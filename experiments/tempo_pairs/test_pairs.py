import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.tempo_pairs import protocol as p
from experiments.tempo_pairs.render import duration_metrics, event_times, pitch_metrics, render, timing_metrics, witness, write_wav
from experiments.tempo_pairs.run import read_source, write_json
from experiments.tempo_pairs.timeline import Timeline


class TimelineTests(unittest.TestCase):
    def test_identity(self):
        t = Timeline((0, 20, 40, 60), (1, 1, 1))
        np.testing.assert_equal(t.to_source([0, 20, 60]), [0, 20, 60])

    def test_duration_and_inverse(self):
        t = Timeline(p.SOURCE_EDGES, p.PROFILES['local_slow_fast'])
        np.testing.assert_allclose(t.output_edges, [0, 20, 45, 61])
        s = np.linspace(0, 60, 123)
        np.testing.assert_allclose(t.to_source(t.to_output(s)), s, atol=1e-12)

    def test_rate_and_phase_inside_interval(self):
        t = Timeline((0, 1, 3), (1, 2))
        phase, rate, valid = t.phase_and_rate([0, 2, 3], [.5, 1, 1.25, 1.5])
        np.testing.assert_allclose(phase, [.25, .5, .75, 1])
        np.testing.assert_allclose(rate, [.5, 1, 1, 2])
        self.assertTrue(valid.all())
        # Incorrect mapped-beat interpolation would give .5/1.5, not .25.
        self.assertNotAlmostEqual(phase[0], .5 / t.to_output(2))

    def test_global_factor(self):
        t = Timeline((0, 60), (1.25,))
        _, rate, valid = t.phase_and_rate([0, 1, 2, 3], [.2, .8, 1.6])
        np.testing.assert_allclose(rate, [1.25]*3)
        self.assertTrue(valid.all())

    def test_no_unbracketed_or_endpoint_labels(self):
        t = Timeline((0, 3), (1,))
        _, _, valid = t.phase_and_rate([.5, 2, 3], [0, .5, 2, 3])
        np.testing.assert_equal(valid, [False, True, True, False])

    def test_invalid_maps(self):
        for edges, rates in [((0, 1), (0,)), ((0, 1), (-1,)), ((0, 1), (np.nan,)),
                             ((1, 2), (1,)), ((0, 0), (1,)), ((0, 1, 2), (1,)),
                             ((0, np.inf), (1,)), ((0,), ())]:
            with self.subTest(edges=edges, rates=rates), self.assertRaises(ValueError):
                Timeline(edges, rates)

    def test_reject_out_of_domain(self):
        t = Timeline((0, 1), (1,))
        for values in ([-.1], [1.1], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                t.to_output(values)

    def test_reject_invalid_beats(self):
        t = Timeline((0, 1), (1,))
        for beats in ([0], [0, 0], [1, 0], [0, np.nan], [-1, 0]):
            with self.assertRaises(ValueError):
                t.phase_and_rate(beats, [.5])

    def test_copies_mutable_arguments(self):
        edges, rates = [0, 1], [1]
        t = Timeline(edges, rates)
        rates[0], edges[1] = 2, 3
        self.assertEqual(t.to_output(1), 1)


class MeasurementTests(unittest.TestCase):
    def test_authored_detector_exact(self):
        for kind in ('clicks', 'clicks_tone'):
            pcm, expected = witness(kind)
            self.assertTrue(timing_metrics(expected, event_times(pcm))['passed'])

    def test_pitch_preserved(self):
        pcm, _ = witness('tone')
        self.assertTrue(pitch_metrics(pcm, [20*p.SR]*3)['passed'])

    def test_pitch_shortcut_rejected(self):
        pcm = .15*np.sin(2*np.pi*550*np.arange(60*p.SR)/p.SR)
        self.assertFalse(pitch_metrics(pcm, [20*p.SR]*3)['passed'])

    def test_missing_and_extra_events_fail(self):
        for observed in ([1, 2], [1, 2, 3, 4], []):
            row = timing_metrics([1, 2, 3], observed)
            self.assertFalse(row['passed'])
            self.assertIsNone(row['timing_p95_ms'])

    def test_delayed_events_fail(self):
        self.assertFalse(timing_metrics([1, 2], [1.1, 2.1])['passed'])

    def test_insensitive_to_rate_control_fails(self):
        events = np.arange(1., 10.)
        self.assertFalse(timing_metrics(events/1.25, events)['passed'])

    def test_reversed_events_fail(self):
        with self.assertRaises(ValueError):
            timing_metrics([1, 2], [2, 1])

    def test_duration_cannot_cancel_at_track_end(self):
        row = duration_metrics([20*p.SR+2000, 20*p.SR-2000, 20*p.SR], [1]*3)
        self.assertEqual(row['cumulative_error_ms'][-1], 0)
        self.assertFalse(row['passed'])

    def test_nominal_duration_passes(self):
        self.assertTrue(duration_metrics([25*p.SR]*3, [.8]*3)['passed'])

    def test_render_rejects_unregistered_request(self):
        with self.assertRaises(ValueError):
            render(np.zeros(8), (1, 1, 1), 'atempo', 'not-called')

    def test_render_process_shape_and_no_padding(self):
        from subprocess import CompletedProcess
        raw = np.zeros(7, dtype='<f4').tobytes()
        with patch('experiments.tempo_pairs.render.subprocess.run', return_value=CompletedProcess([], 0, raw, b'')) as call:
            out, sizes = render(np.zeros(60*p.SR), (1, .8, 1.25), 'atempo', 'renderer')
        self.assertEqual(len(out), 21)
        self.assertEqual(sizes, [7, 7, 7])
        self.assertEqual(call.call_count, 3)
        self.assertEqual(call.call_args.kwargs['timeout'], 60)
        self.assertNotIn('shell', call.call_args.kwargs)


class ContractTests(unittest.TestCase):
    def test_selection_roles_and_licenses(self):
        rows = p.selected_sources()
        self.assertEqual(tuple(r['id'] for r in rows), p.IDS)
        self.assertEqual(len({r['work'] for r in rows}), 3)
        self.assertTrue(all(r['role'] == 'fit' for r in rows))

    def test_selection_rule_not_result_cherry_pick(self):
        from experiments.clock_readout.data import plan
        old = plan()
        selected = [next(r['id'] for r in old['cases'] if r['role'] == 'fit' and r['work'] == work)
                    for work in old['fit_works'][:3]]
        self.assertEqual(selected, list(p.IDS))

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'a.json'
            write_json(path, {'a': 1})
            with self.assertRaises(FileExistsError):
                write_json(path, {'a': 2})
            self.assertEqual(json.loads(path.read_text()), {'a': 1})

    def test_wav_clipping_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_wav(Path(tmp)/'a.wav', np.array([1.]))

    def test_wav_roundtrip_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pcm = np.resize(np.array([-32768, -1234, 0, 4321, 32767], dtype=np.float32)/32768, 60*p.SR)
            write_wav(root/'a.wav', pcm)
            item = dict(audio_hint='a.wav', audio_sha256=p.sha(root/'a.wav'))
            np.testing.assert_equal(read_source(root, item), pcm)
            with self.assertRaises(ValueError):
                read_source(root, dict(item, audio_sha256='0'*64))

    def test_closure_has_protocol_and_no_output_dependency(self):
        closure = p.source_closure()
        self.assertIn('experiments/tempo_pairs/PROTOCOL-v1.md', closure)
        self.assertFalse(any('results' in name for name in closure))


if __name__ == '__main__':
    unittest.main()
