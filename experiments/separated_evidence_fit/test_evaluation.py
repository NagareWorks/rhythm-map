"""Generated independent fields, support and non-admission regression controls."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.coupled_clock.measurement import KEYS, clock_fields, independent_fields, measure
from experiments.coupled_clock.run import state_hash
from experiments.phase_sync_fit import run as old
from experiments.phase_sync_fit.test_fit import passing_gate
from experiments.separated_evidence_fit import evaluation as ev
from experiments.separated_evidence_fit.register import initial_models, ARMS


class EvaluationTests(unittest.TestCase):
    def test_direct_cell_geometry_and_tempo_only_count_never_fabricate_a_clock(self):
        q = np.arange(251, dtype=np.float64) * .04
        vector = np.stack((np.cos(2 * np.pi * q), np.sin(2 * np.pi * q)), 1)
        rate, phase = ev.evidence_fields(np.full(250, -1.), vector)
        self.assertEqual(rate.shape, (250,))
        self.assertEqual(phase.shape, (251,))
        metric = measure((rate, phase), q, np.ones(251, bool))
        self.assertLess(metric[KEYS[2]], 1e-12)
        self.assertLess(metric[KEYS[3]], 1e-12)
        # A half-cycle disagreement cannot be hidden by correct tempo counts.
        shifted = measure(ev.evidence_fields(np.full(250, -1.), -vector), q, np.ones(251, bool))
        self.assertAlmostEqual(shifted[KEYS[2]], .5)
        self.assertLess(shifted[KEYS[3]], 1e-12)
        for wrong in (np.full(251, -1.), np.array([])):
            with self.assertRaisesRegex(ValueError, 'geometry'):
                ev.evidence_fields(wrong, vector)

    def test_zero_phase_and_overflow_do_not_shrink_required_support(self):
        q, valid = np.arange(251) * .04, np.ones(251, bool)
        vector = np.stack((np.cos(2 * np.pi * q), np.sin(2 * np.pi * q)), 1)
        vector[0] = 0
        metric = measure(ev.evidence_fields(np.full(250, -1.), vector), q, valid)
        self.assertEqual((metric['reference_points'], metric['available_points']), (251, 250))
        self.assertIsNone(metric[KEYS[2]])
        period = np.full(250, -1.)
        period[-1] = -2000
        metric = measure(ev.evidence_fields(period, vector), q, valid)
        self.assertEqual(metric['reference_cells'], 250)
        self.assertEqual(metric['available_cells'], 249)
        self.assertIsNone(metric[KEYS[0]])
        self.assertIsNone(metric[KEYS[3]])

    def test_all_ten_selected_pairs_preserve_sources_weights_and_baseline_support(self):
        frames = 251
        q, valid = np.arange(frames) * .04, np.ones(frames, bool)
        v1 = np.stack((np.full(frames, -1.), np.cos(2 * np.pi * q), np.sin(2 * np.pi * q)), 1)
        hidden = np.random.default_rng(111).normal(0, .1, (frames, 512)).astype(np.float32)
        row = dict(id='authored', role='diagnostic', work='authored', frames=frames,
            packet_sha256='not-musical', hidden=hidden, reference=q, valid=valid, raw=q.copy(),
            raw_valid=valid.copy(), v1=v1)
        row['raw_valid'][0:3] = False
        common = valid & row['raw_valid']
        metric = measure(clock_fields(q), q, common)
        parent = dict(id='authored', role='diagnostic', work='authored', frames=frames,
            audio_common=metric, raw_common=metric, v1_common=measure(independent_fields(v1), q, common))
        parents = {k: {'cases': [parent]} for k in ('coupled', 'trajectory', 'protected')}
        _, template = initial_models('cpu')
        models = {k: copy.deepcopy(template).eval() for k in ARMS}
        before, inputs, calls = {k: state_hash(v) for k, v in models.items()}, hidden.copy(), []
        handles = [model.register_forward_hook(lambda m, a, b, key=key: calls.append(key)) for key, model in models.items()]
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                rows = ev.evaluate(models, [row], 'cpu', root, parents, lambda: None)
                self.assertEqual(len(calls), 10)
                self.assertEqual(calls.count('shared-natural'), 4)
                self.assertEqual(calls.count('shared-zero'), 1)
                with np.load(root / 'authored.evidence.npz', allow_pickle=False) as packet:
                    self.assertEqual(len(packet.files), 20)
                    self.assertEqual(packet['shared_audio_log_period'].shape, (frames - 1,))
                for architecture in ev.ARCHITECTURES:
                    result = rows[architecture][0]
                    self.assertEqual(result['audio']['reference_points'], frames)
                    self.assertEqual(result['audio_common']['reference_points'], frames - 3)
                    self.assertEqual(result['audio_common']['reference_cells'], frames - 4)
                    self.assertEqual(result['raw_common'], metric)
                self.assertEqual(before, {k: state_hash(v) for k, v in models.items()})
                self.assertTrue(np.array_equal(inputs, hidden))
        finally:
            for handle in handles:
                handle.remove()

    def passing(self):
        rows, summary, fits = {}, {}, []
        for architecture in ev.ARCHITECTURES:
            r, s, f = passing_gate()
            for row in r:
                row['id'] = row['role']
            if architecture == 'separated':
                for row in r:
                    for name in ('audio', 'audio_common'):
                        row[name] = {k: .4 for k in KEYS}
                for role in s:
                    for name in ('audio', 'audio_common'):
                        s[role][name] = {k: .4 for k in KEYS}
            rows[architecture], summary[architecture] = r, s
            fits.extend(dict(value, arm=architecture + '-' + variant, architecture=architecture)
                        for value, variant in zip(f, ('natural', 'zero')))
        return rows, summary, fits

    def test_original_numerical_gates_are_reused_but_never_become_clock_admission(self):
        self.assertIs(ev.comparison_gate, old.comparison_gate)
        result = ev.compare(*self.passing())
        self.assertTrue(result['supports_fusion_proposal'])
        self.assertTrue(result['numerical_gates']['separated']['passed'])
        self.assertFalse(any(result[k] for k in ('coherent_clock', 'independent_acceptance', 'production_change')))

    def test_baseline_regression_controls_and_missing_metrics_cannot_be_overridden(self):
        for kind, key in (('raw_common', KEYS[1]), ('coupled_common', KEYS[3]), ('trajectory_common', KEYS[2]),
                          ('same_zero_common', KEYS[2]), ('time_mean_common', KEYS[2]), ('half_roll_common', KEYS[2])):
            rows, summary, fits = self.passing()
            summary['separated']['diagnostic'][kind][key] = .01
            self.assertFalse(ev.compare(rows, summary, fits)['supports_fusion_proposal'], kind)
        rows, summary, fits = self.passing()
        rows['separated'][0]['same_zero_common'][KEYS[2]] = None
        self.assertFalse(ev.compare(rows, summary, fits)['supports_fusion_proposal'])

    def test_equal_architectures_do_not_justify_added_capacity(self):
        rows, summary, fits = self.passing()
        rows['separated'] = copy.deepcopy(rows['shared'])
        summary['separated'] = copy.deepcopy(summary['shared'])
        gate = ev.compare(rows, summary, fits)
        self.assertTrue(gate['separation_nonregression'])
        self.assertFalse(gate['separation_strict_development_gain'])
        self.assertFalse(gate['supports_fusion_proposal'])

    def test_one_good_task_cannot_conceal_the_other_tasks_failure(self):
        rows, summary, fits = self.passing()
        summary['separated']['development']['zero'][KEYS[2]] = .1
        gate = ev.compare(rows, summary, fits)
        self.assertTrue(gate['per_task_audio_signal']['separated']['tempo'])
        self.assertFalse(gate['per_task_audio_signal']['separated']['phase'])
        self.assertFalse(gate['supports_fusion_proposal'])


if __name__ == '__main__':
    unittest.main()
