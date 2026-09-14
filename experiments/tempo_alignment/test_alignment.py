import unittest
from unittest.mock import patch

import numpy as np

from experiments.tempo_alignment.alignment import (
    DELAY_S, Features, RESOLUTIONS, SR, extract, measure, paired_support, region, summarize)
from experiments.tempo_alignment.run import changed_context, interior_gate, negative_summary
from experiments.tempo_pairs.timeline import Timeline


def feature_fixture():
    values = np.random.default_rng(412).normal(size=(1001, 8))
    return Features(0., .01, values)


def views(feature):
    return {n: feature for n in RESOLUTIONS}


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.source = feature_fixture()
        self.timeline = Timeline((0, 10), (1,))

    def test_identity(self):
        r = measure(views(self.source), views(self.source), self.timeline, 4)
        self.assertEqual(r['status'], 'aligned')
        self.assertAlmostEqual(r['lag_s'], 0)

    def test_known_delay_and_paired_gate(self):
        shifted = Features(.12, .01, self.source.values)
        a = measure(views(self.source), views(self.source), self.timeline, 4)
        b = measure(views(self.source), views(shifted), self.timeline, 4)
        self.assertEqual(b['status'], 'displaced')
        self.assertAlmostEqual(b['lag_s'], .12)
        self.assertTrue(paired_support(a, b))

    def test_wrong_delay_direction_not_accepted(self):
        shifted = Features(-.12, .01, self.source.values)
        a = measure(views(self.source), views(self.source), self.timeline, 4)
        b = measure(views(self.source), views(shifted), self.timeline, 4)
        self.assertAlmostEqual(b['lag_s'], -.12)
        self.assertFalse(paired_support(a, b))

    def test_control_unchanged_cannot_pass(self):
        a = measure(views(self.source), views(self.source), self.timeline, 4)
        self.assertFalse(paired_support(a, a))

    def test_gain_and_dc_offset_do_not_move_features(self):
        transformed = Features(0, .01, 2*self.source.values + np.arange(8))
        self.assertEqual(measure(views(self.source), views(transformed), self.timeline, 4)['status'], 'aligned')

    def test_flat_signal_abstains(self):
        f = Features(0, .01, np.ones((1001, 8)))
        self.assertEqual(measure(views(f), views(f), self.timeline, 4)['status'], 'uninformative')

    def test_repeated_signal_abstains(self):
        values = np.tile(np.random.default_rng(4).normal(size=(10, 8)), (101, 1))
        f = Features(0, .01, values)
        self.assertNotEqual(measure(views(f), views(f), self.timeline, 4)['status'], 'aligned')

    def test_search_boundary_is_not_alignment(self):
        f = Features(.25, .01, self.source.values)
        self.assertEqual(measure(views(self.source), views(f), self.timeline, 4)['status'], 'search_boundary')

    def test_real_support_required(self):
        r = measure(views(self.source), views(self.source), self.timeline, .5)
        self.assertEqual(r['status'], 'unsupported')
        with self.assertRaises(ValueError):
            self.source.at([-1])

    def test_piecewise_map_not_ignored(self):
        timeline = Timeline((0, 4, 10), (.8, 1.25))
        ts = np.arange(0, timeline.output_edges[-1]+.0001, .01)
        mapped = Features(0, .01, self.source.at(timeline.to_source(ts)))
        positive = measure(views(self.source), views(mapped), timeline, 3)
        wrong = measure(views(self.source), views(self.source), timeline, 3)
        self.assertEqual(positive['status'], 'aligned')
        self.assertNotEqual(wrong['status'], 'aligned')

    def test_two_resolutions_required(self):
        with self.assertRaises(ValueError):
            measure({512: self.source}, views(self.source), self.timeline, 4)

    def test_disagreeing_resolutions_reject(self):
        rows = [dict(status='qualified', lag_s=x, score=1, margin=.5) for x in (0, .1)]
        with patch('experiments.tempo_alignment.alignment.one_resolution', side_effect=rows):
            r = measure(views(self.source), views(self.source), self.timeline, 4)
        self.assertEqual(r['status'], 'resolution_disagreement')
        self.assertIsNone(r['lag_s'])

    def test_input_nonmutation(self):
        before = self.source.values.copy()
        measure(views(self.source), views(self.source), self.timeline, 4)
        np.testing.assert_array_equal(before, self.source.values)
        self.assertFalse(self.source.values.flags.writeable)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            Features(0, .01, np.full((4, 4), np.nan))
        with self.assertRaises(ValueError):
            measure(views(self.source), views(self.source), self.timeline, np.nan)


class AudioTests(unittest.TestCase):
    @staticmethod
    def audio():
        t = np.arange(8*SR)/SR
        pcm = np.zeros(len(t))
        for i, center in enumerate((1.2, 1.73, 2.05, 2.63, 3.01, 3.48, 4.25, 4.51, 5.03, 6.4)):
            pcm += .4*np.exp(-.5*((t-center)/.025)**2)*np.sin(2*np.pi*(400+137*i)*t)
        return pcm

    def test_real_pcm_delay_recovers_both_resolutions(self):
        pcm = self.audio()
        delayed = np.r_[np.zeros(round(DELAY_S*SR)), pcm[:-round(DELAY_S*SR)]]
        a = {n: extract(pcm, n) for n in RESOLUTIONS}
        b = {n: extract(delayed, n) for n in RESOLUTIONS}
        timeline = Timeline((0, 8), (1,))
        natural = measure(a, a, timeline, 3.5)
        shifted = measure(a, b, timeline, 3.5)
        self.assertTrue(paired_support(natural, shifted))

    def test_real_tone_not_timing_evidence(self):
        pcm = .15*np.sin(2*np.pi*440*np.arange(8*SR)/SR)
        a = {n: extract(pcm, n) for n in RESOLUTIONS}
        self.assertNotEqual(measure(a, a, Timeline((0, 8), (1,)), 3.5)['status'], 'aligned')

    def test_extract_no_padding_and_reject_bad_input(self):
        f = extract(self.audio(), 2048)
        self.assertGreater(f.start, 0)
        self.assertLess(f.end, 8)
        with self.assertRaises(ValueError):
            extract(np.zeros(10), 2048)
        with self.assertRaises(ValueError):
            extract(np.ones(2048)*np.inf, 512)


class PopulationTests(unittest.TestCase):
    def test_seams_and_edges_not_silently_dropped(self):
        self.assertEqual(region(.5, 60, [20, 40]), 'edge')
        self.assertEqual(region(20, 60, [20, 40]), 'seam')
        self.assertEqual(region(30, 60, [20, 40]), 'interior')
        rows = [dict(region='interior', natural={'status':'aligned', 'lag_s':0}, feature_capture_supported=True),
                dict(region='interior', natural={'status':'ambiguous', 'lag_s':None}, feature_capture_supported=False)]
        s = summarize(rows)
        self.assertEqual(s['interior']['supported_fraction'], .5)
        self.assertFalse(interior_gate(s, s))

    def test_empty_populations_cannot_pass(self):
        s = summarize([])
        self.assertFalse(interior_gate(s, s))
        self.assertFalse(negative_summary([])['passed'])

    def test_unsupported_cannot_improve_negative_denominator(self):
        rows = [dict(changed_context=True, wrong_map={'status':'aligned'})]
        rows += [dict(changed_context=True, wrong_map={'status':'unsupported'}) for _ in range(99)]
        s = negative_summary(rows)
        self.assertEqual(s['false_aligned_fraction'], 1)
        self.assertFalse(s['passed'])

    def test_changed_context_excludes_unit_rate_and_seams(self):
        timeline = Timeline((0, 20, 40, 60), (1, .8, 1.25))
        self.assertFalse(changed_context(timeline, 5))
        self.assertFalse(changed_context(timeline, 20))
        self.assertTrue(changed_context(timeline, 30))


if __name__ == '__main__':
    unittest.main()
