"""CPU tests without music, pretrained weights, downloads or GPU allocation."""
import copy
import unittest

import numpy as np
import torch

from experiments.separated_evidence.model import SharedEvidence, SeparatedEvidence
from experiments.relative_tempo import data
from experiments.relative_tempo.model import query, paired_loss
from experiments.relative_tempo.run import decision, natural_metrics


class SparseContractTests(unittest.TestCase):
    def point(self, t=10.5, **changes):
        p = dict(natural=dict(center_s=t), index=10, feature_capture_supported=True, region='interior')
        p.update(changes)
        return p

    def test_unsupported_is_never_admitted(self):
        r = data.point_record(dict(source_id='a', profile='fast'), self.point(feature_capture_supported=False))
        self.assertFalse(r['admitted'])
        self.assertEqual(r['exclusion'], 'alignment_not_supported')

    def test_source_and_output_context_clearance(self):
        case = dict(source_id='a', profile='slow')
        self.assertFalse(data.point_record(case, self.point(3.0))['admitted'])  # source 2.4 s
        self.assertTrue(data.point_record(case, self.point(10.5))['admitted'])
        self.assertFalse(data.point_record(case, self.point(24.5))['admitted'])

    def test_no_edge_or_seam_admission(self):
        for region in ('edge', 'seam'):
            self.assertFalse(data.point_record(dict(source_id='a', profile='fast'), self.point(region=region))['admitted'])

    def test_unseen_schedules_cannot_enter_fit(self):
        for p in data.PROFILES:
            row = data.point_record(dict(source_id='a', profile=p), self.point())
            self.assertEqual(row['pair_role'] == 'fit', p in data.FIT_PROFILES)

    def test_population_preserves_all_sources_and_denominator(self):
        rows = data.population()
        self.assertEqual(len(rows), 915)
        self.assertEqual({r['source_id'] for r in rows if r['admitted']}, set(data.IDS))
        self.assertEqual(len({(r['source_id'], r['profile']) for r in rows if r['admitted']}), 15)
        self.assertTrue(any(not r['admitted'] for r in rows))
        self.assertEqual(len({(r['source_id'], r['profile'], r['index']) for r in rows}), len(rows))

    def test_cell_center_query_not_frame_start(self):
        np.testing.assert_allclose(data.query_numpy(np.arange(10.), [.01, .02, .03]), [0, .5, 1])

    def test_query_never_extrapolates(self):
        for t in (-1., 0., 1., np.nan):
            with self.assertRaises(ValueError):
                data.query_numpy(np.arange(10.), [t])

    def test_missing_or_nonfinite_prediction_fails(self):
        for p, y in (([], []), ([np.nan], [1]), ([1, 2], [1])):
            with self.assertRaises(ValueError):
                data.relative_metrics(p, y)

    def test_zero_change_baseline_has_no_response(self):
        metric = data.relative_metrics([0, 0], [-.3, .3])
        self.assertAlmostEqual(metric['rmse'], metric['zero_change_rmse'])
        self.assertEqual(metric['slope'], 0)
        self.assertEqual(metric['sign_correct'], 0)

    def test_correct_and_wrong_speed_direction(self):
        for factor in (1, -1):
            y = np.array([-.3, .3])
            m = data.relative_metrics(y*factor, y)
            self.assertAlmostEqual(m['slope'], factor)
            self.assertEqual(m['sign_correct'], int(factor > 0))

    def test_identity_has_no_fake_change_score(self):
        m = data.relative_metrics([0, .1], [0, 0])
        self.assertIsNone(m['slope'])
        self.assertIsNone(m['sign_correct'])


class QueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(142)
        self.model = SeparatedEvidence(SharedEvidence()).tempo
        self.features = torch.randn(800, 512)

    def test_patch_matches_full_forward_and_gradients(self):
        times = [4.5, 7.371]
        full = self.model(self.features[None])[0]
        a, b, f = data.query_indices(times, len(full))
        f = torch.tensor(f, dtype=full.dtype)
        expected = full[a]*(1-f)+full[b]*f
        actual = query(self.model, self.features, times)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        g1 = torch.autograd.grad(expected.sum(), tuple(self.model.parameters()))
        g2 = torch.autograd.grad(actual.sum(), tuple(self.model.parameters()))
        for x, y in zip(g1, g2):
            torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4)

    def test_missing_context_rejected(self):
        with self.assertRaises(ValueError):
            query(self.model, self.features, [.5])

    def test_pair_sign_and_both_branches_differentiate(self):
        source = self.features.clone().requires_grad_()
        transformed = torch.randn_like(source, requires_grad=True)
        row = dict(source_s=5., output_s=4., rate=1.25)
        left = query(self.model, source, [5.])
        right = query(self.model, transformed, [4.])
        loss = paired_loss(self.model, source, transformed, [row])
        torch.testing.assert_close(loss, (right-left+np.log2(1.25)).square().mean())
        loss.backward()
        self.assertGreater(float(source.grad.abs().sum()), 0)
        self.assertGreater(float(transformed.grad.abs().sum()), 0)

    def test_identity_query_exactly_cancels(self):
        row = dict(source_s=5., output_s=5., rate=1.)
        self.assertEqual(float(paired_loss(self.model, self.features, self.features, [row]).detach()), 0)

    def test_zero_audio_does_not_know_speed(self):
        zero = torch.zeros_like(self.features)
        left = query(self.model, zero, [4.5])
        right = query(self.model, zero, [7.5])
        torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_one_real_optimizer_step_changes_weights(self):
        before = copy.deepcopy(self.model.state_dict())
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.0003, weight_decay=.0001)
        row = dict(source_s=5., output_s=4., rate=1.25)
        loss = paired_loss(self.model, self.features, self.features.roll(17, 0), [row])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in self.model.state_dict().items()))


class OutcomeTests(unittest.TestCase):
    def fixture(self):
        names = ('initial', 'native-only', 'native-plus-relative')
        natural = [dict(role=role, work='work', metrics={n: dict(mse=1., tempo_median_error_percent=10.,
                    tempo_p95_error_percent=20.) for n in names}) for role in ('fit', 'development', 'diagnostic')]
        pairs = []
        for source in data.IDS:
            for profile in data.PROFILES:
                changed = dict(rmse=.1, zero_change_rmse=.3, slope=1., sign_correct=1.)
                metrics = {n: dict(all=dict(mean_abs_prediction=.01), changed=dict(changed)) for n in names}
                metrics['native-only']['changed']['rmse'] = .3
                pairs.append(dict(source_id=source, profile=profile,
                    role='fit' if profile in data.FIT_PROFILES else 'schedule_diagnostic', metrics=metrics))
        return natural, pairs

    def test_all_gates_required(self):
        n, p = self.fixture()
        self.assertTrue(decision(n, p)[1]['passed'])
        for key, value in (('rmse', .4), ('slope', .2), ('sign_correct', .5)):
            altered = copy.deepcopy(p)
            altered[-1]['metrics']['native-plus-relative']['changed'][key] = value
            self.assertFalse(decision(n, altered)[1]['passed'])

    def test_natural_regression_cannot_be_hidden_by_pairs(self):
        n, p = self.fixture()
        n[1]['metrics']['native-plus-relative']['mse'] = 1.2
        self.assertFalse(decision(n, p)[1]['natural_retention_passed'])

    def test_identity_regression_fails(self):
        n, p = self.fixture()
        p[0]['metrics']['native-plus-relative']['all']['mean_abs_prediction'] = .06
        self.assertFalse(decision(n, p)[1]['identity_passed'])

    def test_missing_pair_fails(self):
        n, p = self.fixture()
        with self.assertRaises(ValueError):
            decision(n, p[:-1])

    def test_native_constant_speed_metrics(self):
        reference = np.arange(500, dtype=np.float64)*2/50
        row = dict(reference=reference, valid=np.ones(500, bool))
        m = natural_metrics(np.full(499, -1.), row)
        self.assertLess(m['mse'], 1e-20)
        self.assertLess(m['tempo_median_error_percent'], 1e-10)
        self.assertIsNone(m['large_change_4s'])


if __name__ == '__main__':
    unittest.main()
