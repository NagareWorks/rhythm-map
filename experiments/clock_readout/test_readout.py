"""Local/isolated runtime tests, not a musical accuracy claim."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

import data
import model
import run

torch.set_num_threads(2)


class DataTests(unittest.TestCase):
    def test_constant_native_phase_and_period(self):
        y, valid = data.targets([0., .5, 1., 1.5, 2.], 101, 2.)
        self.assertEqual(int(valid.sum()), 100)
        np.testing.assert_array_equal(y[valid, 0], -np.ones(100))
        np.testing.assert_allclose(y[0], [-1, 1, 0], atol=1e-6)
        np.testing.assert_allclose(y[25], [-1, 1, 0], atol=1e-6)

    def test_targets_follow_real_interval_change_not_raw_event_omission(self):
        constant, _ = data.targets([0., .5, 1., 1.5, 2.], 100, 2.)
        changed, _ = data.targets([0., .5, 1., 2.], 100, 2.)
        self.assertEqual(constant[75, 0], -1)
        self.assertEqual(changed[75, 0], 0)

    def test_unbracketed_reference_and_physical_tail_not_zero_labels(self):
        _, valid = data.targets([.5, 1., 1.5], 101, 2.)
        self.assertEqual(np.where(valid)[0].tolist(), list(range(25, 75)))

    def test_invalid_timelines_fail(self):
        for beats in ([0], [0, 0], [1, 0], [0, np.nan]):
            with self.assertRaises(ValueError):
                data.targets(beats, 100, 2.)

    def test_work_versions_stay_together_and_artbeat_never_fits(self):
        p = data.plan()
        self.assertEqual(len(p['cases']), 40)
        self.assertEqual(len(p['fit_works']), 9)
        self.assertEqual(len(p['development_works']), 3)
        self.assertFalse(set(p['fit_works']) & set(p['development_works']))
        for work in {r['work'] for r in p['cases']}:
            self.assertEqual(len({r['role'] for r in p['cases'] if r['work'] == work}), 1)
        self.assertTrue(all(r['role'] == 'diagnostic' for r in p['cases'] if r['cohort'] == 'artbeat'))
        self.assertFalse(p['independent_acceptance'])
        self.assertTrue(p['prior_calibration_reused_for_exploratory_fitting'])

    def test_tail_windows_retained_without_overlap(self):
        self.assertEqual(data.windows(401), [(0, 200), (200, 400), (400, 401)])

    def test_missing_or_nonfinite_prediction_cannot_improve_headline(self):
        y, valid = data.targets([0., .5, 1.], 50, 1.)
        bad = y.copy()
        bad[0, 0] = np.inf
        result = data.metrics(bad, y, valid)
        self.assertIsNone(result['tempo_median_error_percent'])
        self.assertEqual(result['period_available_frames'], 49)
        bad = y.copy()
        bad[:, 1:] = 0
        self.assertIsNone(data.metrics(bad, y, valid)['phase_mean_absolute_cycles'])

    def test_perfect_and_octave_period_metrics_are_distinct(self):
        y, valid = data.targets([0., .5, 1.], 50, 1.)
        self.assertEqual(data.metrics(y, y, valid)['tempo_median_error_percent'], 0)
        shifted = y.copy()
        shifted[:, 0] += 1
        self.assertEqual(data.metrics(shifted, y, valid)['tempo_median_error_percent'], 50)

    def test_phase_period_coherence_is_measured_without_repair(self):
        y, valid = data.targets([0., .5, 1.], 50, 1.)
        before = y.copy()
        consistent = run.coherence(y, valid)
        self.assertEqual(consistent['forward_phase_fraction'], 1)
        self.assertLess(consistent['mean_phase_period_disagreement_cycles_per_frame'], 1e-6)
        y[:, 2] *= -1
        self.assertEqual(run.coherence(y, valid)['forward_phase_fraction'], 0)
        np.testing.assert_array_equal(y[:, 0], before[:, 0])


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(142)
        self.net = model.ClockReadout().eval()

    def test_architecture_and_direct_output_no_candidate_input(self):
        self.assertEqual(sum(p.numel() for p in self.net.parameters()), 24003)
        self.assertEqual(model.HALO, 126)
        self.assertEqual(tuple(self.net(torch.zeros(2, 452, 512)).shape), (2, 452, 3))

    def test_whole_and_chunked_output_match_including_physical_edges(self):
        x = torch.randn(615, 512)
        with torch.inference_mode():
            full = self.net(torch.nn.functional.pad(x.T, (model.HALO, model.HALO)).T[None])[0, model.HALO:model.HALO+len(x)]
        np.testing.assert_allclose(run.predict(self.net, x), full.numpy(), rtol=1e-5, atol=2e-6)

    def test_frames_beyond_receptive_field_do_not_affect_center(self):
        x = torch.randn(1, 700, 512)
        changed = x.clone()
        changed[:, :200] *= 10
        with torch.inference_mode():
            self.assertTrue(torch.equal(self.net(x)[:, 350], self.net(changed)[:, 350]))

    def test_zero_audio_control_is_independent_of_actual_features(self):
        a, b = torch.randn(451, 512), torch.randn(451, 512) * 100
        np.testing.assert_array_equal(run.predict(self.net, a, True), run.predict(self.net, b, True))

    def test_prediction_does_not_mutate_input_or_model(self):
        x = torch.randn(251, 512)
        old, state = x.clone(), copy.deepcopy(self.net.state_dict())
        run.predict(self.net, x)
        self.assertTrue(torch.equal(old, x))
        self.assertTrue(all(torch.equal(value, self.net.state_dict()[name]) for name, value in state.items()))

    def test_real_gradient_update_reaches_temporal_and_output_weights(self):
        x = torch.randn(2, 452, 512)
        y = torch.randn(2, 200, 3)
        loss = run.supervised_loss(self.net(x)[:, 126:326], y).mean()
        loss.backward()
        for name, parameter in self.net.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(self.net.blocks[-1].depthwise.weight.grad.abs().sum()), 0)


class StoredResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(__file__).parent
        cls.result = json.loads((cls.folder / 'results-v1.json').read_bytes())
        cls.inputs = json.loads((cls.folder / 'inputs-v1.json').read_bytes())
        cls.execution = json.loads((cls.folder / 'execution-v1.json').read_bytes())

    def test_frozen_source_and_artifact_identities(self):
        self.assertEqual(data.sha(self.folder / 'results-v1.json'),
                         '990f7a2c71b2d3d61ab77c81daa69c3c5add3a78ead43d776fbf03d25ed07736')
        self.assertEqual(data.sha(self.folder / 'inputs-v1.json'), self.result['inputs_sha256'])
        self.assertEqual(self.execution['inputs_sha256'], self.result['inputs_sha256'])
        self.assertEqual(self.execution['plan_sha256'], self.inputs['plan_sha256'])
        for name, expected in self.result['plan']['source_sha256'].items():
            self.assertEqual(data.sha(self.folder / name), expected, name)
        self.assertEqual(data.sha(self.folder / 'run.py'), self.execution['runner_sha256'])

    def test_complete_population_and_coverage_are_retained(self):
        cases = self.result['cases']
        self.assertEqual(len(cases), 40)
        self.assertEqual({role: sum(x['role'] == role for x in cases)
                          for role in ('fit', 'development', 'diagnostic')},
                         dict(fit=20, development=5, diagnostic=15))
        self.assertEqual([(x['id'], x['role'], x['work']) for x in cases],
                         [(x['id'], x['role'], x['work']) for x in self.inputs['cases']])
        for result, source in zip(cases, self.inputs['cases']):
            self.assertEqual(result['reference_frames'], source['valid_reference_frames'])
            self.assertEqual(result['frames'], source['frames'])
            for key in ('audio', 'zero'):
                self.assertEqual(result[key]['period_available_frames'], result['reference_frames'])
                self.assertEqual(result[key]['phase_available_frames'], result['reference_frames'])
        for key in ('independent_acceptance', 'holdout_access', 'production_change'):
            self.assertIs(self.result[key], False)
        self.assertIs(self.result['training'], True)

    def test_group_summaries_regressions_and_failed_gate_recompute(self):
        for role, summary in self.result['summary'].items():
            rows = [r for r in self.result['cases'] if r['role'] == role]
            for kind, expected in summary.items():
                self.assertEqual(run.aggregate(rows, kind), expected)
        for row in self.result['cases']:
            for key, regressed in row['regressions_vs_raw_common'].items():
                self.assertEqual(row['audio_common'][key] > row['raw_common'][key], regressed)
        s = self.result['summary']['development']
        improvement = 1 - s['audio']['tempo_median_error_percent'] / s['zero']['tempo_median_error_percent']
        self.assertLess(improvement, .1)
        self.assertIs(self.result['development_gate_passed'], False)
        self.assertEqual(self.result['decision'], 'close_this_fixed_direct_readout_no_automatic_sweep')

    def test_checkpoints_use_only_development_loss_and_budgets_hold(self):
        self.assertEqual(len(self.result['fits']), 2)
        for fit in self.result['fits']:
            best = min(fit['history'], key=lambda row: row['development_loss'])
            self.assertEqual(fit['selected_epoch'], best['epoch'])
            self.assertEqual(fit['development_loss'], best['development_loss'])
            self.assertEqual(fit['seed'], 142)
            self.assertEqual(fit['epochs'], 20)
            self.assertLess(fit['elapsed_s'], 1800)
            self.assertIs(fit['budget_exhausted'], False)


if __name__ == '__main__':
    unittest.main()
