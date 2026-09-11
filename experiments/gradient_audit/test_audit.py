import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from experiments.coupled_clock.run import state_hash
from experiments.gradient_audit.audit import (
    audit_record, cosine, comparisons, direct_prior, group, linearity, period_signs,
)
from experiments.gradient_audit.register import sources
from experiments.gradient_audit.run import execute, flatten, persist_case, summarize
from experiments.phase_sync.model import PhaseSyncReadout
from experiments.scaled_adjoint.scaled import Gradient


class VectorTests(unittest.TestCase):
    def test_cosine_ignores_scale_without_materializing_large_gradients(self):
        a = Gradient.create({'x': np.array([1., 2.])}, 5000)
        b = Gradient.create({'x': np.array([-2., -4.])}, -5000)
        self.assertAlmostEqual(cosine(a, b), -1)
        self.assertIsNone(cosine(a, Gradient.create({'x': np.zeros(2)})))

    def test_cancellation_and_zero_are_not_positive_agreement(self):
        a = Gradient.create({'x': np.array([1., -2.])}, 2000)
        g = dict(phase=a, count=a.multiply(-1), total=a.multiply(0), prior=a)
        self.assertEqual(linearity(g), dict(residual_relative_to_component_norm_sum=0., total_to_component_norm_sum=0.))
        self.assertIsNone(comparisons(g)['total_prior_cosine'])

    def test_linearity_exposes_wrong_total(self):
        a = Gradient.create({'x': np.array([1., -2.])})
        result = linearity(dict(phase=a, count=a, total=a))
        self.assertAlmostEqual(result['residual_relative_to_component_norm_sum'], .5)

    def test_informative_signs_keep_zero_gradients_and_weights(self):
        a = Gradient.create({'periods': np.array([[1., -1., 0., 9.]])})
        d = Gradient.create({'periods': np.array([[2., 4., -2., 0.]])})
        r = period_signs(a, d, np.ones((1, 4), bool))
        self.assertEqual(r['informative_cells'], 3)
        for key in ('agreement_fraction', 'disagreement_fraction', 'zero_gradient_fraction'):
            self.assertAlmostEqual(r[key], 1 / 3)
        self.assertEqual(r['weighted_disagreement_fraction'], .5)
        r = period_signs(a, d.multiply(0), np.ones((1, 4), bool))
        self.assertEqual(r['informative_cells'], 0)
        self.assertIsNone(r['agreement_fraction'])

    def test_head_ownership_excludes_other_output_rows(self):
        a = Gradient.create({'output.weight': np.arange(6.).reshape(3, 2, 1),
                             'output.bias': np.array([1., 9., 9.]), 'projection.weight': np.array([3.]),
                             'blocks.0.weight': np.array([4.]), 'gain_logit': np.array(6.)})
        head, shared = group(a, 'period_head'), group(a, 'shared_cnn')
        self.assertAlmostEqual(head.norm_log2(), math.log2(math.sqrt(2)))
        self.assertAlmostEqual(shared.norm_log2(), math.log2(5))

    def test_summary_uses_works_and_preserves_undefined_denominator(self):
        rows = [dict(work=w, metrics={'x': x}) for w, x in (('a', 0.), ('a', 2.), ('b', 10.))]
        self.assertEqual(summarize(rows)['x'], dict(work_macro=5.5, undefined_recordings=0))
        rows[0]['metrics']['x'] = None
        self.assertEqual(summarize(rows)['x'], dict(work_macro=None, undefined_recordings=1))

    def test_flatten_rejects_nonfinite_instead_of_json_nan(self):
        self.assertEqual(flatten({'a': {'b': 1}, 'c': None}), {'a.b': 1, 'c': None})
        with self.assertRaises(ValueError):
            flatten({'a': np.nan})

    def test_completed_metrics_survive_a_later_case_failure_without_final_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = dict(id='synthetic', role='fit', work='authored', frames=251)
            result = dict(loss={'phase': .4, 'count': .7}, reference_cells=250,
                          native_cnn_vjp_passes={'phase': 1, 'count': 2, 'total': 2})
            saved = persist_case(root, row, 'audio', result, {'marker': torch.tensor([1.])})
            with mock.patch('experiments.gradient_audit.run.flatten', side_effect=ValueError('later failure')):
                with self.assertRaisesRegex(ValueError, 'later failure'):
                    persist_case(root, {**row, 'id': 'later'}, 'audio', result, {})
            self.assertEqual(json.loads((root / 'synthetic.audio.case.json').read_bytes()), saved)
            self.assertFalse((root / 'report.json').exists())
            self.assertFalse((root / 'later.audio.case.json').exists())
            with self.assertRaises(FileExistsError):
                persist_case(root, row, 'audio', result, {})


class GradientPathTests(unittest.TestCase):
    def fixture(self, device='cpu', dtype=torch.float32):
        torch.manual_seed(142)
        model = PhaseSyncReadout().to(device=device, dtype=dtype).eval()
        x = torch.randn(1, 251, 512, device=device, dtype=dtype) * .1
        reference = torch.arange(251, device=device, dtype=torch.float64)[None] / 25
        return model, dict(features=x, reference=reference, valid=torch.ones_like(reference, dtype=torch.bool))

    def check_fixed_audit(self, device):
        model, payload = self.fixture(device)
        with torch.inference_mode():
            fields = model.fields(payload['features'])[0].cpu().numpy()
            cycles = model(payload['features']).cycles[0].cpu().numpy()
        before = state_hash(model)
        with (mock.patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('optimizer step')),
              mock.patch.object(torch.optim.Optimizer, '__init__', side_effect=AssertionError('optimizer init'))):
            result, packet, parameters = audit_record(model, payload, fields, cycles)
        self.assertEqual(state_hash(model), before)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(set(parameters), {'phase', 'count', 'total', 'prior'})
        self.assertEqual(set(packet), {'field', 'parameter'})
        self.assertAlmostEqual(result['loss']['total'], result['loss']['phase'] + result['loss']['count'])
        self.assertEqual(result['reference_cells'], 250)
        self.assertLess(result['field_linearity']['residual_relative_to_component_norm_sum'], 1e-10)
        self.assertLess(result['parameter_linearity']['residual_relative_to_component_norm_sum'], 5e-5)

    def test_cpu_fixed_forward_gradients_and_no_optimizer(self):
        self.check_fixed_audit('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_fixed_forward_gradients_and_no_optimizer(self):
        self.check_fixed_audit('cuda')

    def test_direct_prior_analytic_field_and_parameter_finite_difference(self):
        model, payload = self.fixture(dtype=torch.float64)
        gradients, fields, loss, prediction = direct_prior(model, payload)
        expected = 2 * (prediction[:, :-1, 0] + 1) / 250
        np.testing.assert_allclose(np.ldexp(fields.blocks['periods'], fields.exponent), expected, atol=1e-15)
        derivative = math.ldexp(float(gradients.blocks['output.bias'][0]), gradients.exponent)
        base = model.output.bias[0].item()
        with torch.no_grad():
            model.output.bias[0] = base + 1e-6
        _, _, plus, _ = direct_prior(model, payload)
        with torch.no_grad():
            model.output.bias[0] = base - 1e-6
        _, _, minus, _ = direct_prior(model, payload)
        self.assertAlmostEqual(derivative, (plus - minus) / 2e-6, places=7)
        self.assertGreater(loss, 0)

    def test_missing_reference_support_is_not_bridged(self):
        model, payload = self.fixture()
        payload['valid'][:, 120:130] = False
        payload['reference'][:, 120:130] = torch.nan
        result, _, _ = audit_record(model, payload)
        self.assertEqual(result['reference_cells'], 239)
        self.assertEqual(result['pairs_by_lag']['200'], 0)

    def test_changed_saved_prediction_fails_closed(self):
        model, payload = self.fixture()
        with torch.inference_mode():
            cycles = model(payload['features']).cycles[0].cpu().numpy()
        cycles[20] += .01
        with self.assertRaisesRegex(ValueError, 'does not reproduce'):
            audit_record(model, payload, saved_cycles=cycles)

    def test_direct_failure_has_no_stale_transport_attribution(self):
        model, payload = self.fixture()
        seen = []
        with mock.patch('experiments.gradient_audit.audit.direct_prior', side_effect=ValueError('direct failure')):
            with self.assertRaisesRegex(ValueError, 'direct failure'):
                audit_record(model, payload, observer=lambda term, transport: seen.append((term, transport)))
        self.assertEqual(seen[-1], ('prior', None))

    def test_term_entry_failure_has_no_previous_terms_clock_trace(self):
        model, payload = self.fixture()
        def observer(term, transport):
            self.assertIsNone(transport.last_trace)
            self.assertEqual(transport.stage, 'idle')
            if term == 'count':
                raise ValueError('term-entry deadline')
        with self.assertRaisesRegex(ValueError, 'term-entry deadline'):
            audit_record(model, payload, observer=observer)

    def test_registration_preserves_closed_source_identities(self):
        identity = sources()
        self.assertIn('experiments/scaled_phase_fit/run.py', identity)
        self.assertIn('experiments/gradient_audit/run.py', identity)
        self.assertIn('experiments/phase_attribution/diagnostic.py', identity)

    def test_bad_plan_hash_rejected_before_cuda_inputs_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = root / 'plan.json'
            plan.write_text('{}')
            with mock.patch('experiments.gradient_audit.run.make_plan', side_effect=AssertionError('runtime')):
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    execute(root / 'missing-inputs', root / 'missing-predictions', plan, '0' * 64, root / 'out')
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
