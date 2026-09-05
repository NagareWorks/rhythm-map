import json
import math
from pathlib import Path
import unittest

import numpy as np

import beat_this_semantics_audit as audit


class BeatThisSemanticsTests(unittest.TestCase):
    def test_frozen_provenance_scope_and_denominators(self):
        here = Path(__file__).parent
        report = json.loads((here / 'beat-this-semantics-v1.json').read_text())
        source = json.loads((here / 'beat-this-semantics-source-v1.json').read_text())
        lock = json.loads((here / 'reference-lock.json').read_text())
        old = json.loads((here / 'dense-clock-evidence-v1.json').read_text())
        self.assertEqual(source['reference_revision'], lock['reference_revision'])
        self.assertEqual(source['checkpoint_sha256'], lock['checkpoint']['sha256'])
        self.assertEqual(set(source['source_sha256']), set(audit.SOURCE_FILES))
        self.assertEqual(source['checkpoint_hyperparameters']['pos_weights'], audit.WEIGHTS)
        self.assertFalse(source['checkpoint_sum_head_present'])
        self.assertTrue(source['effective_sum_head_from_pinned_loader_default'])
        self.assertEqual(report['upstream'], source)
        self.assertEqual(report['script_sha256'], audit.dense.sha(Path(audit.__file__).read_bytes()))
        self.assertEqual(report['source_record_sha256'], audit.dense.sha((here / 'beat-this-semantics-source-v1.json').read_bytes()))
        self.assertEqual(report['dense_report_sha256'], audit.dense.sha((here / 'dense-clock-evidence-v1.json').read_bytes()))
        for name, identity in report['helper_sha256'].items():
            self.assertEqual(identity, audit.dense.sha((here / name).read_bytes()))
        for key in ('holdout_opened', 'training_run', 'decoder_replayed', 'fitted_parameters',
                    'production_observations_changed', 'accuracy_improvement_claimed', 'adapter_accepted'):
            self.assertIs(report[key], False)
        self.assertEqual(len(report['cohorts']), 2)
        for actual, prior in zip(report['cohorts'], old['cohorts']):
            self.assertTrue(actual['complete'])
            for key in ('cohort', 'frozen_evidence_sha256', 'capture_summary_sha256', 'source_hashes'):
                self.assertEqual(actual[key], prior[key])
            self.assertEqual(actual['pooled']['frames'], prior['total_frames_per_head'])
            self.assertEqual([(t['id'], t['capture_sha256']) for t in actual['cases']],
                             [(t['id'], t['capture_sha256']) for t in prior['cases']])
            for key, value in actual['pooled'].items():
                self.assertEqual(value, sum(t['counts'][key] for t in actual['cases']))
            for t in actual['cases']:
                c = t['counts']
                self.assertTrue(0 <= c['after_weight_offset_downbeat_gt_beat'] <= c['downbeat_gt_beat'] <= c['frames'])

    def test_shifted_annotation_can_have_low_score_without_extra_loss(self):
        controls = audit.controls()
        frozen = json.loads(Path(__file__).with_name('beat-this-semantics-v1.json').read_text())['controls']
        for live, old in zip(controls['shifted_pulse'], frozen['shifted_pulse']):
            self.assertAlmostEqual(live['loss'], old['loss'], places=14)
            self.assertEqual(live['positive_pooled_logit'], 4.)
            self.assertEqual(live['exact_annotation_logit'], 4. if live['offset_frames'] == 0 else -4.)
        self.assertEqual(len({r['loss'] for r in controls['shifted_pulse']}), 1)

    def test_loss_masks_edges_and_ignored_neighbors(self):
        values, targets = [-4.] * 41, [int(t == 20) for t in range(41)]
        values[21] = 4.
        loss, rows = audit.shift_loss(values, targets)
        self.assertEqual([r['center'] for r in rows], list(range(6, 35)))
        self.assertEqual([r['center'] for r in rows if not r['active']],
                         [t for t in range(14, 27) if t != 20])
        # 16 negative centers and one weight-19 positive; all have softplus(-4).
        self.assertAlmostEqual(loss, 35 * math.log1p(math.exp(-4)) / 29)
        masked, _ = audit.shift_loss(values, targets, mask=[0] * 41)
        self.assertEqual(masked, 0.)
        # Positive evidence farther than tolerance is no longer equivalent.
        values[21], values[24] = -4., 4.
        self.assertGreater(audit.shift_loss(values, targets)[0], loss)
        plain, _ = audit.shift_loss([0., 0.], [0, 1], tolerance=0)
        self.assertAlmostEqual(plain, 10 * math.log(2))

    def test_ordinary_weighted_bce_inverse_is_not_a_likelihood(self):
        for p in (.01, .1, .5, .9):
            for weight in (1, 19, 86):
                z = math.log(p / (1 - p)) + math.log(weight)
                self.assertAlmostEqual(audit.sigmoid(z - math.log(weight)), p)
                # Analytic population BCE derivative vanishes at this logit.
                s = audit.sigmoid(z)
                self.assertAlmostEqual(weight * p * (s - 1) + (1 - p) * s, 0.)
        for p0 in (.02, .2):
            # Same posterior .5 implies different likelihood ratio by prior.
            self.assertAlmostEqual((.5 / .5) / (p0 / (1 - p0)), (1 - p0) / p0)
        self.assertFalse(audit.controls()['ordinary_weighted_bce_only']['applicable_to_shift_tolerant_final0'])

    def test_sum_head_does_not_ensure_nested_probabilities(self):
        counter = audit.controls()['sum_head_counterexample']
        self.assertEqual(counter['beat'], counter['internal_u'] + counter['internal_v'])
        self.assertLess(counter['naive_plain_mass'], 0.)
        self.assertLess(counter['weight_offset_plain_mass'], 0.)
        counts = audit.head_counts([-2., 2., 1000.], [2., -2., 1001.])
        self.assertEqual(counts['downbeat_gt_beat'], 2)  # sigmoid saturation must not hide it
        self.assertEqual(counts['after_weight_offset_downbeat_gt_beat'], 1)
        self.assertEqual(counts, audit.head_counts(np.array([-2., 2., 1000.]), np.array([2., -2., 1001.])))
        json.dumps(audit.head_counts(np.array([-2., 2.]), np.array([2., -2.])))

    def test_class_normalization_is_not_observation_normalization(self):
        # Deliberately non-symmetric, normalized class-conditional sensor.
        sensor = [[.6, .3, .1], [.2, .5, .3], [.1, .2, .7]]
        priors = [.7, .2, .1]
        marginal = [sum(priors[c] * sensor[c][x] for c in range(3)) for x in range(3)]
        posterior = [[priors[c] * sensor[c][x] / marginal[x] for x in range(3)] for c in range(3)]
        for x in range(3):
            self.assertAlmostEqual(sum(posterior[c][x] for c in range(3)), 1.)
        self.assertNotAlmostEqual(sum(posterior[0]), 1.)
        for c in range(3):
            for x in range(3):
                recovered = posterior[c][x] / posterior[0][x] * priors[0] / priors[c]
                self.assertAlmostEqual(recovered, sensor[c][x] / sensor[0][x])

    def test_invalid_inputs_and_ast_default(self):
        for a, b in (([], []), ([1], []), ([math.nan], [1]), ([1], [math.inf])):
            with self.assertRaises(ValueError):
                audit.head_counts(a, b)
        for args in (([0], [0], None, 19, 3), ([0], [2], None, 19, 0),
                     ([math.inf], [0], None, 19, 0), ([0], [0], [], 19, 0)):
            with self.assertRaises(ValueError):
                audit.shift_loss(*args)
        self.assertTrue(audit.default_arg('class Model:\n def __init__(self, a=1, sum_head=True): pass',
                                          'Model', 'sum_head'))


if __name__ == '__main__':
    unittest.main()
