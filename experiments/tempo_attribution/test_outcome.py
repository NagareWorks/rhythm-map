"""Audit the closed scalar diagnosis without private arrays or inference."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.phase_attribution.run import audit_metric
from experiments.tempo_attribution import run
from experiments.tempo_attribution.diagnostic import summarize


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((run.HERE / 'results-v1.json').read_bytes())
        cls.old = json.loads((run.FIT / 'results-v1.json').read_bytes())

    def test_sources_and_all_prerequisites_still_match(self):
        self.assertEqual(run.sha(run.HERE / 'results-v1.json'),
                         '4ef7f051af28c8bdf38dd4b34d983298aad80df540fd013aadd732a09d2a3814')
        self.assertEqual(self.result['source_sha256'], run.source_hashes())
        self.assertEqual(self.result['prerequisite_sha256'],
                         {p.relative_to(run.ROOT).as_posix(): s for p, s in run.PINS.items()})
        for path, digest in run.PINS.items():
            run.verify(path, digest)

    def test_every_population_variant_and_support_remains(self):
        keys = ('id', 'role', 'work', 'frames', 'packet_sha256', 'prediction_sha256')
        self.assertEqual([tuple(r[k] for k in keys) for r in self.result['cases']],
                         [tuple(r[k] for k in keys) for r in self.old['cases']['shared']])
        names = {a + '_' + v for a in run.ARCHITECTURES for v in run.VARIANTS}
        for row in self.result['cases']:
            self.assertEqual(set(row['native']), names)
            self.assertEqual(set(row['common']), names | {'raw', 'v1'})
            self.assertEqual(len({r['offset_shape']['cells'] for r in row['common'].values()}), 1)

    def test_all_720_original_metric_blocks_remain_unchanged(self):
        audited = 0
        for index, row in enumerate(self.result['cases']):
            for architecture in run.ARCHITECTURES:
                old = self.old['cases'][architecture][index]
                for support in ('native', 'common'):
                    for variant in run.VARIANTS:
                        if support == 'common' or variant in ('audio', 'zero'):
                            key = variant + '_common' if support == 'common' else variant
                            audit_metric(row[support][architecture + '_' + variant]['original'], old[key])
                            audited += 1
                for kind in ('raw', 'v1'):
                    audit_metric(row['common'][kind]['original'], old[kind + '_common'])
                    audited += 1
        self.assertEqual(audited, self.result['original_metric_blocks_audited'])
        self.assertEqual(audited, 720)

    def test_all_macros_recompute_with_missing_record_counts(self):
        self.assertEqual(self.result['summary'], summarize(self.result['cases']))

    def test_error_accounting_and_reference_only_partitions(self):
        for row in self.result['cases']:
            for support in ('native', 'common'):
                for result in row[support].values():
                    detail = result['offset_shape']
                    self.assertAlmostEqual(detail['log2_mse'], detail['squared_bias'] + detail['residual_log2_mse'])
                    self.assertEqual(detail['cells'], result['original']['reference_cells'])
                    self.assertAlmostEqual(sum(v for k, v in result['reference'].items()
                                               if k.startswith('fraction_')), 1.)
                    self.assertAlmostEqual(sum(v for k, v in result['octave'].items()
                                               if k.endswith('_fraction')), 1.)
                    for change in result['changes'].values():
                        self.assertEqual(change['all']['pairs'], change['large']['pairs'] + change['small']['pairs'])
                        self.assertEqual(change['large']['pairs'], change['all']['large_change_pairs'])
                        self.assertEqual(change['small']['large_change_pairs'], 0)

    def test_no_new_acceptance_or_model_execution_claim(self):
        for key in ('training', 'model_forward', 'weights_loaded', 'hidden_features_loaded',
                    'holdout_access', 'independent_acceptance', 'production_change'):
            self.assertIs(self.result[key], False)
        self.assertEqual(self.result['prior_gate'], self.old['gate'])
        self.assertEqual(self.result['prior_decision'], self.old['decision'])
        self.assertEqual(self.result['decision'], 'retained_tempo_diagnosis_only_no_automatic_fit_or_fusion')

    def test_wrong_prerequisite_stops_before_array_access(self):
        with patch.object(run, 'PINS', {run.INPUT: '0' * 64}), \
             patch.object(run.np, 'load', side_effect=AssertionError('must not open arrays')):
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                run.run(Path('missing-inputs'), Path('missing-predictions'))


if __name__ == '__main__':
    unittest.main()
