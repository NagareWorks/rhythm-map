"""Recompute published outcome decisions without musical inputs or inference."""
import copy
import json
from pathlib import Path
import unittest

from experiments.phase_sync_fit.register import parent_reports
from experiments.phase_sync_fit.run import same_support
from experiments.separated_evidence_fit.register import ARMS, sha, sources
from experiments.separated_evidence_fit.run import audit_receipts, execution_gate
from experiments.separated_evidence_fit.evaluation import summarize, compare

HERE = Path(__file__).resolve().parents[1]


class OutcomeAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((HERE / 'plan-v1.json').read_bytes())
        cls.report = json.loads((HERE / 'results-v1.json').read_bytes())
        cls.execution = json.loads((HERE / 'execution-v1.json').read_bytes())

    def test_frozen_identity_and_runtime(self):
        self.assertEqual(sha(HERE / 'plan-v1.json'),
                         'f7798c4b78f06af3eb378e21660fbda6f9b372b734b02ba855a557fafd940408')
        self.assertEqual(self.report['plan_sha256'], sha(HERE / 'plan-v1.json'))
        self.assertEqual(self.report['execution_sha256'], sha(HERE / 'execution-v1.json'))
        self.assertEqual(self.execution['plan'], self.plan)
        self.assertEqual(self.execution['plan_sha256'], self.report['plan_sha256'])
        self.assertEqual(self.report['source_sha256'], self.plan['source_sha256'])
        self.assertEqual(sources(), self.plan['source_sha256'])
        self.assertEqual(self.report['inputs_sha256'], self.plan['inputs_sha256'])
        self.assertEqual(self.plan['runtime']['torch'], '2.10.0+cu129')
        self.assertEqual(self.plan['runtime']['gpu'], 'Tesla T4')
        self.assertFalse(self.plan['old_weights_loaded'])

    def test_population_support_and_frozen_comparators(self):
        keys = ('id', 'role', 'work', 'frames', 'packet_sha256')
        parents = parent_reports()
        parents['protected'] = json.loads((HERE.parent / 'protected_phase_fit/results-v1.json').read_bytes())
        for rows in self.report['cases'].values():
            self.assertEqual([tuple(r[k] for k in keys) for r in rows],
                             [tuple(r[k] for k in keys) for r in self.plan['population']])
            self.assertEqual([sum(r['role'] == role for r in rows)
                              for role in ('fit', 'development', 'diagnostic')], [20, 5, 15])
            for name, parent in parents.items():
                for row, old in zip(rows, parent['cases']):
                    same_support(row, old)
                    self.assertEqual(row[name + '_common'], old['audio_common'])
        self.assertEqual([r['prediction_sha256'] for r in self.report['cases']['shared']],
                         [r['prediction_sha256'] for r in self.report['cases']['separated']])

    def test_every_metric_macro_and_decision_recomputes(self):
        report = self.report
        summary = summarize(report['cases'])
        self.assertEqual(summary, report['summary'])
        gate = compare(report['cases'], summary, report['fits'])
        gate['execution_complete'] = execution_gate(report['fits'], self.plan)
        self.assertEqual(gate, report['gate'])
        expected = ('propose_coherent_fusion_validation' if gate['supports_fusion_proposal']
                    else 'close_fixed_evidence_comparison_inspect_isolated_tasks_no_automatic_sweep')
        self.assertEqual(report['decision'], expected)

    def test_complete_native_update_receipts_and_work_weighting(self):
        population = [r for r in self.plan['population'] if r['role'] == 'fit']
        self.assertEqual([f['arm'] for f in self.report['fits']], list(ARMS))
        for fit in self.report['fits']:
            path = HERE / 'checks' / (fit['arm'] + '.receipts-v1.jsonl')
            self.assertEqual(sha(path), fit['journal_sha256'])
            self.assertEqual(audit_receipts(path, population, self.plan, fit['architecture']), fit['receipts'])

    def test_selection_is_per_task_complete_epoch_earliest_tie(self):
        for fit in self.report['fits']:
            self.assertEqual([h['paired_updates'] for h in fit['history']], list(range(5, 101, 5)))
            for task in ('tempo', 'phase'):
                best = min(fit['history'], key=lambda h: h['development'][task])
                self.assertEqual(fit['selected'][task]['epoch'], best['epoch'])
                self.assertEqual(fit['selected'][task]['loss'], best['development'][task])
            self.assertEqual(fit['export_parameters'], 47907)
            self.assertEqual(fit['export_trunk_evaluations'], 2)
            self.assertLessEqual(fit['elapsed_before_report_s'], fit['elapsed_s'])
            self.assertLess(fit['elapsed_s'], 1800)
            self.assertEqual(set(fit['snapshot_sha256']),
                             {'initial.pt', 'before.pt', 'after.pt', 'best-tempo.pt', 'best-phase.pt'})

    def test_incomplete_or_altered_execution_is_rejected(self):
        self.assertTrue(execution_gate(self.report['fits'], self.plan))
        self.assertFalse(execution_gate(self.report['fits'][:-1], self.plan))
        for index, field, value in ((0, 'updates', 99), (2, 'epochs', 19),
                                    (1, 'elapsed_s', 1801), (3, 'budget_exhausted', True)):
            altered = copy.deepcopy(self.report['fits'])
            altered[index][field] = value
            self.assertFalse(execution_gate(altered, self.plan))

    def test_whole_experiment_receipts_and_no_product_claim(self):
        report = self.report
        events = [json.loads(s) for s in (HERE / 'journal-v1.jsonl').read_text().splitlines()]
        self.assertEqual(len([e for e in events if e['event'] == 'epoch']), 80)
        self.assertEqual([e['recording'] for e in events if e['event'] == 'recording_evaluated'], list(range(1, 41)))
        self.assertEqual(events[-1], dict(event='experiment_complete', gate=report['gate']))
        self.assertEqual((report['optimizer_fits'], report['optimizer_calls_total'],
                          report['logical_training_record_presentations'], report['evidence_pair_evaluations']),
                         (4, 600, 1600, 400))
        self.assertLess(report['elapsed_before_report_s'], self.plan['watchdog_seconds'])
        self.assertTrue(report['training'])
        for name in ('independent_acceptance', 'encoder_inference', 'holdout_access',
                     'production_change', 'coherent_clock', 'automatic_retry'):
            self.assertFalse(report[name])


if __name__ == '__main__':
    unittest.main()
