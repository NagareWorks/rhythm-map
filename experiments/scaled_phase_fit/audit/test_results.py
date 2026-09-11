"""Recompute retained outcome/receipts without music inputs, forward or training."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.coupled_clock.run import work_weights
from experiments.coupled_clock.measurement import macro
from experiments.phase_sync_fit.register import parent_reports
from experiments.phase_sync_fit.run import ALL_KINDS, comparison_gate, same_support
from experiments.scaled_phase_fit.register import source_hashes, PINNED, sha
from experiments.scaled_phase_fit.recorder import conditioning
from experiments.scaled_phase_fit.run import execution_gate

HERE = Path(__file__).resolve().parents[1]


class OutcomeAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((HERE / 'plan-v1.json').read_bytes())
        cls.report = json.loads((HERE / 'results-v1.json').read_bytes())
        cls.execution = json.loads((HERE / 'execution-v1.json').read_bytes())
        cls.receipts = [json.loads(line) for line in (HERE / 'receipts-v1.jsonl').read_text().splitlines()]

    def test_plan_source_runtime_and_previous_failure_remain_fixed(self):
        plan, report = self.plan, self.report
        self.assertEqual(sha(HERE / 'plan-v1.json'), report['plan_sha256'])
        self.assertEqual(sha(HERE / 'execution-v1.json'), report['execution_sha256'])
        self.assertEqual(self.execution['plan'], plan)
        self.assertEqual(plan['source_sha256'], source_hashes())
        self.assertEqual(report['source_sha256'], plan['source_sha256'])
        self.assertEqual(report['prerequisite_sha256'], PINNED)
        self.assertEqual(plan['runtime']['torch'], '2.10.0+cu129')
        self.assertEqual(plan['runtime']['gpu'], 'Tesla T4')
        self.assertTrue(plan['previous_run_closed'])

    def test_all_population_support_and_baselines_are_preserved(self):
        rows = self.report['cases']
        self.assertEqual([(r['id'], r['role'], r['work'], r['frames'], r['packet_sha256']) for r in rows],
                         [(r['id'], r['role'], r['work'], r['frames'], r['packet_sha256']) for r in self.plan['population']])
        self.assertEqual([sum(r['role'] == role for r in rows)
                          for role in ('fit', 'development', 'diagnostic')], [20, 5, 15])
        self.assertEqual(self.report['checkpoint_sequence_evaluations'], 200)
        for name, parent in parent_reports().items():
            for row, old in zip(rows, parent['cases']):
                same_support(row, old)
                self.assertEqual(row[name + '_common'], old['audio_common'])

    def test_every_metric_macro_and_accuracy_gate_recomputes(self):
        report = self.report
        summary = {role: {kind: macro([r for r in report['cases'] if r['role'] == role], kind) for kind in ALL_KINDS}
                   for role in ('fit', 'development', 'diagnostic')}
        self.assertEqual(report['summary'], summary)
        gate = comparison_gate(report['cases'], summary, report['fits'])
        gate['observed_execution_complete'] = execution_gate(report['fits'], self.plan)
        gate['passed'] = all(v for k, v in gate.items() if k != 'passed')
        self.assertEqual(report['gate'], gate)
        expected = ('fresh_independent_validation_required' if gate['passed']
                    else 'close_fixed_scaled_phase_fit_no_automatic_sweep')
        self.assertEqual(report['decision'], expected)

    def test_complete_shuffles_weights_and_durable_update_counts(self):
        population = [r for r in self.plan['population'] if r['role'] == 'fit']
        weights = work_weights(population)
        for fit in self.report['fits']:
            events = [e for e in self.receipts if e['kind'] == fit['kind']]
            records = [e for e in events if e['event'] == 'record_gradient']
            updates = [e for e in events if e['event'] == 'update_completed']
            timing = [e for e in events if e['event'] == 'update_timing']
            self.assertEqual((len(records), len(updates), len(timing)), (400, 100, 100))
            self.assertEqual(conditioning(events), fit['conditioning'])
            self.assertEqual([e['seconds'] for e in timing], fit['update_seconds'])
            self.assertEqual([e['completed_updates'] for e in updates], list(range(1, 101)))
            for epoch in range(20):
                order = np.random.default_rng(self.plan['seed'] + epoch).permutation(20)
                for index, recording_index in enumerate(order):
                    row, receipt = population[recording_index], records[epoch * 20 + index]
                    self.assertEqual(receipt['recording'], row['id'])
                    self.assertEqual(receipt['cursor']['epoch'], epoch + 1)
                    self.assertEqual(receipt['cursor']['batch'], index // 4)
                    self.assertEqual(receipt['completed_updates'], epoch * 5 + index // 4)
                    self.assertEqual(receipt['scale'], weights[row['work']] / 4)
                    self.assertEqual(receipt['alignment_underflows'], 0)
            self.assertTrue(execution_gate(self.report['fits'], self.plan))
            changed = copy.deepcopy(self.report['fits'])
            changed[0]['conditioning']['record_gradients'] = 399
            self.assertFalse(execution_gate(changed, self.plan))

    def test_checkpoint_selection_and_io_budget_are_not_partial_epoch_or_lowest_training_loss(self):
        for fit in self.report['fits']:
            history = fit['history']
            self.assertEqual([r['epoch'] for r in history], list(range(1, 21)))
            self.assertEqual([r['updates'] for r in history], list(range(5, 101, 5)))
            selected = min(history, key=lambda r: r['development_loss'])
            self.assertEqual(fit['selected_epoch'], selected['epoch'])
            self.assertEqual(fit['development_loss'], selected['development_loss'])
            self.assertEqual(fit['initial_state_sha256'], self.plan['initial_state_sha256'])
            self.assertLessEqual(fit['elapsed_before_report_s'], fit['elapsed_s'])
            self.assertLess(fit['elapsed_s'], 1800)
            self.assertEqual(len(fit['fit_evidence_sha256']), 64)
            self.assertIn('best.pt', fit['snapshot_sha256'])

    def test_numerical_completion_does_not_claim_release_or_independent_admission(self):
        report = self.report
        self.assertTrue(report['training'])
        self.assertEqual(report['optimizer_fits'], 2)
        self.assertTrue(report['gate']['observed_execution_complete'])
        for name in ('independent_acceptance', 'encoder_inference', 'holdout_access', 'production_change'):
            self.assertFalse(report[name])


if __name__ == '__main__':
    unittest.main()
