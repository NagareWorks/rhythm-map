"""Recompute the retained musical outcome without inputs, inference or fitting."""
import copy
import json
import math
from pathlib import Path
import unittest

import numpy as np

from experiments.coupled_clock.run import work_weights
from experiments.coupled_clock.measurement import macro
from experiments.phase_sync_fit.register import parent_reports
from experiments.phase_sync_fit.run import ALL_KINDS, comparison_gate, same_support
from experiments.protected_phase_fit.register import source_hashes, sha
from experiments.protected_phase_fit.recorder import conditioning
from experiments.protected_phase_fit.run import execution_gate, compare_previous

HERE = Path(__file__).resolve().parents[1]


class OutcomeAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((HERE / 'plan-v1.json').read_bytes())
        cls.report = json.loads((HERE / 'results-v1.json').read_bytes())
        cls.execution = json.loads((HERE / 'execution-v1.json').read_bytes())
        cls.receipts = [json.loads(line) for line in (HERE / 'receipts-v1.jsonl').read_text().splitlines()]

    def test_registered_identity_and_target_runtime(self):
        plan, report = self.plan, self.report
        self.assertEqual(sha(HERE / 'plan-v1.json'), report['plan_sha256'])
        self.assertEqual(sha(HERE / 'execution-v1.json'), report['execution_sha256'])
        self.assertEqual(self.execution['plan'], plan)
        self.assertEqual(plan['source_sha256'], source_hashes())
        for key in ('source_sha256', 'prerequisite_sha256', 'inputs_sha256'):
            self.assertEqual(report[key], plan[key])
        self.assertEqual(plan['runtime']['torch'], '2.10.0+cu129')
        self.assertEqual(plan['runtime']['gpu'], 'Tesla T4')
        self.assertFalse(plan['old_weights_loaded'])

    def test_all_population_support_and_baselines_preserved(self):
        rows = self.report['cases']
        keys = ('id', 'role', 'work', 'frames', 'packet_sha256')
        self.assertEqual([tuple(r[k] for k in keys) for r in rows],
                         [tuple(r[k] for k in keys) for r in self.plan['population']])
        self.assertEqual([sum(r['role'] == role for r in rows)
                          for role in ('fit', 'development', 'diagnostic')], [20, 5, 15])
        self.assertEqual(self.report['checkpoint_sequence_evaluations'], 200)
        for name, parent in parent_reports().items():
            for row, old in zip(rows, parent['cases']):
                same_support(row, old)
                self.assertEqual(row[name + '_common'], old['audio_common'])

    def test_every_macro_gate_and_previous_comparison_recomputes(self):
        report = self.report
        summary = {role: {kind: macro([r for r in report['cases'] if r['role'] == role], kind)
                          for kind in ALL_KINDS} for role in ('fit', 'development', 'diagnostic')}
        self.assertEqual(report['summary'], summary)
        gate = comparison_gate(report['cases'], summary, report['fits'])
        gate['protected_execution_complete'] = execution_gate(report['fits'], self.plan)
        gate['passed'] = all(v for k, v in gate.items() if k != 'passed')
        self.assertEqual(report['gate'], gate)
        self.assertEqual(report['previous_scaled_comparison'], compare_previous(report['cases'], summary))
        expected = ('fresh_independent_validation_required' if gate['passed']
                    else 'close_fixed_protected_phase_fit_no_automatic_sweep')
        self.assertEqual(report['decision'], expected)

    def test_complete_shuffles_weights_and_durable_counts(self):
        population = [r for r in self.plan['population'] if r['role'] == 'fit']
        weights = work_weights(population)
        for fit in self.report['fits']:
            events = [e for e in self.receipts if e['kind'] == fit['kind']]
            records = [e for e in events if e['event'] == 'record_gradient']
            updates = [e for e in events if e['event'] == 'update_completed']
            timing = [e for e in events if e['event'] == 'update_timing']
            self.assertEqual((len(records), len(updates), len(timing)), (800, 100, 100))
            self.assertEqual(conditioning(events), fit['conditioning'])
            self.assertEqual([e['seconds'] for e in timing], fit['update_seconds'])
            self.assertEqual([e['completed_updates'] for e in updates], list(range(1, 101)))
            for epoch in range(20):
                order = np.random.default_rng(self.plan['seed'] + epoch).permutation(20)
                for index, recording_index in enumerate(order):
                    row = population[recording_index]
                    for term_index, term in enumerate(('phase', 'advance')):
                        receipt = records[(epoch * 20 + index) * 2 + term_index]
                        self.assertEqual((receipt['recording'], receipt['term']), (row['id'], term))
                        self.assertEqual(receipt['cursor']['epoch'], epoch + 1)
                        self.assertEqual(receipt['cursor']['batch'], index // 4)
                        self.assertEqual(receipt['completed_updates'], epoch * 5 + index // 4)
                        self.assertEqual(receipt['scale'], weights[row['work']] / 4)
                        self.assertEqual(receipt['alignment_underflows'], 0)

    def test_every_protected_receipt_and_finite_loss_audit(self):
        for fit in self.report['fits']:
            events = [e for e in self.receipts if e['kind'] == fit['kind']]
            steps = [e for e in events if e['event'] == 'protected_step']
            moved = 0
            for number, step in enumerate(steps, 1):
                self.assertEqual(step['committed_steps'], number)
                self.assertEqual(step['completed_updates'], number)
                moved += any(g['moved'] for g in step['groups'].values())
                self.assertEqual(step['moved_steps'], moved)
                self.assertEqual(set(step['groups']), {'period_head', 'shared_cnn', 'phase_parameters'})
                for group in step['groups'].values():
                    self.assertTrue(group['final_sign'] is None or group['final_sign'] <= 0)
                    self.assertIn(group['action'], ('unchanged_proposal', 'projected', 'rounded_projection_veto'))
                    if group['action'] == 'rounded_projection_veto':
                        self.assertFalse(group['moved'])
                        self.assertEqual(group['final_sign'], 0)
                records = [e for e in events if e['event'] == 'post_step_record_loss'
                           and e['completed_updates'] == number]
                self.assertEqual(len(records), 4)
                self.assertEqual([e['cursor']['recording'] for e in records], step['cursor']['batch_recordings'])
                for term in ('phase', 'advance'):
                    self.assertTrue(math.isfinite(step['before_losses'][term]))
                    self.assertEqual(step['after_losses'][term],
                                     math.fsum(r['scale'] * r['losses'][term] for r in records))
                self.assertEqual(step['finite_count_increased'],
                                 step['after_losses']['advance'] > step['before_losses']['advance'])

    def test_complete_epoch_selection_and_exact_budget(self):
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
            self.assertIn('best.pt', fit['snapshot_sha256'])
        self.assertTrue(execution_gate(self.report['fits'], self.plan))
        for key, value in (('record_gradients', 799), ('committed_steps', 99), ('adverse_defined_groups', 1)):
            altered = copy.deepcopy(self.report['fits'])
            altered[0]['conditioning'][key] = value
            self.assertFalse(execution_gate(altered, self.plan))

    def test_completion_does_not_claim_product_or_independent_admission(self):
        report = self.report
        self.assertTrue(report['training'])
        self.assertEqual(report['optimizer_fits'], 2)
        self.assertTrue(report['gate']['protected_execution_complete'])
        for name in ('independent_acceptance', 'encoder_inference', 'holdout_access', 'production_change'):
            self.assertFalse(report[name])


if __name__ == '__main__':
    unittest.main()
