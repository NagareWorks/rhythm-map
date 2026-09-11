"""Audit measured reports and their frozen source/coverage contract, not timings in CI."""
import copy
import json
from pathlib import Path
import statistics
import unittest

import torch

from experiments.observed_adjoint.benchmark import cpu_gate, source_hashes
from experiments.observed_adjoint.fixtures import fixture

HERE = Path(__file__).resolve().parent


class CostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((HERE / 'contract-v1.json').read_bytes())
        cls.reports = {kind: json.loads((HERE / f'cost-{kind}-v1.json').read_bytes()) for kind in ('cpu', 'cuda')}

    def test_reports_pin_current_sources_and_recompute_all_samples(self):
        for kind, report in self.reports.items():
            self.assertEqual(report['device'], kind)
            self.assertEqual(report['source_sha256'], source_hashes())
            self.assertEqual(report['contract'], self.contract)
            self.assertEqual(report['optimizer_steps'], 8)
            self.assertFalse(report['musical_accuracy_measured'])
            self.assertFalse(report['automatic_retry'])
            self.assertNotIn(report['filesystem'], (None, 'tmpfs', 'ramfs'))
            self.assertEqual(set(report['cases']), {'natural', 'opposed'})
            for case, row in report['cases'].items():
                self.assertEqual(len(row['seconds']), 3)
                self.assertEqual(row['median_seconds'], statistics.median(row['seconds']))
                self.assertEqual(len(row['updates']), 3)
                self.assertEqual(len(row['initial_state_sha256']), 64)
                self.assertTrue(all(value > 0 for value in row['seconds']))
                for step in row['updates']:
                    self.assertEqual(step['bands'], [5, 5] if case == 'opposed' else [1, 1])
                    self.assertEqual(step['completed_updates'], 1)
                    self.assertEqual(step['clip_report']['alignment_underflows'], 0)
                    self.assertGreater(step['snapshot_bytes'], 0)
                self.assertEqual(row['feature_sha256'], self.reports['cpu']['cases'][case]['feature_sha256'])

    def test_cpu_acceptance_is_not_cuda_or_musical_admission(self):
        cpu, cuda = self.reports['cpu'], self.reports['cuda']
        self.assertTrue(cpu['cpu_gate_applies'])
        self.assertTrue(cpu['cpu_gate_pass'])
        self.assertTrue(cpu_gate(cpu['cases'], cpu['process_peak_rss_bytes'], self.contract, cpu['filesystem']))
        self.assertFalse(cuda['cpu_gate_applies'])
        self.assertIsNone(cuda['cpu_gate_pass'])
        self.assertEqual(cpu['environment']['torch'], '2.10.0+cpu')
        self.assertEqual(cuda['environment']['gpu'], 'Tesla T4')

    def test_gate_rejects_missing_coverage_unknown_memory_and_boundaries(self):
        cases = self.reports['cpu']['cases']
        for memory, filesystem in ((None, 'ext2/ext3'), (2 * 1024**3, 'ext2/ext3'), (100, 'tmpfs'), (100, None)):
            self.assertFalse(cpu_gate(cases, memory, self.contract, filesystem))
        for value in (2., float('nan'), float('inf'), 0.):
            bad = copy.deepcopy(cases)
            bad['natural']['median_seconds'] = value
            self.assertFalse(cpu_gate(bad, 100, self.contract, 'ext2/ext3'))
        for bands in ([], [3, 5]):
            bad = copy.deepcopy(cases)
            bad['opposed']['updates'][0]['bands'] = bands
            self.assertFalse(cpu_gate(bad, 100, self.contract, 'ext2/ext3'))
        bad = copy.deepcopy(cases)
        bad['opposed']['updates'] = []
        self.assertFalse(cpu_gate(bad, 100, self.contract, 'ext2/ext3'))

    def test_declared_fixture_geometry_weights_and_limits(self):
        contract = self.contract
        self.assertEqual((contract['cells_per_record'], contract['fps'], contract['cpu_threads']), (3000, 50, 4))
        self.assertEqual((contract['warmups'], contract['iterations']), (1, 3))
        self.assertEqual(contract['cpu_median_observed_update_limit_seconds'], 2.)
        self.assertEqual(contract['cpu_process_peak_rss_limit_bytes'], 2 * 1024**3)
        self.assertEqual(contract['opposed_minimum_vjp_bands_per_record'], 4)
        model, rows = fixture('natural', 3, contract['seed'])
        self.assertEqual(len(rows), contract['records_per_update'])
        self.assertEqual([row['scale'] for row in rows], contract['record_weights'])
        self.assertEqual([float(row['payload']['reference'][0, 0]) for row in rows], contract['reference_origin_cycles'])
        self.assertEqual(rows[0]['payload']['features'].shape, (1, 4, contract['feature_width']))
        self.assertEqual(rows[0]['payload']['features'].dtype, torch.float32)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 24037)


if __name__ == '__main__':
    unittest.main()
