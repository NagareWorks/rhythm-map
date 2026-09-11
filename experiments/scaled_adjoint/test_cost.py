"""Audit fixed measurement/source identity; never rerun a timing assertion in CI."""
import json
from pathlib import Path
import statistics
import unittest

from experiments.phase_sync.benchmark import cpu_gate
from experiments.scaled_adjoint.benchmark import source_hashes

HERE = Path(__file__).resolve().parent


class CostTests(unittest.TestCase):
    def test_measurements_recompute_and_pin_all_sources(self):
        contract = json.loads((HERE / 'contract-v1.json').read_bytes())
        hashes = source_hashes()
        reports = [json.loads((HERE / f'cost-{kind}-v1.json').read_bytes()) for kind in ('cpu', 'cuda')]
        self.assertEqual(reports[0]['feature_sha256'], reports[1]['feature_sha256'])
        for report, kind in zip(reports, ('cpu', 'cuda')):
            self.assertEqual(report['device'], kind)
            self.assertEqual(report['source_sha256'], hashes)
            self.assertEqual(report['contract'], contract)
            self.assertEqual(len(report['seconds']), contract['iterations'])
            self.assertEqual(report['median_seconds'], statistics.median(report['seconds']))
            self.assertTrue(all(s > 0 for s in report['seconds']))
            self.assertEqual(report['parameters'], 24037)
            self.assertEqual(len(report['iterations']), contract['iterations'])
            self.assertTrue(all(step == report['iterations'][0] for step in report['iterations']))
            # This fixture exercises one band, NOT the maximum permitted cost.
            self.assertTrue(all(step['neural_vjp_passes'] == 1 and step['alignment_underflows'] == 0
                                for step in report['iterations']))
            self.assertFalse(report['musical_accuracy_measured'])
            self.assertFalse(report['recorder_io_measured'])
            self.assertEqual(report['optimizer_steps'], 0)

    def test_cpu_gate_is_separate_from_cuda_and_accuracy(self):
        cpu, cuda = [json.loads((HERE / f'cost-{kind}-v1.json').read_bytes()) for kind in ('cpu', 'cuda')]
        self.assertTrue(cpu['cpu_gate_applies'])
        self.assertTrue(cpu['cpu_gate_pass'])
        self.assertTrue(cpu_gate(cpu['median_seconds'], cpu['process_peak_rss_bytes'], cpu['contract']))
        self.assertFalse(cuda['cpu_gate_applies'])
        self.assertIsNone(cuda['cpu_gate_pass'])
        self.assertEqual(cpu['environment']['torch'], '2.10.0+cpu')
        self.assertIsNone(cpu['environment']['cuda_runtime'])
        self.assertEqual(cuda['environment']['gpu'], 'Tesla T4')

    def test_cost_contract_limits_are_predeclared(self):
        contract = json.loads((HERE / 'contract-v1.json').read_bytes())
        self.assertEqual((contract['cells'], contract['fps'], contract['cpu_threads']), (3000, 50, 4))
        self.assertEqual((contract['warmups'], contract['iterations']), (1, 5))
        self.assertEqual(contract['cpu_median_forward_backward_limit_seconds'], 1.)
        self.assertEqual(contract['cpu_process_peak_rss_limit_bytes'], 2 * 1024 ** 3)
        for seconds, memory in ((1., 100), (.1, 2 * 1024 ** 3), (.1, None), (float('nan'), 100)):
            self.assertFalse(cpu_gate(seconds, memory, contract))


if __name__ == '__main__':
    unittest.main()
