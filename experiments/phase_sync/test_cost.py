"""Audit the measured, synthetic-only cost gate without rerunning timings."""
import json
from pathlib import Path
import statistics
import unittest

from experiments.phase_sync.benchmark import cpu_gate, source_hashes

HERE = Path(__file__).resolve().parent


class CostReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((HERE / 'contract-v1.json').read_bytes())
        cls.reports = {kind: json.loads((HERE / f'cost-{kind}-v1.json').read_bytes())
                       for kind in ('cpu', 'cuda')}

    def test_measured_sources_and_contract_match(self):
        for kind, report in self.reports.items():
            self.assertEqual(report['device'], kind)
            self.assertEqual(report['source_sha256'], source_hashes())
            self.assertEqual(report['contract'], self.contract)
        self.assertEqual(self.reports['cpu']['feature_sha256'], self.reports['cuda']['feature_sha256'])
        self.assertEqual(self.contract['cells'], 3000)
        self.assertEqual(self.contract['cpu_threads'], 4)
        self.assertEqual(self.contract['iterations'], 5)

    def test_summaries_recompute_and_all_models_present(self):
        for report in self.reports.values():
            self.assertEqual(set(report['models']), {'direct', 'integrated', 'synchronized'})
            for name, expected in (('direct', 24003), ('integrated', 23970), ('synchronized', 24037)):
                model = report['models'][name]
                self.assertEqual(model['parameters'], expected)
                self.assertEqual(len(model['seconds']), 5)
                self.assertTrue(all(v > 0 for v in model['seconds']))
                self.assertEqual(model['median_seconds'], statistics.median(model['seconds']))
            sync = report['models']['synchronized']['median_seconds']
            for baseline in ('direct', 'integrated'):
                self.assertAlmostEqual(report[f'synchronized_over_{baseline}_ratio'],
                                       sync / report['models'][baseline]['median_seconds'])

    def test_cpu_gate_pass_is_not_gpu_or_musical_admission(self):
        cpu, cuda = self.reports['cpu'], self.reports['cuda']
        self.assertTrue(cpu['cpu_gate_applies'])
        self.assertTrue(cpu['cpu_gate_pass'])
        self.assertTrue(cpu_gate(cpu['models']['synchronized']['median_seconds'],
                                 cpu['process_peak_rss_bytes'], self.contract))
        self.assertFalse(cuda['cpu_gate_applies'])
        self.assertIsNone(cuda['cpu_gate_pass'])
        self.assertEqual(cpu['environment']['torch'], '2.10.0+cpu')
        self.assertIsNone(cpu['environment']['cuda_runtime'])
        self.assertEqual(cuda['environment']['gpu'], 'Tesla T4')
        for report in self.reports.values():
            self.assertFalse(report['musical_accuracy_measured'])
            self.assertEqual(report['optimizer_steps'], 0)

    def test_budget_boundaries_and_unknown_memory_fail_closed(self):
        self.assertEqual(self.contract['cpu_median_forward_backward_limit_seconds'], 1.)
        self.assertEqual(self.contract['cpu_process_peak_rss_limit_bytes'], 2 * 1024 ** 3)
        for seconds, memory in ((1., 100), (.1, 2 * 1024 ** 3), (.1, None),
                                (float('nan'), 100), (float('inf'), 100), (0., 100), (.1, 0)):
            self.assertFalse(cpu_gate(seconds, memory, self.contract))


if __name__ == '__main__':
    unittest.main()
