import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parents[1]


class OutcomeTests(unittest.TestCase):
    def test_fixed_authored_result_and_source_identity(self):
        report = json.loads((DIRECTORY / 'authored-windows-v1.json').read_text())
        self.assertEqual(report['schema'], 'protected-update-authored-v1')
        self.assertEqual((report['returned_calls'], report['committed_steps'], report['moved_steps']), (40, 40, 40))
        self.assertTrue(report['gate_passed'])
        self.assertEqual(report['musical_updates'], 0)
        self.assertFalse(report['admission'])
        self.assertEqual([r['step'] for r in report['steps']], list(range(1, 41)))
        self.assertEqual(report['final'], report['steps'][-1]['losses'])
        self.assertTrue(all(report['final'][k] < report['initial'][k] for k in ('phase', 'advance')))
        actions = {}
        previous = report['initial']
        for step in report['steps']:
            self.assertEqual(step['finite_count_increased'], step['losses']['advance'] > previous['advance'])
            previous = step['losses']
            for r in step['receipt'].values():
                self.assertTrue(r['final_sign'] is None or r['final_sign'] <= 0)
                actions[r['action']] = actions.get(r['action'], 0) + 1
        self.assertEqual(actions, report['actions'])
        for name, expected in report['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)


if __name__ == '__main__':
    unittest.main()
