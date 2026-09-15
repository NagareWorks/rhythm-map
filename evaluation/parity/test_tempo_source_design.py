"""Frozen source intervention and role separation; no torch dependency."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.tempo_source_expansion import design


class SourceDesignTests(unittest.TestCase):
    def test_frozen_schedule_replays(self):
        value = design.plan()
        self.assertEqual(value, json.loads((design.HERE/'plan-v1.json').read_bytes()))
        self.assertEqual(len(value['schedule']), 100)
        self.assertEqual(len(value['primary_record_draw_counts']), 93)
        self.assertEqual(sorted(value['primary_record_draw_counts'].values()), [1]*86+[2]*7)

    def test_each_epoch_preserves_all_twenty_old_fit_draws(self):
        p = design.plan()
        fit = {k for k, v in p['old_roles'].items() if v['role'] == 'fit'}
        for epoch in range(1, 21):
            rows = [r for r in p['schedule'] if r['epoch'] == epoch]
            self.assertEqual(len(rows), 5)
            self.assertEqual(set(i for r in rows for i in r['shared_fit_ids']), fit)
            self.assertTrue(all(r['control_extra_id'] in fit for r in rows))

    def test_no_development_or_diagnostic_admission(self):
        old = json.loads((design.ROOT/'experiments/coupled_clock/inputs-v1.json').read_bytes())
        records = [r for r in old['cases'] if r['role'] == 'fit']
        records[0] = next(r for r in old['cases'] if r['role'] == 'development')
        with self.assertRaises(ValueError):
            design.schedule(records, [f'{i:04d}' for i in range(1, 94)])

    def test_favorable_subset_is_rejected(self):
        old = json.loads((design.ROOT/'experiments/coupled_clock/inputs-v1.json').read_bytes())
        records = [r for r in old['cases'] if r['role'] == 'fit']
        with self.assertRaises(ValueError):
            design.schedule(records, ['0001'])

    def test_does_not_claim_training_admission_or_more_independent_works(self):
        p = design.plan()
        self.assertEqual(p['optimizer_steps'], 0)
        self.assertEqual(p['encoder_calls'], 0)
        self.assertTrue(p['admission_pending'])
        self.assertFalse(p['holdout_access'] or p['production_change'])
        self.assertEqual({r['primary_extra_group'] for r in p['schedule']}, {'brid-corpus-v1'})


if __name__ == '__main__':
    unittest.main()
