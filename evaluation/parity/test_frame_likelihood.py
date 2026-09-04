import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class FrameLikelihoodReportTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((ROOT / "evaluation/parity/frame-likelihood-v1.json").read_text())

    def test_source_identity_and_authored_coverage(self):
        for field, relative in (
            ("scorer_source_sha256", "crates/rhythm-map-eval/examples/support/frame_likelihood.rs"),
            ("audit_source_sha256", "crates/rhythm-map-eval/examples/frame_likelihood.rs"),
        ):
            self.assertEqual(self.report[field], hashlib.sha256((ROOT / relative).read_bytes()).hexdigest())
        rows = self.report["ideal_meter_phase_cases"]
        self.assertEqual(len(rows), 27)
        self.assertEqual({(r["authored_meter"], r["authored_phase"]) for r in rows},
                         {(meter, phase) for meter in range(2, 8) for phase in range(meter)})
        for row in rows:
            self.assertEqual((row["best_meter"], row["best_phase"]), (row["authored_meter"], row["authored_phase"]))
            self.assertGreater(row["score_margin_not_confidence"], 0)
        self.assertEqual(len(self.report["contradiction_cases"]), 5)
        for row in self.report["contradiction_cases"]:
            self.assertEqual(row["scored_frames"], self.report["frames"])
            self.assertGreater(row["score_loss_not_confidence"], 0)
        self.assertEqual(len(self.report["corruption_cases"]), 2)
        self.assertEqual(self.report["flat_head_abstention_cases"], 3)

    def test_weak_omission_failure_cannot_be_hidden_by_restricted_ranking(self):
        weak = self.report["weak_repeated_bar_diagnostic"]
        self.assertTrue(weak["correct_unique_top_among_meters_2_through_7"])
        self.assertFalse(weak["correct_beats_omission_hypotheses"])
        self.assertGreater(weak["omitted_alternate_bars_log_score"], weak["correct_log_score"])
        self.assertGreater(weak["no_bars_log_score"], weak["omitted_alternate_bars_log_score"])
        for field in ("production_output_changed", "training_run", "holdout_opened",
                      "real_music_evaluated", "clock_decoder_implemented"):
            self.assertIs(self.report[field], False)


if __name__ == "__main__":
    unittest.main()
