import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DropoutLikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((ROOT / "evaluation/parity/dropout-likelihood-v1.json").read_text())
        self.cases = {row["case"]: row for row in self.report["cases"]}

    def test_frozen_sources_scope_and_identical_input_witness(self):
        for field, path in (
            ("scorer_source_sha256", "crates/rhythm-map-eval/examples/support/dropout_likelihood.rs"),
            ("audit_source_sha256", "crates/rhythm-map-eval/examples/dropout_likelihood.rs"),
        ):
            self.assertEqual(self.report[field], hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
        self.assertEqual(len(self.cases), 8)
        self.assertEqual(self.report["fixed_missing_rate"], .1)
        self.assertEqual(self.report["log_density_measure"], "real_logit_axis")
        for field in ("production_output_changed", "training_run", "holdout_opened",
                      "real_music_evaluated", "clock_decoder_implemented"):
            self.assertIs(self.report[field], False)
        erased = self.cases["constant_erased_alternating"]
        slow = self.cases["half_speed_intact"]
        self.assertEqual(erased["input_f64_le_sha256"], slow["input_f64_le_sha256"])
        self.assertEqual(erased["given_path_scores"], slow["given_path_scores"])
        self.assertNotEqual(erased["authored_path"], slow["authored_path"])

    def test_intact_success_cannot_hide_weak_change_erasure(self):
        for name in ("constant_intact", "half_speed_intact", "double_speed_intact", "non_octave_intact"):
            self.assertTrue(self.cases[name]["authored_path_wins"])
        for name in ("constant_weak_alternating", "constant_erased_alternating",
                     "double_speed_weak_alternating", "constant_all_weak"):
            self.assertFalse(self.cases[name]["authored_path_wins"])
        self.assertEqual(self.cases["double_speed_weak_alternating"]["best_given_path"], "constant_125")
        self.assertEqual(self.cases["constant_all_weak"]["best_given_path"], "all_absent")
        for case in self.cases.values():
            self.assertEqual(len(case["given_path_scores"]), 5)
            for candidate in case["given_path_scores"]:
                self.assertEqual(candidate["score"]["scored_frames"], self.report["frames"])
                self.assertEqual(candidate["score"]["unavailable_frames"], 0)


if __name__ == "__main__":
    unittest.main()
