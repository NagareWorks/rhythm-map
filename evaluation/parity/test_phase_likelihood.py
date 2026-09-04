import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class PhaseLikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((ROOT / "evaluation/parity/phase-likelihood-v1.json").read_text())
        self.rows = {row["case"]: row for row in self.report["cases"]}

    def test_frozen_sources_and_identical_prior_inputs(self):
        for field, path in (
            ("scorer_source_sha256", "crates/rhythm-map-eval/examples/support/phase_likelihood.rs"),
            ("audit_source_sha256", "crates/rhythm-map-eval/examples/phase_likelihood.rs"),
        ):
            self.assertEqual(self.report[field], hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
        prior = json.loads((ROOT / "evaluation/parity/dropout-likelihood-v1.json").read_text())
        self.assertEqual(len(self.rows), 8)
        self.assertEqual(set(self.rows), {row["case"] for row in prior["cases"]})
        for row in prior["cases"]:
            self.assertEqual(row["input_f64_le_sha256"], self.rows[row["case"]]["input_f64_le_sha256"])
        for field in ("production_output_changed", "training_run", "holdout_opened",
                      "real_music_evaluated", "clock_decoder_implemented"):
            self.assertIs(self.report[field], False)

    def test_weak_success_and_identical_input_limit_remain_separate(self):
        self.assertEqual(sum(row["authored_path_unique_top"] for row in self.rows.values()), 7)
        for name, row in self.rows.items():
            self.assertEqual(row["authored_path_unique_top"], name != "constant_erased_alternating")
            for candidate in row["given_path_scores"]:
                self.assertEqual(candidate["score"]["scored_frames"], self.report["frames"])
                self.assertEqual(candidate["score"]["available_frames_in_unscored_cells"], 0)
        erased = self.rows["constant_erased_alternating"]
        slow = self.rows["half_speed_intact"]
        self.assertEqual(erased["given_path_scores"], slow["given_path_scores"])
        self.assertNotEqual(erased["authored_path"], slow["authored_path"])
        for control in self.report["meter_controls"]:
            scores = {x["meter"]: x["score"]["log_ratio_to_null"] for x in control["given_meter_scores"]}
            self.assertGreater(scores[4], max(scores[2], scores[8]))

    def test_noise_search_and_flat_cells_do_not_become_beat_confidence(self):
        for row in self.report["coherence_controls"]:
            measured = row["measurement"]
            self.assertGreater(measured["per_cell_max_score_not_valid_evidence"], 0)
            self.assertEqual(measured["shared_phase_marginal_log_ratio_to_null"] > 0,
                             row["case"] == "coherent_weak_pulses")
        self.assertEqual(self.report["flat_control"]["log_ratio_to_null"], 0)
        self.assertEqual(self.report["flat_control"]["neutral_flat_cells"], 48)


if __name__ == "__main__":
    unittest.main()
