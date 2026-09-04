import hashlib
import json
from pathlib import Path
import unittest

from time_clock_diagnosis import RATE, diagnose

ROOT = Path(__file__).resolve().parents[2]


class TimeClockTests(unittest.TestCase):
    def setUp(self):
        self.current = json.loads((ROOT / "evaluation/parity/time-clock-v1.json").read_bytes())
        self.previous = json.loads((ROOT / "evaluation/parity/joint-clock-v1.json").read_bytes())

    def test_source_frozen_prior_only_intervention_and_unchanged_controls(self):
        for field, path in (
            ("decoder_source_sha256", "crates/rhythm-map-eval/examples/support/time_clock.rs"),
            ("audit_source_sha256", "crates/rhythm-map-eval/examples/time_clock.rs"),
            ("prior_source_sha256", "crates/rhythm-map-eval/examples/support/time_prior.rs"),
        ):
            self.assertEqual(self.current[field], hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
        self.assertAlmostEqual(self.current["rate_per_frame"], RATE, places=12)
        self.assertEqual(len(self.current["cases"]), 7)
        self.assertEqual(len(self.current["controls"]), 8)
        for kind in ("cases", "controls"):
            for old, new in zip(self.previous[kind], self.current[kind], strict=True):
                for field in ("case", "beat_f64_le_sha256", "bar_f64_le_sha256"):
                    self.assertEqual(old[field], new[field])
                for decoded in (old["decoded"], new["decoded"]):
                    self.assertTrue(decoded["runs"])
                self.assertEqual(old["decoded"]["unavailable_spans"], new["decoded"]["unavailable_spans"])
                self.assertEqual([(r["start"], r["end"]) for r in old["decoded"]["runs"]],
                                 [(r["start"], r["end"]) for r in new["decoded"]["runs"]])
        for field in ("production_output_changed", "training_run", "holdout_opened",
                      "real_music_evaluated", "truth_supplied_to_decoder"):
            self.assertIs(self.current[field], False)
        sources = [(ROOT / f"crates/rhythm-map-eval/examples/support/{name}_clock.rs").read_text()
                   for name in ("joint", "time")]
        # Exact same emissions, output contract, availability and domain checks.
        self.assertEqual(*[s.split("/// Same cyclic", 1)[1].split("fn search(", 1)[0] for s in sources])

    def test_independent_traceback_and_complete_duration_cost(self):
        saved = json.loads((ROOT / "evaluation/parity/time-clock-diagnosis-v1.json").read_bytes())
        measured = diagnose()
        for key in ("current_report_sha256", "previous_report_sha256", "diagnosis_source_sha256",
                    "shared_diagnosis_source_sha256"):
            self.assertEqual(saved[key], measured[key])
        for cost in measured["constant_duration_costs"].values():
            self.assertAlmostEqual(cost, -1152 * RATE, places=10)
        for old, new in zip(saved["cases"], measured["cases"], strict=True):
            self.assertEqual(old["case"], new["case"])
            for group in ("current_map", "old_map_with_new_prior", "authored_timing_with_new_prior", "authored_minus_old_map"):
                for key in old[group]:
                    self.assertAlmostEqual(old[group][key], new[group][key], places=9)

    def test_duration_invariance_does_not_guarantee_true_weak_jumps_win(self):
        measured = diagnose()
        weak = next(r for r in measured["cases"] if r["case"] == "double_speed_weak_alternating")
        delta = weak["authored_minus_old_map"]
        self.assertAlmostEqual(delta["beat_evidence"], 4.8532892342987, places=9)
        self.assertAlmostEqual(delta["duration_prior"], -10.569343596051482, places=9)
        self.assertAlmostEqual(delta["log_unnormalized_weight"], -5.716054361752782, places=9)
        self.assertAlmostEqual(delta["bar_evidence"], 0, places=9)
        self.assertAlmostEqual(delta["meter_prior"], 0, places=9)

    def test_flat_tempo_improvement_does_not_hide_unchanged_or_worse_controls(self):
        for old, new in zip(self.previous["cases"], self.current["cases"], strict=True):
            self.assertEqual(old["decoded"]["runs"][0]["map_ticks"], new["decoded"]["runs"][0]["map_ticks"])
            for key in ("exact_frame_matches", "authored_beats", "ticks_with_authored_period"):
                self.assertEqual(old[key], new[key])
        before = {r["case"]: r["decoded"] for r in self.previous["controls"]}
        after = {r["case"]: r["decoded"] for r in self.current["controls"]}
        middle = after["flat_middle"]["runs"][0]
        self.assertEqual([t["frame"] for t in middle["map_ticks"]], list(range(4, 1132, 24)))
        self.assertEqual({t["period_frames"] for t in middle["map_ticks"]}, {24})
        self.assertEqual(sum(t["prior_only"] for t in middle["map_ticks"]), 7)
        self.assertIn(6, {t["meter"] for t in middle["map_ticks"]})
        self.assertEqual(middle["edge_reference_frames"], 24)
        noise = after["fixed_seed_noise"]["runs"][0]
        self.assertTrue(noise["clock_supported"])
        self.assertGreater(noise["clock_log_ratio_to_null"], before["fixed_seed_noise"]["runs"][0]["clock_log_ratio_to_null"])
        flat = after["flat"]["runs"][0]
        self.assertFalse(flat["clock_supported"])
        self.assertAlmostEqual(flat["clock_log_ratio_to_null"], 0, places=10)
        self.assertTrue(all(t["prior_only"] for t in flat["map_ticks"]))


if __name__ == "__main__":
    unittest.main()
