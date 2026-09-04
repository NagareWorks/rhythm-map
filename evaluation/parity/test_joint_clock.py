import hashlib
import json
from pathlib import Path
import unittest

from joint_clock_diagnosis import diagnose


ROOT = Path(__file__).resolve().parents[2]


class JointClockTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((ROOT / "evaluation/parity/joint-clock-v1.json").read_text())

    def test_frozen_truth_free_sources_and_prior_beat_inputs(self):
        for field, path in (
            ("decoder_source_sha256", "crates/rhythm-map-eval/examples/support/joint_clock.rs"),
            ("audit_source_sha256", "crates/rhythm-map-eval/examples/joint_clock.rs"),
        ):
            self.assertEqual(self.report[field], hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
        prior = json.loads((ROOT / "evaluation/parity/phase-likelihood-v1.json").read_text())
        old = {row["case"]: row for row in prior["cases"]}
        self.assertEqual(len(self.report["cases"]), 7)
        self.assertEqual(len(self.report["controls"]), 8)
        for row in self.report["cases"]:
            self.assertEqual(row["beat_f64_le_sha256"], old[row["case"]]["input_f64_le_sha256"])
        for field in ("production_output_changed", "training_run", "holdout_opened",
                      "real_music_evaluated", "truth_supplied_to_decoder"):
            self.assertIs(self.report[field], False)

    def test_trace_has_no_overlap_and_edges_are_not_removed(self):
        for row in self.report["cases"] + self.report["controls"]:
            for run in row["decoded"]["runs"]:
                ticks = run["map_ticks"]
                self.assertLessEqual(run["map_log_probability_given_clock"], 1e-9)
                self.assertEqual(run["clock_supported"], run["clock_log_ratio_to_null"] > 1e-9)
                self.assertTrue(ticks)
                span = run["map_complete_bar_span"]
                self.assertEqual(ticks[0]["frame"], span[0])
                cursor = span[0]
                for tick in ticks:
                    self.assertEqual(tick["frame"], cursor)
                    self.assertIn(tick["period_frames"], range(10, 76))
                    self.assertIn(tick["meter"], range(2, 8))
                    self.assertIn(tick["beat_in_bar"], range(tick["meter"]))
                    cursor += tick["period_frames"]
                self.assertEqual(cursor, span[1])
                self.assertGreaterEqual(span[0], run["start"])
                self.assertLessEqual(span[1], run["end"])
                self.assertEqual(run["edge_reference_frames"], span[0] - run["start"] + run["end"] - span[1])

    def test_missing_frames_split_graph_and_flat_evidence_is_not_a_detection(self):
        controls = {row["case"]: row["decoded"] for row in self.report["controls"]}
        flat = controls["flat"]["runs"][0]
        self.assertFalse(flat["clock_supported"])
        self.assertAlmostEqual(flat["clock_log_ratio_to_null"], 0, places=9)
        self.assertTrue(all(tick["prior_only"] for tick in flat["map_ticks"]))
        gap = controls["unavailable_gap"]
        self.assertEqual(gap["unavailable_spans"], [[480, 672]])
        self.assertEqual([(r["start"], r["end"]) for r in gap["runs"]], [(0, 480), (672, 1152)])
        middle = controls["flat_middle"]
        self.assertEqual(middle["unavailable_spans"], [])
        self.assertEqual(len(middle["runs"]), 1)

    def test_frozen_failure_decomposition_keeps_evidence_and_prior_separate(self):
        saved = json.loads((ROOT / "evaluation/parity/joint-clock-diagnosis-v1.json").read_text())
        measured = diagnose(ROOT / "evaluation/parity/joint-clock-v1.json")
        for field in ("input_report_sha256", "diagnosis_source_sha256"):
            self.assertEqual(saved[field], measured[field])
        for old, new in zip(saved["cases"], measured["cases"], strict=True):
            self.assertEqual(old["case"], new["case"])
            for group in ("map_score", "authored_timing_graph_score", "authored_minus_map"):
                for field in old[group]:
                    self.assertAlmostEqual(old[group][field], new[group][field], places=9)
        weak = next(r for r in measured["cases"] if r["case"] == "double_speed_weak_alternating")
        delta = weak["authored_minus_map"]
        self.assertGreater(delta["beat_evidence"], 0)
        self.assertLess(delta["duration_prior"], -delta["beat_evidence"])
        self.assertAlmostEqual(delta["bar_evidence"], 0, places=9)
        self.assertLess(delta["log_unnormalized_weight"], 0)

    def test_rejected_reference_failures_remain_explicit(self):
        rows = {row["case"]: row for row in self.report["cases"]}
        weak = rows["double_speed_weak_alternating"]
        self.assertEqual((weak["exact_frame_matches"], weak["authored_beats"]), (47, 64))
        self.assertEqual(weak["ticks_with_authored_period"], 31)
        for row in rows.values():
            run = row["decoded"]["runs"][0]
            self.assertEqual(run["map_ticks"][-1]["meter"], 3)
            self.assertEqual(run["edge_reference_frames"], 24)
        controls = {row["case"]: row["decoded"] for row in self.report["controls"]}
        self.assertTrue(controls["fixed_seed_noise"]["runs"][0]["clock_supported"])
        self.assertIn(72, [tick["period_frames"] for tick in controls["flat_middle"]["runs"][0]["map_ticks"]])
        changed = controls["intra_bar_change"]["runs"][0]["map_ticks"]
        self.assertEqual({t["period_frames"] for t in changed}, {24, 32})
        self.assertEqual({t["meter"] for t in changed}, {4})
        edge = controls["edge_phase_zero"]["runs"][0]
        self.assertEqual([t["frame"] for t in edge["map_ticks"]], list(range(0, 1152, 24)))
        self.assertEqual({t["meter"] for t in edge["map_ticks"]}, {4})
        self.assertEqual(edge["edge_reference_frames"], 0)
        triple = controls["three_beat_meter"]["runs"][0]["map_ticks"]
        self.assertEqual({t["meter"] for t in triple[:-2]}, {3})
        self.assertEqual(triple[-1]["meter"], 2)
        extras = controls["extra_offbeat_pulses"]["runs"][0]["map_ticks"]
        self.assertEqual([t["frame"] for t in extras], list(range(4, 1132, 24)))


if __name__ == "__main__":
    unittest.main()
