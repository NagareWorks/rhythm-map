"""Authored tests only; never open private calibration or holdout data."""
import copy
import hashlib
import json
import math
from pathlib import Path
import unittest

from clock_phase_evidence import BANDS, FEATURES, feature_summary, label_case, matches, phase_features


def observations(beats, peaks=()):
    return dict(beats=[dict(time_s=t) for t in beats], beat_candidates=[], onsets=[
        dict(time_s=i / 100, **{b: float(any(abs(i / 100 - p) < 1e-9 for p in peaks)) for b in BANDS})
        for i in range(601)])


class ClockPhaseEvidenceTests(unittest.TestCase):
    def test_frozen_report_identity_and_complete_denominators(self):
        root = Path(__file__).resolve().parent
        report = json.loads((root / "clock-phase-evidence-v1.json").read_bytes())
        self.assertEqual(report["source_sha256"], hashlib.sha256((root / "clock_phase_evidence.py").read_bytes()).hexdigest())
        self.assertEqual(report["auc_source_sha256"], hashlib.sha256((root / "candidate_evidence_audit.py").read_bytes()).hexdigest())
        self.assertEqual([len(c["cases"]) for c in report["cohorts"]], [15, 25])
        for cohort in report["cohorts"]:
            self.assertEqual(cohort["raw_intervals"], sum(cohort["classes"].values()))
            for feature in cohort["features"].values():
                self.assertEqual(feature["positive_available"] + feature["positive_missing"], cohort["classes"]["missing_one"])
                self.assertEqual(feature["negative_available"] + feature["negative_missing"], cohort["classes"]["one_beat"])
            for values in cohort["matched_sample_comparison"].values():
                self.assertEqual(len({(v["positive_available"], v["negative_available"]) for v in values.values()}), 1)
        self.assertFalse(report["inference_run"] or report["training_run"] or report["inferred_beats_emitted"])

    def test_midpoint_and_quadrature_have_opposite_contrast(self):
        positive = phase_features(observations([1., 2.], [1.5]))[0]
        negative = phase_features(observations([1., 2.], [1.25, 1.75]))[0]
        self.assertEqual(positive["contrast_strength"], 1.)
        self.assertEqual(negative["contrast_strength"], -1.)

    def test_short_interval_windows_do_not_overlap(self):
        feature = phase_features(observations([1., 1.16], [1.04, 1.12]))[0]
        self.assertEqual(feature["midpoint_strength"], 0.)
        self.assertEqual(feature["contrast_strength"], -1.)

    def test_missing_evidence_is_not_silence(self):
        obs = observations([1., 2.])
        obs["onsets"] = []
        self.assertTrue(all(v is None for v in phase_features(obs)[0].values()))
        silent = phase_features(observations([1., 2.]))[0]
        self.assertEqual(silent["contrast_strength"], 0.)

    def test_no_extrapolation_or_partial_edge_window(self):
        row = phase_features(observations([5., 9.], [5.5]))[0]
        self.assertIsNone(row["midpoint_strength"])
        self.assertIsNone(row["contrast_strength"])

    def test_five_interval_context_is_fixed_and_edges_remain_missing(self):
        rows = phase_features(observations([0., 1., 2., 3., 4., 5.], [.5, 1.5, 2.5, 3.5, 4.5]))
        self.assertEqual(rows[2]["sequence_strength"], 1.)
        self.assertTrue(all(rows[i]["sequence_strength"] is None for i in (0, 1, 3, 4)))

    def test_nonfinite_or_unsorted_evidence_is_rejected(self):
        obs = observations([1., 2.])
        obs["onsets"][0]["strength"] = math.nan
        with self.assertRaises(ValueError):
            phase_features(obs)
        with self.assertRaises(ValueError):
            phase_features(observations([2., 1.]))

    def test_labels_come_after_features_and_preserve_unreachable_misses(self):
        obs = observations([0., 1., 2., 3.])
        case = dict(observations=obs, truth_times_s=[0., .3, 1., 2., 3.], beat_tolerance_s=.07,
                    raw_truth_pairs=[[0, 0], [1, 2], [2, 3], [3, 4]])
        original = copy.deepcopy(obs)
        features = phase_features(obs)
        rows, counts = label_case(case)
        self.assertEqual([r["features"] for r in rows], features)
        self.assertEqual(obs, original)
        self.assertEqual(rows[0]["label"], "missing_one")
        self.assertFalse(rows[0]["midpoint_reaches_truth"])
        self.assertFalse(rows[0]["missed_truth_has_candidate"])
        self.assertEqual(counts["classes"], dict(missing_one=1, one_beat=2))
        self.assertEqual(counts["missing_one_midpoint_unreachable"], 1)

    def test_unmatched_anchors_are_not_false_negative_labels(self):
        obs = observations([0., .4, 1., 2.])
        case = dict(observations=obs, truth_times_s=[0., 1., 2.], beat_tolerance_s=.07,
                    raw_truth_pairs=[[0, 0], [2, 1], [3, 2]])
        rows, counts = label_case(case)
        self.assertEqual(counts["classes"], dict(unmatched_anchor=2, one_beat=1))
        self.assertEqual(feature_summary(rows, "contrast_strength")["positive_available"], 0)

    def test_replay_identity_mismatch_fails_closed(self):
        case = dict(observations=observations([0., 1.]), truth_times_s=[0., 1.],
                    beat_tolerance_s=.07, raw_truth_pairs=[])
        with self.assertRaises(ValueError):
            label_case(case)

    def test_chronological_match_preserves_identities(self):
        self.assertEqual(matches([0., .03, 1.], [0., 1.], .07), [[0, 0], [2, 1]])

    def test_auc_direction_is_not_flipped_and_missing_values_are_counted(self):
        rows = [dict(label=label, features=dict.fromkeys(FEATURES, value))
                for label, value in (("missing_one", 0.), ("missing_one", None), ("one_beat", 1.))]
        result = feature_summary(rows, "contrast_strength")
        self.assertEqual(result["auc_larger_favors_missing"], 0.)
        self.assertEqual(result["positive_missing"], 1)
        self.assertIsNone(feature_summary([], "contrast_strength")["auc_larger_favors_missing"])


if __name__ == "__main__":
    unittest.main()
