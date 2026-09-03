import copy
import json
import unittest

from candidate_evidence_audit import (
    DIRECTORY, ROOT, DOMINANCE, POSITIVE, SHIPPING, annotation_label, auc, build_summary,
    candidate_features, case_rows, cohort_summary, digest, dominate, validate_inputs,
)


def beat(time_s, confidence=.8):
    return dict(time_s=time_s, confidence=confidence, downbeat_confidence=.1)


def observation():
    times = [0, .5, 1, 2, 2.5, 3]
    return dict(beats=[beat(t) for t in times], beat_candidates=[beat(1.5, .4)],
                onsets=[dict(time_s=t, strength=.5, low_strength=.1, mid_strength=.3, high_strength=.1)
                        for t in [0, .5, 1, 1.5, 2, 2.5, 3]],
                activity=[dict(time_s=1.5, relative_db=-3, rms=.25)],
                harmonic_changes=[dict(time_s=1.5, strength=.2)])


class CandidateEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.lock = json.loads((DIRECTORY / "candidate-evidence-lock-v1.json").read_bytes())

    def case(self):
        return dict(id=self.lock["probe_case_id"], observations=observation(),
                    truth_times_s=[0, .5, 1, 1.5, 2, 2.5, 3], raw_truth_pairs=[[0, 0], [1, 1], [2, 2], [3, 4], [4, 5], [5, 6]],
                    beat_tolerance_s=.07, score_replay_exact=True, tags=["synthetic-test"])

    def test_two_sided_context_excludes_gap_and_uses_real_samples(self):
        features = candidate_features(observation(), beat(1.5, .4), self.lock)
        self.assertEqual(features["context_period_s"], .5)
        self.assertEqual(features["context_dispersion"], 0)
        self.assertEqual(features["double_gap_residual"], 0)
        self.assertEqual(features["midpoint_error_ratio"], 0)
        self.assertEqual(features["onset_relative_to_anchors"], 1)
        self.assertEqual(features["confidence_relative_to_anchors"], .5)
        self.assertEqual(features["relative_db"], -3)

    def test_missing_edge_context_is_not_zero_or_one_sided(self):
        for time in (-.1, .25, 2.75, 3.1):
            with self.subTest(time=time):
                features = candidate_features(observation(), beat(time, .4), self.lock)
                self.assertIsNone(features["double_gap_residual"])
                self.assertIsNone(features["context_dispersion"])
        observations = observation()
        observations["onsets"] = []
        features = candidate_features(observations, beat(1.5, .4), self.lock)
        self.assertIsNone(features["onset_strength"])
        self.assertIsNone(features["onset_relative_to_anchors"])
        self.assertIsNone(candidate_features(observation(), beat(1.6), self.lock)["harmonic_strength"])

    def test_annotations_do_not_influence_features(self):
        case = self.case()
        before, _ = case_rows(case, self.lock)
        case["truth_times_s"] = [0, 1, 2, 3]
        case["raw_truth_pairs"] = []
        after, _ = case_rows(case, self.lock)
        self.assertEqual(before[0]["features"], after[0]["features"])
        self.assertEqual(before[0]["label"], POSITIVE)
        self.assertEqual(after[0]["label"], "offbeat_subdivision_aligned")

    def test_missed_covered_ambiguous_and_unanchored_labels_differ(self):
        label = lambda time, truth, covered: annotation_label(time, truth, covered, .07, self.lock)[0]
        self.assertEqual(label(.5, [0, .5, 1], {0, 2}), POSITIVE)
        self.assertEqual(label(.54, [0, .5, 1], {1}), "covered_truth_duplicate")
        self.assertEqual(label(.5, [.46, .54], set()), "ambiguous_truth_window")
        self.assertEqual(label(.5, [0, 1], set()), "offbeat_subdivision_aligned")
        self.assertEqual(label(.1, [0, 1], set()), "offbeat_other")
        self.assertEqual(label(1.2, [0, 1], set()), "offbeat_unanchored")

    def test_one_frame_accepted_exclusion_and_positive_logit_cohort(self):
        case = self.case()
        case["observations"]["beat_candidates"] = [beat(1.02, .1), beat(1.03, .2), beat(1.5, .51)]
        rows, counts = case_rows(case, self.lock)
        self.assertEqual(counts["accepted_candidate_exclusions"], 1)
        self.assertEqual(rows[0]["label"], "covered_truth_duplicate")
        self.assertEqual(rows[1]["cohort"], "positive_logit_unselected")

    def test_multiple_peaks_do_not_inflate_distinct_missed_beat_count(self):
        case = self.case()
        case["observations"]["beat_candidates"] = [beat(1.46, .4), beat(1.54, .3)]
        rows, _ = case_rows(case, self.lock)
        summary = cohort_summary(rows)
        self.assertEqual(summary["labels"][POSITIVE], 2)
        self.assertEqual(summary["distinct_missed_truth_supported"], 1)
        self.assertEqual(summary["extra_support_candidates_for_same_truth"], 1)

    def test_auc_direction_ties_and_absent_classes(self):
        self.assertEqual(auc([1, 2], [0, 1], 1), .875)
        self.assertEqual(auc([1, 2], [0, 1], -1), .125)
        self.assertEqual(auc([1], [1], 1), .5)
        self.assertIsNone(auc([], [1], 1))
        self.assertIsNone(auc([1], [], 1))

    def test_probe_is_never_pooled_and_dominance_requires_all_coordinates(self):
        case = self.case()
        baseline = copy.deepcopy(case)
        baseline["observations"]["beat_candidates"] = []
        evidence = dict(cases=[baseline], probe=case, lock_sha256="test", suite_sha256="test", cache_hits=1)
        report, rows = build_summary(evidence, self.lock)
        self.assertEqual(report["cohorts"]["subthreshold"]["candidate_count"], 0)
        self.assertEqual(rows, [])
        probe = report["fixed_probe"]["candidate"]
        self.assertTrue(dominate(probe, probe))
        incomplete = copy.deepcopy(probe)
        incomplete["features"][DOMINANCE[0]] = None
        self.assertFalse(dominate(incomplete, probe))

    def fixture(self):
        suite = json.loads((ROOT / "evaluation/suites/artbeat-v1.json").read_bytes())
        frozen = json.loads((DIRECTORY / "reference-resampler-calibration-v1.json").read_bytes())["suites"][0]["cases"]
        cases = []
        for definition, original in zip(suite["cases"], frozen):
            truth = json.loads((ROOT / "evaluation/suites" / definition["input"]["truth"]).read_bytes())
            case = dict(id=original["id"], pcm_sha256=original["current_pcm_sha256"],
                        truth_sha256=original["truth_sha256"], audio_sha256=original["audio_sha256"],
                        truth_times_s=[b["time_s"] for b in truth["beats"]], score_replay_exact=True,
                        sample_rate=22050, beat_tolerance_s=.07, raw_truth_pairs=[], observations=dict(beats=[]))
            cases.append(case)
        probe = copy.deepcopy(next(c for c in cases if c["id"] == self.lock["probe_case_id"]))
        audit = json.loads((DIRECTORY / "resampler-regression-event-v1.json").read_bytes())
        probe.update(source_trace_sha256=audit["after_trace_sha256"], observation_contract=SHIPPING + "+phase-exact-bh2-256-v1",
                     pcm_sha256=next(c["candidate_pcm_sha256"] for c in frozen if c["id"] == probe["id"]))
        evidence = dict(schema_version=1, purpose="private_calibration_candidate_evidence", cases=cases, probe=probe,
                        lock_sha256=digest(DIRECTORY / "candidate-evidence-lock-v1.json"), suite_sha256=self.lock["suite_sha256"],
                        cache_hits=15, neural_inferences=0, cache_writes=0, observation_contract=SHIPPING)
        for field, relative in {
            "source_sha256": "crates/rhythm-map-eval/src/candidate_evidence.rs",
            "cache_source_sha256": "crates/rhythm-map-eval/src/observation_cache.rs",
            "engine_source_sha256": "crates/rhythm-map-core/src/engine.rs",
            "estimator_source_sha256": "crates/rhythm-map-core/src/estimator.rs",
            "model_manifest_sha256": "models/beat-this-full-v1.json",
        }.items():
            evidence[field] = digest(ROOT / relative)
        return evidence

    def test_changed_role_contract_source_lock_and_missing_cache_are_rejected(self):
        evidence = self.fixture()
        validate_inputs(evidence, self.lock)
        for key, value in (("purpose", "holdout"), ("cache_hits", 14), ("neural_inferences", 1), ("cache_writes", 1),
                           ("lock_sha256", "changed"), ("observation_contract", "changed"), ("source_sha256", "changed")):
            with self.subTest(key=key):
                changed = copy.deepcopy(evidence)
                changed[key] = value
                with self.assertRaises(ValueError):
                    validate_inputs(changed, self.lock)

    def test_tempo_only_truth_stale_pcm_changed_tolerance_and_nonfinite_are_rejected(self):
        for key, value in (("truth_times_s", []), ("pcm_sha256", "changed"), ("score_replay_exact", False),
                           ("beat_tolerance_s", .08), ("raw_truth_pairs", [[0, 0]]),
                           ("observations", dict(beats=[beat(float("nan"))]))):
            with self.subTest(key=key):
                evidence = self.fixture()
                evidence["cases"][0][key] = value
                with self.assertRaises(ValueError):
                    validate_inputs(evidence, self.lock)

    def test_frozen_aggregate_is_source_linked_and_does_not_claim_recovery(self):
        report = json.loads((DIRECTORY / "candidate-evidence-separability-v1.json").read_bytes())
        self.assertEqual(report["analysis_source_sha256"], digest(DIRECTORY / "candidate_evidence_audit.py"))
        self.assertEqual(report["lock_sha256"], digest(DIRECTORY / "candidate-evidence-lock-v1.json"))
        self.assertEqual(report["sources"]["source_sha256"], digest(ROOT / "crates/rhythm-map-eval/src/candidate_evidence.rs"))
        self.assertEqual(report["replay_exact_count"], 15)
        self.assertTrue(report["probe_replay_exact"])
        self.assertEqual(report["neural_inferences"], 0)
        self.assertEqual(report["cache_writes"], 0)
        for key in ("production_changed", "promotion", "threshold_search", "holdout_used"):
            self.assertFalse(report[key])
        cohort = report["cohorts"]["subthreshold"]
        self.assertEqual(cohort["candidate_count"], 1180)
        self.assertEqual(sum(cohort["labels"].values()), 1180)
        self.assertEqual(cohort["distinct_missed_truth_supported"], 119)
        self.assertEqual(sum(c["missed_truth_count"] for c in report["cases"]), 128)
        self.assertEqual(cohort["labels"][POSITIVE], 144)
        self.assertEqual(report["fixed_probe"]["candidate"]["time_s"], 1.5)
        self.assertEqual(report["fixed_probe"]["comparable_negative_count"], 823)
        self.assertEqual(report["fixed_probe"]["dominating_negative_count"], 0)
        document = (ROOT / "evaluation/baselines/beat-this-candidate-evidence-v1.md").read_text(encoding="utf-8")
        for label, feature in (("Confidence, higher", "confidence"), ("Onset strength, higher", "onset_strength"),
                               ("Onset relative to anchors, higher", "onset_relative_to_anchors"),
                               ("Midpoint error ratio, lower", "midpoint_error_ratio"),
                               ("Double-gap residual, lower", "double_gap_residual"),
                               ("Context dispersion, lower", "context_dispersion")):
            all_stats = report["primary_features_vs_all_anchored_offbeats"][feature]
            grid_stats = report["primary_features_vs_subdivision_aligned"][feature]
            line = f"| {label} | {all_stats['pooled_auc']:.3f} / {all_stats['macro_track_auc']:.3f} | {grid_stats['pooled_auc']:.3f} |"
            self.assertIn(line, document)


if __name__ == "__main__":
    unittest.main()
