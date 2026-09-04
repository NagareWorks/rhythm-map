import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dense_clock_evidence as dense


def case():
    return dict(id="authored", truth_times_s=[.5, 1, 1.5], beat_tolerance_s=.07,
                raw_truth_pairs=[[0, 0]], score_replay_exact=True,
                audio_sha256="audio", pcm_sha256="pcm", sample_count=44100, sample_rate=22050,
                observations=dict(duration_s=2, beats=[dict(time_s=.5)],
                                  beat_candidates=[dict(time_s=.5), dict(time_s=1)], source={}))


def capture(frozen):
    values = [-2.] * 101
    values[25] = 2.
    return dict(schema_version=1, purpose="private_full_recording_dense_evidence",
                case_id=frozen["id"], suite_id=dense.SUITES["artbeat"][0],
                suite_sha256=dense.SUITES["artbeat"][1], frozen_evidence_sha256=dense.INPUTS["artbeat"][1],
                model_manifest_sha256=dense.MODEL, observation_contract=dense.SHIPPING,
                audio_sha256="audio", pcm_sha256="pcm", sample_count=44100, sample_rate=22050,
                replay=dict(exact=True), frame_rate_hz=50, start_time_s=0, frame_count=101,
                beat_logits=values, downbeat_logits=[-2.] * 101,
                observations=dict(copy.deepcopy(frozen["observations"]), activations=None,
                                  activity=[], onsets=[], harmonic_changes=[]))


class DenseClockEvidenceTests(unittest.TestCase):
    def test_checked_report_provenance_and_denominators(self):
        report = json.loads(Path(__file__).with_name("dense-clock-evidence-v1.json").read_text(encoding="utf-8"))
        self.assertEqual(report["script_sha256"], dense.sha(Path(dense.__file__).read_bytes()))
        self.assertTrue(report["truth_assisted"])
        self.assertFalse(report["automatic_decoder"])
        self.assertFalse(report["accuracy_improvement_claimed"])
        self.assertFalse(report["holdout_opened"])
        for name, identity in report["helper_source_sha256"].items():
            self.assertEqual(identity, dense.sha(Path(__file__).with_name(name).read_bytes()))
        for cohort in report["cohorts"]:
            count, identity = dense.INPUTS[cohort["cohort"]]
            self.assertEqual((cohort["tracks"], len(cohort["cases"])), (count, count))
            self.assertEqual(cohort["frozen_evidence_sha256"], identity)
            self.assertEqual(cohort["total_frames_per_head"], sum(c["frame_count"] for c in cohort["cases"]))
            for key, path in dense.SOURCES.items():
                self.assertEqual(cohort["source_hashes"][key], dense.sha((dense.ROOT / path).read_bytes()))
            for key in dense.STRATA:
                stats = cohort["pooled"][key]
                self.assertEqual(stats["queries"], sum(c["strata"][key]["queries"] for c in cohort["cases"]))
                self.assertEqual(stats["paired_queries"], stats["canonical_wins"] + stats["ties"] + stats["half_phase_wins"])
                self.assertEqual(stats["queries"], stats["paired_queries"] + stats["unavailable_queries"])
            for track in cohort["cases"]:
                self.assertTrue(track["independent_default_events_exact"])
                coverage, strata = track["coverage"], track["strata"]
                self.assertEqual(coverage["truth_beats"], coverage["interval_queries"] + 1)
                self.assertEqual(strata["all"]["queries"], coverage["interval_queries"])
                self.assertEqual(strata["raw_missed"]["queries"], strata["raw_missed_with_candidate"]["queries"] +
                                 strata["raw_missed_without_candidate"]["queries"])
                self.assertEqual(strata["all"]["queries"], strata["raw_matched"]["queries"] + strata["raw_missed"]["queries"])

    def test_ideal_pulse_preference_is_not_recovery(self):
        values = [-2.] * 101
        values[25] = values[50] = 2.
        rows, coverage = dense.template_rows(values, case())
        stats = dense.stratified(rows)
        self.assertEqual(stats["all"]["canonical_wins"], 2)
        self.assertEqual(stats["raw_missed_with_candidate"]["queries"], 1)
        self.assertEqual(coverage["raw_missed_truth_beats"], 2)
        self.assertEqual(coverage["excluded_final_truth_beats"], 1)
        self.assertFalse(coverage["excluded_final_truth_raw_matched"])

    def test_shifted_pulse_is_not_direction_flipped(self):
        values = [-2.] * 101
        values[37] = values[63] = 2.
        rows, _ = dense.template_rows(values, case())
        self.assertEqual(dense.summarize(rows)["half_phase_wins"], 2)
        self.assertEqual(dense.summarize(rows)["mean_logit_margin"], -4)

    def test_template_values_do_not_use_raw_anchors(self):
        frozen = case()
        first, _ = dense.template_rows(list(range(101)), frozen)
        frozen["observations"]["beats"] = []
        frozen["raw_truth_pairs"] = []
        second, _ = dense.template_rows(list(range(101)), frozen)
        self.assertEqual([(r["canonical"], r["half_phase"]) for r in first],
                         [(r["canonical"], r["half_phase"]) for r in second])
        self.assertNotEqual(first[0]["strata"], second[0]["strata"])

    def test_missing_candidate_partition(self):
        frozen = case()
        frozen["observations"]["beat_candidates"] = []
        rows, _ = dense.template_rows([-1.] * 101, frozen)
        stats = dense.stratified(rows)
        self.assertEqual(stats["raw_missed_without_candidate"]["queries"], 1)
        self.assertEqual(stats["raw_missed_with_candidate"]["queries"], 0)
        self.assertIsNone(stats["raw_missed_with_candidate"]["mean_logit_margin"])
        self.assertEqual(stats["all"]["ties"], 2)
        self.assertEqual(stats["all"]["canonical_above_zero"], 0)

    def test_uncovered_windows_are_not_padded_or_dropped(self):
        self.assertIsNone(dense.local_peak([1.] * 51, 0, .05))
        self.assertIsNone(dense.local_peak([1.] * 51, 1, .05))
        self.assertIsNone(dense.local_peak([1.] * 51, .01, .001))
        self.assertEqual(dense.local_peak([1.] * 51, .5, .05), 1)
        rows, _ = dense.template_rows([-2.] * 51, case())
        stats = dense.summarize(rows)
        self.assertEqual((stats["queries"], stats["paired_queries"], stats["unavailable_queries"]), (2, 1, 1))

    def test_changed_raw_match_identity_rejected(self):
        frozen = case()
        frozen["raw_truth_pairs"] = [[0, 1]]
        with self.assertRaisesRegex(ValueError, "identity replay"):
            dense.template_rows([-2.] * 101, frozen)

    def test_truth_and_window_validation(self):
        for truth in ([.5], [.5, .5], [.5, float("nan")]):
            frozen = case()
            frozen["truth_times_s"] = truth
            with self.assertRaises(ValueError):
                dense.template_rows([-2.] * 101, frozen)
        with self.assertRaises(ValueError):
            dense.local_peak([1.] * 51, .5, 0)

    def test_valid_capture_independently_reconstructs_events(self):
        frozen = case()
        self.assertEqual(len(dense.validate_capture(capture(frozen), {}, frozen, "artbeat", {})), 101)

    def test_producer_claim_does_not_hide_changed_dense_events(self):
        frozen = case()
        payload = capture(frozen)
        payload["beat_logits"][50] = 3.
        with self.assertRaisesRegex(ValueError, "event reconstruction"):
            dense.validate_capture(payload, {}, frozen, "artbeat", {})

    def test_changed_observations_source_and_metadata_rejected(self):
        frozen = case()
        payload = capture(frozen)
        payload["observations"]["beats"][0]["time_s"] = .52
        with self.assertRaisesRegex(ValueError, "observation comparison"):
            dense.validate_capture(payload, {}, frozen, "artbeat", {})
        with self.assertRaisesRegex(ValueError, "implementation"):
            dense.validate_capture(capture(frozen), {}, frozen, "artbeat", {"exporter_source_sha256": "other"})
        with self.assertRaisesRegex(ValueError, "metadata"):
            dense.validate_capture(capture(frozen), {"frame_count": 20}, frozen, "artbeat", {})

    def test_truncated_nonfinite_or_misaligned_heads_rejected(self):
        frozen = case()
        for key, value in (("downbeat_logits", [-2.] * 100), ("beat_logits", [float("nan")] * 101),
                           ("start_time_s", 1), ("frame_rate_hz", 100)):
            payload = capture(frozen)
            payload[key] = value
            with self.assertRaises(ValueError):
                dense.validate_capture(payload, {}, frozen, "artbeat", {})
        payload = capture(frozen)
        payload.update(beat_logits=[-2.] * 50, downbeat_logits=[-2.] * 50, frame_count=50)
        with self.assertRaisesRegex(ValueError, "complete recording"):
            dense.validate_capture(payload, {}, frozen, "artbeat", {})

    def test_summary_rejects_partial_duplicate_or_changed_scope(self):
        summary = dict(schema_version=1, purpose="full_recording_dense_capture_summary", complete=True,
                       expected_case_count=1, completed_inference_count=1, cache_writes=0,
                       training_run=False, production_observations_changed=False,
                       accuracy_improvement_claimed=False, cases=[dict(case_id="authored")])
        self.assertEqual(len(dense.validate_summary(summary, [case()], 1)), 1)
        for key, value in (("complete", False), ("completed_inference_count", 0),
                           ("cases", []), ("training_run", True)):
            changed = dict(summary, **{key: value})
            with self.assertRaises(ValueError):
                dense.validate_summary(changed, [case()], 1)
        with self.assertRaises(ValueError):
            dense.validate_summary(summary, [case(), case()], 2)

    def test_capture_byte_hash_is_verified_before_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "capture.json"
            path.write_text(json.dumps({"value": 1}), encoding="utf-8")
            data, identity = dense.read_json(path)
            self.assertEqual(data, {"value": 1})
            self.assertEqual(dense.read_json(path, identity)[0], data)
            path.write_text(json.dumps({"value": 2}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                dense.read_json(path, identity)

    def test_full_pipeline_keeps_frames_and_coordinates_out_of_report(self):
        frozen = case()
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            evidence_path = directory / "evidence.json"
            evidence_path.write_text(json.dumps(dict(cases=[frozen])), encoding="utf-8")
            evidence_hash = dense.sha(evidence_path.read_bytes())
            with patch.dict(dense.INPUTS, artbeat=(1, evidence_hash)):
                payload = capture(frozen)
                payload.update({k: dense.sha((dense.ROOT / v).read_bytes()) for k, v in dense.SOURCES.items()})
                payload["inference_elapsed_s"] = 1.
                capture_path = directory / "authored.json"
                capture_path.write_text(json.dumps(payload), encoding="utf-8")
                record = {k: v for k, v in payload.items() if k not in
                          ("beat_logits", "downbeat_logits", "observations")}
                record["capture_sha256"] = dense.sha(capture_path.read_bytes())
                summary = dict(schema_version=1, purpose="full_recording_dense_capture_summary", complete=True,
                               expected_case_count=1, completed_inference_count=1, cache_writes=0,
                               training_run=False, production_observations_changed=False,
                               accuracy_improvement_claimed=False, cases=[record])
                (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                report = dense.audit("artbeat", evidence_path, directory)
                self.assertEqual(report["total_frames_per_head"], 101)
                self.assertEqual(report["pooled"]["all"]["queries"], 2)
                self.assertEqual(report["macro"]["raw_missed_without_candidate"]["tracks_with_paired_queries"], 0)
                self.assertTrue(report["cases"][0]["independent_default_events_exact"])
                public = json.dumps(report)
                for private in (str(directory), "beat_logits", "downbeat_logits", "truth_times_s", '"time_s"'):
                    self.assertNotIn(private, public)
                capture_path.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "bytes changed"):
                    dense.audit("artbeat", evidence_path, directory)


if __name__ == "__main__":
    unittest.main()
