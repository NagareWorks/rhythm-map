import copy
import hashlib
import json
import unittest

import numpy as np

from resampler_event_audit import DIRECTORY, ROOT, SHIPPING, default_events, explain_removed, profile, validate_inputs
from verify_bounded_resampler import digest, pcm_bytes


class EventAuditTests(unittest.TestCase):
    def test_strict_zero_gate_and_peak_competition_are_independent(self):
        before = [-1.0] * 20
        before[13] = 0.008
        after = before.copy()
        after[13] = -0.002
        self.assertEqual(explain_removed(before, after, 0.26)["cause"], "strict_zero_threshold_crossing")
        after[13] = 0.0
        self.assertFalse(profile(after, 0.26)["above_threshold"])
        after[13], after[14] = 0.1, 0.2
        self.assertEqual(explain_removed(before, after, 0.26)["cause"], "local_peak_competition")
        after[13] = -0.1
        self.assertEqual(explain_removed(before, after, 0.26)["cause"], "threshold_and_local_peak_competition")

    def test_default_decoder_plateau_and_float32_time_contract(self):
        values = [-1.0] * 20
        values[3] = values[4] = 1.0
        values[15] = 2.0
        self.assertEqual(default_events(values), np.asarray([0.07, 0.3], dtype=np.float32).tolist())
        self.assertEqual(default_events([0.0]), [])
        self.assertEqual(default_events([1.0]), [0.0])
        self.assertTrue(profile([1.0], 0)["local_maximum"])

    def test_invalid_inputs_and_out_of_range_probes_fail(self):
        for values in ([], [float("nan")], [float("inf")], [[1.0]]):
            with self.assertRaises(ValueError):
                default_events(values)
        for time in (-1, float("nan"), 2):
            with self.assertRaises(ValueError):
                profile([0.5], time)

    def fixture(self):
        pcm = [0.25] * 4
        pcm_sha = hashlib.sha256(pcm_bytes(pcm)).hexdigest()
        source_sha = digest(ROOT / "crates/rhythm-map-eval/examples/support/reference_resampler.rs")
        lock = dict(schema_version=1, purpose="calibration_resampler_event_loss_diagnosis",
                    frame_rate_hz=50, decoder_logit_threshold=0, decoder_local_max_radius_frames=3,
                    diagnostic_window_radius_frames=3, suite_id="test", suite_sha256="a" * 64,
                    case_id="test-loss", audio_sha256="b" * 64, model_manifest_sha256="c" * 64,
                    model_sample_count=4, current_pcm_sha256=pcm_sha, candidate_pcm_sha256=pcm_sha,
                    candidate_source_sha256=source_sha)
        before = dict(schema_version=1, purpose="calibration_parity_private", sample_rate=22050,
                      observation_contract=SHIPPING, mono_samples=pcm, decoded_sample_count=4,
                      adapter_source_sha256="d" * 64, audio_preprocessing_sha256="e" * 64,
                      mel_shape=[1, 1, 128], beat_logits=[1.0], downbeat_logits=[1.0],
                      upstream_beats=[0.0], observations=dict(beats=[dict(time_s=0.0)]))
        before.update({k: lock[k] for k in ("suite_id", "suite_sha256", "case_id", "audio_sha256", "model_manifest_sha256")})
        after = copy.deepcopy(before)
        after.update(observation_contract=SHIPPING + "+phase-exact-bh2-256-v1",
                     preprocessing_candidate="phase-exact-bh2-256-v1", candidate_source_sha256=source_sha,
                     beat_logits=[-1.0], upstream_beats=[], observations=dict(beats=[]))
        case = dict(id=lock["case_id"], raw_beat_count=0, **{k: lock[k] for k in
                    ("audio_sha256", "model_sample_count", "current_pcm_sha256", "candidate_pcm_sha256")})
        suite = dict(suite_id="test", cases=[case], summary=dict(metrics=dict(beat_f1=dict(regressed=[lock["case_id"]]))),
                     sources={k: before[k] for k in ("adapter_source_sha256", "audio_preprocessing_sha256")})
        return lock, dict(suites=[suite]), before, after

    def test_changed_pcm_cropped_trace_holdout_and_decoder_changes_are_rejected(self):
        valid = self.fixture()
        validate_inputs(*valid)
        mutations = [(0, "decoder_logit_threshold", -0.1), (0, "diagnostic_window_radius_frames", 5),
                     (2, "purpose", "holdout"), (2, "decoded_sample_count", 5),
                     (2, "mono_samples", [0.5] * 4), (2, "adapter_source_sha256", "f" * 64),
                     (3, "observation_contract", SHIPPING), (3, "upstream_beats", [0.0])]
        for index, field, value in mutations:
            with self.subTest(field=field):
                args = copy.deepcopy(valid)
                args[index][field] = value
                with self.assertRaises(ValueError):
                    validate_inputs(*args)

    def test_frozen_real_counterexample_is_not_mistaken_for_a_safe_fix(self):
        event = json.loads((DIRECTORY / "resampler-regression-event-v1.json").read_bytes())
        official = json.loads((DIRECTORY / "resampler-regression-reference-v1.json").read_bytes())
        self.assertEqual(event["lock_sha256"], digest(DIRECTORY / "resampler-regression-lock-v1.json"))
        self.assertEqual(event["calibration_report_sha256"], digest(DIRECTORY / "reference-resampler-calibration-v1.json"))
        self.assertTrue(event["controls_passed"])
        self.assertTrue(all(event["controls"].values()))
        self.assertFalse(event["promotion"])
        self.assertEqual(event["beat_delta"]["removed_source_times_s"], [1.5])
        self.assertEqual(event["beat_delta"]["added_source_times_s"], [])
        self.assertEqual(event["beat_delta"]["max_matched_offset_s"], 0)
        probe, = event["removed_event_probes"]
        self.assertEqual(probe["cause"], "strict_zero_threshold_crossing")
        self.assertTrue(probe["before"]["local_maximum"] and probe["after"]["local_maximum"])
        self.assertTrue(probe["nearest_truth"]["within_suite_tolerance"])
        self.assertGreater(event["before_selected_metrics"]["metrics"]["beats"]["f1"],
                           event["after_selected_metrics"]["metrics"]["beats"]["f1"])
        self.assertTrue(official["passed"] and official["reference_complete"])
        case, = official["cases"]
        self.assertEqual(case["trace_sha256"], event["after_trace_sha256"])
        self.assertEqual(len(case["stages"]), 16)
        self.assertTrue(all(stage["passed"] for stage in case["stages"].values()))
        self.assertEqual(case["stages"]["beat_source_audio_event_agreement"]["left_shape"], [32])


if __name__ == "__main__":
    unittest.main()
