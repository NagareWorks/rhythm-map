"""Small, model-free tests for parity reporting; no checkpoint is imported."""
import unittest
from pathlib import Path

import numpy as np

from compare_reference import (RESAMPLER_CANDIDATE, RESAMPLER_CANDIDATE_CONTRACT,
                               compare, digest, events, validate_trace, waveform_diagnostic)


class ComparisonTests(unittest.TestCase):
    def test_candidate_identity_cannot_claim_shipping_or_stale_implementation(self):
        trace = {"schema_version": 1, "purpose": "calibration_parity_private",
                 "model_manifest_sha256": "digest", "sample_rate": 22050,
                 "observation_contract": RESAMPLER_CANDIDATE_CONTRACT,
                 "preprocessing_candidate": RESAMPLER_CANDIDATE,
                 "candidate_source_sha256": digest(Path(__file__).resolve().parents[2] / "crates/rhythm-map-eval/examples/support/reference_resampler.rs"),
                 "audio_preprocessing_sha256": "a" * 64,
                 "mel_shape": [1, 1, 128], "beat_logits": [1], "downbeat_logits": [1]}
        validate_trace(trace, "digest")
        for key, value in (("observation_contract", "beat-this-rten-observations-v2+decode-audio-v2"),
                           ("candidate_source_sha256", "b" * 64), ("preprocessing_candidate", "other")):
            with self.assertRaises(ValueError):
                validate_trace(dict(trace, **{key: value}), "digest")

    def test_preprocessing_revision_is_explicit_and_historical_traces_still_work(self):
        trace = {"schema_version": 1, "purpose": "calibration_parity_private",
                 "model_manifest_sha256": "digest", "sample_rate": 22050,
                 "observation_contract": "beat-this-rten-observations-v1+decode-audio-v1",
                 "mel_shape": [1, 1, 128], "beat_logits": [1], "downbeat_logits": [1]}
        validate_trace(trace, "digest")
        trace["observation_contract"] = "beat-this-rten-observations-v2+decode-audio-v2"
        with self.assertRaises(ValueError):
            validate_trace(trace, "digest")
        trace["audio_preprocessing_sha256"] = "a" * 64
        validate_trace(trace, "digest")
        trace["observation_contract"] = "unreviewed-contract"
        with self.assertRaises(ValueError):
            validate_trace(trace, "digest")

    def test_trace_role_and_identity_are_required(self):
        with self.assertRaises(ValueError):
            validate_trace({"schema_version": 1, "purpose": "holdout"}, "digest")
        with self.assertRaises(ValueError):
            validate_trace({"schema_version": 1, "purpose": "calibration_parity_private",
                            "model_manifest_sha256": "other"}, "digest")

    def test_shape_mismatch_fails_without_broadcasting(self):
        self.assertFalse(compare([1, 2], [[1, 2]], 0.001)["passed"])

    def test_nonfinite_fails(self):
        self.assertFalse(compare([float("nan")], [float("nan")], 0.001)["passed"])
        self.assertFalse(compare([float("inf")], [float("inf")], 0.001)["passed"])

    def test_numeric_budget(self):
        self.assertTrue(compare([1], [1.00001], 0.001)["passed"])
        self.assertFalse(compare([1], [1.1], 0.001)["passed"])

    def test_events_use_absolute_time_tolerance(self):
        self.assertFalse(events([1000.0], [1000.01])["passed"])
        self.assertTrue(events([1000.0], [1000.000001])["passed"])
        self.assertTrue(events([], [])["passed"])
        self.assertFalse(events([1.0], [])["passed"])
        self.assertFalse(events([1.0], [[1.0]])["passed"])
        self.assertFalse(events([float("inf")], [float("inf")])["passed"])
        self.assertTrue(events([1.0], [1.02], 0.020001)["passed"])
        self.assertFalse(events([1000.0], [1000.03], 0.020001)["passed"])

    def test_delay_is_diagnostic_and_does_not_mutate_audio(self):
        signal = np.random.default_rng(0).normal(size=1000)
        delayed = np.concatenate([np.zeros(32), signal])
        original = delayed.copy()
        result = waveform_diagnostic(signal, delayed)
        self.assertEqual(result["best_rust_delay_samples"], 32)
        self.assertEqual(result["sample_count_delta"], 32)
        np.testing.assert_array_equal(delayed, original)


if __name__ == "__main__":
    unittest.main()
