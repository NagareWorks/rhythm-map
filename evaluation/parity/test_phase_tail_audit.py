"""Model-free tests for counterfactual construction and diagnostic boundaries."""
import copy
import unittest

import numpy as np

from phase_tail_audit import event_delta, make_variants, run_audit, validate_audit_lock


def fixture():
    current = np.arange(100, dtype=np.float32) / 100
    legacy = np.concatenate([[-0.1, -0.2], current[:-3]]).astype(np.float32)
    case = {"case_id": "test", "suite_id": "calibration", "suite_sha256": "suite",
            "audio_sha256": "audio", "prefix_seconds": 35,
            "legacy_origin_samples": 2, "probe_times_s": [0.4]}
    trace = {**case, "purpose": "calibration_parity_private",
             "observation_contract": "beat-this-rten-observations-v2+decode-audio-v2",
             "sample_rate": 22050, "mono_samples": current.tolist(), "decoded_sample_count": 100,
             "legacy_audio": {"implementation": "beat-this-1.0.0", "sample_rate": 22050,
                              "mono_samples": legacy.tolist(), "decoded_sample_count": 99}}
    lock = {"schema_version": 1, "purpose": "calibration_phase_tail_diagnosis",
            "reference_lock_sha256": "reference", "cases": [case]}
    return trace, lock


class PhaseTailTests(unittest.TestCase):
    def test_factors_reconstruct_legacy_without_mutating_trace(self):
        trace, _ = fixture()
        saved = copy.deepcopy(trace)
        v = make_variants(trace, 2)
        np.testing.assert_array_equal(v["origin_and_tail_restored"][0], v["actual_v1"][0])
        np.testing.assert_array_equal(v["origin_restored_only"][0][2:], v["v2"][0])
        np.testing.assert_array_equal(v["tail_trimmed_only"][0], v["v2"][0][:-3])
        v["v2"][0][0] = 123
        self.assertEqual(trace, saved)

    def test_partial_nonfinite_or_missing_legacy_is_rejected(self):
        trace, _ = fixture()
        trace["decoded_sample_count"] += 1
        with self.assertRaises(ValueError):
            make_variants(trace, 2)
        trace, _ = fixture()
        trace["legacy_audio"]["mono_samples"][0] = float("nan")
        with self.assertRaises(ValueError):
            make_variants(trace, 2)
        trace["legacy_audio"] = None
        with self.assertRaises(ValueError):
            make_variants(trace, 2)

    def test_invalid_origin_is_rejected(self):
        trace, _ = fixture()
        for shift in (0, -1, 99, 1.5):
            with self.assertRaises(ValueError):
                make_variants(trace, shift)

    def test_lock_requires_exact_case_set_identity_and_calibration(self):
        trace, lock = fixture()
        self.assertEqual(set(validate_audit_lock(lock, "reference", [trace])), {"test"})
        for traces in ([], [trace, trace]):
            with self.assertRaises(ValueError):
                validate_audit_lock(lock, "reference", traces)
        with self.assertRaises(ValueError):
            validate_audit_lock(lock, "different-reference", [trace])
        for field, value in (("purpose", "holdout"), ("audio_sha256", "other"),
                             ("suite_sha256", "other"), ("observation_contract", "v1")):
            invalid = dict(trace, **{field: value})
            with self.assertRaises(ValueError):
                validate_audit_lock(lock, "reference", [invalid])

    def test_event_correspondence_is_one_to_one_and_keeps_negative_origin(self):
        delta = event_delta([-0.002, 1.0, 2.0], [0.0, 1.01, 1.02, 3.0])
        self.assertEqual(delta["matched"], 2)
        self.assertEqual(delta["removed_source_times_s"], [2.0])
        self.assertEqual(delta["added_source_times_s"], [1.02, 3.0])
        self.assertEqual(event_delta([], [])["matched"], 0)
        for invalid in ([2, 1], [float("nan")], [[1]]):
            with self.assertRaises(ValueError):
                event_delta(invalid, [1])

    def test_audit_reuses_baseline_and_reports_controlled_pairs(self):
        trace, lock = fixture()
        calls = []
        def predict(pcm):
            calls.append(len(pcm))
            return {"beat": pcm.copy(), "downbeat": pcm.copy()}
        def decode(**logits):
            return tuple(np.flatnonzero(logits[key] > 0.5) / 50 for key in ("beat", "downbeat"))
        current = {key: np.asarray(trace["mono_samples"]) for key in ("beat", "downbeat")}
        report = run_audit(trace, lock["cases"][0], predict, decode, current)
        self.assertEqual(calls, [97, 102, 99, 99])
        self.assertEqual(report["trimmed_tail_samples"], 3)
        self.assertTrue(report["reconstruction"]["waveform_passed"])
        self.assertTrue(report["reconstruction"]["logits_passed"])
        self.assertEqual(len(report["effects"]), 6)
        self.assertNotIn("mono_samples", report)


if __name__ == "__main__":
    unittest.main()
