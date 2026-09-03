import copy
import json
from pathlib import Path
import unittest

import numpy as np

from native_pcm_audit import (CONTRACT, REFERENCE_LOCK, make_matrix, mono,
                              pcm_summary, probe_summary, validate_identity)


class NativePcmTests(unittest.TestCase):
    def test_frozen_report_keeps_failed_parity_separate_from_controls(self):
        report = json.loads(Path(__file__).with_name("native-pcm-v2-audit.json").read_text(encoding="utf-8"))
        self.assertTrue(report["controls_passed"])
        self.assertTrue(all(report["controls"].values()))
        self.assertFalse(report["source_event_parity_passed"])
        self.assertFalse(report["product_or_threshold_changed"])
        for label in ("decode_effect_rust_resampler", "decode_effect_soxr_resampler"):
            for key in ("beat", "downbeat"):
                self.assertEqual(report["effects"][label][key + "_event_parity"]["max_abs"], 0)
        for label in ("resampler_effect_rust_decode", "resampler_effect_reference_decode"):
            effect = report["effects"][label]["beat_events"]
            self.assertEqual(effect["removed_source_times_s"], [19.08])
            self.assertEqual(effect["added_source_times_s"], [20.84, 22.26])
        # The redistributable report must not accidentally include private traces.
        forbidden = {"mono_samples", "rust_native_mono", "mel_values", "legacy_audio"}
        def check_keys(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for item in value.values():
                    check_keys(item)
            elif isinstance(value, list):
                for item in value:
                    check_keys(item)
        check_keys(report)

    def test_downmix_and_invalid_pcm(self):
        np.testing.assert_array_equal(mono([[0.25, 0.75], [-0.5, 0.5]]), [0.5, 0])
        for value in ([], [float("nan")], [float("inf")], np.zeros((2, 0)), np.zeros((2, 2, 2))):
            with self.assertRaises(ValueError):
                mono(value)

    def test_matrix_changes_one_factor_and_keeps_rust_results(self):
        rust, reference = np.array([1., 2.], dtype=np.float32), np.array([3., 4.])
        seen = []
        def controlled_resampler(value):
            seen.append(value.copy())
            self.assertEqual(value.dtype, np.float64)
            return value + 10
        result = make_matrix(rust, reference, [5, 6], [7, 8], controlled_resampler)
        np.testing.assert_array_equal(result["rust_decode_rust_resample"], [5, 6])
        np.testing.assert_array_equal(result["reference_decode_rust_resample"], [7, 8])
        np.testing.assert_array_equal(result["rust_decode_soxr_resample"], [11, 12])
        np.testing.assert_array_equal(result["reference_decode_soxr_resample"], [13, 14])
        np.testing.assert_array_equal(seen, [[1, 2], [3, 4]])
        result["rust_decode_soxr_resample"][0] = 100
        np.testing.assert_array_equal(rust, [1, 2])

    def test_normalization_is_explicit_and_not_f64_native_preservation(self):
        value = np.array([1 + 2 ** -30, 2 + 2 ** -29], dtype=np.float64)
        seen = []
        def resample(pcm):
            seen.append(pcm.copy())
            return pcm
        make_matrix(value, value, [1], [2], resample)
        self.assertFalse(np.array_equal(seen[0], value))
        np.testing.assert_array_equal(seen[0], value.astype(np.float32).astype(np.float64))

    def test_probes_do_not_treat_threshold_crossing_as_selection(self):
        logits = np.full(100, -1.)
        logits[50] = 0.1
        value = probe_summary(logits, [], [1.0])[0]
        self.assertGreater(value["nearby_peak_probability"], 0.5)
        self.assertFalse(value["selected"])
        self.assertTrue(probe_summary(logits, [1.0], [1.0])[0]["selected"])
        with self.assertRaises(ValueError):
            probe_summary(logits, [], [20.0])

    def test_pcm_summary_retains_length_and_bit_identity(self):
        value = np.zeros(500, dtype=np.float32)
        value[200] = 1
        same = pcm_summary(value, value.copy())
        self.assertTrue(same["float32_bit_exact"])
        self.assertEqual(same["full_max_abs"], 0)
        self.assertEqual(same["best_rust_delay_samples"], 0)
        changed = pcm_summary(value, np.concatenate([value, [0.]]))
        self.assertEqual(changed["sample_count_delta"], 1)
        self.assertNotIn("float32_bit_exact", changed)

    def test_locked_identity_rejects_crop_source_and_holdout(self):
        case = {"case_id": "case", "suite_id": "calibration", "suite_sha256": "suite",
                "audio_sha256": "audio", "prefix_seconds": 35}
        trace = dict(case, schema_version=1, purpose="calibration_parity_private",
                     model_manifest_sha256=REFERENCE_LOCK["model_manifest_sha256"],
                     sample_rate=22050, observation_contract=CONTRACT,
                     audio_preprocessing_sha256="a" * 64, mel_shape=[1, 1, 128],
                     beat_logits=[0.], downbeat_logits=[0.], mono_samples=[1.],
                     decoded_sample_count=1)
        validate_identity(trace, case)
        for key, wrong in (("case_id", "other"), ("purpose", "holdout"),
                           ("suite_sha256", "other"), ("audio_sha256", "other"),
                           ("decoded_sample_count", 2), ("observation_contract", "unknown")):
            changed = copy.deepcopy(trace)
            changed[key] = wrong
            with self.assertRaises(ValueError):
                validate_identity(changed, case)


if __name__ == "__main__":
    unittest.main()
