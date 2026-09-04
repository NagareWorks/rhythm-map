"""Keep the model-free input proof source-locked and distinct from cache reuse."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
PARITY = ROOT / "evaluation/parity"


class RubatoPcmEquivalenceTests(unittest.TestCase):
    def test_source_identities(self):
        report = json.loads((PARITY / "rubato-pcm-equivalence-v1.json").read_bytes())
        sources = {
            "lock_sha256": "evaluation/parity/rubato-pcm-equivalence-lock-v1.json",
            "suite_sha256": "evaluation/suites/rubato-calibration-v1.json",
            "auditor_sha256": "crates/rhythm-map-eval/examples/rubato_pcm_equivalence.rs",
            "support_sha256": "crates/rhythm-map-eval/examples/support/mod.rs",
            "cargo_lock_sha256": "Cargo.lock",
            "adapter_sha256": "crates/rhythm-map-beat-this/src/lib.rs",
            "preprocessing_sha256": "crates/rhythm-map-beat-this/src/audio.rs",
        }
        for field, path in sources.items():
            with self.subTest(field=field):
                self.assertEqual(report[field], hashlib.sha256((ROOT / path).read_bytes()).hexdigest())

    def test_complete_locked_cohort_and_no_cache_promotion(self):
        lock = json.loads((PARITY / "rubato-pcm-equivalence-lock-v1.json").read_bytes())
        report = json.loads((PARITY / "rubato-pcm-equivalence-v1.json").read_bytes())
        suite = json.loads((ROOT / "evaluation/suites/rubato-calibration-v1.json").read_bytes())
        self.assertEqual(suite["purpose"], "calibration")
        self.assertEqual(report["suite_sha256"], lock["suite_sha256"])
        self.assertEqual(report["purpose"], "calibration_pcm_equivalence_summary")
        self.assertEqual(report["case_count"], lock["case_count"])
        self.assertEqual(report["bit_identical_cases"], 25)
        self.assertFalse(report["cache_reuse_authorized"])
        for key in ("model_inference_runs", "cache_reads", "cache_writes"):
            self.assertEqual(report[key], 0)
        self.assertEqual([c["id"] for c in report["cases"]], [c["id"] for c in suite["cases"]])
        for actual, expected in zip(report["cases"], suite["cases"]):
            with self.subTest(case=actual["id"]):
                self.assertEqual(actual["audio_sha256"], expected["input"]["audio"]["sha256"])
                self.assertEqual(actual["native_sample_rate_hz"], lock["required_native_rate_hz"])
                comparison = actual["comparison"]
                self.assertTrue(comparison["bit_identical"])
                self.assertGreater(comparison["legacy_sample_count"], 0)
                self.assertEqual(comparison["legacy_sample_count"], comparison["shipping_sample_count"])
                self.assertEqual(comparison["legacy_pcm_sha256"], comparison["shipping_pcm_sha256"])
                self.assertEqual(comparison["differing_shared_samples"], 0)
                self.assertEqual(comparison["unpaired_samples"], 0)


if __name__ == "__main__":
    unittest.main()
