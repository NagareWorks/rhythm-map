"""A locked historical replay is not a freshly measured v2 inference result."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
PARITY = ROOT / "evaluation/parity"


def read(path):
    return json.loads(path.read_bytes())


class RubatoCacheReplayTests(unittest.TestCase):
    def test_frozen_source_model_truth_and_pcm_identities(self):
        lock = read(PARITY / "rubato-cache-replay-lock-v1.json")
        report = read(PARITY / "rubato-cache-replay-v1.json")
        proof = read(PARITY / "rubato-pcm-equivalence-v1.json")
        suite = read(ROOT / "evaluation/suites/rubato-calibration-v1.json")
        for field, path in {
            "lock_sha256": "evaluation/parity/rubato-cache-replay-lock-v1.json",
            "pcm_proof_sha256": "evaluation/parity/rubato-pcm-equivalence-v1.json",
            "exporter_sha256": "crates/rhythm-map-eval/src/rubato_cache_replay.rs",
            "suite_sha256": "evaluation/suites/rubato-calibration-v1.json",
            "model_manifest_sha256": "models/beat-this-full-v1.json",
        }.items():
            with self.subTest(field=field):
                self.assertEqual(report[field], hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
        self.assertEqual(report["sources"], lock["sources"])
        for path, digest in lock["sources"].items():
            self.assertEqual(digest, hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
        for field in ("pcm_proof_sha256", "suite_sha256", "model_manifest_sha256", "baseline_sha256"):
            self.assertEqual(report[field], lock[field])
        self.assertEqual(len(report["cases"]), 25)
        self.assertEqual([c["id"] for c in report["cases"]], [c["id"] for c in suite["cases"]])
        for actual, pinned, pcm, case in zip(report["cases"], lock["cases"], proof["cases"], suite["cases"]):
            with self.subTest(case=actual["id"]):
                for field in ("id", "audio_sha256", "truth_sha256", "cache_entry_sha256"):
                    self.assertEqual(actual[field], pinned[field])
                self.assertEqual(actual["truth_sha256"], hashlib.sha256(
                    (ROOT / "evaluation/suites" / case["input"]["truth"]).read_bytes()).hexdigest())
                self.assertEqual(actual["pcm_sha256"], pcm["comparison"]["shipping_pcm_sha256"])
                self.assertEqual(actual["sample_count"], pcm["comparison"]["shipping_sample_count"])

    def test_replay_success_is_distinct_from_musical_accuracy(self):
        report = read(PARITY / "rubato-cache-replay-v1.json")
        self.assertEqual(report["purpose"], "calibration_read_only_v1_cache_replay_summary")
        self.assertEqual(report["cache_hits"], 25)
        self.assertEqual(report["neural_inferences"], 0)
        self.assertEqual(report["cache_writes"], 0)
        self.assertFalse(report["cache_relabeling"])
        self.assertFalse(report["production_fallback"])
        self.assertNotEqual(report["source_observation_contract"], report["shipping_observation_contract"])
        self.assertEqual(sum(c["selected_score"]["passed"] for c in report["cases"]), 1)
        for case in report["cases"]:
            for field in ("raw_events_exact", "score_replay_exact", "source_metadata_exact"):
                self.assertTrue(case[field])
            self.assertGreater(case["onset_point_count"], 0)
            self.assertGreater(case["activity_point_count"], 0)
            self.assertNotIn("observations", case)
            self.assertNotIn("truth_times_s", case)


if __name__ == "__main__":
    unittest.main()
