"""Byte-addressed evidence must survive Git checkout without EOL conversion."""
import json
from pathlib import Path
import subprocess
import unittest

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parent.parent


class ArtifactIdentityTests(unittest.TestCase):
    def test_frozen_json_has_checkout_stable_attributes(self):
        paths = sorted(str(path.relative_to(ROOT)).replace("\\", "/") for path in DIRECTORY.glob("*.json"))
        result = subprocess.run(["git", "check-attr", "-z", "text", "diff", "--", *paths], cwd=ROOT,
                                check=True, stdout=subprocess.PIPE).stdout.decode().split("\0")[:-1]
        self.assertEqual(len(result), len(paths) * 6)
        for path, attribute, value in zip(result[::3], result[1::3], result[2::3]):
            with self.subTest(path=path, attribute=attribute):
                self.assertEqual(value, "unset" if attribute == "text" else "set")

    def test_frozen_artifacts_remain_readable_json_not_audio_exports(self):
        for path in DIRECTORY.glob("*.json"):
            with self.subTest(path=path.name):
                value = json.loads(path.read_bytes())
                self.assertIsInstance(value, dict)
                self.assertNotEqual(value.get("purpose"), "calibration_parity_private")
                self.assertNotEqual(value.get("purpose"), "private_calibration_candidate_evidence")
                self.assertFalse(str(value.get("purpose", "")).startswith("private_"))
                self.assertNotIn("mono_samples", value)
                self.assertNotIn("beat_logits", value)


if __name__ == "__main__":
    unittest.main()
