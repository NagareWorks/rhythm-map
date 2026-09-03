import unittest
import json
from pathlib import Path

import numpy as np

from characterize_resampler import impulse_response, summarize_case, vector, waveform


class CharacterizationTests(unittest.TestCase):
    def test_frozen_signal_and_model_reports_do_not_claim_product_promotion(self):
        directory = Path(__file__).parent
        signal = json.loads((directory / "resampler-characterization-v1.json").read_text(encoding="utf-8"))
        self.assertEqual(len(signal["cases"]), 99)
        self.assertTrue(signal["all_lengths_passed"])
        self.assertIn("musical_accuracy", signal["not_checked"])
        self.assertTrue(all(c["comparisons"]["candidate"]["rmse"] <= c["comparisons"]["current"]["rmse"] + 1e-12 for c in signal["cases"]))
        neural = json.loads((directory / "reference-resampler-v1-audit.json").read_text(encoding="utf-8"))
        self.assertTrue(neural["passed"])
        self.assertEqual(sum(len(c["stages"]) for c in neural["cases"]), 64)
        for case in neural["cases"]:
            self.assertEqual(case["preprocessing_candidate"], signal["candidate"])
            self.assertTrue(case["observation_contract"].endswith("+" + signal["candidate"]))
            self.assertEqual(case["candidate_source_sha256"], signal["sources"]["candidate_source_sha256"])
            self.assertTrue(all(s["passed"] for s in case["stages"].values()))

    def test_invalid_pcm_and_length_mismatch_are_explicit(self):
        for value in ([], [np.nan], [np.inf], [[1.0]]):
            with self.assertRaises(ValueError):
                vector(value)
        self.assertFalse(waveform([0.0], [0.0, 1.0])["equal_length"])
        self.assertEqual(waveform([0.25, 0.5], [0.25, 0.5])["rmse"], 0)

    def test_phase_fit_detects_a_delay_not_just_an_impulse_peak(self):
        pcm = np.zeros(4096)
        pcm[1001] = 1
        response = impulse_response(pcm, 22050, 1000 / 22050)
        self.assertAlmostEqual(response["passband_delay_output_samples"], 1, places=8)
        self.assertEqual(response["peak_time_error_samples"], 1)
        self.assertIsNone(response["minus_3_db_relative_nyquist"])

    def test_native_rate_bypasses_resampling_and_keeps_candidate_separate(self):
        def forbidden(*_):
            self.fail("native PCM must be unchanged")
        case = {"sample_rate": 22050, "signal": "dc", "parameter": None,
                "input_pcm": [0.25] * 3000, "current_pcm": [0.25] * 3000,
                "candidate_pcm": [0.25] * 3000, "current_elapsed_ms": 1,
                "candidate_elapsed_ms": 2}
        result = summarize_case(case, forbidden)
        self.assertTrue(result["lengths_passed"])
        self.assertEqual(set(result["comparisons"]), {"current", "candidate"})
        self.assertEqual(result["comparisons"]["candidate"]["rmse"], 0)

    def test_wrong_reference_duration_is_not_silently_cropped(self):
        case = {"sample_rate": 44100, "signal": "dc", "parameter": None,
                "input_pcm": [0.25] * 6000, "current_pcm": [0.25] * 3000,
                "current_elapsed_ms": 1}
        with self.assertRaises(ValueError):
            summarize_case(case, lambda *_: [0.25] * 2999)


if __name__ == "__main__":
    unittest.main()
