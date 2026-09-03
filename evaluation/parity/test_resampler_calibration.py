import copy
import json
from pathlib import Path
import unittest

from summarize_resampler_calibration import summarize
from verify_bounded_resampler import digest


class CalibrationSummaryTests(unittest.TestCase):
    def case(self, identity, before_f1, after_f1, before_error, after_error):
        def score(f1, error):
            return dict(passed=f1 >= 0.9 and error <= 5, metrics=dict(
                beats=dict(f1=f1, median_absolute_error_ms=1.0, p95_absolute_error_ms=2.0),
                tempo=dict(median_absolute_error_percent=error, p95_absolute_error_percent=error),
                changes=dict(f1=1.0, recall=1.0)))
        return dict(id=identity, baseline=score(before_f1, before_error), candidate=score(after_f1, after_error))

    def test_improved_average_does_not_hide_lost_pass_or_regression(self):
        cases = [self.case("gain", 0.5, 1.0, 10, 0), self.case("loss", 1.0, 0.89, 0, 6)]
        summary = summarize(cases)
        self.assertGreater(summary["metrics"]["beat_f1"]["delta_mean"], 0)
        self.assertEqual(summary["metrics"]["beat_f1"]["improved"], ["gain"])
        self.assertEqual(summary["metrics"]["beat_f1"]["regressed"], ["loss"])
        self.assertEqual(summary["metrics"]["tempo_median_error_percent"]["improved"], ["gain"])
        self.assertEqual(summary["gained_passes"], ["gain"])
        self.assertEqual(summary["lost_passes"], ["loss"])

    def test_tempo_only_does_not_claim_beat_or_change_accuracy(self):
        case = self.case("tempo", 1, 1, 2, 1)
        for side in ("baseline", "candidate"):
            case[side]["metrics"].pop("beats")
            case[side]["metrics"].pop("changes")
        self.assertEqual(set(summarize([case], tempo_only=True)["metrics"]),
                         {"tempo_median_error_percent", "tempo_p95_error_percent"})

    def test_null_error_is_explicit_not_zero_or_silently_excluded(self):
        case = self.case("missing", 0, 1, 5, 5)
        case["baseline"]["metrics"]["beats"]["median_absolute_error_ms"] = None
        metric = summarize([case])["metrics"]["beat_median_error_ms"]
        self.assertFalse(metric["available"])
        self.assertNotIn("before_mean", metric)

    def test_duplicate_empty_and_exact_equality(self):
        case = self.case("equal", 1, 1, 1, 1)
        for cases in ([], [case, copy.deepcopy(case)]):
            with self.assertRaises(ValueError):
                summarize(cases)
        self.assertEqual(summarize([case])["metrics"]["beat_f1"]["unchanged"], ["equal"])

    def test_frozen_bounded_evidence_links_to_unmodified_original(self):
        directory = Path(__file__).parent
        old = json.loads((directory / "resampler-characterization-v1.json").read_bytes())
        bounded = json.loads((directory / "resampler-bounded-v1.json").read_bytes())
        self.assertEqual(bounded["before_trace_sha256"], old["trace_sha256"])
        self.assertEqual(bounded["before_source_sha256"], old["sources"]["candidate_source_sha256"])
        self.assertTrue(bounded["passed"])
        self.assertEqual(len(bounded["cases"]), 99)
        self.assertTrue(all(c["passed"] and all(c["bitwise_equal"].values()) for c in bounded["cases"]))
        self.assertEqual(bounded["coefficient_budget_bytes"], 8 * 1024 * 1024)

    def test_complete_frozen_calibration_retains_regressions_and_does_not_promote(self):
        directory = Path(__file__).parent
        report = json.loads((directory / "reference-resampler-calibration-v1.json").read_bytes())
        self.assertFalse(report["promotion"])
        self.assertEqual(report["bounded_identity_report_sha256"], digest(directory / "resampler-bounded-v1.json"))
        self.assertEqual(report["historical_parity_report_sha256"], digest(directory / "reference-resampler-v1-audit.json"))
        self.assertEqual(len(report["historical_parity_pcm_links"]), 4)
        self.assertTrue(all(link["complete_pcm_bitwise_equal"] for link in report["historical_parity_pcm_links"]))
        self.assertEqual([s["suite_id"] for s in report["suites"]], ["artbeat-v1", "fsld-tempo-v1"])
        self.assertEqual(report["suites"][0]["sources"], report["suites"][1]["sources"])
        for suite in report["suites"]:
            self.assertEqual(len(suite["cases"]), 15)
            self.assertEqual(suite["baseline_cache_hits"], 15)
            self.assertEqual(suite["candidate_cache_hits"], 0)
            self.assertTrue(suite["baseline_replay_exact"])
            self.assertTrue(all(c["oracle_unchanged"] for c in suite["cases"]))
            self.assertEqual(suite["summary"], summarize(suite["cases"], suite["tempo_only"]))
            for tag, summary in suite["slices"].items():
                self.assertEqual(summary, summarize([c for c in suite["cases"] if tag in c["tags"]], suite["tempo_only"]))
        artbeat, fsld = (s["summary"] for s in report["suites"])
        self.assertEqual(artbeat["metrics"]["beat_f1"]["regressed"], ["artbeat-14-240-to-96"])
        self.assertEqual(fsld["metrics"]["tempo_p95_error_percent"]["regressed"], ["fsld-360687-100-bpm"])


if __name__ == "__main__":
    unittest.main()
