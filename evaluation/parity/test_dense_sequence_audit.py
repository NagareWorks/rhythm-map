import hashlib
import json
import unittest

import dense_sequence_audit as audit


class DenseSequenceAuditTests(unittest.TestCase):
    def test_fixed_control_coverage_and_exact_input_witness(self):
        controls = {identity: (frames, truth) for identity, frames, truth in audit.controls()}
        self.assertEqual(len(controls), 42)
        first = controls["authored-120-120-120-erased_alternating-no-bar"]
        second = controls["authored-120-60-120-intact-no-bar"]
        self.assertEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])
        self.assertEqual(sum(not x for x in controls["authored-explicit-unavailable"][0]["available"]), 200)

    def test_weak_evidence_differs_without_default_peaks(self):
        controls = {identity: frames for identity, frames, _ in audit.controls()}
        weak = controls["authored-120-120-120-weak_alternating-no-bar"]
        erased = controls["authored-120-120-120-erased_alternating-no-bar"]
        self.assertNotEqual(weak["beat_logits"], erased["beat_logits"])
        self.assertEqual(audit.default_events(weak["beat_logits"]), audit.default_events(erased["beat_logits"]))

    def test_metrics_keep_empty_and_timing_denominators(self):
        stats, pairs = audit.beat_metrics([], [0., 1.])
        self.assertEqual((stats["precision"], stats["recall"], stats["f1"]), (1., 0., 0.))
        self.assertEqual(pairs, [])
        self.assertIsNone(stats["p95_absolute_error_ms"])
        self.assertEqual(audit.quantile([1., 2., 3., 4.], .5), 3.)

    def test_identical_f1_cannot_hide_lost_truth_identity(self):
        truth = dict(beats=[dict(time_s=float(i), downbeat=False) for i in range(4)],
                     tempo_segments=[dict(start_s=0, end_s=4, start_bpm=60, end_bpm=60)], change_points=[])
        baseline = dict(beats=[dict(time_s=float(i), downbeat=False) for i in range(3)],
                        tempo_segments=truth["tempo_segments"], change_points=[])
        ticks = [dict(frame=i * 50, period_frames=50, bar_phase=i - 1, missing_component=False,
                      positive_pulse_window=True, pulse_contrast=8.) for i in range(1, 4)]
        decoded = dict(components=[dict(start_frame=0, end_frame=200, ticks=ticks, meter_hypothesis_not_estimate=4)],
                       unavailable_frames=0, uninformative_frames=0, max_backpointer_bytes=0)
        measurement = audit.measure(truth, dict(decoded=decoded, baseline=baseline, elapsed_s=0.), [0., 1., 2.])
        self.assertEqual(measurement["primary_beats"]["f1"], measurement["inferred_clock_beats"]["f1"])
        self.assertEqual(measurement["lost_primary_truth_count"], 1)
        self.assertIn("lost_primary_truth_identities", measurement["regression_reasons"])

    def test_unavailable_and_endpoint_prior_are_distinct(self):
        decoded = dict(components=[dict(start_frame=50, end_frame=150,
                                       ticks=[dict(frame=75, period_frames=25), dict(frame=100, period_frames=25)])])
        self.assertEqual(audit.clock_tempo(decoded, .5), (None, False))
        self.assertEqual(audit.clock_tempo(decoded, 1.1), (120., True))
        self.assertEqual(audit.clock_tempo(decoded, 1.8), (120., False))
        self.assertEqual(audit.clock_tempo(decoded, 2.8), (120., True))
        self.assertEqual(audit.clock_tempo(decoded, 3.), (None, False))

    def test_missing_tempo_is_not_zero_error(self):
        truth = dict(beats=[dict(time_s=float(i)) for i in range(3)],
                     tempo_segments=[dict(start_s=0, end_s=3, start_bpm=60, end_bpm=60)])
        measured = audit.tempo_measure(truth, lambda _: (None, False))
        self.assertEqual((measured["queries"], measured["unavailable"]), (2, 2))
        self.assertIsNone(measured["p95_error_percent"])

    def test_null_error_and_smaller_coverage_are_not_improvements(self):
        before, _ = audit.beat_metrics([0.], [0.])
        after, _ = audit.beat_metrics([], [0.])
        reasons = audit.compare_metrics(before, after)
        self.assertIn("matched", reasons)
        self.assertIn("p95_absolute_error_ms", reasons)

    def test_frozen_failed_report_has_complete_cohorts_and_source_identity(self):
        report = json.loads((audit.ROOT / "evaluation/parity/dense-sequence-v1.json").read_text())
        for key, path in (("decoder_source_sha256", audit.DECODER),
                          ("runner_source_sha256", audit.RUNNER),
                          ("audit_source_sha256", audit.ROOT / "evaluation/parity/dense_sequence_audit.py"),
                          ("estimator_source_sha256", audit.CORE)):
            self.assertEqual(report[key], hashlib.sha256(path.read_bytes()).hexdigest())
        for key in ("promoted", "decoder_uses_truth", "training_run", "holdout_opened", "production_output_changed"):
            self.assertIs(report[key], False)
        self.assertEqual(len(report["authored"]), 42)
        self.assertEqual({c["name"]: len(c["cases"]) for c in report["cohorts"]}, {"artbeat": 15, "rubato": 25})
        for cohort in report["cohorts"]:
            self.assertTrue(cohort["complete"])
            self.assertEqual(cohort["regression_case_count"], len(cohort["cases"]))
            for case in cohort["cases"]:
                self.assertTrue(case["primary_beat_score_replay"])
                self.assertTrue(case["measurement"]["regression_reasons"])
                self.assertFalse(case["measurement"]["no_regression"])
        witnesses = list(report["identical_input_witness"].values())
        self.assertEqual(len(witnesses), 2)
        self.assertEqual(witnesses[0], witnesses[1])

    def test_bar_superset_objective_defect_is_not_a_training_verdict(self):
        # Characterization of the frozen bar term on a fixed beat path, not a
        # replacement implementation or a passing acceptance test for v1.
        contrasts = [8., 0., 0., 0., 8., 0., 0., 0.]
        score = lambda meter: sum(contrasts[::meter])
        self.assertEqual(score(2), score(4))
        contrasts[2] = 1.
        self.assertGreater(score(2), score(4))
        report = json.loads((audit.ROOT / "evaluation/parity/dense-sequence-v1.json").read_text())
        regular = [c for c in report["authored"] if c["id"].startswith("authored-120-")]
        self.assertEqual(len(regular), 40)
        self.assertTrue(all(c["measurement"]["meter_hypotheses"] == [2] for c in regular))
        intact = next(c["measurement"] for c in regular if c["id"] == "authored-120-120-120-intact-bar")
        self.assertEqual(intact["inferred_clock_beats"]["f1"], 1.)
        self.assertEqual(intact["inferred_clock_downbeats"]["precision"], .5)
        self.assertFalse(intact["no_regression"])


if __name__ == "__main__":
    unittest.main()
