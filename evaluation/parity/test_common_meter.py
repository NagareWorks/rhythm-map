"""Independent count-augmented probability-space reconstruction of the audit."""
import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss

from joint_clock_diagnosis import heads

ROOT = Path(__file__).resolve().parents[2]


def independent_inference(values):
    scores = np.asarray(values) - max(values)
    n, counts = len(scores), (len(scores) + 1) // 2 + 1
    # Explicit polynomial multiplication, rather than the Rust log-space DP.
    coefficients = np.array([1.])
    for x in scores:
        coefficients = np.convolve(coefficients, [1., np.exp(x)])
    reference_means = coefficients[:counts] / [math.comb(n, k) for k in range(counts)]
    nodes, weights = leggauss((n // 2 + 3) // 2)
    rates, weights = (nodes + 1) / 2, weights / 2
    labels = [(m, p) for m in range(2, 8) for p in range(m)]
    downbeat = np.array([p == 0 for _, p in labels])
    transition = np.zeros((len(rates), 27, 27))
    for i, (meter, phase) in enumerate(labels):
        for j, (next_meter, next_phase) in enumerate(labels):
            if phase + 1 < meter:
                transition[:, i, j] = float((next_meter, next_phase) == (meter, phase + 1))
            elif next_phase == 0:
                transition[:, i, j] = 1 - rates if next_meter == meter else rates / 5
    forward = np.zeros((n, len(rates), counts, 27))
    for j, (meter, phase) in enumerate(labels):
        k = int(phase == 0)
        forward[0, :, k, j] = np.exp(scores[0] * k) / (6 * meter)
    for t in range(1, n):
        before_mark = np.einsum("qki,qij->qkj", forward[t - 1], transition)
        for j in range(27):
            if downbeat[j]:
                forward[t, :, 1:, j] = before_mark[:, :-1, j] * np.exp(scores[t])
            else:
                forward[t, :, :, j] = before_mark[:, :, j]
    backward = np.zeros_like(forward)
    backward[-1] = (1 / reference_means)[None, :, None]
    for t in range(n - 2, -1, -1):
        after_mark = backward[t + 1].copy()
        for j in range(27):
            if downbeat[j]:
                after_mark[:, :-1, j] = backward[t + 1, :, 1:, j] * np.exp(scores[t + 1])
                after_mark[:, -1, j] = 0
        backward[t] = np.einsum("qij,qkj->qki", transition, after_mark)
    terminal = forward[-1].sum(axis=2) / reference_means
    evidence = terminal.sum(axis=1)
    partition = weights @ evidence
    occupancy = np.einsum("tqks,tqks,q->ts", forward, backward, weights) / partition
    meters = np.array([[sum(row[i] for i, (m, _) in enumerate(labels) if m == meter)
                        for meter in range(2, 8)] for row in occupancy])
    downbeats = occupancy[:, downbeat].sum(axis=1)
    return (np.log(partition), weights @ (rates * evidence) / partition, meters,
            downbeats, weights @ terminal / partition)


class CommonMeterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / "evaluation/parity/common-meter-v1.json").read_bytes())
        cls.previous = json.loads((ROOT / "evaluation/parity/censored-meter-v1.json").read_bytes())

    def test_frozen_sources_scope_and_unchanged_clock_inputs(self):
        for field, path in (
            ("decoder_source_sha256", "crates/rhythm-map-eval/examples/support/common_meter.rs"),
            ("audit_source_sha256", "crates/rhythm-map-eval/examples/common_meter.rs"),
            ("previous_report_sha256", "evaluation/parity/censored-meter-v1.json"),
            ("clock_report_sha256", "evaluation/parity/time-clock-v1.json"),
        ):
            self.assertEqual(self.report[field], hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
        for field in ("production_output_changed", "training_run", "holdout_opened",
                      "real_music_evaluated", "truth_supplied_to_decoder", "beat_clock_searched"):
            self.assertIs(self.report[field], False)
        self.assertIs(self.report["conditions_on_frozen_inferred_ticks"], True)
        for kind, count in (("cases", 7), ("controls", 8)):
            self.assertEqual(len(self.report[kind]), count)
            for old, new in zip(self.previous[kind], self.report[kind], strict=True):
                for field in ("case", "beat_f64_le_sha256", "bar_f64_le_sha256"):
                    self.assertEqual(old[field], new[field])
                self.assertEqual(old["conditional_meter"]["unavailable_spans"], new["conditional_meter"]["unavailable_spans"])
                for a, b in zip(old["conditional_meter"]["runs"], new["conditional_meter"]["runs"], strict=True):
                    for key in ("frozen_ticks", "start", "end", "unchanged_edge_reference_frames"):
                        self.assertEqual(a[key], b[key])

    def test_common_context_does_not_change_the_raw_pulse_inputs(self):
        for row in self.report["cases"]:
            _, bar, _ = heads(row["case"])
            for run in row["conditional_meter"]["runs"]:
                expected = []
                for t in run["frozen_ticks"]:
                    start, duration = t["frame"], t["period_frames"]
                    expected.append(0.25 * bar[start + duration - 1] + 0.5 * bar[start] + 0.25 * bar[start + 1])
                np.testing.assert_allclose(run["mark_scores"], expected, atol=1e-12, rtol=0)
        for row in self.report["crop_controls"]:
            expected = [-2 if (i + row["authored_initial_phase"]) % row["authored_meter"] == 0 else -8
                        for i in range(row["visible_beats"])]
            self.assertEqual(row["mark_scores"], expected)
        for row in self.report["meter_change_controls"]:
            expected = [-2 if (i % 24) % meter == 0 else -8 for i, meter in enumerate(row["authored_meters"])]
            self.assertEqual(row["mark_scores"], expected)

    def test_every_probability_matches_independent_counted_reconstruction(self):
        entries = [(run["mark_scores"], run["meter"])
                   for row in self.report["cases"] + self.report["controls"]
                   for run in row["conditional_meter"]["runs"]]
        entries += [(row["mark_scores"], row["inference"])
                    for row in self.report["crop_controls"] + self.report["meter_change_controls"]]
        for marks, actual in entries:
            logz, rate, meters, downbeats, counts = independent_inference(marks)
            self.assertAlmostEqual(actual["log_ratio_to_reference"], logz, places=9)
            self.assertAlmostEqual(actual["mean_change_probability_per_bar"], rate, places=9)
            np.testing.assert_allclose([p["meter_probabilities"] for p in actual["positions"]], meters, atol=1e-10, rtol=0)
            np.testing.assert_allclose([p["downbeat_probability"] for p in actual["positions"]], downbeats, atol=1e-10, rtol=0)
            np.testing.assert_allclose(actual["count_probabilities"], counts, atol=1e-10, rtol=0)
            np.testing.assert_allclose(meters.sum(axis=1), 1, atol=1e-10, rtol=0)
            self.assertAlmostEqual(sum(downbeats), np.arange(len(counts)) @ counts, places=9)

    def test_all_crops_and_true_meter_changes_pass_without_clock_recovery_claims(self):
        self.assertEqual(len(self.report["crop_controls"]), 139)
        self.assertEqual(len(self.report["meter_change_controls"]), 4)
        for row in self.report["crop_controls"]:
            best = [int(np.argmax(p["meter_probabilities"])) + 2 for p in row["inference"]["positions"]]
            self.assertEqual(best, [row["authored_meter"]] * row["visible_beats"])
        for row in self.report["meter_change_controls"]:
            best = [int(np.argmax(p["meter_probabilities"])) + 2 for p in row["inference"]["positions"]]
            self.assertEqual(best, row["authored_meters"])
        rows = {row["case"]: row for row in self.report["cases"] + self.report["controls"]}
        for name in ("constant_intact", "constant_all_weak", "flat_middle"):
            positions = rows[name]["conditional_meter"]["runs"][0]["meter"]["positions"]
            self.assertTrue(all(int(np.argmax(p["meter_probabilities"])) + 2 == 4 for p in positions))
        weak = rows["double_speed_weak_alternating"]["conditional_meter"]["runs"][0]
        self.assertEqual(len(weak["frozen_ticks"]), 47)  # Authored clock still has 64 beats.
        best = [int(np.argmax(p["meter_probabilities"])) + 2 for p in weak["meter"]["positions"]]
        self.assertEqual(best.count(2), 16)  # Wrong supplied clock still explains doubling as meter.
        self.assertEqual(best.count(4), 31)
        flat = rows["flat"]["conditional_meter"]["runs"][0]["meter"]
        self.assertAlmostEqual(flat["log_ratio_to_reference"], 0, places=12)
        noise = rows["fixed_seed_noise"]["conditional_meter"]["runs"][0]["meter"]
        self.assertLess(noise["log_ratio_to_reference"], 0)  # One draw, not a false-positive rate.


if __name__ == "__main__":
    unittest.main()
