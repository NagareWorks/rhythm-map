"""Independent probability-space check of the conditional, censored meter audit."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss

from joint_clock_diagnosis import cell, heads

ROOT = Path(__file__).resolve().parents[2]


def independent_inference(marks):
    """Dense matrices in probability space, not the Rust log-space graph code."""
    n = len(marks)
    nodes, weights = leggauss((n // 2 + 3) // 2)
    rates, weights = (nodes + 1) / 2, weights / 2
    labels = [(m, p) for m in range(2, 8) for p in range(m)]
    transition = np.zeros((len(rates), 27, 27))
    for i, (meter, phase) in enumerate(labels):
        for j, (next_meter, next_phase) in enumerate(labels):
            if phase + 1 < meter:
                transition[:, i, j] = float((next_meter, next_phase) == (meter, phase + 1))
            elif next_phase == 0:
                transition[:, i, j] = 1 - rates if next_meter == meter else rates / 5
    emission = np.array([[np.exp(mark) if phase == 0 else 1 for _, phase in labels]
                         for mark in marks])
    forward = np.empty((n, len(rates), 27))
    forward[0] = emission[0] * [1 / (6 * meter) for meter, _ in labels]
    for t in range(1, n):
        forward[t] = np.einsum("qi,qij->qj", forward[t - 1], transition) * emission[t]
    backward = np.ones_like(forward)
    for t in range(n - 2, -1, -1):
        backward[t] = np.einsum("qij,qj->qi", transition, backward[t + 1] * emission[t + 1])
    evidence = forward[-1].sum(axis=1)
    partition = weights @ evidence
    occupancy = np.einsum("tqs,tqs,q->ts", forward, backward, weights) / partition
    meters = np.array([[sum(row[i] for i, (m, _) in enumerate(labels) if m == meter)
                        for meter in range(2, 8)] for row in occupancy])
    downbeats = occupancy[:, [i for i, (_, phase) in enumerate(labels) if phase == 0]].sum(axis=1)
    return np.log(partition), weights @ (rates * evidence) / partition, meters, downbeats


class CensoredMeterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / "evaluation/parity/censored-meter-v1.json").read_bytes())
        cls.previous = json.loads((ROOT / "evaluation/parity/time-clock-v1.json").read_bytes())

    def test_frozen_sources_and_scope(self):
        for field, path in (
            ("decoder_source_sha256", "crates/rhythm-map-eval/examples/support/censored_meter.rs"),
            ("audit_source_sha256", "crates/rhythm-map-eval/examples/censored_meter.rs"),
            ("cell_source_sha256", "crates/rhythm-map-eval/examples/support/phase_likelihood.rs"),
            ("clock_report_sha256", "evaluation/parity/time-clock-v1.json"),
        ):
            self.assertEqual(self.report[field], hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
        for field in ("production_output_changed", "training_run", "holdout_opened",
                      "real_music_evaluated", "truth_supplied_to_decoder", "beat_clock_searched"):
            self.assertIs(self.report[field], False)
        self.assertIs(self.report["conditions_on_frozen_inferred_ticks"], True)
        self.assertEqual(len(self.report["cases"]), 7)
        self.assertEqual(len(self.report["controls"]), 8)
        for kind in ("cases", "controls"):
            for old, new in zip(self.previous[kind], self.report[kind], strict=True):
                for field in ("case", "beat_f64_le_sha256", "bar_f64_le_sha256"):
                    self.assertEqual(old[field], new[field])
                self.assertEqual(old["decoded"]["unavailable_spans"],
                                 new["conditional_meter"]["unavailable_spans"])
                for a, b in zip(old["decoded"]["runs"], new["conditional_meter"]["runs"], strict=True):
                    self.assertEqual(a["map_ticks"], b["frozen_ticks"])
                    self.assertEqual(a["edge_reference_frames"], b["unchanged_edge_reference_frames"])
                    self.assertEqual((a["start"], a["end"]), (b["start"], b["end"]))
                    self.assertEqual(len(b["frozen_ticks"]), len(b["meter"]["positions"]))
                    # Both frozen clocks and all mark cells remain inside their available run.
                    self.assertTrue(all(b["start"] <= t["frame"] < t["frame"] + t["period_frames"] <= b["end"]
                                        for t in b["frozen_ticks"]))

    def test_independently_reconstructed_mark_inputs(self):
        for row in self.report["cases"]:
            _, bar, _ = heads(row["case"])
            for run in row["conditional_meter"]["runs"]:
                expected = [cell(bar[t["frame"]:t["frame"] + t["period_frames"]]) for t in run["frozen_ticks"]]
                np.testing.assert_allclose(run["mark_log_ratios"], expected, atol=1e-12, rtol=0)
        pulse = cell([-2., -2.] + [-8.] * 21 + [-2.])
        for row in self.report["crop_controls"]:
            expected = [pulse if (i + row["authored_initial_phase"]) % row["authored_meter"] == 0 else 0
                        for i in range(row["visible_beats"])]
            np.testing.assert_allclose(row["mark_log_ratios"], expected, atol=1e-12, rtol=0)
        for row in self.report["meter_change_controls"]:
            expected = [pulse if (i % 24) % meter == 0 else 0
                        for i, meter in enumerate(row["authored_meters"])]
            np.testing.assert_allclose(row["mark_log_ratios"], expected, atol=1e-12, rtol=0)

    def test_every_frozen_marginal_and_hyperprior_moment(self):
        entries = [(run["mark_log_ratios"], run["meter"])
                   for row in self.report["cases"] + self.report["controls"]
                   for run in row["conditional_meter"]["runs"]]
        entries += [(row["mark_log_ratios"], row["inference"])
                    for row in self.report["crop_controls"] + self.report["meter_change_controls"]]
        for marks, actual in entries:
            logz, rate, meters, downbeats = independent_inference(marks)
            self.assertAlmostEqual(actual["log_ratio_to_reference"], logz, places=10)
            self.assertAlmostEqual(actual["mean_change_probability_per_bar"], rate, places=10)
            np.testing.assert_allclose([p["meter_probabilities"] for p in actual["positions"]], meters, atol=1e-11, rtol=0)
            np.testing.assert_allclose([p["downbeat_probability"] for p in actual["positions"]], downbeats, atol=1e-11, rtol=0)
            np.testing.assert_allclose(meters.sum(axis=1), 1, atol=1e-12, rtol=0)

    def test_crop_and_true_change_failures_remain_visible(self):
        rows = self.report["crop_controls"]
        self.assertEqual(len(rows), 139)
        successes = {m: 0 for m in range(2, 8)}
        for row in rows:
            best = [int(np.argmax(p["meter_probabilities"])) + 2 for p in row["inference"]["positions"]]
            successes[row["authored_meter"]] += all(m == row["authored_meter"] for m in best)
        self.assertEqual(successes, {2: 4, 3: 9, 4: 0, 5: 25, 6: 0, 7: 49})
        expected = {"four_to_three": 40, "three_to_four": 48, "four_to_two": 24, "two_to_four": 24}
        for row in self.report["meter_change_controls"]:
            best = [int(np.argmax(p["meter_probabilities"])) + 2 for p in row["inference"]["positions"]]
            self.assertEqual(sum(a == b for a, b in zip(best, row["authored_meters"], strict=True)), expected[row["case"]])
        # A flat cell has R=1: adding predicted bar starts there costs no emission.
        marks = next(r["mark_log_ratios"] for r in rows if r["authored_meter"] == 4
                     and r["authored_initial_phase"] == 0 and r["visible_beats"] == 48)
        self.assertAlmostEqual(sum(marks[::2]), sum(marks[::4]), places=12)

    def test_tail_improvement_is_not_confident_metadata_or_a_timing_fix(self):
        for row in self.report["cases"]:
            p = row["conditional_meter"]["runs"][0]["meter"]["positions"][-1]["meter_probabilities"]
            if row["case"] == "double_speed_weak_alternating":
                self.assertEqual(int(np.argmax(p)) + 2, 2)
            else:
                self.assertEqual(int(np.argmax(p)) + 2, 4)
                self.assertLess(p[2], 0.5)
        controls = {row["case"]: row for row in self.report["controls"]}
        flat = controls["flat"]["conditional_meter"]["runs"][0]["meter"]
        self.assertAlmostEqual(flat["log_ratio_to_reference"], 0, places=12)
        self.assertAlmostEqual(flat["mean_change_probability_per_bar"], 0.5, places=12)
        # Noise's small positive conditional score is NOT a clock detection decision.
        noise = controls["fixed_seed_noise"]["conditional_meter"]["runs"][0]["meter"]
        self.assertGreater(noise["log_ratio_to_reference"], 0)
        self.assertLess(max(noise["positions"][-1]["meter_probabilities"]), 0.25)


if __name__ == "__main__":
    unittest.main()
