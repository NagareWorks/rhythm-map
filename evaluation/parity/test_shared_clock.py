"""Independent shared-frame normalizer and conditional meter reconstruction."""
import hashlib
import json
import math
import struct
from pathlib import Path
import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss
from joint_clock_diagnosis import heads, PERIODS
from time_clock_diagnosis import RATE, OFF_MASS

ROOT = Path(__file__).resolve().parents[2]


def independent_inference(values, norm):
    observed = np.array([v is not None for v in values])
    scores = np.array([0. if v is None else v for v in values])
    n, counts = len(scores), len(norm)
    shift = min(norm)
    terminal_factor = np.exp(shift - np.asarray(norm))
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
        k = int(phase == 0 and observed[0])
        forward[0, :, k, j] = np.exp(scores[0] * k) / (6 * meter)
    for t in range(1, n):
        before_mark = np.einsum("qki,qij->qkj", forward[t - 1], transition)
        for j in range(27):
            if downbeat[j] and observed[t]:
                forward[t, :, 1:, j] = before_mark[:, :-1, j] * np.exp(scores[t])
            else:
                forward[t, :, :, j] = before_mark[:, :, j]
    backward = np.zeros_like(forward)
    backward[-1] = terminal_factor[None, :, None]
    for t in range(n - 2, -1, -1):
        after_mark = backward[t + 1].copy()
        for j in range(27):
            if downbeat[j] and observed[t + 1]:
                after_mark[:, :-1, j] = backward[t + 1, :, 1:, j] * np.exp(scores[t + 1])
                after_mark[:, -1, j] = 0
        backward[t] = np.einsum("qij,qkj->qki", transition, after_mark)
    terminal = forward[-1].sum(axis=2) * terminal_factor
    evidence = terminal.sum(axis=1)
    partition = weights @ evidence
    occupancy = np.einsum("tqks,tqks,q->ts", forward, backward, weights) / partition
    meters = np.array([[sum(row[i] for i, (m, _) in enumerate(labels) if m == meter)
                        for meter in range(2, 8)] for row in occupancy])
    downbeats = occupancy[:, downbeat].sum(axis=1)
    return (np.log(partition) - shift, weights @ (rates * evidence) / partition, meters,
            downbeats, weights @ terminal / partition)


def inputs(name):
    beat, bar, _ = heads(name if name in PERIODS else "constant_intact")
    available = np.ones(1152, dtype=bool)
    if name.startswith("constant_erased"):
        for t in range(412, 768, 48):
            beat[t-1:t+2] = [-8.] * 3
        if name.endswith("and_bars"):
            for t in range(484, 768, 192):
                bar[t-1:t+2] = [-8.] * 3
    if name == "flat":
        beat, bar = [-8.] * 1152, [-8.] * 1152
    if name == "flat_middle":
        beat[480:672], bar[480:672] = [-8.] * 192, [-8.] * 192
    if name == "unavailable_gap":
        available[480:672] = False
    if name == "all_unavailable":
        available[:] = False
    if name == "fixed_seed_noise":
        seed = 0x13572468
        for array in (beat, bar):
            for i in range(1152):
                seed = (seed * 1664525 + 1013904223) & 0xffffffff
                array[i] = float(np.float32(-8) + np.float32(np.float32(seed >> 24) / np.float32(255)) * np.float32(6))
    return np.array(beat), np.array(bar), available


def feature_table(beat, bar, available, contextual):
    scores = np.full((1152, 2), np.nan)
    indices = np.flatnonzero(available)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    pairs = np.stack((beat, bar), axis=1)
    for run in runs:
        if len(run) == 0:
            continue
        smooth = np.empty((len(run), 2))
        for offset, t in enumerate(run):
            near = run[max(0, offset-1):offset+2]
            weights = np.where(near == t, 2., 1.)
            smooth[offset] = (pairs[near] * weights[:, None]).sum(axis=0) / sum(weights)
        for offset, t in enumerate(run):
            scores[t] = smooth[offset]
            if contextual:
                window = smooth[max(0, offset-4):offset+5]
                scores[t] -= np.logaddexp.reduce(window, axis=0) - math.log(len(window))
    if len(indices):
        scores[indices] -= scores[indices].max(axis=0)
    a_max, d_max = min(64, len(indices)), min(32, len(indices))
    coeff = np.full((a_max+1, d_max+1), -np.inf)
    coeff[0, 0] = 0
    for b, d in scores[indices]:
        old = coeff.copy()
        coeff[1:] = np.logaddexp(coeff[1:], old[:-1] + b)
        coeff[:, 1:] = np.logaddexp(coeff[:, 1:], old[:, :-1] + b + d)
    for a in range(a_max+1):
        for d in range(min(d_max, len(indices)-a)+1):
            coeff[a, d] -= (math.lgamma(len(indices)+1) - math.lgamma(a+1)
                            - math.lgamma(d+1) - math.lgamma(len(indices)-a-d+1))
    return scores, coeff


class SharedClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / "evaluation/parity/shared-clock-v1.json").read_bytes())

    def test_source_identities_and_bounded_supplied_clock_scope(self):
        for field, path in (
            ("audit_source_sha256", "crates/rhythm-map-eval/examples/shared_clock.rs"),
            ("feature_source_sha256", "crates/rhythm-map-eval/examples/support/shared_frames.rs"),
            ("meter_source_sha256", "crates/rhythm-map-eval/examples/support/frame_meter.rs"),
            ("prior_source_sha256", "crates/rhythm-map-eval/examples/support/time_prior.rs"),
            ("clock_report_sha256", "evaluation/parity/time-clock-v1.json"),
        ):
            self.assertEqual(self.report[field], hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
        for key in ("production_output_changed", "training_run", "holdout_opened", "real_music_evaluated", "unrestricted_clock_search"):
            self.assertIs(self.report[key], False)
        for key in ("supplied_clock_templates", "truth_assisted_clock_family", "meter_paths_searched"):
            self.assertIs(self.report[key], True)
        self.assertEqual(len(self.report["cases"]), 14)

    def test_all_inputs_normalizers_marginals_and_family_weights(self):
        for row in self.report["cases"]:
            beat, bar, available = inputs(row["case"])
            for name, array in (("beat", beat), ("bar", bar)):
                self.assertEqual(row[name+"_f64_le_sha256"], hashlib.sha256(struct.pack("<1152d", *array)).hexdigest())
            for mode in ("raw", "contextual"):
                out = row[mode]
                features, coeff = feature_table(beat, bar, available, mode == "contextual")
                self.assertEqual(out["available_frames"], sum(available))
                logs, priors = [], []
                for clock in out["clocks"]:
                    path = clock["given_ticks"]
                    ids = [t for t, _ in path if available[t]]
                    b = len(ids)
                    self.assertEqual(clock["visible_beats"], b)
                    norm = [coeff[b-d, d] for d in range(min(b, (len(path)+1)//2)+1)]
                    np.testing.assert_allclose(clock["paired_log_normalizers"], norm, atol=1e-8, rtol=0)
                    marks = [float(features[t, 1]) if available[t] else None for t, _ in path]
                    np.testing.assert_allclose([v for v in clock["bar_mark_scores"] if v is not None],
                                               [v for v in marks if v is not None], atol=1e-12, rtol=0)
                    numerator = features[ids, 0].sum()
                    self.assertAlmostEqual(clock["beat_score_sum"], numerator, places=9)
                    z, rate, meters, downbeats, counts = independent_inference(marks, norm)
                    result = clock["meter"]
                    self.assertAlmostEqual(result["log_ratio_to_reference"], z, places=8)
                    self.assertAlmostEqual(result["mean_change_probability_per_bar"], rate, places=9)
                    np.testing.assert_allclose([p["meter_probabilities"] for p in result["positions"]], meters, atol=1e-9, rtol=0)
                    np.testing.assert_allclose([p["downbeat_probability"] for p in result["positions"]], downbeats, atol=1e-9, rtol=0)
                    np.testing.assert_allclose(result["count_probabilities"], counts, atol=1e-9, rtol=0)
                    self.assertAlmostEqual(sum(p for p, m in zip(downbeats, marks) if m is not None), np.arange(len(counts))@counts, places=8)
                    duration = -math.log(66) - RATE * (1152-path[-1][0])
                    for (_, p), (_, q) in zip(path, path[1:]):
                        duration += (-RATE*p if p == q else math.log(-math.expm1(-RATE*p))-math.log(OFF_MASS[p])-math.log(100)*abs(math.log2(p/q)))
                    self.assertAlmostEqual(clock["duration_prior_log_weight"], duration, places=9)
                    self.assertAlmostEqual(clock["joint_log_ratio"], z+numerator, places=8)
                    priors.append(duration)
                    logs.append(duration+z+numerator)
                prior_mass = np.logaddexp.reduce(priors)
                total = np.logaddexp.reduce(logs)
                self.assertAlmostEqual(out["clock_family_log_ratio"], total-prior_mass, places=8)
                np.testing.assert_allclose(out["clock_family_probabilities"], np.exp(np.array(logs)-total), atol=1e-9, rtol=0)

    def test_weak_doubling_progress_does_not_hide_half_speed_regression_or_prior_only_choices(self):
        rows = {r["case"]: r for r in self.report["cases"]}
        weak = rows["double_speed_weak_alternating"]
        self.assertEqual(np.argmax(weak["raw"]["clock_family_probabilities"]), 0)
        self.assertEqual(np.argmax(weak["contextual"]["clock_family_probabilities"]), 2)
        self.assertEqual(np.argmax(rows["half_speed_intact"]["raw"]["clock_family_probabilities"]), 1)
        self.assertEqual(np.argmax(rows["half_speed_intact"]["contextual"]["clock_family_probabilities"]), 0)
        erased, half = rows["constant_erased_beats_and_bars"], rows["half_speed_intact"]
        for key in ("beat_f64_le_sha256", "bar_f64_le_sha256", "raw", "contextual"):
            self.assertEqual(erased[key], half[key])
        for name in ("flat", "all_unavailable"):
            for mode in ("raw", "contextual"):
                out = rows[name][mode]
                self.assertAlmostEqual(out["clock_family_log_ratio"], 0, places=8)
                np.testing.assert_allclose(out["clock_family_probabilities"], np.exp(out["normalized_duration_prior"]), atol=1e-9, rtol=0)
                self.assertGreater(out["clock_family_probabilities"][0], 0.99)


if __name__ == "__main__":
    unittest.main()
