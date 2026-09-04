"""Frozen calibration-only acoustic phase evidence; no decoder or model fitting.

Feature extraction sees every consecutive raw-event interval, never truth or a
case ID. Compare onset maxima at 1/2 phase against the mean at 1/4 and 3/4 phase.
Use radius min(50 ms, interval/16), separately in four existing onset bands.
Also average contrast across exactly five consecutive raw intervals; incomplete
contexts stay unavailable. Larger values are declared favorable before running.
Labels are attached afterward: two matched anchors advance either one truth
beat (negative) or two (one missed beat, positive). Other intervals are reported
but not assigned either label. No threshold, event insertion or strategy choice.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from candidate_evidence_audit import auc

INPUTS = {
    "artbeat": (15, "3f1ba43fd4f373579727a48668d8de8e00166523d2d1141e072bc3471a71ab3e"),
    "rubato": (25, "ce5e678276888a0e430c004444dce4b27f0cfac0761767736abee2ec3fc05937"),
}
BANDS = ("strength", "low_strength", "mid_strength", "high_strength")
FEATURES = tuple(f"{kind}_{band}" for kind in ("midpoint", "contrast", "sequence") for band in BANDS)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordered(times):
    return all(math.isfinite(t) and t >= 0 for t in times) and all(a < b for a, b in zip(times, times[1:]))


def phase_features(observations):
    """Truth-free extraction; missing samples are None, not silence or negatives."""
    beats = [b["time_s"] for b in observations["beats"]]
    onsets = observations.get("onsets", [])
    times = [p["time_s"] for p in onsets]
    require(ordered(beats) and ordered(times), "invalid event/onset ordering")
    require(all(all(math.isfinite(p[b]) and 0 <= p[b] <= 1 for b in BANDS) for p in onsets),
            "invalid onset value")

    def peak(center, radius, band):
        if not times or center - radius < times[0] or center + radius > times[-1]:
            return None
        left, right = bisect_left(times, center - radius), bisect_right(times, center + radius)
        return max((p[band] for p in onsets[left:right]), default=None)

    rows = []
    for start, end in zip(beats, beats[1:]):
        span = end - start
        radius = min(.05, span / 16)
        features = dict.fromkeys(FEATURES)
        for band in BANDS:
            quarter, middle, three_quarters = [peak(start + phase * span, radius, band)
                                               for phase in (.25, .5, .75)]
            features[f"midpoint_{band}"] = middle
            if all(v is not None for v in (quarter, middle, three_quarters)):
                features[f"contrast_{band}"] = middle - (quarter + three_quarters) / 2
        rows.append(features)
    for index in range(2, len(rows) - 2):
        for band in BANDS:
            values = [rows[j][f"contrast_{band}"] for j in range(index - 2, index + 3)]
            if all(v is not None for v in values):
                rows[index][f"sequence_{band}"] = statistics.mean(values)
    return rows


def matches(predicted, truth, tolerance):
    """Existing chronological one-to-one raw/truth convention, identities retained."""
    require(ordered(predicted) and ordered(truth), "invalid scoring order")
    require(math.isfinite(tolerance) and tolerance > 0, "invalid tolerance")
    i = j = 0
    result = []
    while i < len(predicted) and j < len(truth):
        delta = predicted[i] - truth[j]
        if abs(delta) <= tolerance:
            result.append([i, j])
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return result


def near(times, t, tolerance):
    index = bisect_left(times, t)
    return any(abs(x - t) <= tolerance for x in times[max(0, index - 1):index + 1])


def label_case(case):
    obs = case["observations"]
    features = phase_features(obs)  # No labels cross this boundary.
    times = [b["time_s"] for b in obs["beats"]]
    truth, tolerance = case["truth_times_s"], case["beat_tolerance_s"]
    pairs = matches(times, truth, tolerance)
    require(pairs == case["raw_truth_pairs"], "raw/truth identity replay failed")
    anchors = dict(pairs)
    candidates = [p["time_s"] for p in obs.get("beat_candidates", [])]
    require(ordered(candidates), "invalid candidate ordering")
    rows = []
    for index, feature in enumerate(features):
        row = dict(features=feature, label="unmatched_anchor", midpoint_reaches_truth=None,
                   missed_truth_has_candidate=None)
        if index in anchors and index + 1 in anchors:
            advance = anchors[index + 1] - anchors[index]
            require(advance > 0, "invalid truth advancement")
            row["label"] = "one_beat" if advance == 1 else "missing_one" if advance == 2 else "missing_multiple"
            if advance == 2:
                target = truth[anchors[index] + 1]
                midpoint = (times[index] + times[index + 1]) / 2
                row["midpoint_reaches_truth"] = abs(target - midpoint) <= tolerance
                row["missed_truth_has_candidate"] = near(candidates, target, tolerance)
        rows.append(row)
    return rows, dict(
        raw_intervals=len(rows), classes=dict(Counter(r["label"] for r in rows)),
        total_missed_truth=len(truth) - len(pairs),
        missing_one_midpoint_reachable=sum(r["midpoint_reaches_truth"] is True for r in rows),
        missing_one_midpoint_unreachable=sum(r["midpoint_reaches_truth"] is False for r in rows),
        missing_one_without_model_candidate=sum(r["missed_truth_has_candidate"] is False for r in rows),
        dense_model_series_present=obs.get("activations") is not None,
        onset_samples=len(obs.get("onsets", [])), harmonic_samples=len(obs.get("harmonic_changes", [])))


def feature_summary(rows, feature, positive_filter=lambda row: True):
    positive = [r for r in rows if r["label"] == "missing_one" and positive_filter(r)]
    negative = [r for r in rows if r["label"] == "one_beat"]
    p, n = [[r["features"][feature] for r in group if r["features"][feature] is not None]
            for group in (positive, negative)]
    return dict(positive_available=len(p), negative_available=len(n),
                positive_missing=len(positive) - len(p), negative_missing=len(negative) - len(n),
                auc_larger_favors_missing=auc(p, n, 1),
                positive_median=statistics.median(p) if p else None,
                negative_median=statistics.median(n) if n else None)


def cohort_report(name, path):
    count, expected = INPUTS[name]
    require(digest(path) == expected, "frozen calibration evidence hash mismatch")
    evidence = json.loads(Path(path).read_bytes())
    cases = evidence["cases"]  # Deliberately excludes ARTBeaT's separate probe.
    require(len(cases) == count and len({c["id"] for c in cases}) == count, "cohort mismatch")
    all_rows, case_reports = [], []
    for case in cases:
        require(case["score_replay_exact"] is True, "historical replay not exact")
        rows, counts = label_case(case)
        case_reports.append(dict(id=case["id"], counts=counts,
                                 features={f: feature_summary(rows, f) for f in FEATURES}))
        all_rows.extend(rows)
    summary = {}
    for feature in FEATURES:
        values = [c["features"][feature]["auc_larger_favors_missing"] for c in case_reports]
        values = [v for v in values if v is not None]
        summary[feature] = dict(**feature_summary(all_rows, feature),
                               macro_track_auc=statistics.mean(values) if values else None,
                               macro_track_count=len(values))
    # A sequence/point contrast comparison must use identical rows, not let
    # sequence-context abstention silently remove its harder examples.
    matched = {}
    for band in BANDS:
        subset = [r for r in all_rows if all(r["features"][f"{k}_{band}"] is not None
                                           for k in ("midpoint", "contrast", "sequence"))]
        matched[band] = {k: feature_summary(subset, f"{k}_{band}")
                         for k in ("midpoint", "contrast", "sequence")}
    return dict(cohort=name, evidence_sha256=expected, cases=case_reports,
                raw_intervals=len(all_rows), classes=dict(Counter(r["label"] for r in all_rows)),
                features=summary, matched_sample_comparison=matched,
                no_model_candidate_positives={f: feature_summary(all_rows, f, lambda r: r["missed_truth_has_candidate"] is False)
                                             for f in FEATURES})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artbeat", type=Path, required=True)
    parser.add_argument("--rubato", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite report")
    report = dict(schema_version=1, purpose="frozen_acoustic_clock_phase_evidence",
                  source_sha256=digest(__file__),
                  auc_source_sha256=digest(Path(__file__).with_name("candidate_evidence_audit.py")),
                  inferred_beats_emitted=False, inference_run=False, training_run=False,
                  decision="descriptive_only_no_automatic_adoption",
                  cohorts=[cohort_report(n, p) for n, p in (("artbeat", args.artbeat), ("rubato", args.rubato))])
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
