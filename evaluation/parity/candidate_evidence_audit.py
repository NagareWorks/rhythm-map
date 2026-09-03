#!/usr/bin/env python3
"""Calibration-only descriptive audit. No fitting, cutoff search, or beat recovery."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parent.parent
SHIPPING = "beat-this-rten-observations-v2+decode-audio-v2"
POSITIVE = "missed_truth_support"
NEGATIVES = {"offbeat_subdivision_aligned", "offbeat_other"}
# Declared directions, not learned weights. Each feature is audited independently.
DIRECTIONS = {
    "confidence": 1, "downbeat_confidence": 1,
    "onset_strength": 1, "onset_low_strength": 1, "onset_mid_strength": 1,
    "onset_high_strength": 1, "harmonic_strength": 1, "relative_db": 1,
    "onset_relative_to_anchors": 1, "confidence_relative_to_anchors": 1,
    "midpoint_error_ratio": -1, "double_gap_residual": -1,
    "context_dispersion": -1,
}
DOMINANCE = ("confidence", "onset_relative_to_anchors", "midpoint_error_ratio", "double_gap_residual")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nearest(points, time_s):
    """Nearest actual evidence sample, no invented zero for missing evidence."""
    return min(points, key=lambda p: (abs(p["time_s"] - time_s), p["time_s"])) if points else None


def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator is not None and denominator > 0 else None


def candidate_features(observations, candidate, lock):
    """This function cannot see annotations, case IDs, or evaluation scores."""
    time_s = candidate["time_s"]
    beats = observations["beats"]
    times = [b["time_s"] for b in beats]
    onset = nearest(observations.get("onsets", []), time_s)
    activity = nearest(observations.get("activity", []), time_s)
    harmonic = nearest(observations.get("harmonic_changes", []), time_s)
    features = {name: None for name in DIRECTIONS}
    if harmonic is not None and abs(harmonic["time_s"] - time_s) > lock["accepted_event_exclusion_s"]:
        harmonic = None
    features.update(confidence=candidate["confidence"], downbeat_confidence=candidate["downbeat_confidence"],
                    relative_db=None if activity is None else activity["relative_db"],
                    harmonic_strength=None if harmonic is None else harmonic["strength"])
    for name in ("strength", "low_strength", "mid_strength", "high_strength"):
        features["onset_" + name] = None if onset is None else onset[name]
    right = bisect_right(times, time_s)
    left = right - 1
    features.update(gap_s=None, context_period_s=None, left_anchor_s=None, right_anchor_s=None)
    if left < 0 or right >= len(times):
        return features
    gap = times[right] - times[left]
    features.update(gap_s=gap, left_anchor_s=times[left], right_anchor_s=times[right],
                    midpoint_error_ratio=abs(time_s - (times[left] + times[right]) / 2) / gap,
                    confidence_relative_to_anchors=ratio(candidate["confidence"],
                        (beats[left]["confidence"] + beats[right]["confidence"]) / 2))
    anchors = [nearest(observations.get("onsets", []), t) for t in (times[left], times[right])]
    if all(p is not None for p in anchors):
        features["onset_relative_to_anchors"] = ratio(features["onset_strength"],
                                                     sum(p["strength"] for p in anchors) / 2)
    width = lock["context_intervals_each_side"]
    if left >= width and right + width < len(times):
        intervals = ([times[i + 1] - times[i] for i in range(left - width, left)] +
                     [times[i + 1] - times[i] for i in range(right, right + width)])
        period = statistics.median(intervals)
        features.update(context_period_s=period, double_gap_residual=abs(gap / (2 * period) - 1),
                        context_dispersion=statistics.median(abs(x - period) for x in intervals) / period)
    return features


def annotation_label(time_s, truth, covered, tolerance_s, lock):
    nearby = [i for i, t in enumerate(truth) if abs(t - time_s) <= tolerance_s]
    if len(nearby) > 1:
        return "ambiguous_truth_window", None
    if nearby:
        index = nearby[0]
        return ("covered_truth_duplicate" if index in covered else POSITIVE), index
    right = bisect_right(truth, time_s)
    if right == 0 or right == len(truth):
        return "offbeat_unanchored", None
    left, end = truth[right - 1], truth[right]
    aligned = any(abs(time_s - (left + fraction * (end - left))) <= lock["subdivision_alignment_s"]
                  for fraction in lock["subdivision_fractions"])
    return ("offbeat_subdivision_aligned" if aligned else "offbeat_other"), None


def case_rows(case, lock):
    observations = case["observations"]
    accepted = [b["time_s"] for b in observations["beats"]]
    covered = {pair[1] for pair in case["raw_truth_pairs"]}
    rows = []
    excluded = 0
    for candidate in observations.get("beat_candidates", []):
        time_s = candidate["time_s"]
        if accepted and min(abs(t - time_s) for t in accepted) <= lock["accepted_event_exclusion_s"]:
            excluded += 1
            continue
        features = candidate_features(observations, candidate, lock)
        # Labels are deliberately attached only after truth-free feature extraction.
        label, truth_index = annotation_label(time_s, case["truth_times_s"], covered, case["beat_tolerance_s"], lock)
        rows.append(dict(id=case["id"], time_s=time_s, features=features, label=label, truth_index=truth_index,
                         cohort="subthreshold" if candidate["confidence"] <= lock["primary_cohort_max_confidence"]
                         else "positive_logit_unselected"))
    return rows, dict(id=case["id"], tags=case.get("tags", []), truth_count=len(case["truth_times_s"]),
                     raw_count=len(accepted), raw_matched_truth_count=len(covered),
                     missed_truth_count=len(case["truth_times_s"]) - len(covered),
                     accepted_candidate_exclusions=excluded)


def auc(positive, negative, direction):
    """P(positive ranks higher), ties count half; no threshold is selected."""
    if not positive or not negative:
        return None
    negatives = sorted(direction * value for value in negative)
    wins = sum((bisect_left(negatives, direction * value) + bisect_right(negatives, direction * value)) / 2
               for value in positive)
    return wins / (len(positive) * len(negative))


def quantiles(values):
    return dict(zip(("min", "p10", "median", "p90", "max"),
                    np.quantile(values, [0, .1, .5, .9, 1]).tolist())) if values else None


def feature_stats(rows, negatives):
    positives = [r for r in rows if r["label"] == POSITIVE]
    negatives = [r for r in rows if r["label"] in negatives]
    result = {}
    for name, direction in DIRECTIONS.items():
        classes = [[r["features"][name] for r in group if r["features"][name] is not None]
                   for group in (positives, negatives)]
        per_track = []
        for case_id in sorted({r["id"] for r in positives + negatives}):
            local = [[r["features"][name] for r in group if r["id"] == case_id and r["features"][name] is not None]
                     for group in (positives, negatives)]
            value = auc(*local, direction)
            if value is not None:
                per_track.append(value)
        result[name] = dict(direction=direction, positive_available=len(classes[0]), negative_available=len(classes[1]),
                            positive_missing=len(positives) - len(classes[0]), negative_missing=len(negatives) - len(classes[1]),
                            positive_quantiles=quantiles(classes[0]), negative_quantiles=quantiles(classes[1]),
                            pooled_auc=auc(*classes, direction),
                            macro_track_auc=statistics.mean(per_track) if per_track else None,
                            tracks_with_both_classes=len(per_track))
    return result


def cohort_summary(rows):
    positive = [r for r in rows if r["label"] == POSITIVE]
    return dict(candidate_count=len(rows), labels=dict(sorted(Counter(r["label"] for r in rows).items())),
                distinct_missed_truth_supported=len({(r["id"], r["truth_index"]) for r in positive}),
                extra_support_candidates_for_same_truth=len(positive) - len({(r["id"], r["truth_index"]) for r in positive}))


def dominate(row, probe):
    return all(row["features"][name] is not None and probe["features"][name] is not None and
               DIRECTIONS[name] * row["features"][name] >= DIRECTIONS[name] * probe["features"][name]
               for name in DOMINANCE)


def build_summary(evidence, lock):
    rows, cases = [], []
    for case in evidence["cases"]:
        local, counts = case_rows(case, lock)
        rows.extend(local)
        primary = [r for r in local if r["cohort"] == "subthreshold"]
        counts.update(cohorts={cohort: cohort_summary([r for r in local if r["cohort"] == cohort])
                              for cohort in ("subthreshold", "positive_logit_unselected")},
                      subthreshold_feature_auc={name: stats["pooled_auc"]
                                               for name, stats in feature_stats(primary, NEGATIVES).items()})
        cases.append(counts)
    primary = [r for r in rows if r["cohort"] == "subthreshold"]
    probe_rows, _ = case_rows(evidence["probe"], lock)
    probes = [r for r in probe_rows if abs(r["time_s"] - lock["probe_time_s"]) < 1e-6]
    require(len(probes) == 1 and probes[0]["label"] == POSITIVE, "fixed missed-beat probe not recovered in evidence")
    probe = probes[0]
    comparable = [r for r in primary if r["label"] in NEGATIVES and
                  all(r["features"][name] is not None for name in DOMINANCE)]
    dominated = [r for r in comparable if dominate(r, probe)]
    tags = sorted({tag for case in cases for tag in case["tags"]})
    summary = dict(schema_version=1, purpose="calibration_candidate_evidence_separability_not_recovery_policy",
                   lock_sha256=evidence["lock_sha256"], suite_sha256=evidence["suite_sha256"],
                   cache_hits=evidence["cache_hits"], neural_inferences=0, cache_writes=0,
                   replay_exact_count=sum(c["score_replay_exact"] for c in evidence["cases"]),
                   probe_replay_exact=evidence["probe"]["score_replay_exact"],
                   production_changed=False, promotion=False, threshold_search=False, holdout_used=False,
                   cohort_definitions=dict(primary="unselected model candidates with confidence <= 0.5",
                       secondary="positive-logit candidates unselected by the existing wider peak decoder",
                       positive="candidate within unchanged beat tolerance of a missed annotated main beat",
                       negative="within annotated span, outside all annotated main-beat tolerance windows",
                       subdivision="aligned with a fixed truth-interval fraction; NOT a verified subdivision note",
                       exclusion="accepted-near candidates, covered-truth duplicates, ambiguous windows, and unanchored negatives excluded from AUC"),
                   limitations=["ARTBeaT calibration only; not independent generalization evidence",
                       "Candidates from one track are correlated; pooled AUC is descriptive, not a significance test",
                       "Evidence uses raw decoder anchors, not a newly recovered or iterated beat sequence",
                       "No subdivision-note annotation or safe recovery decision is claimed",
                       "Probe uses candidate resampling and remains separate from shipping cohort"],
                   cohorts={cohort: cohort_summary([r for r in rows if r["cohort"] == cohort])
                            for cohort in ("subthreshold", "positive_logit_unselected")},
                   primary_features_vs_all_anchored_offbeats=feature_stats(primary, NEGATIVES),
                   primary_features_vs_subdivision_aligned=feature_stats(primary, {"offbeat_subdivision_aligned"}),
                   cases=cases,
                   tag_slices={tag: cohort_summary([r for r in primary if r["id"] in
                                                  {c["id"] for c in cases if tag in c["tags"]}]) for tag in tags},
                   fixed_probe=dict(candidate=probe, dominance_features=list(DOMINANCE),
                       all_features_available=all(probe["features"][name] is not None for name in DOMINANCE),
                       comparable_negative_count=len(comparable), dominating_negative_count=len(dominated),
                       dominating_negative_track_count=len({r["id"] for r in dominated}),
                       representative_counterexamples=sorted(dominated, key=lambda r: (r["id"], r["time_s"]))[:5]))
    return summary, rows


def validate_inputs(evidence, lock):
    require(evidence.get("schema_version") == 1 and evidence.get("purpose") == "private_calibration_candidate_evidence",
            "not a private calibration export")
    require(evidence["lock_sha256"] == digest(DIRECTORY / "candidate-evidence-lock-v1.json"), "changed lock")
    require(evidence["suite_sha256"] == lock["suite_sha256"] == digest(ROOT / "evaluation/suites/artbeat-v1.json"), "changed suite")
    require(evidence["cache_hits"] == 15 and evidence["neural_inferences"] == evidence["cache_writes"] == 0,
            "requires all 15 cache-only replays")
    require(evidence["observation_contract"] == SHIPPING, "not shipping observation contract")
    for field, relative in {
        "source_sha256": "crates/rhythm-map-eval/src/candidate_evidence.rs",
        "cache_source_sha256": "crates/rhythm-map-eval/src/observation_cache.rs",
        "engine_source_sha256": "crates/rhythm-map-core/src/engine.rs",
        "estimator_source_sha256": "crates/rhythm-map-core/src/estimator.rs",
        "model_manifest_sha256": "models/beat-this-full-v1.json",
    }.items():
        require(evidence[field] == digest(ROOT / relative), f"changed source/manifest: {field}")
    for filename, key in (("reference-resampler-calibration-v1.json", "calibration_report_sha256"),
                          ("resampler-regression-event-v1.json", "event_audit_sha256")):
        require(digest(DIRECTORY / filename) == lock[key], "changed frozen historical evidence")
    frozen = json.loads((DIRECTORY / "reference-resampler-calibration-v1.json").read_bytes())["suites"][0]["cases"]
    audit = json.loads((DIRECTORY / "resampler-regression-event-v1.json").read_bytes())
    require(len(evidence["cases"]) == len(frozen) == 15, "incomplete cohort")
    suite = json.loads((ROOT / "evaluation/suites/artbeat-v1.json").read_bytes())
    for case, original, definition in zip(evidence["cases"], frozen, suite["cases"]):
        require(case["id"] == original["id"] and case["pcm_sha256"] == original["current_pcm_sha256"]
                and case["truth_sha256"] == original["truth_sha256"] and case["audio_sha256"] == original["audio_sha256"],
                "changed case identity")
        truth_path = ROOT / "evaluation/suites" / definition["input"]["truth"]
        require(digest(truth_path) == case["truth_sha256"] and case["truth_times_s"] ==
                [b["time_s"] for b in json.loads(truth_path.read_bytes())["beats"]], "changed truth values")
    probe = evidence["probe"]
    require(probe["id"] == lock["probe_case_id"] and probe["source_trace_sha256"] == audit["after_trace_sha256"]
            and probe["observation_contract"] == SHIPPING + "+phase-exact-bh2-256-v1", "changed fixed probe")
    original_probe = next(c for c in frozen if c["id"] == probe["id"])
    require(probe["pcm_sha256"] == original_probe["candidate_pcm_sha256"] and probe["truth_times_s"] ==
            next(c["truth_times_s"] for c in evidence["cases"] if c["id"] == probe["id"]), "changed probe PCM/truth")
    for case in evidence["cases"] + [probe]:
        require(case["score_replay_exact"] and case["sample_rate"] == 22050, "replay not verified")
        truth = case["truth_times_s"]
        require(truth and all(math.isfinite(t) for t in truth) and all(b > a for a, b in zip(truth, truth[1:])),
                "timestamp truth required")
        require(case["beat_tolerance_s"] == .07, "changed label tolerance")
        pairs = case["raw_truth_pairs"]
        require(len({p[0] for p in pairs}) == len(pairs) == len({p[1] for p in pairs}), "reused event match")
        require(all(0 <= p[0] < len(case["observations"]["beats"]) and 0 <= p[1] < len(truth) and
                    abs(case["observations"]["beats"][p[0]]["time_s"] - truth[p[1]]) <= .07 for p in pairs),
                "invalid frozen raw/truth match")
        for key in ("beats", "beat_candidates", "onsets", "activity", "harmonic_changes"):
            points = case["observations"].get(key, [])
            require(all(math.isfinite(v) for p in points for v in p.values()), "non-finite evidence")
            require(all(b["time_s"] > a["time_s"] for a, b in zip(points, points[1:])), "unsorted evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--private-rows", type=Path)
    args = parser.parse_args()
    lock = json.loads((DIRECTORY / "candidate-evidence-lock-v1.json").read_bytes())
    evidence = json.loads(args.evidence.read_bytes())
    validate_inputs(evidence, lock)
    report, rows = build_summary(evidence, lock)
    report.update(private_evidence_sha256=digest(args.evidence), analysis_source_sha256=digest(__file__),
                  sources={k: v for k, v in evidence.items() if k.endswith("_sha256")})
    if args.private_rows:
        require(not args.private_rows.resolve().is_relative_to(ROOT.resolve()), "dense rows must stay outside repository")
        with args.private_rows.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(rows, stream, allow_nan=False)
            stream.write("\n")
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(cases=len(report["cases"]), cohorts=report["cohorts"],
                          probe_dominating_negatives=report["fixed_probe"]["dominating_negative_count"]), indent=2))


if __name__ == "__main__":
    main()
