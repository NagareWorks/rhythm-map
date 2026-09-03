#!/usr/bin/env python3
"""Summarize both complete frozen suites without hiding per-case regressions."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from verify_bounded_resampler import digest, pcm_bytes

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parent.parent
SUITES = ("artbeat-v1", "fsld-tempo-v1")
SOURCE_PATHS = {
    "candidate_source_sha256": "crates/rhythm-map-eval/examples/support/reference_resampler.rs",
    "runner_source_sha256": "crates/rhythm-map-eval/examples/resampler_calibration.rs",
    "adapter_source_sha256": "crates/rhythm-map-beat-this/src/lib.rs",
    "audio_preprocessing_sha256": "crates/rhythm-map-beat-this/src/audio.rs",
    "core_engine_sha256": "crates/rhythm-map-core/src/engine.rs",
    "core_estimator_sha256": "crates/rhythm-map-core/src/estimator.rs",
    "metrics_source_sha256": "crates/rhythm-map-eval/src/metrics.rs",
}
# Display classification only. Exact scores and suite acceptance remain unchanged.
DISPLAY_EPSILON = 1e-9
METRICS = {
    "beat_f1": ("beats", "f1", True),
    "beat_median_error_ms": ("beats", "median_absolute_error_ms", False),
    "beat_p95_error_ms": ("beats", "p95_absolute_error_ms", False),
    "tempo_median_error_percent": ("tempo", "median_absolute_error_percent", False),
    "tempo_p95_error_percent": ("tempo", "p95_absolute_error_percent", False),
    "change_f1": ("changes", "f1", True),
    "change_recall": ("changes", "recall", True),
}


def summarize(cases, tempo_only=False):
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("empty or duplicate cases")
    summary = dict(case_count=len(cases), before_passed=sum(c["baseline"]["passed"] for c in cases),
                   after_passed=sum(c["candidate"]["passed"] for c in cases),
                   gained_passes=[c["id"] for c in cases if not c["baseline"]["passed"] and c["candidate"]["passed"]],
                   lost_passes=[c["id"] for c in cases if c["baseline"]["passed"] and not c["candidate"]["passed"]],
                   metrics={})
    for name, (group, key, higher_better) in METRICS.items():
        if tempo_only and group != "tempo":
            continue
        before = [c["baseline"]["metrics"][group][key] for c in cases]
        after = [c["candidate"]["metrics"][group][key] for c in cases]
        if any(value is None for value in before + after):
            summary["metrics"][name] = dict(available=False, reason="missing values are not averaged away")
            continue
        changes = [(b - a) * (1 if higher_better else -1) for a, b in zip(before, after)]
        summary["metrics"][name] = dict(available=True, before_mean=statistics.mean(before),
            after_mean=statistics.mean(after), delta_mean=statistics.mean(after) - statistics.mean(before),
            improved=[c["id"] for c, d in zip(cases, changes) if d > DISPLAY_EPSILON],
            regressed=[c["id"] for c, d in zip(cases, changes) if d < -DISPLAY_EPSILON],
            unchanged=[c["id"] for c, d in zip(cases, changes) if abs(d) <= DISPLAY_EPSILON])
    return summary


def validate_report(path, expected_suite):
    report = json.loads(path.read_bytes())
    suite_path = ROOT / "evaluation/suites" / f"{expected_suite}.json"
    suite = json.loads(suite_path.read_bytes())
    lock = json.loads((DIRECTORY / "resampling-v2-calibration.json").read_bytes())
    baseline = next(s for s in lock["suites"] if s["suite_id"] == expected_suite)
    if (report["suite_id"] != expected_suite or report["suite_purpose"] != "calibration"
        or report["suite_sha256"] != digest(suite_path)
        or report["purpose"] != "paired_resampler_calibration_not_release_acceptance"
        or report["baseline_report_sha256"] != baseline["after_report_sha256"]
        or report["model_manifest_sha256"] != baseline["model_manifest_sha256"]
        or report["shipping_observation_contract"] != baseline["after_contract"]):
        raise ValueError("suite, baseline, model, or purpose identity mismatch")
    if (report["baseline_replay_exact"] is not True or report["baseline_cache_hits"] != 15
        or report["candidate_cache_hits"] != 0 or len(report["cases"]) != 15):
        raise ValueError("incomplete paired inference or changed shipping replay")
    for key, relative in SOURCE_PATHS.items():
        if report[key] != digest(ROOT / relative):
            raise ValueError(f"stale source: {key}")
    bounded = json.loads((DIRECTORY / "resampler-bounded-v1.json").read_bytes())
    if (bounded["passed"] is not True or report["candidate_source_sha256"] != bounded["after_source_sha256"]
        or report["coefficient_budget_bytes"] != bounded["coefficient_budget_bytes"]
        or report["candidate"] != bounded["candidate"]
        or report["candidate_observation_contract"] != report["shipping_observation_contract"] + "+" + report["candidate"]):
        raise ValueError("candidate does not match the bit-identity evidence")
    for case, spec in zip(report["cases"], suite["cases"]):
        if (case["id"] != spec["id"] or case["baseline"]["id"] != spec["id"]
            or case["candidate"]["id"] != spec["id"] or case["oracle_unchanged"] is not True
            or case["audio_sha256"] != spec["input"]["audio"]["sha256"]
            or case["truth_sha256"] != digest(suite_path.parent / spec["input"]["truth"])):
            raise ValueError("case identity, truth, or oracle mismatch")
        wanted = (case["source_sample_count"] * 22050 + case["source_sample_rate"] // 2) // case["source_sample_rate"]
        if case["model_sample_count"] != wanted:
            raise ValueError("candidate duration changed")
        case["tags"] = spec["tags"]
    return report


def suite_result(path, suite):
    report = validate_report(path, suite)
    cases = report["cases"]
    tempo_only = suite == "fsld-tempo-v1"
    tags = sorted({tag for case in cases for tag in case["tags"]})
    timing = {key: dict(mean=statistics.mean(c[key] for c in cases), max=max(c[key] for c in cases),
                        total=sum(c[key] for c in cases))
              for key in ("current_resample_ms", "candidate_resample_ms", "model_and_analysis_ms")}
    return dict(suite_id=suite, report_sha256=digest(path), suite_sha256=report["suite_sha256"],
                baseline_report_sha256=report["baseline_report_sha256"],
                model_manifest_sha256=report["model_manifest_sha256"],
                sources={key: report[key] for key in SOURCE_PATHS},
                baseline_replay_exact=True, baseline_cache_hits=15, candidate_cache_hits=0,
                tempo_only=tempo_only, summary=summarize(cases, tempo_only),
                slices={tag: summarize([c for c in cases if tag in c["tags"]], tempo_only) for tag in tags},
                timings_ms=timing, cases=cases)


def trace_links(paths, suites):
    """Prove the four pre-optimization parity inputs survive the complete rerun."""
    parity = json.loads((DIRECTORY / "reference-resampler-v1-audit.json").read_bytes())
    expected = {c["case_id"]: c for c in parity["cases"]}
    cases = {c["id"]: c for s in suites for c in s["cases"]}
    results = []
    seen = set()
    for path in paths:
        trace = json.loads(path.read_bytes())
        identity = trace["case_id"]
        if identity not in expected or identity in seen:
            raise ValueError("unexpected or duplicate historical trace")
        seen.add(identity)
        old = expected[identity]
        if digest(path) != old["trace_sha256"]:
            raise ValueError("historical trace hash mismatch")
        # All four frozen traces cover the complete short recording, not a crop.
        pcm = trace["mono_samples"]
        current = cases[identity]
        pcm_hash = hashlib.sha256(pcm_bytes(pcm)).hexdigest()
        if (trace["audio_sha256"] != current["audio_sha256"] or trace["sample_rate"] != 22050
            or trace["decoded_sample_count"] != len(pcm) or len(pcm) != current["model_sample_count"]
            or pcm_hash != current["candidate_pcm_sha256"]):
            raise ValueError("full-run PCM differs from the frozen neural-parity input")
        results.append(dict(case_id=identity, historical_trace_sha256=digest(path),
                            candidate_pcm_sha256=pcm_hash, complete_pcm_bitwise_equal=True))
    if seen != expected.keys():
        raise ValueError("all four historical parity traces are required")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artbeat", type=Path, required=True)
    parser.add_argument("--fsld", type=Path, required=True)
    parser.add_argument("--parity-trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace evidence")
    suites = [suite_result(path, suite) for path, suite in zip((args.artbeat, args.fsld), SUITES)]
    report = dict(schema_version=1, purpose="paired_resampler_calibration_not_release_acceptance",
        summarizer_sha256=digest(Path(__file__)), display_delta_epsilon=DISPLAY_EPSILON,
        bounded_identity_report_sha256=digest(DIRECTORY / "resampler-bounded-v1.json"),
        historical_parity_report_sha256=digest(DIRECTORY / "reference-resampler-v1-audit.json"),
        historical_parity_pcm_links=trace_links(args.parity_trace, suites), suites=suites,
        promotion=False, not_checked=["holdout", "platform_parity", "stable_performance_benchmark"],
        timing_note="Sequential VDI wall times include setup. Shipping neural work used cache; do not compare model speed.",
        labels_note="FSLD provides tempo only; no beat/downbeat accuracy is inferred. ARTBeaT has no downbeat labels.")
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for suite in suites:
        print(json.dumps(dict(suite_id=suite["suite_id"], **suite["summary"]), indent=2))


if __name__ == "__main__":
    main()
