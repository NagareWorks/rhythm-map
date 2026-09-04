"""Compare Rust candidates to immutable private Python calibration experiments.

Does not generate candidates, select a strategy or inspect holdout. Raw inputs and
timestamp reports remain outside Git; only aggregate output is publication-safe.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


LOCKS = {
    "rubato": {
        "count": 25,
        "frozen": "49324840abfb1b8ecbe824bc9deab16aa0f88992f7a79d6639bb16b6c06d138e",
        "evidence": "ce5e678276888a0e430c004444dce4b27f0cfac0761767736abee2ec3fc05937",
        "baseline": "2ecb565b59e05d8e637f0c7275c6089c69ecc302ff31dfe5848b1fedeb4350f8",
    },
    "artbeat": {
        "count": 15,
        "frozen": "886fd8c3bec7e9834c0b0656c3692dca80daf690d331328c5d0dc38eedffa8f2",
        "evidence": "3f1ba43fd4f373579727a48668d8de8e00166523d2d1141e072bc3471a71ab3e",
        "baseline": "44d17c8dba13b4494d869914a8830840bccf32000adc0a08af0a4fcd07108e43",
    },
}
ESTIMATOR = "3d2bc3ca875025b5d08e511dcecf38351fc8f62e27daf8d49147f9f8a68bf8f1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_by_id(rows):
    result = {row["id"]: row for row in rows}
    require(len(result) == len(rows), "duplicate case id")
    return result


def metrics(predicted, truth, tolerance):
    """Same chronological event matching as the evaluation contract."""
    i = j = tp = 0
    while i < len(predicted) and j < len(truth):
        delta = predicted[i] - truth[j]
        if abs(delta) <= tolerance:
            tp += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    fp, fn = len(predicted) - tp, len(truth) - tp
    return dict(tp=tp, fp=fp, fn=fn, f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0)


def assert_metrics(actual, expected):
    require(all(actual[k] == expected[k] for k in ("tp", "fp", "fn")), "event metric mismatch")
    require(abs(actual["f1"] - expected["f1"]) <= 1e-12, "F1 mismatch")


def compare_case(row, frozen, evidence, history):
    generated = row["generated"]
    require(row["primary_replay_exact"] is True, "primary replay was not exact")
    require(generated["generator"] == "active-interval-path-v1", "unknown generator")
    for key in ("silence_regions", "unknown_gaps"):
        require(generated[key] == frozen[key], key + " mismatch")
    primary = next(h["beat_times_s"] for h in history["pulse_hypothesis_coverage"]["hypotheses"] if h["id"] == "selected")
    forced = primary[:]
    proposals = generated["proposals"]
    require(len(proposals) == len(frozen["proposals"]), "component count mismatch")
    candidate_times = {c["time_s"] for c in evidence["observations"]["beat_candidates"]}
    for actual, expected in zip(proposals, frozen["proposals"]):
        for key in ("start_s", "end_s", "status"):
            require(actual[key] == expected[key], key + " mismatch")
        a, b = actual["start_s"], actual["end_s"]
        original = [t for t in primary if a <= t <= b]
        require(actual["original_times_s"] == original, "primary component mismatch")
        if "candidate_count" in expected:
            require(actual["candidate_count"] == expected["candidate_count"], "candidate count mismatch")
        if actual["status"] == "proposal":
            path = actual["proposal_times_s"]
            require(path == expected["proposal_times_s"], "proposal timestamps mismatch")
            require(len(path) >= 8 and all(t in candidate_times and a <= t <= b for t in path), "unsupported path")
            require(all(60 / 320 <= y - x <= 60 / 40 for x, y in zip(path, path[1:])), "invalid path interval")
            require(not any(lo <= t <= hi for t in path for lo, hi in generated["silence_regions"]), "path in silence")
            forced = sorted([t for t in forced if not a <= t <= b] + path)
            # Counts account for every edit, never for shared anchors. These
            # timestamps are frozen exact model events, so exact sets suffice.
            diffs = actual["disagreements"]
            require(sum(d["primary_only_beat_count"] for d in diffs) == len(set(original) - set(path)), "primary edit count")
            require(sum(d["alternative_only_beat_count"] for d in diffs) == len(set(path) - set(original)), "alternative edit count")
        else:
            require(actual["status"] == "fallback_no_valid_path" and actual["proposal_times_s"] is None and not actual["disagreements"], "invalid fallback")
    require(forced == sorted(set(forced)), "stitched path not unique/ordered")
    truth, tol = evidence["truth_times_s"], evidence["beat_tolerance_s"]
    assert_metrics(metrics(primary, truth, tol), frozen["selected"])
    assert_metrics(metrics(forced, truth, tol), frozen["forced_active_paths"])
    if "forced_times_s" in frozen:
        require(forced == frozen["forced_times_s"], "forced timestamps mismatch")


def verify(cohort, evidence_path, baseline_path, frozen_path, result_path, source_path):
    lock = LOCKS[cohort]
    for key, path in (("evidence", evidence_path), ("baseline", baseline_path), ("frozen", frozen_path)):
        require(digest(path) == lock[key], key + " identity mismatch")
    evidence, history, frozen, result = [json.loads(p.read_text(encoding="utf-8")) for p in (evidence_path, baseline_path, frozen_path, result_path)]
    require(result["purpose"] == "private_calibration_active_region_rust_replay", "result purpose")
    require(result["evidence_sha256"] == lock["evidence"] and result["baseline_sha256"] == lock["baseline"], "result provenance")
    require(result["generator_source_sha256"] == digest(source_path), "generator source identity")
    require(result["estimator_source_sha256"] == ESTIMATOR, "default estimator changed")
    require(result["inference_run"] is False and result["adoption_enabled"] is False and result["extra_probe_excluded"] is True, "diagnostic boundary")
    maps = [rows_by_id(doc["cases"]) for doc in (evidence, history, frozen, result)]
    require(all(len(m) == lock["count"] and m.keys() == maps[0].keys() for m in maps), "case set mismatch (probe/holdout?)")
    for case_id in maps[0]:
        compare_case(maps[3][case_id], maps[2][case_id], maps[0][case_id], maps[1][case_id])
    rows = result["cases"]
    parts = [p for r in rows for p in r["generated"]["proposals"]]
    times = [r["generation_elapsed_s"] for r in rows]
    require(all(math.isfinite(t) and t >= 0 for t in times), "invalid runtime")
    return dict(cohort=cohort, exact_cases=len(rows), components=len(parts),
        proposals=sum(p["status"] == "proposal" for p in parts),
        fallback_components=sum(p["status"] != "proposal" for p in parts),
        generation_total_s=sum(times), generation_max_case_s=max(times),
        peak_dp_capacity_bytes=max(r["generated"]["work"]["peak_dp_capacity_bytes"] for r in rows),
        pair_states=sum(r["generated"]["work"]["pair_states"] for r in rows),
        transitions=sum(r["generated"]["work"]["transitions"] for r in rows),
        generator_source_sha256=result["generator_source_sha256"],
        frozen_sha256=lock["frozen"], result_sha256=digest(result_path),
        automatic_adoption=False, accuracy_improvement_claimed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=LOCKS, required=True)
    for name in ("evidence", "baseline", "frozen", "result"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2] / "crates/rhythm-map-eval/src/active_regions.rs")
    args = parser.parse_args()
    print(json.dumps(verify(args.cohort, args.evidence, args.baseline, args.frozen, args.result, args.source), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
