"""Truth-assisted dense pulse-template audit, never an automatic decoder.

Freeze the comparison before inspecting calibration results: beat-head local
maximum at each annotated beat versus its following half-beat location, using
radius min(50 ms, truth interval / 16) at both positions. No selected-event phase
anchor, fitted threshold, direction reversal, or invented frame is permitted.
The last annotated beat has no following interval and is explicitly excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from clock_phase_evidence import INPUTS, matches, near, ordered, require
from resampler_event_audit import RATE, SHIPPING, default_events, vector

ROOT = Path(__file__).resolve().parents[2]
MODEL = "ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d"
SUITES = {
    "artbeat": ("artbeat-v1", "21f3d44bacbfe9c50dfbc889990c563d44e406d56558492627402d21e5a7e81b"),
    "rubato": ("rubato-calibration-v1", "c10c229bbf7b89ebd23dd2b4ff2a2d19aaec9b5f28d2b5eb6d121d950fb62653"),
}
SOURCES = {
    "exporter_source_sha256": "crates/rhythm-map-eval/examples/dense_beat_evidence.rs",
    "adapter_source_sha256": "crates/rhythm-map-beat-this/src/lib.rs",
    "audio_source_sha256": "crates/rhythm-map-beat-this/src/audio.rs",
    "cargo_lock_sha256": "Cargo.lock",
}
STRATA = ("all", "raw_matched", "raw_missed", "raw_missed_with_candidate", "raw_missed_without_candidate")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path, expected=None):
    data = Path(path).read_bytes()
    identity = sha(data)
    require(expected is None or expected == identity, "input bytes changed")
    return json.loads(data), identity


def validate_summary(summary, cases, count):
    require(summary.get("schema_version") == 1 and
            summary.get("purpose") == "full_recording_dense_capture_summary", "invalid capture summary")
    require(summary.get("complete") is True and len(cases) == count and
            summary.get("expected_case_count") == count and
            summary.get("completed_inference_count") == count, "incomplete cohort")
    require(summary.get("cache_writes") == 0 and all(summary.get(k) is False for k in
            ("training_run", "production_observations_changed", "accuracy_improvement_claimed")),
            "invalid capture scope")
    records = summary.get("cases", [])
    ids = [c["id"] for c in cases]
    require(len(set(ids)) == count and [r["case_id"] for r in records] == ids, "case order/coverage changed")
    require(all(i and all(c.isascii() and (c.isalnum() or c in "-_") for c in i) for i in ids),
            "unsafe case identity")
    return records


def validate_capture(payload, record, case, cohort, source_hashes):
    require(all(payload.get(k) == v for k, v in record.items() if k != "capture_sha256"),
            "capture metadata differs from summary")
    require(payload.get("schema_version") == 1 and
            payload.get("purpose") == "private_full_recording_dense_evidence", "invalid capture schema")
    suite_id, suite_hash = SUITES[cohort]
    require(payload.get("case_id") == case["id"] and payload.get("suite_id") == suite_id and
            payload.get("suite_sha256") == suite_hash and
            payload.get("frozen_evidence_sha256") == INPUTS[cohort][1], "capture identity mismatch")
    require(payload.get("model_manifest_sha256") == MODEL and
            payload.get("observation_contract") == SHIPPING, "model/contract changed")
    require(all(payload.get(k) == v for k, v in source_hashes.items()), "capture implementation changed")
    require(all(payload.get(k) == case[k] for k in
                ("audio_sha256", "pcm_sha256", "sample_count", "sample_rate")), "full PCM identity changed")
    require(case.get("score_replay_exact") is True and payload.get("replay", {}).get("exact") is True,
            "producer replay failed")
    require(payload.get("frame_rate_hz") == RATE and payload.get("start_time_s") == 0,
            "frame timing changed")
    beats, downbeats = vector(payload["beat_logits"]), vector(payload["downbeat_logits"])
    require(len(beats) == len(downbeats) == payload.get("frame_count"), "head length mismatch")
    duration = case["sample_count"] / case["sample_rate"]
    # Native 22,050 Hz cohorts: retain the centered frontend's final frame, with
    # no prefix crop, padding, or extrapolation to the audio endpoint.
    require(case["sample_rate"] == 22050 and case["sample_count"] > 0 and
            0 <= duration - (len(beats) - 1) / RATE < 1 / RATE + 1e-9,
            "dense timeline does not cover the complete recording")
    expected, actual = case["observations"], payload["observations"]
    require(all(actual.get(k) == expected[k] for k in ("beats", "beat_candidates", "source", "duration_s")),
            "independent raw observation comparison failed")
    require(actual["duration_s"] == duration and actual.get("activations") is None and
            all(actual.get(k) == [] for k in ("activity", "onsets", "harmonic_changes")), "not raw-only capture")
    require(default_events(beats) == [b["time_s"] for b in expected["beats"]],
            "independent default event reconstruction failed")
    return beats


def local_peak(values, center, radius):
    """Only actual frames in a fully covered window; unavailable is not zero."""
    require(math.isfinite(center) and math.isfinite(radius) and radius > 0, "invalid template window")
    if center - radius < 0 or center + radius > (len(values) - 1) / RATE:
        return None
    left, right = math.ceil((center - radius) * RATE), math.floor((center + radius) * RATE)
    return max((float(v) for v in values[left:right + 1]), default=None)


def template_rows(values, case):
    """Annotations choose ideal templates, not a truth-free proposed beat path."""
    truth = case["truth_times_s"]
    require(len(truth) >= 2 and ordered(truth), "invalid truth sequence")
    obs, tolerance = case["observations"], case["beat_tolerance_s"]
    pairs = matches([b["time_s"] for b in obs["beats"]], truth, tolerance)
    require(pairs == case["raw_truth_pairs"], "raw/truth identity replay failed")
    matched = {t for _, t in pairs}
    candidates = [p["time_s"] for p in obs["beat_candidates"]]
    require(ordered(candidates), "invalid candidate ordering")
    rows = []
    for index, (start, end) in enumerate(zip(truth, truth[1:])):
        radius = min(.05, (end - start) / 16)
        canonical = local_peak(values, start, radius)
        control = local_peak(values, (start + end) / 2, radius)
        strata = ["all", "raw_matched" if index in matched else "raw_missed"]
        if index not in matched:
            strata.append("raw_missed_with_candidate" if near(candidates, start, tolerance)
                          else "raw_missed_without_candidate")
        rows.append(dict(canonical=canonical, half_phase=control, strata=strata))
    return rows, dict(truth_beats=len(truth), interval_queries=len(rows),
                     excluded_final_truth_beats=1, raw_matched_truth_beats=len(matched),
                     raw_missed_truth_beats=len(truth) - len(matched),
                     excluded_final_truth_raw_matched=len(truth) - 1 in matched)


def summarize(rows):
    complete = [r for r in rows if r["canonical"] is not None and r["half_phase"] is not None]
    margins = [r["canonical"] - r["half_phase"] for r in complete]
    n = len(complete)
    return dict(queries=len(rows), paired_queries=n, unavailable_queries=len(rows) - n,
                canonical_wins=sum(m > 0 for m in margins), ties=sum(m == 0 for m in margins),
                half_phase_wins=sum(m < 0 for m in margins),
                canonical_win_fraction=sum(m > 0 for m in margins) / n if n else None,
                mean_logit_margin=statistics.mean(margins) if n else None,
                median_logit_margin=statistics.median(margins) if n else None,
                canonical_above_zero=sum(r["canonical"] > 0 for r in complete),
                half_phase_above_zero=sum(r["half_phase"] > 0 for r in complete))


def stratified(rows):
    return {key: summarize([r for r in rows if key in r["strata"]]) for key in STRATA}


def audit(cohort, evidence_path, capture_dir):
    count, evidence_hash = INPUTS[cohort]
    evidence, _ = read_json(evidence_path, evidence_hash)
    summary, summary_hash = read_json(Path(capture_dir) / "summary.json")
    records = validate_summary(summary, evidence["cases"], count)
    source_hashes = {k: sha((ROOT / v).read_bytes()) for k, v in SOURCES.items()}
    all_rows, tracks = [], []
    for record, case in zip(records, evidence["cases"]):
        payload, _ = read_json(Path(capture_dir) / f'{case["id"]}.json', record["capture_sha256"])
        values = validate_capture(payload, record, case, cohort, source_hashes)
        rows, coverage = template_rows(values, case)
        all_rows.extend(rows)
        tracks.append(dict(id=case["id"], capture_sha256=record["capture_sha256"],
                           frame_count=len(values), independent_default_events_exact=True,
                           coverage=coverage, strata=stratified(rows)))
    macro = {}
    for key in STRATA:
        available = [t["strata"][key]["canonical_win_fraction"] for t in tracks
                     if t["strata"][key]["canonical_win_fraction"] is not None]
        macro[key] = dict(tracks_with_paired_queries=len(available), total_tracks=count,
                          mean_track_canonical_win_fraction=statistics.mean(available) if available else None)
    return dict(cohort=cohort, frozen_evidence_sha256=evidence_hash, capture_summary_sha256=summary_hash,
                source_hashes=source_hashes, complete=True, tracks=count,
                total_frames_per_head=sum(t["frame_count"] for t in tracks),
                total_inference_elapsed_s=sum(r["inference_elapsed_s"] for r in records),
                pooled=stratified(all_rows), macro=macro, cases=tracks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for cohort in INPUTS:
        parser.add_argument(f"--{cohort}-evidence", type=Path, required=True)
        parser.add_argument(f"--{cohort}-captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohorts = [audit(name, getattr(args, f"{name}_evidence"), getattr(args, f"{name}_captures")) for name in INPUTS]
    report = dict(schema_version=1, purpose="calibration_truth_assisted_dense_pulse_template_audit",
                  script_sha256=sha(Path(__file__).read_bytes()),
                  helper_source_sha256={name: sha(Path(__file__).with_name(name).read_bytes()) for name in
                                        ("resampler_event_audit.py", "clock_phase_evidence.py")},
                  truth_assisted=True, automatic_decoder=False, accuracy_improvement_claimed=False,
                  training_run=False, production_observations_changed=False, holdout_opened=False,
                  frame_rate_hz=RATE, heads_retained=["beat", "downbeat"], template_head="beat",
                  radius="min(0.05 seconds, truth interval / 16)",
                  control="following half-beat location; not an annotated nonmusical-event class",
                  favorable_direction="canonical logit greater than half-phase logit",
                  cohorts=cohorts)
    # All validation completes before a result is written; never overwrite an
    # earlier audit or publish a partial cohort after a late replay failure.
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(report, target, indent=2, allow_nan=False)
        target.write("\n")


if __name__ == "__main__":
    main()
