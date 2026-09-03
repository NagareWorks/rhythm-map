#!/usr/bin/env python3
"""Explain the frozen calibration regression without tuning a decoder or filter."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from compare_reference import digest, events, validate_trace
from phase_tail_audit import EVENT_MATCH_S, event_delta, logit_delta
from verify_bounded_resampler import pcm_bytes

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parent.parent
RATE = 50
RADIUS = 3
SHIPPING = "beat-this-rten-observations-v2+decode-audio-v2"


def vector(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("expected nonempty finite logit vector")
    return values


def default_events(values):
    """Independent replay of the unchanged strict-zero/radius-three decoder."""
    values = vector(values)
    peaks = [i for i, value in enumerate(values) if value > 0 and
             np.all(values[max(0, i - RADIUS):i + RADIUS + 1] <= value)]
    groups = []
    if peaks:
        mean, count = float(peaks[0]), 1
        for index in peaks[1:]:
            if index - mean <= 1:
                count += 1
                mean += (index - mean) / count
            else:
                groups.append(mean)
                mean, count = float(index), 1
        groups.append(mean)
    return np.asarray([frame / RATE for frame in groups], dtype=np.float32).tolist()


def profile(values, time_s):
    values = vector(values)
    if not math.isfinite(time_s) or time_s < 0:
        raise ValueError("invalid probe time")
    frame = int(math.floor(time_s * RATE + 0.5))
    if frame >= len(values):
        raise ValueError("probe outside model frames")
    left, right = max(0, frame - RADIUS), min(len(values), frame + RADIUS + 1)
    value = float(values[frame])
    peak = left + int(np.argmax(values[left:right]))
    neighbors = [i for i in range(left, right) if i != frame]
    neighbor = max(neighbors, key=lambda i: values[i]) if neighbors else None
    probability = 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))
    return dict(frame=frame, time_s=frame / RATE, logit=value, probability=probability,
                above_threshold=value > 0, local_maximum=value >= float(values[peak]),
                window_peak_frame=peak, window_peak_logit=float(values[peak]),
                strongest_neighbor_frame=neighbor,
                neighbor_margin=None if neighbor is None else value - float(values[neighbor]))


def explain_removed(before, after, time_s):
    old, new = profile(before, time_s), profile(after, time_s)
    if not old["above_threshold"] or not old["local_maximum"]:
        cause = "requires_plateau_or_event_correspondence_inspection"
    elif not new["above_threshold"] and new["local_maximum"]:
        cause = "strict_zero_threshold_crossing"
    elif not new["above_threshold"]:
        cause = "threshold_and_local_peak_competition"
    elif not new["local_maximum"]:
        cause = "local_peak_competition"
    else:
        cause = "requires_deduplication_or_event_correspondence_inspection"
    return dict(event_time_s=time_s, before=old, after=new, cause=cause)


def validate_inputs(lock, calibration, before, after):
    if (lock.get("schema_version") != 1 or lock.get("purpose") != "calibration_resampler_event_loss_diagnosis"
        or lock.get("frame_rate_hz") != RATE or lock.get("decoder_logit_threshold") != 0
        or lock.get("decoder_local_max_radius_frames") != RADIUS
        or lock.get("diagnostic_window_radius_frames") != RADIUS):
        raise ValueError("invalid lock or altered diagnostic/decoder settings")
    suite = next(s for s in calibration["suites"] if s["suite_id"] == lock["suite_id"])
    if suite["summary"]["metrics"]["beat_f1"]["regressed"] != [lock["case_id"]]:
        raise ValueError("case must be the sole reported beat-F1 regression")
    case = next(c for c in suite["cases"] if c["id"] == lock["case_id"])
    for key in ("audio_sha256", "model_sample_count", "current_pcm_sha256", "candidate_pcm_sha256"):
        if case[key] != lock[key]:
            raise ValueError("calibration case identity mismatch: " + key)
    for trace, name in ((before, "current"), (after, "candidate")):
        validate_trace(trace, lock["model_manifest_sha256"])
        for key in ("suite_id", "suite_sha256", "case_id", "audio_sha256"):
            if trace[key] != lock[key]:
                raise ValueError("trace identity mismatch: " + key)
        for key in ("adapter_source_sha256", "audio_preprocessing_sha256"):
            if trace[key] != suite["sources"][key]:
                raise ValueError("production code changed")
        samples = trace["mono_samples"]
        if len(samples) != lock["model_sample_count"] or len(samples) != trace["decoded_sample_count"]:
            raise ValueError("cropped trace or changed duration")
        if hashlib.sha256(pcm_bytes(samples)).hexdigest() != lock[name + "_pcm_sha256"]:
            raise ValueError("PCM differs from the completed calibration")
        if not events(default_events(trace["beat_logits"]), trace["upstream_beats"])["passed"]:
            raise ValueError("independent fixed decoder replay differs from Rust")
        if not events(trace["upstream_beats"], [b["time_s"] for b in trace["observations"]["beats"]])["passed"]:
            raise ValueError("port/adapter event mismatch")
    if before["observation_contract"] != SHIPPING or before.get("preprocessing_candidate") is not None:
        raise ValueError("baseline must be the shipping preprocessor")
    if (after["candidate_source_sha256"] != lock["candidate_source_sha256"]
        or after["observation_contract"] != SHIPPING + "+phase-exact-bh2-256-v1"):
        raise ValueError("candidate identity changed")
    if len(after["upstream_beats"]) != case["raw_beat_count"]:
        raise ValueError("candidate raw count changed from calibration")
    return suite, case


def audit(lock, calibration, before, after, baseline, truth, beat_tolerance_ms):
    suite, case = validate_inputs(lock, calibration, before, after)
    baseline_case = next(c for c in baseline["cases"] if c["id"] == lock["case_id"])
    raw_before = baseline_case["observations"]["raw_beats"]
    for field in ("time_s", "confidence", "downbeat_confidence"):
        if [b[field] for b in raw_before] != [b[field] for b in before["observations"]["beats"]]:
            raise ValueError("fresh shipping trace differs from frozen observations: " + field)
    if baseline_case["end_to_end"] != case["baseline"]:
        raise ValueError("frozen selected scores changed")
    if truth["id"] != lock["case_id"] or not truth["beats"]:
        raise ValueError("expected matching annotated calibration truth")
    delta = event_delta(before["upstream_beats"], after["upstream_beats"])
    probes = []
    for time_s in delta["removed_source_times_s"]:
        probe = explain_removed(before["beat_logits"], after["beat_logits"], time_s)
        nearest = min((b["time_s"] for b in truth["beats"]), key=lambda t: abs(t - time_s))
        probe["nearest_truth"] = dict(time_s=nearest, error_ms=abs(nearest - time_s) * 1000,
                                      within_suite_tolerance=abs(nearest - time_s) * 1000 <= beat_tolerance_ms)
        probes.append(probe)
    return dict(schema_version=1, purpose=lock["purpose"], case_id=lock["case_id"], controls_passed=True,
        controls=dict(frozen_pcm_reproduced=True, frozen_shipping_observations_exact=True,
                      fixed_decoder_replayed=True, port_adapter_events_agree=True,
                      candidate_raw_count_reproduced=True, full_recording=True),
        decoder=dict(logit_threshold=0, local_max_radius_frames=RADIUS, deduplicate_width_frames=1,
                     frame_rate_hz=RATE), diagnostic_event_correspondence_s=EVENT_MATCH_S,
        beat_delta=delta, downbeat_delta=event_delta(before["upstream_downbeats"], after["upstream_downbeats"]),
        logit_difference=logit_delta(before["beat_logits"], after["beat_logits"]), removed_event_probes=probes,
        before_selected_metrics=case["baseline"], after_selected_metrics=case["candidate"],
        sources=suite["sources"], audio_sha256=lock["audio_sha256"], model_manifest_sha256=lock["model_manifest_sha256"],
        not_checked=["official_source_pipeline", "holdout", "safe_recovery_rule"], promotion=False,
        interpretation="A diagnostic peak/threshold classification is not permission to lower a threshold or synthesize beats.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace evidence")
    lock_path = DIRECTORY / "resampler-regression-lock-v1.json"
    lock = json.loads(lock_path.read_bytes())
    calibration_path = DIRECTORY / "reference-resampler-calibration-v1.json"
    if digest(calibration_path) != lock["calibration_report_sha256"]:
        raise ValueError("completed calibration identity changed")
    calibration = json.loads(calibration_path.read_bytes())
    suite = next(s for s in calibration["suites"] if s["suite_id"] == lock["suite_id"])
    if digest(args.baseline) != suite["baseline_report_sha256"]:
        raise ValueError("shipping baseline identity changed")
    suite_path = ROOT / "evaluation/suites" / (lock["suite_id"] + ".json")
    if digest(suite_path) != lock["suite_sha256"]:
        raise ValueError("suite identity changed")
    spec = json.loads(suite_path.read_bytes())
    if spec["purpose"] != "calibration":
        raise ValueError("truth-assisted diagnostics reject holdout and regression")
    case = next(c for c in spec["cases"] if c["id"] == lock["case_id"])
    truth_path = suite_path.parent / case["input"]["truth"]
    expected = next(c for c in suite["cases"] if c["id"] == lock["case_id"])
    if digest(truth_path) != expected["truth_sha256"]:
        raise ValueError("truth identity changed")
    before, after, baseline, truth = (json.loads(path.read_bytes()) for path in
                                     (args.before, args.after, args.baseline, truth_path))
    tolerance = case.get("thresholds", spec["thresholds"])["beat_tolerance_ms"]
    report = audit(lock, calibration, before, after, baseline, truth, tolerance)
    report.update(auditor_sha256=digest(Path(__file__)), lock_sha256=digest(lock_path),
                  calibration_report_sha256=digest(calibration_path), baseline_report_sha256=digest(args.baseline),
                  before_trace_sha256=digest(args.before), after_trace_sha256=digest(args.after),
                  truth_sha256=digest(truth_path))
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(beat_delta=report["beat_delta"], probes=report["removed_event_probes"]), indent=2))


if __name__ == "__main__":
    main()
