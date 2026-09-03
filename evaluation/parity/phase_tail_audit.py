"""Calibration-only input counterfactuals, never a production resampling policy.

The two factors are restoring the exact former pre-origin prefix and truncating
the corrected signal at the former last available source sample. Length changes
are explicit: this is not a claim of shift-invariance under identical padding.
"""
from __future__ import annotations

import numpy as np

EVENT_MATCH_S = 0.040001  # Diagnostic correspondence, NOT an accuracy tolerance.
RECONSTRUCTION_ATOL = 1e-5
LOGIT_ATOL = 2e-3


def validate_audit_lock(lock, reference_digest, traces):
    if (lock.get("schema_version") != 1 or
            lock.get("purpose") != "calibration_phase_tail_diagnosis" or
            lock.get("reference_lock_sha256") != reference_digest):
        raise ValueError("invalid phase/tail lock or reference identity")
    cases = {case["case_id"]: case for case in lock["cases"]}
    ids = [trace["case_id"] for trace in traces]
    if len(cases) != len(lock["cases"]) or len(set(ids)) != len(ids) or set(ids) != set(cases):
        raise ValueError("phase/tail traces must match all locked cases exactly once")
    for trace in traces:
        case = cases[trace["case_id"]]
        for field in ("suite_id", "suite_sha256", "audio_sha256", "prefix_seconds"):
            if trace.get(field) != case[field]:
                raise ValueError("phase/tail trace identity mismatch: " + field)
        if (trace.get("purpose") != "calibration_parity_private" or
                trace.get("observation_contract") != "beat-this-rten-observations-v2+decode-audio-v2"):
            raise ValueError("phase/tail diagnosis requires a v2 calibration trace")
        make_variants(trace, case["legacy_origin_samples"])
    return cases


def make_variants(trace, delay):
    legacy = trace.get("legacy_audio")
    if (not legacy or legacy.get("implementation") != "beat-this-1.0.0" or
            legacy.get("sample_rate") != 22050 or trace.get("sample_rate") != 22050):
        raise ValueError("exact legacy PCM export at model rate is required")
    current = np.asarray(trace["mono_samples"], dtype=np.float32)
    old = np.asarray(legacy["mono_samples"], dtype=np.float32)
    if (current.ndim != 1 or old.ndim != 1 or
            not np.isfinite(current).all() or not np.isfinite(old).all()):
        raise ValueError("expected finite mono PCM")
    if (len(current) != trace.get("decoded_sample_count") or
            len(old) != legacy.get("decoded_sample_count")):
        raise ValueError("tail diagnosis rejects cropped traces")
    if not isinstance(delay, int) or delay <= 0 or not delay < len(old) <= len(current):
        raise ValueError("invalid predeclared legacy origin or sample lengths")
    keep = len(old) - delay
    # Copy instead of mutating/reusing the trace as writable experiment state.
    prefix = old[:delay]
    return {
        "v2": (current.copy(), 0),
        "tail_trimmed_only": (current[:keep].copy(), 0),
        "origin_restored_only": (np.concatenate([prefix, current]), delay),
        "origin_and_tail_restored": (np.concatenate([prefix, current[:keep]]), delay),
        "actual_v1": (old.copy(), delay),
    }


def event_delta(left, right):
    """One-to-one chronological matching; no truth or beat-grid synthesis."""
    left, right = np.asarray(left), np.asarray(right)
    if any(x.ndim != 1 or not np.isfinite(x).all() or np.any(np.diff(x) < 0) for x in (left, right)):
        raise ValueError("expected finite sorted event vectors")
    i = j = 0
    removed, added, offsets = [], [], []
    while i < len(left) and j < len(right):
        delta = float(right[j] - left[i])
        if abs(delta) <= EVENT_MATCH_S:
            offsets.append(abs(delta))
            i, j = i + 1, j + 1
        elif delta > 0:
            removed.append(float(left[i]))
            i += 1
        else:
            added.append(float(right[j]))
            j += 1
    removed.extend(map(float, left[i:]))
    added.extend(map(float, right[j:]))
    return {"left_count": len(left), "right_count": len(right), "matched": len(offsets),
            "removed_source_times_s": removed, "added_source_times_s": added,
            "max_matched_offset_s": max(offsets, default=0.0)}


def logit_delta(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("invalid logits")
    count = min(len(left), len(right))
    if not count:
        raise ValueError("empty logits")
    diff = left[:count] - right[:count]
    return {"left_frames": len(left), "right_frames": len(right), "common_frames": count,
            "unshifted_max_abs": float(np.abs(diff).max()),
            "unshifted_rmse": float(np.sqrt(np.mean(diff ** 2)))}


def run_audit(trace, case, predict_pcm, decode, current_logits):
    """Callbacks execute the already-verified official model, frontend and decoder."""
    delay = case["legacy_origin_samples"]
    variants = make_variants(trace, delay)
    old, reconstructed = variants["actual_v1"][0], variants["origin_and_tail_restored"][0]
    waveform_error = old.astype(np.float64) - reconstructed.astype(np.float64)
    reconstruction = {"max_abs": float(np.abs(waveform_error).max()),
                      "rmse": float(np.sqrt(np.mean(waveform_error ** 2))),
                      "waveform_passed": bool(np.allclose(old, reconstructed, atol=RECONSTRUCTION_ATOL, rtol=0))}
    runs, summaries = {}, {}
    for name, (pcm, shift) in variants.items():
        logits = current_logits if name == "v2" else predict_pcm(pcm)
        raw_events = decode(**logits)
        offset = shift / 22050
        source_events = [np.asarray(values, dtype=np.float64) - offset for values in raw_events]
        # Retain negative pre-origin detections explicitly, not clipped to time zero.
        runs[name] = {"logits": logits, "events": source_events}
        probes = []
        beat_logits = np.asarray(logits["beat"])
        for time in case.get("probe_times_s", []):
            frame = int(round((time + offset) * 50))
            first, last = max(0, frame - 2), min(len(beat_logits), frame + 3)
            if first >= last:
                continue
            peak = first + int(np.argmax(beat_logits[first:last]))
            probability = float(1 / (1 + np.exp(-np.clip(float(beat_logits[peak]), -80, 80))))
            probes.append({"source_time_s": time, "nearby_peak_source_time_s": peak / 50 - offset,
                           "nearby_peak_probability": probability,
                           "selected_within_match_window": bool(np.any(np.abs(source_events[0] - time) <= EVENT_MATCH_S))})
        summaries[name] = {"sample_count": len(pcm), "input_origin_shift_samples": shift,
                           "beat_count": len(raw_events[0]), "downbeat_count": len(raw_events[1]),
                           "pre_origin_beat_count": int(np.sum(source_events[0] < 0)), "probes": probes}

    pairs = {
        "tail_effect_at_v2_origin": ("v2", "tail_trimmed_only"),
        "tail_effect_at_legacy_origin": ("origin_restored_only", "origin_and_tail_restored"),
        "origin_effect_with_full_tail": ("v2", "origin_restored_only"),
        "origin_effect_with_trimmed_tail": ("tail_trimmed_only", "origin_and_tail_restored"),
        "actual_v1_to_v2": ("actual_v1", "v2"),
        "reconstructed_vs_actual_v1": ("actual_v1", "origin_and_tail_restored"),
    }
    effects = {}
    for label, (left, right) in pairs.items():
        effect = {}
        for index, key in enumerate(("beat", "downbeat")):
            effect[key + "_events"] = event_delta(runs[left]["events"][index], runs[right]["events"][index])
            effect[key + "_logits"] = logit_delta(runs[left]["logits"][key], runs[right]["logits"][key])
        effects[label] = effect
    same = effects["reconstructed_vs_actual_v1"]
    reconstruction["logits_passed"] = all(
        same[key + "_logits"]["left_frames"] == same[key + "_logits"]["right_frames"] and
        same[key + "_logits"]["unshifted_max_abs"] <= LOGIT_ATOL for key in ("beat", "downbeat"))
    return {"purpose": "counterfactual_input_diagnosis_not_accuracy_or_policy_selection",
            "event_time_normalization": "subtract input prefix delay for diagnostic comparison only; no product timestamps modified",
            "event_match_window_s": EVENT_MATCH_S, "waveform_atol": RECONSTRUCTION_ATOL,
            "reconstruction_logit_atol": LOGIT_ATOL,
            "trimmed_tail_samples": len(variants["v2"][0]) - len(variants["tail_trimmed_only"][0]),
            "reconstruction": reconstruction, "variants": summaries, "effects": effects}
