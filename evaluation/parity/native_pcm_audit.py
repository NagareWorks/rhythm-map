#!/usr/bin/env python3
"""Locked calibration 2x2 decoder/resampler diagnosis; no truth or policy tuning.

Exports private native PCM through the actual shipping Rust preprocessing.
Writes only digests, numerical summaries, and fixed-probe/event deltas publicly.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from compare_reference import (REFERENCE_LOCK, REFERENCE_REVISION, LOGIT_ATOL,
                               SOURCE_EVENT_ATOL, compare, digest, events,
                               validate_trace, waveform_diagnostic)
from phase_tail_audit import EVENT_MATCH_S, event_delta, logit_delta

TARGET_CASE = "artbeat-15-85-to-127-5"
CONTRACT = "beat-this-rten-observations-v2+decode-audio-v2"


def mono(signal):
    signal = np.asarray(signal)
    if signal.ndim == 2 and signal.shape[1] > 0:
        signal = signal.mean(1)
    if signal.ndim != 1 or not signal.size or not np.isfinite(signal).all():
        raise ValueError("expected nonempty finite mono or frame-major multichannel PCM")
    return signal


def validate_identity(trace, case):
    validate_trace(trace, REFERENCE_LOCK["model_manifest_sha256"])
    if trace.get("observation_contract") != CONTRACT:
        raise ValueError("native diagnosis requires current v2 preprocessing")
    for key in ("case_id", "suite_id", "suite_sha256", "audio_sha256", "prefix_seconds"):
        if trace.get(key) != case[key]:
            raise ValueError("locked calibration identity mismatch: " + key)
    samples = mono(trace["mono_samples"])
    if len(samples) != trace.get("decoded_sample_count"):
        raise ValueError("native diagnosis rejects cropped traces")


def pcm_summary(left, right):
    left, right = mono(left), mono(right)
    result = waveform_diagnostic(left, right)
    if left.shape == right.shape:
        diff = left.astype(np.float64) - right.astype(np.float64)
        result.update(full_max_abs=float(np.abs(diff).max()),
                      full_rmse=float(np.sqrt(np.mean(diff ** 2))),
                      float32_bit_exact=bool(np.array_equal(
                          left.astype(np.float32).view(np.uint32),
                          right.astype(np.float32).view(np.uint32))))
    return result


def make_matrix(rust_native, reference_native, rust_on_rust, rust_on_reference, resample):
    """Both decoders supply f32 PCM; the soxr side receives its f64 promotion.

    Rust resampler results are provided by the diagnostic executable. This code
    never substitutes a Python imitation for the actual shipping resampler.
    """
    sources = {"rust": mono(rust_native).astype(np.float32),
               "reference": mono(reference_native).astype(np.float32)}
    return {
        "rust_decode_rust_resample": mono(rust_on_rust).astype(np.float32).copy(),
        "reference_decode_rust_resample": mono(rust_on_reference).astype(np.float32).copy(),
        "rust_decode_soxr_resample": mono(resample(sources["rust"].astype(np.float64))).astype(np.float32),
        "reference_decode_soxr_resample": mono(resample(sources["reference"].astype(np.float64))).astype(np.float32),
    }


def probe_summary(logits, beat_events, times):
    values = np.asarray(logits, dtype=np.float64)
    result = []
    for time in times:
        frame = int(round(time * 50))
        first, last = max(0, frame - 2), min(len(values), frame + 3)
        if first >= last:
            raise ValueError("fixed probe is outside the clip")
        peak = first + int(np.argmax(values[first:last]))
        result.append({"time_s": time, "nearby_peak_time_s": peak / 50,
                       "nearby_peak_probability": float(1 / (1 + np.exp(-np.clip(values[peak], -80, 80)))),
                       "selected": bool(np.any(np.abs(np.asarray(beat_events) - time) <= EVENT_MATCH_S))})
    return result


def summarize_runs(pcms, predict, decode, probe_times):
    runs, summaries = {}, {}
    for name, pcm in pcms.items():
        print("Official model: " + name, flush=True)
        logits = predict(pcm)
        decoded = decode(**logits)
        runs[name] = {"logits": logits, "events": decoded}
        summaries[name] = {"sample_count": len(pcm), "beat_count": len(decoded[0]),
                           "downbeat_count": len(decoded[1]),
                           "probes": probe_summary(logits["beat"], decoded[0], probe_times)}
    pairs = {
        "decode_effect_rust_resampler": ("reference_decode_rust_resample", "rust_decode_rust_resample"),
        "decode_effect_soxr_resampler": ("reference_decode_soxr_resample", "rust_decode_soxr_resample"),
        "resampler_effect_rust_decode": ("rust_decode_soxr_resample", "rust_decode_rust_resample"),
        "resampler_effect_reference_decode": ("reference_decode_soxr_resample", "reference_decode_rust_resample"),
        "native_f32_normalization_control": ("official_original_file", "reference_decode_soxr_resample"),
        "original_file_to_shipping": ("official_original_file", "rust_decode_rust_resample"),
    }
    effects = {}
    for label, (left, right) in pairs.items():
        effects[label] = {"left": left, "right": right, "waveform": pcm_summary(pcms[left], pcms[right])}
        for i, key in enumerate(("beat", "downbeat")):
            effects[label][key + "_events"] = event_delta(runs[left]["events"][i], runs[right]["events"][i])
            effects[label][key + "_event_parity"] = events(runs[left]["events"][i], runs[right]["events"][i], SOURCE_EVENT_ATOL)
            effects[label][key + "_logits"] = logit_delta(runs[left]["logits"][key], runs[right]["logits"][key])
    return runs, summaries, effects


def write_new(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, allow_nan=False, indent=2)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "checkpoint", "trace", "suite", "source-audio", "rust-exporter", "private-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.private_dir.resolve().is_relative_to(root) or args.output.exists():
        parser.error("private PCM directory must be outside the repository; report must be new")
    if args.private_dir.exists():
        parser.error("private PCM directory must be new")
    lock_path = Path(__file__).with_name("regression-lock-v2.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["reference_lock_sha256"] != digest(Path(__file__).with_name("reference-lock.json")):
        parser.error("reference identity differs from the regression lock")
    case = next(c for c in lock["cases"] if c["case_id"] == TARGET_CASE)
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    validate_identity(trace, case)
    # Freeze the previous failed case/trace rather than replacing its evidence.
    spot = json.loads(Path(__file__).with_name("source-threshold-probes-v2.json").read_text(encoding="utf-8"))
    if digest(args.trace) != spot["trace_sha256"]:
        parser.error("expected the frozen source-mismatch trace")
    if digest(args.source_audio) != case["audio_sha256"] or digest(args.suite) != case["suite_sha256"]:
        parser.error("source audio or calibration suite differs from lock")
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True).strip()
    if revision != REFERENCE_REVISION or dirty:
        parser.error("reference checkout must be clean at the pinned revision")
    if (digest(args.checkpoint) != REFERENCE_LOCK["checkpoint"]["sha256"] or
            args.checkpoint.stat().st_size != REFERENCE_LOCK["checkpoint"]["size_bytes"]):
        parser.error("checkpoint differs from lock; no downloads performed")
    sys.path.insert(0, str(args.upstream.resolve()))
    import torch
    import torchaudio
    import soxr
    from beat_this.preprocessing import load_audio, LogMelSpect
    from beat_this.inference import load_model, split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source, rate = load_audio(str(args.source_audio))
    native64 = mono(source)
    if rate != case["source_sample_rate"] or len(native64) > rate * 60:
        parser.error("unexpected native rate or clip length")
    native32 = native64.astype(np.float32)
    args.private_dir.mkdir(parents=True)
    reference_path, rust_path = args.private_dir / "reference-native.json", args.private_dir / "rust-pcm.json"
    write_new(reference_path, {"schema_version": 1, "purpose": "calibration_native_pcm_private",
                              "case_id": TARGET_CASE, "suite_sha256": case["suite_sha256"],
                              "audio_sha256": case["audio_sha256"], "sample_rate": rate,
                              "mono_samples": native32.tolist()})
    print("Exporting actual Rust native decode and two controlled resampling paths", flush=True)
    subprocess.run([str(args.rust_exporter.resolve()), "--suite", str(args.suite.resolve()),
                    "--case", TARGET_CASE, "--audio-dir", str(args.source_audio.resolve().parent),
                    "--reference-native", str(reference_path.resolve()), "--output", str(rust_path.resolve())], check=True)
    rust = json.loads(rust_path.read_text(encoding="utf-8"))
    for key in ("case_id", "suite_sha256", "audio_sha256"):
        if rust[key] != case[key]:
            raise ValueError("Rust native export identity mismatch")
    if (rust["reference_native_sha256"] != digest(reference_path) or rust["observation_contract"] != CONTRACT or
            rust["native_sample_rate"] != rate or rust["model_sample_rate"] != 22050 or
            rust["shipping_reconstruction_bit_exact"] is not True):
        raise ValueError("Rust native export provenance/rate/reconstruction mismatch")
    sources = {"exporter_sha256": "crates/rhythm-map-eval/examples/beat_this_pcm.rs",
               "support_sha256": "crates/rhythm-map-eval/examples/support/mod.rs",
               "adapter_source_sha256": "crates/rhythm-map-beat-this/src/lib.rs",
               "audio_preprocessing_sha256": "crates/rhythm-map-beat-this/src/audio.rs"}
    if any(rust[key] != digest(root / path) for key, path in sources.items()):
        raise ValueError("Rust exporter is stale relative to its source files")
    pcms = make_matrix(rust["rust_native_mono"], native32, rust["rust_native_rust_resampled"],
                       rust["reference_native_rust_resampled"],
                       lambda pcm: soxr.resample(pcm, rate, 22050))
    pcms["official_original_file"] = soxr.resample(native64, rate, 22050).astype(np.float32)
    historical = np.asarray(trace["mono_samples"], dtype=np.float32)
    shipping = pcms["rust_decode_rust_resample"]
    bit_exact = np.array_equal(shipping.view(np.uint32), historical.view(np.uint32))
    if not bit_exact:
        raise ValueError("native-stage refactor changed the frozen shipping PCM")

    model, frontend, decode = load_model(str(args.checkpoint), "cpu"), LogMelSpect(device="cpu"), Postprocessor(type="minimal")
    def predict(pcm):
        with torch.inference_mode():
            mel = frontend(torch.from_numpy(pcm))
            return split_predict_aggregate(mel, 1500, 6, "keep_first", model)
    runs, summaries, effects = summarize_runs(pcms, predict, decode, case["probe_times_s"])
    controls = {"native_stages_reconstruct_shipping_bit_exact": True,
                "shipping_equals_frozen_v2_trace_bit_exact": bool(bit_exact)}
    for i, key in enumerate(("beat", "downbeat")):
        current = runs["rust_decode_rust_resample"]
        controls[key + "_same_pcm_logit_parity"] = compare(np.asarray(current["logits"][key]), trace[key + "_logits"], LOGIT_ATOL)["passed"]
        controls[key + "_same_pcm_event_parity"] = events(current["events"][i], trace["upstream_" + key + "s"])["passed"]
        controls[key + "_f32_normalization_event_parity"] = effects["native_f32_normalization_control"][key + "_event_parity"]["passed"]
    report = {"schema_version": 1, "purpose": "native_pcm_factor_diagnosis_not_accuracy",
              "case_id": TARGET_CASE, "suite_sha256": case["suite_sha256"], "audio_sha256": case["audio_sha256"],
              "reference_revision": revision, "checkpoint_sha256": digest(args.checkpoint),
              "reference_lock_sha256": digest(Path(__file__).with_name("reference-lock.json")),
              "regression_lock_sha256": digest(lock_path), "historical_trace_sha256": digest(args.trace),
              "model_manifest_sha256": trace["model_manifest_sha256"], "observation_contract": CONTRACT,
              "auditor_sha256": digest(Path(__file__)), "exporter_executable_sha256": digest(args.rust_exporter),
              "comparison_helper_sha256": digest(Path(__file__).with_name("compare_reference.py")),
              "event_helper_sha256": digest(Path(__file__).with_name("phase_tail_audit.py")),
              "rust_sources": {key: rust[key] for key in sources},
              "private_reference_native_sha256": digest(reference_path), "private_rust_pcm_sha256": digest(rust_path),
              "source_sample_rate": rate, "official_native_dtype": str(native64.dtype),
              "official_native_shape_before_downmix": list(np.asarray(source).shape),
              "native_pcm_comparison": pcm_summary(native32, rust["rust_native_mono"]),
              "native_precision_control": pcm_summary(native64, native32),
              "resampling_dtype_contract": "2x2 native PCM normalized to f32; Rust consumes f32, soxr consumes its f64 promotion; original-file control retains upstream f64",
              "soxr_quality": "HQ (unchanged upstream default)",
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchaudio", "numpy", "soxr", "soundfile")},
              "torchaudio_available_backends": torchaudio.list_audio_backends(),
              "event_parity_atol_s": SOURCE_EVENT_ATOL, "event_correspondence_window_s": EVENT_MATCH_S,
              "controls": controls, "controls_passed": all(controls.values()),
              "variants": summaries, "effects": effects,
              "source_event_parity_passed": all(effects["original_file_to_shipping"][key + "_event_parity"]["passed"] for key in ("beat", "downbeat")),
              "not_checked": ["music_accuracy", "holdout", "other_codecs_or_rates", "resampler_replacement_candidate"],
              "product_or_threshold_changed": False}
    write_new(args.output, report)
    print(json.dumps({"controls_passed": report["controls_passed"], "source_event_parity_passed": report["source_event_parity_passed"]}))
    if not report["controls_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
