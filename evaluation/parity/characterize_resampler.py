#!/usr/bin/env python3
"""Analyze generated physical signals, never music, models, or beat labels."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from compare_reference import digest

RATE = 22050


def vector(value):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError("expected nonempty finite mono PCM")
    return value


def waveform(reference, actual):
    reference, actual = vector(reference), vector(actual)
    result = {"reference_frames": len(reference), "actual_frames": len(actual)}
    if reference.shape != actual.shape:
        return dict(result, equal_length=False)
    diff = reference.astype(np.float64) - actual.astype(np.float64)
    interior = diff[1024:-1024] if len(diff) > 2048 else diff
    return dict(result, equal_length=True, max_abs=float(np.abs(diff).max()),
                rmse=float(np.sqrt(np.mean(diff ** 2))),
                interior_rmse=float(np.sqrt(np.mean(interior ** 2))))


def impulse_response(samples, source_rate, position_s):
    samples = vector(samples)
    spectrum = np.fft.rfft(samples.astype(np.float64), n=131072)
    frequencies = np.fft.rfftfreq(131072, 1 / RATE)
    dc = float(np.abs(spectrum[0]))
    if dc < 1e-12:
        raise ValueError("zero DC response")
    magnitude_db = 20 * np.log10(np.maximum(np.abs(spectrum) / dc, 1e-12))
    reference_nyquist = min(source_rate, RATE) / 2
    result = {"dc_gain": dc, "peak_time_error_samples": float(np.argmax(np.abs(samples)) - position_s * RATE)}
    for db in (3, 6, 60, 100):
        crossing = np.flatnonzero((frequencies > reference_nyquist * 0.5) & (magnitude_db <= -db))
        result[f"minus_{db}_db_relative_nyquist"] = float(frequencies[crossing[0]] / reference_nyquist) if crossing.size else None
    phase_mask = (frequencies > reference_nyquist * 0.1) & (frequencies < reference_nyquist * 0.8)
    omega = 2 * np.pi * frequencies[phase_mask] / RATE
    corrected = spectrum[phase_mask] * np.exp(1j * omega * position_s * RATE)
    phase = np.unwrap(np.angle(corrected))
    result["passband_delay_output_samples"] = float(-np.polyfit(omega, phase, 1)[0])
    return result


def summarize_case(case, resample):
    source = vector(case["input_pcm"])
    rate = case["sample_rate"]
    reference = source.copy() if rate == RATE else vector(resample(source.astype(np.float64), rate, RATE))
    outputs = {"current": vector(case["current_pcm"])}
    if "candidate_pcm" in case:
        outputs["candidate"] = vector(case["candidate_pcm"])
    wanted = (len(source) * RATE + rate // 2) // rate
    if len(reference) != wanted:
        raise ValueError("reference duration differs from integer-rounded contract")
    comparisons = {key: waveform(reference, pcm) for key, pcm in outputs.items()}
    result = {"sample_rate": rate, "signal": case["signal"], "parameter": case["parameter"],
              "input_frames": len(source), "wanted_frames": wanted,
              "comparisons": comparisons,
              "lengths_passed": all(len(pcm) == wanted for pcm in outputs.values()),
              "elapsed_ms": {key: case[key + "_elapsed_ms"] for key in outputs}}
    if case["signal"] == "impulse_center":
        result["responses"] = {key: impulse_response(pcm, rate, case["parameter"])
                               for key, pcm in dict(reference=reference, **outputs).items()}
    if case["signal"].startswith("impulse_"):
        result["impulse_peak_errors_samples"] = {key: float(np.argmax(np.abs(pcm)) - case["parameter"] * RATE)
                                                 for key, pcm in dict(reference=reference, **outputs).items()}
    if case["signal"].startswith("tone_"):
        result["tone_interior_gain_db"] = {
            key: float(20 * np.log10(max(np.sqrt(np.mean(pcm[1024:-1024].astype(np.float64) ** 2)) / (0.5 / np.sqrt(2)), 1e-12)))
            for key, pcm in dict(reference=reference, **outputs).items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a report")
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    if (trace.get("schema_version") != 1 or trace.get("purpose") != "generated_resampler_characterization" or
            trace.get("model_sample_rate") != RATE):
        parser.error("expected generated resampler trace")
    root = Path(__file__).resolve().parents[2]
    sources = {"probe_source_sha256": "crates/rhythm-map-eval/examples/resampler_probe.rs",
               "adapter_source_sha256": "crates/rhythm-map-beat-this/src/lib.rs",
               "audio_preprocessing_sha256": "crates/rhythm-map-beat-this/src/audio.rs"}
    if "candidate_source_sha256" in trace:
        sources["candidate_source_sha256"] = "crates/rhythm-map-eval/examples/support/reference_resampler.rs"
    if any(trace[key] != digest(root / path) for key, path in sources.items()):
        parser.error("trace source identities differ; rebuild the probe")
    import soxr
    results = [summarize_case(case, soxr.resample) for case in trace["cases"]]
    report = {"schema_version": 1, "purpose": "generated_resampler_characterization_not_music_accuracy",
              "trace_sha256": digest(args.trace), "analyzer_sha256": digest(Path(__file__)),
              "sources": {key: trace[key] for key in sources},
              "reference": {"python_soxr": importlib.metadata.version("soxr"), "libsoxr": soxr.__libsoxr_version__, "quality": "HQ"},
              "observation_contract": trace["observation_contract"], "candidate": trace.get("candidate"),
              "cases": results, "all_lengths_passed": all(r["lengths_passed"] for r in results),
              "not_checked": ["musical_accuracy", "model_parity", "holdout", "short_clip_lengths"],
              "timing_note": "one sequential processing call including initialization; not a stable benchmark"}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"cases": len(results), "all_lengths_passed": report["all_lengths_passed"],
                      "impulse_centers": [r for r in results if r["signal"] == "impulse_center"]}, indent=2))


if __name__ == "__main__":
    main()
