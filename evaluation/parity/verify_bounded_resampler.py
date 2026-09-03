#!/usr/bin/env python3
"""Link a frozen pre-optimization probe to a bit-identical bounded implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pcm_bytes(value):
    array = np.asarray(value, dtype="<f4")
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("expected finite nonempty mono PCM")
    return array.tobytes()


def compare_cases(before, after):
    if not before or len(before) != len(after):
        raise ValueError("case count changed")
    results, identities = [], set()
    for old, new in zip(before, after):
        identity = tuple(old[k] for k in ("sample_rate", "signal", "parameter"))
        if identity in identities or identity != tuple(new[k] for k in ("sample_rate", "signal", "parameter")):
            raise ValueError("case identities changed or duplicated")
        identities.add(identity)
        checks = {key: pcm_bytes(old[key]) == pcm_bytes(new[key])
                  for key in ("input_pcm", "current_pcm", "candidate_pcm")}
        results.append(dict(sample_rate=identity[0], signal=identity[1], parameter=identity[2],
                            bitwise_equal=checks, passed=all(checks.values())))
    return results


def verify(before_path, after_path):
    lock_path = DIRECTORY / "resampler-characterization-v1.json"
    lock = json.loads(lock_path.read_bytes())
    if digest(before_path) != lock["trace_sha256"]:
        raise ValueError("before trace is not the frozen characterization input")
    before, after = (json.loads(path.read_bytes()) for path in (before_path, after_path))
    for key in ("candidate", "observation_contract", "model_sample_rate", "purpose", "probe_source_sha256",
                "adapter_source_sha256", "audio_preprocessing_sha256"):
        if before[key] != after[key]:
            raise ValueError(f"unrelated trace identity changed: {key}")
    source = ROOT / "crates/rhythm-map-eval/examples/support/reference_resampler.rs"
    if after["candidate_source_sha256"] != digest(source):
        raise ValueError("bounded trace does not match the checked-out implementation")
    if before["candidate_source_sha256"] != lock["sources"]["candidate_source_sha256"]:
        raise ValueError("original source identity mismatch")
    cases = compare_cases(before["cases"], after["cases"])
    if len(cases) != 99:
        raise ValueError("the frozen probe must have 99 cases")
    return dict(schema_version=1, purpose="bounded_resampler_bit_identity_not_music_accuracy",
                verifier_sha256=digest(Path(__file__)), characterization_report_sha256=digest(lock_path),
                before_trace_sha256=digest(before_path), after_trace_sha256=digest(after_path),
                candidate=after["candidate"], before_source_sha256=before["candidate_source_sha256"],
                after_source_sha256=after["candidate_source_sha256"], coefficient_budget_bytes=8 * 1024 * 1024,
                memory_scope="coefficient allocation only; excludes input/output audio and model memory",
                cases=cases, passed=all(c["passed"] for c in cases),
                not_checked=["musical_accuracy", "peak_process_memory", "stable_performance_benchmark"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace evidence")
    report = verify(args.before, args.after)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Bitwise identity: {sum(c['passed'] for c in report['cases'])}/{len(report['cases'])}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
