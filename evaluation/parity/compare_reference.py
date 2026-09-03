#!/usr/bin/env python3
"""Compare private Rust traces with pinned upstream Python and ONNX Runtime.

No training, annotation loading, policy selection, or automatic downloads.
Only aggregate numerical differences are written to the report, never PCM.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REFERENCE_LOCK = json.loads(Path(__file__).with_name("reference-lock.json").read_text(encoding="utf-8"))
REFERENCE_REVISION = REFERENCE_LOCK["reference_revision"]
# Fixed before comparing real traces; these are numerical parity budgets,
# not accuracy thresholds or audio-dependent tuning parameters.
MEL_ATOL = 1e-3
LOGIT_ATOL = 2e-3
RTOL = 1e-4
EVENT_ATOL = 1e-5
SOURCE_EVENT_ATOL = 0.020001  # One 50 Hz frame, matching upstream integration tests.
RESAMPLER_CANDIDATE = "phase-exact-bh2-256-v1"
RESAMPLER_CANDIDATE_CONTRACT = "beat-this-rten-observations-v2+decode-audio-v2+" + RESAMPLER_CANDIDATE


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def compare(left, right, atol: float) -> dict:
    left, right = np.asarray(left), np.asarray(right)
    result = {"left_shape": list(left.shape), "right_shape": list(right.shape)}
    if left.shape != right.shape:
        return dict(result, passed=False, reason="shape_mismatch")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return dict(result, passed=False, reason="nonfinite")
    error = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return dict(result, passed=bool(np.allclose(left, right, atol=atol, rtol=RTOL)),
                max_abs=float(error.max()) if error.size else 0.0,
                mean_abs=float(error.mean()) if error.size else 0.0,
                p95_abs=float(np.quantile(error, 0.95)) if error.size else 0.0)


def events(left, right, atol=EVENT_ATOL) -> dict:
    left, right = np.asarray(left), np.asarray(right)
    result = compare(left, right, atol)
    # Timestamp agreement is absolute, not relative to the track duration.
    if left.ndim != 1 or right.ndim != 1:
        return dict(result, passed=False, reason="events_must_be_vectors")
    if left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all():
        result["passed"] = bool(np.allclose(left, right, atol=atol, rtol=0))
    return result


def waveform_diagnostic(reference, actual) -> dict:
    """Different resampling filters need not be bit-identical. Diagnose delay only."""
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    count = min(len(reference), len(actual), 44100)
    if count < 256 or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("invalid/too short waveform for resampling diagnosis")
    errors = []
    for lag in range(-128, 129):
        left_start, right_start = max(0, -lag), max(0, lag)
        n = count - abs(lag)
        error = reference[left_start:left_start + n] - actual[right_start:right_start + n]
        errors.append(float(np.mean(error ** 2)))
    best = int(np.argmin(errors))
    return {"reference_sample_count": len(reference), "rust_sample_count": len(actual),
            "sample_count_delta": len(actual) - len(reference),
            "diagnostic_prefix_samples": count,
            "best_rust_delay_samples": best - 128,
            "unshifted_rmse": float(np.sqrt(errors[128])),
            "best_shift_rmse": float(np.sqrt(errors[best])),
            "timestamps_shifted": False}


def validate_trace(trace: dict, manifest_digest: str) -> None:
    if trace.get("schema_version") != 1 or trace.get("purpose") != "calibration_parity_private":
        raise ValueError("expected a calibration-only private trace")
    if trace["model_manifest_sha256"] != manifest_digest:
        raise ValueError("trace and verified model pack differ")
    if trace["sample_rate"] != 22050:
        raise ValueError("expected actual shipping decoded PCM at 22050 Hz")
    contract = trace["observation_contract"]
    if contract not in ("beat-this-rten-observations-v1+decode-audio-v1",
                        "beat-this-rten-observations-v2+decode-audio-v2", RESAMPLER_CANDIDATE_CONTRACT):
        raise ValueError("unknown observation contract; audit the reference first")
    if contract == RESAMPLER_CANDIDATE_CONTRACT:
        if trace.get("preprocessing_candidate") != RESAMPLER_CANDIDATE:
            raise ValueError("candidate identity differs from its observation contract")
        if trace.get("candidate_source_sha256") != digest(Path(__file__).resolve().parents[2] / "crates/rhythm-map-eval/examples/support/reference_resampler.rs"):
            raise ValueError("candidate trace is stale relative to its frozen implementation")
    elif trace.get("preprocessing_candidate"):
        raise ValueError("candidate must not claim the shipping observation contract")
    if "decode-audio-v2" in contract:
        source_digest = trace.get("audio_preprocessing_sha256", "")
        if len(source_digest) != 64 or any(c not in "0123456789abcdef" for c in source_digest):
            raise ValueError("v2 trace requires the audio preprocessing source digest")
    shape = trace["mel_shape"]
    if len(shape) != 3 or shape[0] != 1 or shape[2] != 128:
        raise ValueError("invalid mel shape")
    if shape[1] != len(trace["beat_logits"]) or shape[1] != len(trace["downbeat_logits"]):
        raise ValueError("logit and mel lengths differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model-pack", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--source-audio", type=Path, action="append",
                        help="Original encoded files, in trace order; each digest must match")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-tail-lock", type=Path,
                        help="Predeclared calibration-only legacy origin/tail counterfactuals")
    args = parser.parse_args()
    if args.source_audio and len(args.source_audio) != len(args.trace):
        parser.error("provide one source audio per trace, in the same order")
    if args.output.exists():
        parser.error("refusing to overwrite a report")
    audit_cases = None
    if args.phase_tail_lock:
        from phase_tail_audit import validate_audit_lock
        if not args.checkpoint or not args.source_audio:
            parser.error("phase/tail diagnosis requires the original checkpoint and source files")
        audit_cases = validate_audit_lock(
            json.loads(args.phase_tail_lock.read_text(encoding="utf-8")),
            digest(Path(__file__).with_name("reference-lock.json")),
            [json.loads(path.read_text(encoding="utf-8")) for path in args.trace])
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != REFERENCE_REVISION:
        parser.error("reference checkout is not the pinned revision")
    if subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True).strip():
        parser.error("reference checkout must be clean")
    pack_digest = digest(args.model_pack)
    if pack_digest != REFERENCE_LOCK["model_manifest_sha256"]:
        parser.error("model pack differs from the audit lock")
    manifest = json.loads(args.model_pack.read_text(encoding="utf-8"))
    artifacts = {}
    for artifact in manifest["artifacts"]:
        path = (args.model_dir / artifact["file"]).resolve()
        if not path.is_relative_to(args.model_dir.resolve()):
            parser.error("model artifact escapes its directory")
        if path.stat().st_size != artifact["size_bytes"] or digest(path) != artifact["sha256"]:
            parser.error("model artifact hash or size mismatch")
        artifacts[artifact["role"]] = path
    checkpoint_digest = None
    if args.checkpoint:
        checkpoint_digest = digest(args.checkpoint)
        if (checkpoint_digest != args.checkpoint_sha256 or
                checkpoint_digest != REFERENCE_LOCK["checkpoint"]["sha256"] or
                args.checkpoint.stat().st_size != REFERENCE_LOCK["checkpoint"]["size_bytes"]):
            parser.error("explicit matching checkpoint SHA-256 is required")

    sys.path.insert(0, str(args.upstream.resolve()))
    import torch
    import onnxruntime as ort
    from beat_this.inference import load_model, split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import LogMelSpect, load_audio
    import soxr

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    def session(path):
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    mel_session, beat_session = session(artifacts["mel_frontend"]), session(artifacts["beat_model"])
    official_model = load_model(str(args.checkpoint), "cpu") if args.checkpoint else None
    frontend, postprocess = LogMelSpect(device="cpu"), Postprocessor(type="minimal")

    def ort_model(chunk):
        outputs = beat_session.run(None, {"spectrogram": chunk.numpy()})
        return {out.name: torch.from_numpy(value) for out, value in zip(beat_session.get_outputs(), outputs)}

    def predict(mel, model):
        with torch.inference_mode():
            return split_predict_aggregate(torch.from_numpy(mel), 1500, 6, "keep_first", model)

    cases = []
    for case_index, trace_path in enumerate(args.trace):
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        validate_trace(trace, pack_digest)
        print("Comparing " + trace["case_id"], flush=True)
        pcm = np.asarray(trace["mono_samples"], dtype=np.float32)
        rust_mel = np.asarray(trace["mel_values"], dtype=np.float32).reshape(trace["mel_shape"])[0]
        rust_logits = {key: torch.tensor(trace[key + "_logits"], dtype=torch.float32) for key in ("beat", "downbeat")}
        with torch.inference_mode():
            official_mel = frontend(torch.from_numpy(pcm)).numpy()
            ort_mel = mel_session.run(None, {"audio_pcm": pcm[None]})[0][0]
        ort_logits = predict(rust_mel, ort_model)
        decoded = postprocess(**rust_logits)
        stages = {
            "mel_onnxruntime_vs_rten_same_pcm": compare(ort_mel, rust_mel, MEL_ATOL),
            "mel_official_vs_rten_same_pcm": compare(official_mel, rust_mel, MEL_ATOL),
        }
        source_diagnostic = None
        for index, key in enumerate(("beat", "downbeat")):
            stages[key + "_onnxruntime_vs_rten_same_mel"] = compare(ort_logits[key].numpy(), rust_logits[key].numpy(), LOGIT_ATOL)
            upstream = trace["upstream_" + ("beats" if index == 0 else "downbeats")]
            product = [b["time_s"] for b in trace["observations"]["beats"] if index == 0 or b["downbeat_confidence"] >= 0.5]
            stages[key + "_official_vs_port_decoder_same_logits"] = events(decoded[index], upstream)
            stages[key + "_port_vs_adapter_decoder_same_logits"] = events(upstream, product)
        if official_model is not None:
            official_same_mel = predict(rust_mel, official_model)
            official_pipeline = predict(official_mel, official_model)
            official_events = postprocess(**official_pipeline)
            for index, key in enumerate(("beat", "downbeat")):
                stages[key + "_checkpoint_vs_onnx_same_mel"] = compare(official_same_mel[key].numpy(), ort_logits[key].numpy(), LOGIT_ATOL)
                stages[key + "_official_vs_rten_same_pcm"] = compare(official_pipeline[key].numpy(), rust_logits[key].numpy(), LOGIT_ATOL)
                stages[key + "_pipeline_event_agreement_same_pcm"] = events(official_events[index], trace["upstream_" + ("beats" if index == 0 else "downbeats")])
        if args.source_audio:
            source_path = args.source_audio[case_index]
            if digest(source_path) != trace["audio_sha256"]:
                raise ValueError("source audio does not match the trace identity")
            signal, sample_rate = load_audio(str(source_path))
            if audit_cases and sample_rate != audit_cases[trace["case_id"]]["source_sample_rate"]:
                raise ValueError("source sample rate differs from phase/tail lock")
            if signal.ndim == 2:
                signal = signal.mean(1)
            elif signal.ndim != 1:
                raise ValueError("unsupported source audio shape")
            if sample_rate != 22050:
                signal = soxr.resample(signal, in_rate=sample_rate, out_rate=22050)
            signal = np.asarray(signal[:trace["prefix_seconds"] * 22050], dtype=np.float32)
            source_diagnostic = waveform_diagnostic(signal, pcm)
            source_diagnostic["source_sample_rate"] = sample_rate
            if official_model is not None:
                with torch.inference_mode():
                    source_mel = frontend(torch.from_numpy(signal)).numpy()
                source_logits = predict(source_mel, official_model)
                source_events = postprocess(**source_logits)
                for index, key in enumerate(("beat", "downbeat")):
                    other = trace["upstream_" + ("beats" if index == 0 else "downbeats")]
                    comparison = events(source_events[index], other, SOURCE_EVENT_ATOL)
                    stages[key + "_source_audio_event_agreement"] = comparison
        audit = None
        if audit_cases:
            from phase_tail_audit import run_audit
            def predict_pcm(samples):
                with torch.inference_mode():
                    mel = frontend(torch.from_numpy(samples)).numpy()
                return predict(mel, official_model)
            print("Ablating origin/tail: " + trace["case_id"], flush=True)
            audit = run_audit(trace, audit_cases[trace["case_id"]], predict_pcm, postprocess, official_pipeline)
        cases.append({"case_id": trace["case_id"], "audio_sha256": trace["audio_sha256"],
                      "suite_sha256": trace["suite_sha256"], "trace_sha256": digest(trace_path),
                      "trace_exporter_sha256": trace.get("trace_exporter_sha256"),
                      "adapter_source_sha256": trace.get("adapter_source_sha256"),
                      "audio_preprocessing_sha256": trace.get("audio_preprocessing_sha256"),
                      "observation_contract": trace["observation_contract"],
                      "preprocessing_candidate": trace.get("preprocessing_candidate"),
                      "candidate_source_sha256": trace.get("candidate_source_sha256"),
                      "sample_count": len(pcm), "mel_frames": len(rust_mel),
                      "source_waveform_diagnostic": source_diagnostic, "stages": stages})
        if audit is not None:
            cases[-1]["phase_tail_audit"] = audit
    report = {"schema_version": 1, "purpose": "numerical_parity_not_accuracy",
              "comparator_sha256": digest(Path(__file__)),
              "reference_lock_sha256": digest(Path(__file__).with_name("reference-lock.json")),
              "reference_revision": revision, "checkpoint_sha256": checkpoint_digest,
              "model_manifest_sha256": pack_digest, "reference_complete": official_model is not None,
              "budgets": {"mel_atol": MEL_ATOL, "logit_atol": LOGIT_ATOL, "rtol": RTOL, "event_atol_s": EVENT_ATOL,
                          "source_event_atol_s": SOURCE_EVENT_ATOL},
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchaudio", "numpy", "onnxruntime", "einops", "rotary-embedding-torch", "soxr")},
              "cases": cases,
              "passed": official_model is not None and all(s["passed"] for c in cases for s in c["stages"].values()),
              "not_checked": (["source_audio_decoder_and_resampler"] if not args.source_audio else []) +
                             ["full_track_accuracy", "holdout"]}
    if audit_cases:
        report["phase_tail_lock_sha256"] = digest(args.phase_tail_lock)
        report["phase_tail_auditor_sha256"] = digest(Path(__file__).with_name("phase_tail_audit.py"))
        report["passed"] = report["passed"] and all(
            c["phase_tail_audit"]["reconstruction"]["waveform_passed"] and
            c["phase_tail_audit"]["reconstruction"]["logits_passed"] for c in cases)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"reference_complete": report["reference_complete"], "passed": report["passed"]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
