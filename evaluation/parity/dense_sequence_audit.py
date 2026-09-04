"""Fixed full-frame clock experiment. Truth is confined to evaluation, not Rust decoding."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import subprocess

import dense_clock_evidence as dense
from clock_phase_evidence import matches, require
from resampler_event_audit import default_events

ROOT = dense.ROOT
DECODER = ROOT / "crates/rhythm-map-eval/examples/support/dense_sequence.rs"
RUNNER = ROOT / "crates/rhythm-map-eval/examples/dense_sequence.rs"
CORE = ROOT / "crates/rhythm-map-core/src/estimator.rs"
CORE_HASH = "3d2bc3ca875025b5d08e511dcecf38351fc8f62e27daf8d49147f9f8a68bf8f1"


def quantile(values, q):
    return sorted(values)[math.ceil((len(values) - 1) * q)] if values else None


def beat_metrics(predicted, truth, tolerance=.07):
    pairs = matches(predicted, truth, tolerance)
    errors = [abs(predicted[i] - truth[j]) * 1000 for i, j in pairs]
    precision = len(pairs) / len(predicted) if predicted else 1.
    recall = len(pairs) / len(truth) if truth else 1.
    return dict(matched=len(pairs), precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.,
                median_absolute_error_ms=quantile(errors, .5), p95_absolute_error_ms=quantile(errors, .95)), pairs


def segment_tempo(segments, time):
    for s in segments:
        if s["start_s"] <= time < s["end_s"]:
            fraction = (time - s["start_s"]) / (s["end_s"] - s["start_s"])
            return s["start_bpm"] + fraction * (s["end_bpm"] - s["start_bpm"])
    return None


def clock_tempo(decoded, time):
    frame = time * 50
    for c in decoded["components"]:
        ticks = c["ticks"]
        if not (c["start_frame"] <= frame < c["end_frame"]) or not ticks:
            continue
        if frame < ticks[0]["frame"]:
            return 3000 / ticks[0]["period_frames"], True
        for left, right in zip(ticks, ticks[1:]):
            if left["frame"] <= frame < right["frame"]:
                return 3000 / right["period_frames"], False
        return 3000 / ticks[-1]["period_frames"], True
    return None, False


def tempo_measure(truth, lookup):
    queries = [(a["time_s"] + b["time_s"]) / 2 for a, b in zip(truth["beats"], truth["beats"][1:])]
    queries = [(t, segment_tempo(truth["tempo_segments"], t)) for t in queries]
    queries = [(t, bpm) for t, bpm in queries if bpm is not None]
    errors, prior = [], 0
    for t, bpm in queries:
        actual, prior_only = lookup(t)
        if actual is not None:
            errors.append(abs(actual / bpm - 1) * 100)
            prior += prior_only
    return dict(queries=len(queries), scored=len(errors), unavailable=len(queries) - len(errors),
                endpoint_prior_only_queries=prior, median_error_percent=quantile(errors, .5),
                p95_error_percent=quantile(errors, .95), maximum_error_percent=max(errors, default=None))


def compare_metrics(before, after):
    regressions = []
    for key in ("matched", "precision", "recall", "f1"):
        if after[key] < before[key] - 1e-12:
            regressions.append(key)
    for key in ("median_absolute_error_ms", "p95_absolute_error_ms"):
        if before[key] is not None and (after[key] is None or after[key] > before[key] + 1e-9):
            regressions.append(key)
    return regressions


def change_measure(truth, decoded, baseline):
    predictions = []
    for c in decoded["components"]:
        for a, b in zip(c["ticks"][1:], c["ticks"][2:]):
            if abs(math.log(b["period_frames"] / a["period_frames"])) >= math.log1p(.12):
                predictions.append(a["frame"] / 50)
    expected = [c["time_s"] for c in truth["change_points"] if c["kind"] == "tempo_jump"]
    base = [c["time_s"] for c in baseline.get("change_points", []) if c["kind"] == "tempo_jump"]
    # Unannotated expressive fluctuations are descriptive counts, not false positives.
    rows = []
    for time in expected:
        row = {}
        for label, at in (("before", time - 1), ("after", time + 1)):
            target = segment_tempo(truth["tempo_segments"], at)
            actual, prior = clock_tempo(decoded, at)
            row[label] = dict(expected_bpm=target, candidate_bpm=actual, endpoint_prior_only=prior,
                              error_percent=abs(actual / target - 1) * 100 if actual and target else None)
        rows.append(row)
    return dict(annotated_tempo_jumps=len(expected), candidate_adjacent_period_jumps=len(predictions),
                candidate_matches_within_1s=len(matches(predictions, expected, 1.)),
                baseline_matches_within_1s=len(matches(base, expected, 1.)),
                mean_nearest_candidate_boundary_error_s=statistics.mean(
                    min(abs(p - t) for p in predictions) for t in expected) if expected and predictions else None,
                before_after_probes=rows)


def measure(truth, response, raw_times, expected_score=None):
    decoded, baseline = response["decoded"], response["baseline"]
    times = [b["time_s"] for b in truth["beats"]]
    ticks = [t for c in decoded["components"] for t in c["ticks"]]
    candidate = [t["frame"] / 50 for t in ticks]
    primary = [b["time_s"] for b in baseline["beats"]]
    raw, raw_pairs = beat_metrics(raw_times, times)
    previous, previous_pairs = beat_metrics(primary, times)
    proposed, proposed_pairs = beat_metrics(candidate, times)
    if expected_score is not None:
        expected = expected_score["metrics"]["beats"]
        require(all(previous[k] == v if v is None else previous[k] is not None and abs(previous[k] - v) < 1e-9
                    for k, v in expected.items()), "primary beat score replay changed")
    down_truth = [b["time_s"] for b in truth["beats"] if b["downbeat"]]
    down_candidate = [t["frame"] / 50 for t in ticks if t["bar_phase"] == 0]
    down_base = [b["time_s"] for b in baseline["beats"] if b["downbeat"]]
    down_metrics, _ = beat_metrics(down_candidate, down_truth)
    down_previous, _ = beat_metrics(down_base, down_truth)
    candidate_ids, primary_ids = {j for _, j in proposed_pairs}, {j for _, j in previous_pairs}
    raw_ids = {j for _, j in raw_pairs}
    tempo_candidate = tempo_measure(truth, lambda t: clock_tempo(decoded, t))
    tempo_primary = tempo_measure(truth, lambda t: (segment_tempo(baseline["tempo_segments"], t), False))
    regressions = compare_metrics(previous, proposed)
    if primary_ids - candidate_ids:
        regressions.append("lost_primary_truth_identities")
    for key in ("median_error_percent", "p95_error_percent"):
        b, a = tempo_primary[key], tempo_candidate[key]
        if b is not None and (a is None or a > b + 1e-9):
            regressions.append("tempo_" + key)
    if tempo_candidate["unavailable"] > tempo_primary["unavailable"]:
        regressions.append("tempo_coverage")
    if down_truth and compare_metrics(down_previous, down_metrics):
        regressions.append("downbeat_metrics")
    changes = change_measure(truth, decoded, baseline)
    if changes["candidate_matches_within_1s"] < changes["baseline_matches_within_1s"]:
        regressions.append("tempo_change_recall")
    return dict(raw_beats=raw, primary_beats=previous, inferred_clock_beats=proposed,
                primary_downbeats=down_previous, inferred_clock_downbeats=down_metrics,
                recovered_raw_truth_count=len(candidate_ids - raw_ids), lost_raw_truth_count=len(raw_ids - candidate_ids),
                recovered_primary_truth_count=len(candidate_ids - primary_ids), lost_primary_truth_count=len(primary_ids - candidate_ids),
                tempo_primary=tempo_primary, tempo_candidate=tempo_candidate, changes=changes,
                regression_reasons=regressions, no_regression=not regressions,
                clock_ticks=len(ticks), mixture_missing_ticks=sum(t["missing_component"] for t in ticks),
                nonpositive_pulse_windows=sum(not t["positive_pulse_window"] for t in ticks),
                nonpositive_pulse_contrast=sum(t["pulse_contrast"] <= 0 for t in ticks),
                period_boundary_ticks=sum(t["period_frames"] in (10, 75) for t in ticks),
                meter_hypotheses=[c["meter_hypothesis_not_estimate"] for c in decoded["components"]],
                unavailable_frames=decoded["unavailable_frames"], uninformative_frames=decoded["uninformative_frames"],
                max_backpointer_bytes=decoded["max_backpointer_bytes"], elapsed_s=response["elapsed_s"])


def controls():
    for tempos in ((120, 120, 120), (120, 60, 120), (120, 90, 120), (120, 240, 120)):
        positions, segments = [], []
        for part, bpm in enumerate(tempos):
            start, end = part * 8, (part + 1) * 8
            count = round(8 * bpm / 60)
            positions.extend(start + i * 60 / bpm for i in range(count))
            segments.append(dict(start_s=start, end_s=end, start_bpm=bpm, end_bpm=bpm, kind="constant"))
        truth = dict(beats=[dict(time_s=t, downbeat=i % 4 == 0) for i, t in enumerate(positions)],
                     tempo_segments=segments, change_points=[dict(time_s=i * 8, kind="tempo_jump") for i in (1, 2)
                                                            if tempos[i] != tempos[i - 1]])
        for mask in ("intact", "weak_alternating", "erased_alternating", "erased_four", "erased_tail"):
            for downbeat in (True, False):
                beat, bar = [-8.] * 1200, [-8.] * 1200
                def pulse(values, time, peak):
                    center = round(time * 50)
                    for frame in range(max(0, center - 4), min(len(values), center + 5)):
                        values[frame] = max(values[frame], -8 + (peak + 8) * (1 - abs(frame - center) / 4))
                for i, time in enumerate(positions):
                    alternate = 8 <= time < 16 and i % 2 == 1
                    erased = (mask == "erased_alternating" and alternate) or (
                        mask == "erased_four" and len(positions) // 2 <= i < len(positions) // 2 + 4) or (
                        mask == "erased_tail" and i >= len(positions) - 8)
                    if not erased:
                        pulse(beat, time, -2 if mask == "weak_alternating" and alternate else 8)
                    if downbeat and i % 4 == 0:
                        pulse(bar, time, 8)
                identity = "authored-" + "-".join(map(str, tempos)) + "-" + mask + ("-bar" if downbeat else "-no-bar")
                yield identity, dict(beat_logits=beat, downbeat_logits=bar, available=None), truth
    identity, frames, truth = next(controls_constant())
    yield identity, frames, truth
    identity, frames, truth = next(controls_constant())
    frames["beat_logits"] = [-8.] * 1200
    frames["downbeat_logits"] = [-8.] * 1200
    frames["available"] = None
    yield "authored-flat-both-heads", frames, truth


def controls_constant():
    # Separate helper avoids recursive generator calls for the two absence cases.
    times = [i / 2 for i in range(48)]
    b, d = [-8.] * 1200, [-8.] * 1200
    for i, time in enumerate(times):
        b[round(time * 50)] = 8.
        if i % 4 == 0:
            d[round(time * 50)] = 8.
    truth = dict(beats=[dict(time_s=t, downbeat=i % 4 == 0) for i, t in enumerate(times)],
                 tempo_segments=[dict(start_s=0, end_s=24, start_bpm=120, end_bpm=120)], change_points=[])
    yield "authored-explicit-unavailable", dict(beat_logits=b, downbeat_logits=d,
                                               available=[not 400 <= i < 600 for i in range(1200)]), truth


def authored_observations(frames):
    events = default_events(frames["beat_logits"])
    bars = default_events(frames["downbeat_logits"])
    return dict(duration_s=len(frames["beat_logits"]) / 50, beats=[dict(time_s=t, confidence=1.,
                downbeat_confidence=float(any(abs(t - d) < .07 for d in bars))) for t in events],
                source=dict(backend="authored-heads", model="controlled", version=None, frame_rate_hz=50))


def run(binary, frames, baseline, private, identity, capture_hash=None):
    request = dict(frames=frames, baseline_observations=baseline)
    process = subprocess.run([str(binary)], input=json.dumps(request), text=True, encoding="utf-8",
                             capture_output=True, check=True, timeout=600)
    response = json.loads(process.stdout)
    require(response["decoder_source_sha256"] == dense.sha(DECODER.read_bytes()) and
            response["runner_source_sha256"] == dense.sha(RUNNER.read_bytes()), "stale candidate binary")
    payload = dict(response=response, capture_sha256=capture_hash,
                   input_sha256=dense.sha(json.dumps(frames, separators=(",", ":")).encode()))
    data = (json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with (private / f"{identity}.json").open("xb") as target:
        target.write(data)
    return response, dense.sha(data), payload["input_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for cohort in dense.INPUTS:
        parser.add_argument(f"--{cohort}-evidence", type=Path, required=True)
        parser.add_argument(f"--{cohort}-captures", type=Path, required=True)
    args = parser.parse_args()
    require(dense.sha(CORE.read_bytes()) == CORE_HASH, "default estimator changed")
    private = args.private_output.resolve()
    require(not private.exists() and not any((p / ".git").exists() for p in (private, *private.parents)),
            "private prediction directory must be new and outside Git")
    private.mkdir()
    authored, witnesses = [], {}
    for identity, frames, truth in controls():
        print(identity, flush=True)
        baseline = authored_observations(frames)
        response, artifact, input_hash = run(args.binary, frames, baseline, private, identity)
        authored.append(dict(id=identity, private_prediction_sha256=artifact,
                             measurement=measure(truth, response, [b["time_s"] for b in baseline["beats"]])))
        if identity in ("authored-120-120-120-erased_alternating-no-bar", "authored-120-60-120-intact-no-bar"):
            witnesses[identity] = dict(input_sha256=input_hash, output_sha256=dense.sha(
                json.dumps(response["decoded"], sort_keys=True).encode()))
    require(len(authored) == 42 and len(witnesses) == 2, "authored coverage changed")
    witness_values = list(witnesses.values())
    require(witness_values[0] == witness_values[1], "identical-input witness diverged")
    cohorts = []
    for name, (count, evidence_hash) in dense.INPUTS.items():
        evidence, _ = dense.read_json(getattr(args, f"{name}_evidence"), evidence_hash)
        captures = getattr(args, f"{name}_captures")
        summary, summary_hash = dense.read_json(captures / "summary.json")
        records = dense.validate_summary(summary, evidence["cases"], count)
        suite_file = ROOT / "evaluation/suites" / ("artbeat-v1.json" if name == "artbeat" else "rubato-calibration-v1.json")
        suite, _ = dense.read_json(suite_file, dense.SUITES[name][1])
        sources = {k: dense.sha((ROOT / v).read_bytes()) for k, v in dense.SOURCES.items()}
        cases = []
        for case, entry, record in zip(evidence["cases"], suite["cases"], records):
            print(f'{name} {len(cases)+1}/{count}: {case["id"]}', flush=True)
            payload, _ = dense.read_json(captures / f'{case["id"]}.json', record["capture_sha256"])
            dense.validate_capture(payload, record, case, name, sources)
            truth, _ = dense.read_json(suite_file.parent / entry["input"]["truth"], case["truth_sha256"])
            frames = dict(beat_logits=payload["beat_logits"], downbeat_logits=payload["downbeat_logits"], available=None)
            response, artifact, _ = run(args.binary, frames, case["observations"], private, case["id"], record["capture_sha256"])
            measurement = measure(truth, response, [b["time_s"] for b in case["observations"]["beats"]], case["selected_score"])
            cases.append(dict(id=case["id"], private_prediction_sha256=artifact, primary_beat_score_replay=True,
                              measurement=measurement))
        cohorts.append(dict(name=name, complete=len(cases) == count, evidence_sha256=evidence_hash,
                            capture_summary_sha256=summary_hash, cases=cases,
                            regression_case_count=sum(not c["measurement"]["no_regression"] for c in cases),
                            regression_reasons=dict(Counter(k for c in cases for k in c["measurement"]["regression_reasons"]))))
    report = dict(schema_version=1, purpose="frozen_truth_free_full_frame_clock_experiment",
                  decoder_source_sha256=dense.sha(DECODER.read_bytes()), runner_source_sha256=dense.sha(RUNNER.read_bytes()),
                  audit_source_sha256=dense.sha(Path(__file__).read_bytes()), estimator_source_sha256=CORE_HASH,
                  decoder_uses_truth=False, training_run=False, holdout_opened=False, production_output_changed=False,
                  promoted=False, authored=authored, identical_input_witness=witnesses, cohorts=cohorts)
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(report, target, indent=2, allow_nan=False)
        target.write("\n")


if __name__ == "__main__":
    main()
