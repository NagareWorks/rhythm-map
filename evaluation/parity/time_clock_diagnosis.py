"""Independent prior-only comparison, after both joint decoders are frozen."""
import hashlib
import json
import math
from pathlib import Path
import struct

from joint_clock_diagnosis import heads, score

ROOT = Path(__file__).resolve().parents[2]
PERIODS = list(range(10, 76))
OFF_MASS = {
    p: sum(math.exp(-math.log(100) * abs(math.log2(p / q))) for q in PERIODS if q != p)
    for p in PERIODS
}
RATE = sum(math.log1p(OFF_MASS[p]) / p for p in PERIODS) / len(PERIODS)


def exposure_score(beat, bar, ticks):
    result = score(beat, bar, ticks)
    duration_prior = -math.log(len(PERIODS))
    for previous, current in zip(ticks, ticks[1:]):
        p, q = previous["period_frames"], current["period_frames"]
        if p == q:
            duration_prior -= RATE * p
        else:
            duration_prior += math.log(-math.expm1(-RATE * p)) - math.log(OFF_MASS[p])
            duration_prior -= math.log(100) * abs(math.log2(p / q))
    duration_prior -= RATE * ticks[-1]["period_frames"]
    result["duration_prior"] = duration_prior
    result["log_unnormalized_weight"] = sum(v for k, v in result.items() if k != "log_unnormalized_weight")
    return result


def diagnose():
    current_path = ROOT / "evaluation/parity/time-clock-v1.json"
    previous_path = ROOT / "evaluation/parity/joint-clock-v1.json"
    current = json.loads(current_path.read_bytes())
    previous = json.loads(previous_path.read_bytes())
    if not math.isclose(current["rate_per_frame"], RATE, abs_tol=1e-12):
        raise ValueError("prior rate differs from source-independent reconstruction")
    old_rows = {r["case"]: r for r in previous["cases"]}
    rows = []
    for row in current["cases"]:
        beat, bar, truth = heads(row["case"])
        old = old_rows[row["case"]]
        for label, values in (("beat", beat), ("bar", bar)):
            digest = hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()
            if digest != row[f"{label}_f64_le_sha256"] or digest != old[f"{label}_f64_le_sha256"]:
                raise ValueError("changed control input")
        run, old_run = row["decoded"]["runs"][0], old["decoded"]["runs"][0]
        first, last = old_run["map_complete_bar_span"]
        candidate = [t for t in truth if first <= t["frame"] and t["frame"] + t["period_frames"] <= last]
        final_count = candidate[-1]["beat_in_bar"] + 1
        for tick in candidate[-final_count:]:
            tick["meter"] = final_count
        selected = exposure_score(beat, bar, run["map_ticks"])
        old_map = exposure_score(beat, bar, old_run["map_ticks"])
        authored = exposure_score(beat, bar, candidate)
        reconstructed = selected["log_unnormalized_weight"] - run["clock_log_ratio_to_null"] - run["log_reference_partition"]
        if not math.isclose(reconstructed, run["map_log_probability_given_clock"], abs_tol=1e-9):
            raise ValueError("independent decomposition differs from selected MAP")
        if max(old_map["log_unnormalized_weight"], authored["log_unnormalized_weight"]) > selected["log_unnormalized_weight"] + 1e-9:
            raise ValueError("search missed an available higher-scoring competitor")
        rows.append(dict(case=row["case"], current_map=selected,
                         old_map_with_new_prior=old_map, authored_timing_with_new_prior=authored,
                         diagnostic_competitor_span=[first, last], current_map_span=run["map_complete_bar_span"],
                         authored_minus_old_map={key: authored[key] - old_map[key] for key in authored}))
    return dict(schema_version=1, purpose="time_exposure_prior_controlled_intervention",
                rate_per_frame=RATE, rate_per_second=RATE * 50,
                constant_duration_costs={str(p): -(1152 // p) * RATE * p for p in (12, 24, 48)},
                current_report_sha256=hashlib.sha256(current_path.read_bytes()).hexdigest(),
                previous_report_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                diagnosis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                shared_diagnosis_source_sha256=hashlib.sha256((ROOT / "evaluation/parity/joint_clock_diagnosis.py").read_bytes()).hexdigest(),
                cases=rows)


if __name__ == "__main__":
    print(json.dumps(diagnose(), indent=2))
