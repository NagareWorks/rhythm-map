"""Truth-assisted decomposition AFTER frozen search; never an inference input."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
PERIODS = {
    "constant_intact": (24, 24, 24),
    "constant_weak_alternating": (24, 24, 24),
    "half_speed_intact": (24, 48, 24),
    "double_speed_intact": (24, 12, 24),
    "double_speed_weak_alternating": (24, 12, 24),
    "non_octave_intact": (24, 32, 24),
    "constant_all_weak": (24, 24, 24),
}


def cell(values):
    if min(values) == max(values):
        return 0.0
    stats = [0.25 * values[i - 1] + 0.5 * x + 0.25 * values[(i + 1) % len(values)]
             for i, x in enumerate(values)]
    maximum = max(stats)
    return stats[0] - maximum - math.log(sum(math.exp(s - maximum) for s in stats) / len(stats))


def heads(name):
    beat, bar, truth = [-8.] * 1152, [-8.] * 1152, []
    for part, period in enumerate(PERIODS[name]):
        for index, frame in enumerate(range(part * 384 + 4, (part + 1) * 384, period)):
            amplitude = -2. if name == "constant_all_weak" else 8.
            beat[frame - 1:frame + 2] = [amplitude] * 3
            if index % 4 == 0:
                bar[frame - 1:frame + 2] = [amplitude] * 3
            truth.append(dict(frame=frame, period_frames=period, beat_in_bar=index % 4, meter=4))
    if "alternating" in name:
        period = PERIODS[name][1]
        for frame in range(384 + 4 + period, 768, period * 2):
            beat[frame - 1:frame + 2] = [-2.] * 3
    return beat, bar, truth


def score(beat, bar, ticks):
    result = dict(beat_evidence=0., bar_evidence=0., duration_prior=-math.log(66),
                  meter_prior=0., edge_prior=-2 * math.log(525))
    previous, bar_start = None, None
    for tick in ticks:
        start, duration = tick["frame"], tick["period_frames"]
        result["beat_evidence"] += cell(beat[start:start + duration])
        if previous is not None:
            weights = [-math.log(100) * abs(math.log2(previous / q)) for q in range(10, 76)]
            result["duration_prior"] += weights[duration - 10] - math.log(sum(map(math.exp, weights)))
        previous = duration
        if tick["beat_in_bar"] == 0:
            bar_start = start
            result["meter_prior"] -= math.log(6)
        if tick["beat_in_bar"] + 1 == tick["meter"]:
            result["bar_evidence"] += cell(bar[bar_start:start + duration])
            bar_start = None
    if bar_start is not None:
        raise ValueError("incomplete diagnostic bar")
    result["log_unnormalized_weight"] = sum(result.values())
    return result


def diagnose(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    rows = []
    for row in report["cases"]:
        beat, bar, truth = heads(row["case"])
        for label, values in (("beat", beat), ("bar", bar)):
            digest = hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()
            if digest != row[f"{label}_f64_le_sha256"]:
                raise ValueError("authored reconstruction differs from frozen input")
        run = row["decoded"]["runs"][0]
        first, last = run["map_complete_bar_span"]
        # Only a diagnostic competitor on exactly the SAME common edge domain.
        # Final 3-beat bar is a graph-legal truncation, not an assertion of truth.
        candidate = [t for t in truth if first <= t["frame"] and t["frame"] + t["period_frames"] <= last]
        if candidate[0]["frame"] != first or candidate[-1]["frame"] + candidate[-1]["period_frames"] != last:
            raise ValueError("authored path cannot share frozen MAP coverage")
        final_count = candidate[-1]["beat_in_bar"] + 1
        for tick in candidate[-final_count:]:
            tick["meter"] = final_count
        selected = score(beat, bar, run["map_ticks"])
        authored = score(beat, bar, candidate)
        reconstructed_map = selected["log_unnormalized_weight"] - run["clock_log_ratio_to_null"] - run["log_reference_partition"]
        if not math.isclose(reconstructed_map, run["map_log_probability_given_clock"], abs_tol=1e-9):
            raise ValueError("independent path decomposition does not reproduce MAP")
        rows.append(dict(case=row["case"], common_span=[first, last],
                         map_score=selected, authored_timing_graph_score=authored,
                         authored_minus_map={key: authored[key] - selected[key] for key in authored}))
    return dict(schema_version=1, purpose="truth_assisted_frozen_joint_clock_decomposition",
                changes_decoder=False, changes_weights=False, training_verdict=False,
                input_report_sha256=hashlib.sha256(raw).hexdigest(),
                diagnosis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "evaluation/parity/joint-clock-v1.json")
    args = parser.parse_args()
    print(json.dumps(diagnose(args.report), indent=2))
