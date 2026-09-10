"""Freeze one source/parent-data plan before the loss-only experiment."""
import argparse
import json
from pathlib import Path

from experiments.trajectory_clock.attribution import ROOT, INPUT_SHA, REPORT_SHA, sha, require, write_json

HERE = Path(__file__).resolve().parent


def source_hashes():
    paths = sorted(HERE.glob('*.py')) + sorted((ROOT / 'experiments/coupled_clock').glob('*.py'))
    paths += [HERE / 'PROTOCOL-v1.md', ROOT / 'experiments/clock_readout/model.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'new private plan required')
    old = ROOT / 'experiments/coupled_clock'
    require(sha(old / 'inputs-v1.json') == INPUT_SHA and sha(old / 'results-v1.json') == REPORT_SHA, 'parent identity changed')
    write_json(args.output, dict(schema='rhythm-map.trajectory-clock-plan.v1', parent_inputs_sha256=INPUT_SHA,
        parent_report_sha256=REPORT_SHA, attribution_sha256=sha(HERE / 'attribution-v1.json'),
        source_sha256=source_hashes(), population=json.loads((old / 'inputs-v1.json').read_bytes())['cases'],
        training=False, holdout_access=False, production_change=False))
    print('TRAJECTORY_PLAN_READY ' + sha(args.output), flush=True)


if __name__ == '__main__':
    main()
