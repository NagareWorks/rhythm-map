"""Freeze the synchronizer's one musical comparison before optimization."""
import argparse
import json
from pathlib import Path

import torch

from experiments.coupled_clock.clock import require
from experiments.coupled_clock.prepare import ROOT, sha, write_json
from experiments.coupled_clock.run import SEED, state_hash
from experiments.phase_sync.model import PhaseSyncReadout

HERE = Path(__file__).resolve().parent
INPUT_SHA = 'b4f05c10ebede84c394b0b8e06f6723b096d1f9a0a48e63024eb81d913fffd9c'
PARENTS = {
    'coupled': ('coupled_clock', 'e33a44db4ce20a35f0676aa3f4fc47806da83f4a2ab80c1cfb555ce5343cda9a'),
    'trajectory': ('trajectory_clock', 'a20e960c8fa7d3ffffdfab24e9e2d73fb2a0ab78069cc8a33d7e1cfd42d5a8e1'),
}
EVIDENCE_SHA = 'b40d7c5af1f8b492256e0a4bdb52e0004eff351df0c8c0dde726c2e61e529e38'


def source_hashes():
    paths = [p for folder in ('clock_readout', 'coupled_clock', 'phase_sync', 'phase_sync_fit')
             for p in sorted((ROOT / 'experiments' / folder).glob('*.py'))]
    paths += [HERE / 'PROTOCOL-v1.md', ROOT / 'experiments/clock_evidence/diagnostic.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def parent_reports():
    reports = {}
    for name, (folder, expected) in PARENTS.items():
        path = ROOT / 'experiments' / folder / 'results-v1.json'
        require(sha(path) == expected, 'parent report identity changed')
        reports[name] = json.loads(path.read_bytes())
    require(sha(ROOT / 'experiments/clock_evidence/results-v1.json') == EVIDENCE_SHA,
            'motivating evidence identity changed')
    return reports


def make_plan():
    manifest_path = ROOT / 'experiments/coupled_clock/inputs-v1.json'
    require(sha(manifest_path) == INPUT_SHA, 'input manifest identity changed')
    population = json.loads(manifest_path.read_bytes())['cases']
    reports = parent_reports()
    identity = [(r['id'], r['role'], r['work'], r['frames']) for r in population]
    for report in reports.values():
        require([(r['id'], r['role'], r['work'], r['frames']) for r in report['cases']] == identity,
                'parent population differs')
    cost_path = ROOT / 'experiments/phase_sync/cost-cpu-v1.json'
    cost = json.loads(cost_path.read_bytes())
    require(cost['cpu_gate_pass'] is True and cost['cpu_gate_applies'] is True
            and all(sha(ROOT / name) == expected for name, expected in cost['source_sha256'].items()),
            'synchronizer component readiness changed')
    torch.manual_seed(SEED)
    return dict(schema='rhythm-map.phase-sync-plan.v1', inputs_sha256=INPUT_SHA,
                parent_report_sha256={k: v[1] for k, v in PARENTS.items()},
                evidence_sha256=EVIDENCE_SHA, component_cost_sha256=sha(cost_path),
                source_sha256=source_hashes(), population=population,
                initial_state_sha256=state_hash(PhaseSyncReadout()),
                objective='circular_phase_plus_native_multilag_log_count',
                seed=SEED, epochs=20, updates=100, seconds_per_fit=1800,
                temporal_controls=['zero', 'time_mean', 'half_roll'],
                training=False, holdout_access=False, production_change=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT),
            'new private plan required')
    write_json(args.output, make_plan())
    print('PHASE_SYNC_PLAN_READY ' + sha(args.output), flush=True)


if __name__ == '__main__':
    main()
