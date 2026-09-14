"""Freeze a four-arm plan without opening music packets or trained weights."""
import argparse
import json
from pathlib import Path

import torch

from experiments.coupled_clock.clock import require
from experiments.coupled_clock.prepare import ROOT, sha, write_json
from experiments.coupled_clock.run import state_hash
from experiments.scaled_phase_fit.register import configure, runtime_identity
from experiments.separated_evidence.model import SharedEvidence, SeparatedEvidence

HERE = Path(__file__).resolve().parent
ARMS = ('shared-natural', 'shared-zero', 'separated-natural', 'separated-zero')
PINS = {
    'experiments/protected_phase_fit/plan-v1.json': 'a138707e2620542b520c97479e162b075f222ceee402b9ef9ed2ec584f24484a',
    'experiments/protected_phase_fit/results-v1.json': '3d33799c6046456c1da60611ee4100ea0e39f54ba1ae7fa43afe3f10f8abe79b',
    'experiments/protected_phase_fit/execution-v1.json': '1079261c20b353d4e8c1fef484a5dddcb9d0e02f2f684557a6d1523ecc4f662c',
    'experiments/separated_evidence/model.py': 'e78c59c0c565f58cffa78c5ef494975cb56373f6bb6c17c188ad75eedeca4f75',
    'experiments/separated_evidence/training.py': '1c12e20b34ff1254ca9d16c29a908c9a6b217b9bd944944fc7d765ef113a7f47',
    'experiments/separated_evidence/supervision.py': 'df88f2a9f3fb8cd895e5ba7bfb82c6d63ee2da78185d24a3561e452eaf8cabd5',
    'experiments/separated_evidence/PROTOCOL-v1.md': '40d7013b74ef74bc66b7c1d4b49202d2f8df22e8f50a425ccd8d2241bce7e9ee',
}


def sources():
    require(all(sha(ROOT / k) == v for k, v in PINS.items()), 'closed prerequisite changed')
    old = json.loads((ROOT / 'experiments/protected_phase_fit/plan-v1.json').read_bytes())
    require(all(sha(ROOT / k) == v for k, v in old['source_sha256'].items()), 'closed source changed')
    paths = set(old['source_sha256']) | set(PINS)
    for folder in (HERE, ROOT / 'experiments/separated_evidence'):
        paths.update(p.relative_to(ROOT).as_posix() for p in folder.glob('*.py'))
        paths.add((folder / 'PROTOCOL-v1.md').relative_to(ROOT).as_posix())
    return {k: sha(ROOT / k) for k in sorted(paths)}


def initial_models(device):
    torch.manual_seed(142)
    shared = SharedEvidence().to(device)
    return shared, SeparatedEvidence(shared)


def make_plan(device):
    configure(device)
    identity = sources()
    old = json.loads((ROOT / 'experiments/protected_phase_fit/plan-v1.json').read_bytes())
    manifest = ROOT / 'experiments/coupled_clock/inputs-v1.json'
    require(sha(manifest) == old['inputs_sha256'], 'input manifest changed')
    require(json.loads(manifest.read_bytes())['cases'] == old['population'], 'population changed')
    shared, separated = initial_models(device)
    return dict(schema='rhythm-map.separated-evidence-plan.v1', source_sha256=identity,
        prerequisite_sha256=dict(PINS), inputs_sha256=old['inputs_sha256'], population=old['population'],
        parent_report_sha256=old['parent_report_sha256'], runtime=runtime_identity(device),
        initial_state_sha256=dict(shared=state_hash(shared), separated=state_hash(separated)),
        initial_task_sha256={k: state_hash(getattr(separated, k)) for k in ('tempo', 'phase')},
        parameters=dict(shared=24003, separated=47907), arms=list(ARMS), seed=142,
        export_parameters=dict(shared=47907, separated=47907), export_trunk_evaluations=2,
        epochs=20, paired_updates=100, batch_recordings=4, seconds_per_arm=1800, watchdog_seconds=7500,
        optimizer=dict(name='AdamW', learning_rate=.001, weight_decay=.0001, max_norm=1.),
        selection='per_task_original_loss_work_macro_complete_epoch_earliest_ties_both_architectures',
        objective='native_cell_log2_period_MSE_and_frame_cos_sin_MSE',
        optimizer_calls=dict(shared=100, separated=200), evidence_pair_evaluations=400,
        old_weights_loaded=False, automatic_retry=False, training=False, encoder_inference=False,
        holdout_access=False, coherent_clock=False, production_change=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'new private plan required')
    write_json(args.output, make_plan(args.device))
    print('SEPARATED_EVIDENCE_PLAN_READY ' + sha(args.output), flush=True)


if __name__ == '__main__':
    main()
