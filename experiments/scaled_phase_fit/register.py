"""Register a new bounded fit without touching music packets or old weights."""
import argparse
import json
import os
from pathlib import Path
import platform

import numpy as np
import torch

from experiments.coupled_clock.clock import require
from experiments.coupled_clock.prepare import ROOT, sha, write_json
from experiments.phase_sync_fit import register as previous
from experiments.observed_adjoint.benchmark import cpu_gate, source_hashes as cost_sources

HERE = Path(__file__).resolve().parent
PINNED = {
    'experiments/phase_sync_fit/plan-v1.json': '4427fbbcbfee04da501b00d2aa0c41e59dbb47c3f396a26c4162ee9a95c772a7',
    'experiments/phase_sync_fit/results-v1.json': '22326b1fd11ffd9e3ba81e5e19f6f6a9d71af9e83d5403a72e192d2e736b8f95',
    'experiments/phase_sync_fit/failure-v1.json': 'e4fae94444703430d500a65762ae6f069283104d2a3f3407520fcb2d603c18c1',
    'experiments/phase_sync_fit/execution-v1.json': '290a0065ab86e4c9cb79161904eae21e89fd058acf6e4664c035b48970a9fff1',
    'experiments/phase_sync_fit/journal-v1.jsonl': 'c6ec2a37739da3086c68fb1ed95c1e2712011afe64753d79703d6200deb6ddf0',
    'experiments/observed_adjoint/cost-cpu-v1.json': 'c27c4ad29e7658b1fe43e7473c1668a69d1b32fd78806244a63e471ac88b3d11',
    'experiments/observed_adjoint/cost-cuda-v1.json': '6a675e6ba88ca331d3d63a07b6d6111eeac4cc5cc4f67406dfc9e5c0bbfa73ec',
}


def configure(device):
    require(device in ('cpu', 'cuda'), 'explicit CPU/CUDA target required')
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda':
        require(torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
                'CUDA deterministic workspace unavailable')


def runtime_identity(device):
    return dict(device=device, torch=str(torch.__version__), numpy=np.__version__,
                python=platform.python_version(), os=platform.system(), machine=platform.machine(),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
                cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark)


def source_hashes():
    paths = {ROOT / name for name in previous.source_hashes()}
    paths.update(ROOT / name for name in cost_sources())
    for folder in ('training_observer', 'scaled_adjoint', 'observed_adjoint', 'scaled_phase_fit'):
        paths.update((ROOT / 'experiments' / folder).glob('*.py'))
    paths.add(HERE / 'PROTOCOL-v1.md')
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def prerequisites():
    require(all(sha(ROOT / name) == value for name, value in PINNED.items()),
            'frozen failure or numerical prerequisite changed')
    old_plan = json.loads((ROOT / 'experiments/phase_sync_fit/plan-v1.json').read_bytes())
    require(old_plan['source_sha256'] == previous.source_hashes(), 'closed fit source changed')
    cpu = json.loads((ROOT / 'experiments/observed_adjoint/cost-cpu-v1.json').read_bytes())
    require(cpu['source_sha256'] == cost_sources()
            and cpu_gate(cpu['cases'], cpu['process_peak_rss_bytes'], cpu['contract'], cpu['filesystem']),
            'observed transport cost prerequisite failed')
    return PINNED.copy()


def make_plan(device):
    configure(device)
    pins = prerequisites()
    old = previous.make_plan()
    return dict(schema='rhythm-map.scaled-phase-plan.v1', inputs_sha256=old['inputs_sha256'],
                parent_report_sha256=old['parent_report_sha256'], population=old['population'],
                source_sha256=source_hashes(), prerequisite_sha256=pins,
                runtime=runtime_identity(device), initial_state_sha256=old['initial_state_sha256'],
                seed=142, epochs=20, updates=100, seconds_per_fit=1800, batch_recordings=4,
                optimizer=dict(name='AdamW', learning_rate=.001, weight_decay=.0001, max_norm=1.),
                objective=old['objective'], temporal_controls=old['temporal_controls'],
                recorder='experiments.scaled_phase_fit.recorder.MusicalRecorder',
                transport='experiments.observed_adjoint.transport.ObservedTransport',
                complete_record_gradients=400, fits=2, watchdog_seconds=3900,
                previous_run_closed=True, automatic_retry=False, training=False,
                encoder_inference=False, holdout_access=False, production_change=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'new private plan required')
    write_json(args.output, make_plan(args.device))
    print('SCALED_PHASE_PLAN_READY ' + sha(args.output), flush=True)


if __name__ == '__main__':
    main()
