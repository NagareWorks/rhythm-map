"""Bounded, independently reset synthetic updates including real snapshot I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch

from experiments.observed_adjoint.fixtures import fixture
from experiments.observed_adjoint.recorder import ScaledFitRecorder
from experiments.training_observer.recorder import cpu_copy

ROOT = Path(__file__).resolve().parents[2]
SOURCES = tuple('experiments/' + name for name in (
    'observed_adjoint/benchmark.py', 'observed_adjoint/contract-v1.json',
    'observed_adjoint/fixtures.py', 'observed_adjoint/transport.py', 'observed_adjoint/recorder.py',
    'training_observer/recorder.py', 'training_observer/phase_trace.py',
    'scaled_adjoint/transport.py', 'scaled_adjoint/scaled.py', 'scaled_adjoint/scan.py',
    'phase_sync/model.py', 'phase_sync/scan.py', 'coupled_clock/model.py',
    'coupled_clock/clock.py', 'coupled_clock/supervision.py', 'clock_readout/model.py',
))


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def state_hash(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode('utf-8'))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cpu_gate(cases, rss_bytes, contract, filesystem):
    return (filesystem not in (None, 'tmpfs', 'ramfs') and rss_bytes is not None
            and 0 < rss_bytes < contract['cpu_process_peak_rss_limit_bytes']
            and set(cases) == set(contract['cases'])
            and all(0 < row['median_seconds'] < contract['cpu_median_observed_update_limit_seconds']
                    for row in cases.values())
            and all(len(row['updates']) == contract['iterations']
                    and all(update['completed_updates'] == 1 and len(update['bands']) == contract['records_per_update']
                            for update in row['updates']) for row in cases.values())
            and all(min(row['bands']) >= contract['opposed_minimum_vjp_bands_per_record']
                    for row in cases['opposed']['updates']))


def run(device, output):
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('available CPU/CUDA device required')
    contract = json.loads((ROOT / 'experiments/observed_adjoint/contract-v1.json').read_bytes())
    hashes = source_hashes()
    torch.set_num_threads(contract['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    filesystem = (subprocess.check_output(['stat', '-f', '-c', '%T', str(output)], text=True).strip()
                  if sys.platform.startswith('linux') else None)
    if filesystem in ('tmpfs', 'ramfs'):
        raise ValueError('snapshot I/O benchmark requires the declared host disk mount')
    cases = {}
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for kind in contract['cases']:
        model, rows = fixture(kind, contract['cells_per_record'], contract['seed'], device)
        initial = cpu_copy(model.state_dict())
        initial_sha = state_hash(initial)
        features_sha = hashlib.sha256(rows[0]['payload']['features'].cpu().numpy().tobytes()).hexdigest()
        offsets_sha = (hashlib.sha256(model.fixed_offsets.cpu().numpy().tobytes()).hexdigest()
                       if kind == 'opposed' else None)
        directory = output / kind
        directory.mkdir()
        times, updates = [], []
        for index in range(contract['warmups'] + contract['iterations']):
            # Independent one-step probes, NOT successive training epochs or a
            # retry after failure. A caught exception terminates the whole run.
            model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=contract['learning_rate'],
                                          weight_decay=contract['weight_decay'])
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            observer = ScaledFitRecorder(directory / str(index), model, optimizer,
                dict(protocol=contract, source_sha256=hashes, kind=kind, feature_sha256=features_sha,
                     fixed_offsets_sha256=offsets_sha, initial_state_sha256=initial_sha,
                     probe=index, musical_data=False), max_updates=1)
            observer.update(1, 0, rows, clip_norm=contract['max_norm'])
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            bands = [record['neural_vjp_passes'] for record in observer.records]
            if kind == 'opposed' and min(bands) < contract['opposed_minimum_vjp_bands_per_record']:
                raise ValueError('authored pressure did not exercise the registered band coverage')
            if index >= contract['warmups']:
                times.append(elapsed)
                updates.append(dict(bands=bands, completed_updates=observer.completed_updates,
                                    clip_report=observer.clip_report,
                                    snapshot_bytes=sum(path.stat().st_size for path in observer.output.iterdir())))
        cases[kind] = dict(seconds=times, median_seconds=statistics.median(times), updates=updates,
                           feature_sha256=features_sha, fixed_offsets_sha256=offsets_sha,
                           initial_state_sha256=initial_sha)
    if hashes != source_hashes():
        raise ValueError('benchmark sources changed')
    rss = None
    if sys.platform.startswith('linux'):
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    applies = device == 'cpu' and sys.platform.startswith('linux')
    return dict(schema=contract['schema'], contract=contract, source_sha256=hashes, device=device,
                cases=cases, process_peak_rss_bytes=rss, filesystem=filesystem,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                cpu_gate_applies=applies, cpu_gate_pass=cpu_gate(cases, rss, contract, filesystem) if applies else None,
                optimizer_steps=len(cases) * (contract['warmups'] + contract['iterations']),
                musical_accuracy_measured=False, automatic_retry=False,
                environment=dict(os=platform.system(), machine=platform.machine(), python=platform.python_version(),
                                 torch=str(torch.__version__), gpu=torch.cuda.get_device_name() if device == 'cuda' else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('new private output outside repository required')
    output.mkdir(exist_ok=False)
    report = run(args.device, output)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(device=report['device'], cpu_gate_pass=report['cpu_gate_pass'],
                         medians={kind: row['median_seconds'] for kind, row in report['cases'].items()},
                         process_peak_rss_bytes=report['process_peak_rss_bytes'], filesystem=report['filesystem'])))
    if report['cpu_gate_pass'] is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
