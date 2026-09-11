"""Predeclared complete-crop transport cost, not a fit or snapshot-I/O budget."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import torch

from experiments.phase_sync.model import PhaseSyncReadout
from experiments.phase_sync.benchmark import cpu_gate
from experiments.scaled_adjoint.scaled import accumulate
from experiments.scaled_adjoint.transport import record_gradient, install_clipped

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    'experiments/scaled_adjoint/benchmark.py', 'experiments/scaled_adjoint/contract-v1.json',
    'experiments/scaled_adjoint/scaled.py', 'experiments/scaled_adjoint/scan.py',
    'experiments/scaled_adjoint/transport.py', 'experiments/training_observer/phase_trace.py',
    'experiments/phase_sync/scan.py', 'experiments/phase_sync/model.py',
    'experiments/phase_sync/benchmark.py', 'experiments/clock_readout/model.py',
    'experiments/coupled_clock/model.py', 'experiments/coupled_clock/clock.py',
    'experiments/coupled_clock/supervision.py',
)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def run(device):
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('available cpu/cuda device required')
    contract = json.loads((ROOT / 'experiments/scaled_adjoint/contract-v1.json').read_bytes())
    hashes = source_hashes()
    torch.set_num_threads(contract['cpu_threads'])
    torch.manual_seed(contract['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    features = torch.randn(contract['batch'], contract['cells'] + 1, contract['feature_width']) * .1
    feature_hash = hashlib.sha256(features.numpy().tobytes()).hexdigest()
    features = features.to(device)
    torch.manual_seed(contract['seed'])
    model = PhaseSyncReadout().to(device).train()
    target = torch.arange(contract['cells'] + 1, device=device, dtype=torch.float64)[None] * .04
    valid = torch.ones_like(target, dtype=torch.bool)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    samples, details = [], []
    for index in range(contract['warmups'] + contract['iterations']):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        model.zero_grad(set_to_none=True)
        packet = record_gradient(model, features, target, valid)
        summary = install_clipped(model, accumulate([packet['gradient'].multiply(1.)]), contract['max_norm'])
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if index >= contract['warmups']:
            samples.append(elapsed)
            details.append(dict(loss=packet['loss'], neural_vjp_passes=packet['neural_vjp_passes'],
                                field_exponent=packet['field_exponent'], **summary))
    for name, value in model.state_dict().items():
        if not torch.equal(value, before[name]):
            raise ValueError('benchmark mutated model state')
    if hashes != source_hashes():
        raise ValueError('sources changed during benchmark')
    rss_bytes = None
    if sys.platform.startswith('linux'):
        import resource
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    applies = device == 'cpu' and sys.platform.startswith('linux')
    median = statistics.median(samples)
    return dict(schema=contract['schema'], contract=contract, source_sha256=hashes,
                device=device, feature_sha256=feature_hash, seconds=samples, median_seconds=median,
                iterations=details, parameters=sum(p.numel() for p in model.parameters()),
                process_peak_rss_bytes=rss_bytes,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                environment=dict(os=platform.system(), machine=platform.machine(), python=platform.python_version(),
                                 torch=str(torch.__version__), cuda_runtime=torch.version.cuda,
                                 gpu=torch.cuda.get_device_name() if device == 'cuda' else None),
                cpu_gate_applies=applies, cpu_gate_pass=cpu_gate(median, rss_bytes, contract) if applies else None,
                optimizer_steps=0, musical_accuracy_measured=False, recorder_io_measured=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError('benchmark output must be private and outside the repository')
    with args.output.open('x', encoding='utf-8') as handle:
        result = run(args.device)
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in ('device', 'median_seconds', 'process_peak_rss_bytes', 'cpu_gate_pass')}))
    if result['cpu_gate_pass'] is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
