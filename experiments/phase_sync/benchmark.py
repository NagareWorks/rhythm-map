"""Fixed complete-crop cost check. No optimizer, checkpoint, or music input."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import torch
from torch.nn import functional as F

from experiments.clock_readout.model import ClockReadout, HALO
from experiments.coupled_clock.model import CoupledClockReadout
from experiments.phase_sync.model import PhaseSyncReadout

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    'experiments/phase_sync/scan.py', 'experiments/phase_sync/model.py',
    'experiments/phase_sync/benchmark.py', 'experiments/phase_sync/contract-v1.json',
    'experiments/clock_readout/model.py', 'experiments/coupled_clock/model.py',
    'experiments/coupled_clock/clock.py',
)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def cpu_gate(median_seconds, rss_bytes, contract):
    return (rss_bytes is not None
            and 0 < median_seconds < contract['cpu_median_forward_backward_limit_seconds']
            and 0 < rss_bytes < contract['cpu_process_peak_rss_limit_bytes'])


def run(device):
    contract = json.loads((ROOT / 'experiments/phase_sync/contract-v1.json').read_text())
    hashes_before = source_hashes()
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    torch.set_num_threads(contract['cpu_threads'])
    torch.manual_seed(contract['seed'])
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    features = torch.randn(contract['batch'], contract['cells'] + 1,
                           contract['feature_width']) * .1
    feature_hash = hashlib.sha256(features.numpy().tobytes()).hexdigest()
    features = features.to(device)
    models = {}
    for name, cls in (('direct', ClockReadout), ('integrated', CoupledClockReadout),
                      ('synchronized', PhaseSyncReadout)):
        torch.manual_seed(contract['seed'])
        models[name] = cls().to(device).train()
    samples = {name: [] for name in models}

    def step(name):
        model = models[name]
        model.zero_grad(set_to_none=True)
        if name == 'direct':
            # Same physical-edge CNN padding and owned full sequence as the
            # integrated heads, not a shorter receptive field at the edges.
            padded = F.pad(features.transpose(1, 2), (HALO, HALO)).transpose(1, 2)
            fields = model(padded)[:, HALO:-HALO]
            period, vector = fields[:, :-1, 0].double(), fields[..., 1:3].double()
        else:
            clock = model(features)
            period, vector = clock.log2_period, clock.phase_vector
        # Authored target (constant unit phase, log-period zero), cost only.
        loss = period.square().mean() + (vector[..., 0] - 1).square().mean() + vector[..., 1].square().mean()
        loss.backward()
        if not torch.isfinite(loss).item() or any(
                p.grad is None or not torch.isfinite(p.grad).all().item() for p in model.parameters()):
            raise ValueError('nonfinite or missing benchmark gradient')

    for _ in range(contract['warmups']):
        for name in models:
            step(name)
    for _ in range(contract['iterations']):
        for name in models:
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            step(name)
            if device == 'cuda':
                torch.cuda.synchronize()
            samples[name].append(time.perf_counter() - start)
    # Linux ru_maxrss is KiB; it includes ALL heads and imported runtimes in this
    # process, not only tensor allocations or incremental scan memory.
    rss_bytes = None
    if sys.platform.startswith('linux'):
        import resource
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    summaries = {name: {'seconds': values, 'median_seconds': statistics.median(values),
                        'parameters': sum(p.numel() for p in models[name].parameters())}
                 for name, values in samples.items()}
    sync_median = summaries['synchronized']['median_seconds']
    hashes_after = source_hashes()
    if hashes_after != hashes_before:
        raise ValueError('source changed during benchmark')
    applies = device == 'cpu' and sys.platform.startswith('linux')
    return {
        'schema': contract['schema'], 'contract': contract, 'source_sha256': hashes_before,
        'feature_sha256': feature_hash, 'device': device,
        'environment': {'os': platform.system(), 'machine': platform.machine(),
                        'python': platform.python_version(), 'torch': torch.__version__,
                        'cuda_runtime': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name() if device == 'cuda' else None},
        'models': summaries, 'process_peak_rss_bytes': rss_bytes,
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
        'synchronized_over_direct_ratio': sync_median / summaries['direct']['median_seconds'],
        'synchronized_over_integrated_ratio': sync_median / summaries['integrated']['median_seconds'],
        'cpu_gate_applies': applies,
        'cpu_gate_pass': cpu_gate(sync_median, rss_bytes, contract) if applies else None,
        'musical_accuracy_measured': False, 'optimizer_steps': 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Refuse overwrite before doing work; exclusive creation also closes races.
    with args.output.open('x', encoding='utf-8') as handle:
        result = run(args.device)
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'device': result['device'], 'models': result['models'],
                      'process_peak_rss_bytes': result['process_peak_rss_bytes'],
                      'cpu_gate_pass': result['cpu_gate_pass']}))
    if result['cpu_gate_pass'] is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
