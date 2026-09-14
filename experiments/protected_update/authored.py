"""One fixed forty-step synthetic observation through the durable recorder."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from experiments.coupled_clock.supervision import clock_loss
from experiments.gradient_routing.authored import teacher_case
from experiments.protected_update.recorder import ProtectedFitRecorder

ROOT = Path(__file__).resolve().parents[2]


def sources():
    paths = [p for p in (ROOT / 'experiments').rglob('*.py')
             if not p.is_relative_to(Path(__file__).parent / 'checks')]
    paths.append(Path(__file__).with_name('PROTOCOL-v1.md'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def run(output, device='cpu'):
    start, identity = time.perf_counter(), sources()
    model, payload = teacher_case(device)
    observer = ProtectedFitRecorder(output, model,
        torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001),
        dict(protocol='protected-update-authored-v1', source_sha256=identity), max_updates=40)
    def losses():
        with torch.no_grad():
            values = clock_loss(model(payload['features']), payload['reference'], payload['valid'])
            return {k: float(values[k]) for k in ('phase', 'advance', 'total')}
    initial, rows = losses(), []
    previous = initial
    for step in range(40):
        observer.update(1, step, [dict(id='authored-teacher', scale=1., payload=payload)])
        current = losses()
        rows.append(dict(step=step + 1, losses=current, receipt=observer.optimizer.receipt,
                         finite_count_increased=current['advance'] > previous['advance']))
        previous = current
    actions = Counter(r['action'] for row in rows for r in row['receipt'].values())
    gate = (all(previous[k] < initial[k] for k in ('phase', 'advance')) and
            all(r['final_sign'] is None or r['final_sign'] <= 0 for row in rows for r in row['receipt'].values()))
    result = dict(schema='protected-update-authored-v1', source_sha256=identity, device=device,
        runtime=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__),
        initial=initial, final=previous, steps=rows, actions=dict(actions), gate_passed=gate,
        returned_calls=observer.completed_updates, committed_steps=observer.optimizer.committed_steps,
        moved_steps=observer.optimizer.moved_steps, elapsed_seconds=time.perf_counter() - start,
        musical_updates=0, admission=False)
    if platform.system() == 'Linux':
        import resource
        result['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    else:
        result['peak_rss_mib'] = None
    assert sources() == identity
    path = Path(output) / 'report.json'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('steps', 'source_sha256')}, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.output, args.device)
