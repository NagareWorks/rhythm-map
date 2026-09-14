"""Fixed synthetic component observation; never reads audio or trained weights."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from experiments.separated_evidence.model import SharedEvidence, SeparatedEvidence
from experiments.separated_evidence.supervision import losses
from experiments.separated_evidence.training import TaskTrainer
from experiments.separated_evidence.test_evidence import case
from experiments.coupled_clock.clock import require

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sources():
    paths = list(HERE.glob('*.py')) + [HERE / 'PROTOCOL-v1.md',
        ROOT / 'experiments/clock_readout/model.py', ROOT / 'experiments/coupled_clock/clock.py']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def run(output, device):
    require(device in ('cpu', 'cuda'), 'unknown device')
    require(not output.exists() and not output.resolve().is_relative_to(ROOT), 'new private output required')
    torch.set_num_threads(2)
    torch.manual_seed(142)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    shared = SharedEvidence().to(device)
    separated = SeparatedEvidence(shared)
    row = case(device)
    models = dict(shared=shared, separated=separated)
    trainers = {k: TaskTrainer(getattr(separated, k), max_updates=40) for k in ('tempo', 'phase')}
    optimizer = torch.optim.AdamW(shared.parameters(), lr=.001, weight_decay=.0001)
    identity = sources()
    output.mkdir()
    shared_steps, stage = 0, 'created'
    started = time.perf_counter()
    def scores():
        with torch.inference_mode():
            return {name: {k: float(v) for k, v in losses(model(row['features']), row['reference'], row['valid']).items()}
                    for name, model in models.items()}
    def snapshot(name):
        # All authored inputs, current task boundaries, gradients, moments and
        # RNG are owned. This is not a musical recorder/partial-epoch selection.
        payload = dict(source_sha256=identity, stage=stage, shared_steps=shared_steps,
            models={k: m.state_dict() for k, m in models.items()}, shared_optimizer=optimizer.state_dict(),
            tasks={k: dict(stage=t.stage, failed=t.failed, returned_updates=t.returned_updates,
                          optimizer=t.optimizer.state_dict()) for k, t in trainers.items()},
            gradients={k: {n: p.grad for n, p in m.named_parameters()} for k, m in models.items()},
            input=row, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else [],
            automatic_retry=False)
        temporary = output / (name + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(output / name)
    history = []
    try:
        initial = scores()
        for index in range(40):
            stage = 'before_pair'
            snapshot('before-update.pt')
            stage = 'shared_backward'
            optimizer.zero_grad(set_to_none=True)
            parts = losses(shared(row['features']), row['reference'], row['valid'])
            sum(parts.values()).backward()
            torch.nn.utils.clip_grad_norm_(shared.parameters(), 1., error_if_nonfinite=True)
            stage = 'shared_optimizer_entered'
            optimizer.step()
            shared_steps += 1
            stage = 'shared_optimizer_returned'
            require(all(torch.isfinite(p).all().item() for p in shared.parameters()) and
                    all(not isinstance(v, torch.Tensor) or torch.isfinite(v).all().item()
                        for state in optimizer.state.values() for v in state.values()),
                    'nonfinite returned shared state')
            receipts = {}
            for task, trainer in trainers.items():
                stage = task
                receipts[task] = trainer.update([row])
            stage = 'paired_update_complete'
            item = dict(step=index + 1, losses=scores(), receipts=receipts)
            require(all(np.isfinite(v) for arm in item['losses'].values() for v in arm.values()), 'nonfinite authored score')
            history.append(item)
            snapshot('after-update.pt')
            with (output / 'journal.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(item, allow_nan=False) + '\n')
        final = history[-1]['losses']
        passed = all(final['separated'][k] < initial['separated'][k] for k in ('tempo', 'phase'))
        require(identity == sources(), 'authored sources changed')
        report = dict(schema='separated-evidence-authored-v1', source_sha256=identity, device=device,
            runtime=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__),
            initial=initial, final=final, steps=history, shared_updates=shared_steps,
            task_updates={k: t.returned_updates for k, t in trainers.items()},
            parameters={k: sum(p.numel() for p in m.parameters()) for k, m in models.items()},
            elapsed_s=time.perf_counter() - started, gate_passed=passed, musical_updates=0,
            admission=False, coherent_clock=False, holdout_access=False, automatic_retry=False,
            snapshot_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.pt'))})
        with (output / 'report.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({k: v for k, v in report.items() if k not in ('steps', 'source_sha256', 'snapshot_sha256')}), flush=True)
        return report
    except BaseException as error:
        try:
            snapshot('failure.pt')
        except BaseException as secondary:
            error.add_note('Authored failure snapshot unavailable: ' + type(secondary).__name__)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    run(args.output, args.device)
