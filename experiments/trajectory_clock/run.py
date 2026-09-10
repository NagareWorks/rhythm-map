"""Loss-only ablation with a frozen copy of the coupled-v1 optimization loop."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.coupled_clock.clock import require
from experiments.coupled_clock.model import CoupledClockReadout, PARAMETERS
from experiments.coupled_clock.run import (
    SEED, EPOCHS, SECONDS, BATCH, KINDS, state_hash, work_weights, work_mean,
    input_tensor, journal, load_records, evaluate,
)
from experiments.coupled_clock.measurement import KEYS, macro, regressions, gates
from experiments.trajectory_clock.attribution import ROOT, INPUT_SHA, REPORT_SHA, sha, write_json
from experiments.trajectory_clock.register import HERE, source_hashes
from experiments.trajectory_clock.supervision import trajectory_loss

# Keep these two loops text-equivalent to the closed coupled-v1 loops except
# the objective call. This explicit snapshot preserves that old experiment;
# there is no global monkey-patch, independent-window reset or optimizer tweak.

def development_loss(model, records, zero_audio):
    totals = {}
    model.eval()
    with torch.inference_mode():
        for row in records:
            if row['role'] != 'development':
                continue
            result = trajectory_loss(model(input_tensor(row, zero_audio)), row['target'][None], row['mask'][None])
            totals.setdefault(row['work'], []).append(float(result['total']))
    require(len(totals) == 3, 'development work split changed')
    return work_mean(totals)


def fit(records, device, zero_audio, output):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = CoupledClockReadout().to(device)
    initial = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    population = [r for r in records if r['role'] == 'fit']
    weights = work_weights(population)
    require(len(population) == 20 and len(weights) == 9, 'fit work split changed')
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    history, best, best_epoch, best_loss, updates = [], None, None, float('inf'), 0
    exhausted = False
    name = 'zero-audio' if zero_audio else 'audio'
    journal(output, dict(event='fit_started', kind=name, initial_state_sha256=initial))
    for epoch in range(EPOCHS):
        model.train()
        order = np.random.default_rng(SEED + epoch).permutation(len(population))
        training = {}
        for offset in range(0, len(order), BATCH):
            batch = [population[i] for i in order[offset:offset + BATCH]]
            optimizer.zero_grad(set_to_none=True)
            for row in batch:
                if time.monotonic() - started >= SECONDS:
                    exhausted = True
                    break
                # One full physical crop and one clock anchor per graph. Separate
                # forwards avoid variable-length padding changing real boundaries.
                result = trajectory_loss(model(input_tensor(row, zero_audio)), row['target'][None], row['mask'][None])
                loss = result['total'] * weights[row['work']] / len(batch)
                loss.backward()
                training.setdefault(row['work'], []).append(float(result['total'].detach()))
            if exhausted:
                break  # Discard incomplete accumulated gradients; no partial update.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
        if exhausted:
            break
        score = development_loss(model, records, zero_audio)
        if time.monotonic() - started >= SECONDS:
            exhausted = True
            break
        item = dict(epoch=epoch + 1, training_work_macro_loss=work_mean(training), development_loss=score, updates=updates)
        history.append(item)
        if score < best_loss:  # Earlier checkpoint wins exact ties.
            best, best_epoch, best_loss = copy.deepcopy(model.state_dict()), epoch + 1, score
        journal(output, dict(event='epoch', kind=name, **item))
    require(best is not None, 'no complete checkpoint before budget exhausted')
    model.load_state_dict(best)
    model.eval()
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    path = output / (name + '.weights.npz')
    with path.open('xb') as stream:
        np.savez(stream, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    report = dict(kind=name, parameters=PARAMETERS, initial_state_sha256=initial, seed=SEED,
                  epochs=len(history), updates=updates, budget_exhausted=exhausted,
                  selected_epoch=best_epoch, development_loss=best_loss, elapsed_s=elapsed,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                  weight_sha256=sha(path), history=history)
    write_json(output / (name + '.fit.json'), report)
    return model, report


def comparison_gate(rows, summary, fits):
    result = gates(rows, summary, fits)
    previous = all(
        summary[role]['audio_common'][key] is not None and summary[role]['coupled_common'][key] is not None
        and summary[role]['audio_common'][key] <= summary[role]['coupled_common'][key]
        for role in ('development', 'diagnostic') for key in KEYS)
    result['previous_coupled_nonregression'] = bool(previous)
    result['passed'] = all(value for key, value in result.items() if key != 'passed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'plan', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'new private output required')
    require(sha(args.plan) == args.expected_plan_sha256, 'plan identity changed')
    plan = json.loads(args.plan.read_bytes())
    require(plan['source_sha256'] == source_hashes(), 'registered code or protocol changed')
    require(plan['parent_inputs_sha256'] == INPUT_SHA == sha(args.inputs / 'inputs.json'), 'input identity changed')
    prior_path = ROOT / 'experiments/coupled_clock/results-v1.json'
    require(plan['parent_report_sha256'] == REPORT_SHA == sha(prior_path), 'parent report changed')
    require(plan['attribution_sha256'] == sha(HERE / 'attribution-v1.json'), 'attribution identity changed')
    manifest = json.loads((args.inputs / 'inputs.json').read_bytes())
    require(plan['population'] == manifest['cases'] and not plan['holdout_access'], 'registered population changed')
    prior = json.loads(prior_path.read_bytes())
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda':
        require(torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUDA/determinism unavailable')
        torch.cuda.set_per_process_memory_fraction(.4)
    torch.manual_seed(SEED)
    require(state_hash(CoupledClockReadout()) == prior['fits'][0]['initial_state_sha256'], 'initialization differs from parent')
    args.output.mkdir()
    write_json(args.output / 'execution.json', dict(schema='rhythm-map.trajectory-clock-execution.v1',
        plan_sha256=args.expected_plan_sha256, parent_inputs_sha256=INPUT_SHA, parent_report_sha256=REPORT_SHA,
        source_sha256=plan['source_sha256'], torch_version=torch.__version__, numpy_version=np.__version__,
        device=args.device, deterministic=True, tf32=False, fits=2, seed=SEED, max_epochs=EPOCHS,
        max_seconds_per_fit=SECONDS, batch_recordings=BATCH, parameters=PARAMETERS,
        objective='integer_origin_trajectory_mse', holdout_access=False, production_change=False))
    try:
        records = load_records(args.inputs, manifest, args.device)
        audio, audio_fit = fit(records, args.device, False, args.output)
        zero, zero_fit = fit(records, args.device, True, args.output)
        require(audio_fit['initial_state_sha256'] == zero_fit['initial_state_sha256'] ==
                prior['fits'][0]['initial_state_sha256'], 'control or parent initialization differs')
        rows = evaluate(dict(audio=audio, zero=zero), records, args.device, args.output)
        for row, previous in zip(rows, prior['cases']):
            require(row['id'] == previous['id'], 'comparison population differs')
            row['coupled_common'] = previous['audio_common']
            row['regressions_vs_coupled_common'] = regressions(row, 'coupled_common')
        summaries = {role: {kind: macro([r for r in rows if r['role'] == role], kind)
                           for kind in KINDS + ('coupled_common',)}
                     for role in ('fit', 'development', 'diagnostic')}
        gate = comparison_gate(rows, summaries, [audio_fit, zero_fit])
        report = dict(schema='rhythm-map.trajectory-clock-learning.v1', plan_sha256=args.expected_plan_sha256,
                      parent_inputs_sha256=INPUT_SHA, parent_report_sha256=REPORT_SHA,
                      execution_sha256=sha(args.output / 'execution.json'), fits=[audio_fit, zero_fit],
                      cases=rows, summary=summaries, gate=gate, training=True,
                      independent_acceptance=False, holdout_access=False, production_change=False,
                      decision='fresh_independent_validation_required' if gate['passed'] else 'close_fixed_trajectory_fit_no_automatic_sweep')
        write_json(args.output / 'report.json', report)
        journal(args.output, dict(event='experiment_complete', gate=gate))
    except Exception as error:
        write_json(args.output / 'failure.json', dict(event='experiment_failed', exception_type=type(error).__name__,
                                                     automatic_retry=False, production_change=False))
        raise


if __name__ == '__main__':
    main()
