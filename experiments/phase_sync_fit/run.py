"""One pre-registered synchronizer fit pair; no tuning after observing results."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.coupled_clock.clock import require
from experiments.coupled_clock.supervision import clock_loss
from experiments.coupled_clock.run import (
    SEED, EPOCHS, SECONDS, BATCH, KINDS, state_hash, work_weights, work_mean,
    input_tensor, development_loss, journal, load_records,
)
from experiments.coupled_clock.measurement import (
    KEYS, clock_fields, independent_fields, measure, macro, regressions, gates,
)
from experiments.clock_evidence.diagnostic import intervention
from experiments.phase_sync.model import PhaseSyncReadout, PARAMETERS
from experiments.phase_sync_fit.register import (
    ROOT, INPUT_SHA, PARENTS, sha, write_json, source_hashes, make_plan, parent_reports,
)

CONTROL_KINDS = ('same_zero_common', 'time_mean_common', 'half_roll_common')
ALL_KINDS = KINDS + ('coupled_common', 'trajectory_common') + CONTROL_KINDS


# Preserve the coupled-v1 optimizer/selection loop, changing only construction.
def fit(records, device, zero_audio, output):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = PhaseSyncReadout().to(device)
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
                result = clock_loss(model(input_tensor(row, zero_audio)), row['target'][None], row['mask'][None])
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


def same_support(row, parent):
    """No new metric denominator may make a comparison look better."""
    for kind in ('audio_common', 'raw_common', 'v1_common'):
        for key in ('reference_points', 'reference_cells', 'reference_4s_intervals'):
            require(row[kind][key] == parent[kind][key], 'parent measurement support differs')
    for kind in ('raw_common', 'v1_common'):
        for key in KEYS:
            a, b = row[kind][key], parent[kind][key]
            require((a is None and b is None) or
                    (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10)),
                    'fixed raw/v1 comparator differs')


def evaluate(models, records, device, output, parents):
    before = {name: state_hash(model) for name, model in models.items()}
    rows = []
    for index, row in enumerate(records):
        common = row['valid'] & row['raw_valid']
        result = dict(id=row['id'], role=row['role'], work=row['work'], frames=row['frames'],
                      packet_sha256=row['packet_sha256'])
        predictions, fields = {}, {}
        for name, model, variant in (
                ('audio', models['audio'], 'natural'), ('zero', models['zero'], 'zero'),
                ('same_zero', models['audio'], 'zero'),
                ('time_mean', models['audio'], 'time_mean'),
                ('half_roll', models['audio'], 'half_roll')):
            x = torch.from_numpy(intervention(row['hidden'], variant)).to(device)[None]
            with torch.inference_mode():
                # Actual model forward defines prediction. A second inexpensive
                # CNN pass retains local fields for later read-only attribution.
                prediction = model(x).cycles[0].cpu().numpy()
                local = model.fields(x)[0].cpu().numpy()
            require(np.isfinite(prediction).all() and (np.diff(prediction) > 0).all()
                    and np.isfinite(local).all(), 'invalid complete prediction')
            predictions[name], fields['fields_' + name] = prediction, local
            if name in ('audio', 'zero'):
                result[name] = measure(clock_fields(prediction), row['reference'], row['valid'])
            result[name + '_common'] = measure(clock_fields(prediction), row['reference'], common)
        result['raw_common'] = measure(clock_fields(row['raw']), row['reference'], common)
        result['v1_common'] = measure(independent_fields(row['v1']), row['reference'], common)
        for name, parent in parents.items():
            old = parent['cases'][index]
            require((row['id'], row['role'], row['work'], row['frames']) ==
                    (old['id'], old['role'], old['work'], old['frames']), 'parent population differs')
            same_support(result, old)
            result[name + '_common'] = old['audio_common']
        for baseline in ('raw_common', 'v1_common', 'coupled_common', 'trajectory_common'):
            result['regressions_vs_' + baseline] = regressions(result, baseline)
        result['natural_better_than_control'] = {
            kind: {key: (None if result[kind][key] is None or result['audio_common'][key] is None
                         else result['audio_common'][key] < result[kind][key]) for key in KEYS}
            for kind in CONTROL_KINDS}
        path = output / (row['id'] + '.prediction.npz')
        with path.open('xb') as stream:
            np.savez(stream, **predictions, **fields)
        result['prediction_sha256'] = sha(path)
        rows.append(result)
        journal(output, dict(event='recording_evaluated', recording=index + 1, total=len(records)))
    require(before == {name: state_hash(model) for name, model in models.items()},
            'evaluation changed weights')
    return rows


def comparison_gate(rows, summary, fits):
    result = gates(rows, summary, fits)
    # Preserve the original gate and make this protocol's exact budget explicit.
    result['finished'] &= all(f['updates'] == 100 and np.isfinite(f['elapsed_s'])
                              and 0 <= f['elapsed_s'] <= SECONDS for f in fits)
    for parent in PARENTS:
        result['previous_' + parent + '_nonregression'] = all(
            summary[role]['audio_common'][key] is not None
            and summary[role][parent + '_common'][key] is not None
            and np.isfinite(summary[role]['audio_common'][key])
            and np.isfinite(summary[role][parent + '_common'][key])
            and summary[role]['audio_common'][key] <= summary[role][parent + '_common'][key]
            for role in ('development', 'diagnostic') for key in KEYS)
    required = [r for r in rows if r['role'] in ('development', 'diagnostic')]
    result['temporal_controls_complete'] = bool(required) and all(
        row[kind][key] is not None and np.isfinite(row[kind][key])
        for row in required for kind in CONTROL_KINDS for key in KEYS)
    key = 'phase_mean_absolute_cycles'
    result['temporal_phase_dependence'] = all(
        summary[role]['audio_common'][key] is not None and summary[role][kind][key] is not None
        and np.isfinite(summary[role]['audio_common'][key]) and np.isfinite(summary[role][kind][key])
        and summary[role]['audio_common'][key] < summary[role][kind][key]
        for role in ('development', 'diagnostic') for kind in CONTROL_KINDS)
    result['passed'] = all(value for key, value in result.items() if key != 'passed')
    return {k: bool(v) for k, v in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'plan', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT),
            'new private output required')
    require(sha(args.plan) == args.expected_plan_sha256, 'plan identity changed')
    plan = json.loads(args.plan.read_bytes())
    require(plan == make_plan(), 'registered source, population, initialization or protocol changed')
    require(sha(args.inputs / 'inputs.json') == INPUT_SHA, 'private input identity changed')
    manifest = json.loads((args.inputs / 'inputs.json').read_bytes())
    parents = parent_reports()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda':
        require(torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
                'CUDA/determinism unavailable')
        torch.cuda.set_per_process_memory_fraction(.4)
    args.output.mkdir()
    write_json(args.output / 'execution.json', dict(schema='rhythm-map.phase-sync-execution.v1',
        plan_sha256=args.expected_plan_sha256, inputs_sha256=INPUT_SHA,
        parent_report_sha256=plan['parent_report_sha256'], source_sha256=plan['source_sha256'],
        initial_state_sha256=plan['initial_state_sha256'], torch_version=torch.__version__,
        numpy_version=np.__version__, device=args.device, deterministic=True, tf32=False,
        fits=2, seed=SEED, max_epochs=EPOCHS, max_seconds_per_fit=SECONDS,
        batch_recordings=BATCH, parameters=PARAMETERS, objective=plan['objective'],
        controls=list(CONTROL_KINDS), encoder_inference=False, holdout_access=False, production_change=False))
    try:
        records = load_records(args.inputs, manifest, args.device)
        audio, audio_fit = fit(records, args.device, False, args.output)
        zero, zero_fit = fit(records, args.device, True, args.output)
        require(audio_fit['initial_state_sha256'] == zero_fit['initial_state_sha256'] ==
                plan['initial_state_sha256'], 'control initialization differs')
        rows = evaluate(dict(audio=audio, zero=zero), records, args.device, args.output, parents)
        summary = {role: {kind: macro([r for r in rows if r['role'] == role], kind) for kind in ALL_KINDS}
                   for role in ('fit', 'development', 'diagnostic')}
        gate = comparison_gate(rows, summary, [audio_fit, zero_fit])
        require(plan['source_sha256'] == source_hashes() and plan == make_plan(),
                'registered sources or parents changed during execution')
        report = dict(schema='rhythm-map.phase-sync-learning.v1',
            plan_sha256=args.expected_plan_sha256, inputs_sha256=INPUT_SHA,
            parent_report_sha256=plan['parent_report_sha256'],
            execution_sha256=sha(args.output / 'execution.json'), fits=[audio_fit, zero_fit],
            cases=rows, summary=summary, gate=gate, training=True, optimizer_fits=2,
            checkpoint_sequence_evaluations=5 * len(records), independent_acceptance=False,
            encoder_inference=False, holdout_access=False, production_change=False,
            selected_gain={kind: float(math.log(2.) * torch.sigmoid(model.gain_logit.detach().double()))
                           for kind, model in (('audio', audio), ('zero', zero))},
            decision='fresh_independent_validation_required' if gate['passed']
            else 'close_fixed_phase_sync_fit_no_automatic_sweep')
        write_json(args.output / 'report.json', report)
        journal(args.output, dict(event='experiment_complete', gate=gate))
    except Exception as error:
        write_json(args.output / 'failure.json', dict(event='experiment_failed',
            exception_type=type(error).__name__, automatic_retry=False, production_change=False))
        raise


if __name__ == '__main__':
    main()
