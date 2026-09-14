"""Execute one fixed four-arm evidence comparison, with no resume or decoder."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from experiments.coupled_clock.clock import require
from experiments.coupled_clock.run import state_hash, work_weights, work_mean, load_records, journal
from experiments.phase_sync_fit.register import parent_reports
from experiments.separated_evidence.supervision import losses
from experiments.separated_evidence_fit.register import ROOT, ARMS, sha, write_json, make_plan, initial_models
from experiments.separated_evidence_fit.recorder import Recorder, TASKS, branch
from experiments.separated_evidence_fit.evaluation import evaluate, summarize, compare


def packet(row, zero, scale=1.):
    x = row['features'][None]
    return dict(id=row['id'], work=row['work'], scale=scale,
                features=torch.zeros_like(x) if zero else x, reference=row['target'][None], valid=row['mask'][None])


def development(model, records, zero, recorder):
    groups = {k: {} for k in TASKS}
    for row in records:
        if row['role'] != 'development':
            continue
        record = packet(row, zero)
        recorder.inference_stage('development', [record])
        with torch.inference_mode():
            result = losses(model(record['features']), record['reference'], record['valid'])
        for k, value in result.items():
            groups[k].setdefault(row['work'], []).append(float(value))
        recorder.check()
    require(all(len(v) == 3 for v in groups.values()), 'development work split changed')
    return {k: work_mean(v) for k, v in groups.items()}


def audit_receipts(path, population, plan, architecture):
    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    trainers = ('shared',) if architecture == 'shared' else TASKS
    returned = [e for e in events if e['event'] == 'update_returned']
    entered = [e for e in events if e['event'] == 'update_entered']
    weights, expected = work_weights(population), []
    counters = {k: 0 for k in trainers}
    for epoch in range(plan['epochs']):
        order = np.random.default_rng(plan['seed'] + epoch).permutation(len(population))
        for offset in range(0, len(order), plan['batch_recordings']):
            batch = [population[i] for i in order[offset:offset + plan['batch_recordings']]]
            for task in trainers:
                expected.append((epoch + 1, offset // plan['batch_recordings'], task,
                    [dict(id=r['id'], work=r['work'], scale=weights[r['work']] / len(batch)) for r in batch]))
    require(len(entered) == len(returned) == len(expected), 'incomplete entered/returned receipts')
    for start, end, (epoch, batch, task, rows) in zip(entered, returned, expected):
        for event in (start, end):
            require(tuple(event['cursor'][k] for k in ('epoch', 'batch', 'task')) == (epoch, batch, task),
                    'receipt order changed')
        counters[task] += 1
        require(end['records'] == rows and end['optimizer_calls'] == counters
                and end['receipt']['returned_updates'] == counters[task], 'receipt schedule or counters differ')
    require(len([e for e in events if e['event'] == 'pair_complete']) == plan['paired_updates']
            and len([e for e in events if e['event'] == 'epoch_complete']) == plan['epochs'],
            'incomplete paired or epoch receipts')
    return dict(optimizer_calls=counters, paired_updates=plan['paired_updates'],
                task_records={k: len(population) * plan['epochs'] for k in TASKS},
                returned_receipts=len(returned))


def fit(records, device, arm, output, plan, plan_sha):
    require(arm in ARMS, 'unregistered arm')
    architecture, variant = arm.split('-')
    zero = variant == 'zero'
    started = time.monotonic()
    np.random.seed(plan['seed'])
    random.seed(plan['seed'])
    shared, separated = initial_models(device)
    model = shared if architecture == 'shared' else separated
    del shared, separated
    require(state_hash(model) == plan['initial_state_sha256'][architecture]
            and all(state_hash(branch(model, task)) == plan['initial_task_sha256'][task] for task in TASKS),
            'registered initialization differs before fit')
    population = [r for r in records if r['role'] == 'fit']
    weights = work_weights(population)
    dev = [r for r in records if r['role'] == 'development']
    require(len(population) == 20 and len(weights) == 9 and len(dev) == 5
            and len({r['work'] for r in dev}) == 3, 'fit/development split changed')
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    recorder, history = None, []
    try:
        recorder = Recorder(output / arm, model, dict(arm=arm, plan_sha256=plan_sha,
            source_sha256=plan['source_sha256'], inputs_sha256=plan['inputs_sha256'],
            initial_state_sha256=plan['initial_state_sha256'][architecture]),
            updates=plan['paired_updates'], deadline=started + plan['seconds_per_arm'])
        recorder.check()
        recorder.snapshot('initial.pt')
        for epoch in range(plan['epochs']):
            order = np.random.default_rng(plan['seed'] + epoch).permutation(len(population))
            training = {k: {} for k in TASKS}
            for offset in range(0, len(order), plan['batch_recordings']):
                batch = [population[i] for i in order[offset:offset + plan['batch_recordings']]]
                weighted = [packet(row, zero, weights[row['work']] / len(batch)) for row in batch]
                values = recorder.update(epoch + 1, offset // plan['batch_recordings'], weighted)
                for task in TASKS:
                    require(len(values[task]) == len(batch), 'incomplete training losses')
                    for row, value in zip(batch, values[task]):
                        training[task].setdefault(row['work'], []).append(value)
            scores = development(model, records, zero, recorder)
            recorder.checkpoint(epoch + 1, scores)
            history.append(dict(epoch=epoch + 1, training={k: work_mean(v) for k, v in training.items()},
                                development=scores, paired_updates=recorder.paired_updates))
            journal(output, dict(event='epoch', arm=arm, **history[-1]))
        require(recorder.paired_updates == plan['paired_updates'], 'incomplete paired fit')
        recorder.inference_stage('selected_export')
        selected = recorder.selection.export(model)
        path = output / (arm + '.weights.npz')
        with path.open('xb') as stream:
            np.savez(stream, **{k: v.detach().cpu().numpy() for k, v in selected.state_dict().items()})
        recorder.check()
        if device == 'cuda':
            torch.cuda.synchronize()
        receipts = audit_receipts(recorder.output / 'journal.jsonl', population, plan, architecture)
        report = dict(arm=arm, architecture=architecture, parameters=plan['parameters'][architecture],
            export_parameters=sum(p.numel() for p in selected.parameters()), export_trunk_evaluations=2,
            initial_state_sha256=plan['initial_state_sha256'][architecture], initial_task_sha256=plan['initial_task_sha256'],
            epochs=len(history), updates=recorder.paired_updates, budget_exhausted=False, history=history,
            receipts=receipts, optimizer_calls=recorder.counts(), update_seconds=recorder.timings,
            selected={k: {field: value for field, value in item.items() if field != 'state'}
                      for k, item in recorder.selection.best.items()},
            selected_state_sha256=state_hash(selected), weight_sha256=sha(path),
            snapshot_sha256={p.name: sha(p) for p in sorted(recorder.output.glob('*.pt'))},
            snapshot_bytes=sum(p.stat().st_size for p in recorder.output.glob('*.pt')),
            journal_sha256=sha(recorder.output / 'journal.jsonl'),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
            coherent_clock=False, elapsed_before_report_s=recorder.check() - started)
        write_json(output / (arm + '.fit.json'), report)
        report['fit_evidence_sha256'] = sha(output / (arm + '.fit.json'))
        report['elapsed_s'] = recorder.check() - started
        return selected, report
    except BaseException as error:
        if recorder is not None:
            recorder.abort(error)
        raise


def execution_gate(fits, plan):
    if [f['arm'] for f in fits] != list(ARMS):
        return False
    for fit in fits:
        architecture = fit['architecture']
        if architecture != fit['arm'].split('-')[0]:
            return False
        expected = ({'shared': plan['paired_updates']} if architecture == 'shared'
                    else {k: plan['paired_updates'] for k in TASKS})
        if not (fit['optimizer_calls'] == fit['receipts']['optimizer_calls'] == expected
                and fit['updates'] == fit['receipts']['paired_updates'] == plan['paired_updates']
                and fit['epochs'] == len(fit['history']) == plan['epochs'] and not fit['budget_exhausted']
                and [r['epoch'] for r in fit['history']] == list(range(1, plan['epochs'] + 1))
                and fit['receipts']['task_records'] == {k: 20 * plan['epochs'] for k in TASKS}
                and fit['receipts']['returned_receipts'] == sum(expected.values())
                and fit['initial_state_sha256'] == plan['initial_state_sha256'][architecture]
                and fit['initial_task_sha256'] == plan['initial_task_sha256']
                and math.isfinite(fit['elapsed_s']) and 0 <= fit['elapsed_s'] <= plan['seconds_per_arm']
                and set(fit['update_seconds']) == set(expected)
                and all(len(fit['update_seconds'][k]) == expected[k]
                        and all(math.isfinite(t) and t >= 0 for t in fit['update_seconds'][k]) for k in expected)):
            return False
        for task in TASKS:
            scores = [r['development'][task] for r in fit['history']]
            if not scores or not all(math.isfinite(v) and v >= 0 for v in scores):
                return False
            selected_epoch = min(range(len(scores)), key=scores.__getitem__) + 1
            if fit['selected'][task]['epoch'] != selected_epoch or fit['selected'][task]['loss'] != scores[selected_epoch - 1]:
                return False
    return True


def execute(records, device, output, plan, plan_sha, parents, check):
    models, fits = {}, []
    for arm in ARMS:
        check()
        model, report = fit(records, device, arm, output, plan, plan_sha)
        models[arm] = model
        fits.append(report)
    require(execution_gate(fits, plan), 'four-arm execution incomplete')
    check()
    rows = evaluate(models, records, device, output, parents, check)
    summary = summarize(rows)
    gate = compare(rows, summary, fits)
    gate['execution_complete'] = True
    return dict(fits=fits, cases=rows, summary=summary, gate=gate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'plan', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    started = time.monotonic()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'new private output required')
    require(sha(args.plan) == args.expected_plan_sha256, 'plan identity changed')
    plan = json.loads(args.plan.read_bytes())
    require(plan == make_plan(args.device), 'registered runtime, source, population or budget changed')
    require(sha(args.inputs / 'inputs.json') == plan['inputs_sha256'], 'private input identity changed')
    manifest = json.loads((args.inputs / 'inputs.json').read_bytes())
    parents = parent_reports()
    parents['protected'] = json.loads((ROOT / 'experiments/protected_phase_fit/results-v1.json').read_bytes())
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.4)
    args.output.mkdir()
    write_json(args.output / 'execution.json', dict(schema='rhythm-map.separated-evidence-execution.v1',
        plan=plan, plan_sha256=args.expected_plan_sha256, optimizer_fits_requested=4,
        training=True, encoder_inference=False, holdout_access=False, production_change=False, coherent_clock=False))
    def check():
        if time.monotonic() - started >= plan['watchdog_seconds']:
            raise TimeoutError('outer wall budget exhausted')
    try:
        check()
        records = load_records(args.inputs, manifest, args.device)
        result = execute(records, args.device, args.output, plan, args.expected_plan_sha256, parents, check)
        require(plan == make_plan(args.device), 'registered identity changed during execution')
        check()
        rss = None
        if plan['runtime']['os'] == 'Linux':
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report = dict(schema='rhythm-map.separated-evidence-learning.v1', **result,
            plan_sha256=args.expected_plan_sha256, source_sha256=plan['source_sha256'],
            inputs_sha256=plan['inputs_sha256'], execution_sha256=sha(args.output / 'execution.json'),
            optimizer_fits=4, evidence_pair_evaluations=10 * len(records), training=True,
            optimizer_calls_total=sum(sum(f['optimizer_calls'].values()) for f in result['fits']),
            logical_training_record_presentations=4 * 20 * plan['epochs'],
            process_peak_rss_bytes=rss, elapsed_before_report_s=time.monotonic() - started,
            independent_acceptance=False, coherent_clock=False, encoder_inference=False,
            holdout_access=False, production_change=False, automatic_retry=False,
            decision='propose_coherent_fusion_validation' if result['gate']['supports_fusion_proposal']
                     else 'close_fixed_evidence_comparison_inspect_isolated_tasks_no_automatic_sweep')
        write_json(args.output / 'report.json', report)
        check()
        journal(args.output, dict(event='experiment_complete', gate=result['gate']))
        check()
    except BaseException as error:
        try:
            write_json(args.output / 'failure.json', dict(event='experiment_failed',
                exception_type=type(error).__name__, automatic_retry=False, production_change=False,
                incomplete_or_unknown_calls_are_not_success=True))
        except BaseException as secondary:
            error.add_note('Failure summary unavailable: ' + type(secondary).__name__)
        raise


if __name__ == '__main__':
    main()
