"""Execute exactly one new observed fit pair; never resume a closed experiment."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from experiments.coupled_clock.clock import require
from experiments.coupled_clock.run import state_hash, work_weights, work_mean, input_tensor, journal, load_records
from experiments.coupled_clock.measurement import macro
from experiments.phase_sync.model import PhaseSyncReadout, PARAMETERS
from experiments.phase_sync_fit.run import evaluate, comparison_gate, ALL_KINDS, CONTROL_KINDS
from experiments.phase_sync_fit.register import parent_reports
from experiments.scaled_phase_fit.register import ROOT, sha, write_json, make_plan
from experiments.scaled_phase_fit.recorder import MusicalRecorder, score, conditioning


def packet(row, zero_audio, scale=1.):
    return dict(id=row['id'], scale=scale, payload=dict(features=input_tensor(row, zero_audio),
                reference=row['target'][None], valid=row['mask'][None]))


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def fit(records, device, zero_audio, output, plan, plan_sha):
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    random.seed(plan['seed'])
    model = PhaseSyncReadout().to(device)
    initial = state_hash(model)
    require(initial == plan['initial_state_sha256'], 'registered initialization differs before fit')
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['optimizer']['learning_rate'],
                                  weight_decay=plan['optimizer']['weight_decay'])
    population = [r for r in records if r['role'] == 'fit']
    weights = work_weights(population)
    require(len(population) == 20 and len(weights) == 9, 'fit work split changed')
    require(sum(r['role'] == 'development' for r in records) == 5, 'development split changed')
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    name = 'zero-audio' if zero_audio else 'audio'
    started = time.monotonic()
    observer = None
    stage = 'recorder_creation'
    history, update_seconds = [], []
    try:
        observer = MusicalRecorder(output / name, model, optimizer,
            dict(plan_sha256=plan_sha, kind=name, initial_state_sha256=initial,
                 source_sha256=plan['source_sha256'], inputs_sha256=plan['inputs_sha256']),
            max_updates=plan['updates'], deadline=started + plan['seconds_per_fit'])
        observer.check_deadline()
        journal(output, dict(event='fit_started', kind=name, initial_state_sha256=initial))
        for epoch in range(plan['epochs']):
            order = np.random.default_rng(plan['seed'] + epoch).permutation(len(population))
            training = {}
            for offset in range(0, len(order), plan['batch_recordings']):
                stage = 'training_update'
                batch = [population[i] for i in order[offset:offset + plan['batch_recordings']]]
                weighted = [packet(row, zero_audio, weights[row['work']] / len(batch)) for row in batch]
                before = time.monotonic()
                losses = observer.update(epoch + 1, offset // plan['batch_recordings'], weighted,
                                         clip_norm=plan['optimizer']['max_norm'])
                duration = time.monotonic() - before
                update_seconds.append(duration)
                observer._event('update_timing', seconds=duration)
                for row, loss in zip(batch, losses):
                    training.setdefault(row['work'], []).append(loss)
            stage = 'development_aggregation'
            totals = {}
            for row in records:
                if row['role'] == 'development':
                    value = observer.development_record(epoch + 1, packet(row, zero_audio), score)
                    totals.setdefault(row['work'], []).append(value)
            require(len(totals) == 3, 'development work split changed')
            development = work_mean(totals)
            observer.check_deadline()
            stage = 'checkpoint'
            observer.checkpoint(epoch + 1, development)
            observer.check_deadline()
            item = dict(epoch=epoch + 1, training_work_macro_loss=work_mean(training),
                        development_loss=development, updates=observer.completed_updates)
            history.append(item)
            journal(output, dict(event='epoch', kind=name, **item))
        require(observer.completed_updates == plan['updates'] and len(history) == plan['epochs'],
                'incomplete registered fit')
        stage = 'selected_export'
        observer.check_deadline()
        model.load_state_dict(observer.best['model'])
        model.eval()
        path = output / (name + '.weights.npz')
        with path.open('xb') as stream:
            np.savez(stream, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
        if device == 'cuda':
            torch.cuda.synchronize()
        events_path = observer.output / 'journal.jsonl'
        diagnostic = conditioning(read_events(events_path))
        require(diagnostic['update_receipts'] == plan['updates']
                and diagnostic['record_gradients'] == plan['complete_record_gradients']
                and diagnostic['alignment_underflows'] == 0, 'incomplete diagnostic receipts')
        observer.check_deadline()
        report = dict(kind=name, parameters=PARAMETERS, initial_state_sha256=initial, seed=plan['seed'],
            epochs=len(history), updates=observer.completed_updates, budget_exhausted=False,
            selected_epoch=observer.best['epoch'], development_loss=observer.best['development_loss'],
            update_seconds=update_seconds, conditioning=diagnostic,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
            weight_sha256=sha(path), selected_state_sha256=state_hash(model), history=history,
            journal_sha256=sha(events_path),
            snapshot_sha256={p.name: sha(p) for p in sorted(observer.output.glob('*.pt'))},
            snapshot_bytes=sum(p.stat().st_size for p in observer.output.glob('*.pt')))
        # The private evidence file has a pre-receipt timestamp. The final
        # returned report measures AFTER its persistence/hash, then enforces
        # the same deadline before allowing the next fit to start.
        report['elapsed_before_report_s'] = observer.check_deadline() - started
        evidence_path = output / (name + '.fit.json')
        write_json(evidence_path, report)
        report['fit_evidence_sha256'] = sha(evidence_path)
        report['elapsed_s'] = observer.check_deadline() - started
        return model, report
    except BaseException as error:
        if observer is not None:
            observer.abort_external(error, stage)
        raise


def execution_gate(fits, plan):
    return len(fits) == 2 and {row['kind'] for row in fits} == {'audio', 'zero-audio'} and all(
        row['initial_state_sha256'] == plan['initial_state_sha256']
        and row['conditioning']['update_receipts'] == plan['updates']
        and row['conditioning']['record_gradients'] == plan['complete_record_gradients']
        and row['conditioning']['alignment_underflows'] == 0
        and len(row['update_seconds']) == plan['updates']
        and all(math.isfinite(v) and v >= 0 for v in row['update_seconds'])
        for row in fits)


def failure_evidence(output):
    result = {}
    for kind in ('audio', 'zero-audio'):
        directory = output / kind
        if not directory.is_dir():
            continue
        path = directory / 'journal.jsonl'
        events = read_events(path) if path.exists() else []
        failed = [e for e in events if e['event'] == 'failed']
        result[kind] = dict(conditioning=conditioning(events), last_failed=failed[-1] if failed else None,
            snapshot_sha256={p.name: sha(p) for p in sorted(directory.glob('*.pt'))},
            journal_sha256=sha(path) if path.exists() else None)
    return result


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
    require(plan == make_plan(args.device), 'registered runtime, source, population or initialization changed')
    require(sha(args.inputs / 'inputs.json') == plan['inputs_sha256'], 'private input identity changed')
    manifest = json.loads((args.inputs / 'inputs.json').read_bytes())
    parents = parent_reports()
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.4)
    args.output.mkdir()
    write_json(args.output / 'execution.json', dict(schema='rhythm-map.scaled-phase-execution.v1',
        plan_sha256=args.expected_plan_sha256, plan=plan, training=True,
        optimizer_fits_requested=2, encoder_inference=False, holdout_access=False, production_change=False))
    try:
        records = load_records(args.inputs, manifest, args.device)
        audio, audio_fit = fit(records, args.device, False, args.output, plan, args.expected_plan_sha256)
        zero, zero_fit = fit(records, args.device, True, args.output, plan, args.expected_plan_sha256)
        rows = evaluate(dict(audio=audio, zero=zero), records, args.device, args.output, parents)
        summary = {role: {kind: macro([r for r in rows if r['role'] == role], kind) for kind in ALL_KINDS}
                   for role in ('fit', 'development', 'diagnostic')}
        fits = [audio_fit, zero_fit]
        gate = comparison_gate(rows, summary, fits)
        gate['observed_execution_complete'] = execution_gate(fits, plan)
        gate['passed'] = all(value for key, value in gate.items() if key != 'passed')
        require(plan == make_plan(args.device), 'registered identity changed during execution')
        rss = None
        if plan['runtime']['os'] == 'Linux':
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report = dict(schema='rhythm-map.scaled-phase-learning.v1',
            plan_sha256=args.expected_plan_sha256, inputs_sha256=plan['inputs_sha256'],
            source_sha256=plan['source_sha256'], prerequisite_sha256=plan['prerequisite_sha256'],
            parent_report_sha256=plan['parent_report_sha256'],
            execution_sha256=sha(args.output / 'execution.json'), fits=fits,
            cases=rows, summary=summary, gate=gate, training=True, optimizer_fits=2,
            checkpoint_sequence_evaluations=5 * len(records), independent_acceptance=False,
            encoder_inference=False, holdout_access=False, production_change=False,
            process_peak_rss_bytes=rss, controls=list(CONTROL_KINDS),
            selected_gain={kind: float(math.log(2.) * torch.sigmoid(model.gain_logit.detach().double()))
                           for kind, model in (('audio', audio), ('zero', zero))},
            decision='fresh_independent_validation_required' if gate['passed']
            else 'close_fixed_scaled_phase_fit_no_automatic_sweep')
        write_json(args.output / 'report.json', report)
        journal(args.output, dict(event='experiment_complete', gate=gate))
    except BaseException as error:
        failure = dict(event='experiment_failed', exception_type=type(error).__name__,
                       automatic_retry=False, production_change=False)
        try:
            failure['evidence'] = failure_evidence(args.output)
        except BaseException as capture_error:
            failure['evidence_capture_error'] = type(capture_error).__name__
        try:
            write_json(args.output / 'failure.json', failure)
        except BaseException as capture_error:
            error.add_note('Failure summary unavailable: ' + type(capture_error).__name__)
        raise


if __name__ == '__main__':
    main()
