"""Inspect frozen gradients and saved displacements, without training."""
import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from experiments.coupled_clock.run import load_records, state_hash, work_weights, journal
from experiments.relative_tempo.model import load_model
from experiments.relative_tempo.run import configure
from experiments.local_tempo.run import load_captures
from experiments.local_tempo.fit import read_weights
from . import data, gradients as g, statistics as stats
from .data import ROOT, require, sha, write_json


def same_metrics(actual, expected):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'metric fields differ')
        for key in expected:
            same_metrics(actual[key], expected[key])
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'metric population differs')
        for a, b in zip(actual, expected):
            same_metrics(a, b)
    elif isinstance(expected, float):
        require(np.isfinite(actual) and abs(actual-expected) <= 1e-7, 'saved metric replay differs')
    else:
        require(actual == expected, 'discrete metric differs')


def first_draw(local_run, previous, records, groups):
    path = local_run/'journal.jsonl'
    require(sha(path) == previous['journal_sha256'], 'old optimizer receipt changed')
    events = [json.loads(line) for line in path.read_text().splitlines()]
    entered, returned = [], []
    for arm in ('local-only', 'local-retained'):
        entered.append(next(e for e in events if e.get('arm') == arm and e['event'] == 'update_entered' and e['update'] == 1))
        returned.append(next(e for e in events if e.get('arm') == arm and e['event'] == 'update_returned' and e['update'] == 1))
    require(entered[0]['natural_ids'] == entered[1]['natural_ids'] and entered[0]['pair_samples'] == entered[1]['pair_samples'],
            'first arm draws differ')
    by_id = {r['id']: r for r in records if r['role'] == 'fit'}
    require(len(entered[0]['natural_ids']) == 4 and set(entered[0]['natural_ids']) <= by_id.keys(), 'first natural draw escaped fit')
    population = [by_id[i] for i in entered[0]['natural_ids']]
    require(len(groups) == len(entered[0]['pair_samples']) == 15, 'first pair draw population differs')
    selected = []
    for group, sample in zip(groups, entered[0]['pair_samples']):
        require((group[0]['source_id'], group[0]['profile']) == (sample['source_id'], sample['profile']), 'first group differs')
        by_index = {r['index']: r for r in group}
        require(len(sample['indices']) == 4 and set(sample['indices']) <= by_index.keys(), 'first pair indices changed')
        selected.append([by_index[i] for i in sample['indices']])
    return population, selected, dict(natural_ids=entered[0]['natural_ids'], pair_samples=entered[0]['pair_samples'], returned=returned)


def movement(rows, packets):
    by_key = {r['key']: r for r in rows}
    result = []
    for arm in ('local-only', 'local-retained'):
        for epoch in range(1, 21):
            left = 'initial' if epoch == 1 else f'{arm}.epoch-{epoch-1:02}'
            right = f'{arm}.epoch-{epoch:02}'
            a, b = by_key[left], by_key[right]
            _, start, before = stats.aggregates(packets[left], a['development'])
            _, end, after = stats.aggregates(packets[right], b['development'])
            require(set(start) == set(end), 'development ownership changed across endpoints')
            delta = packets[right]['theta']-packets[left]['theta']
            result.append(dict(arm=arm, from_epoch=epoch-1, to_epoch=epoch,
                development={k: stats.observed(start[k], end[k], delta, before[k], after[k]) for k in start}))
    return result


def run(args):
    started = time.monotonic()
    configure(args.device)
    previous, source = data.prior(), data.closure()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'fresh private output required')
    if os.name == 'nt':
        require(args.output.resolve().drive.lower() != os.environ.get('SystemDrive', 'C:').lower(), 'use data drive')
    require(sha(args.local_run/'report.json') == data.PREVIOUS_SHA, 'private closed report changed')
    require(sha(args.local_run/'plan.json') == previous['plan_sha256'], 'private closed plan changed')
    manifest_path = ROOT/'experiments/coupled_clock/inputs-v1.json'
    require(sha(args.inputs/'inputs.json') == sha(manifest_path), 'natural manifest changed')
    manifest = json.loads(manifest_path.read_bytes())
    slots = data.slots(previous)
    for slot in slots:
        require(sha(args.local_run/slot['file']) == slot['weight_sha256'], 'saved epoch identity changed')
    require(sha(args.weights) == data.old.old.WEIGHTS_SHA, 'initial export changed')
    args.output.mkdir(parents=True)
    initial = load_model(args.weights, args.device).eval()
    initial_hash = state_hash(initial)
    require(initial_hash == slots[0]['state_sha256'], 'initial state differs from epoch zero')
    plan = dict(schema='rhythm-map.tempo-conflict-plan.v1', source_sha256=source, previous_report_sha256=data.PREVIOUS_SHA,
                slots=slots, parameter_geometry=g.geometry(initial), natural_manifest_sha256=sha(manifest_path),
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, device=args.device,
                pair_batch=g.PAIR_BATCH, optimizer_steps=0, encoder_forwards=0, holdout_access=False,
                development_role='selection_exposed_gradient_diagnostic_only', independent_acceptance=False, production_change=False)
    write_json(args.output/'plan.json', plan)

    def check():
        require(time.monotonic()-started < 1700, 'diagnostic wall budget exceeded')

    rows, packets = [], {}
    try:
        loaded = load_records(args.inputs, manifest, args.device)
        records = [r for r in loaded if r['role'] in ('fit', 'development')]
        require(len(loaded) == 40 and len(records) == 25, 'natural role population changed')
        del loaded
        captures = load_captures(args.old_run, args.device)
        groups = data.old.groups(data.old.fit_points(), data.old.old.PROFILES)
        fit_rows = [r for r in records if r['role'] == 'fit']
        require(len(fit_rows) == 20 and len(work_weights(fit_rows)) == 9, 'fit population changed')
        with torch.no_grad():
            teacher_targets = {r['id']: initial(r['features'][None]).detach().clone() for r in fit_rows}
        for slot in slots:
            check()
            if slot['key'] in packets:
                require(slot['epoch'] == 0 and slot['state_sha256'] == initial_hash, 'unexpected duplicate state slot')
                continue
            journal(args.output, dict(event='state_entered', key=slot['key']))
            model = read_weights(initial, args.local_run/slot['file'], slot['weight_sha256'])
            require(state_hash(model) == slot['state_sha256'], 'loaded epoch state differs')
            vectors, metrics = g.inspect(model, records, captures, groups, teacher_targets, check)
            saved = next(f for f in previous['fits'] if f['arm'] == slot['arm'])['history'][slot['epoch']]
            same_metrics(metrics['development'], saved['development'])
            pair_loss = sum(metrics['losses'][k] for k in ('pair_identity', 'pair_global', 'pair_local'))
            require(abs(pair_loss-saved['training_pair_mse']) <= 5e-6, 'full-pair objective differs from saved score')
            path = args.output/(slot['key']+'.vectors.npz')
            with path.open('xb') as stream:
                np.savez(stream, **vectors)
            item = dict(key=slot['key'], **metrics, geometry=stats.summarize(vectors, metrics['development']), vector_sha256=sha(path))
            write_json(args.output/(slot['key']+'.json'), item)
            packets[slot['key']] = vectors
            rows.append(item)
            journal(args.output, dict(event='state_returned', key=slot['key'], vector_sha256=item['vector_sha256']))
        require(len(rows) == 41, 'incomplete checkpoint coverage')
        population, chosen, receipt = first_draw(args.local_run, previous, records, groups)
        vectors, losses = g.native(initial, population, teacher_targets, work_weights(fit_rows), len(population), check)
        pair_vectors, pair_losses, detail = g.paired(initial, captures, chosen, check)
        vectors.update(pair_vectors)
        losses.update(pair_losses)
        for key, value in packets['initial'].items():
            if key.startswith('dev_'):
                vectors[key] = value
        total = vectors['native']+sum(vectors[k] for k in ('pair_identity', 'pair_global', 'pair_local'))
        norm = float(np.linalg.norm(total))
        for returned in receipt['returned']:
            require(np.isclose(norm, returned['unclipped_grad_norm'], rtol=1e-5, atol=1e-6), 'recorded first-batch gradient norm differs')
            require(np.isclose(losses['native'], returned['native_loss'], rtol=1e-5, atol=1e-6), 'first native loss differs')
            require(np.isclose(sum(pair_losses.values()), returned['relative_loss'], rtol=1e-5, atol=1e-6), 'first pair loss differs')
        require(np.count_nonzero(vectors['retention']) == 0 and losses['retention'] == 0., 'initial retention was not zero')
        path = args.output/'first-batch.vectors.npz'
        with path.open('xb') as stream:
            np.savez(stream, **vectors)
        first = dict(receipt=receipt, losses=losses, pair_groups=detail, recomputed_unclipped_norm=norm,
                     geometry=stats.summarize(vectors, rows[0]['development']), vector_sha256=sha(path))
        intervals = movement(rows, packets)
        require(state_hash(initial) == initial_hash and all(p.grad is None for p in initial.parameters()), 'initial/teacher mutated')
        require(source == data.closure(), 'audit code changed during run')
        for slot in slots:
            require(sha(args.local_run/slot['file']) == slot['weight_sha256'], 'input epoch file changed')
        check()
        report = dict(schema='rhythm-map.tempo-conflict-result.v1', completed=True, source_sha256=source,
                      plan_sha256=sha(args.output/'plan.json'), states=rows, first_batch=first, intervals=intervals,
                      state_coverage=41, epoch_exports_verified=42, natural_records_loaded=40, natural_records_differentiated=25,
                      optimizer_steps=0, encoder_forwards=0, art_forwards=0, fresh_schedule_forwards=0,
                      model_states_unchanged=True, gradient_buffers_unchanged=True, teacher_unchanged=True,
                      holdout_access=False, independent_acceptance=False, production_change=False,
                      elapsed_s=time.monotonic()-started, journal_sha256=sha(args.output/'journal.jsonl'))
        write_json(args.output/'report.json', report)
        print(json.dumps(dict(completed=True, states=41, intervals=40, optimizer_steps=0, elapsed_s=report['elapsed_s'])), flush=True)
    except BaseException as error:
        write_json(args.output/'partial-report.json', dict(completed=False, failure_type=type(error).__name__,
                    completed_states=[r['key'] for r in rows], elapsed_s=time.monotonic()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('local-run', 'old-run', 'inputs', 'weights', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    run(parser.parse_args())
