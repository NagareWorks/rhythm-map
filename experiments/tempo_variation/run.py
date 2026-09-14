"""One fixed native-offset ablation on already exposed data."""
import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from experiments.coupled_clock.run import load_records, state_hash, journal
from experiments.relative_tempo.model import load_model
from experiments.relative_tempo.run import configure
from experiments.local_tempo.run import load_captures
from experiments.local_tempo.fit import read_weights
from . import data
from .data import ROOT, require, sha, write_json
from .fit import ARMS, EPOCHS, STEPS, fit
from .evaluate import evaluate, ALIASES


def control_replay(report, previous):
    old = next(r for r in previous['fits'] if r['arm'] == 'local-retained')
    require(len(report['history']) == len(old['history']) == 21, 'control history incomplete')
    for now, then in zip(report['history'], old['history']):
        require(now['epoch'] == then['epoch'] and now['state_sha256'] == then['state_sha256'], 'control state failed exact replay')
        require(now['development'] == then['development'] and now['training_pair_mse'] == then['training_pair_mse'],
                'control metrics failed exact replay')
    require(report['selected_epoch'] == old['selected_epoch'], 'control selection differs')
    return dict(passed=True, epoch_states=21, bit_exact=True, selected_epoch=report['selected_epoch'])


def verify_freeze(output, exports):
    require(set(exports) == {a+s for a in ARMS for s in ('-selected', '-final')}, 'freeze population changed')
    require(all(sha(output/(k+'.weights.npz')) == v for k, v in exports.items()), 'frozen export changed')


def receipts(output, fits):
    events = [json.loads(line) for line in (output/'journal.jsonl').read_text().splitlines()]
    draws = []
    for arm, report in zip(ARMS, fits):
        require(report['arm'] == arm and report['updates'] == STEPS, 'incomplete fit')
        for event in ('backward_entered', 'update_entered', 'update_returned'):
            require([e['update'] for e in events if e.get('arm') == arm and e['event'] == event] == list(range(1, STEPS+1)),
                    'incomplete update receipts')
        draws.append([(e['natural_ids'], e['pair_samples']) for e in events if e.get('arm') == arm and e['event'] == 'backward_entered'])
        for item in report['losses']:
            prefix = f"{arm}.update-{item['update']:03}"
            for suffix, key in (('.before.pt', 'before_sha256'), ('.gradients.npz', 'gradient_sha256'), ('.after.pt', 'after_sha256')):
                require(sha(output/(prefix+suffix)) == item[key], 'update artifact changed')
    require(draws[0] == draws[1], 'arm draws differ')
    return dict(matched_draws=True, complete_updates=2*STEPS, before_after_optimizer_packets=4*STEPS, gradient_packets=2*STEPS)


def run(args):
    started = time.monotonic()
    configure(args.device)
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'fresh private output required')
    if os.name == 'nt':
        require(args.output.resolve().drive.lower() != os.environ.get('SystemDrive', 'C:').lower(), 'use data drive')
    previous, source, points = data.prior(), data.closure(), data.points()
    require(sha(args.local_run/'report.json') == data.audit.PREVIOUS_SHA, 'private local run changed')
    manifest_path = ROOT/'experiments/coupled_clock/inputs-v1.json'
    require(sha(args.inputs/'inputs.json') == sha(manifest_path), 'natural manifest changed')
    require(sha(args.weights) == data.old.old.WEIGHTS_SHA, 'initial export changed')
    args.output.mkdir(parents=True)
    plan = dict(schema='rhythm-map.tempo-variation-plan.v1', source_sha256=source,
                local_report_sha256=data.audit.PREVIOUS_SHA, conflict_report_sha256=data.AUDIT_SHA,
                natural_manifest_sha256=sha(manifest_path), initial_weights_sha256=data.old.old.WEIGHTS_SHA,
                points=points, arms=list(ARMS), primary=ARMS[1], epochs=EPOCHS, updates_per_arm=STEPS,
                lr=.0003, weight_decay=.0001, clip_norm=1., native_seed=142, pair_seed=8142,
                pair_coefficient=1., retention_coefficient=1., samples_per_group=4,
                prediction_artifact_aliases=ALIASES, python=platform.python_version(), torch=torch.__version__,
                numpy=np.__version__, device=args.device, encoder_calls=0, holdout_access=False,
                independent_acceptance=False, production_change=False)
    write_json(args.output/'plan.json', plan)

    def check():
        require(time.monotonic()-started < 5300, 'global wall budget exceeded')

    try:
        captures = load_captures(args.old_run, args.device)
        records = load_records(args.inputs, json.loads(manifest_path.read_bytes()), args.device)
        initial = load_model(args.weights, args.device).eval()
        require(sum(p.numel() for p in initial.parameters()) == 23937, 'architecture changed')
        initial_hash = state_hash(initial)
        old_global = next(r for r in data.old.prior()['fits'] if r['arm'] == 'native-plus-relative')
        models = {'initial': initial, 'previous-global': read_weights(initial, args.old_run/'native-plus-relative.weights.npz', old_global['weight_sha256'])}
        fits = []
        replay = None
        for arm in ARMS:
            chosen, terminal, report = fit(initial, records, captures, points, arm, args.output, check)
            models[arm+'-selected'], models[arm+'-final'] = chosen, terminal
            fits.append(report)
            if arm == ARMS[0]:
                replay = control_replay(report, previous)
                write_json(args.output/'control-replay.json', replay)
        exports = {a+s: sha(args.output/(a+s+'.weights.npz')) for a in ARMS for s in ('-selected', '-final')}
        verify_freeze(args.output, exports)
        write_json(args.output/'selection-freeze.json', dict(exports=exports))
        journal(args.output, dict(event='selection_frozen', exports=exports))
        captures.update(data.load_exposed(args.local_run, args.device))
        natural, pairs, summary, gate, decomposition = evaluate(models, records, captures, points, args.device, args.output, fits[1]['selected_epoch'], check)
        receipt = receipts(args.output, fits)
        verify_freeze(args.output, exports)
        require(data.closure() == source and state_hash(initial) == initial_hash, 'frozen source/initial changed')
        require(sha(args.inputs/'inputs.json') == sha(manifest_path) and sha(args.weights) == data.old.old.WEIGHTS_SHA,
                'original inputs changed')
        check()
        report = dict(schema='rhythm-map.tempo-variation-result.v1', completed=True, source_sha256=source,
                      plan_sha256=sha(args.output/'plan.json'), fits=fits, control_replay=replay, receipts=receipt,
                      natural=natural, pairs=pairs, summary=summary, gate=gate, native_decomposition=decomposition,
                      optimizer_steps=2*STEPS, encoder_calls=0, selection_freeze_sha256=sha(args.output/'selection-freeze.json'),
                      journal_sha256=sha(args.output/'journal.jsonl'), elapsed_s=time.monotonic()-started,
                      independent_acceptance=False, holdout_access=False, production_change=False)
        write_json(args.output/'report.json', report)
        print(json.dumps(dict(event='completed', gate=gate, elapsed_s=report['elapsed_s'])), flush=True)
    except BaseException as error:
        write_json(args.output/'partial-report.json', dict(completed=False, failure_type=type(error).__name__, elapsed_s=time.monotonic()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('local-run', 'old-run', 'inputs', 'weights', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    run(parser.parse_args())
