"""One fixed two-arm local-tempo experiment; fresh forwards follow selection."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from experiments.coupled_clock.run import load_records, state_hash, journal
from experiments.relative_tempo.model import load_model
from experiments.relative_tempo.run import configure
from . import data
from .capture import capture, verify_freeze
from .fit import ARMS, EPOCHS, STEPS, SEED, fit, read_weights
from .evaluate import evaluate, decision
from .data import ROOT, require, sha, write_json


def load_preparation(path, expected_sha, closure):
    require(sha(path) == expected_sha, 'preparation report changed')
    report = json.loads(path.read_bytes())
    require(report['completed'] and report['gate_passed'] and report['source_sha256'] == closure, 'preparation did not pass frozen gate')
    require(report['model_calls'] == report['optimizer_steps'] == 0 and not report['holdout_access'], 'preparation crossed task boundary')
    require([(p['source_id'], p['profile']) for p in report['pairs']] == [(i, p) for i in data.IDS for p in data.FRESH],
            'fresh preparation population changed')
    points = [r for p in report['pairs'] for r in p['points']]
    require(len(points) == 366 and all(r['pair_role'] == 'fresh_schedule_diagnostic' for r in points), 'fresh grid/role changed')
    require(all(p['passed'] and p['negative_control']['passed'] and p['admitted_changed_points'] >= 10 for p in report['pairs']),
            'fresh preparation pair gate failed')
    return report, points


def load_captures(old_run, device):
    report = data.prior()
    require(sha(old_run/'report.json') == data.OLD_REPORT_SHA, 'old private report changed')
    items = report['capture']['cases']
    require([(r['source_id'], r['profile']) for r in items] == [(i, p) for i in data.IDS for p in ('source', *data.old.PROFILES)],
            'old capture population changed')
    captures = {}
    for row in items:
        path = old_run/f"{row['source_id']}.{row['profile']}.features.npz"
        require(sha(path) == row['feature_sha256'], 'old captured tensor changed')
        with np.load(path, allow_pickle=False) as archive:
            x = archive['hidden'].copy()
        require(x.shape == (row['frames'], 512) and x.dtype == np.float32 and np.isfinite(x).all(), 'invalid captured tensor')
        captures[row['source_id'], row['profile']] = torch.from_numpy(x).to(device)
    return captures


def run(args):
    started = time.monotonic()
    configure(args.device)
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'fresh private output required')
    if os.name == 'nt':
        require(args.output.resolve().drive.lower() != os.environ.get('SystemDrive', 'C:').lower(), 'use data drive')
    source = data.closure()
    preparation, fresh = load_preparation(args.prepared/'report.json', args.preparation_sha256, source)
    points = data.fit_points()
    manifest_path = ROOT/'experiments/coupled_clock/inputs-v1.json'
    require(sha(args.inputs/'inputs.json') == sha(manifest_path), 'natural manifest changed')
    require(sha(args.weights) == data.old.WEIGHTS_SHA, 'starting weights changed')
    upstream = {p.relative_to(args.assets).as_posix(): sha(p) for p in (args.assets/'upstream').rglob('*.py')}
    dependencies = {p.relative_to(args.deps).as_posix(): sha(p) for p in args.deps.rglob('*.py')}
    previous = data.prior()
    old_plan = json.loads((data.old.HERE/'plan-v1.json').read_bytes())
    require(upstream == old_plan['upstream_sha256'] and dependencies == old_plan['dependency_sha256'], 'encoder dependency sources changed')
    args.output.mkdir(parents=True)
    plan = dict(schema='rhythm-map.local-tempo-plan.v1', source_sha256=source, points=points+fresh,
                preparation_sha256=args.preparation_sha256, natural_inputs_sha256=sha(manifest_path),
                initial_weights_sha256=data.old.WEIGHTS_SHA, encoder_checkpoint_sha256=data.old.CHECKPOINT_SHA,
                upstream_sha256=upstream, dependency_sha256=dependencies, arms=list(ARMS), primary='local-retained',
                seed=SEED, epochs=EPOCHS, steps_per_arm=STEPS, learning_rate=.0003, pair_coefficient=1.,
                retention_coefficient=1., samples_per_group=4, selection='development-constrained-training-pair-minimum',
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, device=args.device,
                holdout_access=False, production_change=False, independent_acceptance=False)
    write_json(args.output/'plan.json', plan)

    def check():
        require(time.monotonic()-started < 5300, 'global wall budget exceeded')

    try:
        captures = load_captures(args.old_run, args.device)
        records = load_records(args.inputs, json.loads(manifest_path.read_bytes()), args.device)
        initial = load_model(args.weights, args.device).eval()
        require(sum(p.numel() for p in initial.parameters()) == 23937, 'model size changed')
        write_json(args.output/'initial.json', dict(state_sha256=state_hash(initial), parameters=23937))
        reference = next(r for r in previous['fits'] if r['arm'] == 'native-plus-relative')
        models = {'initial': initial, 'previous-global': read_weights(initial, args.old_run/'native-plus-relative.weights.npz', reference['weight_sha256'])}
        fits = []
        for arm in ARMS:
            selected, terminal, report = fit(initial, records, captures, points, arm, args.output, check)
            models[arm+'-selected'], models[arm+'-final'] = selected, terminal
            fits.append(report)
        freeze = {arm+suffix: sha(args.output/(arm+suffix+'.weights.npz')) for arm in ARMS for suffix in ('-selected', '-final')}
        verify_freeze(args.output, freeze)
        write_json(args.output/'selection-freeze.json', dict(exports=freeze, fits=[sha(args.output/(a+'.fit.json')) for a in ARMS]))
        journal(args.output, dict(event='selection_frozen', exports=freeze))
        sys.path.insert(0, str(args.deps))
        new_captures, capture_report = capture(args.prepared, preparation, args.assets, args.output, args.device, freeze, check)
        captures.update(new_captures)
        natural, pairs = evaluate(models, records, captures, points+fresh, args.device, args.output, check)
        summary, gate = decision(natural, pairs, fits[1]['selected_epoch'])
        require(data.closure() == source and sha(args.prepared/'report.json') == args.preparation_sha256, 'frozen inputs changed')
        require(all(sha(args.assets/k) == v for k, v in upstream.items()) and all(sha(args.deps/k) == v for k, v in dependencies.items()),
                'encoder dependency changed')
        require(all(f['initial_state_sha256'] == state_hash(initial) for f in fits), 'arm initialization differs')
        events = [json.loads(line) for line in (args.output/'journal.jsonl').read_text().splitlines()]
        draws = []
        for arm in ARMS:
            for event in ('update_entered', 'update_returned'):
                require([e['update'] for e in events if e.get('arm') == arm and e['event'] == event] == list(range(1, STEPS+1)), 'incomplete update receipts')
            draws.append([(e['natural_ids'], e['pair_samples']) for e in events if e.get('arm') == arm and e['event'] == 'update_entered'])
        require(draws[0] == draws[1], 'arm draws differ')
        verify_freeze(args.output, freeze)
        check()
        report = dict(schema='rhythm-map.local-tempo-result.v1', completed=True, plan_sha256=sha(args.output/'plan.json'),
                      source_sha256=source, preparation_sha256=args.preparation_sha256, capture=capture_report, fits=fits,
                      natural=natural, pairs=pairs, summary=summary, gate=gate, optimizer_steps=2*STEPS,
                      selection_freeze_sha256=sha(args.output/'selection-freeze.json'), journal_sha256=sha(args.output/'journal.jsonl'),
                      elapsed_s=time.monotonic()-started, independent_acceptance=False, holdout_access=False, production_change=False)
        write_json(args.output/'report.json', report)
        print(json.dumps(dict(event='completed', gate=gate, elapsed_s=report['elapsed_s'])), flush=True)
    except BaseException as error:
        write_json(args.output/'partial-report.json', dict(completed=False, failure_type=type(error).__name__, elapsed_s=time.monotonic()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'old-run', 'assets', 'deps', 'inputs', 'weights', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--preparation-sha256', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    run(parser.parse_args())
