"""One two-arm continuation with final-checkpoint-only evaluation."""
import argparse
import copy
import json
import os
import platform
import sys
import time

import numpy as np
import torch

from experiments.coupled_clock.run import load_records, state_hash, work_weights, work_mean, journal
from experiments.separated_evidence.supervision import task_loss
from . import data
from .data import ROOT, IDS, PROFILES, require, sha, write_json
from .model import load_model, paired_loss

EPOCHS, STEPS, SEED = 20, 100, 142
ARMS = ('native-only', 'native-plus-relative')


def configure(device):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA unavailable')
        torch.cuda.set_per_process_memory_fraction(.4)


def fit(initial, records, captures, points, arm, output, check):
    model = copy.deepcopy(initial).train()
    initial_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    population = [r for r in records if r['role'] == 'fit']
    weights = work_weights(population)
    require(len(population) == 20 and len(weights) == 9, 'natural fit split changed')
    groups = [[r for r in points if r['admitted'] and r['source_id'] == i and r['profile'] == p]
              for i in IDS for p in data.FIT_PROFILES]
    require(all(groups) and all(r['pair_role'] == 'fit' for g in groups for r in g), 'invalid paired fit support')
    rng, history, updates = np.random.default_rng(8142), [], 0
    started = time.monotonic()
    for epoch in range(EPOCHS):
        order = np.random.default_rng(SEED+epoch).permutation(len(population))
        for offset in range(0, len(order), 4):
            check()
            require(time.monotonic()-started < 1800, 'arm budget exceeded')
            optimizer.zero_grad(set_to_none=True)
            batch = [population[i] for i in order[offset:offset+4]]
            native_loss = 0.
            for row in batch:
                prediction = model(row['features'][None])
                loss = task_loss('tempo', prediction, row['target'][None], row['mask'][None])*weights[row['work']]/len(batch)
                loss.backward()
                native_loss += float(loss.detach())
            relative, sampled = 0., []
            if arm == 'native-plus-relative':
                for group in groups:
                    chosen = [group[j] for j in rng.integers(len(group), size=4)]
                    identity, profile = chosen[0]['source_id'], chosen[0]['profile']
                    loss = paired_loss(model, captures[identity, 'source'], captures[identity, profile], chosen)/len(groups)
                    require(torch.isfinite(loss).item(), 'nonfinite pair loss')
                    loss.backward()
                    relative += float(loss.detach())
                    sampled.append(dict(source_id=identity, profile=profile, indices=[r['index'] for r in chosen]))
            require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
                    'missing or nonfinite trainable gradient')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            journal(output, dict(event='update_entered', arm=arm, update=updates+1,
                                 natural_ids=[r['id'] for r in batch], pair_samples=sampled))
            optimizer.step()
            updates += 1
            journal(output, dict(event='update_returned', arm=arm, update=updates,
                                 native_loss=native_loss, relative_loss=relative, unclipped_grad_norm=float(norm)))
            history.append(dict(update=updates, epoch=epoch+1, native_loss=native_loss, relative_loss=relative))
    require(updates == STEPS, 'incomplete training')
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), 'nonfinite final weights')
    path = output/(arm+'.weights.npz')
    with path.open('xb') as stream:
        np.savez(stream, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    report = dict(arm=arm, updates=updates, epochs=EPOCHS, selected_update=STEPS,
                  initial_state_sha256=initial_hash, final_state_sha256=state_hash(model),
                  weight_sha256=sha(path), history=history, elapsed_s=time.monotonic()-started)
    write_json(output/(arm+'.fit.json'), report)
    return model.eval(), report


def natural_metrics(prediction, row):
    valid = row['valid'][:-1] & row['valid'][1:]
    reference = row['reference'].astype(np.float64)
    target = -np.log2(np.diff(reference)[valid]*50)
    values = prediction[valid]
    require(len(target) > 0 and np.isfinite(values).all() and np.isfinite(target).all(), 'invalid natural evaluation')
    error = np.abs(np.exp2(target-values)-1)*100
    require(np.isfinite(error).all(), 'BPM metric overflow')
    # 4-second differences of local native-cell fields, fixed supported endpoints.
    dense_target = np.zeros(len(prediction), np.float64)
    dense_target[valid] = target
    supported = valid[:-200] & valid[200:]
    expected_change = -(dense_target[200:]-dense_target[:-200])[supported]
    actual_change = -(prediction[200:]-prediction[:-200])[supported]
    large = np.abs(expected_change) >= np.log2(1.1)
    changes = data.relative_metrics(actual_change[large], expected_change[large]) if large.any() else None
    return dict(cells=int(valid.sum()), mse=float(np.mean((values-target)**2)),
                tempo_median_error_percent=float(np.median(error)), tempo_p95_error_percent=float(np.percentile(error, 95)),
                large_change_4s=changes)


def evaluate(models, records, captures, points, device, output, check):
    natural, pairs = [], []
    before = {k: state_hash(v) for k, v in models.items()}
    for row in records:
        check()
        metrics, predictions = {}, {}
        for name, model in models.items():
            with torch.inference_mode():
                p = model(row['features'].to(device)[None])[0].cpu().numpy()
            predictions[name] = p
            metrics[name] = natural_metrics(p, row)
        path = output/(row['id']+'.natural.npz')
        with path.open('xb') as stream:
            np.savez(stream, **predictions)
        natural.append(dict(id=row['id'], role=row['role'], work=row['work'], metrics=metrics,
                            prediction_sha256=sha(path)))
    for identity in IDS:
        fields = {}
        for profile in ('source', *PROFILES):
            check()
            x = captures[identity, profile][None]
            for name, model in models.items():
                with torch.inference_mode():
                    fields[name, profile] = model(x)[0].cpu().numpy()
            with torch.inference_mode():
                fields['paired-zero', profile] = models['native-plus-relative'](torch.zeros_like(x))[0].cpu().numpy()
        path = output/(identity+'.paired-fields.npz')
        with path.open('xb') as stream:
            np.savez(stream, **{k[0]+'__'+k[1]: v for k, v in fields.items()})
        for profile in PROFILES:
            all_rows = [r for r in points if r['source_id'] == identity and r['profile'] == profile]
            admitted = [r for r in all_rows if r['admitted']]
            observed = {}
            for name in (*models, 'paired-zero'):
                observed[name] = -(data.query_numpy(fields[name, profile], [r['output_s'] for r in admitted])
                                   - data.query_numpy(fields[name, 'source'], [r['source_s'] for r in admitted]))
            target = np.log2([r['rate'] for r in admitted])
            changed = target != 0
            metrics = {name: dict(all=data.relative_metrics(p, target),
                                  changed=data.relative_metrics(p[changed], target[changed]) if changed.any() else None)
                       for name, p in observed.items()}
            pairs.append(dict(source_id=identity, profile=profile, role=admitted[0]['pair_role'],
                              total_grid_points=len(all_rows), admitted_points=len(admitted), metrics=metrics,
                              observations=[dict(index=r['index'], target_log2_rate=float(target[j]),
                                                 prediction={k: float(v[j]) for k, v in observed.items()})
                                            for j, r in enumerate(admitted)], prediction_sha256=sha(path)))
    require(before == {k: state_hash(v) for k, v in models.items()}, 'evaluation changed weights')
    return natural, pairs


def decision(natural, pairs):
    summary = {}
    for role in ('fit', 'development', 'diagnostic'):
        summary[role] = {}
        for name in ('initial', *ARMS):
            summary[role][name] = {}
            for metric in ('mse', 'tempo_median_error_percent', 'tempo_p95_error_percent'):
                groups = {}
                for r in natural:
                    if r['role'] == role:
                        groups.setdefault(r['work'], []).append(r['metrics'][name][metric])
                summary[role][name][metric] = work_mean(groups)
    identities, transfer = [], []
    for pair in pairs:
        p = pair['metrics']['native-plus-relative']
        if pair['profile'] == 'identity_seams':
            identities.append(p['all']['mean_abs_prediction'] <= .05)
        if pair['role'] == 'schedule_diagnostic':
            p, b = p['changed'], pair['metrics']['native-only']['changed']
            transfer.append(p is not None and p['rmse'] <= .75*p['zero_change_rmse'] and
                            p['rmse'] <= .8*b['rmse'] and .5 <= p['slope'] <= 1.5 and p['sign_correct'] >= .8)
    retention = all(summary[role]['native-plus-relative'][k] <= 1.05*summary[role][baseline][k]
                    for role in ('development', 'diagnostic') for baseline in ('initial', 'native-only')
                    for k in ('mse', 'tempo_median_error_percent'))
    require(len(identities) == 3 and len(transfer) == 6, 'decision population incomplete')
    return summary, dict(identity_passed=all(identities), schedule_transfer_passed=all(transfer),
                         schedule_pairs_passed=sum(transfer), natural_retention_passed=retention,
                         passed=all(identities) and all(transfer) and retention)


def run(args):
    started = time.monotonic()
    configure(args.device)
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'fresh private output required')
    if os.name == 'nt':
        require(args.output.resolve().drive.lower() != os.environ.get('SystemDrive', 'C:').lower(), 'use data drive')
    source = data.closure()
    points = data.population()
    manifest_path = ROOT/'experiments/coupled_clock/inputs-v1.json'
    require(sha(args.inputs/'inputs.json') == sha(manifest_path), 'natural input manifest changed')
    require(sha(args.weights) == data.WEIGHTS_SHA, 'starting weights changed')
    upstream = {p.relative_to(args.assets).as_posix(): sha(p) for p in (args.assets/'upstream').rglob('*.py')}
    dependencies = {p.relative_to(args.deps).as_posix(): sha(p) for p in args.deps.rglob('*.py')}
    require(bool(upstream) and bool(dependencies), 'missing pinned dependency sources')
    args.output.mkdir(parents=True)
    plan = dict(schema='rhythm-map.relative-tempo-plan.v1', source_sha256=source, points=points,
                natural_inputs_sha256=sha(manifest_path), initial_weights_sha256=data.WEIGHTS_SHA,
                encoder_checkpoint_sha256=data.CHECKPOINT_SHA, upstream_sha256=upstream, dependency_sha256=dependencies,
                arms=list(ARMS), seed=SEED, epochs=EPOCHS, steps_per_arm=STEPS,
                learning_rate=.0003, pair_coefficient=1., samples_per_group=4, selection='final_update_only',
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, device=args.device,
                holdout_access=False, production_change=False, independent_acceptance=False)
    write_json(args.output/'plan.json', plan)

    def check():
        require(time.monotonic()-started < 5300, 'global wall budget exceeded')

    try:
        sys.path.insert(0, str(args.deps))
        from .capture import capture
        captures, capture_report = capture(args.pairs, args.assets, args.output, args.device, check)
        records = load_records(args.inputs, json.loads(manifest_path.read_bytes()), args.device)
        initial = load_model(args.weights, args.device).eval()
        write_json(args.output/'initial.json', dict(state_sha256=state_hash(initial), parameters=sum(p.numel() for p in initial.parameters())))
        models, fits = {'initial': initial}, []
        for arm in ARMS:
            models[arm], report = fit(initial, records, captures, points, arm, args.output, check)
            fits.append(report)
        natural, pairs = evaluate(models, records, captures, points, args.device, args.output, check)
        summary, gate = decision(natural, pairs)
        require(data.closure() == source, 'experiment code changed during run')
        require(all(sha(args.assets/k) == v for k, v in upstream.items()) and
                all(sha(args.deps/k) == v for k, v in dependencies.items()), 'upstream source changed during run')
        require(fits[0]['initial_state_sha256'] == fits[1]['initial_state_sha256'] == state_hash(initial), 'arm initialization differs')
        events = [json.loads(line) for line in (args.output/'journal.jsonl').read_text().splitlines()]
        for arm in ARMS:
            for event in ('update_entered', 'update_returned'):
                require([e['update'] for e in events if e['arm'] == arm and e['event'] == event] == list(range(1, STEPS+1)),
                        'incomplete optimizer receipts')
        report = dict(schema='rhythm-map.relative-tempo-result.v1', completed=True, plan_sha256=sha(args.output/'plan.json'),
                      source_sha256=source, capture=capture_report, fits=fits, natural=natural, pairs=pairs,
                      summary=summary, gate=gate, elapsed_s=time.monotonic()-started, optimizer_steps=2*STEPS,
                      journal_sha256=sha(args.output/'journal.jsonl'), independent_acceptance=False,
                      holdout_access=False, production_change=False)
        write_json(args.output/'report.json', report)
        print(json.dumps(dict(event='completed', gate=gate, elapsed_s=report['elapsed_s'])), flush=True)
    except BaseException as error:
        write_json(args.output/'partial-report.json', dict(completed=False, failure_type=type(error).__name__,
                                                         elapsed_s=time.monotonic()-started))
        raise


if __name__ == '__main__':
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pairs', 'assets', 'deps', 'inputs', 'weights', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    run(parser.parse_args())
