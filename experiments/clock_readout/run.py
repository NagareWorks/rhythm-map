"""One bounded frozen-encoder/direct-readout experiment; private artifacts only."""
import argparse
import copy
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

import data
from model import ClockReadout, HALO, PARAMETERS
from prepare import write_json

CENTERS = 200
EPOCHS = 20
FIT_SECONDS = 1800


def read_pinned(path, expected):
    data.require(data.sha(path) == expected, 'input identity changed')
    return json.loads(Path(path).read_bytes())


def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def context(features, start):
    """Canonical raw-feature zero padding; central ownership is non-overlapping."""
    out = features.new_zeros((CENTERS + 2 * HALO, features.shape[1]))
    a, b = max(0, start - HALO), min(len(features), start + CENTERS + HALO)
    out[a - start + HALO:b - start + HALO] = features[a:b]
    return out


def predict(model, features, zero_audio=False):
    model.eval()
    rows = []
    with torch.inference_mode():
        for start, end in data.windows(len(features)):
            x = context(features, start)
            if zero_audio:
                x.zero_()
            rows.append(model(x[None])[0, HALO:HALO + end - start].cpu().numpy())
    return np.concatenate(rows)


def supervised_loss(output, target):
    return (output[..., 0] - target[..., 0]).square() + (output[..., 1:] - target[..., 1:]).square().mean(dim=-1)


def development_loss(model, records, zero_audio):
    totals = {}
    for row in records:
        if row['role'] != 'development':
            continue
        prediction = predict(model, row['features'], zero_audio)
        y, valid = row['target_numpy'], row['valid_numpy']
        error = (prediction[:, 0] - y[:, 0]) ** 2 + ((prediction[:, 1:] - y[:, 1:]) ** 2).mean(axis=1)
        current = totals.setdefault(row['work'], [0., 0])
        current[0] += float(error[valid].sum())
        current[1] += int(valid.sum())
    data.require(len(totals) == 3 and all(v[1] > 0 for v in totals.values()), 'development group lacks labels')
    value = float(np.mean([a / b for a, b in totals.values()]))
    data.require(np.isfinite(value), 'nonfinite development loss')
    return value


def fit(records, device, zero_audio, output):
    torch.manual_seed(data.SEED)
    np.random.seed(data.SEED)
    model = ClockReadout().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    population, denominators = [], {}
    for row in records:
        if row['role'] != 'fit':
            continue
        denominators[row['work']] = denominators.get(row['work'], 0) + int(row['valid_numpy'].sum())
        population.extend((row, a, b) for a, b in data.windows(len(row['features'])) if row['valid_numpy'][a:b].any())
    data.require(len(denominators) == 9 and all(v > 0 for v in denominators.values()), 'fit group lacks labels')
    started = time.monotonic()
    best, best_epoch, best_loss, history = None, None, float('inf'), []
    exhausted = False
    for epoch in range(EPOCHS):
        model.train()
        order = np.random.default_rng(data.SEED + epoch).permutation(len(population))
        running = []
        for offset in range(0, len(order), 4):
            if time.monotonic() - started >= FIT_SECONDS:
                exhausted = True
                break
            entries = [population[i] for i in order[offset:offset + 4]]
            x = torch.stack([context(row['features'], a) for row, a, _ in entries])
            if zero_audio:
                x.zero_()
            prediction = model(x)[:, HALO:HALO + CENTERS]
            terms = []
            for p, (row, a, b) in zip(prediction, entries):
                error = supervised_loss(p[:b-a], row['target'][a:b])
                terms.append(error[row['valid'][a:b]].sum() / denominators[row['work']])
            # Equal expected contribution from each work, not each crop/frame.
            loss = torch.stack(terms).sum() * len(population) / (len(denominators) * len(entries))
            data.require(torch.isfinite(loss).item(), 'nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            running.append(float(loss.detach()))
        if exhausted:
            break
        score = development_loss(model, records, zero_audio)
        history.append(dict(epoch=epoch + 1, training_loss_mean=float(np.mean(running)), development_loss=score))
        if score < best_loss:  # Strict comparison preserves earlier ties.
            best, best_epoch, best_loss = copy.deepcopy(model.state_dict()), epoch + 1, score
        print(json.dumps(dict(fit='zero_audio' if zero_audio else 'audio', **history[-1])), flush=True)
    data.require(best is not None, 'no complete checkpoint before budget exhausted')
    model.load_state_dict(best)
    name = 'zero-audio' if zero_audio else 'audio'
    path = output / (name + '.weights.npz')
    with path.open('xb') as stream:
        np.savez(stream, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    return model, dict(kind=name, parameters=PARAMETERS, seed=data.SEED, epochs=len(history),
        budget_exhausted=exhausted, selected_epoch=best_epoch, development_loss=best_loss,
        elapsed_s=time.monotonic() - started, history=history, weight_sha256=data.sha(path))


def capture_features(args, cases, output):
    """Same final0 checkpoint and chunk ownership; one GPU forward per chunk."""
    sys.path.insert(0, str(args.upstream.resolve()))
    sys.path.insert(0, str(data.ROOT / 'evaluation/parity'))
    from beat_this.model.beat_tracker import BeatThis
    from beat_this.utils import replace_state_dict_key
    import prehead_capture as geometry
    data.require(data.sha(args.checkpoint) == '8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331', 'checkpoint changed')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    hparams = {k: v for k, v in checkpoint['hyper_parameters'].items() if k in inspect.signature(BeatThis).parameters}
    encoder = BeatThis(**hparams).eval().to(args.device)
    encoder.load_state_dict(replace_state_dict_key(checkpoint['state_dict'], 'model.', ''))
    del checkpoint
    for p in encoder.parameters():
        p.requires_grad_(False)
    before = state_hash(encoder)
    captured, records = [], []
    begin = time.monotonic()
    for item in cases:
        data.require(time.monotonic() - begin < 1800, 'encoder GPU capture budget exceeded')
        path = args.inputs / (item['id'] + '.input.npz')
        data.require(data.sha(path) == item['input_sha256'], 'input archive changed')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        mel = arrays['mel']
        chunks, forward_errors = [], []
        for piece in geometry.split(mel):
            hidden = []
            handle = encoder.task_heads.register_forward_pre_hook(lambda module, inp: hidden.append(inp[0].detach().clone()))
            try:
                with torch.inference_mode():
                    heads = encoder(torch.from_numpy(piece.copy()).to(args.device)[None])
                    data.require(len(hidden) == 1, 'nonunique hidden export')
                    recreated = encoder.task_heads(hidden[0])
                    for key in ('beat', 'downbeat'):
                        data.require(torch.allclose(heads[key], recreated[key], atol=2e-5, rtol=1e-6), 'head reconstruction changed')
                    chunks.append(dict(hidden=hidden[0][0].cpu().numpy(), **{k: heads[k][0].cpu().numpy() for k in ('beat', 'downbeat')}))
            finally:
                handle.remove()
        result = geometry.stitch(len(mel), chunks)
        for key in ('beat', 'downbeat'):
            check = geometry.difference(result[key], arrays[key], atol=.002, rtol=.0001)
            forward_errors.append(check['max_abs'])
        archive_path = output / (item['id'] + '.features.npz')
        with archive_path.open('xb') as stream:
            np.savez(stream, **result)
        captured.append(dict(id=item['id'], frames=len(mel), input_sha256=item['input_sha256'],
            feature_sha256=data.sha(archive_path), max_logit_error_vs_rten=max(forward_errors)))
        records.append(dict(item, features=torch.from_numpy(result['hidden']).to(args.device),
            target=torch.from_numpy(arrays['target']).to(args.device), valid=torch.from_numpy(arrays['valid']).to(args.device),
            target_numpy=arrays['target'], valid_numpy=arrays['valid'], baseline=arrays['baseline'], baseline_valid=arrays['baseline_valid']))
        print('Captured ' + item['id'], flush=True)
    data.require(state_hash(encoder) == before, 'encoder state changed')
    del encoder
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    return records, dict(cases=captured, elapsed_s=time.monotonic() - begin, encoder_state_unchanged=True, state_sha256=before)


def aggregate(rows, name, keys=('tempo_median_error_percent', 'tempo_p95_error_percent', 'phase_mean_absolute_cycles')):
    # Each work contributes equally, with equal recording weight within work.
    result = {}
    for key in keys:
        groups = {}
        for row in rows:
            groups.setdefault(row['work'], []).append(row[name][key])
        result[key] = (float(np.mean([np.mean(v) for v in groups.values()]))
                       if groups and all(x is not None and np.isfinite(x) for v in groups.values() for x in v) else None)
    return result


def coherence(prediction, valid):
    """Describe incompatible heads; never integrate away their disagreement."""
    phase = np.arctan2(prediction[:, 2], prediction[:, 1])
    amplitude = np.hypot(prediction[:, 1], prediction[:, 2])
    forward = np.angle(np.exp(1j * np.diff(phase))) / (2 * np.pi)
    with np.errstate(over='ignore', invalid='ignore'):
        expected = np.exp2(-prediction[:-1, 0]) / data.FPS
    reference = valid[:-1] & valid[1:]
    available = (reference & np.isfinite(prediction[:-1]).all(axis=1) & np.isfinite(prediction[1:]).all(axis=1)
                 & (amplitude[:-1] > 1e-6) & (amplitude[1:] > 1e-6) & np.isfinite(expected))
    n = int(reference.sum())
    complete = n > 0 and np.all(available[reference])
    return dict(reference_adjacent_frames=n, available_adjacent_frames=int(available.sum()),
        forward_phase_fraction=float(np.mean(forward[reference] > 0)) if complete else None,
        mean_phase_period_disagreement_cycles_per_frame=float(np.mean(np.abs(forward[reference] - expected[reference]))) if complete else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'upstream', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    data.require(not args.output.exists() and not args.output.resolve().is_relative_to(data.ROOT), 'new private output outside code required')
    original = json.loads((args.inputs / 'inputs.json').read_bytes())
    recipe = read_pinned(args.inputs / 'plan.json', original['plan_sha256'])
    data.require(original['complete'] and len(original['cases']) == 40 and not recipe['holdout_access'], 'incomplete or forbidden population')
    for name, expected in recipe['source_sha256'].items():
        data.require(data.sha(Path(__file__).parent / name) == expected, 'registered preparation code changed')
    data.require([(r['id'], r['role'], r['work']) for r in original['cases']] ==
                 [(r['id'], r['role'], r['work']) for r in recipe['cases']], 'split changed')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda':
        data.require(torch.cuda.is_available(), 'CUDA unavailable')
        torch.cuda.set_per_process_memory_fraction(.4)
    args.output.mkdir()
    write_json(args.output / 'execution-plan.json', dict(inputs_sha256=data.sha(args.inputs / 'inputs.json'),
        plan_sha256=original['plan_sha256'], runner_sha256=data.sha(__file__), torch_version=torch.__version__,
        numpy_version=np.__version__, device=args.device, deterministic=True, tf32=False,
        fits=2, max_epochs=EPOCHS, max_seconds_per_fit=FIT_SECONDS, parameters=PARAMETERS))
    records, captured = capture_features(args, original['cases'], args.output)
    write_json(args.output / 'capture.json', captured)
    audio, main_fit = fit(records, args.device, False, args.output)
    zero, zero_fit = fit(records, args.device, True, args.output)
    rows = []
    for row in records:
        predictions = dict(audio=predict(audio, row['features']), zero=predict(zero, row['features'], True))
        common = row['valid_numpy'] & row['baseline_valid']
        result = dict(id=row['id'], role=row['role'], work=row['work'], frames=row['frames'],
            reference_frames=int(row['valid_numpy'].sum()), raw_supported_reference_frames=int(common.sum()))
        for kind, prediction in predictions.items():
            result[kind] = data.metrics(prediction, row['target_numpy'], row['valid_numpy'])
            result[kind + '_common'] = data.metrics(prediction, row['target_numpy'], common)
            result[kind + '_coherence'] = coherence(prediction, row['valid_numpy'])
        result['raw_common'] = data.metrics(row['baseline'], row['target_numpy'], common)
        result['regressions_vs_raw_common'] = {
            key: (None if result['audio_common'][key] is None or result['raw_common'][key] is None else
                  result['audio_common'][key] > result['raw_common'][key])
            for key in ('tempo_median_error_percent', 'tempo_p95_error_percent', 'phase_mean_absolute_cycles')}
        with (args.output / (row['id'] + '.prediction.npz')).open('xb') as stream:
            np.savez(stream, **predictions)
        rows.append(result)
    development = [r for r in rows if r['role'] == 'development']
    summaries = {role: {kind: aggregate([r for r in rows if r['role'] == role], kind)
                       for kind in ('audio', 'zero', 'audio_common', 'raw_common')}
                 for role in ('fit', 'development', 'diagnostic')}
    s = summaries['development']
    keys = ('tempo_median_error_percent', 'phase_mean_absolute_cycles')
    complete = all(r['audio']['period_available_frames'] == r['audio']['phase_available_frames'] == r['reference_frames'] for r in development)
    passed = (complete and not main_fit['budget_exhausted'] and not zero_fit['budget_exhausted'] and
        all(s['audio'][k] is not None and s['zero'][k] is not None and s['audio_common'][k] is not None and
            s['raw_common'][k] is not None and s['audio'][k] <= .9 * s['zero'][k] and
            s['audio_common'][k] <= s['raw_common'][k] for k in keys))
    report = dict(schema='rhythm-map.direct-clock-learning.v1', plan=recipe,
        inputs_sha256=data.sha(args.inputs / 'inputs.json'), capture=captured, fits=[main_fit, zero_fit],
        cases=rows, summary=summaries, development_gate_passed=bool(passed), training=True,
        independent_acceptance=False, holdout_access=False, production_change=False,
        decision='independent_coherence_and_product_gates_required' if passed else 'close_this_fixed_direct_readout_no_automatic_sweep')
    write_json(args.output / 'report.json', report)
    print(json.dumps(dict(summary=summaries, development_gate_passed=bool(passed))), flush=True)
    print('DIRECT_CLOCK_EXPERIMENT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
