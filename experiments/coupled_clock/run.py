"""Execute the pre-registered pair of complete-crop fits, once, on cached features."""
import argparse
import copy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.coupled_clock.clock import require
from experiments.coupled_clock.model import CoupledClockReadout, PARAMETERS
from experiments.coupled_clock.supervision import clock_loss
from experiments.coupled_clock.measurement import clock_fields, independent_fields, measure, macro, regressions, gates
from experiments.coupled_clock.prepare import ROOT, OLD_INPUTS, sha, write_json, source_hashes

SEED, EPOCHS, SECONDS, BATCH = 142, 20, 1800, 4
KINDS = ('audio', 'zero', 'audio_common', 'zero_common', 'raw_common', 'v1_common')


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def work_weights(records):
    counts = Counter(r['work'] for r in records)
    require(bool(counts), 'empty work population')
    return {work: len(records) / (len(counts) * count) for work, count in counts.items()}


def work_mean(values):
    require(bool(values) and all(v and np.isfinite(v).all() for v in values.values()), 'invalid work loss')
    return float(np.mean([np.mean(v) for v in values.values()]))


def input_tensor(row, zero_audio):
    x = row['features'][None]
    return torch.zeros_like(x) if zero_audio else x


def development_loss(model, records, zero_audio):
    totals = {}
    model.eval()
    with torch.inference_mode():
        for row in records:
            if row['role'] != 'development':
                continue
            result = clock_loss(model(input_tensor(row, zero_audio)), row['target'][None], row['mask'][None])
            totals.setdefault(row['work'], []).append(float(result['total']))
    require(len(totals) == 3, 'development work split changed')
    return work_mean(totals)


def journal(output, value):
    # An append-only execution artifact, not an overwriteable checkpoint log.
    with (output / 'journal.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(value, allow_nan=False), flush=True)


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


def load_records(inputs, manifest, device):
    original = ROOT / 'experiments/clock_readout/inputs-v1.json'
    require(sha(original) == OLD_INPUTS, 'original population identity changed')
    cases = json.loads(original.read_bytes())['cases']
    require(manifest['complete'] and not manifest['holdout_access'] and
            [(r['id'], r['role'], r['work'], r['frames'], r['duration_s']) for r in manifest['cases']] ==
            [(r['id'], r['role'], r['work'], r['frames'], r['duration_s']) for r in cases], 'population changed')
    records = []
    for item in manifest['cases']:
        path = inputs / (item['id'] + '.npz')
        require(sha(path) == item['packet_sha256'], 'packet identity changed')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        require(arrays['hidden'].shape == (item['frames'], 512) and arrays['hidden'].dtype == np.float32 and
                np.isfinite(arrays['hidden']).all(), 'invalid hidden fields')
        row = dict(item, **arrays)
        # Diagnostics remain CPU-only until fitting AND checkpoint choice finish.
        target_device = device if item['role'] != 'diagnostic' else 'cpu'
        row.update(features=torch.from_numpy(arrays['hidden']).to(target_device),
                   target=torch.from_numpy(arrays['reference']).to(target_device),
                   mask=torch.from_numpy(arrays['valid']).to(target_device))
        records.append(row)
    return records


def evaluate(models, records, device, output):
    rows = []
    for row in records:
        features = row['features'].to(device)
        predictions = {}
        with torch.inference_mode():
            for name, model in models.items():
                x = torch.zeros_like(features) if name == 'zero' else features
                predictions[name] = model(x[None]).cycles[0].cpu().numpy()
        common = row['valid'] & row['raw_valid']
        result = dict(id=row['id'], role=row['role'], work=row['work'], frames=row['frames'])
        for name, prediction in predictions.items():
            result[name] = measure(clock_fields(prediction), row['reference'], row['valid'])
            result[name + '_common'] = measure(clock_fields(prediction), row['reference'], common)
        result['raw_common'] = measure(clock_fields(row['raw']), row['reference'], common)
        result['v1_common'] = measure(independent_fields(row['v1']), row['reference'], common)
        for baseline in ('raw_common', 'v1_common'):
            result['regressions_vs_' + baseline] = regressions(result, baseline)
        path = output / (row['id'] + '.prediction.npz')
        with path.open('xb') as stream:
            np.savez(stream, **predictions)
        result['prediction_sha256'] = sha(path)
        rows.append(result)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--expected-inputs-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'new private output required')
    require(sha(args.inputs / 'inputs.json') == args.expected_inputs_sha256, 'manifest changed')
    manifest = json.loads((args.inputs / 'inputs.json').read_bytes())
    require(source_hashes() == manifest['source_sha256'], 'registered code or protocol changed')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda':
        require(torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUDA/determinism unavailable')
        torch.cuda.set_per_process_memory_fraction(.4)
    args.output.mkdir()
    write_json(args.output / 'execution.json', dict(schema='rhythm-map.coupled-clock-execution.v1',
        inputs_sha256=args.expected_inputs_sha256, source_sha256=manifest['source_sha256'],
        torch_version=torch.__version__, numpy_version=np.__version__, device=args.device,
        deterministic=True, tf32=False, fits=2, seed=SEED, max_epochs=EPOCHS, max_seconds_per_fit=SECONDS,
        batch_recordings=BATCH, parameters=PARAMETERS, holdout_access=False, production_change=False))
    try:
        records = load_records(args.inputs, manifest, args.device)
        audio, audio_fit = fit(records, args.device, False, args.output)
        zero, zero_fit = fit(records, args.device, True, args.output)
        require(audio_fit['initial_state_sha256'] == zero_fit['initial_state_sha256'], 'control initialization differs')
        rows = evaluate(dict(audio=audio, zero=zero), records, args.device, args.output)
        summaries = {role: {kind: macro([r for r in rows if r['role'] == role], kind) for kind in KINDS}
                     for role in ('fit', 'development', 'diagnostic')}
        gate = gates(rows, summaries, [audio_fit, zero_fit])
        report = dict(schema='rhythm-map.coupled-clock-learning.v1', inputs_sha256=args.expected_inputs_sha256,
                      execution_sha256=sha(args.output / 'execution.json'), fits=[audio_fit, zero_fit],
                      cases=rows, summary=summaries, gate=gate, training=True, independent_acceptance=False,
                      holdout_access=False, production_change=False,
                      decision='fresh_independent_validation_required' if gate['passed'] else 'close_fixed_coupled_fit_no_automatic_sweep')
        write_json(args.output / 'report.json', report)
        journal(args.output, dict(event='experiment_complete', gate=gate))
    except Exception as error:
        # Keep partial journals/checkpoints and a small public-safe failure type.
        write_json(args.output / 'failure.json', dict(event='experiment_failed', exception_type=type(error).__name__,
                                                     automatic_retry=False, production_change=False))
        raise


if __name__ == '__main__':
    main()
