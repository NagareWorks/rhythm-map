"""Reuse pinned v1 captures; prepare a private, immutable complete-crop bundle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.clock_readout.model import ClockReadout, HALO
from experiments.coupled_clock.clock import require
from experiments.coupled_clock.supervision import reference_clock

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD_INPUTS = '2e931a64145b1e425803bc08b7eb9d868f4c41b2d8ddea5a90b455c1f0a292df'
OLD_REPORT = '990f7a2c71b2d3d61ab77c81daa69c3c5add3a78ead43d776fbf03d25ed07736'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def source_hashes():
    paths = sorted(HERE.glob('*.py')) + [HERE / 'PROTOCOL-v1.md', ROOT / 'experiments/clock_readout/model.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def replay_v1(model, hidden):
    """Unchanged v1 raw-feature halo/200-frame central ownership, no new fit."""
    rows = []
    with torch.inference_mode():
        for start in range(0, len(hidden), 200):
            end = min(start + 200, len(hidden))
            x = np.zeros((200 + 2 * HALO, 512), np.float32)
            a, b = max(0, start - HALO), min(len(hidden), start + 200 + HALO)
            x[a - start + HALO:b - start + HALO] = hidden[a:b]
            rows.append(model(torch.from_numpy(x)[None])[0, HALO:HALO + end - start].numpy())
    return np.concatenate(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-inputs', 'old-results', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--trace-root', type=Path, action='append', required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'new private output required')
    require(sha(args.old_inputs / 'inputs.json') == OLD_INPUTS and sha(args.old_results / 'report.json') == OLD_REPORT,
            'v1 source manifest changed')
    inputs = json.loads((args.old_inputs / 'inputs.json').read_bytes())
    old = json.loads((args.old_results / 'report.json').read_bytes())
    cases = inputs['cases']
    require(inputs['complete'] and len(cases) == 40 and not inputs['holdout_access'], 'incomplete or forbidden population')
    capture = {r['id']: r for r in old['capture']['cases']}
    weights = args.old_results / 'audio.weights.npz'
    require(sha(weights) == old['fits'][0]['weight_sha256'], 'v1 weights changed')
    torch.set_num_threads(2)
    model = ClockReadout().eval()
    with np.load(weights, allow_pickle=False) as archive:
        model.load_state_dict({k: torch.from_numpy(archive[k].copy()) for k in archive.files})
    args.output.mkdir()
    prepared = []
    for row in cases:
        identity = row['id']
        features = args.old_results / (identity + '.features.npz')
        truth = ROOT / row['truth']
        require(sha(features) == capture[identity]['feature_sha256'] and sha(truth) == row['truth_sha256'], 'features or truth changed')
        matches = [p for root in args.trace_root for p in (root / (identity + '.trace.private.json'), root / (identity + '.json'))
                   if p.is_file() and sha(p) == row['trace_sha256']]
        require(bool(matches), 'missing pinned trace: ' + identity)
        trace = json.loads(matches[0].read_bytes())
        reference = reference_clock([b['time_s'] for b in json.loads(truth.read_bytes())['beats']], row['frames'], row['duration_s'])
        raw = reference_clock([b['time_s'] for b in trace['observations']['beats']], row['frames'], row['duration_s'])
        with np.load(features, allow_pickle=False) as archive:
            hidden = archive['hidden'].copy()
        require(hidden.dtype == np.float32 and hidden.shape == (row['frames'], 512) and np.isfinite(hidden).all(), 'invalid features')
        previous = replay_v1(model, hidden)
        path = args.output / (identity + '.npz')
        with path.open('xb') as stream:
            np.savez(stream, hidden=hidden, reference=reference.cycles, valid=reference.valid,
                     raw=raw.cycles, raw_valid=raw.valid, v1=previous)
        prepared.append(dict(id=identity, role=row['role'], work=row['work'], frames=row['frames'], duration_s=row['duration_s'],
                             packet_sha256=sha(path), feature_sha256=sha(features), truth_sha256=sha(truth),
                             trace_sha256=row['trace_sha256'], original_input_sha256=row['input_sha256']))
        print('Prepared ' + identity, flush=True)
    manifest = dict(schema='rhythm-map.coupled-clock-inputs.v1', old_inputs_sha256=OLD_INPUTS,
                    old_report_sha256=OLD_REPORT, v1_weight_sha256=sha(weights), source_sha256=source_hashes(),
                    cases=prepared, complete=True, holdout_access=False, training=False)
    write_json(args.output / 'inputs.json', manifest)
    print('COUPLED_INPUTS_READY ' + sha(args.output / 'inputs.json'), flush=True)


if __name__ == '__main__':
    main()
