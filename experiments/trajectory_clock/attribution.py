"""Truth-assisted error attribution of retained predictions, without new fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT_SHA = 'b4f05c10ebede84c394b0b8e06f6723b096d1f9a0a48e63024eb81d913fffd9c'
REPORT_SHA = 'e33a44db4ce20a35f0676aa3f4fc47806da83f4a2ab80c1cfb555ce5343cda9a'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def wrap(value):
    return (value + .5) % 1 - .5


def circular_median(error):
    """Exact circular L1 minimum over ONE constant shift, O(N log N).

    A piecewise-linear circular-distance sum has a minimum at a data phase
    (or on a flat interval with such an endpoint). Antipodes are downward
    slope changes, not strict minima. A doubled sorted array/prefix sum
    scores all data phases. Recompute the winner directly to avoid cancellation.
    """
    error = np.asarray(error, dtype=np.float64)
    require(error.ndim == 1 and len(error) > 0 and np.isfinite(error).all(), 'invalid phase errors')
    phase = np.sort(error % 1)
    n = len(phase)
    doubled = np.r_[phase, phase + 1]
    prefix = np.r_[0., np.cumsum(doubled)]
    start = np.arange(n)
    middle = np.minimum(np.searchsorted(doubled, phase + .5, side='right'), start + n)
    left = prefix[middle] - prefix[start] - (middle - start) * phase
    right = (start + n - middle) * (phase + 1) - (prefix[start + n] - prefix[middle])
    shift = float(phase[int(np.argmin(left + right))])
    return dict(shift_cycles=shift, mean_absolute_cycles=float(np.abs(wrap(error - shift)).mean()))


def spans(valid):
    changes = np.diff(np.r_[False, valid, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def attribute(prediction, reference, valid):
    prediction, reference, valid = np.asarray(prediction), np.asarray(reference), np.asarray(valid)
    require(valid.dtype == bool and valid.ndim == 1 and prediction.shape == reference.shape == valid.shape,
            'attribution geometry mismatch')
    require(valid.any() and np.isfinite(prediction[valid]).all() and np.isfinite(reference[valid]).all(),
            'missing required predictions or reference')
    blocks = spans(valid)
    error = prediction[valid].astype(np.float64) - reference[valid].astype(np.float64)
    oracle = circular_median(error)
    centered_energy, linear_energy, residual_energy, block_rows = 0., 0., 0., []
    for a, b in blocks:
        e = prediction[a:b].astype(np.float64) - reference[a:b].astype(np.float64)
        centered = e - e.mean()
        time = np.arange(len(e), dtype=np.float64) / 50
        time -= time.mean()
        denominator = float(time @ time)
        slope = float((time @ centered) / denominator) if denominator > 0 else 0.
        linear = slope * time
        residual = centered - linear
        centered_energy += float(centered @ centered)
        linear_energy += float(linear @ linear)
        residual_energy += float(residual @ residual)
        block_rows.append(dict(start_frame=int(a), end_frame_exclusive=int(b), points=int(b - a),
                               native_rate_bias_bpm=60 * slope if b - a > 1 else None,
                               terminal_drift_cycles=float(e[-1] - e[0]),
                               max_first_aligned_drift_cycles=float(np.max(np.abs(e - e[0])))))
    require(np.isclose(centered_energy, linear_energy + residual_energy, atol=1e-9, rtol=1e-10), 'OLS decomposition failed')
    n = len(error)
    # No count bridge across annotation gaps. Singleton spans cannot measure drift.
    has_intervals = any(b - a > 1 for a, b in blocks)
    return dict(points=n, reference_spans=block_rows,
                observed_phase_error_cycles=float(np.abs(wrap(error)).mean()),
                first_supported_phase_error_cycles=float(abs(wrap(error[0]))),
                first_supported_aligned_phase_error_cycles=float(np.abs(wrap(error - error[0])).mean()),
                best_constant_phase_error_cycles=oracle['mean_absolute_cycles'],
                oracle_shift_cycles=oracle['shift_cycles'],
                minimum_circular_training_loss=float(1 - abs(np.exp(2j * np.pi * (error % 1)).mean())),
                centered_clock_rmse_cycles=float(np.sqrt(centered_energy / n)) if has_intervals else None,
                constant_rate_component_rmse_cycles=float(np.sqrt(linear_energy / n)) if has_intervals else None,
                nonlinear_residual_rmse_cycles=float(np.sqrt(residual_energy / n)) if has_intervals else None)


SUMMARY_KEYS = ('observed_phase_error_cycles', 'best_constant_phase_error_cycles',
                'minimum_circular_training_loss', 'centered_clock_rmse_cycles',
                'constant_rate_component_rmse_cycles', 'nonlinear_residual_rmse_cycles')


def summarize(rows, kind):
    result = {}
    for key in SUMMARY_KEYS:
        groups = {}
        for row in rows:
            groups.setdefault(row['work'], []).append(row[kind][key])
        result[key] = (float(np.mean([np.mean(v) for v in groups.values()]))
                       if groups and all(x is not None for v in groups.values() for x in v) else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'new private output required')
    source = ROOT / 'experiments/coupled_clock/results-v1.json'
    require(sha(source) == REPORT_SHA and sha(args.inputs / 'inputs.json') == INPUT_SHA, 'fixed experiment changed')
    previous = json.loads(source.read_bytes())
    manifest = json.loads((args.inputs / 'inputs.json').read_bytes())
    require(len(manifest['cases']) == len(previous['cases']) == 40, 'population changed')
    rows = []
    for item, old in zip(manifest['cases'], previous['cases']):
        require(item['id'] == old['id'], 'case order changed')
        packet = args.inputs / (item['id'] + '.npz')
        prediction = args.predictions / (item['id'] + '.prediction.npz')
        require(sha(packet) == item['packet_sha256'] and sha(prediction) == old['prediction_sha256'], 'private bytes changed')
        with np.load(packet, allow_pickle=False) as archive:
            reference, valid = archive['reference'], archive['valid'] & archive['raw_valid']
        result = dict(id=item['id'], role=item['role'], work=item['work'])
        with np.load(prediction, allow_pickle=False) as archive:
            for kind in ('audio', 'zero'):
                result[kind] = attribute(archive[kind], reference, valid)
        for baseline in ('raw_common', 'v1_common'):
            value = old[baseline]['phase_mean_absolute_cycles']
            result[baseline + '_phase_error_cycles'] = value
            result['oracle_anchor_still_worse_than_' + baseline] = result['audio']['best_constant_phase_error_cycles'] > value
        require(abs(result['audio']['observed_phase_error_cycles'] - old['audio_common']['phase_mean_absolute_cycles']) < 1e-12,
                'original phase metric changed')
        rows.append(result)
    report = dict(schema='rhythm-map.clock-error-attribution.v1', old_inputs_sha256=INPUT_SHA, old_report_sha256=REPORT_SHA,
                  source_sha256=sha(__file__), post_fit_truth_assisted=True, training=False, holdout_access=False,
                  product_change=False, independent_acceptance=False, cases=rows,
                  summary={role: {kind: summarize([r for r in rows if r['role'] == role], kind) for kind in ('audio', 'zero')}
                  for role in ('fit', 'development', 'diagnostic')})
    write_json(args.output, report)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
