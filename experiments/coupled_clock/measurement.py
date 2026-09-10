"""Fixed cell/phase/count measurements; never infer extra beat observations."""
import numpy as np

from experiments.coupled_clock.clock import FPS, require

KEYS = ('tempo_median_error_percent', 'tempo_p95_error_percent',
        'phase_mean_absolute_cycles', 'count_4s_mean_absolute_cycles')


def intervals(valid, lag):
    valid = np.asarray(valid)
    require(valid.ndim == 1 and valid.dtype == bool and type(lag) is int and lag > 0,
            'invalid interval mask')
    if lag >= len(valid):
        return np.zeros(0, dtype=bool)
    missing = np.r_[0, np.cumsum(~valid)]
    return (missing[lag + 1:] - missing[:-(lag + 1)]) == 0


def clock_fields(cycles):
    cycles = np.asarray(cycles, dtype=np.float64)
    require(cycles.ndim == 1 and len(cycles) >= 2, 'invalid clock geometry')
    return np.diff(cycles) * FPS, cycles


def independent_fields(prediction):
    """Declared trapezoidal cell-rate projection; keep original head phase."""
    prediction = np.asarray(prediction, dtype=np.float64)
    require(prediction.ndim == 2 and prediction.shape[1] == 3, 'invalid v1 fields')
    with np.errstate(over='ignore', invalid='ignore'):
        rates = np.exp2(-prediction[:, 0])
        cells = rates[:-1] / 2 + rates[1:] / 2
        phase = np.arctan2(prediction[:, 2], prediction[:, 1]) / (2 * np.pi)
    amplitude = np.hypot(prediction[:, 1], prediction[:, 2])
    phase[(amplitude <= 1e-6) | ~np.isfinite(amplitude)] = np.nan
    return cells, phase


def measure(fields, reference, valid):
    rate, phase = fields
    reference = np.asarray(reference, dtype=np.float64)
    valid = np.asarray(valid)
    require(valid.dtype == bool and reference.ndim == 1 and valid.shape == reference.shape
            and rate.shape == (len(reference) - 1,) and phase.shape == reference.shape,
            'measurement geometry mismatch')
    cells, spans = intervals(valid, 1), intervals(valid, 200)
    expected_rate = np.diff(reference) * FPS
    require(np.isfinite(reference[valid]).all() and
            np.isfinite(expected_rate[cells]).all() and (expected_rate[cells] > 0).all(),
            'invalid measurement reference')
    finite_rate = np.isfinite(rate) & (rate > 0)
    finite_phase = np.isfinite(phase)
    # Zero filling only prevents NaN poisoning outside supported intervals.
    # Required bad cells invalidate metrics; they are never removed from masks.
    integrated = np.r_[0., np.cumsum(np.where(finite_rate, rate / FPS, 0.))]
    missing = np.r_[0, np.cumsum(~finite_rate)]
    available_spans = ((missing[200:] - missing[:-200]) == 0) if len(rate) >= 200 else np.zeros(0, bool)
    out = dict(reference_points=int(valid.sum()), available_points=int((valid & finite_phase).sum()),
               reference_cells=int(cells.sum()), available_cells=int((cells & finite_rate).sum()),
               reference_4s_intervals=int(spans.sum()),
               available_4s_intervals=int((spans & available_spans).sum()))
    out.update({key: None for key in KEYS})
    if cells.any() and finite_rate[cells].all():
        error = 100 * np.abs(rate[cells] - expected_rate[cells]) / expected_rate[cells]
        if np.isfinite(error).all():
            out[KEYS[0]], out[KEYS[1]] = float(np.median(error)), float(np.percentile(error, 95))
    if valid.any() and finite_phase[valid].all():
        out[KEYS[2]] = float(np.abs((phase[valid] - reference[valid] + .5) % 1 - .5).mean())
    if spans.any() and available_spans[spans].all():
        error = ((integrated[200:] - integrated[:-200]) -
                 (reference[200:] - reference[:-200]))[spans]
        if np.isfinite(error).all():
            out[KEYS[3]] = float(np.abs(error).mean())
    return out


def macro(rows, name):
    result = {}
    for key in KEYS:
        groups = {}
        for row in rows:
            groups.setdefault(row['work'], []).append(row[name][key])
        values = [v for group in groups.values() for v in group]
        result[key] = (float(np.mean([np.mean(v) for v in groups.values()]))
                       if values and all(v is not None and np.isfinite(v) for v in values) else None)
    return result


def regressions(row, baseline):
    return {k: (None if row['audio_common'][k] is None or row[baseline][k] is None else
                row['audio_common'][k] > row[baseline][k]) for k in KEYS}


def gates(rows, summaries, fits):
    def finite_comparison(a, b, factor=1.):
        return a is not None and b is not None and np.isfinite(a) and np.isfinite(b) and a <= factor * b

    required = [r for r in rows if r['role'] in ('development', 'diagnostic')]
    complete = bool(required) and all(
        r[name][k] is not None and np.isfinite(r[name][k])
        for r in required for name in ('audio', 'zero', 'audio_common', 'zero_common', 'raw_common', 'v1_common')
        for k in KEYS)
    finished = len(fits) == 2 and all(f['epochs'] == 20 and not f['budget_exhausted'] for f in fits)
    dev = summaries['development']
    signal = all(finite_comparison(dev['audio'][k], dev['zero'][k], .9) for k in (KEYS[0], KEYS[2]))
    nonregression = all(finite_comparison(summaries[role]['audio_common'][k], summaries[role][baseline][k])
                        for role in ('development', 'diagnostic') for baseline in ('raw_common', 'v1_common') for k in KEYS)
    breadth = True
    for role in ('development', 'diagnostic'):
        subset = [r for r in rows if r['role'] == role]
        breadth &= bool(subset) and all(
            sum(regressions(r, 'raw_common')[k] is True for r in subset) * 2 <= len(subset)
            and all(regressions(r, 'raw_common')[k] is not None for r in subset)
            for k in (KEYS[0], KEYS[2]))
    result = dict(complete=bool(complete), finished=bool(finished), audio_signal=bool(signal),
                  common_nonregression=bool(nonregression), recording_breadth=bool(breadth))
    result['passed'] = all(result.values())
    return result
