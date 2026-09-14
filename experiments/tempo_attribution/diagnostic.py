"""Fixed offset/shape and native-speed-change accounting; not a decoder."""
import numpy as np

from experiments.coupled_clock.clock import FPS, require
from experiments.coupled_clock.measurement import intervals
from experiments.phase_attribution.diagnostic import rate_diagnosis, OCTAVE_WIDTH

LAGS = (50, 200)
BANDS = (0., 60., 90., 120., 180., float('inf'))
SLOPE_EPS = 1e-10


def average(values):
    return float(np.mean(values)) if len(values) else None


def response(dx, dy):
    require(dx.shape == dy.shape and dx.ndim == 1
            and np.isfinite(dx).all() and np.isfinite(dy).all(), 'invalid change vectors')
    energy = average(dx * dx)
    defined = energy is not None and energy > SLOPE_EPS ** 2
    large = np.abs(dx) >= OCTAVE_WIDTH
    return dict(pairs=len(dx), reference_rms_log2=None if energy is None else float(np.sqrt(energy)),
                error_rmse_log2=None if not len(dx) else float(np.sqrt(np.mean((dy - dx) ** 2))),
                response_slope=None if not defined else average(dx * dy) / energy,
                large_change_pairs=int(large.sum()),
                large_change_direction_agreement=average(np.sign(dx[large]) == np.sign(dy[large])))


def coverage(reference_rate, cells):
    require(reference_rate.shape == cells.shape and cells.dtype == bool, 'invalid reference support')
    bpm = reference_rate[cells] * 60
    require(np.isfinite(bpm).all() and (bpm > 0).all(), 'invalid reference speed')
    out = dict(cells=int(cells.sum()), seconds=float(cells.sum() / FPS))
    for p in (0, 5, 50, 95, 100):
        out['bpm_p' + str(p)] = float(np.percentile(bpm, p)) if len(bpm) else None
    for low, high in zip(BANDS[:-1], BANDS[1:]):
        out['fraction_' + str(int(low)) + '_' + (str(int(high)) if np.isfinite(high) else 'inf')] = (
            average((bpm >= low) & (bpm < high)))
    return out


def analyze(rate, reference, valid):
    rate, reference, valid = np.asarray(rate, np.float64), np.asarray(reference, np.float64), np.asarray(valid)
    require(reference.ndim == 1 and len(reference) >= 2 and valid.dtype == bool
            and valid.shape == reference.shape and rate.shape == (len(reference) - 1,), 'invalid rate geometry')
    cells = intervals(valid, 1)
    expected = np.diff(reference) * FPS
    require(np.isfinite(reference[valid]).all() and np.isfinite(expected[cells]).all()
            and (expected[cells] > 0).all() and np.isfinite(rate[cells]).all()
            and (rate[cells] > 0).all(), 'invalid supported rate')
    x, y = np.zeros_like(rate), np.zeros_like(rate)
    x[cells], y[cells] = np.log2(expected[cells]), np.log2(rate[cells])
    e = y[cells] - x[cells]
    bias, mse = average(e), average(e * e)
    residual = None if bias is None else average((e - bias) ** 2)
    centered_x, centered_y = x[cells] - (average(x[cells]) or 0.), y[cells] - (average(y[cells]) or 0.)
    shape = response(centered_x, centered_y)
    if bias is not None:
        require(np.isclose(mse, bias * bias + residual, rtol=1e-12, atol=1e-12), 'decomposition differs')
    detail = dict(cells=int(cells.sum()), log2_mse=mse, signed_bias_log2=bias,
                  squared_bias=None if bias is None else bias * bias, residual_log2_mse=residual,
                  reference_log2_std=shape['reference_rms_log2'],
                  predicted_log2_std=None if not len(e) else float(np.sqrt(np.mean(centered_y ** 2))),
                  centered_response_slope=shape['response_slope'])
    changes = {}
    for lag in LAGS:
        mask = intervals(cells, lag)
        dx, dy = (x[lag:] - x[:-lag])[mask], (y[lag:] - y[:-lag])[mask]
        large = np.abs(dx) >= OCTAVE_WIDTH
        changes[str(lag)] = {name: response(dx[subset], dy[subset]) for name, subset in
                            (('all', np.ones(len(dx), bool)), ('large', large), ('small', ~large))}
    return dict(offset_shape=detail, octave=rate_diagnosis(rate, expected, cells),
                reference=coverage(expected, cells), changes=changes)


def flatten(item, prefix=''):
    result = {}
    for key, value in item.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(flatten(value, name + '.'))
        else:
            result[name] = value
    return result


def macro(rows, support, kind):
    groups = {}
    for row in rows:
        groups.setdefault(row['work'], []).append(flatten(row[support][kind]))
    require(bool(groups), 'empty cohort')
    keys = next(iter(groups.values()))[0].keys()
    require(all(r.keys() == keys for group in groups.values() for r in group), 'incomplete diagnostic fields')
    result = {}
    for key in keys:
        values = [r[key] for group in groups.values() for r in group]
        missing = sum(v is None or not np.isfinite(v) for v in values)
        result[key] = dict(work_macro=None if missing else float(np.mean(
            [np.mean([r[key] for r in group]) for group in groups.values()])),
            recordings=len(values), works=len(groups), missing_recordings=missing)
    return result


def summarize(rows):
    return {role: {support: {kind: macro([r for r in rows if r['role'] == role], support, kind)
        for kind in rows[0][support]} for support in ('native', 'common')}
        for role in ('fit', 'development', 'diagnostic')}
