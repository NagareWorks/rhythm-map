"""Descriptive algebra on saved clocks/fields; no model or optimizer calls."""
import math

import numpy as np

from experiments.coupled_clock.clock import FPS, require
from experiments.coupled_clock.measurement import clock_fields, intervals, measure

KINDS = ('audio', 'zero', 'same_zero', 'time_mean', 'half_roll', 'raw')
LAGS = (1, 25, 100, 200)
BINS = ('native', 'half', 'double', 'other_octave', 'off_grid')
OCTAVE_WIDTH = math.log2(1.1)


def components(mask):
    """Half-open runs; gaps are never joined."""
    edge = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def mean(value):
    return float(np.mean(value)) if len(value) else None


def median(value):
    return float(np.median(value)) if len(value) else None


def rate_diagnosis(rate, reference_rate, cells):
    require(rate.shape == reference_rate.shape == cells.shape, 'rate geometry mismatch')
    require(np.isfinite(rate[cells]).all() and (rate[cells] > 0).all()
            and np.isfinite(reference_rate[cells]).all() and (reference_rate[cells] > 0).all(),
            'invalid supported rate')
    z = np.log2(rate[cells]) - np.log2(reference_rate[cells])
    k = np.floor(z + .5)
    residual = z - k
    level = float(np.floor(np.median(z) + .5)) if len(z) else None
    out = dict(log2_ratio_median=median(z), log2_error_median=median(abs(z)),
               folded_log2_error_median=median(abs(residual)), oracle_recording_level=level,
               oracle_recording_tempo_error_percent=(median(100 * abs(np.exp2(z - level) - 1))
                                                     if level is not None else None),
               oracle_cell_folded_tempo_error_percent=median(100 * abs(np.exp2(residual) - 1)))
    near = abs(residual) <= OCTAVE_WIDTH
    categories = (near & (k == 0), near & (k == -1), near & (k == 1),
                  near & (abs(k) > 1), ~near)
    for name, category in zip(BINS, categories):
        full = np.zeros_like(cells)
        full[cells] = category
        out[name + '_fraction'] = mean(category)
        out[name + '_longest_s'] = (max((b - a for a, b in components(full)), default=0) / FPS
                                   if cells.any() else None)
    return out


def count_diagnosis(q, reference, valid):
    rows, drifts, integers = [], [], []
    for a, b in components(valid):
        # One point has phase support, but no count interval.
        if b - a < 2:
            continue
        d = (q[a:b] - q[a]) - (reference[a:b] - reference[a])
        integer = np.floor(d + .5)
        rows.append(dict(start_frame=int(a), end_frame=int(b - 1),
                         reference_advance=float(reference[b - 1] - reference[a]),
                         predicted_advance=float(q[b - 1] - q[a]),
                         terminal_drift_cycles=float(d[-1]),
                         integer_min=float(integer.min()), integer_max=float(integer.max()),
                         integer_boundary_transitions=int(np.count_nonzero(np.diff(integer)))))
        drifts.extend(abs(d))
        integers.extend(abs(integer))
    return dict(relative_drift_mae_cycles=mean(drifts), relative_integer_offset_mae=mean(integers),
                terminal_abs_drift_mean_cycles=mean([abs(r['terminal_drift_cycles']) for r in rows]),
                integer_range_max=(max(r['integer_max'] - r['integer_min'] for r in rows) if rows else None)), rows


def loss_diagnosis(q, reference, valid):
    error = q[valid] - reference[valid]
    out = dict(circular_loss=mean(1 - np.cos(2 * np.pi * (error % 1.))))
    terms = []
    for lag in LAGS:
        pairs = intervals(valid, lag)
        a, b = (q[lag:] - q[:-lag])[pairs], (reference[lag:] - reference[:-lag])[pairs]
        value = mean((np.log2(a) - np.log2(b)) ** 2)
        out[f'count_log2_mse_lag_{lag}'] = value
        if value is not None:
            terms.append(value)
    out['count_loss'] = mean(terms)
    out['total_loss'] = (out['circular_loss'] + out['count_loss']
                         if out['circular_loss'] is not None and out['count_loss'] is not None else None)
    return out


def feedback_diagnosis(q, fields, reference, valid, gain):
    require(fields.shape == (len(q), 4) and np.isfinite(fields).all()
            and math.isfinite(gain) and 0 <= gain <= math.log(2), 'invalid local fields/gain')
    require(np.isfinite(q).all() and (np.diff(q) > 0).all(), 'invalid saved complete clock')
    prior = np.exp2(-fields[:-1, 0].astype(np.float64))
    actual = np.diff(q) * FPS
    require(np.isfinite(prior).all() and (prior > 0).all(), 'invalid local prior')
    u, v = fields[1:, 1].astype(np.float64), fields[1:, 2].astype(np.float64)
    amplitude = np.hypot(u, v)
    norm = np.hypot(1., amplitude)
    angle = 2 * np.pi * ((q[:-1] + prior / FPS) % 1.)
    local_log2 = gain * (v / norm * np.cos(angle) - u / norm * np.sin(angle)) / math.log(2)
    correction = np.log2(actual) - np.log2(prior)
    require(np.allclose(correction, local_log2, atol=2e-9, rtol=1e-8), 'saved feedback equation differs')
    require(math.isclose(q[0], float(fields[0, 3]), abs_tol=1e-12, rel_tol=1e-12),
            'saved origin differs')
    cells = intervals(valid, 1)
    expected = np.diff(reference) * FPS
    required = np.log2(expected[cells]) - np.log2(prior[cells])
    bound = gain * (amplitude / norm) / math.log(2)
    obs_valid = valid[1:]
    observed = obs_valid & (amplitude > 0)
    # A zero vector has no angle: retain coverage and null complete metric.
    phase = np.arctan2(v, u) / (2 * np.pi)
    phase_error = abs((phase[observed] - reference[1:][observed] + .5) % 1 - .5)
    return dict(prior_tempo_median_error_percent=median(100 * abs(np.exp2(-required) - 1)),
                prior_log2_error_median=median(abs(required)),
                correction_log2_mean=mean(correction[cells]),
                correction_abs_log2_mean=mean(abs(correction[cells])),
                correction_abs_log2_p95=(float(np.percentile(abs(correction[cells]), 95)) if cells.any() else None),
                reference_outside_feedback_bound_fraction=mean(abs(required) > bound[cells]),
                feedback_improves_cell_fraction=mean(abs(np.log2(actual[cells]) - np.log2(expected[cells])) < abs(required)),
                observation_available_fraction=(float(observed.sum() / obs_valid.sum()) if obs_valid.any() else None),
                observation_phase_mae_cycles=(mean(phase_error) if observed.sum() == obs_valid.sum() else None),
                feedback_equation_max_abs_log2_residual=float(np.max(abs(correction - local_log2))))


def analyze(q, reference, valid, fields=None, gain=None):
    q, reference = np.asarray(q, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    valid = np.asarray(valid)
    require(q.ndim == 1 and len(q) >= 2 and q.shape == reference.shape == valid.shape
            and valid.dtype == bool, 'clock/support geometry mismatch')
    require(np.isfinite(q[valid]).all() and np.isfinite(reference[valid]).all(), 'invalid supported clock')
    rate, expected, cells = np.diff(q) * FPS, np.diff(reference) * FPS, intervals(valid, 1)
    original = measure(clock_fields(q), reference, valid)
    detail = rate_diagnosis(rate, expected, cells)
    count, spans = count_diagnosis(q, reference, valid)
    detail.update(count)
    detail.update(loss_diagnosis(q, reference, valid))
    if fields is not None:
        detail.update(feedback_diagnosis(q, fields, reference, valid, gain))
    return dict(original=original, detail=detail, components=spans)


def macro(rows, support, kind, section):
    groups = {}
    for row in rows:
        groups.setdefault(row['work'], []).append(row[support][kind][section])
    keys = next(iter(groups.values()))[0].keys() if groups else ()
    return {key: (float(np.mean([np.mean([r[key] for r in group]) for group in groups.values()]))
                  if all(r[key] is not None and np.isfinite(r[key]) for group in groups.values() for r in group)
                  else None) for key in keys}
