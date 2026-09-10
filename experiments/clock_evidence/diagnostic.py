"""Input interventions and measurements, not new training or a product policy."""
import numpy as np

from experiments.clock_readout.model import HALO
from experiments.coupled_clock.clock import require
from experiments.coupled_clock.measurement import KEYS

VARIANTS = ('natural', 'zero', 'time_mean', 'half_roll')


def intervention(features, variant):
    x = np.asarray(features)
    require(x.ndim == 2 and len(x) >= 2 and x.dtype == np.float32
            and np.isfinite(x).all(), 'invalid frozen features')
    require(variant in VARIANTS, 'unregistered intervention')
    if variant == 'natural':
        return x.copy()
    if variant == 'zero':
        return np.zeros_like(x)
    if variant == 'time_mean':
        return np.broadcast_to(x.mean(axis=0, dtype=np.float64).astype(np.float32), x.shape).copy()
    return np.roll(x, len(x) // 2, axis=0)


def interior(frames, halo=HALO):
    """Same support for all field comparisons, free of source/destination edges.

    A half-roll introduces a seam. Both the output frame and the original frame
    supplying it must be at least HALO from physical edges. This excludes the
    seam's convolution neighborhood too. Musical metrics do NOT use this trim;
    globally integrated clocks can carry an earlier disturbance past the seam.
    """
    require(type(frames) is int and frames >= 2 and type(halo) is int and halo >= 0,
            'invalid interior geometry')
    index = np.arange(frames)
    source = (index - frames // 2) % frames
    return ((index >= halo) & (index < frames - halo)
            & (source >= halo) & (source < frames - halo))


def field_response(natural, modified, valid, dense_phase):
    natural, modified, valid = np.asarray(natural), np.asarray(modified), np.asarray(valid)
    require(natural.shape == modified.shape and natural.ndim == 2
            and natural.shape[1] == (3 if dense_phase else 2)
            and valid.dtype == bool and valid.shape == (len(natural),)
            and np.isfinite(natural).all() and np.isfinite(modified).all(), 'field geometry mismatch')
    result = dict(points=int(valid.sum()), log2_period_rms_change=None,
                  dense_phase_mean_absolute_change_cycles=None)
    if valid.any():
        result['log2_period_rms_change'] = float(np.sqrt(np.mean(
            (natural[valid, 0].astype(np.float64) - modified[valid, 0]) ** 2)))
        if dense_phase:
            a, b = natural[valid, 1:], modified[valid, 1:]
            if (np.linalg.norm(a, axis=1) > 1e-6).all() and (np.linalg.norm(b, axis=1) > 1e-6).all():
                angle = np.arctan2(a[:, 1], a[:, 0]) - np.arctan2(b[:, 1], b[:, 0])
                result['dense_phase_mean_absolute_change_cycles'] = float(np.abs(
                    (angle / (2 * np.pi) + .5) % 1 - .5).mean())
    return result


def paired_summary(rows, variant):
    """Within-checkpoint intervention deltas; positive means natural is better.

    A positive response is descriptive, not a significance, independence or
    promotion gate. Temporal mean/roll/zero can all be out of distribution.
    """
    result = {}
    for key in KEYS:
        groups, signs = {}, []
        for row in rows:
            a, b = row['metrics']['natural'][key], row['metrics'][variant][key]
            value = None if a is None or b is None else b - a
            groups.setdefault(row['work'], []).append(value)
            signs.append(None if value is None else bool(value > 0))
        values = [v for group in groups.values() for v in group]
        result[key] = dict(
            intervention_minus_natural_work_macro=(float(np.mean([np.mean(v) for v in groups.values()]))
                if values and all(v is not None and np.isfinite(v) for v in values) else None),
            natural_better_recordings=sum(v is True for v in signs),
            missing_recordings=sum(v is None for v in signs), recordings=len(rows))
    return result
