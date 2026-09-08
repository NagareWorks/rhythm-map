"""Fixed, label-free recurrence score for supplied clocks, not a tempo decoder.

No learned projection, score fitting, phase search or automatic boundary search.
Candidate clocks are oracle-assisted in the paired calibration protocol.
"""
from fractions import Fraction
import math

import numpy as np

import oracle_period_paired_audit as oracle
from prehead_capture import finite_f32, require

NAMES = oracle.NAMES
FRAMES, WIDTH = 200, 512
MARGIN_EPS = 1e-6  # Numerical tie budget, not a confidence threshold.
NORM_EPS = 1e-12


def query_plan(anchor, pre, post, boundary):
    """Exact previous-cycle and previous-half-cycle coordinates for all clocks.

    Half-phase offsets cancel from phase differences. Absolute anchors cancel
    too, including across the continuous step; the four supplied phases remain
    registered conditions, not four independent votes.
    """
    anchor, pre, post, boundary = map(Fraction, (anchor, pre, post, boundary))
    require(pre > 0 and post > 0 and 0 < boundary < FRAMES, 'invalid supplied clock')
    queries = {}
    for name in NAMES:
        shape, density = name.split('/')
        period_after = pre if shape == 'constant' else post
        scale, _ = oracle.SCALES[density]
        rows = []
        for frame in range(FRAMES):
            phase = oracle.phase_at(Fraction(frame), anchor, pre, period_after, boundary)
            rows.append(tuple(oracle.time_at(phase - lag * scale, anchor, pre,
                                             period_after, boundary)
                              for lag in (Fraction(1), Fraction(1, 2))))
        queries[name] = rows
    # Every clock uses exactly the same target frames and interpolation support.
    # No extrapolation, wrapping at a file edge, or using a neighboring window.
    selected = [t for t in range(FRAMES) if all(
        0 <= q <= FRAMES - 1 for rows in queries.values() for q in rows[t])]
    return dict(targets=selected, queries=queries, boundary=boundary)


def sample(values, coordinate):
    """Interpolate raw features in float64, then normalize; never round a lag."""
    left, right = math.floor(coordinate), math.ceil(coordinate)
    weight = float(coordinate - left)
    return (values[left].astype(np.float64) * (1 - weight) +
            values[right].astype(np.float64) * weight)


def unit(value):
    norm = float(np.linalg.norm(value))
    require(math.isfinite(norm) and norm > NORM_EPS, 'unavailable zero-norm feature')
    return value / norm


def compare(left, right):
    require(math.isfinite(left) and math.isfinite(right), 'nonfinite score')
    delta = left - right
    return 'left' if delta > MARGIN_EPS else 'right' if delta < -MARGIN_EPS else 'unresolved'


def evaluate(hidden, owner, anchor, pre, post, boundary):
    """Score all eight clocks on common support; label and role are not inputs.

    Owner IDs describe the original complete-recording capture. Cross-owner
    comparisons are retained and counted, not repaired by local re-inference.
    A zero vector at any required sample invalidates the entire window.
    """
    finite_f32(hidden, (FRAMES, WIDTH))
    require(isinstance(owner, np.ndarray) and owner.dtype == np.int32 and
            owner.shape == (FRAMES,) and np.all(owner >= 0) and
            np.all(np.diff(owner.astype(np.int64)) >= 0), 'invalid feature ownership')
    plan = query_plan(anchor, pre, post, boundary)
    targets = plan['targets']
    base = dict(target_frames=len(targets), excluded_edge_frames=FRAMES - len(targets),
                pre_boundary_frames=sum(t < plan['boundary'] for t in targets),
                post_boundary_frames=sum(t >= plan['boundary'] for t in targets))
    if not targets:
        return dict(base, status='no_common_support', clocks=None)
    clocks = {}
    try:
        normalized = {t: unit(hidden[t].astype(np.float64)) for t in targets}
        for name, rows in plan['queries'].items():
            differences, same, half = [], [], []
            crossings = 0
            for t in targets:
                a, b = rows[t]
                x, y = unit(sample(hidden, a)), unit(sample(hidden, b))
                same.append(float(np.clip(np.dot(normalized[t], x), -1., 1.)))
                half.append(float(np.clip(np.dot(normalized[t], y), -1., 1.)))
                differences.append(same[-1] - half[-1])
                crossings += any(owner[i] != owner[t] for q in (a, b)
                                 for i in (math.floor(q), math.ceil(q)))
            clocks[name] = dict(score=float(np.mean(differences)),
                                same_cycle_cosine=float(np.mean(same)),
                                half_cycle_cosine=float(np.mean(half)),
                                owner_crossing_targets=crossings)
    except ValueError as error:
        if str(error) != 'unavailable zero-norm feature':
            raise
        return dict(base, status='zero_norm_support', clocks=None)
    return dict(base, status='eligible', clocks=clocks)


def verdict(windows):
    """Apply truth only after scoring; preserve all four phases and both roles."""
    require(set(windows) == {'constant', 'change'} and
            all(len(rows) == 4 for rows in windows.values()), 'incomplete paired phases')
    if any(row['status'] != 'eligible' for rows in windows.values() for row in rows):
        return dict(status='pair_rejected', shape_phases=0, density_phases=0,
                    shape_supported=False, density_resolved=False)
    shapes, densities = [], []
    for phase in range(4):
        shape_ok, density_ok = True, True
        for role, rows in windows.items():
            correct = ('constant' if role == 'constant' else 'step') + '/native'
            rival = ('step' if role == 'constant' else 'constant') + '/native'
            scores = rows[phase]['clocks']
            require(set(scores) == set(NAMES), 'incomplete hypothesis population')
            shape_ok &= compare(scores[correct]['score'], scores[rival]['score']) == 'left'
            density_ok &= all(compare(scores[correct]['score'], scores[name]['score']) == 'left'
                              for name in NAMES if name != correct)
        shapes.append(bool(shape_ok))
        densities.append(bool(density_ok))
    return dict(status='eligible', shape_phases=sum(shapes), density_phases=sum(densities),
                shape_supported=all(shapes), density_resolved=all(densities))


def authored_hidden(pre, post, boundary=100, *, kind='clean', multiplier=1):
    """Artificial feature trajectories, NOT audio or outputs of a real encoder.

    The first two channels explicitly carry a clock. A small third-channel
    pulse can disappear or weaken without erasing that authored information.
    This checks the comparison's behavior if such a cue exists, not that Beat
    This preserves it. The remaining 509 channels are zero by construction.
    """
    require(kind in ('clean', 'omitted', 'weak', 'flat', 'zero'), 'unknown authored fixture')
    phase = np.asarray([float(oracle.phase_at(Fraction(t), Fraction(0), Fraction(pre),
                                             Fraction(post), Fraction(boundary)))
                        for t in range(FRAMES)]) * multiplier
    out = np.zeros((FRAMES, WIDTH), np.float32)
    if kind == 'zero':
        return out
    if kind == 'flat':
        out[:, 0] = 1
        return out
    out[:, 0], out[:, 1] = np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase)
    pulse = 0.25 * (np.minimum(phase % 1, 1 - phase % 1) < 1 / 16)
    if kind == 'omitted':
        pulse[(np.floor(phase).astype(int) % 2) == 1] = 0
    elif kind == 'weak':
        pulse[(np.floor(phase).astype(int) % 2) == 1] *= 0.125
    out[:, 2] = pulse
    return out
