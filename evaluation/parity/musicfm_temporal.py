"""One frozen 25 Hz context/recurrence hypothesis, not a product tempo decoder."""
from fractions import Fraction
import math

import numpy as np

from musicfm_loader import require

RATE, STRIDE, WIDTH, FRAMES = 24000, 960, 1024, 100
CHUNK, HOP = 8 * RATE, 4 * RATE
MAX_SAMPLES = 1800 * RATE
SCALES = {'native': Fraction(1), 'half_zero': Fraction(2),
          'half_one': Fraction(2), 'double': Fraction(1, 2)}
NAMES = tuple(f'{shape}/{density}' for shape in ('constant', 'step') for density in SCALES)
TIE, ZERO = 1e-6, 1e-12
CASES = ('constant-clean', 'constant-omitted', 'constant-weak',
         'step-fast', 'step-slow', 'silence')


def tokens(samples):
    require(type(samples) is int and 1025 <= samples <= MAX_SAMPLES, 'invalid complete input length')
    return (samples // 240 + 3) // 4


def chunks(samples):
    """Grid-aligned 8 s contexts, 4 s hops; fixed midpoint ownership, no padding.

    Final context retains 4..8 s (except a one-context short input). No context
    is aligned to a detected beat or change. Nominal centers are not latency.
    """
    total = tokens(samples)
    starts = [0]
    while starts[-1] + CHUNK < samples:
        starts.append(starts[-1] + HOP)
    seams = [0] + [(s + HOP // 2) // STRIDE for s in starts[1:]] + [total]
    return [dict(id=i, sample_start=s, sample_stop=min(s + CHUNK, samples),
                 token_start=s // STRIDE, token_count=tokens(min(CHUNK, samples - s)),
                 own_start=seams[i], own_stop=seams[i + 1]) for i, s in enumerate(starts)]


def stitch(features, plan, samples):
    """No crossfade, normalization fit, duplicate owner or score-selected seam."""
    require(plan == chunks(samples) and len(features) == len(plan), 'incomplete or altered context plan')
    result = np.empty((tokens(samples), WIDTH), np.float32)
    owner = np.full(len(result), -1, np.int32)
    for x, row in zip(features, plan):
        require(isinstance(x, np.ndarray) and x.dtype == np.float32 and
                x.shape == (row['token_count'], WIDTH) and np.isfinite(x).all(), 'invalid context features')
        left, right = row['own_start'], row['own_stop']
        start = left - row['token_start']
        require(0 <= start < start + right - left <= len(x) and np.all(owner[left:right] == -1),
                'unowned or multiply owned coordinate')
        result[left:right] = x[start:start + right - left]
        owner[left:right] = row['id']
    require(np.all(owner >= 0), 'ownership gap')
    return result, owner


def phase_at(t, anchor, pre, post, boundary):
    return (t - anchor) / pre if t <= boundary else (boundary - anchor) / pre + (t - boundary) / post


def time_at(phase, anchor, pre, post, boundary):
    crossing = (boundary - anchor) / pre
    return anchor + phase * pre if phase <= crossing else boundary + (phase - crossing) * post


def queries(anchor, pre, post, boundary):
    anchor, pre, post, boundary = map(Fraction, (anchor, pre, post, boundary))
    require(pre > 0 and post > 0 and 0 < boundary < FRAMES, 'invalid supplied clock')
    rows = {}
    for name in NAMES:
        shape, density = name.split('/')
        after = pre if shape == 'constant' else post
        rows[name] = [tuple(time_at(phase_at(Fraction(t), anchor, pre, after, boundary) - lag * SCALES[density],
                                         anchor, pre, after, boundary)
                            for lag in (1, Fraction(1, 2))) for t in range(FRAMES)]
    targets = [t for t in range(FRAMES) if all(0 <= q <= FRAMES - 1
               for values in rows.values() for q in values[t])]
    return targets, rows, boundary


def sample(hidden, q):
    left, right = math.floor(q), math.ceil(q)
    weight = float(q - left)
    return hidden[left].astype(np.float64) * (1 - weight) + hidden[right].astype(np.float64) * weight


def unit(x):
    norm = float(np.linalg.norm(x))
    require(math.isfinite(norm) and norm > ZERO, 'zero-norm support')
    return x / norm


def compare(a, b):
    require(math.isfinite(a) and math.isfinite(b), 'nonfinite score')
    return 'left' if a - b > TIE else 'right' if a - b < -TIE else 'unresolved'


def evaluate(hidden, owner, anchor, pre, post, boundary):
    """Same full-cycle minus half-cycle score as prehead v1, on native 25 Hz.

    Labels are not score inputs. Supplied periods/boundary are oracle assistance,
    not label-free automatic detection. Required zero vectors reject all clocks.
    """
    require(isinstance(hidden, np.ndarray) and hidden.dtype == np.float32 and
            hidden.shape == (FRAMES, WIDTH) and np.isfinite(hidden).all(), 'invalid window features')
    require(isinstance(owner, np.ndarray) and owner.dtype == np.int32 and owner.shape == (FRAMES,) and
            np.all(owner >= 0) and np.all(np.diff(owner.astype(np.int64)) >= 0), 'invalid window ownership')
    targets, rows, boundary = queries(anchor, pre, post, boundary)
    base = dict(target_frames=len(targets), excluded_edge_frames=FRAMES - len(targets),
                pre_boundary_frames=sum(t < boundary for t in targets),
                post_boundary_frames=sum(t >= boundary for t in targets))
    if not targets:
        return dict(base, status='no_common_support', clocks=None)
    scores = {}
    try:
        normalized = {t: unit(hidden[t].astype(np.float64)) for t in targets}
        for name, values in rows.items():
            same, half, crossings = [], [], 0
            for t in targets:
                a, b = values[t]
                same.append(float(np.clip(normalized[t] @ unit(sample(hidden, a)), -1, 1)))
                half.append(float(np.clip(normalized[t] @ unit(sample(hidden, b)), -1, 1)))
                crossings += any(owner[i] != owner[t] for q in (a, b)
                                 for i in (math.floor(q), math.ceil(q)))
            scores[name] = dict(score=float(np.mean(np.asarray(same) - half)),
                                same_cycle_cosine=float(np.mean(same)), half_cycle_cosine=float(np.mean(half)),
                                owner_crossing_targets=crossings)
    except ValueError as error:
        if str(error) != 'zero-norm support':
            raise
        return dict(base, status='zero_norm_support', clocks=None)
    return dict(base, status='eligible', clocks=scores)


def verdict(constant, change):
    """Both members required; no orphan win or substitution of favorable cases."""
    if any(row['status'] != 'eligible' for row in (constant, change)):
        return dict(status='pair_rejected', shape_supported=False, density_resolved=False)
    shape, density = True, True
    for row, correct, opposite in ((constant, 'constant/native', 'step/native'),
                                   (change, 'step/native', 'constant/native')):
        require(set(row['clocks']) == set(NAMES), 'incomplete clock ledger')
        scores = {name: value['score'] for name, value in row['clocks'].items()}
        shape &= compare(scores[correct], scores[opposite]) == 'left'
        density &= all(compare(scores[correct], score) == 'left' for name, score in scores.items() if name != correct)
    return dict(status='eligible', shape_supported=bool(shape), density_resolved=bool(density))


def window_from_50hz(start, anchor, pre, post, boundary):
    """Keep the physical four-second interval; never invent 50 Hz features.

    Arguments are absolute legacy 50 Hz coordinates except the two periods.
    Odd starts shift the first native target by 20 ms, explicitly reported.
    """
    require(type(start) is int and start >= 0, 'invalid legacy window start')
    origin = (start + 1) // 2
    a, p, q, b = Fraction(anchor, 2) - origin, Fraction(pre, 2), Fraction(post, 2), Fraction(boundary, 2) - origin
    queries(a, p, q, b)
    return dict(token_start=origin, token_stop=origin + FRAMES, first_target_offset_seconds=str(Fraction(origin, 25) - Fraction(start, 50)),
                anchor=a, pre=p, post=q, boundary=b)


def authored_hidden(post, kind='clean', multiplier=1):
    require(kind in ('clean', 'omitted', 'weak', 'flat', 'zero'), 'unknown hidden recipe')
    phase = np.asarray([float(phase_at(Fraction(t), 0, Fraction(25, 2), Fraction(post), 50))
                        for t in range(FRAMES)]) * multiplier
    out = np.zeros((FRAMES, WIDTH), np.float32)
    if kind == 'zero':
        return out
    if kind == 'flat':
        out[:, 0] = 1
        return out
    out[:, 0], out[:, 1] = np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase)
    pulse = 0.25 * (np.minimum(phase % 1, 1 - phase % 1) < 1 / 16)
    odd = np.floor(phase).astype(int) % 2 == 1
    if kind == 'omitted':
        pulse[odd] = 0
    if kind == 'weak':
        pulse[odd] *= 0.125
    out[:, 2] = pulse
    return out


def authored_pcm(name):
    """Self-authored audio, no pretrained features or files used to choose it.

    120 BPM carrier-modulation cue persists when alternate percussive pulses
    disappear/weaken. This conditional cue is explicit, not a missing-beat oracle.
    """
    require(name in CASES, 'unknown PCM recipe')
    count = 12 * RATE + 240
    t = np.arange(count, dtype=np.float64) / RATE
    post = 0.25 if name == 'step-fast' else 0.8 if name == 'step-slow' else 0.5
    phase = np.where(t <= 6, t / 0.5, 12 + (t - 6) / post)
    age = (phase % 1) * np.where(t <= 6, 0.5, post)
    amp = np.ones(count, np.float64)
    odd = np.floor(phase).astype(np.int64) % 2 == 1
    if name == 'constant-omitted':
        amp[odd] = 0
    if name == 'constant-weak':
        amp[odd] = 0.125
    cue = 0.04 * (1 + np.cos(2 * np.pi * phase)) * np.sin(2 * np.pi * 220 * t)
    pulse = 0.3 * amp * np.exp(-age / 0.015) * (age < 0.08) * np.sin(2 * np.pi * 880 * age)
    return np.zeros(count, np.float32) if name == 'silence' else (cue + pulse).astype(np.float32)
