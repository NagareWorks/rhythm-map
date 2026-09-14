"""Two-resolution local spectral correspondence with explicit abstention."""
from dataclasses import dataclass

import numpy as np

SR = 22050
HOP = 220
RESOLUTIONS = (512, 2048)
LAGS = np.arange(-25, 26, dtype=float) / 100
CONTEXT = np.arange(-50, 51, dtype=float) / 50
DELAY_S = .12
LIMITS = dict(min_spread=.005, min_score=.8, min_margin=.02,
              exclusion_s=.04, agreement_s=.02, aligned_s=.02,
              min_interior_supported_fraction=.8)


@dataclass(frozen=True)
class Features:
    start: float
    step: float
    values: np.ndarray

    def __post_init__(self):
        a = np.array(self.values, dtype=float, copy=True)
        if (a.ndim != 2 or min(a.shape) < 2 or not np.isfinite(a).all()
                or not np.isfinite(self.start) or not np.isfinite(self.step) or self.step <= 0):
            raise ValueError('invalid feature grid')
        a.flags.writeable = False
        object.__setattr__(self, 'values', a)

    @property
    def end(self):
        return self.start + (len(self.values) - 1) * self.step

    def covers(self, times):
        a = np.asarray(times)
        return bool(np.isfinite(a).all() and np.all(a >= self.start) and np.all(a <= self.end))

    def at(self, times):
        t = np.asarray(times, dtype=float)
        if not self.covers(t):
            raise ValueError('no extrapolation outside real feature support')
        u = (t - self.start) / self.step
        i = np.minimum(np.floor(u).astype(int), len(self.values) - 2)
        fraction = (u - i)[..., None]
        return self.values[i] * (1 - fraction) + self.values[i + 1] * fraction


def extract(pcm, n_fft):
    """Unpadded Hann STFT; pooled log amplitude, not a learned representation."""
    pcm = np.asarray(pcm, dtype=float)
    if (pcm.ndim != 1 or n_fft not in RESOLUTIONS or len(pcm) < n_fft + HOP
            or not np.isfinite(pcm).all()):
        raise ValueError('invalid PCM or resolution')
    window = np.hanning(n_fft)
    edges = np.unique(np.rint(np.geomspace(1, n_fft//2 + 1, 33)).astype(int))
    frames = np.lib.stride_tricks.sliding_window_view(pcm, n_fft)[::HOP]
    blocks = []
    for start in range(0, len(frames), 256):
        power = np.abs(np.fft.rfft(frames[start:start+256] * window))**2 / window.sum()**2
        bands = np.stack([power[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
        blocks.append(np.log1p(100 * np.sqrt(bands)))
    return Features(n_fft/2/SR, HOP/SR, np.concatenate(blocks))


def one_resolution(source, output, timeline, center):
    context = center + CONTEXT
    if context[0] < 0 or context[-1] > timeline.output_edges[-1]:
        return dict(status='unsupported')
    source_times = timeline.to_source(context)
    observed_times = context[None, :] + LAGS[:, None]
    if not source.covers(source_times) or not output.covers(observed_times):
        return dict(status='unsupported')
    a = source.at(source_times)
    b = output.at(observed_times)
    # Remove each band's local temporal mean, not temporal variation itself.
    a -= a.mean(axis=0, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    na = np.sqrt(np.sum(a*a))
    nb = np.sqrt(np.sum(b*b, axis=(1, 2)))
    spread = float(np.sqrt(np.mean(a*a)))
    spread_out = np.sqrt(np.mean(b*b, axis=(1, 2)))
    if spread < LIMITS['min_spread'] or np.all(spread_out < LIMITS['min_spread']):
        return dict(status='uninformative', source_spread=spread)
    scores = np.full(len(LAGS), -1.)
    usable = spread_out >= LIMITS['min_spread']
    scores[usable] = np.einsum('ij,kij->k', a, b[usable]) / (na * nb[usable])
    best = int(np.argmax(scores))
    # A second nearly as good shift is uncertainty, not a chosen best alignment.
    outside = np.abs(LAGS - LAGS[best]) > LIMITS['exclusion_s'] + 1e-9
    margin = float(scores[best] - scores[outside].max())
    if best in (0, len(LAGS)-1):
        status = 'search_boundary'
    elif scores[best] < LIMITS['min_score']:
        status = 'mismatch'
    elif margin < LIMITS['min_margin']:
        status = 'ambiguous'
    else:
        status = 'qualified'
    return dict(status=status, lag_s=float(LAGS[best]), score=float(scores[best]),
                margin=margin, source_spread=spread, output_spread=float(spread_out[best]))


def measure(source_views, output_views, timeline, center):
    if set(source_views) != set(RESOLUTIONS) or set(output_views) != set(RESOLUTIONS):
        raise ValueError('both resolutions required')
    if not np.isfinite(center) or not 0 <= center <= timeline.output_edges[-1]:
        raise ValueError('invalid center')
    rows = [dict(n_fft=n, **one_resolution(source_views[n], output_views[n], timeline, center))
            for n in RESOLUTIONS]
    status = next((r['status'] for r in rows if r['status'] != 'qualified'), None)
    lag = None
    if status is None:
        lags = [r['lag_s'] for r in rows]
        if max(lags) - min(lags) > LIMITS['agreement_s'] + 1e-9:
            status = 'resolution_disagreement'
        else:
            lag = float(np.mean(lags))
            status = 'aligned' if max(abs(v) for v in lags) <= LIMITS['aligned_s'] + 1e-9 else 'displaced'
    return dict(center_s=float(center), status=status, lag_s=lag, resolutions=rows)


def region(center, duration, seams):
    if center < 2 or center > duration - 2:
        return 'edge'
    if any(abs(center - seam) <= 1.25 for seam in seams):
        return 'seam'
    return 'interior'


def paired_support(natural, delayed):
    """Same music must report the independently injected +120 ms displacement."""
    if natural['center_s'] != delayed['center_s']:
        raise ValueError('control center differs')
    if natural['status'] != 'aligned' or delayed['status'] != 'displaced':
        return False
    return all(abs((b['lag_s'] - a['lag_s']) - DELAY_S) <= .02 + 1e-9
               for a, b in zip(natural['resolutions'], delayed['resolutions']))


def summarize(rows):
    result = {}
    for kind in ('all', 'interior', 'seam', 'edge'):
        selected = [r for r in rows if kind == 'all' or r['region'] == kind]
        statuses = {s: sum(r['natural']['status'] == s for r in selected)
                    for s in sorted({r['natural']['status'] for r in selected})}
        eligible = sum(r['feature_capture_supported'] for r in selected)
        lags = [abs(r['natural']['lag_s'])*1000 for r in selected if r['natural']['lag_s'] is not None]
        result[kind] = dict(points=len(selected), statuses=statuses, supported_points=eligible,
                           supported_fraction=eligible/len(selected) if selected else None,
                           qualified_points=len(lags),
                           qualified_p95_abs_lag_ms=float(np.percentile(lags, 95)) if lags else None,
                           qualified_max_abs_lag_ms=max(lags) if lags else None)
    return result
