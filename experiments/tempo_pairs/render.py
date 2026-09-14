"""External offline FFmpeg renderer and authored signal measurements."""
import subprocess
import wave

import numpy as np

from .protocol import BACKENDS, LIMITS, SOURCE_EDGES, SR
from .timeline import Timeline


def render(pcm, rates, backend, executable):
    pcm = np.asarray(pcm)
    if (pcm.shape != (60 * SR,) or not np.isfinite(pcm).all()
            or backend not in BACKENDS or len(rates) != 3
            or any(r not in (.8, 1., 1.25) for r in rates)):
        raise ValueError('unregistered render request')
    parts, sizes = [], []
    for index, rate in enumerate(rates):
        audio_filter = f'atempo={rate}' if backend == 'atempo' else f'rubberband=tempo={rate}:pitch=1'
        command = [str(executable), '-hide_banner', '-loglevel', 'error', '-nostdin', '-threads', '1',
                   '-filter_threads', '1', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', 'pipe:0',
                   '-af', audio_filter, '-f', 'f32le', '-ac', '1', '-ar', str(SR), 'pipe:1']
        result = subprocess.run(command, input=np.asarray(pcm[index*20*SR:(index+1)*20*SR],
                                dtype='<f4').tobytes(), capture_output=True, check=True, timeout=60)
        part = np.frombuffer(result.stdout, dtype='<f4').copy()
        if not len(part) or not np.isfinite(part).all():
            raise ValueError('empty or nonfinite rendered audio')
        parts.append(part)
        sizes.append(len(part))
    # No padding, cropping, crossfade, normalization or post-hoc time rescale.
    return np.concatenate(parts), sizes


def write_wav(path, pcm):
    # Preview uses PCM16; the raw float renderer hash/measurements remain separate.
    quantized = np.rint(np.asarray(pcm, dtype=np.float64) * 32768)
    if not np.isfinite(quantized).all() or np.any(quantized < -32768) or np.any(quantized > 32767):
        raise ValueError('preview would clip')
    with path.open('xb') as stream, wave.open(stream, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(quantized.astype('<i2').tobytes())


def witness(kind):
    if kind not in ('clicks', 'clicks_tone', 'tone'):
        raise ValueError('unknown witness')
    t = np.arange(60 * SR) / SR
    pcm = np.zeros(len(t))
    events = np.arange(.75, 59.5, .5)
    if kind != 'tone':
        # An isolated deterministic broadband burst has a unique envelope peak.
        n = np.arange(-220, 221)
        burst = .65 * np.exp(-.5 * (n / 22.)**2) * np.cos(2*np.pi*.31*n)
        for event in events:
            center = round(event * SR)
            pcm[center-220:center+221] += burst
    if kind != 'clicks':
        pcm += (.15 if kind == 'tone' else .015) * np.sin(2*np.pi*440*t)
    return pcm.astype(np.float32), events


def event_times(pcm):
    """Detect all above-threshold burst groups, not just nearest expected peaks."""
    indices = np.flatnonzero(np.abs(pcm) > .12)
    if not len(indices):
        return np.empty(0)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > round(.04 * SR)) + 1)
    return np.asarray([g[np.argmax(np.abs(pcm[g]))] / SR for g in groups])


def timing_metrics(expected, observed):
    expected, observed = np.asarray(expected), np.asarray(observed)
    if (expected.ndim != 1 or observed.ndim != 1 or not len(expected)
            or not np.isfinite(expected).all() or not np.isfinite(observed).all()
            or np.any(np.diff(expected) <= 0) or np.any(np.diff(observed) <= 0)):
        raise ValueError('invalid events')
    complete = len(expected) == len(observed)
    errors = (observed - expected) * 1000 if complete else None
    p95 = float(np.percentile(np.abs(errors), 95)) if complete else None
    maximum = float(np.max(np.abs(errors))) if complete else None
    return dict(expected_events=len(expected), observed_events=len(observed), complete=complete,
                timing_p95_ms=p95, timing_max_ms=maximum,
                signed_errors_ms=errors.tolist() if complete else None,
                passed=complete and p95 <= LIMITS['timing_p95_ms'] and maximum <= LIMITS['timing_max_ms'])


def duration_metrics(sizes, rates):
    timeline = Timeline(SOURCE_EDGES, rates)
    error = (np.cumsum(sizes) / SR - timeline.output_edges[1:]) * 1000
    return dict(actual_segment_samples=list(sizes), cumulative_error_ms=error.tolist(),
                passed=bool(np.max(np.abs(error)) <= LIMITS['duration_max_ms']))


def pitch_metrics(pcm, sizes):
    # Use two interior seconds in each rendered segment, far from reset seams.
    rows = []
    for offset, length in zip(np.r_[0, np.cumsum(sizes)[:-1]], sizes):
        center = int(offset + length // 2)
        segment = pcm[center-SR:center+SR].astype(float)
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
        k = int(np.argmax(spectrum[1:]) + 1)
        frequency = k * SR / len(segment)
        rows.append(float(1200*np.log2(frequency/440.)))
    return dict(cents_per_segment=rows,
                passed=bool(np.max(np.abs(rows)) <= LIMITS['pitch_max_cents']))
