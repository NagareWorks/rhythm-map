"""Evaluation-only normalized pre-head capture; never a production decoder.

Chunk geometry follows the pinned MIT-licensed Beat This implementation (see
licenses/beat-this-rs-MIT.txt). NumPy controls need no model or PyTorch import.
"""
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

CHUNK, BORDER, STRIDE, WIDTH = 1500, 6, 1488, 512
MAX_FRAMES = 4096  # This gate covers short controls/reference prefixes only.
KEYS = ('hidden', 'beat', 'downbeat')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_hash(value):
    value = np.asarray(value)
    # Stable little-endian, C-order bytes; shape/dtype are recorded separately.
    return hashlib.sha256(value.astype(value.dtype.newbyteorder('<'), copy=False).tobytes()).hexdigest()


def finite_f32(value, shape):
    require(isinstance(value, np.ndarray) and value.dtype == np.float32 and
            value.shape == tuple(shape) and np.isfinite(value).all(),
            'expected finite float32 tensor with exact shape')


def layout(frames):
    require(type(frames) is int and 1 <= frames <= MAX_FRAMES, 'invalid frame budget')
    starts = list(range(-BORDER, frames - BORDER, STRIDE))
    if frames > STRIDE:
        starts[-1] = frames - (CHUNK - BORDER)
    rows = []
    for start in starts:
        left, right = max(0, -start), max(0, min(BORDER, start + CHUNK - frames))
        source_start, source_end = max(0, start), min(start + CHUNK, frames)
        size = left + source_end - source_start + right
        rows.append(dict(start=start, left=left, right=right, size=size,
                         source_start=source_start, source_end=source_end,
                         write_start=start + BORDER, write_end=start + size - BORDER))
    return rows


def split(mel):
    require(isinstance(mel, np.ndarray) and mel.ndim == 2, 'expected unbatched mel')
    rows = layout(len(mel))
    finite_f32(mel, (len(mel), 128))
    return [np.pad(mel[r['source_start']:r['source_end']],
                   ((r['left'], r['right']), (0, 0))) for r in rows]


def stitch(frames, chunks):
    rows = layout(frames)
    require(len(chunks) == len(rows), 'missing or extra chunk')
    for row, chunk in zip(rows, chunks):
        require(set(chunk) == set(KEYS), 'unexpected chunk fields')
        for key in KEYS:
            finite_f32(chunk[key], (row['size'], WIDTH) if key == 'hidden' else (row['size'],))
    result = {key: np.empty((frames, WIDTH) if key == 'hidden' else frames, np.float32)
              for key in KEYS}
    owner = np.full(frames, -1, dtype=np.int32)
    # Earlier chunks overwrite later chunks, for every channel together.
    for index in reversed(range(len(rows))):
        row = rows[index]
        target = slice(row['write_start'], row['write_end'])
        require(0 <= row['write_start'] < row['write_end'] <= frames, 'invalid ownership range')
        for key in KEYS:
            result[key][target] = chunks[index][key][BORDER:-BORDER]
        owner[target] = index
    require(np.all(owner >= 0), 'uncovered frame')
    return dict(result, owner=owner)


def difference(left, right, *, atol=0.0, rtol=0.0, exact=False):
    require(np.isfinite(atol) and np.isfinite(rtol) and atol >= 0 and rtol >= 0,
            'invalid numerical budget')
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape and left.dtype == right.dtype and
            left.size > 0 and np.isfinite(left).all() and np.isfinite(right).all(),
            'incompatible/nonfinite comparison')
    error = np.abs(left.astype(np.float64) - right.astype(np.float64))
    same_bits = left.tobytes() == right.tobytes()
    passed = same_bits if exact else bool(np.all(error <= atol + rtol * np.abs(right)))
    require(passed, 'parity budget exceeded')
    return dict(passed=True, bit_exact=same_bits, max_abs=float(error.max()))


def event_identity(left, right, atol=1e-5):
    """Absolute time AND identical ordered half-frame identities, never rematching."""
    left, right = np.asarray(left), np.asarray(right)
    require(np.isfinite(atol) and atol >= 0 and left.ndim == right.ndim == 1 and
            left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all() and
            np.all(left >= 0) and np.all(right >= 0) and
            np.all(np.diff(left) > 0) and np.all(np.diff(right) > 0), 'invalid event vectors')
    require(np.all(np.abs(left - right) <= atol) and
            np.array_equal(np.rint(left * 100), np.rint(right * 100)), 'event identity changed')
    return dict(passed=True, count=len(left), max_abs=float(np.max(np.abs(left - right))) if len(left) else 0.0)


def capture_chunk(model, mel, reconstruction_atol, reconstruction_rtol):
    """Run untouched and hooked copies; remove the hook even on model failure.

    The cloned task-head input is the normalized transformer output. The hook
    returns None and cannot substitute the actual forward input or output.
    """
    import torch
    finite_f32(mel, (len(mel), 128))
    require(not any(m.training for m in model.modules()), 'model must be in evaluation mode')
    require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters()),
            'capture requires float32 CPU parameters')
    require(not model.task_heads._forward_pre_hooks, 'task head already has pre-hooks')
    x = torch.from_numpy(mel.copy()).unsqueeze(0)
    before = x.clone()
    captured = []

    def tap(_module, args):
        require(len(args) == 1, 'unexpected task-head inputs')
        captured.append(args[0].detach().clone())

    with torch.inference_mode():
        begin = time.perf_counter()
        baseline = model(x)
        baseline_s = time.perf_counter() - begin
        require(torch.equal(before, x), 'untapped pass mutated input')
        handle = model.task_heads.register_forward_pre_hook(tap)
        begin = time.perf_counter()
        try:
            observed = model(x)
        finally:
            handle.remove()
        tapped_s = time.perf_counter() - begin
        require(torch.equal(before, x) and len(captured) == 1, 'input mutation or nonunique tap')
        hidden = captured[0]
        require(hidden.device.type == 'cpu' and hidden.dtype == torch.float32 and
                tuple(hidden.shape) == (1, len(mel), WIDTH), 'wrong feature tap')
        recreated = model.task_heads(hidden)
        result = dict(hidden=hidden[0].numpy().copy())
        checks, base = {}, {}
        for key in ('beat', 'downbeat'):
            base[key] = baseline[key][0].numpy().copy()
            result[key] = observed[key][0].numpy().copy()
            checks[key + '_tap_vs_untapped'] = difference(result[key], base[key], exact=True)
            checks[key + '_reconstructed'] = difference(
                recreated[key][0].numpy(), result[key], atol=reconstruction_atol, rtol=reconstruction_rtol)
        for key in KEYS:
            finite_f32(result[key], (len(mel), WIDTH) if key == 'hidden' else (len(mel),))
    return result, base, dict(checks=checks, untapped_s=baseline_s, tapped_s=tapped_s)


def save_capture(path, arrays):
    """Exclusive private artifact creation, with pickle-free round-trip checks."""
    path = Path(path)
    require(set(arrays) == {*KEYS, 'owner'}, 'unexpected archive fields')
    frames = len(arrays['beat'])
    layout(frames)
    for key in KEYS:
        finite_f32(arrays[key], (frames, WIDTH) if key == 'hidden' else (frames,))
    expected_owner = np.full(frames, -1, np.int32)
    for index, row in reversed(list(enumerate(layout(frames)))):
        expected_owner[row['write_start']:row['write_end']] = index
    require(arrays['owner'].dtype == np.int32 and arrays['owner'].shape == (frames,) and
            np.array_equal(arrays['owner'], expected_owner), 'invalid archive ownership')
    begin = time.perf_counter()
    with path.open('xb') as target:
        np.savez(target, **arrays)
    elapsed = time.perf_counter() - begin
    with np.load(path, allow_pickle=False) as replay:
        require(set(replay.files) == set(arrays), 'archive fields changed')
        for key, value in arrays.items():
            difference(replay[key], value, exact=True)
    return dict(size_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                serialization_s=elapsed, roundtrip_bit_exact=True)


def peak_rss_bytes():
    """Process-lifetime high-water RSS, not per-model or per-case allocation."""
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('peak_rss', 'rss', 'peak_pool', 'pool',
                                                   'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
        kernel, psapi = ctypes.WinDLL('kernel32', use_last_error=True), ctypes.WinDLL('psapi', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        value = Counters()
        value.cb = ctypes.sizeof(value)
        require(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb),
                'cannot measure process peak memory')
        return value.peak_rss
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak * 1024)
