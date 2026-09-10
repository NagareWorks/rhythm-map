"""Native annotation convention and work-disjoint development split, not a holdout."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FPS = 50
PREFIX_SECONDS = 60
SEED = 142


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def targets(beats, frames, duration):
    """Inter-beat native annotation phase and period. No inferred change labels.

    Only bracketing annotation intervals are supervised. This does not assert
    that native beat level is the only perceptually valid one, or label a rest
    as beatless. Truth never enters feature/model inputs.
    """
    beats = np.asarray(beats, dtype=np.float64)
    require(beats.ndim == 1 and len(beats) >= 2 and np.isfinite(beats).all()
            and np.all(beats >= 0) and np.all(np.diff(beats) > 0), 'invalid beat timeline')
    require(type(frames) is int and frames > 0 and np.isfinite(duration) and duration > 0, 'invalid extent')
    t = np.arange(frames, dtype=np.float64) / FPS
    index = np.searchsorted(beats, t, side='right') - 1
    valid = (index >= 0) & (index < len(beats) - 1) & (t < duration)
    y = np.zeros((frames, 3), dtype=np.float32)
    i = index[valid]
    period = beats[i + 1] - beats[i]
    phase = 2 * np.pi * (t[valid] - beats[i]) / period
    y[valid, 0] = np.log2(period)
    y[valid, 1] = np.cos(phase)
    y[valid, 2] = np.sin(phase)
    return y, valid


def work_key(filename):
    match = re.match(r'(.+?)_(?:AR|OV)(?:-|_)', filename)
    require(match is not None, 'unknown RUBATO work naming')
    return match[1]


def plan():
    """Fixed metadata-only split; old calibration remains development-exposed.

    ARTBeaT never fits or selects checkpoints. Twelve RUBATO works are ordered
    by a fixed salted hash, nine fit and three validate. Performers and encoder
    training overlap are not certified disjoint by work names.
    """
    selection = json.loads((ROOT / 'evaluation/datasets/rubato-calibration-v1-selection.json').read_bytes())
    works = {work_key(r['filename']) for r in selection['tracks']}
    require(len(works) == 12, 'RUBATO work population changed')
    ordered = sorted(works, key=lambda w: hashlib.sha256(('direct-clock-v1/' + w).encode()).hexdigest())
    fit_works = set(ordered[:9])
    rows = []
    for cohort, suite_id in (('rubato', 'rubato-calibration-v1'), ('artbeat', 'artbeat-v1')):
        path = ROOT / 'evaluation/suites' / (suite_id + '.json')
        suite = json.loads(path.read_bytes())
        require(suite['purpose'] == 'calibration', 'only already exposed calibration is allowed')
        for case in suite['cases']:
            audio = case['input']['audio']
            if cohort == 'rubato':
                work = work_key(Path(audio['local_file_hint']).stem)
                require(work in works, 'unknown work')
                role = 'fit' if work in fit_works else 'development'
            else:
                work, role = 'artbeat-shared-source', 'diagnostic'
            truth = (path.parent / case['input']['truth']).resolve()
            require(truth.is_relative_to((path.parent / 'truth' / suite_id).resolve()), 'truth outside selected calibration')
            rows.append(dict(id=case['id'], cohort=cohort, suite=suite_id, suite_sha256=sha(path),
                work=work, role=role, audio_hint=audio['local_file_hint'], audio_sha256=audio['sha256'],
                truth=str(truth.relative_to(ROOT)).replace('\\', '/'), truth_sha256=sha(truth)))
    require(len(rows) == 40 and sum(r['cohort'] == 'rubato' for r in rows) == 25, 'case population changed')
    return dict(schema='rhythm-map.direct-clock-development.v1', seed=SEED, prefix_seconds=PREFIX_SECONDS,
        fit_works=ordered[:9], development_works=ordered[9:], cases=rows,
        independent_acceptance=False, prior_calibration_reused_for_exploratory_fitting=True,
        native_annotation_convention_not_unique_perceived_level=True, holdout_access=False,
        performer_disjointness='not_certified', encoder_recording_overlap='unknown')


def windows(length, size=200):
    require(type(length) is int and length > 0, 'invalid feature length')
    return [(a, min(a + size, length)) for a in range(0, length, size)]


def metrics(prediction, target, valid):
    """No silent missing-prediction drop; headline errors require full coverage."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    valid = np.asarray(valid)
    require(prediction.shape == target.shape == (len(valid), 3) and valid.dtype == bool, 'metric shape mismatch')
    n = int(valid.sum())
    finite = np.isfinite(prediction).all(axis=1)
    # Very large finite log-period can still overflow tempo conversion.
    with np.errstate(over='ignore', invalid='ignore'):
        relative = np.abs(np.exp2(target[:, 0] - prediction[:, 0]) - 1) * 100
    period_ok = valid & finite & np.isfinite(relative)
    amplitude = np.hypot(prediction[:, 1], prediction[:, 2])
    phase_ok = valid & finite & (amplitude > 1e-6)
    phase = np.arctan2(prediction[:, 2], prediction[:, 1])
    truth_phase = np.arctan2(target[:, 2], target[:, 1])
    phase_error = np.abs(np.angle(np.exp(1j * (phase - truth_phase)))) / (2 * np.pi)
    return dict(reference_frames=n, period_available_frames=int(period_ok.sum()), phase_available_frames=int(phase_ok.sum()),
        tempo_median_error_percent=float(np.median(relative[valid])) if n and np.all(period_ok[valid]) else None,
        tempo_p95_error_percent=float(np.percentile(relative[valid], 95)) if n and np.all(period_ok[valid]) else None,
        phase_mean_absolute_cycles=float(phase_error[valid].mean()) if n and np.all(phase_ok[valid]) else None)
