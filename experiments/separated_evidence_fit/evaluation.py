"""Unchanged metric denominators applied to explicitly independent evidence."""
import math

import numpy as np
import torch

from experiments.coupled_clock.clock import require
from experiments.coupled_clock.prepare import sha
from experiments.coupled_clock.run import state_hash, journal
from experiments.coupled_clock.measurement import KEYS, clock_fields, independent_fields, measure, macro
from experiments.clock_evidence.diagnostic import intervention
from experiments.phase_sync_fit.run import comparison_gate, same_support, ALL_KINDS

ARCHITECTURES = ('shared', 'separated')


def evidence_fields(period, vector):
    period, vector = np.asarray(period, np.float64), np.asarray(vector, np.float64)
    require(period.ndim == 1 and len(period) >= 1 and vector.shape == (len(period) + 1, 2)
            and np.isfinite(period).all() and np.isfinite(vector).all(), 'invalid independent evidence geometry')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        rate = np.exp2(-period)
        phase = np.arctan2(vector[:, 1], vector[:, 0]) / (2 * np.pi)
        amplitude = np.hypot(vector[:, 0], vector[:, 1])
    phase[(amplitude <= 1e-6) | ~np.isfinite(amplitude)] = np.nan
    return rate, phase


def evaluate(models, records, device, output, parents, check):
    before = {k: state_hash(v) for k, v in models.items()}
    rows = {k: [] for k in ARCHITECTURES}
    for index, row in enumerate(records):
        check()
        common, predictions = row['valid'] & row['raw_valid'], {}
        for architecture in ARCHITECTURES:
            result = {k: row[k] for k in ('id', 'role', 'work', 'frames', 'packet_sha256')}
            for kind, arm, variant in (
                ('audio', architecture + '-natural', 'natural'), ('zero', architecture + '-zero', 'zero'),
                ('same_zero', architecture + '-natural', 'zero'),
                ('time_mean', architecture + '-natural', 'time_mean'),
                ('half_roll', architecture + '-natural', 'half_roll')):
                check()
                features = torch.from_numpy(intervention(row['hidden'], variant)).to(device)[None]
                with torch.inference_mode():
                    evidence = models[arm](features)
                    period = evidence.log2_cell_period[0].cpu().numpy()
                    vector = evidence.phase_vector[0].cpu().numpy()
                fields = evidence_fields(period, vector)
                predictions[architecture + '_' + kind + '_log_period'] = period
                predictions[architecture + '_' + kind + '_phase_vector'] = vector
                if kind in ('audio', 'zero'):
                    result[kind] = measure(fields, row['reference'], row['valid'])
                result[kind + '_common'] = measure(fields, row['reference'], common)
                check()
            result['raw_common'] = measure(clock_fields(row['raw']), row['reference'], common)
            result['v1_common'] = measure(independent_fields(row['v1']), row['reference'], common)
            for name, parent in parents.items():
                old = parent['cases'][index]
                require(all(row[k] == old[k] for k in ('id', 'role', 'work', 'frames')), 'parent population differs')
                same_support(result, old)
                result[name + '_common'] = old['audio_common']
            rows[architecture].append(result)
        path = output / (row['id'] + '.evidence.npz')
        with path.open('xb') as stream:
            np.savez(stream, **predictions)
        digest = sha(path)
        for architecture in ARCHITECTURES:
            rows[architecture][-1]['prediction_sha256'] = digest
        journal(output, dict(event='recording_evaluated', recording=index + 1, total=len(records)))
        check()
    require(before == {k: state_hash(v) for k, v in models.items()}, 'evaluation changed selected weights')
    return rows


def summarize(rows):
    return {architecture: {role: {kind: macro([r for r in cases if r['role'] == role], kind)
        for kind in ALL_KINDS + ('protected_common',)} for role in ('fit', 'development', 'diagnostic')}
        for architecture, cases in rows.items()}


def compare(rows, summary, fits):
    result = {}
    for architecture in ARCHITECTURES:
        pair = [dict(f, kind='audio' if f['arm'].endswith('-natural') else 'zero-audio')
                for f in fits if f['architecture'] == architecture]
        result[architecture] = comparison_gate(rows[architecture], summary[architecture], pair)

    def le(a, b, factor=1.):
        return a is not None and b is not None and math.isfinite(a) and math.isfinite(b) and a <= factor * b

    roles = ('development', 'diagnostic')
    shared, separate = summary['shared'], summary['separated']
    nonregression = all(le(separate[role]['audio_common'][k], shared[role]['audio_common'][k])
                        for role in roles for k in KEYS)
    strict = any(le(separate['development']['audio_common'][k], shared['development']['audio_common'][k])
                 and separate['development']['audio_common'][k] < shared['development']['audio_common'][k]
                 for k in (KEYS[0], KEYS[2]))
    breadth = True
    for role in roles:
        a = [r for r in rows['separated'] if r['role'] == role]
        b = [r for r in rows['shared'] if r['role'] == role]
        require([r['id'] for r in a] == [r['id'] for r in b], 'architecture population differs')
        breadth &= bool(a) and all(sum(not le(x['audio_common'][k], y['audio_common'][k])
                                      for x, y in zip(a, b)) * 2 <= len(a) for k in (KEYS[0], KEYS[2]))
    signal = {architecture: {task: le(summary[architecture]['development']['audio'][key],
               summary[architecture]['development']['zero'][key], .9)
               for task, key in (('tempo', KEYS[0]), ('phase', KEYS[2]))} for architecture in ARCHITECTURES}
    return dict(numerical_gates=result, per_task_audio_signal=signal,
        separation_nonregression=bool(nonregression), separation_strict_development_gain=bool(strict),
        separation_recording_breadth=bool(breadth),
        supports_fusion_proposal=bool(result['separated']['passed'] and nonregression and strict and breadth),
        coherent_clock=False, independent_acceptance=False, production_change=False)
