"""Audit pinned saved arrays and emit attribution; no inference or training."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.coupled_clock.clock import require
from experiments.phase_attribution.diagnostic import KINDS, analyze, macro

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPORT = ROOT / 'experiments/scaled_phase_fit/results-v1.json'
PLAN = ROOT / 'experiments/scaled_phase_fit/plan-v1.json'
INPUT = ROOT / 'experiments/coupled_clock/inputs-v1.json'
PINS = {
    REPORT: '73454ce089cd3821c51d622ccbd491c56ecc53765e9552f6910d4a56e6c840a5',
    PLAN: 'c159b0f90369793d67cae5d8fc8c94264252d152dfb2e3de6eec257967f55193',
    INPUT: 'b4f05c10ebede84c394b0b8e06f6723b096d1f9a0a48e63024eb81d913fffd9c',
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(path, expected):
    require(sha(path) == expected, 'artifact identity differs: ' + path.name)


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in
            (HERE / '__init__.py', HERE / 'PROTOCOL-v1.md', HERE / 'diagnostic.py', HERE / 'run.py')}


def audit_metric(actual, expected):
    require(actual.keys() == expected.keys(), 'original metric keys differ')
    for key in actual:
        a, b = actual[key], expected[key]
        require((a is None and b is None) or (a is not None and b is not None
                and np.isclose(a, b, atol=1e-10, rtol=1e-12)), 'original metric differs: ' + key)


def run(inputs, predictions):
    started = time.monotonic()
    sources = source_hashes()
    for path, digest in PINS.items():
        verify(path, digest)
    report, plan = json.loads(REPORT.read_bytes()), json.loads(PLAN.read_bytes())
    verify(inputs / 'inputs.json', PINS[INPUT])
    manifest = json.loads(INPUT.read_bytes())
    for name, digest in plan['source_sha256'].items():
        verify(ROOT / name, digest)
    population = [(r['id'], r['role'], r['work'], r['frames'], r['packet_sha256']) for r in report['cases']]
    for other in (manifest['cases'], plan['population']):
        require(population == [(r['id'], r['role'], r['work'], r['frames'], r['packet_sha256']) for r in other],
                'population differs')
    # Audit every identity first, before any value is read or partial result made.
    artifacts = []
    for row in report['cases']:
        artifacts.extend(((inputs / (row['id'] + '.npz'), row['packet_sha256']),
                          (predictions / (row['id'] + '.prediction.npz'), row['prediction_sha256'])))
    for path, digest in artifacts:
        verify(path, digest)
    rows = []
    audited = 0
    for old in report['cases']:
        row = {key: old[key] for key in ('id', 'role', 'work', 'frames', 'packet_sha256', 'prediction_sha256')}
        with np.load(inputs / (row['id'] + '.npz'), allow_pickle=False) as packet:
            # NPZ is lazy: hidden features and v1 weights/fields are not loaded.
            reference, valid, raw, raw_valid = (packet[k] for k in ('reference', 'valid', 'raw', 'raw_valid'))
        require(reference.shape == valid.shape == raw.shape == raw_valid.shape == (row['frames'],)
                and valid.dtype == raw_valid.dtype == bool, 'packet geometry differs')
        with np.load(predictions / (row['id'] + '.prediction.npz'), allow_pickle=False) as saved:
            require(set(saved.files) == {k for n in KINDS[:-1] for k in (n, 'fields_' + n)},
                    'prediction arrays differ')
            for support, mask in (('native', valid), ('common', valid & raw_valid)):
                row[support] = {}
                for kind in KINDS:
                    if kind == 'raw' and support == 'native':
                        continue  # Raw is not extrapolated outside its own support.
                    q = raw if kind == 'raw' else saved[kind]
                    fields = None if kind == 'raw' else saved['fields_' + kind]
                    gain = None if kind == 'raw' else report['selected_gain']['zero' if kind == 'zero' else 'audio']
                    result = analyze(q, reference, mask, fields, gain)
                    row[support][kind] = result
                    old_key = kind + '_common' if support == 'common' else kind
                    if old_key in old:
                        audit_metric(result['original'], old[old_key])
                        audited += 1
        rows.append(row)
    summary = {role: {support: {kind: {section: macro(
        [r for r in rows if r['role'] == role], support, kind, section)
        for section in ('original', 'detail')} for kind in rows[0][support]}
        for support in ('native', 'common')} for role in ('fit', 'development', 'diagnostic')}
    for path, digest in artifacts + list(PINS.items()):
        verify(path, digest)
    for name, digest in plan['source_sha256'].items():
        verify(ROOT / name, digest)
    require(source_hashes() == sources, 'diagnostic changed during execution')
    return dict(schema='rhythm-map.phase-attribution.v1', source_sha256=sources,
                prerequisite_sha256={p.relative_to(ROOT).as_posix(): d for p, d in PINS.items()},
                cases=rows, summary=summary, original_metric_blocks_audited=audited,
                prior_decision=report['decision'], training=False, model_forward=False,
                weights_loaded=False, hidden_features_loaded=False, holdout_access=False,
                independent_acceptance=False, production_change=False,
                elapsed_s=time.monotonic() - started,
                decision='fixed_prediction_diagnosis_only_no_automatic_fit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'predictions', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'new private output required')
    result = run(args.inputs, args.predictions)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(records=len(result['cases']), audited=result['original_metric_blocks_audited'],
                          seconds=result['elapsed_s'], training=False)))


if __name__ == '__main__':
    main()
