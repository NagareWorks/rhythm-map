"""Inspect all retained four-arm rate fields; no forward or optimizer call."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.coupled_clock.clock import require
from experiments.coupled_clock.measurement import clock_fields, independent_fields, measure
from experiments.phase_attribution.run import sha, verify, audit_metric
from experiments.separated_evidence_fit.evaluation import evidence_fields
from experiments.tempo_attribution.diagnostic import analyze, summarize

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIT = ROOT / 'experiments/separated_evidence_fit'
INPUT = ROOT / 'experiments/coupled_clock/inputs-v1.json'
PINS = {FIT / 'results-v1.json': '251dcb1f421d1588bf8b6d11709e05696a15af57624d659600e51a15059984a8',
        FIT / 'plan-v1.json': 'f7798c4b78f06af3eb378e21660fbda6f9b372b734b02ba855a557fafd940408',
        INPUT: 'b4f05c10ebede84c394b0b8e06f6723b096d1f9a0a48e63024eb81d913fffd9c'}
ARCHITECTURES = ('shared', 'separated')
VARIANTS = ('audio', 'zero', 'same_zero', 'time_mean', 'half_roll')


def source_hashes():
    paths = [HERE / name for name in ('__init__.py', 'PROTOCOL-v1.md', 'diagnostic.py', 'run.py')]
    paths += [ROOT / 'experiments/phase_attribution' / name for name in ('diagnostic.py', 'run.py')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def run(inputs, predictions):
    started, identity = time.monotonic(), source_hashes()
    for path, digest in PINS.items():
        verify(path, digest)
    old = json.loads((FIT / 'results-v1.json').read_bytes())
    plan = json.loads((FIT / 'plan-v1.json').read_bytes())
    manifest = json.loads(INPUT.read_bytes())
    keys = ('id', 'role', 'work', 'frames', 'packet_sha256')
    population = [tuple(r[k] for k in keys) for r in manifest['cases']]
    require([tuple(r[k] for k in keys) for r in plan['population']] == population, 'plan population changed')
    for architecture in ARCHITECTURES:
        require([tuple(r[k] for k in keys) for r in old['cases'][architecture]] == population, 'result population changed')
    require([r['prediction_sha256'] for r in old['cases']['shared']] ==
            [r['prediction_sha256'] for r in old['cases']['separated']], 'prediction identities differ')
    artifacts = [(inputs / 'inputs.json', PINS[INPUT])]
    for row in old['cases']['shared']:
        artifacts += [(inputs / (row['id'] + '.npz'), row['packet_sha256']),
                      (predictions / (row['id'] + '.evidence.npz'), row['prediction_sha256'])]
    prerequisites = list(PINS.items()) + [(ROOT / p, digest) for p, digest in plan['source_sha256'].items()]
    for path, digest in prerequisites + artifacts:
        verify(path, digest)
    rows, audited = [], 0
    for index, item in enumerate(old['cases']['shared']):
        row = {k: item[k] for k in keys + ('prediction_sha256',)}
        with np.load(inputs / (row['id'] + '.npz'), allow_pickle=False) as packet:
            reference, valid, raw, raw_valid, v1 = (packet[k] for k in ('reference', 'valid', 'raw', 'raw_valid', 'v1'))
        require(reference.shape == valid.shape == raw.shape == raw_valid.shape == (row['frames'],)
                and valid.dtype == raw_valid.dtype == bool, 'packet geometry differs')
        with np.load(predictions / (row['id'] + '.evidence.npz'), allow_pickle=False) as saved:
            expected_names = {a + '_' + v + '_' + field for a in ARCHITECTURES for v in VARIANTS
                              for field in ('log_period', 'phase_vector')}
            require(set(saved.files) == expected_names, 'prediction fields differ')
            fields = {a + '_' + v: evidence_fields(saved[a + '_' + v + '_log_period'],
                       saved[a + '_' + v + '_phase_vector']) for a in ARCHITECTURES for v in VARIANTS}
        fields.update(raw=clock_fields(raw), v1=independent_fields(v1))
        for support, mask in (('native', valid), ('common', valid & raw_valid)):
            row[support] = {}
            for name, value in fields.items():
                if support == 'native' and name in ('raw', 'v1'):
                    continue
                result = dict(original=measure(value, reference, mask), **analyze(value[0], reference, mask))
                row[support][name] = result
                if name in ('raw', 'v1'):
                    for architecture in ARCHITECTURES:
                        audit_metric(result['original'], old['cases'][architecture][index][name + '_common'])
                        audited += 1
                else:
                    architecture, variant = name.split('_', 1)
                    if support == 'common' or variant in ('audio', 'zero'):
                        key = variant + '_common' if support == 'common' else variant
                        audit_metric(result['original'], old['cases'][architecture][index][key])
                        audited += 1
        rows.append(row)
    summary = summarize(rows)
    for path, digest in prerequisites + artifacts:
        verify(path, digest)
    require(identity == source_hashes(), 'analysis source changed')
    require(audited == 720, 'incomplete original metric audit')
    return dict(schema='rhythm-map.tempo-attribution.v1', source_sha256=identity,
                prerequisite_sha256={p.relative_to(ROOT).as_posix(): digest for p, digest in PINS.items()},
                cases=rows, summary=summary, original_metric_blocks_audited=audited,
                prior_decision=old['decision'], prior_gate=old['gate'], elapsed_s=time.monotonic() - started,
                training=False, model_forward=False, weights_loaded=False, hidden_features_loaded=False,
                holdout_access=False, independent_acceptance=False, production_change=False,
                decision='retained_tempo_diagnosis_only_no_automatic_fit_or_fusion')


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
                          seconds=result['elapsed_s'], training=False, model_forward=False)))


if __name__ == '__main__':
    main()
