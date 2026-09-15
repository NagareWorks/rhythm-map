"""Freeze both source-slot schedules without audio, encoder or optimizer access."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PINS = {
    'experiments/coupled_clock/inputs-v1.json': 'b4f05c10ebede84c394b0b8e06f6723b096d1f9a0a48e63024eb81d913fffd9c',
    'experiments/tempo_variation/results-v1.json': '774b86f5f1ddf7a24758af8f742a14e544e7778eeb0c571af285562c02330dc9',
    'evaluation/datasets/brid-reference-v1.json': 'ad70e2ed3a005c1d349368e25be405abb5f6a779f1efcf22d2f495291aa34a6a',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule(records, ids):
    require(len(records) == 20 and len({r['id'] for r in records}) == 20
            and all(r['role'] == 'fit' for r in records), 'FIT population differs')
    works = sorted({r['work'] for r in records})
    require(len(works) == 9 and tuple(ids) == tuple(f'{i:04d}' for i in range(1, 94)),
            'source groups or BRID membership differ')
    old_rng, new_rng = np.random.default_rng(24142), np.random.default_rng(34142)
    new_order = list(new_rng.permutation(ids)) + list(new_rng.permutation(ids))[:7]
    grouped = {w: [r['id'] for r in records if r['work'] == w] for w in works}
    draws = []
    for epoch in range(20):
        order = np.random.default_rng(142+epoch).permutation(20)
        for offset in range(0, 20, 4):
            work = works[int(old_rng.integers(len(works)))]
            extra = grouped[work][int(old_rng.integers(len(grouped[work])))]
            update = len(draws)+1
            draws.append(dict(update=update, epoch=epoch+1,
                              shared_fit_ids=[records[i]['id'] for i in order[offset:offset+4]],
                              control_extra_id=extra, control_extra_work=work,
                              primary_extra_id=str(new_order[update-1]),
                              primary_extra_group='brid-corpus-v1'))
    return draws


def plan():
    require(all(sha(ROOT/p) == digest for p, digest in PINS.items()), 'pinned source changed')
    old = json.loads((ROOT/'experiments/coupled_clock/inputs-v1.json').read_bytes())
    refs = json.loads((ROOT/'evaluation/datasets/brid-reference-v1.json').read_bytes())
    records = [r for r in old['cases'] if r['role'] == 'fit']
    draws = schedule(records, [r['recording_id'] for r in refs['records']])
    counts = Counter(r['primary_extra_id'] for r in draws)
    closure = dict(PINS)
    for relative in ('experiments/tempo_source_expansion/PROTOCOL-v1.md',
                     'experiments/tempo_source_expansion/design.py',
                     'experiments/local_tempo/selection.py',
                     'experiments/coupled_clock/supervision.py',
                     'experiments/separated_evidence/supervision.py'):
        closure[relative] = sha(ROOT/relative)
    return dict(schema='rhythm-map.tempo-source-expansion-design.v1',
                source_sha256=closure, status='design_frozen_execution_not_admitted',
                arms=['old-source-control', 'brid-source-primary'],
                initial_weights_sha256='1b25f43474ba780d845dccd032a8e4c150ebabc77c50f31cc975e9419c4e214a',
                updates_per_arm=100, epochs=20, shared_native_records_per_update=4,
                extra_native_records_per_update=1, natural_seeds=[142+i for i in range(20)],
                pair_seed=8142, control_extra_seed=24142, primary_extra_seed=34142,
                loss_coefficients=dict(shared_native=.9, extra_native=.1, pair=1., old_fit_retention=1.),
                primary_record_draw_counts=dict(sorted(counts.items())), schedule=draws,
                old_roles={r['id']: dict(role=r['role'], work=r['work']) for r in old['cases']},
                admission_pending=['source_alignment_disposition', 'all_93_feature_packet_capture_and_replay',
                                   'old_input_identity_replay', 'executable_runner_closure_and_execution_freeze'],
                optimizer_steps=0, encoder_calls=0, holdout_access=False, production_change=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--check', type=Path)
    args = parser.parse_args()
    value = plan()
    if args.check:
        require(value == json.loads(args.check.read_bytes()), 'retained design changed')
    else:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
