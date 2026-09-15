"""Offline independent archive/identity replay, not an accuracy evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.parity import prehead_capture as geometry
from .prepare import (ROOT, IDS, CHECKPOINT_SHA, array_manifest, load_inputs,
                      require, sha, write_json)

ENCODER_STATE_SHA = 'db0c7b0d5e2b42aab3651477a058e72b6a82e236ef538cc1566131670da468ce'


def verify_features(row, arrays, frames):
    require(row['frames'] == frames and set(arrays) == {*geometry.KEYS, 'owner'}, 'wrong feature extent/fields')
    for key in geometry.KEYS:
        geometry.finite_f32(arrays[key], (frames, 512) if key == 'hidden' else (frames,))
    owner = np.full(frames, -1, np.int32)
    layout = geometry.layout(frames)
    for index, part in reversed(list(enumerate(layout))):
        owner[part['write_start']:part['write_end']] = index
    require(arrays['owner'].dtype == np.int32 and np.array_equal(owner, arrays['owner']), 'chunk ownership changed')
    require(row['arrays'] == array_manifest(arrays), 'feature array identity changed')
    require(row['chunks'] == len(layout) and row['forward_calls'] == 3*len(layout)
            and row['untapped_heads_bit_exact'] is True and row['hidden_replay_bit_exact'] is True
            and row['roundtrip_bit_exact'] is True
            and np.isfinite(row['max_head_reconstruction_error'])
            and row['max_head_reconstruction_error'] >= 0, 'capture checks incomplete')


def audit(inputs, features, input_sha):
    source = load_inputs(inputs, input_sha)
    registration = json.loads((features/'registration.json').read_bytes())
    captured = json.loads((features/'report.json').read_bytes())
    require(not (features/'failure.json').exists(), 'failed capture is not admitted')
    require(captured['registration_sha256'] == sha(features/'registration.json')
            and captured['input_report_sha256'] == registration['input_report_sha256'] == input_sha
            and registration['source_sha256'] == source['source_sha256']
            and registration['checkpoint_sha256'] == CHECKPOINT_SHA
            and registration['recording_ids'] == IDS
            and registration['passes_per_chunk'] == 3 and registration['optimizer_steps'] == 0,
            'registration/source identity mismatch')
    require([r['recording_id'] for r in captured['records']] == IDS
            and captured['recording_count'] == 93 and captured['frames'] == source['frames'] == 133247
            and captured['encoder_state_sha256'] == ENCODER_STATE_SHA and captured['encoder_unchanged'] is True
            and captured['optimizer_steps'] == 0 and captured['training_ready'] is False
            and captured['holdout_access'] is False and captured['production_change'] is False,
            'encoder/population/admission contract differs')
    rows = []
    for old, row in zip(source['records'], captured['records']):
        require(row['input_sha256'] == old['input_sha256'], 'feature input join differs')
        path = features/f"{row['recording_id']}.features.npz"
        require(sha(path) == row['feature_sha256'], 'feature packet changed')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        verify_features(row, arrays, old['frames'])
        rows.append(dict(row, audio_sha256=old['audio_sha256'], pcm_sha256=old['pcm_sha256'],
                         input_arrays=old['arrays']))
    require(captured['encoder_forward_calls'] == sum(r['forward_calls'] for r in rows), 'forward accounting differs')
    return dict(schema='rhythm-map.brid-feature-audit.v1',
        status='frozen_features_prepared_not_fit_admission',
        input_report_sha256=input_sha, capture_report_sha256=sha(features/'report.json'),
        audit_source_sha256=sha(Path(__file__)), registration=registration,
        records=rows, recording_count=93, frames=133247,
        source_groups=1, alignment_disposition=source['alignment_disposition'], label_shifts=0,
        encoder_state_sha256=ENCODER_STATE_SHA, encoder_unchanged=True,
        encoder_forward_calls=captured['encoder_forward_calls'],
        capture_elapsed_s=captured['elapsed_s'], peak_cuda_allocated_bytes=captured['peak_cuda_allocated_bytes'],
        archives_replayed=True, optimizer_steps=0, training_ready=False, holdout_access=False,
        production_change=False,
        admission_pending=['old_input_identity_replay', 'executable_runner_closure_and_execution_freeze'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'features'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--input-report-sha256', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--check', type=Path)
    args = parser.parse_args()
    report = audit(args.inputs, args.features, args.input_report_sha256)
    if args.output:
        write_json(args.output, report)
    else:
        require(report == json.loads(args.check.read_bytes()), 'retained feature audit changed')
    print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'registration')}), flush=True)
