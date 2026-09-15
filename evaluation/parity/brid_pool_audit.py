"""Full-pool BRID byte/PCM audit; no inference, fit admission or musical grading."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

from brid_audio_audit import (INVENTORY_SHA, ROOT, analyse_wav, check_decoder,
                              load_pins, quantiles, safe_file, verified_bytes)
from brid_training_inventory import parse_beats, require


POOL_LOCK_SHA = 'e19481eb14707777736b0087302924448b3c7bdf6338d004694664a59d8440f3'
IDS = tuple(f'{index:04d}' for index in range(1, 94))


def load_pool():
    inventory, smoke = load_pins()
    raw = (ROOT / 'evaluation/datasets/brid-training-mixtures-v1.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == POOL_LOCK_SHA, 'full-pool lock drift')
    lock = json.loads(raw)
    assets = {asset['path']: asset for asset in lock['assets']}
    require(len(assets) == len(lock['assets']) == 279, 'duplicate or incomplete full pool')
    require(set(assets) == {f'{folder}/{recording_id}.{extension}' for recording_id in IDS
            for folder, extension in (('audio', 'wav'), ('annotations', 'beats'), ('annotations', 'bpm'))},
            'full-pool membership differs')
    for original in smoke['assets']:
        require(assets[original['path']] == original, 'frozen smoke asset drift')
    return inventory, assets


def summarize(rows):
    require([row['recording_id'] for row in rows] == list(IDS), 'expected exact 93 IDs, no filtering')
    pcm_groups = defaultdict(list)
    for row in rows:
        pcm_groups[row['native_audio']['pcm_sha256']].append(row['recording_id'])
    return dict(
        schema='rhythm-map.brid-audio-pool.v1',
        status='verified_source_bytes_and_audio_structure_not_musical_acceptance',
        inventory_sha256=INVENTORY_SHA, fetch_lock_sha256=POOL_LOCK_SHA,
        selection_rule='all_93_released_beat_annotated_acoustic_mixtures_before_model_outputs',
        source_assets_verified=279, audio_recordings_decoded=len(rows),
        audio_bytes=sum(row['audio_size_bytes'] for row in rows),
        duration_seconds=round(sum(row['native_audio']['duration_seconds'] for row in rows), 9),
        annotated_span_seconds=round(sum(row['annotated_span_seconds'] for row in rows), 6),
        beat_count=sum(row['beat_count'] for row in rows),
        downbeat_count=sum(row['downbeat_count'] for row in rows),
        style_counts=dict(sorted(Counter(row['style'] for row in rows).items())),
        structural_gate_passed=all(row['native_audio']['structural_gate_passed'] for row in rows),
        structural_failure_ids=[row['recording_id'] for row in rows
                                if not row['native_audio']['structural_gate_passed']],
        zero_channel_ids=[row['recording_id'] for row in rows if row['native_audio']['zero_channels']],
        full_scale_sample_ids=[row['recording_id'] for row in rows
                               if any(row['native_audio']['full_scale_sample_fraction_by_channel'])],
        exact_duplicate_native_pcm_groups=[group for group in pcm_groups.values() if len(group) > 1],
        semantic_annotation_review='pending_not_replaced_by_onset_diagnostics',
        independent_work_count=None, conservative_leakage_group_count=1,
        admitted_training_recordings=0, framewise_tempo_targets_created=False,
        tempo_change_labels_created=False, feature_access=False, model_inference=False,
        optimizer_steps=0, project_holdout_access=False, public_default_changed=False,
        records=rows)


def build_report(directory):
    inventory, assets = load_pool()
    sources = {row['recording_id']: row for row in inventory['records']}
    rows = []
    for recording_id in IDS:
        source = sources[recording_id]
        asset = assets[f'audio/{recording_id}.wav']
        payload = verified_bytes(directory, asset)
        require(f'{zlib.crc32(payload):08x}' == source['audio_candidate']['zip_crc32'],
                'original audio CRC differs')
        times, positions = parse_beats(verified_bytes(directory, assets[f'annotations/{recording_id}.beats']))
        bpm = float(verified_bytes(directory, assets[f'annotations/{recording_id}.bpm']).decode())
        require(bpm == source['published_global_bpm'] and len(times) == source['beat_count'] and
                positions.count(1) == source['downbeat_count'], 'original annotation summary drift')
        native = analyse_wav(payload, times)
        decoder = json.loads(safe_file(directory, f'decoder-{recording_id}.json').read_text(encoding='utf-8'))
        check_decoder(decoder, asset, native['duration_seconds'])
        rows.append(dict(recording_id=recording_id, style=source['style'],
                         intended_role='training_candidate_only', leakage_group='brid-corpus-v1',
                         audio_sha256=asset['sha256'], audio_size_bytes=asset['size_bytes'],
                         beat_count=len(times), downbeat_count=positions.count(1),
                         annotated_span_seconds=round(times[-1]-times[0], 6),
                         published_global_bpm=bpm, ibi_seconds_quantiles=quantiles(np.diff(times)),
                         native_audio=native, rust_decoder=decoder))
    return summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir', type=Path, required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument('--output', type=Path)
    output.add_argument('--check', type=Path)
    args = parser.parse_args()
    report = build_report(args.audio_dir)
    if args.check:
        require(json.loads(args.check.read_text(encoding='utf-8')) == report, 'retained pool report differs')
    else:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()
