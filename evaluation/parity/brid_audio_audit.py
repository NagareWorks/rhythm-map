"""Hash-pinned BRID smoke audio audit, never a musical-accuracy or fit gate."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import wave
import zlib

import numpy as np

from brid_training_inventory import parse_beats, require


SMOKE = ('0001', '0003', '0010', '0028')
INVENTORY_SHA = '5532c49ae45af376803bd2a7529966e26e6a934765c7968ef9d3443320332157'
LOCK_SHA = 'e7f495df75c1058f8d2c43f958fb3260aadaf2b78459bbd8c5daf0e2f838f158'
ROOT = Path(__file__).resolve().parents[2]


def safe_file(root, relative):
    path = PurePosixPath(relative)
    require(not path.is_absolute() and path.parts and
            all(part not in ('.', '..') for part in path.parts) and
            '\\' not in relative and ':' not in relative, 'unsafe asset path')
    require(not root.is_symlink(), 'linked input root')
    root = root.resolve(strict=True)
    candidate = root
    for part in path.parts:
        candidate = candidate / part
        require(not candidate.is_symlink(), 'linked input component')
    require(candidate.is_file() and candidate.resolve().is_relative_to(root), 'missing/escaped file')
    return candidate


def verified_bytes(root, asset):
    path = safe_file(root, asset['path'])
    with path.open('rb') as stream:
        data = stream.read(asset['size_bytes'] + 1)
    require(len(data) == asset['size_bytes'], 'asset size differs')
    require(hashlib.sha256(data).hexdigest() == asset['sha256'], 'asset SHA-256 differs')
    return data


def quantiles(values):
    if len(values) == 0:
        return None
    return [round(float(value), 6) for value in np.quantile(values, [0, .25, .5, .75, 1])]


def acoustic_diagnostics(pcm, sample_rate, beats):
    """5 ms multichannel RMS rises; descriptive only, no time correction."""
    require(pcm.ndim == 2 and len(pcm) > 0 and np.isfinite(pcm).all(), 'invalid PCM')
    block = max(1, round(.005 * sample_rate))
    count = len(pcm) // block
    require(count >= 3, 'audio too short for diagnostics')
    frames = pcm[:count * block].reshape(count, block, pcm.shape[1])
    rms = np.sqrt(np.mean(frames * frames, axis=(1, 2)))
    centers = (np.arange(count) * block + (block - 1) / 2) / sample_rate
    rise = np.maximum(np.diff(rms, prepend=rms[0]), 0)
    peaks = np.flatnonzero((rise[1:-1] >= rise[:-2]) &
                          (rise[1:-1] >= rise[2:]) & (rise[1:-1] > 0)) + 1
    peak_times = centers[peaks]
    distances = [float(np.min(np.abs(peak_times - beat))) * 1000 for beat in beats] if len(peaks) else []
    offsets = np.arange(-30, 31, dtype=np.float64) * .005
    common = np.asarray([beat for beat in beats if centers[0] + .15 <= beat <= centers[-1] - .15])
    profile = np.asarray([np.mean(np.interp(common + offset, centers, rise))
                          for offset in offsets]) if len(common) else np.array([])
    return dict(
        block_samples=block, block_seconds=block/sample_rate,
        complete_block_count=count, discarded_tail_samples=len(pcm)-count*block,
        rms_quantiles=quantiles(rms),
        nearest_rms_rise_peak_distance_ms_quantiles=quantiles(distances),
        offset_profile_common_beats=len(common),
        best_diagnostic_offset_seconds=(round(float(offsets[np.argmax(profile)]), 6)
                                        if len(profile) and np.max(profile) > 0 else None),
        offset_profile_mean_rise=[round(float(x), 10) for x in profile],
        offset_is_correction=False, musical_alignment_certified=False)


def analyse_wav(payload, beats):
    require(len(beats) >= 2 and np.isfinite(beats).all() and
            all(left < right for left, right in zip(beats, beats[1:])),
            'beats must be finite, ordered and nonempty')
    with wave.open(io.BytesIO(payload), 'rb') as audio:
        channels, width, rate, count, compression, _ = audio.getparams()
        require(compression == 'NONE' and width == 2 and channels in (1, 2), 'unsupported PCM encoding')
        require(rate > 0 and count > 0, 'empty or invalid WAV')
        data = audio.readframes(count + 1)
    require(len(data) == count * channels * width, 'truncated/inconsistent WAV data')
    samples = np.frombuffer(data, dtype='<i2').reshape(count, channels)
    pcm = samples.astype(np.float64) / 32768
    duration = count / rate
    out_of_bounds = [index for index, beat in enumerate(beats) if not 0 <= beat < duration]
    silent_channels = [i for i in range(channels) if not np.any(samples[:, i])]
    problems = []
    if out_of_bounds:
        problems.append('reference_beats_outside_half_open_audio')
    if len(silent_channels) == channels:
        problems.append('entire_audio_is_zero')
    return dict(
        source_sample_rate_hz=rate, channels=channels, sample_width_bytes=width,
        pcm_frames=count, duration_seconds=duration,
        pcm_sha256=hashlib.sha256(data).hexdigest(),
        rms_by_channel=[round(float(x), 10) for x in np.sqrt(np.mean(pcm * pcm, axis=0))],
        peak_absolute_by_channel=[round(float(x), 10) for x in np.max(np.abs(pcm), axis=0)],
        full_scale_sample_fraction_by_channel=[round(float(x), 10) for x in
            np.mean((samples == -32768) | (samples == 32767), axis=0)],
        zero_channels=silent_channels, out_of_bounds_beat_indices=out_of_bounds,
        unannotated_leading_seconds=beats[0],
        unannotated_trailing_seconds=duration-beats[-1],
        structural_findings=problems, structural_gate_passed=not problems,
        acoustic_diagnostics=acoustic_diagnostics(pcm, rate, beats))


def check_decoder(report, asset, duration):
    require(report['schema_version'] == 1 and report['sha256'] == asset['sha256'] and
            report['size_bytes'] == asset['size_bytes'], 'Rust decoder identity differs')
    require(report['decoded_sample_rate_hz'] == 22050 and
            np.isfinite(report['duration_s']) and report['duration_s'] > 0 and
            abs(report['duration_s'] - duration) <= 1/22050,
            'Rust/native duration differs by more than one decoded sample')


def load_pins():
    inventory_bytes = (ROOT / 'evaluation/datasets/brid-training-candidate-v1.json').read_bytes()
    lock_bytes = (ROOT / 'evaluation/datasets/brid-training-smoke-v1.json').read_bytes()
    require(hashlib.sha256(inventory_bytes).hexdigest() == INVENTORY_SHA, 'inventory drift')
    require(hashlib.sha256(lock_bytes).hexdigest() == LOCK_SHA, 'smoke lock drift')
    inventory, lock = json.loads(inventory_bytes), json.loads(lock_bytes)
    return inventory, lock


def build_report(directory):
    inventory, lock = load_pins()
    assets = {asset['path']: asset for asset in lock['assets']}
    require(len(assets) == len(lock['assets']) == 12, 'duplicate or missing asset')
    expected_paths = {path for recording_id in SMOKE for path in
                      (f'audio/{recording_id}.wav', f'annotations/{recording_id}.beats',
                       f'annotations/{recording_id}.bpm')}
    require(set(assets) == expected_paths, 'smoke membership drift')
    sources = {row['recording_id']: row for row in inventory['records']}
    rows = []
    for recording_id in SMOKE:
        source = sources[recording_id]
        asset = assets[f'audio/{recording_id}.wav']
        payload = verified_bytes(directory, asset)
        require(f'{zlib.crc32(payload):08x}' == source['audio_candidate']['zip_crc32'],
                'original audio directory CRC differs')
        times, positions = parse_beats(verified_bytes(directory, assets[f'annotations/{recording_id}.beats']))
        bpm = float(verified_bytes(directory, assets[f'annotations/{recording_id}.bpm']).decode())
        require(bpm == source['published_global_bpm'] and len(times) == source['beat_count'] and
                positions.count(1) == source['downbeat_count'], 'source annotation summary drift')
        audio = analyse_wav(payload, times)
        decoder = json.loads(safe_file(directory, f'decoder-{recording_id}.json').read_text(encoding='utf-8'))
        check_decoder(decoder, asset, audio['duration_seconds'])
        rows.append(dict(recording_id=recording_id, style=source['style'],
                         intended_role='training_candidate_only', leakage_group='brid-corpus-v1',
                         audio_sha256=asset['sha256'], audio_size_bytes=asset['size_bytes'],
                         beat_count=len(times), downbeat_count=positions.count(1),
                         annotated_span_seconds=round(times[-1]-times[0], 6),
                         published_global_bpm=bpm, ibi_seconds_quantiles=quantiles(np.diff(times)),
                         native_audio=audio, rust_decoder=decoder))
    return dict(
        schema='rhythm-map.brid-audio-smoke.v1',
        status='verified_source_bytes_and_audio_structure_not_musical_acceptance',
        inventory_sha256=INVENTORY_SHA, fetch_lock_sha256=LOCK_SHA,
        selection_rule='first_numeric_mixture_id_per_annotated_style_before_model_outputs',
        recording_ids=list(SMOKE), source_assets_verified=len(assets), audio_recordings_decoded=len(rows),
        audio_bytes=sum(row['audio_size_bytes'] for row in rows),
        duration_seconds=sum(row['native_audio']['duration_seconds'] for row in rows),
        annotated_span_seconds=round(sum(row['annotated_span_seconds'] for row in rows), 6),
        beat_count=sum(row['beat_count'] for row in rows),
        downbeat_count=sum(row['downbeat_count'] for row in rows),
        structural_gate_passed=all(row['native_audio']['structural_gate_passed'] for row in rows),
        semantic_annotation_review='pending_not_replaced_by_onset_diagnostics',
        independent_work_count=None, admitted_training_recordings=0,
        framewise_tempo_targets_created=False, tempo_change_labels_created=False,
        feature_access=False, model_inference=False, optimizer_steps=0,
        project_holdout_access=False, public_default_changed=False, records=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir', type=Path, required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument('--output', type=Path)
    output.add_argument('--check', type=Path)
    args = parser.parse_args()
    report = build_report(args.audio_dir)
    if args.check:
        require(json.loads(args.check.read_text(encoding='utf-8')) == report, 'retained report differs')
    else:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()
