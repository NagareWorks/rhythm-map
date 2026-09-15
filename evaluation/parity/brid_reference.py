"""Offline BRID native-reference packets; no encoder, fitting or label repair."""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import wave

import numpy as np

from brid_audio_audit import ROOT, check_decoder, quantiles, verified_bytes
from brid_pool_audit import IDS, POOL_LOCK_SHA, load_pool
from brid_training_inventory import parse_beats, require
from prehead_capture import array_hash

AUDIT_PATH = 'evaluation/datasets/brid-audio-pool-v1.json'
AUDIT_SHA = '7ce039472dea873397c2a8f1078e59205edb16ae11cfbc3c5ca1aaddf755083b'
FPS = 50


def source_hashes():
    paths = [f'evaluation/parity/{name}.py' for name in (
        'brid_reference', 'brid_audio_audit', 'brid_pool_audit',
        'brid_training_inventory', 'prehead_capture')]
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def reference_arrays(beats, positions, frames, duration, support=None):
    """Same bracketed q(t) as coupled_clock.reference_clock, using only NumPy.

    Frame t lives at t/50; period cell t covers [t/50,(t+1)/50].
    A cell needs both reference endpoints. The last beat is an exclusive bound,
    matching the existing loss contract. No downbeat phase is inferred from q.
    """
    beats = np.asarray(beats, np.float64)
    positions = np.asarray(positions)
    require(beats.ndim == 1 and len(beats) >= 2 and np.isfinite(beats).all()
            and (beats >= 0).all() and (np.diff(beats) > 0).all(), 'invalid beats')
    require(positions.shape == beats.shape and positions.dtype.kind in 'iu'
            and np.isin(positions, [1, 2]).all(), 'invalid beat positions')
    require(type(frames) is int and 2 <= frames <= 4096
            and np.isfinite(duration) and duration > 0 and beats[-1] < duration,
            'invalid extent or beat outside audio')
    require(np.all(positions[1:] != positions[:-1]), 'position discontinuity needs review')
    times = np.arange(frames, dtype=np.float64) / FPS
    index = np.searchsorted(beats, times, side='right') - 1
    valid = (index >= 0) & (index < len(beats)-1) & (times < duration)
    if support is not None:
        support = np.asarray(support)
        require(support.shape == (frames,) and support.dtype == bool, 'invalid support')
        valid &= support
    reference = np.full(frames, np.nan, np.float64)
    i = index[valid]
    reference[valid] = i + (times[valid]-beats[i])/(beats[i+1]-beats[i])
    cell_valid = valid[:-1] & valid[1:]
    require(cell_valid.any(), 'no supported native tempo cells')
    rate = np.diff(reference)[cell_valid] * FPS
    require(np.isfinite(rate).all() and (rate > 0).all(), 'nonpositive native rate')
    # Canonical packets stop before transcendental functions: NumPy log2 can
    # differ by one float64 ULP across supported CPUs/platforms. The unchanged
    # training loss derives its log target from this exact reference and mask.
    return dict(beats=beats, positions=positions.astype(np.int8), reference=reference,
                valid=valid, cell_valid=cell_valid)


def tempo_target(arrays):
    """Runtime-derived target for numeric checks, deliberately not serialized."""
    cells = arrays['cell_valid']
    period = np.full(len(cells), np.nan, np.float64)
    period[cells] = -np.log2(np.diff(arrays['reference'])[cells]*FPS)
    return period


def alignment_review(payload, beats):
    """Recheck *every* recording using equal-beat and temporal-split summaries.

    RMS rise is still not a beat oracle. Fixed +/-150 ms candidates and interior
    population match the earlier diagnostic; no result changes a timestamp/mask.
    """
    with wave.open(io.BytesIO(payload), 'rb') as audio:
        channels, width, rate, count, compression, _ = audio.getparams()
        require(width == 2 and compression == 'NONE' and channels == 2 and rate == 44100,
                'unexpected source PCM format')
        raw = audio.readframes(count+1)
    require(len(raw) == count*channels*width, 'incomplete PCM')
    pcm = np.frombuffer(raw, dtype='<i2').reshape(count, channels).astype(np.float64)/32768
    block = round(.005*rate)
    n = count//block
    require(n >= 3, 'insufficient diagnostic audio')
    rms = np.sqrt(np.mean(pcm[:n*block].reshape(n, block, channels)**2, axis=(1, 2)))
    centers = (np.arange(n)*block + (block-1)/2)/rate
    rise = np.maximum(np.diff(rms, prepend=rms[0]), 0)
    common = np.asarray([b for b in beats if centers[0]+.15 <= b <= centers[-1]-.15])
    require(len(common) >= 2, 'insufficient interior beats for review')
    offsets = np.arange(-30, 31)*.005
    values = np.stack([np.interp(common+offset, centers, rise) for offset in offsets], axis=1)
    # Normalization keeps one loud attack from determining the entire profile.
    maxima = values.max(axis=1)
    nonzero = maxima > 0
    require(nonzero.any(), 'no acoustic rise evidence')
    normalized = np.zeros_like(values)
    normalized[nonzero] = values[nonzero]/maxima[nonzero, None]

    def best(profile):
        return round(float(offsets[np.argmax(profile)]), 6) if profile.max() > 0 else None

    midpoint = len(common)//2
    winners = offsets[values[nonzero].argmax(axis=1)]*1000
    return dict(common_beats=len(common), zero_rise_windows=int((~nonzero).sum()),
                mean_rise_best_offset_s=best(values.mean(axis=0)),
                equal_beat_best_offset_s=best(normalized.mean(axis=0)),
                first_half_best_offset_s=best(values[:midpoint].mean(axis=0)),
                second_half_best_offset_s=best(values[midpoint:].mean(axis=0)),
                per_beat_winning_offset_ms_quantiles=quantiles(winners),
                applied_shift_seconds=0, musical_alignment_certified=False)


def load_sources():
    inventory, assets = load_pool()
    raw = (ROOT/AUDIT_PATH).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == AUDIT_SHA, 'audio audit drift')
    audit = json.loads(raw)
    require(audit['structural_gate_passed'] and audit['fetch_lock_sha256'] == POOL_LOCK_SHA
            and [r['recording_id'] for r in audit['records']] == list(IDS), 'invalid audit population')
    return inventory, assets, {r['recording_id']: r for r in audit['records']}


def array_manifest(arrays):
    return {key: dict(shape=list(value.shape), dtype=value.dtype.name, sha256=array_hash(value))
            for key, value in sorted(arrays.items())}


def build(directory, output=None):
    inventory, assets, audited = load_sources()
    rows = []
    for source in inventory['records']:
        identity = source['recording_id']
        audio = assets[f'audio/{identity}.wav']
        annotation = assets[f'annotations/{identity}.beats']
        payload = verified_bytes(directory, audio)
        beats, positions = parse_beats(verified_bytes(directory, annotation))
        bpm = float(verified_bytes(directory, assets[f'annotations/{identity}.bpm']).decode('ascii'))
        require(bpm == source['published_global_bpm'], 'descriptive BPM drift')
        old = audited[identity]
        decoder = old['rust_decoder']
        check_decoder(decoder, audio, old['native_audio']['duration_seconds'])
        samples = round(decoder['duration_s']*22050)
        require(abs(samples/22050-decoder['duration_s']) <= 1e-12, 'nonintegral decoded sample count')
        # Centered Beat This mel: 1 + floor(decoded samples / 441). A future
        # capture must confirm this geometry, not crop/pad targets to fit it.
        frames = 1 + samples//441
        arrays = reference_arrays(beats, positions, frames, samples/22050)
        review = alignment_review(payload, beats)
        require(review['mean_rise_best_offset_s'] ==
                old['native_audio']['acoustic_diagnostics']['best_diagnostic_offset_seconds'],
                'earlier diagnostic failed replay')
        if output is not None:
            packet = output/f'{identity}.reference.npz'
            with packet.open('xb') as stream:
                np.savez(stream, **arrays)
            with np.load(packet, allow_pickle=False) as archive:
                require(array_manifest({k: archive[k] for k in archive.files}) == array_manifest(arrays),
                        'reference packet roundtrip differs')
        row = dict(recording_id=identity, style=source['style'], role='training_candidate_only',
                   leakage_group='brid-corpus-v1', audio_sha256=audio['sha256'],
                   beats_sha256=annotation['sha256'], frames=frames, duration_s=samples/22050,
                   reference_interval_s=[beats[0], beats[-1]], interval_end_exclusive=True,
                   reference_frames=int(arrays['valid'].sum()), tempo_cells=int(arrays['cell_valid'].sum()),
                   unsupported_frames=int((~arrays['valid']).sum()),
                   arrays=array_manifest(arrays), alignment_review=review)
        rows.append(row)
    require([r['recording_id'] for r in rows] == list(IDS), 'incomplete reference population')
    return dict(schema='rhythm-map.brid-reference.v1', fetch_lock_sha256=POOL_LOCK_SHA,
                source_sha256=source_hashes(),
                audio_audit_sha256=AUDIT_SHA, status='reference_packets_prepared_not_fit_admission',
                records=rows, recording_count=len(rows), leakage_group_count=1,
                frame_count=sum(r['frames'] for r in rows),
                reference_frames=sum(r['reference_frames'] for r in rows),
                tempo_cells=sum(r['tempo_cells'] for r in rows),
                supported_cell_seconds=round(sum(r['tempo_cells'] for r in rows)/FPS, 6),
                mean_offset_counts=dict(sorted(Counter(str(r['alignment_review']['mean_rise_best_offset_s'])
                                                       for r in rows).items())),
                semantic_review='source_labels_preserved_not_independently_certified',
                downbeat_positions_preserved=True, inferred_downbeat_phase=False,
                tempo_change_labels_created=False, timestamp_shifts_applied=0,
                encoder_calls=0, optimizer_steps=0, training_ready=False,
                holdout_access=False, production_change=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--check', type=Path)
    args = parser.parse_args()
    if args.output:
        output = args.output.resolve()
        require(not output.exists() and not output.is_relative_to(ROOT), 'fresh private output required')
        if os.name == 'nt':
            require(output.drive.lower() != os.environ.get('SystemDrive', 'C:').lower(), 'use data drive')
        output.mkdir(parents=True)
        report = build(args.audio_dir, output)
        with (output/'report.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        report = build(args.audio_dir)
        require(report == json.loads(args.check.read_text(encoding='utf-8')), 'reference replay differs')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
