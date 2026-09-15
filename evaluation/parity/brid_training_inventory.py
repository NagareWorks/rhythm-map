"""Offline BRID training-candidate inventory; no audio reads, fitting or acceptance."""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import re
import stat
import statistics
import zipfile


SOURCE = 'https://zenodo.org/records/14051323'
ANNOTATIONS = dict(size_bytes=84989, md5='678b2fa99c8d220cddd9f5e20d55d0c1',
                   sha256='605ad57dd9ad86fb3dbb40368a961c7d0db91595a3574aa15b8eb0bbe35ac6dd')
AUDIO = dict(size_bytes=944409073, md5='3514b53d66515181f95619adb71a59b4')
TAIL = dict(offset=943360497, size_bytes=1048576,
            sha256='5edb4eab075d8dc08ad1ce03030984a8b66f4c40295f5afb56b5f568e7c0aa00')
NAME = re.compile(r'\[(\d{4})\] M([234])-(\d{2})-(SA|PA|SE|MA)', re.ASCII)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_locked(path, lock):
    with path.open('rb') as stream:
        data = stream.read(lock['size_bytes'] + 1)
    require(len(data) == lock['size_bytes'], 'source size differs')
    require(hashlib.sha256(data).hexdigest() == lock['sha256'], 'source SHA-256 differs')
    if 'md5' in lock:
        require(hashlib.md5(data).hexdigest() == lock['md5'], 'published MD5 differs')
    return data


def parse_beats(payload):
    times, positions = [], []
    for line in payload.decode('ascii').splitlines():
        fields = line.split()
        require(len(fields) == 2, 'expected timestamp and beat position')
        timestamp = float(fields[0])
        require(math.isfinite(timestamp) and timestamp >= 0, 'invalid timestamp')
        require(not times or timestamp > times[-1], 'timestamps must increase')
        require(fields[1] in ('1', '2'), 'expected 2/4 beat position')
        times.append(timestamp)
        positions.append(int(fields[1]))
    require(len(times) >= 2, 'insufficient beats')
    return times, positions


def parse_annotations(data):
    """Preserve position discontinuities as findings, never repair the source."""
    members = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for entry in archive.infolist():
            require(entry.filename not in members, 'duplicate ZIP member')
            require(not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16),
                    'expected regular annotation member')
            match = re.fullmatch(r'Annotations/(beats|tempo)/(.+)\.(beats|bpm)', entry.filename)
            require(match is not None, 'unexpected annotation path')
            folder, stem, extension = match.groups()
            require((folder, extension) in (('beats', 'beats'), ('tempo', 'bpm')),
                    'annotation folder and extension differ')
            require(NAME.fullmatch(stem) is not None, 'unexpected recording name')
            require(0 < entry.file_size <= 65536, 'annotation member size out of bounds')
            members[entry.filename] = archive.read(entry)
    stems = sorted({Path(name).stem for name in members})
    require(stems, 'no annotation records')
    rows = []
    for stem in stems:
        beat_path = f'Annotations/beats/{stem}.beats'
        tempo_path = f'Annotations/tempo/{stem}.bpm'
        require(beat_path in members and tempo_path in members, 'missing annotation pair')
        times, positions = parse_beats(members[beat_path])
        fields = members[tempo_path].decode('ascii').split()
        require(len(fields) == 1, 'expected one global BPM')
        bpm = float(fields[0])
        require(math.isfinite(bpm) and bpm > 0, 'invalid global BPM')
        match = NAME.fullmatch(stem)
        recording_id, instruments, _, style = match.groups()
        intervals = [b - a for a, b in zip(times, times[1:])]
        rows.append(dict(
            recording_id=recording_id, name=stem, instrument_count=int(instruments), style=style,
            intended_role='training_candidate_only', leakage_group='brid-corpus-v1',
            beat_count=len(times), downbeat_count=positions.count(1),
            first_beat_seconds=times[0], last_beat_seconds=times[-1],
            annotated_span_seconds=round(times[-1] - times[0], 6),
            position_repeat_count=sum(a == b for a, b in zip(positions, positions[1:])),
            published_global_bpm=bpm,
            descriptive_median_interval_bpm=round(60 / statistics.median(intervals), 6),
            minimum_interval_seconds=round(min(intervals), 6),
            maximum_interval_seconds=round(max(intervals), 6),
            annotation_assets=[dict(member_path=path, size_bytes=len(members[path]),
                                    sha256=hashlib.sha256(members[path]).hexdigest())
                               for path in (beat_path, tempo_path)]))
    require(len({row['recording_id'] for row in rows}) == len(rows), 'duplicate recording ID')
    return rows


def attach_audio_directory(rows, tail):
    """Read central-directory metadata only, not compressed audio payloads."""
    with zipfile.ZipFile(io.BytesIO(tail)) as archive:
        entries = archive.infolist()
    names = [entry.filename for entry in entries]
    require(len(set(names)) == len(names), 'duplicate audio ZIP member')
    files = {entry.filename: entry for entry in entries if not entry.is_dir()}
    require(len(files) == 367, 'audio inventory differs')
    matched = set()
    for row in rows:
        path = f"Data/Acoustic Mixtures/{row['instrument_count']} Instruments/{row['name']}.wav"
        require(path in files, 'annotation lacks matching audio')
        entry = files[path]
        require(not stat.S_ISLNK(entry.external_attr >> 16), 'audio member is a link')
        require(entry.compress_type == zipfile.ZIP_DEFLATED and not entry.flag_bits & 1,
                'unsupported audio ZIP encoding')
        require(entry.file_size > 0 and entry.compress_size > 0, 'empty audio')
        row['audio_candidate'] = dict(member_path=path, size_bytes=entry.file_size,
                                      compressed_size_bytes=entry.compress_size,
                                      zip_crc32=f'{entry.CRC:08x}', sha256=None,
                                      payload_verified=False)
        matched.add(path)
    mixtures = {name for name in files if name.startswith('Data/Acoustic Mixtures/')}
    require(matched == mixtures, 'mixture and annotation inventory differ')
    return len(files)


def build_report(annotations, tail):
    rows = parse_annotations(annotations)
    require([row['recording_id'] for row in rows] == [f'{i:04d}' for i in range(1, 94)],
            'expected exact 93 mixture IDs')
    audio_count = attach_audio_directory(rows, tail)
    return dict(
        schema='rhythm-map.brid-training-inventory.v1',
        status='verified_annotation_and_directory_inventory_not_training_admission',
        inspected_on='2026-09-15', source_url=SOURCE,
        license='CC-BY-4.0', license_url='https://creativecommons.org/licenses/by/4.0/',
        attribution='Brazilian Rhythmic Instruments Dataset (BRID), Maia et al., 2018',
        annotation_archive=ANNOTATIONS, audio_archive=AUDIO, audio_tail=TAIL,
        source_audio_count=audio_count, beat_annotated_recording_count=len(rows),
        audio_without_released_beat_files=audio_count-len(rows),
        style_counts=dict(sorted(Counter(row['style'] for row in rows).items())),
        instrument_count_counts=dict(sorted(Counter(str(row['instrument_count']) for row in rows).items())),
        beat_count=sum(row['beat_count'] for row in rows),
        downbeat_count=sum(row['downbeat_count'] for row in rows),
        annotated_span_seconds=round(sum(row['annotated_span_seconds'] for row in rows), 6),
        candidate_audio_uncompressed_bytes=sum(row['audio_candidate']['size_bytes'] for row in rows),
        candidate_audio_compressed_bytes=sum(row['audio_candidate']['compressed_size_bytes'] for row in rows),
        position_repeat_recordings=[row['recording_id'] for row in rows if row['position_repeat_count']],
        encoder_exposure='not_named_in_beat_this_annotations_v1.0_not_a_no_overlap_guarantee',
        independent_work_count=None, conservative_leakage_group_count=1,
        audio_payloads_acquired=0, admitted_training_recordings=0,
        feature_access=False, optimizer_steps=0, project_holdout_access=False,
        public_default_changed=False, records=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--audio-tail', type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--output', type=Path)
    destination.add_argument('--check', type=Path)
    args = parser.parse_args()
    report = build_report(read_locked(args.annotations, ANNOTATIONS), read_locked(args.audio_tail, TAIL))
    if args.check:
        require(json.loads(args.check.read_text(encoding='utf-8')) == report, 'retained report differs')
    else:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()
