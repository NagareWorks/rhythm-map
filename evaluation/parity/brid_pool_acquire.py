"""Polite, resumable source-only acquisition of the frozen BRID mixture pool.

One bounded HTTP range per missing WAV, no burst concurrency or automatic
retries. Reuse the frozen local ZIP directory and annotation archive. This is
research intake tooling, not a shipping backend or training admission gate.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib

from brid_audio_audit import INVENTORY_SHA, ROOT, load_pins, safe_file
from brid_training_inventory import ANNOTATIONS, AUDIO, TAIL, read_locked, require


URL = 'https://zenodo.org/api/records/14051323/files/audio.zip/content'
MIN_REQUEST_INTERVAL_SECONDS = 2.0
MAX_RANGE_BYTES = 16 * 1024 * 1024


def plan_ranges(tail, inventory):
    require(hashlib.sha256(tail).hexdigest() == TAIL['sha256'], 'audio directory drift')
    with zipfile.ZipFile(io.BytesIO(tail)) as archive:
        entries = sorted(archive.infolist(), key=lambda entry: entry.header_offset)
    indices = {entry.filename: index for index, entry in enumerate(entries)}
    require(len(indices) == len(entries), 'duplicate source ZIP member')
    plans = []
    for row in inventory['records']:
        source = row['audio_candidate']
        index = indices[source['member_path']]
        require(index + 1 < len(entries), 'missing following local header')
        entry, following = entries[index:index + 2]
        start = entry.header_offset + TAIL['offset']
        end = following.header_offset + TAIL['offset'] - 1
        require(entry.compress_type == zipfile.ZIP_DEFLATED and not entry.flag_bits & 1,
                'unsupported source ZIP encoding')
        require(entry.file_size == source['size_bytes'] and
                entry.compress_size == source['compressed_size_bytes'] and
                f'{entry.CRC:08x}' == source['zip_crc32'], 'directory metadata differs')
        require(0 <= start <= end < TAIL['offset'] and
                entry.compress_size + 30 <= end-start+1 <= MAX_RANGE_BYTES,
                'source range outside bounded payload region')
        plans.append(dict(recording_id=row['recording_id'], start=start, end=end,
                          member_path=entry.filename, flags=entry.flag_bits,
                          compressed_size_bytes=entry.compress_size,
                          size_bytes=entry.file_size, zip_crc32=f'{entry.CRC:08x}'))
    require([plan['recording_id'] for plan in plans] == [f'{i:04d}' for i in range(1, 94)],
            'expected all 93 preregistered mixtures')
    return plans


def extract_member(data, plan):
    require(len(data) == plan['end']-plan['start']+1 and len(data) >= 30, 'range size differs')
    header = struct.unpack_from('<IHHHHHIIIHH', data)
    signature, _, flags, method, _, _, crc, compressed, size, name_size, extra_size = header
    require(signature == 0x04034B50 and flags == plan['flags'] and method == 8,
            'local ZIP header differs')
    require(data[30:30+name_size] == plan['member_path'].encode('ascii'), 'local member name differs')
    if not flags & 8:
        require((crc, compressed, size) == (int(plan['zip_crc32'], 16),
                plan['compressed_size_bytes'], plan['size_bytes']), 'local size/CRC differs')
    payload_start = 30 + name_size + extra_size
    payload_end = payload_start + plan['compressed_size_bytes']
    require(payload_end <= len(data), 'truncated compressed member')
    decoder = zlib.decompressobj(-15)
    audio = decoder.decompress(data[payload_start:payload_end], plan['size_bytes'] + 1)
    require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
            'incomplete or excess deflate payload')
    verify_audio(audio, plan)
    return audio


def verify_audio(data, plan):
    require(len(data) == plan['size_bytes'], 'audio size differs')
    require(f'{zlib.crc32(data):08x}' == plan['zip_crc32'], 'audio CRC differs')


def check_response(status, headers, plan):
    require(status == 206, 'source did not return partial content')
    require(headers.get('Content-Range') == f"bytes {plan['start']}-{plan['end']}/{AUDIO['size_bytes']}",
            'source Content-Range differs')
    require(headers.get('Content-Encoding', 'identity') == 'identity', 'encoded range response')
    length = headers.get('Content-Length')
    if length is not None:
        require(int(length) == plan['end']-plan['start']+1, 'source Content-Length differs')


class RangeFetcher:
    def __init__(self):
        self.next_request = 0.0
        self.opener = urllib.request.build_opener()

    def fetch(self, plan):
        delay = self.next_request - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.next_request = time.monotonic() + MIN_REQUEST_INTERVAL_SECONDS
        request = urllib.request.Request(URL, headers={
            'Range': f"bytes={plan['start']}-{plan['end']}", 'Accept-Encoding': 'identity',
            'User-Agent': 'rhythm-map-source-intake/1'})
        # A 429 propagates immediately. Respect Retry-After before a later run;
        # no identity rotation, retry loop or partial lock promotion here.
        with self.opener.open(request, timeout=60) as response:
            require(response.geturl() == URL, 'unexpected source redirect')
            check_response(response.status, response.headers, plan)
            data = response.read(plan['end']-plan['start']+2)
        return extract_member(data, plan)


def read_audio(directory, plan):
    path = safe_file(directory, f"audio/{plan['recording_id']}.wav")
    with path.open('rb') as stream:
        data = stream.read(plan['size_bytes'] + 1)
    verify_audio(data, plan)
    return data


def install(directory, relative, data):
    # Output root and fixed parent folders have been checked before any access.
    require(re.fullmatch(r'(audio/[0-9]{4}\.wav|annotations/[0-9]{4}\.(beats|bpm))', relative),
            'unexpected destination path')
    parent = directory / relative.split('/')[0]
    require(parent.is_dir() and not parent.is_symlink(), 'unsafe destination parent')
    path = directory / relative
    require(not path.exists() and not path.is_symlink(), 'refusing destination overwrite')
    temporary = path.with_suffix(path.suffix + '.part')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    # No concurrent writers are supported; final files are never accepted from
    # temporary names. A leftover .part causes an explicit failure, not deletion.
    require(not path.exists() and not path.is_symlink(), 'destination appeared during write')
    temporary.rename(path)


def acquire(directory, reuse_directories, inventory, plans, annotations, fetcher, verify_only=False):
    require(not directory.is_symlink(), 'linked output root')
    if verify_only:
        require(directory.is_dir(), 'verification requires existing pool')
        safe_file(directory, 'lock.json')
    else:
        directory.mkdir(exist_ok=True)
    for name in ('audio', 'annotations'):
        child = directory / name
        require(not child.is_symlink(), 'linked output child')
        if verify_only:
            require(child.is_dir(), 'missing pool directory')
        else:
            child.mkdir(exist_ok=True)
    require(verify_only or not (directory / 'lock.json').exists(),
            'pool already locked; use --verify-only or dataset-fetch')
    rows = {row['recording_id']: row for row in inventory['records']}
    assets = []
    with zipfile.ZipFile(io.BytesIO(annotations)) as archive:
        for index, plan in enumerate(plans, 1):
            recording_id = plan['recording_id']
            relative = f'audio/{recording_id}.wav'
            roots = [directory] if verify_only else [directory, *reuse_directories]
            source = next((root for root in roots
                           if (root / relative).exists() or (root / relative).is_symlink()), None)
            if source is not None:
                data = read_audio(source, plan)
                state = 'reused'
            else:
                require(not verify_only, 'missing audio in verification-only mode')
                data = fetcher.fetch(plan)
                state = 'downloaded'
            if source != directory:
                install(directory, relative, data)
            assets.append(dict(path=relative, url=URL, size_bytes=len(data),
                               sha256=hashlib.sha256(data).hexdigest(), role='audio',
                               zip_member=dict(archive_size_bytes=AUDIO['size_bytes'],
                                               member_path=plan['member_path'])))
            for original in rows[recording_id]['annotation_assets']:
                data = archive.read(original['member_path'])
                require(len(data) == original['size_bytes'] and
                        hashlib.sha256(data).hexdigest() == original['sha256'], 'annotation drift')
                relative = f"annotations/{recording_id}.{original['member_path'].rsplit('.', 1)[1]}"
                if (directory / relative).exists() or (directory / relative).is_symlink():
                    path = safe_file(directory, relative)
                    with path.open('rb') as stream:
                        require(stream.read(len(data) + 1) == data, 'existing annotation drift')
                else:
                    require(not verify_only, 'missing annotation in verification-only mode')
                    install(directory, relative, data)
                assets.append(dict(path=relative,
                    url='https://zenodo.org/api/records/14051323/files/annotations.zip/content',
                    size_bytes=len(data), sha256=original['sha256'], role='annotation_source',
                    zip_member=dict(archive_size_bytes=ANNOTATIONS['size_bytes'],
                                    member_path=original['member_path'])))
            print(json.dumps(dict(recording_id=recording_id, completed=index, total=len(plans),
                                  audio=state)), flush=True)
    require(len(assets) == 279, 'incomplete pool cannot become a lock')
    _, smoke_lock = load_pins()
    # Continuity with already acquired, cryptographically pinned smoke bytes.
    known = {asset['path']: asset for asset in smoke_lock['assets']}
    for asset in assets:
        if asset['path'] in known:
            require(asset['sha256'] == known[asset['path']]['sha256'], 'smoke byte identity drift')
    lock = dict(smoke_lock, id='brid-training-mixtures-v1',
                version='14051323-all-acoustic-mixtures', assets=assets)
    if verify_only:
        require(json.loads(safe_file(directory, 'lock.json').read_text(encoding='utf-8')) == lock,
                'existing lock differs from verified source bytes')
    else:
        with (directory / 'lock.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(lock, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print('Complete source lock verified; no training admission or model inference.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-tail', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse', type=Path, action='append', default=[])
    parser.add_argument('--verify-only', action='store_true',
                        help='Verify the existing pool and lock offline, with no writes or downloads.')
    args = parser.parse_args()
    load_pins()
    smoke_bytes = (ROOT / 'evaluation/datasets/brid-audio-smoke-v1.json').read_bytes()
    require(hashlib.sha256(smoke_bytes).hexdigest() ==
            '96d9a70f055cfa9573eaab315b80b2bc0a8f14fa54556809912b2ec35aedbe19',
            'structural smoke evidence drift')
    require(json.loads(smoke_bytes)['structural_gate_passed'], 'structural smoke failed')
    raw = (ROOT / 'evaluation/datasets/brid-training-candidate-v1.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == INVENTORY_SHA, 'inventory drift')
    inventory = json.loads(raw)
    tail = read_locked(args.audio_tail, TAIL)
    annotations = read_locked(args.annotations, ANNOTATIONS)
    plans = plan_ranges(tail, inventory)
    try:
        acquire(args.output, args.reuse, inventory, plans, annotations, RangeFetcher(), args.verify_only)
    except urllib.error.HTTPError as error:
        print(json.dumps(dict(status='source_http_failure', code=error.code,
                              retry_after=error.headers.get('Retry-After'),
                              partial_files_retained=True, training_admission=False)), flush=True)
        raise


if __name__ == '__main__':
    main()
