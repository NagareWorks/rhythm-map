"""Offline, hash-pinned FSD50K development metadata screen; never fetch audio."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import stat
import zipfile


SOURCE = 'https://zenodo.org/records/4060432'
ATTRIBUTION = ('FSD50K: Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font '
               'and Xavier Serra, Music Technology Group, Universitat Pompeu Fabra')
METADATA = dict(
    file='FSD50K.metadata.zip', size_bytes=6700838,
    md5='b9ea0c829a411c1d42adb9da539ed237',
    sha256='9a738e032546f9a2c6e3d04566928d04a65fb79b422cc9d78bb781723537bd19',
    member='FSD50K.metadata/dev_clips_info_FSD50K.json', member_size_bytes=22494560,
    member_sha256='b31a2eb130aea6d545d4b48511b24dfd0120420567a8d1066c52d54968bb3735')
DOCUMENT = dict(
    file='FSD50K.doc.zip', size_bytes=6984,
    md5='3516162b82dc2945d3e7feba0904e800',
    sha256='a5c314ad2bf93d01b5921aa4270c70c90e28ac9a283ba9f08167490e728deb85',
    member='FSD50K.doc/LICENSE-DATASET', member_size_bytes=1599,
    member_sha256='ebee1f320405cc09abfcc3c77e14fbc3adb9734d53f2ef5274d35eddcbb476f1')
LICENSES = {
    'http://creativecommons.org/publicdomain/zero/1.0/': 'CC0-1.0',
    'http://creativecommons.org/licenses/by/3.0/': 'CC-BY-3.0',
    'http://creativecommons.org/licenses/by-nc/3.0/': 'CC-BY-NC-3.0',
    'http://creativecommons.org/licenses/sampling+/1.0/': 'Sampling+-1.0',
}
SCREEN = {'CC0-1.0', 'CC-BY-3.0'}
FIELDS = {'title', 'description', 'tags', 'license', 'uploader'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_verified_member(path, lock):
    """Read one bounded member, never extract or inspect other member payloads."""
    with path.open('rb') as stream:
        archive = stream.read(lock['size_bytes'] + 1)
    require(len(archive) == lock['size_bytes'], 'archive size differs')
    require(hashlib.md5(archive).hexdigest() == lock['md5'], 'published MD5 differs')
    require(hashlib.sha256(archive).hexdigest() == lock['sha256'], 'archive SHA-256 differs')
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        entries = [entry for entry in zipped.infolist() if entry.filename == lock['member']]
        require(len(entries) == 1, 'expected exactly one allowed member')
        entry = entries[0]
        require(not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16),
                'member is not a regular payload')
        require(entry.file_size == lock['member_size_bytes'], 'member size differs')
        with zipped.open(entry) as stream:
            payload = stream.read(lock['member_size_bytes'] + 1)
    require(len(payload) == lock['member_size_bytes'], 'decoded member size differs')
    require(hashlib.sha256(payload).hexdigest() == lock['member_sha256'],
            'member SHA-256 differs')
    return payload


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f'non-JSON numeric constant: {value}')


def parse_dev(payload):
    rows = json.loads(payload, object_pairs_hook=unique_object, parse_constant=reject_constant)
    require(isinstance(rows, dict) and bool(rows), 'expected nonempty development map')
    for clip_id, row in rows.items():
        require(clip_id.isascii() and clip_id.isdecimal() and clip_id != '0'
                and clip_id == str(int(clip_id)), 'noncanonical clip id')
        require(isinstance(row, dict) and set(row) == FIELDS, 'unexpected clip schema')
        for field in FIELDS - {'tags'}:
            require(isinstance(row[field], str), f'{field} must be text')
        require(all(row[field].strip() for field in ('title', 'uploader', 'license')),
                'empty attribution or license')
        require(isinstance(row['tags'], list)
                and all(isinstance(tag, str) for tag in row['tags']), 'invalid tags')
        require(row['license'] in LICENSES, 'unrecognized license; review before screening')
    return rows


def candidate_records(rows):
    """Preliminary per-clip facts, not source selection, rights clearance or labels."""
    for clip_id in sorted(rows, key=int):
        row = rows[clip_id]
        family = LICENSES[row['license']]
        if family not in SCREEN:
            continue
        yield dict(
            source_id='fsd50k-dev-4060432', source_version='1.0', clip_id=clip_id,
            source_url=f'https://freesound.org/s/{clip_id}/',
            metadata_sha256=METADATA['sha256'], dev_member_sha256=METADATA['member_sha256'],
            status='license_screen_only_not_admitted', pcm_sha256=None,
            rights=dict(
                audio_license_url=row['license'], audio_license_id=family,
                annotation_license_url=None,
                dataset_license_url='https://creativecommons.org/licenses/by/4.0/',
                attribution=dict(title=row['title'], uploader=row['uploader']),
                training_permission='unreviewed', evaluation_permission='unreviewed',
                redistribution_permission='unreviewed',
                dataset_conditions='commercial_contact_note_unresolved',
                evidence_urls=[SOURCE]),
            provenance=dict(
                uploader_hint=row['uploader'], work_id=None, recording_id=None,
                creator_ids=[], derived_asset_ids=[], synthesis_generator_id=None,
                encoder_overlap='unreviewed', independent_group_id=None),
            split='unassigned', windows=[], label_status='not_collected')


def encoded_records(rows):
    for record in candidate_records(rows):
        yield (json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(',', ':'),
                          allow_nan=False) + '\n').encode('utf-8')


def uploader_counts(rows):
    counts = Counter(row['uploader'] for row in rows)
    return dict(exact_strings=len(counts), casefolded_strings=len({key.casefold() for key in counts}),
                max_clips_per_exact_string=max(counts.values(), default=0),
                single_clip_exact_strings=sum(value == 1 for value in counts.values()),
                identity_resolution='unreviewed_not_independent_groups')


def build_report(rows):
    counts = Counter(LICENSES[row['license']] for row in rows.values())
    screened = [row for row in rows.values() if LICENSES[row['license']] in SCREEN]
    digest = hashlib.sha256()
    size = 0
    for line in encoded_records(rows):
        digest.update(line)
        size += len(line)
    return dict(
        schema='rhythm-map.rhythm-support-inventory.v1',
        status='verified_metadata_screen_not_rights_or_independence_clearance',
        source_url=SOURCE, source_version='1.0', attribution=ATTRIBUTION,
        archive_locks=[METADATA, DOCUMENT],
        member_payloads_inspected=[METADATA['member'], DOCUMENT['member']],
        archive_contains_uninspected_eval_metadata=True, eval_metadata_inspected=False,
        metadata_clip_count=len(rows), license_counts=dict(sorted(counts.items())),
        initial_license_screen=sorted(SCREEN), screened_clip_count=len(screened),
        excluded_clip_count=len(rows) - len(screened),
        all_uploader_hints=uploader_counts(rows.values()),
        screened_uploader_hints=uploader_counts(screened),
        candidate_jsonl=dict(record_count=len(screened), size_bytes=size, sha256=digest.hexdigest(),
                             serialization='ASCII sorted-key compact JSON plus LF; numeric clip-id order'),
        dataset_license_url='https://creativecommons.org/licenses/by/4.0/',
        dataset_conditions_review='commercial_contact_note_unresolved',
        annotation_license_review='unreviewed', encoder_overlap_review='unreviewed',
        independent_source_group_count=None, rights_cleared_clip_count=0,
        admitted_independent_group_count=0, collected_region_labels=0,
        audio_selection_made=False, audio_acquired=False, feature_access=False,
        new_inference=False, project_holdout_access=False, training_authorized=False,
        production_change=False)


def load_report(metadata, document):
    # Verify both complete archives and the only two allowed members before interpreting data.
    payload = read_verified_member(metadata, METADATA)
    read_verified_member(document, DOCUMENT)
    rows = parse_dev(payload)
    return rows, build_report(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--document', type=Path, required=True)
    paths = parser.add_mutually_exclusive_group(required=True)
    paths.add_argument('--output', type=Path)
    paths.add_argument('--check', type=Path)
    parser.add_argument('--rows', type=Path, help='optional new local JSONL; no audio URLs or labels')
    args = parser.parse_args()
    rows, report = load_report(args.metadata, args.document)
    if args.check:
        require(json.loads(args.check.read_text(encoding='utf-8')) == report, 'retained report differs')
    else:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    if args.rows:
        with args.rows.open('xb') as stream:
            for line in encoded_records(rows):
                stream.write(line)


if __name__ == '__main__':
    main()
