"""Deterministic metadata-review sample, not audio selection or data admission."""
import argparse
import hashlib
import json
from pathlib import Path

from rhythm_support_inventory import LICENSES, SCREEN, load_report, require


DOMAIN = 'rhythm-support-metadata-review-v1\n'
PER_LICENSE = 4
ROOT = Path(__file__).resolve().parent.parent
EXPOSURE_PATH = ROOT / 'datasets/fsld-tempo-v1.json'


def rank(clip_id):
    return hashlib.sha256((DOMAIN + clip_id).encode('ascii')).hexdigest()


def select_review_sample(rows, exposed_ids):
    """Freeze four rows per screened license before reading their descriptions.

    Case-folded uploader strings only prevent obvious reuse in this tiny sample;
    they do not establish creator or recording independence. Never fill a quota
    with NC/unknown licenses, known exposed IDs or repeated uploader hints.
    """
    selected = []
    seen_uploaders = set()
    for license_id in sorted(SCREEN):
        eligible = [clip_id for clip_id, row in rows.items()
                    if LICENSES[row['license']] == license_id and clip_id not in exposed_ids]
        chosen = 0
        for clip_id in sorted(eligible, key=lambda value: (rank(value), int(value))):
            row = rows[clip_id]
            uploader_key = row['uploader'].casefold()
            if uploader_key in seen_uploaders:
                continue
            seen_uploaders.add(uploader_key)
            selected.append(dict(clip_id=clip_id, license_id=license_id, rank_sha256=rank(clip_id),
                                 source_url=f'https://freesound.org/s/{clip_id}/'))
            chosen += 1
            if chosen == PER_LICENSE:
                break
        require(chosen == PER_LICENSE, 'insufficient distinct uploader hints for fixed review sample')
    return selected


def exposure_ids(payload):
    manifest = json.loads(payload)
    require(manifest['id'] == 'freesound-loop-tempo-v1', 'wrong exposure manifest')
    ids = []
    for asset in manifest['assets']:
        if asset['role'] == 'audio':
            path = Path(asset['path'])
            clip_id = path.stem
            require(asset['path'] == f'audio/{clip_id}.wav' and clip_id.isascii()
                    and clip_id.isdecimal() and clip_id == str(int(clip_id)) and int(clip_id) > 0,
                    'noncanonical exposed source ID')
            ids.append(clip_id)
    require(len(ids) == 15 and len(set(ids)) == 15, 'expected fifteen distinct exposed FSLD IDs')
    return set(ids)


def build_report(rows, prior, exposure_payload):
    exposed = exposure_ids(exposure_payload)
    matches = sorted(exposed.intersection(rows), key=int)
    screened_matches = [clip_id for clip_id in matches if LICENSES[rows[clip_id]['license']] in SCREEN]
    return dict(
        schema='rhythm-map.rhythm-support-provenance.v1',
        status='metadata_review_only_not_admitted',
        input_archive_locks=prior['archive_locks'],
        exposure_manifest=dict(path='evaluation/datasets/fsld-tempo-v1.json',
                               sha256=hashlib.sha256(exposure_payload).hexdigest(),
                               exposed_audio_id_count=len(exposed)),
        exposure_matches=dict(all_development_ids=matches, screened_ids=screened_matches,
                              meaning='known_project_regression_exposure_not_encoder_training_overlap'),
        selection_rule=dict(domain=DOMAIN, per_license=PER_LICENSE,
                            license_order=sorted(SCREEN), ordering='SHA-256 then numeric clip ID',
                            uploader_exclusion='casefold hint across both quotas, not identity proof',
                            exposed_id_exclusion=True, tags_or_descriptions_used=False,
                            model_scores_used=False, sample_role='source_metadata_review_only'),
        review_sample=select_review_sample(rows, exposed),
        population_rhythm_coverage=None, encoder_recording_overlap='unknown',
        independent_groups_admitted=0, listener_labels_collected=0,
        audio_selection_made=False, audio_acquired=False, feature_access=False,
        training_authorized=False, project_holdout_access=False, production_change=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--document', type=Path, required=True)
    paths = parser.add_mutually_exclusive_group(required=True)
    paths.add_argument('--output', type=Path)
    paths.add_argument('--check', type=Path)
    args = parser.parse_args()
    rows, prior = load_report(args.metadata, args.document)
    retained = json.loads((ROOT / 'datasets/rhythm-support-inventory-v1.json').read_text(encoding='utf-8'))
    require(prior == retained, 'verified inventory differs')
    report = build_report(rows, prior, EXPOSURE_PATH.read_bytes())
    if args.check:
        require(json.loads(args.check.read_text(encoding='utf-8')) == report, 'retained report differs')
    else:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()
