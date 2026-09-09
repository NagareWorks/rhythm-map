"""Authored metadata controls; CI needs neither network nor the upstream ZIPs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import rhythm_support_inventory as inventory


def clip(license_url='http://creativecommons.org/licenses/by/3.0/', uploader='author'):
    return dict(title='Authored test title', description='No audio in this fixture',
                tags=['music'], license=license_url, uploader=uploader)


def rows():
    return {'10': clip(), '2': clip(uploader='AUTHOR'),
            '3': clip('http://creativecommons.org/publicdomain/zero/1.0/'),
            '4': clip('http://creativecommons.org/licenses/by-nc/3.0/'),
            '5': clip('http://creativecommons.org/licenses/sampling+/1.0/')}


class MetadataInventoryTests(unittest.TestCase):
    def test_parse_valid_exact_schema(self):
        self.assertEqual(inventory.parse_dev(json.dumps(rows()).encode()), rows())

    def test_duplicate_json_keys_at_both_levels_are_rejected(self):
        row = json.dumps(clip())
        for payload in ('{"2":' + row + ',"2":' + row + '}',
                        '{"2":' + row.replace('{', '{"title":"duplicate",', 1) + '}'):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, 'duplicate'):
                inventory.parse_dev(payload)

    def test_nonstandard_constants_and_empty_maps_are_rejected(self):
        for payload in ('{}', '[]', 'null', '{"2":NaN}', '{"2":Infinity}'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                inventory.parse_dev(payload)

    def test_clip_ids_are_canonical_positive_ascii_decimals(self):
        for key in ('0', '02', '-2', '2.0', '../2', '２', ''):
            with self.subTest(key=key), self.assertRaises(ValueError):
                inventory.parse_dev(json.dumps({key: clip()}))

    def test_schema_drift_is_not_silently_accepted(self):
        for change in ('extra', 'missing', 'not_object'):
            value = clip()
            if change == 'extra':
                value['duration'] = 12
            elif change == 'missing':
                del value['description']
            else:
                value = []
            with self.subTest(change=change), self.assertRaises(ValueError):
                inventory.parse_dev(json.dumps({'2': value}))

    def test_attribution_and_tag_types_are_checked(self):
        for field, value in [('title', ''), ('uploader', '  '), ('description', None),
                             ('tags', 'music'), ('tags', [1]), ('license', False)]:
            row = clip()
            row[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                inventory.parse_dev(json.dumps({'2': row}))

    def test_unknown_license_fails_closed_instead_of_prefix_matching(self):
        for value in ('http://creativecommons.org/licenses/by/3.0/extra',
                      'https://creativecommons.org/licenses/by/4.0/', '', 'CC BY'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                inventory.parse_dev(json.dumps({'2': clip(value)}))

    def test_only_initial_screen_survives_in_numeric_id_order(self):
        records = list(inventory.candidate_records(rows()))
        self.assertEqual([row['clip_id'] for row in records], ['2', '3', '10'])
        self.assertEqual({row['rights']['audio_license_id'] for row in records}, inventory.SCREEN)

    def test_screen_is_not_permission_or_a_split_or_a_label(self):
        for record in inventory.candidate_records(rows()):
            self.assertEqual(record['status'], 'license_screen_only_not_admitted')
            self.assertIsNone(record['pcm_sha256'])
            for key in ('training_permission', 'evaluation_permission', 'redistribution_permission'):
                self.assertEqual(record['rights'][key], 'unreviewed')
            self.assertIsNone(record['rights']['annotation_license_url'])
            for key in ('work_id', 'recording_id', 'independent_group_id'):
                self.assertIsNone(record['provenance'][key])
            self.assertEqual(record['provenance']['creator_ids'], [])
            self.assertEqual(record['provenance']['encoder_overlap'], 'unreviewed')
            self.assertEqual(record['split'], 'unassigned')
            self.assertEqual(record['windows'], [])
            self.assertEqual(record['label_status'], 'not_collected')

    def test_original_attribution_is_retained_but_tags_do_not_become_truth(self):
        source = {'2': clip(uploader='Unicode \u97f3\u4e50')}
        first = list(inventory.candidate_records(source))
        self.assertEqual(first[0]['rights']['attribution']['uploader'], 'Unicode \u97f3\u4e50')
        source['2']['tags'] = ['silence', 'noise', 'non-music']
        source['2']['description'] = 'Untrusted source text is not instructions or labels.'
        self.assertEqual(first, list(inventory.candidate_records(source)))

    def test_uploader_spelling_is_not_normalized_to_independence(self):
        result = inventory.uploader_counts([clip(), clip(), clip(uploader='AUTHOR')])
        self.assertEqual(result['exact_strings'], 2)
        self.assertEqual(result['casefolded_strings'], 1)
        self.assertEqual(result['max_clips_per_exact_string'], 2)
        self.assertEqual(result['single_clip_exact_strings'], 1)
        self.assertEqual(result['identity_resolution'], 'unreviewed_not_independent_groups')

    def test_jsonl_identity_is_deterministic_and_matches_retained_bytes(self):
        source = rows()
        encoded = b''.join(inventory.encoded_records(source))
        reversed_source = dict(reversed(list(source.items())))
        self.assertEqual(encoded, b''.join(inventory.encoded_records(reversed_source)))
        report = inventory.build_report(source)
        identity = report['candidate_jsonl']
        self.assertEqual(identity['record_count'], len(encoded.splitlines()))
        self.assertEqual(identity['size_bytes'], len(encoded))
        self.assertEqual(identity['sha256'], hashlib.sha256(encoded).hexdigest())
        self.assertEqual([json.loads(line)['clip_id'] for line in encoded.splitlines()], ['2', '3', '10'])

    def test_zero_screen_candidates_does_not_mean_zero_possible_groups(self):
        report = inventory.build_report({'2': clip('http://creativecommons.org/licenses/by-nc/3.0/')})
        self.assertEqual(report['screened_clip_count'], 0)
        self.assertEqual(report['screened_uploader_hints']['max_clips_per_exact_string'], 0)
        self.assertIsNone(report['independent_source_group_count'])
        self.assertEqual(report['rights_cleared_clip_count'], 0)
        self.assertEqual(report['collected_region_labels'], 0)
        for key in ('audio_selection_made', 'audio_acquired', 'feature_access', 'new_inference',
                    'project_holdout_access', 'training_authorized', 'production_change'):
            self.assertFalse(report[key])

    def test_load_checks_document_before_parsing_development_rows(self):
        with patch.object(inventory, 'read_verified_member', side_effect=[b'{}', ValueError('bad doc')]), \
                patch.object(inventory, 'parse_dev') as parse, self.assertRaisesRegex(ValueError, 'bad doc'):
            inventory.load_report(Path('metadata'), Path('document'))
        parse.assert_not_called()

    def run_cli(self, arguments):
        with patch('sys.argv', ['inventory', '--metadata', 'unused', '--document', 'unused',
                                *arguments]), \
                patch.object(inventory, 'load_report', return_value=(rows(), inventory.build_report(rows()))):
            inventory.main()

    def test_cli_does_not_overwrite_existing_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'summary.json'
            target.write_bytes(b'owned existing data')
            with self.assertRaises(FileExistsError):
                self.run_cli(['--output', str(target)])
            self.assertEqual(target.read_bytes(), b'owned existing data')

    def test_cli_does_not_overwrite_existing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'summary.json'
            report.write_text(json.dumps(inventory.build_report(rows())), encoding='utf-8')
            target = Path(directory) / 'rows.jsonl'
            target.write_bytes(b'owned existing data')
            with self.assertRaises(FileExistsError):
                self.run_cli(['--check', str(report), '--rows', str(target)])
            self.assertEqual(target.read_bytes(), b'owned existing data')

    def test_cli_mismatch_prevents_row_export(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'summary.json'
            report.write_text('{}', encoding='utf-8')
            target = Path(directory) / 'rows.jsonl'
            with self.assertRaisesRegex(ValueError, 'report differs'):
                self.run_cli(['--check', str(report), '--rows', str(target)])
            self.assertFalse(target.exists())

    def test_frozen_report_is_a_screen_and_keeps_v1_failure_history(self):
        root = Path(__file__).resolve().parent.parent
        report = json.loads((root / 'datasets/rhythm-support-inventory-v1.json').read_text())
        old = json.loads((root / 'datasets/rhythm-support-worksheet-v1.json').read_text())
        self.assertEqual(report['archive_locks'], [inventory.METADATA, inventory.DOCUMENT])
        self.assertEqual(report['metadata_clip_count'], 40966)
        self.assertEqual(report['screened_clip_count'], 34976)
        self.assertEqual(report['excluded_clip_count'], 5990)
        self.assertEqual(sum(report['license_counts'].values()), report['metadata_clip_count'])
        self.assertEqual(report['candidate_jsonl']['record_count'], report['screened_clip_count'])
        self.assertIsNone(report['independent_source_group_count'])
        self.assertFalse(report['eval_metadata_inspected'])
        self.assertEqual(old['sources'][0]['metadata_status'], 'not_acquired')
        self.assertIsNone(old['sources'][0]['metadata_sha256'])


class VerifiedArchiveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'authored.zip'
        self.payload = json.dumps({'2': clip()}).encode()
        self.member = 'FSD50K.metadata/dev_clips_info_FSD50K.json'

    def archive(self, duplicate=False, symlink=False):
        with zipfile.ZipFile(self.path, 'w') as zipped:
            # Invalid JSON in an out-of-scope member must never be opened.
            zipped.writestr('FSD50K.metadata/eval_clips_info_FSD50K.json', b'NOT JSON')
            info = zipfile.ZipInfo(self.member)
            if symlink:
                info.create_system = 3
                info.external_attr = 0o120777 << 16
            zipped.writestr(info, self.payload)
            if duplicate:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    zipped.writestr(self.member, self.payload)
        data = self.path.read_bytes()
        return dict(size_bytes=len(data), md5=hashlib.md5(data).hexdigest(),
                    sha256=hashlib.sha256(data).hexdigest(), member=self.member,
                    member_size_bytes=len(self.payload), member_sha256=hashlib.sha256(self.payload).hexdigest())

    def test_only_allowed_member_is_opened(self):
        lock = self.archive()
        opened = []
        original = zipfile.ZipFile.open

        def record(zipped, entry, *args, **kwargs):
            opened.append(entry.filename)
            return original(zipped, entry, *args, **kwargs)

        with patch.object(zipfile.ZipFile, 'open', record):
            self.assertEqual(inventory.read_verified_member(self.path, lock), self.payload)
        self.assertEqual(opened, [self.member])

    def test_each_size_and_hash_is_checked(self):
        lock = self.archive()
        for key in ('size_bytes', 'md5', 'sha256', 'member_size_bytes', 'member_sha256'):
            bad = copy.deepcopy(lock)
            bad[key] = bad[key] + 1 if isinstance(bad[key], int) else '0' * len(bad[key])
            with self.subTest(key=key), self.assertRaises(ValueError):
                inventory.read_verified_member(self.path, bad)

    def test_missing_duplicate_and_symlink_members_are_rejected(self):
        lock = self.archive()
        lock['member'] = '../not-the-allowed-member'
        with self.assertRaises(ValueError):
            inventory.read_verified_member(self.path, lock)
        for options in ({'duplicate': True}, {'symlink': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                inventory.read_verified_member(self.path, self.archive(**options))

    def test_wrong_archive_is_rejected_before_opening_zip(self):
        lock = self.archive()
        lock['sha256'] = '0' * 64
        with patch.object(zipfile, 'ZipFile') as reader, self.assertRaises(ValueError):
            inventory.read_verified_member(self.path, lock)
        reader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
