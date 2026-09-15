"""Authored transport controls; no network or BRID corpus in CI."""
import io
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import urllib.error
import zipfile
import zlib

import brid_pool_acquire as acquisition


def fixture():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('authored.wav', b'authored payload')
    complete = stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(complete)) as archive:
        entry, = archive.infolist()
        end = archive.start_dir - 1
    return complete[:end+1], dict(recording_id='0001', start=0, end=end,
        member_path=entry.filename, flags=entry.flag_bits, compressed_size_bytes=entry.compress_size,
        size_bytes=entry.file_size, zip_crc32=f'{entry.CRC:08x}')


class BridPoolAcquisitionTests(unittest.TestCase):
    def test_tail_directory_offsets_address_all_authored_members_in_original_archive(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for index in range(1, 94):
                archive.writestr(f'sample{index:04d}.wav', f'authored {index}'.encode())
            archive.writestr('following-unselected.bin', b'not selected')
        original = stream.getvalue()
        with zipfile.ZipFile(io.BytesIO(original)) as archive:
            offset = archive.start_dir
            entries = archive.infolist()[:-1]
        tail = original[offset:]
        records = [dict(recording_id=f'{index:04d}', audio_candidate=dict(
            member_path=entry.filename, size_bytes=entry.file_size,
            compressed_size_bytes=entry.compress_size, zip_crc32=f'{entry.CRC:08x}'))
            for index, entry in enumerate(entries, 1)]
        with mock.patch.dict(acquisition.TAIL, offset=offset, sha256=hashlib.sha256(tail).hexdigest()):
            plans = acquisition.plan_ranges(tail, dict(records=records))
        self.assertEqual(len(plans), 93)
        for index, plan in enumerate(plans, 1):
            self.assertEqual(acquisition.extract_member(original[plan['start']:plan['end']+1], plan),
                             f'authored {index}'.encode())

    def test_extracts_exact_named_payload(self):
        data, plan = fixture()
        self.assertEqual(acquisition.extract_member(data, plan), b'authored payload')

    def test_local_header_name_crc_and_truncation_fail_closed(self):
        data, plan = fixture()
        for changed in (data[:-1], b'xxxx' + data[4:], data[:30] + b'X' + data[31:]):
            with self.subTest(data=changed), self.assertRaises((ValueError, zlib.error)):
                acquisition.extract_member(changed, plan)
        for change in (dict(size_bytes=1), dict(zip_crc32='00000000'), dict(flags=1),
                       dict(compressed_size_bytes=plan['compressed_size_bytes'] - 1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                acquisition.extract_member(data, dict(plan, **change))

    def test_data_descriptor_flag_still_requires_valid_deflate_and_crc(self):
        data, plan = fixture()
        changed = bytearray(data)
        struct.pack_into('<H', changed, 6, 8)
        struct.pack_into('<III', changed, 14, 0, 0, 0)
        self.assertEqual(acquisition.extract_member(changed, dict(plan, flags=8)), b'authored payload')
        with self.assertRaises(ValueError):
            acquisition.extract_member(changed, dict(plan, flags=8, zip_crc32='00000000'))

    def test_http_range_requires_exact_status_bounds_and_identity_encoding(self):
        _, plan = fixture()
        headers = {'Content-Range': f"bytes 0-{plan['end']}/{acquisition.AUDIO['size_bytes']}",
                   'Content-Length': str(plan['end']+1)}
        acquisition.check_response(206, headers, plan)
        for status, changed in ((200, headers), (206, dict(headers, **{'Content-Range': 'bytes 0-0/1'})),
                                (206, dict(headers, **{'Content-Length': '1'})),
                                (206, dict(headers, **{'Content-Encoding': 'gzip'}))):
            with self.subTest(status=status, headers=changed), self.assertRaises(ValueError):
                acquisition.check_response(status, changed, plan)

    def test_429_is_not_retried(self):
        _, plan = fixture()
        fetcher = acquisition.RangeFetcher()
        failure = urllib.error.HTTPError(acquisition.URL, 429, 'limited', {'Retry-After': '60'}, None)
        with mock.patch.object(fetcher.opener, 'open', side_effect=failure) as request:
            with self.assertRaises(urllib.error.HTTPError):
                fetcher.fetch(plan)
        self.assertEqual(request.call_count, 1)

    def test_start_times_are_spaced_without_concurrency(self):
        _, plan = fixture()
        fetcher = acquisition.RangeFetcher()
        fetcher.next_request = 12
        with mock.patch.object(acquisition.time, 'monotonic', return_value=11), \
             mock.patch.object(acquisition.time, 'sleep') as sleep, \
             mock.patch.object(fetcher.opener, 'open', side_effect=OSError('authored stop')):
            with self.assertRaises(OSError):
                fetcher.fetch(plan)
        sleep.assert_called_once_with(1)
        self.assertEqual(fetcher.next_request, 13)

    def test_install_reuses_nothing_silently_and_leaves_no_partial_after_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'audio').mkdir()
            acquisition.install(root, 'audio/0001.wav', b'first')
            self.assertEqual((root / 'audio/0001.wav').read_bytes(), b'first')
            self.assertFalse((root / 'audio/0001.wav.part').exists())
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                acquisition.install(root, 'audio/0001.wav', b'second')

    def test_partial_install_is_preserved_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'audio').mkdir()
            temporary = root / 'audio/0001.wav.part'
            temporary.write_bytes(b'previous interrupted write')
            with self.assertRaises(FileExistsError):
                acquisition.install(root, 'audio/0001.wav', b'new')
            self.assertEqual(temporary.read_bytes(), b'previous interrupted write')

    def test_existing_audio_must_pass_size_and_crc(self):
        _, plan = fixture()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'audio').mkdir()
            (root / 'audio/0001.wav').write_bytes(b'authored payload')
            self.assertEqual(acquisition.read_audio(root, plan), b'authored payload')
            (root / 'audio/0001.wav').write_bytes(b'Xuthored payload')
            with self.assertRaisesRegex(ValueError, 'CRC'):
                acquisition.read_audio(root, plan)

    def test_directory_pin_fails_before_parse(self):
        with self.assertRaisesRegex(ValueError, 'directory drift'):
            acquisition.plan_ranges(b'invented ZIP', {'records': []})

    def test_rejects_noncanonical_destination_before_writing(self):
        for relative in ('../escape', 'audio/../../escape.wav', '/audio/0001.wav',
                         'audio/1.wav', 'annotations/0001.csv', 'audio\\0001.wav'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                acquisition.install(Path('unused'), relative, b'x')

    def test_interrupted_pool_resumes_verified_audio_without_refetch_or_lock(self):
        _, plan = fixture()
        metadata = {'records': [{'recording_id': '0001', 'annotation_assets': []}]}
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w'):
            pass
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fetcher = mock.Mock()
            fetcher.fetch.return_value = b'authored payload'
            for attempt in range(2):
                with self.assertRaisesRegex(ValueError, 'incomplete pool'):
                    acquisition.acquire(root, [], metadata, [plan], stream.getvalue(), fetcher)
                self.assertFalse((root / 'lock.json').exists())
                self.assertEqual((root / 'audio/0001.wav').read_bytes(), b'authored payload')
                self.assertEqual(fetcher.fetch.call_count, 1)

    def test_source_failure_does_not_create_final_or_partial_audio(self):
        _, plan = fixture()
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w'):
            pass
        metadata = {'records': [{'recording_id': '0001', 'annotation_assets': []}]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fetcher = mock.Mock()
            fetcher.fetch.side_effect = OSError('authored transport failure')
            with self.assertRaises(OSError):
                acquisition.acquire(root, [], metadata, [plan], stream.getvalue(), fetcher)
            self.assertEqual(list((root / 'audio').iterdir()), [])
            self.assertFalse((root / 'lock.json').exists())

    def test_offline_verification_cannot_create_directory_or_fetch_missing_audio(self):
        _, plan = fixture()
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w'):
            pass
        metadata = {'records': [{'recording_id': '0001', 'annotation_assets': []}]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'pool'
            fetcher = mock.Mock()
            with self.assertRaisesRegex(ValueError, 'existing pool'):
                acquisition.acquire(root, [], metadata, [plan], stream.getvalue(), fetcher, True)
            self.assertFalse(root.exists())
            root.mkdir()
            (root / 'audio').mkdir()
            (root / 'annotations').mkdir()
            (root / 'lock.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'missing audio'):
                acquisition.acquire(root, [], metadata, [plan], stream.getvalue(), fetcher, True)
            fetcher.fetch.assert_not_called()
            self.assertEqual((root / 'lock.json').read_text(), '{}')


if __name__ == '__main__':
    unittest.main()
