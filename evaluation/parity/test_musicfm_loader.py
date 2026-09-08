"""Model-free fail-closed controls; real torch controls use the private probe."""
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import musicfm_loader as loader


class Tensor:
    def __init__(self, shape=(2,), dtype='torch.float32', device='cpu', finite=True):
        self.shape, self.dtype = shape, dtype
        self.device = types.SimpleNamespace(type=device)
        self.layout, self.is_quantized, self.finite = 'strided', False, finite


TORCH = types.SimpleNamespace(Tensor=Tensor, strided='strided',
                             isfinite=lambda value: types.SimpleNamespace(all=lambda: value.finite))


class MusicFMLoaderControls(unittest.TestCase):
    def test_previous_screen_and_single_candidate_remain_pinned(self):
        screen = loader.screen()
        self.assertEqual(screen['reserved_probe']['layer_ix'], 7)
        self.assertEqual(screen['reserved_probe']['checkpoint'], 'pretrained_fma.pt')
        self.assertFalse(screen['model_payload_downloaded'])
        self.assertFalse(screen['production_pack_approved'])
        self.assertEqual(len(loader.SOURCE), 4)

    def test_modified_screen_fails_before_parsing(self):
        with patch.object(loader.Path, 'read_bytes', return_value=b'{}'), \
                patch.object(loader, 'strict_json') as parse:
            with self.assertRaisesRegex(ValueError, 'screen changed'):
                loader.screen()
            parse.assert_not_called()

    def test_json_duplicates_nonfinite_and_nonobject_rejected(self):
        for data in (b'{"x":1,"x":2}', b'{"nested":{"x":0,"x":1}}',
                     b'{"x":NaN}', b'{"x":Infinity}', b'[]', b'null', b'{'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                loader.strict_json(data)
        self.assertEqual(loader.strict_json(b'{"x":[1,2]}'), {'x': [1, 2]})

    def test_same_verified_handle_is_rewound_and_restricted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'authored.pt'
            path.write_bytes(b'authored bytes')
            observed = []

            def deserialize(stream, **kwargs):
                observed.append((stream.tell(), stream.read(), kwargs))
                return {'authored': True}

            spec = dict(size_bytes=14, sha256=loader.digest(b'authored bytes'))
            self.assertEqual(loader.restricted_load(path, spec, deserialize), {'authored': True})
            self.assertEqual(observed, [(0, b'authored bytes', dict(map_location='cpu', weights_only=True))])

    def test_wrong_size_hash_missing_and_directory_never_deserialize(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'authored.pt'
            path.write_bytes(b'abc')
            cases = [(path, 2, loader.digest(b'ab')), (path, 3, '0' * 64),
                     (path.with_name('missing'), 3, '0' * 64), (Path(directory), 3, '0' * 64)]
            for file, size, sha in cases:
                call = Mock()
                with self.subTest(file=file, size=size), self.assertRaises((ValueError, OSError)):
                    loader.restricted_load(file, dict(size_bytes=size, sha256=sha), call)
                call.assert_not_called()

    def test_file_identity_rejects_bool_negative_oversize_and_invalid_hash(self):
        for size, sha in ((True, '0' * 64), (0, '0' * 64), (-1, '0' * 64),
                          (1_400_000_001, '0' * 64), (1, 'A' * 64), (1, '0' * 63)):
            with self.subTest(size=size, sha=sha), self.assertRaises(ValueError):
                with loader.verified_file('must-not-open', size, sha):
                    self.fail('invalid identity accepted')

    def test_link_flag_is_rejected_without_opening(self):
        with patch.object(loader.Path, 'lstat', return_value=types.SimpleNamespace(st_mode=0o100600, st_size=3)), \
                patch.object(loader.Path, 'is_symlink', return_value=True), \
                patch.object(loader.Path, 'open') as opened:
            with self.assertRaisesRegex(ValueError, 'non-link'):
                with loader.verified_file('link', 3, loader.digest(b'abc')):
                    self.fail('link accepted')
            opened.assert_not_called()

    def test_stream_growth_and_truncation_are_not_accepted(self):
        for content in (b'abcd', b'ab'):
            with patch.object(loader.Path, 'lstat', return_value=types.SimpleNamespace(st_mode=0o100600, st_size=3)), \
                    patch.object(loader.Path, 'is_symlink', return_value=False), \
                    patch.object(loader.Path, 'open', return_value=io.BytesIO(content)):
                with self.assertRaises(ValueError):
                    with loader.verified_file('authored', 3, loader.digest(b'abc')):
                        self.fail('changed stream accepted')

    def test_deserializer_exception_propagates_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'authored.pt'
            path.write_bytes(b'abc')
            call = Mock(side_effect=RuntimeError('restricted rejection'))
            with self.assertRaisesRegex(RuntimeError, 'restricted rejection'):
                loader.restricted_load(path, dict(size_bytes=3, sha256=loader.digest(b'abc')), call)
            self.assertEqual(call.call_count, 1)
            self.assertTrue(call.call_args.args[0].closed)

    def test_small_file_budget_stops_before_open(self):
        with patch.object(loader, 'verified_file') as opened:
            with self.assertRaisesRegex(ValueError, 'small-file budget'):
                loader.small_file('unused', dict(size_bytes=65537, sha256='0' * 64))
            opened.assert_not_called()

    def test_config_dimensions_types_and_remote_code_are_not_silently_fixed(self):
        config = dict(model_type='wav2vec2-conformer', hidden_size=1024,
                      num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096,
                      position_embeddings_type='rotary', rotary_embedding_base=10000,
                      conv_depthwise_kernel_size=31, layerdrop=0.0, layer_norm_eps=1e-5)
        self.assertEqual(loader.validate_config(config.copy()), config)
        for key, value in [('hidden_size', 512), ('num_hidden_layers', 12),
                           ('num_attention_heads', 16.0), ('layerdrop', False),
                           ('position_embeddings_type', 'relative'), ('auto_map', {})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                loader.validate_config(dict(config, **{key: value}))

    def test_geometry_has_ceil_subsampling_and_fixed_sample_origin(self):
        for n in (1025, 1199, 1200, 1919, 1920, 24000, 24001, 48000):
            row = loader.token_geometry(n)
            mel = n // 240
            # Independent two convolution output-length calculations.
            once = (mel + 2 - 3) // 2 + 1
            twice = (once + 2 - 3) // 2 + 1
            self.assertEqual(row['tokens'], twice)
            self.assertEqual(row['origin_samples'], 0)
            self.assertEqual(row['stride_samples'], 960)
            self.assertLess(row['last_center_samples'], n)
        self.assertEqual(loader.token_geometry(1200)['tokens'], 2)

    def test_no_automatic_short_padding_long_clip_or_sample_rate_assumption(self):
        for n in (0, -1, 1024, 48001, True, 1200.0, '1200'):
            with self.subTest(n=n), self.assertRaises(ValueError):
                loader.token_geometry(n)

    def test_network_guard_denies_dns_and_restores_functions(self):
        previous = socket.getaddrinfo
        with self.assertRaisesRegex(ValueError, 'network attempt detected'):
            with loader.no_network():
                socket.getaddrinfo('invalid.example', 80)
        self.assertIs(socket.getaddrinfo, previous)

    def test_swallowed_network_error_still_rejects_result(self):
        with self.assertRaisesRegex(ValueError, 'network attempt detected'):
            with loader.no_network():
                try:
                    socket.create_connection(('invalid.example', 80))
                except RuntimeError:
                    pass

    def test_network_guard_denies_existing_socket_connect_and_sendto(self):
        for method in ('connect', 'connect_ex', 'sendto'):
            with socket.socket() as sock:
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'network attempt detected'):
                    with loader.no_network():
                        getattr(sock, method)(b'x', ('127.0.0.1', 9)) if method == 'sendto' else \
                            getattr(sock, method)(('127.0.0.1', 9))

    def test_network_guard_preserves_nonnetwork_errors(self):
        with self.assertRaisesRegex(RuntimeError, 'ordinary failure'):
            with loader.no_network():
                raise RuntimeError('ordinary failure')

    def test_ipv6_probe_is_suppressed_and_restored_without_allowing_bind(self):
        previous = socket.has_ipv6
        with loader.no_network():
            self.assertFalse(socket.has_ipv6)
        self.assertEqual(socket.has_ipv6, previous)
        with socket.socket() as sock, self.assertRaisesRegex(ValueError, 'network attempt detected'):
            with loader.no_network():
                sock.bind(('127.0.0.1', 0))

    def test_occupied_module_namespace_and_missing_sources_rejected(self):
        with patch.dict(sys.modules, {'musicfm': types.ModuleType('musicfm')}):
            with self.assertRaisesRegex(ValueError, 'occupied'):
                with loader.source_module({}):
                    self.fail('namespace hijacked')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            with loader.source_module({}):
                self.fail('incomplete source imported')

    def test_source_module_cleanup_after_import_failure(self):
        authored = {name: b'raise RuntimeError("authored import failure")' for name in loader.SOURCE}
        with self.assertRaisesRegex(RuntimeError, 'authored import failure'):
            with loader.source_module(authored):
                self.fail('failed import yielded')
        self.assertFalse(any(n == 'musicfm' or n.startswith('musicfm.') for n in sys.modules))

    def test_changed_runtime_is_rejected(self):
        with patch.object(loader.importlib.metadata, 'version', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'versions differ'):
                loader.runtime_versions({'torch': '2.8.0+cpu'})

    def test_runtime_lock_matches_all_wheels_and_refuses_drift(self):
        lock = loader.runtime_lock()
        self.assertEqual(len(lock['packages']), 28)
        self.assertEqual(lock['expected_missing_wheel_notice_files'], ['tokenizers'])
        self.assertEqual(next(p['version'] for p in lock['packages'] if p['name'] == 'torch'), '2.8.0+cpu')
        for p in lock['packages']:
            self.assertTrue(p['filename'].endswith('.whl'))
            self.assertGreater(p['size'], 0)
            self.assertEqual(len(p['sha256']), 64)
        with patch.object(loader.Path, 'read_bytes', return_value=b'{}'):
            with self.assertRaisesRegex(ValueError, 'runtime lock changed'):
                loader.runtime_lock()

    def test_bad_config_fails_before_source_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad-config.json'
            path.write_bytes(b'{}')
            with patch.object(loader, 'source_module') as imported:
                with self.assertRaisesRegex(ValueError, 'size mismatch'):
                    loader.construct_meta('unused', path, 'unused')
                imported.assert_not_called()

    def test_bad_stats_and_source_fail_before_import(self):
        # Exercise later rejection boundaries with authored objects, not downloads.
        config = dict(model_type='wav2vec2-conformer', hidden_size=1024,
                      num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096,
                      position_embeddings_type='rotary', rotary_embedding_base=10000,
                      conv_depthwise_kernel_size=31, layerdrop=0.0, layer_norm_eps=1e-5)
        spec = next(a for a in loader.screen()['artifacts'] if a['path'] == 'fma_stats.json')
        with patch.object(loader, 'small_file', side_effect=[json.dumps(config), '{}']):
            with self.assertRaisesRegex(ValueError, 'statistics mismatch'):
                loader.verified_inputs('unused', 'config', 'stats')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / next(iter(loader.SOURCE))
            source.parent.mkdir(parents=True)
            source.write_bytes(b'changed source')
            stats = {k: spec[k] for k in ('melspec_2048_mean', 'melspec_2048_std')}
            with patch.object(loader, 'small_file', side_effect=[json.dumps(config), json.dumps(stats)]):
                with self.assertRaisesRegex(ValueError, 'source identity mismatch'):
                    loader.verified_inputs(directory, 'config', 'stats')

    def test_recorded_prerequisite_is_not_pretrained_fidelity(self):
        report = loader.strict_json((loader.HERE / 'musicfm-loader-prerequisite-v1.json').read_bytes())
        self.assertTrue(report['complete'])
        self.assertEqual(report['decision'], 'loader_prerequisites_only_not_pretrained_fidelity')
        for key in ('pretrained_checkpoint_acquired', 'pretrained_checkpoint_loaded',
                    'pretrained_musicfm_inference', 'music_access', 'training', 'production_changed'):
            self.assertFalse(report[key])
        self.assertEqual(report['state_key_count'], 465)
        self.assertEqual(report['meta_parameter_elements'], 328932480)
        self.assertEqual(report['process_memory_limit_bytes'], 4 * 1024 ** 3)
        self.assertEqual(len(report['authored_frontend_geometry']), 7)
        for row in report['authored_frontend_geometry']:
            for key, value in loader.token_geometry(row['samples']).items():
                self.assertEqual(row[key], value)
        self.assertEqual(len(report['dependency_notices']), 28)
        self.assertEqual([p['name'] for p in report['dependency_notices'] if not p['notices']], ['tokenizers'])
        for name, sha in report['source_sha256'].items():
            raw = (loader.HERE / name).read_bytes()
            self.assertEqual(loader.digest(raw), sha, name)

    def test_tensor_schema_prefix_keys_shapes_dtype_and_finiteness(self):
        expected = {'weight': dict(shape=[2], dtype='torch.float32')}
        tensor = Tensor()
        mapped = loader.checked_state({'state_dict': {'model.weight': tensor}}, expected, TORCH)
        self.assertIs(mapped['weight'], tensor)
        cases = [{}, {'state_dict': []}, {'state_dict': {}},
                 {'state_dict': {'other.weight': Tensor()}},
                 {'state_dict': {'model.unknown': Tensor()}},
                 {'state_dict': {'model.weight': Tensor(shape=(1, 2))}},
                 {'state_dict': {'model.weight': Tensor(dtype='torch.float16')}},
                 {'state_dict': {'model.weight': Tensor(device='meta')}},
                 {'state_dict': {'model.weight': Tensor(finite=False)}},
                 {'state_dict': {'model.weight': [0, 0]}},
                 {'state_dict': {'model.weight': Tensor(), 'model.extra': Tensor()}}]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                loader.checked_state(case, expected, TORCH)

    def test_tensor_subclasses_quantized_and_sparse_layout_rejected(self):
        class Derived(Tensor):
            pass
        sparse, quantized = Tensor(), Tensor()
        sparse.layout, quantized.is_quantized = 'sparse_coo', True
        expected = {'weight': dict(shape=[2], dtype='torch.float32')}
        for value in (Derived(), sparse, quantized):
            with self.assertRaises(ValueError):
                loader.checked_state({'state_dict': {'model.weight': value}}, expected, TORCH)


if __name__ == '__main__':
    unittest.main()
