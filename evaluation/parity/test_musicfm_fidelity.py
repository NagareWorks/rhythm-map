"""Model-free protocol/refusal checks. Neural execution uses the private probe."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import musicfm_fidelity as fidelity
import musicfm_loader as loader
import musicfm_checkpoint_inventory as inventory
import musicfm_fidelity_compat as compat


class MusicFMFidelityControls(unittest.TestCase):
    def test_protocol_pins_one_artifact_and_no_music_or_training(self):
        plan = fidelity.protocol()
        previous = next(a for a in loader.screen()['artifacts'] if a['path'] == 'pretrained_fma.pt')
        self.assertEqual(plan['checkpoint']['sha256'], previous['sha256'])
        self.assertEqual(plan['checkpoint']['size_bytes'], previous['size_bytes'])
        self.assertEqual(plan['layer_ix'], 7)
        self.assertEqual(plan['dtype'], 'float32')
        for key in ('is_flash', 'music_access', 'holdout_access', 'training', 'production_change',
                    'unsafe_pickle_fallback', 'checkpoint_or_layer_or_precision_fallback'):
            self.assertFalse(plan[key])
        self.assertTrue(plan['stop_on_failure'])

    def test_protocol_changes_stop_before_helper_or_runtime_access(self):
        with patch.object(Path, 'read_bytes', return_value=b'{}'), patch.object(loader, 'runtime_lock') as runtime:
            with self.assertRaisesRegex(ValueError, 'protocol changed'):
                fidelity.protocol()
            runtime.assert_not_called()

    def test_changed_helper_is_rejected(self):
        raw = (loader.HERE / 'musicfm-fidelity-lock-v1.json').read_bytes()
        with patch.object(Path, 'read_bytes', side_effect=[raw, b'changed helper']):
            with self.assertRaisesRegex(ValueError, 'helper changed'):
                fidelity.protocol()

    def test_no_metric_or_clip_selection_and_exact_comparison(self):
        plan = fidelity.protocol()
        self.assertEqual([(c['samples'], c['kind']) for c in plan['cases']],
                         [(1025, 'silence'), (1199, 'impulse'), (1200, 'tone'), (1919, 'tone'),
                          (1920, 'impulse'), (24001, 'tone'), (48000, 'silence')])
        self.assertEqual(plan['comparison']['atol'], 0)
        self.assertEqual(plan['comparison']['rtol'], 0)
        self.assertEqual(plan['comparison']['passes_per_case'], 3)
        self.assertEqual(plan['num_executed_blocks'], 12)
        self.assertEqual(plan['num_hidden_states'], 13)
        for case in plan['cases']:
            self.assertGreater(loader.token_geometry(case['samples'])['tokens'], 0)

    def test_resource_boundaries_and_no_bool_as_byte_count(self):
        limits = fidelity.protocol()['resources']
        good = dict(available_physical_bytes=limits['min_available_physical_bytes'],
                    available_commit_bytes=limits['min_available_commit_bytes'],
                    free_disk_bytes=limits['min_free_disk_bytes'])
        fidelity.validate_resources(good, limits)
        for key in good:
            for value in (good[key] - 1, -1, True, None, 'enough'):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'insufficient resource'):
                    fidelity.validate_resources(dict(good, **{key: value}), limits)

    def test_cpu_memory_and_time_limits_are_not_user_modes(self):
        limits = fidelity.protocol()['resources']
        self.assertEqual(limits['torch_threads'], 2)
        self.assertEqual(limits['process_memory_limit_bytes'], 4 * 1024 ** 3)
        self.assertEqual(limits['timeout_seconds'], 300)
        self.assertEqual(limits['max_tensor_hash_chunk_bytes'], 1024 ** 2)

    def test_offline_environment_overrides_inherited_online_settings(self):
        previous = dict(os.environ)
        with patch.dict(os.environ, {'PYTHONPATH': 'untrusted', 'HF_HOME': 'shared', 'HF_HUB_OFFLINE': '0'}):
            env = fidelity.offline_environment(Path('/private/authored'))
        self.assertEqual(dict(os.environ), previous)
        self.assertNotIn('PYTHONPATH', env)
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'HF_HUB_DISABLE_XET'):
            self.assertEqual(env[key], '1')
        for key in ('USE_TF', 'USE_FLAX'):
            self.assertEqual(env[key], '0')
        self.assertEqual(env['HF_HOME'], str(Path('/private/authored/hf-cache')))

    def test_nonwindows_output_is_rejected(self):
        with patch.object(fidelity.sys, 'platform', 'linux'):
            with self.assertRaisesRegex(ValueError, 'Windows system drive'):
                fidelity.private_output('unused')

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(fidelity.sys, 'platform', 'win32'), \
                patch.dict(os.environ, {'SystemDrive': 'Z:'}):
            with self.assertRaisesRegex(ValueError, 'fresh'):
                fidelity.private_output(directory)

    def test_invalid_pcm_length_and_recipe_stop_without_torch(self):
        for case in [dict(samples=1024, kind='tone'), dict(samples=48001, kind='tone'),
                     dict(samples=1200, kind='music-file'), dict(samples=True, kind='silence')]:
            with self.assertRaises(ValueError):
                fidelity.pcm(case, None)

    def test_hash_loop_does_not_make_whole_checkpoint_byte_copies(self):
        # The real Torch authored controls test tensor byte identity separately.
        source = Path(fidelity.__file__).read_text(encoding='utf-8')
        self.assertIn('memoryview(', source)
        self.assertIn('1024 * 1024', source)
        self.assertNotIn('.tobytes()', source)

    def test_inventory_differences_preserve_missing_and_extra_denominators(self):
        one = dict(shape=[2], dtype='torch.float32')
        two = dict(shape=[3], dtype='torch.float32')
        result = inventory.differences({'same': one, 'changed': two, 'extra': one},
                                       {'same': one, 'changed': one, 'missing': one})
        self.assertEqual(result['missing'], ['missing'])
        self.assertEqual(result['unexpected'], ['extra'])
        self.assertEqual(result['exact_entries'], 1)
        self.assertEqual(result['actual_count'], 3)
        self.assertEqual(result['expected_count'], 3)
        self.assertEqual(result['mismatches'], [dict(key='changed', actual=two, expected=one)])

    def test_strict_failure_is_retained_without_any_pretrained_cases(self):
        report = loader.strict_json((loader.HERE / 'musicfm-fidelity-v1.json').read_bytes())
        self.assertFalse(report['complete'])
        self.assertTrue(report['checkpoint_bytes_verified'])
        self.assertFalse(report['checkpoint_loaded'])
        self.assertFalse(report['pretrained_inference_started'])
        self.assertEqual(report['decision'], 'stopped_at_restricted_checkpoint_load')
        self.assertEqual(report['cases'], [])
        self.assertEqual(report['protocol_sha256'], fidelity.LOCK_SHA)
        self.assertEqual(report['source_sha256'], loader.digest(Path(fidelity.__file__).read_bytes()))

    def test_compatibility_scope_has_exact_named_twenty_entries(self):
        plan = compat.compatibility()
        self.assertEqual(len(plan['aliases']), 2)
        self.assertEqual(len(plan['counter_keys']), 18)
        self.assertEqual(len(set(plan['counter_keys'])), 18)
        self.assertTrue(all(k.endswith('.num_batches_tracked') for k in plan['counter_keys']))
        self.assertEqual(plan['unchanged_non_counter_tensors'], 447)
        self.assertFalse(plan['learned_tensor_casts'])
        self.assertFalse(plan['learned_tensor_reshapes'])
        self.assertTrue(plan['fidelity_comparison_unchanged'])

    def test_changed_compatibility_lock_refused(self):
        with patch.object(Path, 'read_bytes', return_value=b'{}'):
            with self.assertRaisesRegex(ValueError, 'compatibility protocol changed'):
                compat.compatibility()

    def test_counter_semantics_reject_fractional_nonfinite_overflow_and_boolean(self):
        for value in (0.0, 17.0, float(2 ** 24)):
            self.assertEqual(compat.counter_integer(value), int(value))
        for value in (-1.0, 0.5, float('nan'), float('inf'), float(2 ** 63), True, 17, '17'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compat.counter_integer(value)

    def test_unknown_compatibility_root_prefix_and_alias_collision_refused(self):
        plan = compat.compatibility()
        for checkpoint in ({}, {'state_dict': []}, {'state_dict': {'wrong.prefix': object()}}):
            with self.assertRaises(ValueError):
                compat.normalize(checkpoint, {}, plan, None)
        old, new = next(iter(plan['aliases'].items()))
        with self.assertRaisesRegex(ValueError, 'both present'):
            compat.normalize({'state_dict': {'model.' + old: object(), 'model.' + new: object()}},
                             {new: {}}, plan, None)

    def test_compatibility_is_bound_to_observed_inventory_not_generic_repairs(self):
        report = loader.strict_json((loader.HERE / 'musicfm-checkpoint-inventory-v1.json').read_bytes())
        plan = compat.compatibility()
        comparison = report['comparison']
        self.assertEqual(comparison['actual_count'], 465)
        self.assertEqual(comparison['expected_count'], 465)
        self.assertEqual(comparison['exact_entries'], 445)
        self.assertEqual(set(comparison['unexpected']), set(plan['aliases']))
        self.assertEqual(set(comparison['missing']), set(plan['aliases'].values()))
        self.assertEqual({row['key'] for row in comparison['mismatches']}, set(plan['counter_keys']))
        self.assertFalse(report['state_assigned'])
        self.assertFalse(report['pretrained_inference'])
        self.assertTrue(report['all_state_tensors_finite'])
        self.assertEqual(report['source_sha256'], loader.digest(Path(inventory.__file__).read_bytes()))

    def test_compatible_authored_pass_retains_all_cases_and_no_music_claim(self):
        report = loader.strict_json((loader.HERE / 'musicfm-fidelity-compat-v1.json').read_bytes())
        self.assertTrue(report['complete'])
        self.assertEqual(report['decision'], 'compatibility_and_authored_fidelity_pass_not_musical_accuracy')
        self.assertTrue(report['checkpoint_loaded'])
        self.assertEqual(report['initial_model_state_sha256'], report['final_model_state_sha256'])
        self.assertEqual(report['compatibility_sha256'], compat.COMPAT_SHA)
        self.assertEqual(report['source_sha256'], loader.digest(Path(compat.__file__).read_bytes()))
        self.assertEqual(report['original_fidelity_source_sha256'], loader.digest(Path(fidelity.__file__).read_bytes()))
        self.assertEqual(len(report['cases']), 7)
        for case, frozen in zip(report['cases'], fidelity.protocol()['cases']):
            for key, value in frozen.items():
                self.assertEqual(case[key], value)
            for key, value in loader.token_geometry(case['samples']).items():
                self.assertEqual(case[key], value)
            self.assertTrue(case['reference_hook_repeat_exact'])
            self.assertTrue(case['state_and_input_unchanged'])
        for key in ('music_access', 'holdout_access', 'training', 'production_change', 'commercial_distribution_approved'):
            self.assertFalse(report[key])


if __name__ == '__main__':
    unittest.main()
