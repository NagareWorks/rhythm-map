"""Authored join failures and fixed identities; no music/model dependency."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.tempo_source_expansion import prepare as p
from brid_reference import reference_arrays, array_manifest


class FeatureInputTests(unittest.TestCase):
    def setUp(self):
        self.pcm = np.zeros(22050, np.float32)
        self.mel = np.zeros((51, 128), np.float32)
        self.arrays = reference_arrays([.1, .4, .8], [1, 2, 1], 51, 1.)
        self.row = dict(recording_id='0001', audio_sha256='a'*64,
                        frames=51, duration_s=1., arrays=array_manifest(self.arrays))
        self.front = dict(recording_id='0001', audio_sha256='a'*64, samples=22050, frames=51)

    def verify(self):
        p.verify_join(self.front, self.row, self.pcm, self.mel, self.arrays)

    def test_exact_native_join(self):
        self.verify()

    def test_shifted_reference_rejected(self):
        self.arrays['reference'][self.arrays['valid']] += .01
        with self.assertRaises(ValueError): self.verify()

    def test_mask_or_position_repair_rejected(self):
        original = copy.deepcopy(self.arrays)
        for key in ('valid', 'cell_valid', 'positions'):
            self.arrays = copy.deepcopy(original)
            self.arrays[key][0] = 1-self.arrays[key][0]
            with self.assertRaises(ValueError): self.verify()

    def test_crop_or_extra_frame_rejected(self):
        self.mel = self.mel[:-1]
        with self.assertRaises(ValueError): self.verify()

    def test_alternate_audio_or_sample_count_rejected(self):
        for key, value in [('audio_sha256', 'b'*64), ('samples', 22049), ('recording_id', '0002')]:
            old = self.front[key]
            self.front[key] = value
            with self.assertRaises(ValueError): self.verify()
            self.front[key] = old

    def test_nonfinite_and_wrong_dtype_rejected(self):
        self.mel[0, 0] = np.nan
        with self.assertRaises(ValueError): self.verify()
        self.mel = np.zeros((51, 128), np.float64)
        with self.assertRaises(ValueError): self.verify()

    def test_frozen_design_closure_still_matches(self):
        value = p.closure()
        self.assertEqual(value[p.REFERENCE], p.REFERENCE_SHA)
        self.assertIn('experiments/tempo_source_expansion/ALIGNMENT-v1.md', value)
        self.assertIn('experiments/tempo_source_expansion/capture.py', value)
        plan = json.loads((p.HERE/'plan-v1.json').read_bytes())
        self.assertEqual(plan['optimizer_steps'], 0)
        self.assertIn('all_93_feature_packet_capture_and_replay', plan['admission_pending'])


if __name__ == '__main__': unittest.main()
