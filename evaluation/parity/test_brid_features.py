"""Feature archive contract with authored boundary/replay faults."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.tempo_source_expansion.audit import verify_features
from experiments.tempo_source_expansion.prepare import ROOT, IDS, REFERENCE, array_manifest, closure, sha
from evaluation.parity import prehead_capture as geometry


class FeatureReplayTests(unittest.TestCase):
    def test_retained_full_population_and_reference_identities(self):
        report = json.loads((ROOT/'evaluation/datasets/brid-features-v1.json').read_bytes())
        refs = json.loads((ROOT/REFERENCE).read_bytes())
        self.assertEqual(report['registration']['source_sha256'], closure())
        self.assertEqual(report['audit_source_sha256'], sha(ROOT/'experiments/tempo_source_expansion/audit.py'))
        self.assertEqual([r['recording_id'] for r in report['records']], IDS)
        self.assertEqual(sum(r['frames'] for r in report['records']), 133247)
        self.assertEqual(sum(r['forward_calls'] for r in report['records']), 387)
        for row, ref in zip(report['records'], refs['records']):
            self.assertEqual({k: v for k, v in row['input_arrays'].items() if k != 'mel'}, ref['arrays'])
            self.assertEqual(row['frames'], ref['frames'])
            self.assertEqual(row['audio_sha256'], ref['audio_sha256'])
        self.assertFalse(report['training_ready'] or report['holdout_access'] or report['production_change'])
        self.assertEqual(report['optimizer_steps'], 0)
        self.assertEqual(report['label_shifts'], 0)
        self.assertEqual(report['source_groups'], 1)
        self.assertTrue(report['admission_pending'])

    def fixture(self, frames=1489):
        chunks = [dict(hidden=np.zeros((r['size'], 512), np.float32),
                       beat=np.zeros(r['size'], np.float32), downbeat=np.zeros(r['size'], np.float32))
                  for r in geometry.layout(frames)]
        arrays = geometry.stitch(frames, chunks)
        row = dict(frames=frames, chunks=len(chunks), forward_calls=3*len(chunks),
                   arrays=array_manifest(arrays), untapped_heads_bit_exact=True,
                   hidden_replay_bit_exact=True, roundtrip_bit_exact=True, max_head_reconstruction_error=0.)
        return row, arrays

    def test_overlap_and_short_extents(self):
        for frames in (17, 1488, 1489, 3001):
            row, arrays = self.fixture(frames)
            verify_features(row, arrays, frames)

    def test_later_owner_rejected_even_with_rehashed_manifest(self):
        row, arrays = self.fixture()
        arrays['owner'][500] = 1
        row['arrays'] = array_manifest(arrays)
        with self.assertRaises(ValueError): verify_features(row, arrays, 1489)

    def test_hidden_mutation_and_nonfinite_rejected(self):
        row, arrays = self.fixture()
        arrays['hidden'][0, 0] = 1
        with self.assertRaises(ValueError): verify_features(row, arrays, 1489)
        arrays['hidden'][0, 0] = np.nan
        row['arrays'] = array_manifest(arrays)
        with self.assertRaises(ValueError): verify_features(row, arrays, 1489)

    def test_geometry_and_missing_replay_rejected(self):
        original, arrays = self.fixture()
        for key, value in [('frames', 1488), ('forward_calls', 2), ('hidden_replay_bit_exact', False),
                           ('untapped_heads_bit_exact', False), ('roundtrip_bit_exact', False)]:
            row = copy.deepcopy(original)
            row[key] = value
            with self.assertRaises(ValueError): verify_features(row, arrays, 1489)


if __name__ == '__main__': unittest.main()
