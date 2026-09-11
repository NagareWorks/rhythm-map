"""Authored pre-fit contracts: no musical inference or optimizer steps."""
import copy
import inspect
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.coupled_clock import run as old
from experiments.coupled_clock.measurement import KEYS, clock_fields, independent_fields, measure
from experiments.phase_sync.model import PhaseSyncReadout
from experiments.phase_sync_fit import run
from experiments.phase_sync_fit.register import make_plan, source_hashes

torch.set_num_threads(1)


def passing_gate():
    rows = []
    for role in ('development', 'diagnostic'):
        row = dict(role=role, work=role)
        for kind in run.ALL_KINDS:
            row[kind] = {key: (.5 if kind in ('audio', 'audio_common') else 1.) for key in KEYS}
        rows.append(row)
    summary = {row['role']: {name: row[name].copy() for name in run.ALL_KINDS} for row in rows}
    fits = [dict(epochs=20, updates=100, budget_exhausted=False, elapsed_s=30.) for _ in range(2)]
    return rows, summary, fits


class MusicalFitContracts(unittest.TestCase):
    def test_optimizer_loop_changes_only_model_constructor(self):
        expected = inspect.getsource(old.fit).replace('CoupledClockReadout()', 'PhaseSyncReadout()').strip()
        self.assertEqual(inspect.getsource(run.fit).strip(), expected)
        self.assertIs(run.development_loss, old.development_loss)
        self.assertIs(run.load_records, old.load_records)
        self.assertEqual((run.SEED, run.EPOCHS, run.SECONDS, run.BATCH), (142, 20, 1800, 4))

    def test_plan_reproduces_without_music_or_weight_loading(self):
        a, b = make_plan(), make_plan()
        self.assertEqual(a, b)
        self.assertEqual(a['source_sha256'], source_hashes())
        self.assertEqual(len(a['population']), 40)
        self.assertEqual([sum(r['role'] == role for r in a['population'])
                          for role in ('fit', 'development', 'diagnostic')], [20, 5, 15])
        self.assertFalse(a['training'])
        self.assertFalse(a['holdout_access'])
        torch.manual_seed(142)
        self.assertEqual(a['initial_state_sha256'], old.state_hash(PhaseSyncReadout()))
        self.assertEqual(a['temporal_controls'], ['zero', 'time_mean', 'half_roll'])

    def test_all_registered_conditions_can_pass(self):
        gate = run.comparison_gate(*passing_gate())
        self.assertTrue(gate['passed'])
        self.assertTrue(all(gate.values()))

    def test_good_phase_cannot_erase_bad_tempo_or_count(self):
        for role in ('development', 'diagnostic'):
            for parent in ('raw', 'v1', 'coupled', 'trajectory'):
                for key in KEYS:
                    rows, summary, fits = passing_gate()
                    summary[role][parent + '_common'][key] = .4
                    with self.subTest(role=role, parent=parent, metric=key):
                        self.assertFalse(run.comparison_gate(rows, summary, fits)['passed'])

    def test_each_temporal_control_is_mandatory(self):
        key = 'phase_mean_absolute_cycles'
        for role in ('development', 'diagnostic'):
            for kind in run.CONTROL_KINDS:
                rows, summary, fits = passing_gate()
                summary[role][kind][key] = summary[role]['audio_common'][key]
                gate = run.comparison_gate(rows, summary, fits)
                self.assertFalse(gate['temporal_phase_dependence'])
                self.assertFalse(gate['passed'])

    def test_incomplete_or_unbounded_fit_cannot_pass(self):
        for field, value in (('epochs', 19), ('updates', 99), ('elapsed_s', 1800.1),
                             ('elapsed_s', float('nan')), ('budget_exhausted', True)):
            rows, summary, fits = passing_gate()
            fits[0][field] = value
            self.assertFalse(run.comparison_gate(rows, summary, fits)['passed'])

    def test_missing_intervention_metric_fails_without_dropping_case(self):
        for kind in run.CONTROL_KINDS:
            for key in KEYS:
                rows, summary, fits = passing_gate()
                rows[0][kind][key] = None
                self.assertFalse(run.comparison_gate(rows, summary, fits)['temporal_controls_complete'])

    def test_same_support_rejects_denominator_and_comparator_changes(self):
        metric = {key: 1. for key in KEYS}
        metric.update(reference_points=250, reference_cells=249, reference_4s_intervals=50)
        row = {name: metric.copy() for name in ('audio_common', 'raw_common', 'v1_common')}
        run.same_support(row, copy.deepcopy(row))
        for kind, key in (('audio_common', 'reference_cells'), ('raw_common', KEYS[0]), ('v1_common', KEYS[2])):
            parent = copy.deepcopy(row)
            parent[kind][key] += 1
            with self.assertRaises(ValueError):
                run.same_support(row, parent)

    def test_evaluation_keeps_fields_controls_and_weights_fixed(self):
        frames = 251
        q = np.arange(frames, dtype=np.float64) * .04
        valid = np.ones(frames, bool)
        v1 = np.stack((np.full(frames, -1.), np.cos(2 * math.pi * q), np.sin(2 * math.pi * q)), 1)
        hidden = np.random.default_rng(111).normal(0, .1, (frames, 512)).astype(np.float32)
        row = dict(id='authored', role='diagnostic', work='authored', frames=frames,
                   packet_sha256='not-a-music-packet', hidden=hidden, reference=q, valid=valid,
                   raw=q.copy(), raw_valid=valid.copy(), v1=v1)
        common = measure(clock_fields(q), q, valid)
        parent = dict(id='authored', role='diagnostic', work='authored', frames=frames,
                      audio_common=common, raw_common=common, v1_common=measure(independent_fields(v1), q, valid))
        parents = {name: {'cases': [parent]} for name in ('coupled', 'trajectory')}
        torch.manual_seed(142)
        audio = PhaseSyncReadout().eval()
        zero = copy.deepcopy(audio)
        before = old.state_hash(audio)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = run.evaluate({'audio': audio, 'zero': zero}, [row], 'cpu', output, parents)[0]
            self.assertEqual(old.state_hash(audio), before)
            with np.load(output / 'authored.prediction.npz', allow_pickle=False) as archive:
                self.assertEqual(len(archive.files), 10)
                np.testing.assert_array_equal(archive['zero'], archive['same_zero'])
                for name in ('audio', 'zero', 'same_zero', 'time_mean', 'half_roll'):
                    self.assertEqual(archive[name].shape, (frames,))
                    self.assertEqual(archive['fields_' + name].shape, (frames, 4))
                    self.assertTrue((np.diff(archive[name]) > 0).all())
            for kind in run.ALL_KINDS:
                self.assertEqual(result[kind]['reference_cells'], frames - 1)
            with self.assertRaises(FileExistsError):
                run.evaluate({'audio': audio, 'zero': zero}, [row], 'cpu', output, parents)


if __name__ == '__main__':
    unittest.main()
