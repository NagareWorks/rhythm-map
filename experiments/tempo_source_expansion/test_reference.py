"""CPU proof that the new NumPy adapter matches the unchanged training target."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'evaluation/parity'))
from brid_reference import reference_arrays, tempo_target
from experiments.coupled_clock.supervision import reference_clock
from experiments.separated_evidence.supervision import task_loss


class ExistingReferenceContractTests(unittest.TestCase):
    def test_exact_original_reference_and_zero_tempo_loss(self):
        for beats in ([.1, .51, .8, 1.4], [0., .4, .9, 1.3], [.011, .5, 1., 1.8]):
            for hole in (False, True):
                support = np.ones(101, bool)
                if hole:
                    support[20:25] = False
                new = reference_arrays(beats, [2, 1, 2, 1], 101, 2., support)
                old = reference_clock(beats, 101, 2., support)
                np.testing.assert_array_equal(new['reference'], old.cycles)
                np.testing.assert_array_equal(new['valid'], old.valid)
                # The production loss requires every prediction finite, even
                # outside support. Only targets carry unknown NaNs.
                prediction = np.nan_to_num(tempo_target(new))
                loss = task_loss('tempo', torch.from_numpy(prediction)[None],
                                 torch.from_numpy(old.cycles)[None], torch.from_numpy(old.valid)[None])
                self.assertLess(float(loss), 1e-28)

    def test_unknown_prediction_still_fails(self):
        a = reference_arrays([.1, .6, 1.1], [1, 2, 1], 101, 2.)
        with self.assertRaisesRegex(ValueError, 'nonfinite task prediction'):
            task_loss('tempo', torch.from_numpy(tempo_target(a))[None],
                      torch.from_numpy(a['reference'])[None], torch.from_numpy(a['valid'])[None])


if __name__ == '__main__':
    unittest.main()
