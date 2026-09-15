"""CPU authored capture contract; real encoder is a separately retained GPU run."""
import unittest
import numpy as np
import torch

from experiments.tempo_source_expansion.capture import capture_chunk, state_hash


class Heads(torch.nn.Module):
    def forward(self, x):
        return dict(beat=x[..., 0], downbeat=x[..., 1])


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.task_heads = Heads()
        self.fail_tapped = False

    def forward(self, x):
        if self.fail_tapped and self.task_heads._forward_pre_hooks:
            raise RuntimeError('authored failure')
        return self.task_heads(x.repeat(1, 1, 4)*self.scale)


class CaptureTests(unittest.TestCase):
    def test_hidden_replay_and_passive_heads(self):
        model = Toy().eval()
        before = state_hash(model)
        mel = np.arange(17*128, dtype=np.float32).reshape(17, 128)/1000
        result, error = capture_chunk(model, mel)
        np.testing.assert_array_equal(result['hidden'], np.tile(mel, (1, 4)))
        self.assertEqual(error, 0)
        self.assertEqual(before, state_hash(model))
        self.assertFalse(model.task_heads._forward_pre_hooks)
        self.assertIsNone(model.scale.grad)

    def test_hook_removed_on_failure(self):
        model = Toy().eval()
        model.fail_tapped = True
        with self.assertRaises(RuntimeError): capture_chunk(model, np.zeros((17, 128), np.float32))
        self.assertFalse(model.task_heads._forward_pre_hooks)

    def test_unfrozen_or_training_model_rejected(self):
        model = Toy()
        with self.assertRaises(ValueError): capture_chunk(model, np.zeros((17, 128), np.float32))
        model.eval().scale.requires_grad_(True)
        with self.assertRaises(ValueError): capture_chunk(model, np.zeros((17, 128), np.float32))

    def test_preexisting_tap_rejected_and_not_removed(self):
        model = Toy().eval()
        handle = model.task_heads.register_forward_pre_hook(lambda *_: None)
        with self.assertRaises(ValueError): capture_chunk(model, np.zeros((17, 128), np.float32))
        self.assertEqual(len(model.task_heads._forward_pre_hooks), 1)
        handle.remove()

    def test_nonfinite_mel_rejected(self):
        with self.assertRaises(ValueError): capture_chunk(Toy().eval(), np.full((17, 128), np.nan, np.float32))


if __name__ == '__main__': unittest.main()
