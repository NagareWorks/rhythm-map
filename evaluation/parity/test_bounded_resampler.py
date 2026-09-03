import copy
import unittest

from verify_bounded_resampler import compare_cases, pcm_bytes


class BoundedResamplerTests(unittest.TestCase):
    def case(self):
        return dict(sample_rate=44100, signal="dc", parameter=None,
                    input_pcm=[0.0, -0.0, 0.25, 0.25], current_pcm=[0.0, 0.25], candidate_pcm=[0.0, 0.25])

    def test_identity_preserves_signed_zero_not_just_numeric_equality(self):
        before = self.case()
        self.assertTrue(compare_cases([before], [copy.deepcopy(before)])[0]["passed"])
        after = copy.deepcopy(before)
        after["candidate_pcm"][0] = -0.0
        self.assertFalse(compare_cases([before], [after])[0]["passed"])

    def test_missing_reordered_and_duplicate_cases_fail(self):
        before = self.case()
        other = dict(before, signal="step")
        for old, new in (([], []), ([before], []), ([before, other], [other, before]),
                         ([before, before], [before, before])):
            with self.assertRaises(ValueError):
                compare_cases(old, new)

    def test_nonfinite_or_empty_pcm_is_rejected(self):
        for value in ([], [float("nan")], [float("inf")], [[0.0]]):
            with self.assertRaises(ValueError):
                pcm_bytes(value)


if __name__ == "__main__":
    unittest.main()
