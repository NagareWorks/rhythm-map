"""Public authored fixtures; no calibration recordings or annotations required."""
import copy
import unittest

from active_region_parity import compare_case, metrics, rows_by_id


class ActiveRegionParityTests(unittest.TestCase):
    def setUp(self):
        primary = [i * 0.5 for i in range(8)]
        self.evidence = dict(observations=dict(beat_candidates=[dict(time_s=t) for t in primary]),
                             truth_times_s=primary, beat_tolerance_s=0.07)
        self.history = dict(pulse_hypothesis_coverage=dict(hypotheses=[dict(id="selected", beat_times_s=primary)]))
        self.part = dict(start_s=0.0, end_s=4.0, status="proposal", candidate_count=8,
                         original_times_s=primary[:], proposal_times_s=primary[:], disagreements=[])
        self.row = dict(primary_replay_exact=True, generated=dict(generator="active-interval-path-v1",
                        silence_regions=[], unknown_gaps=[], proposals=[copy.deepcopy(self.part)]))
        self.frozen = dict(silence_regions=[], unknown_gaps=[], proposals=[copy.deepcopy(self.part)],
                           selected=dict(tp=8, fp=0, fn=0, f1=1.0),
                           forced_active_paths=dict(tp=8, fp=0, fn=0, f1=1.0), forced_times_s=primary)

    def check(self):
        compare_case(self.row, self.frozen, self.evidence, self.history)

    def test_exact_candidate_and_metrics(self):
        self.check()

    def test_fallback_preserves_primary(self):
        for doc in (self.row["generated"], self.frozen):
            doc["proposals"][0].update(status="fallback_no_valid_path", proposal_times_s=None)
        self.check()

    def test_omitted_component_rejected(self):
        self.row["generated"]["proposals"].clear()
        with self.assertRaisesRegex(ValueError, "component count"):
            self.check()

    def test_modified_path_rejected(self):
        self.row["generated"]["proposals"][0]["proposal_times_s"][3] += 0.01
        with self.assertRaisesRegex(ValueError, "timestamps mismatch"):
            self.check()

    def test_changed_unknown_or_silence_boundary_rejected(self):
        for key in ("unknown_gaps", "silence_regions"):
            self.row["generated"][key] = [[0.0, 0.1]]
            with self.assertRaisesRegex(ValueError, "mismatch"):
                self.check()
            self.row["generated"][key] = []

    def test_fallback_cannot_hide_replacement(self):
        for doc in (self.row["generated"], self.frozen):
            doc["proposals"][0]["status"] = "fallback_no_valid_path"
        with self.assertRaisesRegex(ValueError, "invalid fallback"):
            self.check()

    def test_primary_edit_counts_checked(self):
        self.row["generated"]["proposals"][0]["disagreements"] = [dict(primary_only_beat_count=1, alternative_only_beat_count=0)]
        with self.assertRaisesRegex(ValueError, "primary edit count"):
            self.check()

    def test_duplicate_case_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            rows_by_id([dict(id="fixture"), dict(id="fixture")])

    def test_metric_matching_is_chronological_not_nearest_neighbor(self):
        self.assertEqual(metrics([0.0, 0.06], [0.05], 0.07), dict(tp=1, fp=1, fn=0, f1=2 / 3))


if __name__ == "__main__":
    unittest.main()
