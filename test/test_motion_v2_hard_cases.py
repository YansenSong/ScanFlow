"""Deterministic hard-case behavior for the geometry v2 prototype."""
import unittest

from test.evaluate_motion_v2_hard_cases import evaluate_hard_cases


class MotionV2HardCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = evaluate_hard_cases()

    def test_opposing_movers_keep_distinct_direction(self):
        self.assertTrue(self.report["J1_two_opposing_movers"]["not_averaged_to_zero"])

    def test_static_and_dynamic_angular_neighbors_stay_separate(self):
        self.assertTrue(self.report["J2_static_plus_dynamic_same_angular_neighborhood"]["same_angular_neighborhood_separated"])

    def test_emergence_is_unsupported_and_does_not_fabricate_velocity(self):
        self.assertTrue(self.report["J3_occlusion_emergence"]["zero_velocity_without_support"])

    def test_tangential_ambiguity_is_low_confidence(self):
        self.assertTrue(self.report["J4_tangential_ambiguity"]["ambiguous_signal_ok"])


if __name__ == "__main__":
    unittest.main()
