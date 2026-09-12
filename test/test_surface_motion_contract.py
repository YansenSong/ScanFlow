"""Finite geometry, missing evidence, and temporal-direction checks."""
import unittest

import numpy as np

from surface_motion import SurfaceDistance, estimate_surface_motion
from generate_dataset import Circle, GeneratorConfig
from test.common import render_history


class SurfaceContract(unittest.TestCase):
    def test_finite_segments_do_not_extend_across_gaps(self):
        surface = SurfaceDistance([dict(points=np.array([[0., 0.], [1., 0.]])),
                                   dict(points=np.array([[3., 0.], [4., 0.]]))])
        np.testing.assert_allclose(surface.distance(np.array([[.5, .2], [2., 0.]])), [.2, 1.])

    def test_missing_history_is_not_static_evidence(self):
        scans = np.full((6, 180), np.inf)
        scans[-1, 90:94] = 2.
        out = estimate_surface_motion(scans, np.zeros((6, 3)), np.arange(6)*.1)
        self.assertFalse(out['beam_supported'].any())

    def test_repeated_geometry_has_zero_motion(self):
        scans = np.full((6, 180), np.inf)
        scans[:, 90:94] = 2.
        out = estimate_surface_motion(scans, np.zeros((6, 3)), np.arange(6)*.1)
        self.assertTrue(out['beam_supported'][90:94].all())
        np.testing.assert_array_equal(out['beam_velocity'], 0.)

    def test_same_current_scan_opposite_histories(self):
        cfg = GeneratorConfig(num_beams=180, lidar_noise_std=0.)
        odom = np.zeros((6, 3))
        current = []
        for speed in (.65, -.65):
            circle = Circle(np.array([3., .35]), .4, np.array([0., speed]), True, 100)
            scans, ids = render_history(cfg, odom, [], [circle])
            current.append(scans[-1])
            out = estimate_surface_motion(scans, odom, np.arange(6)*.1)
            self.assertGreater(speed * out['beam_velocity'][ids[-1] >= 0, 1].mean(), 0.)
        np.testing.assert_array_equal(*current)


if __name__ == '__main__':
    unittest.main()
