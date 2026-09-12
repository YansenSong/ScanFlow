"""Physical invariants and empty-observation behavior of geometry baseline."""
import unittest

import numpy as np

from geometric_motion import GeometryConfig, estimate_motion, extract_surfaces


class GeometryContract(unittest.TestCase):
    def test_empty_history_has_no_supported_motion(self):
        scans = np.full((6, 180), np.inf)
        scans[-1, 90:94] = 2.
        out = estimate_motion(scans, np.zeros((6, 3)), np.arange(6)*.1)
        self.assertFalse(out['beam_supported'].any())
        np.testing.assert_array_equal(out['beam_velocity'], 0.)

    def test_repeated_scan_zero_velocity_and_full_support(self):
        scans = np.full((6, 180), np.inf)
        scans[:, 90:94] = 2.
        out = estimate_motion(scans, np.zeros((6, 3)), np.arange(6)*.1)
        self.assertTrue(out['beam_supported'][90:94].all())
        np.testing.assert_array_equal(out['beam_velocity'], 0.)

    def test_alignment_and_circular_support(self):
        scan = np.full(180, np.inf)
        scan[[179, 0, 1]] = 2.
        surfaces = extract_surfaces(scan, np.array([1., 0., np.pi/2]), np.zeros(3), GeometryConfig())
        self.assertEqual(len(surfaces), 1)
        surface = surfaces[0]
        i = list(surface['beams']).index(0)
        np.testing.assert_allclose(surface['points'][i], [1., -2.], atol=1e-12)
        np.testing.assert_array_equal(np.sort(surface['beams']), [0, 1, 179])

    def test_time_order_is_required(self):
        with self.assertRaises(ValueError):
            estimate_motion(np.ones((3, 180)), np.zeros((3, 3)), [0., 0., 1.])

    def test_nonzero_motion_scales_with_elapsed_time(self):
        scans = np.full((6, 180), np.inf)
        scans[:, 90:94] = (2. + np.arange(6)*.05)[:, None]
        times = np.arange(6)*.1
        out = estimate_motion(scans, np.zeros((6, 3)), times)
        slower = estimate_motion(scans, np.zeros((6, 3)), 2*times)
        self.assertGreater(np.linalg.norm(out['beam_velocity'][91]), .4)
        np.testing.assert_allclose(slower['beam_velocity'], out['beam_velocity']/2, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
