"""Public geometry layer contracts and legacy-compatible cost volume."""
import unittest

import numpy as np

from motion_geometry import (
    GeometryConfig,
    FiniteSurfaceDistance,
    align_history_to_current,
    compute_motion_cost_volume,
    extract_finite_surfaces,
    scan_to_xy,
    surface_distance,
    velocity_candidates,
)


class MotionGeometryContract(unittest.TestCase):
    def test_scan_to_xy_and_invalid_range(self):
        scan = np.array([1., np.inf, 0.01, 2.])
        points, valid, angles = scan_to_xy(scan, GeometryConfig())
        self.assertEqual(points.shape, (4, 2))
        np.testing.assert_array_equal(valid, [True, False, False, True])
        self.assertEqual(angles.shape, (4,))
        np.testing.assert_array_equal(points[~valid], 0.)

    def test_alignment_identity_and_se2(self):
        scan = np.full(180, np.inf)
        scan[90:94] = 2.
        identity, valid_a, _ = align_history_to_current(scan, np.zeros(3), np.zeros(3))
        moved, valid_b, _ = align_history_to_current(scan, np.array([1., 0., .2]), np.array([1., 0., .2]))
        np.testing.assert_allclose(identity, moved, atol=1e-12)
        np.testing.assert_array_equal(valid_a, valid_b)

    def test_surface_distance_respects_finite_endpoints(self):
        distance = FiniteSurfaceDistance([dict(points=np.array([[0., 0.], [1., 0.]])),
                                           dict(points=np.array([[3., 0.], [4., 0.]]))])
        np.testing.assert_allclose(distance.distance(np.array([[.5, .2], [2., 0.]])), [.2, 1.])
        np.testing.assert_allclose(surface_distance([dict(points=np.array([[0., 0.], [1., 0.]]))],
                                                    np.array([[.5, .2]])), [.2])

    def test_cost_volume_exposes_metric_evidence(self):
        scans = np.full((6, 180), np.inf)
        scans[:, 90:94] = 2.
        volume = compute_motion_cost_volume(scans, np.zeros((6, 3)), np.arange(6)*.1)
        self.assertEqual(volume['features'].shape[-1], 8)
        self.assertEqual(volume['extended_features'].shape[-1], 11)
        self.assertEqual(volume['features'].shape[1], len(velocity_candidates()))
        self.assertEqual(volume['support_frames'].shape, volume['median_cost'].shape)
        self.assertTrue(np.isfinite(volume['features']).all())

    def test_missing_history_remains_unsupported(self):
        scans = np.full((6, 180), np.inf)
        scans[-1, 90:94] = 2.
        volume = compute_motion_cost_volume(scans, np.zeros((6, 3)), np.arange(6)*.1)
        self.assertFalse(volume['supported'].any())

    def test_surface_extraction_has_beam_support(self):
        scan = np.full(180, np.inf)
        scan[90:94] = 2.
        surfaces = extract_finite_surfaces(scan, np.zeros(3), np.zeros(3))
        self.assertEqual(sum(len(surface['beams']) for surface in surfaces), 4)


if __name__ == '__main__':
    unittest.main()
