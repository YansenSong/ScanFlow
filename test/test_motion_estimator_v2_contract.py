"""v2 API, separation of support/confidence, and top-k refinement contracts."""
import unittest

import numpy as np
import torch

from generate_dataset import Circle, GeneratorConfig
from motion_estimator_v2 import MotionEstimatorV2
from test.common import render_history


class MotionEstimatorV2Contract(unittest.TestCase):
    def test_output_shapes_and_batching(self):
        scans = np.full((2, 6, 180), np.inf)
        scans[:, :, 90:94] = 2.
        odom = np.zeros((2, 6, 3))
        times = np.broadcast_to(np.arange(6)*.1, (2, 6))
        out = MotionEstimatorV2(score_mode='geometry', device='cpu', top_k=3)(scans, odom, times)
        self.assertEqual(out['points'].shape, (2, 180, 2))
        self.assertEqual(out['velocity'].shape, (2, 180, 2))
        self.assertEqual(out['candidate_scores'].shape[-1], 113)
        self.assertEqual(out['topk_velocity'].shape, (2, 180, 3, 2))
        self.assertEqual(out['motion_supported'].dtype, torch.bool)
        self.assertTrue(torch.isfinite(out['candidate_entropy']).all())
        self.assertEqual(out['timing_ms'].shape, (2, 4))
        self.assertTrue(torch.isfinite(out['timing_ms']).all())

    def test_no_history_is_unsupported_and_zero_velocity(self):
        scans = np.full((6, 180), np.inf)
        scans[-1, 90:94] = 2.
        out = MotionEstimatorV2(score_mode='geometry', device='cpu')(scans, np.zeros((6, 3)), np.arange(6)*.1)
        self.assertFalse(out['motion_supported'].any())
        self.assertTrue(torch.allclose(out['velocity'], torch.zeros_like(out['velocity'])))
        self.assertTrue(torch.allclose(out['motion_confidence'], torch.zeros_like(out['motion_confidence'])))

    def test_history_direction_changes_refined_velocity(self):
        cfg = GeneratorConfig(num_beams=180, lidar_noise_std=0.)
        odom = np.zeros((6, 3))
        values = []
        for speed in (.65, -.65):
            circle = Circle(np.array([3., .35]), .4, np.array([0., speed]), True, 100)
            scans, ids = render_history(cfg, odom, [], [circle])
            out = MotionEstimatorV2(score_mode='geometry', device='cpu', top_k=3)(scans, odom, np.arange(6)*.1)
            mask = ids[-1] >= 0
            values.append(float(out['velocity'][0, mask, 1].mean()))
        self.assertGreater(values[0], .3)
        self.assertLess(values[1], -.3)

    def test_refinement_is_continuous_relative_to_coarse(self):
        cfg = GeneratorConfig(num_beams=180, lidar_noise_std=0.)
        odom = np.zeros((6, 3))
        circle = Circle(np.array([3., .35]), .4, np.array([0., .65]), True, 100)
        scans, ids = render_history(cfg, odom, [], [circle])
        coarse = MotionEstimatorV2(score_mode='geometry', device='cpu', top_k=1, refine=False)(
            scans, odom, np.arange(6)*.1)
        refined = MotionEstimatorV2(score_mode='geometry', device='cpu', top_k=1, refine=True)(
            scans, odom, np.arange(6)*.1)
        mask = ids[-1] >= 0
        self.assertGreater(float(torch.linalg.vector_norm(refined['velocity'][0, mask], dim=-1).mean()),
                           float(torch.linalg.vector_norm(coarse['velocity'][0, mask], dim=-1).mean()) - .1)
        self.assertLess(float(torch.abs(refined['velocity'][0, mask, 1]).max()), 1.51)


if __name__ == '__main__':
    unittest.main()
