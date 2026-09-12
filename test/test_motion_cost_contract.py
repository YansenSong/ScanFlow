import unittest

import numpy as np
import torch

from motion_cost_model import matching_features, MotionCandidateScorer, velocity_candidates


class CostContract(unittest.TestCase):
    def test_empty_evidence_is_finite_and_unsupported(self):
        features, supported = matching_features(np.full((6, 180), np.inf), np.zeros((6, 3)), np.arange(6)*.1)
        self.assertTrue(np.isfinite(features).all())
        self.assertFalse(supported.any())
        model = MotionCandidateScorer()
        velocity, probability = model.predict(torch.from_numpy(features), torch.from_numpy(supported))
        np.testing.assert_array_equal(velocity.detach().numpy(), 0.)
        torch.testing.assert_close(probability.sum(-1), torch.ones(180))

    def test_static_candidate_is_unique_and_in_bounds(self):
        grid = velocity_candidates()
        self.assertEqual(int((np.linalg.norm(grid, axis=-1) == 0).sum()), 1)
        self.assertLessEqual(float(np.linalg.norm(grid, axis=-1).max()), 1.5)

    def test_initial_logits_preserve_geometry_prior(self):
        model = MotionCandidateScorer()
        features = torch.rand(2, 12, len(velocity_candidates()), 8)
        torch.testing.assert_close(model(features), -4.*features[..., 2]-.08*features[..., 5])
        model(features).sum().backward()
        self.assertTrue(torch.isfinite(model.net[-1].weight.grad).all())


if __name__ == '__main__':
    unittest.main()
