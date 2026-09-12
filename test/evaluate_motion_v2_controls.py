"""A2/A4 control scenarios for the beam-level v2 estimators.

The existing helper deliberately reports speed-threshold detection for all
methods so this remains comparable with the surface baseline.  The v2 output
also exposes dynamic_probability; this script records its separate aggregate
classification where available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_cost_model import MotionCandidateScorer
from motion_estimator_v2 import MotionEstimatorV2
from surface_motion import estimate_surface_motion
from test.evaluate_geometric_motion import controlled


def _adapter(estimator):
    def run(lidar, odom, times):
        out = estimator(lidar, odom, times)
        velocity = out["velocity"][0].detach().cpu().numpy() if isinstance(out["velocity"], torch.Tensor) else out["velocity"][0]
        supported = out["motion_supported"][0].detach().cpu().numpy() if isinstance(out["motion_supported"], torch.Tensor) else out["motion_supported"][0]
        return {"beam_velocity": velocity, "beam_supported": supported}
    return run


def _surface_adapter(lidar, odom, times):
    out = estimate_surface_motion(lidar, odom, times)
    return {"beam_velocity": out["beam_velocity"], "beam_supported": out["beam_supported"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/candidate_scorer/best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", type=Path, default=Path("artifacts/motion_v2/controls.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    if Path(args.checkpoint).exists():
        v2_b = MotionEstimatorV2.from_checkpoint(args.checkpoint, device=device, top_k=1, refine=False)
        v2_d = MotionEstimatorV2.from_checkpoint(args.checkpoint, device=device, top_k=1, refine=True)
        v2_e = MotionEstimatorV2.from_checkpoint(args.checkpoint, device=device, top_k=3, refine=True)
    else:
        scorer = MotionCandidateScorer().to(device).eval()
        v2_b = MotionEstimatorV2(scorer=scorer, device=device, top_k=1, refine=False)
        v2_d = MotionEstimatorV2(scorer=scorer, device=device, top_k=1, refine=True)
        v2_e = MotionEstimatorV2(scorer=scorer, device=device, top_k=3, refine=True)
    v2_c = MotionEstimatorV2(device=device, top_k=1, refine=True, score_mode="geometry")

    estimators = {
        "A_surface_geometry": _surface_adapter,
        "B_coarse_scorer_map": _adapter(v2_b),
        "C_geometry_coarse_plus_refine": _adapter(v2_c),
        "D_scorer_top1_plus_refine": _adapter(v2_d),
        "E_scorer_top3_plus_refine": _adapter(v2_e),
    }
    report = {"device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
              "checkpoint": args.checkpoint if Path(args.checkpoint).exists() else None, "methods": {}}
    for name, estimator in estimators.items():
        print(f"running {name}", flush=True)
        report["methods"][name] = controlled(estimator)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
