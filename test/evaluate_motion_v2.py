"""Unified A-E evaluation for the first ScanFlow v2 prototype.

Every method sees the same independently generated scenes.  The learned
scorer is selected before this script from a train/validation split; this
script never updates a checkpoint or tunes a threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_dataset import GeneratorConfig, generate_one_sample
from geometric_motion import extract_finite_surfaces
from motion_cost_model import MotionCandidateScorer
from motion_estimator_v2 import MotionEstimatorV2
from surface_motion import estimate_surface_motion


def _signal_metrics(pred, signal, gt, dynamic, valid, supported, points, entropy=None, margin=None):
    pred = np.asarray(pred, dtype=np.float64)
    signal = np.asarray(signal, dtype=bool)
    gt = np.asarray(gt, dtype=np.float64)
    dynamic = np.asarray(dynamic, dtype=bool) & np.asarray(valid, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    static = valid & ~dynamic
    positive = signal & valid
    tp = int((positive & dynamic).sum())
    fp = int((positive & static).sum())
    fn = int((~positive & dynamic).sum())
    epe_values = np.linalg.norm(pred[dynamic] - gt[dynamic], axis=-1)
    zero_values = np.linalg.norm(gt[dynamic], axis=-1)
    speeds = np.linalg.norm(pred, axis=-1)
    gt_speeds = np.linalg.norm(gt, axis=-1)
    angular_mask = dynamic & (gt_speeds > 1e-6) & (speeds > 1e-6)
    if angular_mask.any():
        cosine = np.sum(pred[angular_mask] * gt[angular_mask], axis=-1) / (
            speeds[angular_mask] * gt_speeds[angular_mask]
        )
        angular_error = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).mean())
    else:
        angular_error = None
    supported = np.asarray(supported, dtype=bool)
    distance = np.linalg.norm(points, axis=-1)

    def bucket(mask):
        mask = mask & valid
        dyn = mask & dynamic
        pos = mask & positive
        local_epe = np.linalg.norm(pred[dyn] - gt[dyn], axis=-1)
        local_zero = np.linalg.norm(gt[dyn], axis=-1)
        return {
            "count": int(mask.sum()),
            "dynamic_count": int(dyn.sum()),
            "f1": 2.0 * int((pos & dynamic).sum()) / max(2 * int((pos & dynamic).sum()) + int((pos & ~dynamic).sum()) + int((~pos & dyn).sum()), 1),
            "recall": int((pos & dyn).sum()) / max(int(dyn.sum()), 1),
            "epe_mps": float(local_epe.mean()) if len(local_epe) else None,
            "zero_epe_mps": float(local_zero.mean()) if len(local_zero) else None,
        }

    out = {
        "dynamic_beams": int(dynamic.sum()),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2.0 * tp / max(2 * tp + fp + fn, 1),
        "dynamic_epe_mps": float(epe_values.mean()) if len(epe_values) else None,
        "zero_epe_mps": float(zero_values.mean()) if len(zero_values) else None,
        "epe_gain_vs_zero": 1.0 - float(epe_values.mean()) / max(float(zero_values.mean()), 1e-8) if len(epe_values) else None,
        "speed_error_mps": float(np.abs(speeds[dynamic] - gt_speeds[dynamic]).mean()) if dynamic.any() else None,
        "angular_error_deg": angular_error,
        "static_mean_speed_mps": float(speeds[static].mean()) if static.any() else None,
        "static_false_dynamic_rate": float(positive[static].mean()) if static.any() else None,
        "supported_coverage": float(supported[valid].mean()) if valid.any() else 0.0,
        "dynamic_supported_recall": float(supported[dynamic].mean()) if dynamic.any() else None,
        "unsupported_fraction": float((~supported[valid]).mean()) if valid.any() else 1.0,
        "distance_bins": {
            "near_lt3m": bucket(distance < 3.0),
            "medium_3to6m": bucket((distance >= 3.0) & (distance < 6.0)),
            "far_ge6m": bucket(distance >= 6.0),
        },
    }
    if entropy is not None:
        entropy = np.asarray(entropy)
        out["candidate_entropy_mean"] = float(entropy[valid].mean()) if valid.any() else None
    if margin is not None:
        margin = np.asarray(margin)
        out["top1_top2_margin_mean"] = float(margin[valid].mean()) if valid.any() else None
    return out


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _run_estimator(estimator, sample):
    output = estimator(sample["lidar_history"], sample["odom_history"], sample["timestamps"])
    return {key: _as_numpy(value) for key, value in output.items()}


def _zero_velocity(sample):
    """A deliberately evidence-free zero-velocity reference."""
    beams = sample["lidar_history"].shape[-1]
    valid = sample["beam_valid"] > 0.5
    return {
        "velocity": np.zeros((1, beams, 2), dtype=np.float32),
        "motion_supported": np.zeros((1, beams), dtype=bool),
        "candidate_entropy": None,
        "candidate_margin": None,
    }


def _collect_samples(samples: int, seed: int, beams: int):
    cfg = GeneratorConfig(num_samples=samples, num_beams=beams, seed=seed, lidar_noise_std=0.01)
    rng = np.random.default_rng(seed)
    collected = []
    for index in range(samples):
        sample = generate_one_sample(rng, cfg)
        sample["timestamps"] = (np.arange(cfg.num_frames, dtype=np.float64) - (cfg.num_frames - 1)) * cfg.scan_dt
        collected.append(sample)
        if (index + 1) % 16 == 0 or index + 1 == samples:
            print(f"generated {index + 1}/{samples}", flush=True)
    return cfg, collected


def _save_raw_data(path: Path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        lidar_history=np.stack([sample["lidar_history"] for sample in samples]),
        odom_history=np.stack([sample["odom_history"] for sample in samples]),
        timestamps=np.stack([sample["timestamps"] for sample in samples]),
        beam_velocity=np.stack([sample["beam_velocity"] for sample in samples]),
        beam_dynamic=np.stack([sample["beam_dynamic"] for sample in samples]),
        beam_valid=np.stack([sample["beam_valid"] for sample in samples]),
    )


def evaluate(samples, checkpoint: str | None, device: str):
    target = torch.device(device)
    if checkpoint:
        learned_top1 = MotionEstimatorV2.from_checkpoint(
            checkpoint, device=target, top_k=1, refine=False,
            config=None,
        )
        learned_refine1 = MotionEstimatorV2.from_checkpoint(
            checkpoint, device=target, top_k=1, refine=True,
            config=None,
        )
        learned_refine3 = MotionEstimatorV2.from_checkpoint(
            checkpoint, device=target, top_k=3, refine=True,
            config=None,
        )
    else:
        scorer = MotionCandidateScorer().to(target).eval()
        learned_top1 = MotionEstimatorV2(scorer=scorer, device=target, top_k=1, refine=False)
        learned_refine1 = MotionEstimatorV2(scorer=scorer, device=target, top_k=1, refine=True)
        learned_refine3 = MotionEstimatorV2(scorer=scorer, device=target, top_k=3, refine=True)
    geometry_refine1 = MotionEstimatorV2(device=target, top_k=1, refine=True, score_mode="geometry")

    methods = {
        "Z_zero_velocity": {"time": [], "values": []},
        "A_surface_geometry": {"time": [], "values": []},
        "B_coarse_scorer_map": {"time": [], "values": []},
        "C_geometry_coarse_plus_refine": {"time": [], "values": []},
        "D_scorer_top1_plus_refine": {"time": [], "values": []},
        "E_scorer_top3_plus_refine": {"time": [], "values": []},
    }
    for index, sample in enumerate(samples):
        current_surfaces = extract_finite_surfaces(
            sample["lidar_history"][-1], sample["odom_history"][-1], sample["odom_history"][-1]
        )
        points = np.zeros((sample["lidar_history"].shape[-1], 2), dtype=np.float64)
        valid = np.zeros(len(points), dtype=bool)
        for surface in current_surfaces:
            beams = np.asarray(surface["beams"], dtype=np.int64)
            points[beams] = np.asarray(surface["points"])
            valid[beams] = True
        gt = sample["beam_velocity"]
        dynamic = sample["beam_dynamic"] > 0.5
        valid &= sample["beam_valid"] > 0.5

        started = time.perf_counter()
        zero = _zero_velocity(sample)
        methods["Z_zero_velocity"]["time"].append(time.perf_counter() - started)
        methods["Z_zero_velocity"]["values"].append(
            _signal_metrics(zero["velocity"][0],
                            np.zeros(len(points), dtype=bool), gt, dynamic, valid,
                            zero["motion_supported"][0], points)
        )

        started = time.perf_counter()
        surface = estimate_surface_motion(sample["lidar_history"], sample["odom_history"], sample["timestamps"])
        methods["A_surface_geometry"]["time"].append(time.perf_counter() - started)
        methods["A_surface_geometry"]["values"].append(
            _signal_metrics(surface["beam_velocity"], np.linalg.norm(surface["beam_velocity"], axis=-1) >= .15,
                            gt, dynamic, valid, surface["beam_supported"], points)
        )

        for name, estimator in [
            ("B_coarse_scorer_map", learned_top1),
            ("C_geometry_coarse_plus_refine", geometry_refine1),
            ("D_scorer_top1_plus_refine", learned_refine1),
            ("E_scorer_top3_plus_refine", learned_refine3),
        ]:
            started = time.perf_counter()
            output = _run_estimator(estimator, sample)
            methods[name]["time"].append(time.perf_counter() - started)
            speed_signal = np.linalg.norm(output["velocity"][0], axis=-1) >= .15
            probability_signal = output["dynamic_probability"][0] >= .5
            primary = _signal_metrics(
                output["velocity"][0], speed_signal, gt, dynamic, valid,
                output["motion_supported"][0], points,
                output["candidate_entropy"][0], output["candidate_margin"][0],
            )
            primary["probability_signal_metrics"] = _signal_metrics(
                output["velocity"][0], probability_signal, gt, dynamic, valid,
                output["motion_supported"][0], points,
            )
            timing = output.get("timing_ms")
            if timing is not None:
                methods[name].setdefault("component_time", []).append(timing[0])
            methods[name]["values"].append(primary)
        if (index + 1) % 8 == 0 or index + 1 == len(samples):
            print(f"evaluated {index + 1}/{len(samples)}", flush=True)

    report = {}
    for name, method_data in methods.items():
        values = method_data["values"]
        # Report the mean of per-scene metrics.  This avoids scenes with many
        # valid beams dominating the summary and keeps the distance buckets
        # comparable across methods.
        keys = ["dynamic_beams", "precision", "recall", "f1", "dynamic_epe_mps", "zero_epe_mps",
                "epe_gain_vs_zero", "speed_error_mps", "angular_error_deg", "static_mean_speed_mps",
                "static_false_dynamic_rate", "supported_coverage", "dynamic_supported_recall",
                "unsupported_fraction", "candidate_entropy_mean", "top1_top2_margin_mean"]
        summary = {key: float(np.mean([v[key] for v in values if v.get(key) is not None]))
                   if any(v.get(key) is not None for v in values) else None for key in keys}
        summary["samples"] = len(values)
        summary["runtime_ms"] = {
            "mean": float(np.mean(method_data["time"]) * 1000.0),
            "p50": float(np.percentile(method_data["time"], 50) * 1000.0),
            "p95": float(np.percentile(method_data["time"], 95) * 1000.0),
        }
        if method_data.get("component_time"):
            component = np.stack(method_data["component_time"])
            summary["component_runtime_ms"] = {
                "geometry": float(np.mean(component[:, 0])),
                "scorer": float(np.mean(component[:, 1])),
                "refinement": float(np.mean(component[:, 2])),
                "total": float(np.mean(component[:, 3])),
            }
        summary["distance_bins"] = {
            key: {metric: float(np.mean([v["distance_bins"][key][metric] for v in values
                                         if v["distance_bins"][key].get(metric) is not None]))
                  if any(v["distance_bins"][key].get(metric) is not None for v in values) else None
                  for metric in ["count", "dynamic_count", "f1", "recall", "epe_mps", "zero_epe_mps"]}
            for key in ["near_lt3m", "medium_3to6m", "far_ge6m"]
        }
        if "probability_signal_metrics" in values[0]:
            summary["probability_signal"] = {
                key: float(np.mean([v["probability_signal_metrics"][key] for v in values
                                    if v["probability_signal_metrics"].get(key) is not None]))
                if any(v["probability_signal_metrics"].get(key) is not None for v in values) else None
                for key in ["precision", "recall", "f1", "static_false_dynamic_rate"]
            }
        report[name] = summary
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--beams", type=int, default=180)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default="artifacts/candidate_scorer/best.pt")
    parser.add_argument("--save", type=Path, default=Path("artifacts/motion_v2/evaluation.json"))
    parser.add_argument("--save-data", type=Path, default=None)
    args = parser.parse_args()
    checkpoint = args.checkpoint if Path(args.checkpoint).exists() else None
    if checkpoint is None:
        print("warning: scorer checkpoint not found; using the untrained geometry-prior scorer", flush=True)
    cfg, samples = _collect_samples(args.samples, args.seed, args.beams)
    if args.save_data:
        _save_raw_data(args.save_data, samples)
    report = {
        "seed": args.seed,
        "samples": args.samples,
        "beams": args.beams,
        "frames": cfg.num_frames,
        "scan_dt": cfg.scan_dt,
        "noise_std": cfg.lidar_noise_std,
        "device": str(args.device),
        "gpu": torch.cuda.get_device_name() if str(args.device).startswith("cuda") and torch.cuda.is_available() else None,
        "checkpoint": checkpoint,
        "methods": evaluate(samples, checkpoint, args.device),
    }
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
