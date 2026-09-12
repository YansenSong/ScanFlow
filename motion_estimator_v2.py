"""ScanFlow v2 beam-level motion estimator.

Geometry creates metric motion hypotheses and performs the final continuous
refinement.  The optional learned scorer only chooses coarse basins; it never
regresses an unconstrained velocity vector.  This module is intentionally
independent from the legacy ``DynamicLiDARNetwork``.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Optional

import numpy as np
import torch

from motion_geometry import (
    GeometryConfig,
    build_local_neighborhood,
    compute_motion_cost_volume,
    motion_cost_for_points,
)
from motion_cost_model import MotionCandidateScorer


@dataclass
class MotionEstimatorV2Config:
    candidate_max_speed: float = 1.5
    candidate_step: float = 0.25
    neighborhood_radius: float = 0.35
    neighborhood_beams: int = 3

    refinement_radius: float = 0.25
    refinement_step: float = 0.05
    refinement_regularizer: float = 0.002

    noise_tolerance: float = 0.025
    max_residual: float = 0.06
    min_static_improvement: float = 0.015
    minimum_history_surfaces: int = 2
    confidence_residual_scale: float = 0.05

    def __post_init__(self) -> None:
        if self.candidate_max_speed <= 0 or self.candidate_step <= 0:
            raise ValueError("candidate_max_speed and candidate_step must be positive.")
        if self.refinement_radius < 0 or self.refinement_step <= 0:
            raise ValueError("refinement_radius must be non-negative and refinement_step positive.")
        if self.neighborhood_radius <= 0 or self.neighborhood_beams < 0:
            raise ValueError("neighborhood geometry is invalid.")
        if self.minimum_history_surfaces < 1:
            raise ValueError("minimum_history_surfaces must be positive.")


def _to_numpy(value, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    return value.astype(dtype, copy=False) if dtype is not None else value


def _as_batch(lidar_history, odom_history, timestamps):
    lidar = _to_numpy(lidar_history, np.float64)
    odom = _to_numpy(odom_history, np.float64)
    times = _to_numpy(timestamps, np.float64)
    if lidar.ndim == 2:
        lidar = lidar[None]
    if odom.ndim == 2:
        odom = odom[None]
    if times.ndim == 1:
        times = times[None]
    if lidar.ndim != 3 or odom.ndim != 3 or times.ndim != 2:
        raise ValueError("Expected lidar [B,K,H], odom [B,K,3], timestamps [B,K].")
    if odom.shape[:2] != lidar.shape[:2] or odom.shape[-1] != 3 or times.shape != lidar.shape[:2]:
        raise ValueError("lidar/odom/timestamps shapes are incompatible.")
    if lidar.shape[1] < 2:
        raise ValueError("v2 requires a current scan and at least one history scan.")
    return lidar, odom, times


def _fine_candidates(seed: np.ndarray, cfg: MotionEstimatorV2Config) -> np.ndarray:
    if cfg.refinement_radius <= 1e-12:
        return seed.reshape(1, 2).astype(np.float64)
    offsets = np.arange(-cfg.refinement_radius, cfg.refinement_radius + cfg.refinement_step * 0.5,
                        cfg.refinement_step, dtype=np.float64)
    offsets = np.stack(np.meshgrid(offsets, offsets), axis=-1).reshape(-1, 2)
    candidates = seed[None] + offsets
    candidates = candidates[np.linalg.norm(candidates, axis=-1) <= cfg.candidate_max_speed + 1e-8]
    # Round only for stable de-duplication; the returned values remain metric.
    return np.unique(np.round(candidates, decimals=8), axis=0)


class MotionEstimatorV2:
    """Beam-level v2 estimator with optional GPU candidate scoring."""

    def __init__(
        self,
        scorer: Optional[MotionCandidateScorer] = None,
        config: Optional[MotionEstimatorV2Config] = None,
        device: Optional[torch.device | str] = None,
        top_k: int = 3,
        refine: bool = True,
        score_mode: str = "model",
        geometry_config: Optional[GeometryConfig] = None,
    ) -> None:
        self.config = config or MotionEstimatorV2Config()
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        if score_mode not in {"model", "geometry"}:
            raise ValueError("score_mode must be 'model' or 'geometry'.")
        self.top_k = int(top_k)
        self.refine = bool(refine)
        self.score_mode = score_mode
        self.geometry_config = geometry_config or GeometryConfig(max_speed=self.config.candidate_max_speed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.scorer = scorer
        if self.score_mode == "model":
            self.scorer = self.scorer or MotionCandidateScorer()
            self.scorer = self.scorer.to(self.device).eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        device: Optional[torch.device | str] = None,
        **kwargs,
    ) -> "MotionEstimatorV2":
        target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(checkpoint, map_location=target, weights_only=True)
        scorer = MotionCandidateScorer().to(target)
        state = payload.get("model", payload.get("model_state_dict"))
        if state is None:
            raise KeyError("Checkpoint must contain model or model_state_dict.")
        scorer.load_state_dict(state, strict=True)
        return cls(scorer=scorer, device=target, **kwargs)

    def _score(self, volume: Dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(volume["features"], dtype=np.float32)
        candidates = np.asarray(volume["candidates"], dtype=np.float32)
        if self.score_mode == "geometry":
            # The legacy geometry scorer uses its local neighborhood cost for
            # the coarse basin; retaining that definition makes C comparable
            # with the surface baseline rather than with a single-beam cost.
            logits = -np.asarray(volume["local_cost"], dtype=np.float32)
        else:
            assert self.scorer is not None
            with torch.no_grad():
                tensor = torch.from_numpy(features).to(self.device)
                logits = self.scorer(tensor).detach().cpu().numpy().astype(np.float32)
        if logits.shape != (features.shape[0], candidates.shape[0]):
            raise RuntimeError("candidate scorer returned an incompatible shape.")
        return logits, candidates

    def _estimate_one(self, lidar: np.ndarray, odom: np.ndarray, times: np.ndarray) -> Dict[str, np.ndarray]:
        cfg = self.config
        total_started = time.perf_counter()
        geometry_started = total_started
        volume = compute_motion_cost_volume(
            lidar,
            odom,
            times,
            geometry_cfg=self.geometry_config,
            candidate_max_speed=cfg.candidate_max_speed,
            candidate_step=cfg.candidate_step,
            neighborhood_radius=cfg.neighborhood_radius,
            neighborhood_beams=cfg.neighborhood_beams,
        )
        geometry_ms = (time.perf_counter() - geometry_started) * 1000.0
        scorer_started = time.perf_counter()
        logits, candidates = self._score(volume)
        scorer_ms = (time.perf_counter() - scorer_started) * 1000.0
        refinement_started = time.perf_counter()
        probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        order = np.argsort(-logits, axis=-1)
        top_k = min(self.top_k, candidates.shape[0])
        top_indices = order[:, :top_k]
        top_scores = np.take_along_axis(logits, top_indices, axis=1)
        coarse_velocity = candidates[top_indices[:, 0]].copy()
        points = np.asarray(volume["points"], dtype=np.float64)
        valid = np.asarray(volume["valid"], dtype=bool)
        base_supported = np.asarray(volume["supported"], dtype=bool)
        history = volume["history"]
        lags = np.asarray(volume["lags"], dtype=np.float64)
        static_cost = np.asarray(volume["static_cost"], dtype=np.float64)[:, 0].copy()
        static_index = int(volume["zero_index"])

        velocity = np.zeros_like(points)
        refined_velocity = np.zeros_like(points)
        fit_residual = np.full(len(points), np.nan, dtype=np.float64)
        static_improvement = np.full(len(points), np.nan, dtype=np.float64)
        motion_supported = np.zeros(len(points), dtype=bool)

        if self.score_mode == "geometry":
            # A geometry-only probability proxy compares the best coarse
            # explanation with the static one.  It is deliberately not called
            # calibrated: its use is for a like-for-like diagnostic only.
            best_geometry = np.asarray(volume["local_cost"], dtype=np.float64).min(axis=1)
            dynamic_probability = 1.0 / (1.0 + np.exp(
                -np.clip((static_cost - best_geometry - cfg.min_static_improvement) / 0.02, -30.0, 30.0)
            ))
        else:
            # Softmax mass is diluted over all coarse candidates and is not a
            # useful presence score.  Use the learned non-static-vs-static
            # logit gap; this remains a ranking proxy until calibration.
            nonstatic = np.delete(logits, static_index, axis=1).max(axis=1)
            dynamic_probability = 1.0 / (1.0 + np.exp(-np.clip(
                nonstatic - logits[:, static_index], -30.0, 30.0
            )))
        if sum(surface.tree is not None for surface in history) < cfg.minimum_history_surfaces:
            dynamic_probability = np.zeros_like(dynamic_probability)
        dynamic_probability = dynamic_probability * valid.astype(np.float32)
        entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)
        # Always expose the full-candidate top1-vs-top2 gap, even for a top-1
        # refinement run.
        margin = logits[np.arange(len(logits)), order[:, 0]] - logits[np.arange(len(logits)), order[:, 1]]
        best_indices = np.full(len(points), static_index, dtype=np.int64)

        for i in np.flatnonzero(valid):
            local_ids = build_local_neighborhood(
                points, valid, int(i), cfg.neighborhood_radius, cfg.neighborhood_beams
            )
            if len(local_ids) < 2 or sum(surface.tree is not None for surface in history) < cfg.minimum_history_surfaces:
                continue
            local_points = points[local_ids]
            static = float(motion_cost_for_points(local_points, np.zeros((1, 2)), history, lags)[0])
            static_cost[i] = static

            if not self.refine:
                selected = coarse_velocity[i]
                selected_cost = float(motion_cost_for_points(local_points, selected[None], history, lags)[0])
                refined_velocity[i] = selected
                fit_residual[i] = selected_cost
                static_improvement[i] = static - selected_cost
                motion_supported[i] = base_supported[i]
                best_indices[i] = top_indices[i, 0]
                velocity[i] = selected if motion_supported[i] else 0.0
                continue

            if static <= cfg.noise_tolerance:
                # Static geometry is already a good explanation; retain the
                # evidence flag while reporting a zero dynamic motion.
                fit_residual[i] = static
                static_improvement[i] = 0.0
                motion_supported[i] = base_supported[i]
                refined_velocity[i] = 0.0
                velocity[i] = 0.0
                best_indices[i] = static_index
                continue

            fine = np.concatenate([_fine_candidates(candidates[index], cfg) for index in top_indices[i]], axis=0)
            fine = np.unique(np.round(fine, decimals=8), axis=0)
            costs = motion_cost_for_points(local_points, fine, history, lags)
            choice = int(np.argmin(costs + cfg.refinement_regularizer * np.linalg.norm(fine, axis=-1)))
            selected = fine[choice]
            selected_cost = float(costs[choice])
            refined_velocity[i] = selected
            fit_residual[i] = selected_cost
            static_improvement[i] = static - selected_cost
            motion_supported[i] = bool(
                base_supported[i]
                and selected_cost <= cfg.max_residual
                and static_improvement[i] >= cfg.min_static_improvement
            )
            velocity[i] = selected if motion_supported[i] else 0.0
            # Candidate index is only a debug pointer; a fine result can lie
            # between grid candidates, so static is used as a sentinel.
            nearest = np.linalg.norm(candidates - selected[None], axis=-1).argmin()
            best_indices[i] = int(nearest)

        # For valid points with insufficient history, preserve a finite metric
        # debug output only when a history surface exists; support remains false.
        no_fit = valid & ~np.isfinite(fit_residual)
        fit_residual[no_fit & np.isfinite(static_cost)] = static_cost[no_fit & np.isfinite(static_cost)]
        confidence = np.zeros(len(points), dtype=np.float32)
        fit_ok = np.isfinite(fit_residual)
        residual_factor = np.exp(-np.nan_to_num(fit_residual, nan=1e6) / max(cfg.confidence_residual_scale, 1e-6))
        margin_factor = 1.0 / (1.0 + np.exp(-np.clip(margin, -30.0, 30.0)))
        entropy_factor = np.clip(1.0 - entropy / max(np.log(len(candidates)), 1e-6), 0.0, 1.0)
        confidence[fit_ok] = (residual_factor * margin_factor * entropy_factor)[fit_ok].astype(np.float32)
        confidence *= motion_supported.astype(np.float32)

        refinement_ms = (time.perf_counter() - refinement_started) * 1000.0
        total_ms = (time.perf_counter() - total_started) * 1000.0

        return {
            "points": points.astype(np.float32),
            "valid": valid,
            "velocity": velocity.astype(np.float32),
            "dynamic_probability": dynamic_probability.astype(np.float32),
            "motion_supported": motion_supported,
            "motion_confidence": confidence,
            "fit_residual": fit_residual.astype(np.float32),
            "candidate_entropy": entropy.astype(np.float32),
            "candidate_margin": margin.astype(np.float32),
            "coarse_velocity": coarse_velocity.astype(np.float32),
            "refined_velocity": refined_velocity.astype(np.float32),
            "candidate_scores": logits.astype(np.float32),
            "topk_indices": top_indices.astype(np.int64),
            "topk_scores": top_scores.astype(np.float32),
            "topk_velocity": candidates[top_indices].astype(np.float32),
            "refined_candidate_index": best_indices,
            "static_cost": static_cost.astype(np.float32),
            "static_improvement": static_improvement.astype(np.float32),
            # [geometry feature extraction, candidate scoring, local
            # refinement/post-processing, total], in milliseconds.  These
            # values are diagnostic and do not change the estimator API's
            # beam-level outputs.
            "timing_ms": np.asarray([geometry_ms, scorer_ms, refinement_ms, total_ms], dtype=np.float32),
        }

    def __call__(self, lidar_history, odom_history, timestamps) -> Dict[str, torch.Tensor]:
        lidar, odom, times = _as_batch(lidar_history, odom_history, timestamps)
        outputs = [self._estimate_one(lidar[b], odom[b], times[b]) for b in range(len(lidar))]
        keys = outputs[0].keys()
        result = {}
        for key in keys:
            result[key] = torch.from_numpy(np.stack([out[key] for out in outputs], axis=0)).to(self.device)
        return result
