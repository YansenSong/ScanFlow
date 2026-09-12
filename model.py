"""
ScanFlow dynamic 2D LiDAR motion-field network.

Historical 2D LiDAR + odometry
    -> SE(2) ego-motion scan warping
    -> current-frame polar reprojection
    -> LiDAR angular patch tokenization
    -> validity-aware local cross-frame matching
    -> residual motion tokens
    -> per-patch temporal Transformer
    -> detection-free token motion field

Inputs
------
lidar_history : [B, K, H], oldest -> current
odom_history  : [B, K, 3], robot [x, y, yaw] in one common odometry frame

Outputs
-------
anchors      : [B, P, 2] current-frame geometric anchors
velocity     : [B, P, 2] obstacle/world motion expressed in current robot frame
confidence   : [B, P, 1] probability that the current patch contains dynamic motion
motion_field : [B, P, 5] [x, y, vx, vy, confidence]
current_valid: [B, P] current LiDAR occupancy/anchor validity
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    num_frames: int = 6
    num_beams: int = 720
    num_patches: int = 36

    angle_min: float = -math.pi
    angle_increment: Optional[float] = None
    full_circle: bool = True
    range_min: float = 0.05
    range_max: float = 10.0

    patch_hidden_dim: int = 64
    d_model: int = 128

    cross_heads: int = 4
    local_patch_radius: int = 2
    cross_dropout: float = 0.1

    temporal_heads: int = 4
    temporal_layers: int = 3
    temporal_ff_dim: int = 256
    temporal_dropout: float = 0.1

    motion_hidden_dim: int = 64
    max_motion_speed: float = 3.0

    def __post_init__(self) -> None:
        if self.num_frames < 2:
            raise ValueError("num_frames must be >= 2 for motion estimation.")
        if self.num_beams % self.num_patches != 0:
            raise ValueError("num_beams must be divisible by num_patches.")
        if self.d_model % self.cross_heads != 0:
            raise ValueError("d_model must be divisible by cross_heads.")
        if self.d_model % self.temporal_heads != 0:
            raise ValueError("d_model must be divisible by temporal_heads.")
        if self.angle_increment is None:
            self.angle_increment = 2.0 * math.pi / self.num_beams


class ScanWarpingSE2(nn.Module):
    """Warp historical scans into the current robot frame and reproject to beams."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_beams = cfg.num_beams
        self.angle_min = float(cfg.angle_min)
        self.angle_increment = float(cfg.angle_increment)
        self.full_circle = bool(cfg.full_circle)
        self.range_min = float(cfg.range_min)
        self.range_max = float(cfg.range_max)
        angles = self.angle_min + torch.arange(cfg.num_beams, dtype=torch.float32) * self.angle_increment
        self.register_buffer("beam_angles", angles, persistent=False)

    def forward(self, lidar: torch.Tensor, odom: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if lidar.ndim != 3:
            raise ValueError(f"lidar must be [B,K,H], got {tuple(lidar.shape)}")
        if odom.ndim != 3 or odom.shape[-1] != 3:
            raise ValueError(f"odom must be [B,K,3], got {tuple(odom.shape)}")
        B, K, H = lidar.shape
        if H != self.num_beams or odom.shape[:2] != (B, K):
            raise ValueError("lidar/odom shapes are incompatible with the configured geometry.")

        dtype, device = lidar.dtype, lidar.device
        angles = self.beam_angles.to(device=device, dtype=dtype)
        raw_valid = torch.isfinite(lidar) & (lidar >= self.range_min) & (lidar <= self.range_max)
        safe_range = torch.nan_to_num(
            lidar, nan=self.range_max, posinf=self.range_max, neginf=self.range_min
        ).clamp(self.range_min, self.range_max)

        c = torch.cos(angles).view(1, 1, H)
        s = torch.sin(angles).view(1, 1, H)
        x_local, y_local = safe_range * c, safe_range * s

        tx, ty = odom[..., 0].unsqueeze(-1), odom[..., 1].unsqueeze(-1)
        yaw = odom[..., 2].unsqueeze(-1)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        x_world = cy * x_local - sy * y_local + tx
        y_world = sy * x_local + cy * y_local + ty

        tx_t = odom[:, -1, 0].view(B, 1, 1)
        ty_t = odom[:, -1, 1].view(B, 1, 1)
        yaw_t = odom[:, -1, 2].view(B, 1, 1)
        ct, st = torch.cos(yaw_t), torch.sin(yaw_t)
        dx, dy = x_world - tx_t, y_world - ty_t
        x_now = ct * dx + st * dy
        y_now = -st * dx + ct * dy

        r_now = torch.sqrt(x_now.square() + y_now.square()).clamp_min(1e-8)
        a_now = torch.atan2(y_now, x_now)
        bin_idx = torch.floor((a_now - self.angle_min) / self.angle_increment + 0.5).long()

        valid = raw_valid & torch.isfinite(r_now)
        valid &= (r_now >= self.range_min) & (r_now <= self.range_max)
        if self.full_circle:
            bin_idx = torch.remainder(bin_idx, H)
        else:
            valid &= (bin_idx >= 0) & (bin_idx < H)
            bin_idx = bin_idx.clamp(0, H - 1)

        BK = B * K
        idx = bin_idx.reshape(BK, H)
        src = r_now.reshape(BK, H)
        vld = valid.reshape(BK, H)
        src = torch.where(vld, src, torch.full_like(src, float("inf")))
        aligned = torch.full((BK, H), float("inf"), dtype=dtype, device=device)
        if not hasattr(aligned, "scatter_reduce_"):
            raise RuntimeError("ScanFlow requires PyTorch with Tensor.scatter_reduce_.")
        aligned.scatter_reduce_(1, idx, src, reduce="amin", include_self=True)

        valid_out = torch.isfinite(aligned)
        aligned = torch.where(valid_out, aligned, torch.full_like(aligned, self.range_max))
        return aligned.view(B, K, H), valid_out.view(B, K, H)


class LiDARPatchTokenizer(nn.Module):
    """Embed contiguous angular range patches together with reprojection validity bits."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_beams = cfg.num_beams
        self.num_patches = cfg.num_patches
        self.patch_size = cfg.num_beams // cfg.num_patches
        self.range_max = float(cfg.range_max)
        self.embedding = nn.Sequential(
            nn.Linear(2 * self.patch_size, cfg.patch_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.patch_hidden_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

    def forward(self, ranges: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if ranges.shape != mask.shape or ranges.ndim != 3:
            raise ValueError("ranges/mask must have identical [B,K,H] shapes.")
        B, K, H = ranges.shape
        if H != self.num_beams:
            raise ValueError(f"Expected {self.num_beams} beams, got {H}.")
        x = torch.nan_to_num(ranges, nan=self.range_max, posinf=self.range_max, neginf=0.0)
        x = x.clamp(0.0, self.range_max) / self.range_max
        x = x.view(B, K, self.num_patches, self.patch_size)
        m = mask.to(x.dtype).view(B, K, self.num_patches, self.patch_size)
        tokens = self.embedding(torch.cat([x, m], dim=-1))
        return tokens, m.bool().any(dim=-1)


class SpatialPatchEmbedding(nn.Module):
    """Add only angular position before cross-frame matching."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_patches = cfg.num_patches
        self.patch_embedding = nn.Parameter(torch.randn(1, 1, cfg.num_patches, cfg.d_model) * 0.02)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4 or tokens.shape[2] != self.num_patches:
            raise ValueError("tokens must be [B,K,P,D] with configured P.")
        return self.norm(tokens + self.patch_embedding)


def build_local_attention_mask(num_patches: int, radius: int, circular: bool = True) -> torch.Tensor:
    idx = torch.arange(num_patches)
    distance = (idx[:, None] - idx[None, :]).abs()
    if circular:
        distance = torch.minimum(distance, num_patches - distance)
    return distance > radius


class LocalCrossFrameAttention(nn.Module):
    """Validity-aware local matching from current patches to each historical frame."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_patches = cfg.num_patches
        self.num_heads = cfg.cross_heads
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.cross_heads, dropout=cfg.cross_dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.cross_dropout),
            nn.Linear(2 * cfg.d_model, cfg.d_model),
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.register_buffer(
            "local_attn_mask",
            build_local_attention_mask(cfg.num_patches, cfg.local_patch_radius, cfg.full_circle),
            persistent=False,
        )

    def forward(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        current_valid: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, P, D = current.shape
        if history.ndim != 4:
            raise ValueError("history must be [B,Kh,P,D].")
        Bh, Kh, Ph, Dh = history.shape
        if (Bh, Ph, Dh) != (B, P, D):
            raise ValueError("current/history shapes are incompatible.")
        if current_valid.shape != (B, P) or history_valid.shape != (B, Kh, P):
            raise ValueError("current_valid/history_valid shapes are incompatible.")

        q = current[:, None].expand(-1, Kh, -1, -1).reshape(B * Kh, P, D)
        kv = history.reshape(B * Kh, P, D)
        hv = history_valid.reshape(B * Kh, P)

        disallowed = self.local_attn_mask[None, :, :] | (~hv[:, None, :])
        matched_valid = (~disallowed).any(dim=-1)

        if (~matched_valid).any():
            bad_b, bad_q = (~matched_valid).nonzero(as_tuple=True)
            disallowed[bad_b, bad_q, bad_q] = False

        attn_mask = disallowed[:, None].expand(-1, self.num_heads, -1, -1)
        attn_mask = attn_mask.reshape(B * Kh * self.num_heads, P, P)
        attn_out, _ = self.attn(q, kv, kv, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ffn(x))

        cv = current_valid[:, None].expand(-1, Kh, -1).reshape(B * Kh, P)
        matched_valid &= cv
        x = torch.where(matched_valid[..., None], x, q)
        return x.view(B, Kh, P, D), matched_valid.view(B, Kh, P)


class ResidualMotionEncoder(nn.Module):
    """[current, matched, difference, absolute difference] -> residual motion token."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.d_model
        self.mlp = nn.Sequential(
            nn.Linear(4 * D, 2 * D),
            nn.GELU(),
            nn.Dropout(cfg.cross_dropout),
            nn.Linear(2 * D, D),
            nn.LayerNorm(D),
        )

    def forward(self, current: torch.Tensor, matched: torch.Tensor) -> torch.Tensor:
        Kh = matched.shape[1]
        cur = current[:, None].expand(-1, Kh, -1, -1)
        diff = cur - matched
        return self.mlp(torch.cat([cur, matched, diff, diff.abs()], dim=-1))


class TemporalLagEmbedding(nn.Module):
    """Add history-lag identity only after residual motion construction."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_history = cfg.num_frames - 1
        self.embedding = nn.Parameter(torch.randn(1, self.num_history, 1, cfg.d_model) * 0.02)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.num_history:
            raise ValueError("Too many historical residual frames.")
        return self.norm(x + self.embedding[:, : x.shape[1]])


class DynamicTemporalTransformer(nn.Module):
    """Temporal attention independently for every spatial patch."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.temporal_heads,
            dim_feedforward=cfg.temporal_ff_dim,
            dropout=cfg.temporal_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.temporal_layers, norm=nn.LayerNorm(cfg.d_model))

    def forward(self, tokens: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, P, D = tokens.shape
        x = tokens.permute(0, 2, 1, 3).contiguous().view(B * P, T, D)
        key_padding = None
        all_invalid = None
        if valid is not None:
            vm = valid.permute(0, 2, 1).contiguous().view(B * P, T)
            all_invalid = ~vm.any(dim=1)
            safe_vm = vm.clone()
            safe_vm[all_invalid, -1] = True
            key_padding = ~safe_vm
        x = self.encoder(x, src_key_padding_mask=key_padding)
        if all_invalid is not None and all_invalid.any():
            x[all_invalid] = 0.0
        return x.view(B, P, T, D).permute(0, 2, 1, 3).contiguous()


class TemporalTokenPooling(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.d_model
        self.score = nn.Sequential(nn.Linear(D, D // 2), nn.Tanh(), nn.Linear(D // 2, 1))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)
        logits = logits.masked_fill(~valid, -1e4)
        weights = F.softmax(logits, dim=1) * valid.to(x.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (x * weights[..., None]).sum(dim=1)


class DynamicFeatureFusion(nn.Module):
    """Fuse temporal motion evidence with the current spatial appearance token."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.d_model
        self.net = nn.Sequential(
            nn.Linear(2 * D, D), nn.GELU(), nn.Dropout(cfg.temporal_dropout), nn.LayerNorm(D)
        )

    def forward(self, motion: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([motion, current], dim=-1))


class MotionFieldHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.max_motion_speed = float(cfg.max_motion_speed)
        self.backbone = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.motion_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.motion_hidden_dim, cfg.motion_hidden_dim),
            nn.GELU(),
        )
        self.velocity_head = nn.Linear(cfg.motion_hidden_dim, 2)
        self.confidence_head = nn.Linear(cfg.motion_hidden_dim, 1)

    def forward(self, features: torch.Tensor, current_valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(features)
        velocity = torch.tanh(self.velocity_head(h)) * self.max_motion_speed
        confidence = torch.sigmoid(self.confidence_head(h))
        confidence = confidence * current_valid[..., None].to(confidence.dtype)
        velocity = velocity * current_valid[..., None].to(velocity.dtype)
        return velocity, confidence


class PatchAnchorExtractor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_beams = cfg.num_beams
        self.num_patches = cfg.num_patches
        self.patch_size = cfg.num_beams // cfg.num_patches
        angles = cfg.angle_min + torch.arange(cfg.num_beams, dtype=torch.float32) * float(cfg.angle_increment)
        self.register_buffer("beam_angles", angles, persistent=False)

    def forward(self, current_range: torch.Tensor, current_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H = current_range.shape
        if H != self.num_beams:
            raise ValueError(f"Expected H={self.num_beams}, got {H}.")
        angles = self.beam_angles.to(current_range.device, current_range.dtype)
        points = torch.stack(
            [current_range * torch.cos(angles)[None], current_range * torch.sin(angles)[None]], dim=-1
        ).view(B, self.num_patches, self.patch_size, 2)
        m = current_mask.view(B, self.num_patches, self.patch_size)
        mf = m[..., None].to(points.dtype)
        count = mf.sum(dim=2).clamp_min(1.0)
        anchors = (points * mf).sum(dim=2) / count
        valid = m.any(dim=-1)
        anchors = anchors * valid[..., None].to(anchors.dtype)
        return anchors, valid


class DynamicLiDARNetwork(nn.Module):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.warp = ScanWarpingSE2(self.cfg)
        self.tokenizer = LiDARPatchTokenizer(self.cfg)
        self.spatial_embedding = SpatialPatchEmbedding(self.cfg)
        self.cross_frame = LocalCrossFrameAttention(self.cfg)
        self.motion_encoder = ResidualMotionEncoder(self.cfg)
        self.lag_embedding = TemporalLagEmbedding(self.cfg)
        self.temporal = DynamicTemporalTransformer(self.cfg)
        self.pool = TemporalTokenPooling(self.cfg)
        self.fusion = DynamicFeatureFusion(self.cfg)
        self.motion_head = MotionFieldHead(self.cfg)
        self.anchor_extractor = PatchAnchorExtractor(self.cfg)

    def forward(
        self,
        lidar_history: torch.Tensor,
        odom_history: torch.Tensor,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if lidar_history.ndim != 3 or lidar_history.shape[1] != self.cfg.num_frames:
            raise ValueError(f"Expected lidar_history [B,{self.cfg.num_frames},H].")

        aligned_range, beam_valid = self.warp(lidar_history, odom_history)
        raw_tokens, patch_valid = self.tokenizer(aligned_range, beam_valid)
        tokens = self.spatial_embedding(raw_tokens)

        current, history = tokens[:, -1], tokens[:, :-1]
        current_valid, history_valid = patch_valid[:, -1], patch_valid[:, :-1]
        matched, matched_valid = self.cross_frame(current, history, current_valid, history_valid)

        residual_motion = self.motion_encoder(current, matched)
        residual_motion = self.lag_embedding(residual_motion)
        dynamic_tokens = self.temporal(residual_motion, matched_valid)
        motion_features = self.pool(dynamic_tokens, matched_valid)
        dynamic_features = self.fusion(motion_features, current)

        anchors, current_patch_valid = self.anchor_extractor(aligned_range[:, -1], beam_valid[:, -1])
        velocity, confidence = self.motion_head(dynamic_features, current_patch_valid)
        motion_field = torch.cat([anchors, velocity, confidence], dim=-1)

        out = {
            "anchors": anchors,
            "velocity": velocity,
            "confidence": confidence,
            "motion_field": motion_field,
            "current_valid": current_patch_valid,
        }
        if return_aux:
            out.update(
                {
                    "aligned_range": aligned_range,
                    "beam_valid": beam_valid,
                    "patch_valid": patch_valid,
                    "tokens": tokens,
                    "matched_history": matched,
                    "matched_valid": matched_valid,
                    "residual_motion_tokens": residual_motion,
                    "dynamic_tokens": dynamic_tokens,
                    "motion_features": motion_features,
                    "dynamic_features": dynamic_features,
                }
            )
        return out


def _smoke_test() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig()
    model = DynamicLiDARNetwork(cfg)
    B, K, H = 2, cfg.num_frames, cfg.num_beams
    lidar = torch.rand(B, K, H) * 7.0 + 0.5
    odom = torch.zeros(B, K, 3)
    odom[:, :, 0] = torch.linspace(0.0, 0.5, K)
    odom[:, :, 2] = torch.linspace(0.0, 0.1, K)
    with torch.no_grad():
        out = model(lidar, odom, return_aux=True)
    assert out["motion_field"].shape == (B, cfg.num_patches, 5)
    assert torch.isfinite(out["motion_field"]).all()
    print("model.py smoke test passed")


if __name__ == "__main__":
    _smoke_test()
