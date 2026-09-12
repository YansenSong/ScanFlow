"""
model.py

Dynamic 2D LiDAR motion-field network.

Pipeline
--------
Historical 2D LiDAR + odometry
    -> SE(2) ego-motion scan warping
    -> current-frame polar reprojection
    -> LiCS-style patch tokenization
    -> local cross-frame attention
    -> residual motion tokens
    -> temporal Transformer
    -> detection-free token motion field

Expected inputs
---------------
lidar_history : [B, K, H]
    Range scans ordered from oldest -> current.
odom_history  : [B, K, 3]
    Robot poses [x, y, yaw] in ONE common/world odometry frame,
    synchronized to each LiDAR timestamp.

Default configuration
---------------------
K = 6 historical/current frames
H = 720 beams per scan
P = 36 angular patches
D = 128 token dimension

The network outputs one motion token per angular patch:
    anchor     : [B, P, 2]  -> current-frame (x, y)
    velocity   : [B, P, 2]  -> estimated (vx, vy)
    confidence : [B, P, 1]  -> dynamic/motion confidence

Notes
-----
1. SE(2) scan warping is geometric preprocessing, not a learnable layer.
2. Reprojection uses the nearest transformed point per angular bin.
3. This file intentionally stops at the motion field. NMPC should be kept
   in a separate planner/controller module (e.g. CasADi/acados).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # Input/history
    num_frames: int = 6          # K
    num_beams: int = 720         # H
    num_patches: int = 36        # P

    # LiDAR geometry
    angle_min: float = -math.pi
    angle_increment: Optional[float] = None  # default: 2*pi / H
    full_circle: bool = True
    range_min: float = 0.05
    range_max: float = 10.0

    # Token dimensions
    patch_hidden_dim: int = 64
    d_model: int = 128

    # Cross-frame attention
    cross_heads: int = 4
    local_patch_radius: int = 2
    cross_dropout: float = 0.1

    # Temporal transformer
    temporal_heads: int = 4
    temporal_layers: int = 3
    temporal_ff_dim: int = 256
    temporal_dropout: float = 0.1

    # Output
    motion_hidden_dim: int = 64
    max_motion_speed: float = 3.0

    def __post_init__(self) -> None:
        if self.num_beams % self.num_patches != 0:
            raise ValueError(
                f"num_beams ({self.num_beams}) must be divisible by "
                f"num_patches ({self.num_patches})."
            )
        if self.d_model % self.cross_heads != 0:
            raise ValueError("d_model must be divisible by cross_heads.")
        if self.d_model % self.temporal_heads != 0:
            raise ValueError("d_model must be divisible by temporal_heads.")
        if self.angle_increment is None:
            self.angle_increment = 2.0 * math.pi / self.num_beams


# ---------------------------------------------------------------------------
# 1. Geometric scan warping
# ---------------------------------------------------------------------------

class ScanWarpingSE2(nn.Module):
    """
    Warp every historical scan into the CURRENT robot coordinate frame.

    Input
    -----
    lidar : [B, K, H]
    odom  : [B, K, 3]  (x, y, yaw) in a common odometry/world frame

    Output
    ------
    aligned_range : [B, K, H]
    valid_mask    : [B, K, H] bool

    The transform is:
        p_world = T_world_robot(k) * p_robot(k)
        p_now   = inv(T_world_robot(t)) * p_world

    Then transformed Cartesian points are reprojected onto the current-frame
    fixed polar grid. If multiple points land in the same beam, the nearest
    point is retained.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_beams = cfg.num_beams
        self.angle_min = float(cfg.angle_min)
        self.angle_increment = float(cfg.angle_increment)
        self.full_circle = bool(cfg.full_circle)
        self.range_min = float(cfg.range_min)
        self.range_max = float(cfg.range_max)

        angles = (
            self.angle_min
            + torch.arange(self.num_beams, dtype=torch.float32)
            * self.angle_increment
        )
        self.register_buffer("beam_angles", angles, persistent=False)

    def _check_shapes(
        self, lidar: torch.Tensor, odom: torch.Tensor
    ) -> Tuple[int, int, int]:
        if lidar.ndim != 3:
            raise ValueError(f"lidar must be [B,K,H], got {tuple(lidar.shape)}")
        if odom.ndim != 3 or odom.shape[-1] != 3:
            raise ValueError(f"odom must be [B,K,3], got {tuple(odom.shape)}")

        B, K, H = lidar.shape
        if H != self.num_beams:
            raise ValueError(f"Expected H={self.num_beams}, got H={H}")
        if odom.shape[:2] != (B, K):
            raise ValueError("lidar and odom must have identical [B,K].")
        return B, K, H

    def forward(
        self,
        lidar: torch.Tensor,
        odom: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, H = self._check_shapes(lidar, odom)

        dtype = lidar.dtype
        device = lidar.device
        angles = self.beam_angles.to(device=device, dtype=dtype)

        # Basic raw-scan validity.
        raw_valid = (
            torch.isfinite(lidar)
            & (lidar >= self.range_min)
            & (lidar <= self.range_max)
        )
        safe_range = torch.nan_to_num(
            lidar,
            nan=self.range_max,
            posinf=self.range_max,
            neginf=self.range_min,
        ).clamp(self.range_min, self.range_max)

        # Polar -> Cartesian in each historical robot frame.
        c = torch.cos(angles).view(1, 1, H)
        s = torch.sin(angles).view(1, 1, H)
        x_local = safe_range * c
        y_local = safe_range * s

        # Robot pose at each historical frame.
        tx = odom[..., 0].unsqueeze(-1)   # [B,K,1]
        ty = odom[..., 1].unsqueeze(-1)
        yaw = odom[..., 2].unsqueeze(-1)
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)

        # Historical robot frame -> world frame.
        x_world = cy * x_local - sy * y_local + tx
        y_world = sy * x_local + cy * y_local + ty

        # World frame -> CURRENT robot frame.
        tx_t = odom[:, -1, 0].view(B, 1, 1)
        ty_t = odom[:, -1, 1].view(B, 1, 1)
        yaw_t = odom[:, -1, 2].view(B, 1, 1)
        ct = torch.cos(yaw_t)
        st = torch.sin(yaw_t)

        dx = x_world - tx_t
        dy = y_world - ty_t

        # R(-yaw_t)
        x_now = ct * dx + st * dy
        y_now = -st * dx + ct * dy

        r_now = torch.sqrt(x_now.square() + y_now.square()).clamp_min(1e-8)
        a_now = torch.atan2(y_now, x_now)

        # Cartesian -> current-frame angular bins.
        bin_float = (a_now - self.angle_min) / self.angle_increment
        bin_idx = torch.floor(bin_float + 0.5).long()

        valid = raw_valid & torch.isfinite(r_now)
        valid &= (r_now >= self.range_min) & (r_now <= self.range_max)

        if self.full_circle:
            bin_idx = torch.remainder(bin_idx, H)
        else:
            valid &= (bin_idx >= 0) & (bin_idx < H)
            bin_idx = bin_idx.clamp(0, H - 1)

        # Scatter nearest transformed point into each angular bin.
        BK = B * K
        idx = bin_idx.reshape(BK, H)
        src = r_now.reshape(BK, H)
        vld = valid.reshape(BK, H)

        inf = torch.tensor(float("inf"), dtype=dtype, device=device)
        src = torch.where(vld, src, inf)

        aligned = torch.full(
            (BK, H),
            float("inf"),
            dtype=dtype,
            device=device,
        )

        # torch.scatter_reduce_ is available in modern PyTorch.
        if not hasattr(aligned, "scatter_reduce_"):
            raise RuntimeError(
                "This implementation requires PyTorch with Tensor.scatter_reduce_. "
                "Please use PyTorch >= 1.12/2.x."
            )

        aligned.scatter_reduce_(
            dim=1,
            index=idx,
            src=src,
            reduce="amin",
            include_self=True,
        )

        valid_out = torch.isfinite(aligned)
        aligned = torch.where(
            valid_out,
            aligned,
            torch.full_like(aligned, self.range_max),
        )

        aligned = aligned.view(B, K, H)
        valid_out = valid_out.view(B, K, H)

        return aligned, valid_out


# ---------------------------------------------------------------------------
# 2. LiCS-style spatial patch tokenizer
# ---------------------------------------------------------------------------

class LiDARPatchTokenizer(nn.Module):
    """
    LiCS-style angular patch embedding adapted to aligned range + validity mask.

    Inputs
    ------
    ranges : [B,K,H]
    mask   : [B,K,H]

    Output
    ------
    tokens      : [B,K,P,D]
    patch_valid : [B,K,P] bool
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_beams = cfg.num_beams
        self.num_patches = cfg.num_patches
        self.patch_size = cfg.num_beams // cfg.num_patches
        self.range_max = float(cfg.range_max)

        in_dim = self.patch_size * 2  # normalized ranges + validity bits

        self.embedding = nn.Sequential(
            nn.Linear(in_dim, cfg.patch_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.patch_hidden_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

    def forward(
        self,
        ranges: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if ranges.shape != mask.shape:
            raise ValueError("ranges and mask must have identical shape.")
        if ranges.ndim != 3:
            raise ValueError("ranges must be [B,K,H].")

        B, K, H = ranges.shape
        if H != self.num_beams:
            raise ValueError(f"Expected H={self.num_beams}, got {H}")

        # Normalize to approximately [0,1].
        x = torch.nan_to_num(
            ranges,
            nan=self.range_max,
            posinf=self.range_max,
            neginf=0.0,
        ).clamp(0.0, self.range_max)
        x = x / self.range_max

        # [B,K,H] -> [B,K,P,W]
        x = x.view(B, K, self.num_patches, self.patch_size)
        m = mask.to(dtype=x.dtype).view(
            B, K, self.num_patches, self.patch_size
        )

        # [range samples | validity samples]
        patch_input = torch.cat([x, m], dim=-1)  # [B,K,P,2W]
        tokens = self.embedding(patch_input)      # [B,K,P,D]

        patch_valid = m.bool().any(dim=-1)        # [B,K,P]
        return tokens, patch_valid


# ---------------------------------------------------------------------------
# 3. Spatial + temporal embeddings
# ---------------------------------------------------------------------------

class SpatioTemporalEmbedding(nn.Module):
    """
    Add learnable angular-patch and frame-index embeddings.

    Input/output: [B,K,P,D]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.d_model
        self.num_frames = cfg.num_frames
        self.num_patches = cfg.num_patches

        self.patch_embedding = nn.Parameter(
            torch.randn(1, 1, self.num_patches, D) * 0.02
        )
        self.time_embedding = nn.Parameter(
            torch.randn(1, self.num_frames, 1, D) * 0.02
        )
        self.norm = nn.LayerNorm(D)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError("tokens must be [B,K,P,D].")

        _, K, P, _ = tokens.shape
        if K > self.num_frames:
            raise ValueError(
                f"Got K={K}, but configured num_frames={self.num_frames}"
            )
        if P != self.num_patches:
            raise ValueError(
                f"Got P={P}, expected num_patches={self.num_patches}"
            )

        x = (
            tokens
            + self.patch_embedding[:, :, :P]
            + self.time_embedding[:, :K]
        )
        return self.norm(x)


# ---------------------------------------------------------------------------
# 4. Local cross-frame attention
# ---------------------------------------------------------------------------

def build_local_attention_mask(
    num_patches: int,
    radius: int,
    circular: bool = True,
) -> torch.Tensor:
    """
    Return bool attention mask [P,P].
    True = disallowed attention edge for nn.MultiheadAttention.
    """
    idx = torch.arange(num_patches)
    q = idx[:, None]
    k = idx[None, :]

    distance = (q - k).abs()
    if circular:
        distance = torch.minimum(distance, num_patches - distance)

    allowed = distance <= radius
    return ~allowed


class LocalCrossFrameAttention(nn.Module):
    """
    Match CURRENT tokens against each historical frame using local angular
    cross-attention.

    Inputs
    ------
    current : [B,P,D]
    history : [B,Kh,P,D], Kh=K-1

    Output
    ------
    matched_history : [B,Kh,P,D]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_patches = cfg.num_patches
        self.local_radius = cfg.local_patch_radius

        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.cross_heads,
            dropout=cfg.cross_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.cross_dropout),
            nn.Linear(2 * cfg.d_model, cfg.d_model),
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)

        local_mask = build_local_attention_mask(
            num_patches=cfg.num_patches,
            radius=cfg.local_patch_radius,
            circular=cfg.full_circle,
        )
        self.register_buffer(
            "local_attn_mask",
            local_mask,
            persistent=False,
        )

    def forward(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        if current.ndim != 3:
            raise ValueError("current must be [B,P,D].")
        if history.ndim != 4:
            raise ValueError("history must be [B,Kh,P,D].")

        B, P, D = current.shape
        Bh, Kh, Ph, Dh = history.shape
        if (Bh, Ph, Dh) != (B, P, D):
            raise ValueError("current/history shapes are incompatible.")

        # Treat every historical frame as a separate batch item.
        q = current.unsqueeze(1).expand(-1, Kh, -1, -1)
        q = q.reshape(B * Kh, P, D)
        kv = history.reshape(B * Kh, P, D)

        attn_out, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            attn_mask=self.local_attn_mask,
            need_weights=False,
        )

        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ffn(x))

        return x.view(B, Kh, P, D)


# ---------------------------------------------------------------------------
# 5. Residual motion token encoder
# ---------------------------------------------------------------------------

class ResidualMotionEncoder(nn.Module):
    """
    Construct residual motion features from current and matched-history tokens.

    For each history frame:
        [cur, matched, cur-matched, |cur-matched|] -> MLP

    Inputs
    ------
    current : [B,P,D]
    matched : [B,Kh,P,D]

    Output
    ------
    motion_tokens : [B,Kh,P,D]
    """

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

    def forward(
        self,
        current: torch.Tensor,
        matched: torch.Tensor,
    ) -> torch.Tensor:
        B, Kh, P, D = matched.shape
        cur = current.unsqueeze(1).expand(-1, Kh, -1, -1)

        feat = torch.cat(
            [
                cur,
                matched,
                cur - matched,
                (cur - matched).abs(),
            ],
            dim=-1,
        )  # [B,Kh,P,4D]

        return self.mlp(feat)


# ---------------------------------------------------------------------------
# 6. Temporal Transformer
# ---------------------------------------------------------------------------

class DynamicTemporalTransformer(nn.Module):
    """
    Temporal modeling is performed INDEPENDENTLY for each angular patch.

    Input
    -----
    tokens : [B,K,P,D]

    Internally:
        [B,K,P,D] -> [B,P,K,D] -> [B*P,K,D]

    Output
    ------
    [B,K,P,D]
    """

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
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=cfg.temporal_layers,
            norm=nn.LayerNorm(cfg.d_model),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError("tokens must be [B,K,P,D].")

        B, K, P, D = tokens.shape

        # Every spatial patch gets its own temporal sequence.
        x = tokens.permute(0, 2, 1, 3).contiguous()  # [B,P,K,D]
        x = x.view(B * P, K, D)                     # [B*P,K,D]

        x = self.encoder(x)

        x = x.view(B, P, K, D)
        x = x.permute(0, 2, 1, 3).contiguous()      # [B,K,P,D]
        return x


# ---------------------------------------------------------------------------
# 7. Temporal pooling
# ---------------------------------------------------------------------------

class TemporalTokenPooling(nn.Module):
    """
    Learned attention pooling over the K temporal features of each patch.

    Input  : [B,K,P,D]
    Output : [B,P,D]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.d_model
        self.score = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.Tanh(),
            nn.Linear(D // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        frame_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [B,K,P,D]
        logits = self.score(x).squeeze(-1)  # [B,K,P]

        if frame_valid is not None:
            if frame_valid.shape != logits.shape:
                raise ValueError(
                    "frame_valid must have shape [B,K,P]."
                )
            # Avoid all -inf if every frame is invalid for a patch.
            any_valid = frame_valid.any(dim=1, keepdim=True)  # [B,1,P]
            safe_valid = torch.where(
                any_valid,
                frame_valid,
                torch.ones_like(frame_valid),
            )
            logits = logits.masked_fill(~safe_valid, -1e4)

        weights = F.softmax(logits, dim=1)  # [B,K,P]
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        return pooled


# ---------------------------------------------------------------------------
# 8. Motion-field regression head
# ---------------------------------------------------------------------------

class MotionFieldHead(nn.Module):
    """
    Input
    -----
    dynamic_features : [B,P,D]

    Output
    ------
    velocity   : [B,P,2]
    confidence : [B,P,1]
    """

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

    def forward(
        self,
        dynamic_features: torch.Tensor,
        current_patch_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(dynamic_features)

        velocity = (
            torch.tanh(self.velocity_head(h))
            * self.max_motion_speed
        )
        confidence = torch.sigmoid(self.confidence_head(h))

        if current_patch_valid is not None:
            confidence = confidence * current_patch_valid.unsqueeze(-1).to(
                confidence.dtype
            )

        return velocity, confidence


# ---------------------------------------------------------------------------
# 9. Geometric patch anchors
# ---------------------------------------------------------------------------

class PatchAnchorExtractor(nn.Module):
    """
    Compute one current-frame Cartesian anchor per angular patch.

    Anchor = centroid of valid points in the patch.

    Inputs
    ------
    current_range : [B,H]
    current_mask  : [B,H]

    Output
    ------
    anchors     : [B,P,2]
    patch_valid : [B,P] bool
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_beams = cfg.num_beams
        self.num_patches = cfg.num_patches
        self.patch_size = cfg.num_beams // cfg.num_patches

        angles = (
            cfg.angle_min
            + torch.arange(cfg.num_beams, dtype=torch.float32)
            * float(cfg.angle_increment)
        )
        self.register_buffer("beam_angles", angles, persistent=False)

    def forward(
        self,
        current_range: torch.Tensor,
        current_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if current_range.ndim != 2:
            raise ValueError("current_range must be [B,H].")

        B, H = current_range.shape
        if H != self.num_beams:
            raise ValueError(f"Expected H={self.num_beams}, got {H}")

        angles = self.beam_angles.to(
            device=current_range.device,
            dtype=current_range.dtype,
        )

        x = current_range * torch.cos(angles).view(1, H)
        y = current_range * torch.sin(angles).view(1, H)

        points = torch.stack([x, y], dim=-1)  # [B,H,2]
        points = points.view(
            B, self.num_patches, self.patch_size, 2
        )

        m = current_mask.view(
            B, self.num_patches, self.patch_size
        )
        m_float = m.unsqueeze(-1).to(points.dtype)

        count = m_float.sum(dim=2).clamp_min(1.0)   # [B,P,1]
        anchors = (points * m_float).sum(dim=2) / count

        patch_valid = m.any(dim=-1)
        anchors = anchors * patch_valid.unsqueeze(-1).to(anchors.dtype)

        return anchors, patch_valid


# ---------------------------------------------------------------------------
# 10. Full model
# ---------------------------------------------------------------------------

class DynamicLiDARNetwork(nn.Module):
    """
    Full neural perception model.

    Inputs
    ------
    lidar_history : [B,K,H]
    odom_history  : [B,K,3]

    Outputs
    -------
    dict:
        anchors          [B,P,2]
        velocity         [B,P,2]
        confidence       [B,P,1]
        motion_field     [B,P,5] = [x,y,vx,vy,confidence]
        current_valid    [B,P]
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else ModelConfig()

        self.warp = ScanWarpingSE2(self.cfg)
        self.tokenizer = LiDARPatchTokenizer(self.cfg)
        self.embedding = SpatioTemporalEmbedding(self.cfg)
        self.cross_frame = LocalCrossFrameAttention(self.cfg)
        self.motion_encoder = ResidualMotionEncoder(self.cfg)
        self.temporal = DynamicTemporalTransformer(self.cfg)
        self.pool = TemporalTokenPooling(self.cfg)
        self.motion_head = MotionFieldHead(self.cfg)
        self.anchor_extractor = PatchAnchorExtractor(self.cfg)

    def forward(
        self,
        lidar_history: torch.Tensor,
        odom_history: torch.Tensor,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        lidar_history : [B,K,H]
        odom_history  : [B,K,3]
        """
        if lidar_history.shape[1] != self.cfg.num_frames:
            raise ValueError(
                f"Expected K={self.cfg.num_frames}, "
                f"got K={lidar_history.shape[1]}"
            )

        # ---------------------------------------------------------------
        # 1) Ego-motion compensation + reprojection
        # ---------------------------------------------------------------
        aligned_range, beam_valid = self.warp(
            lidar_history, odom_history
        )
        # aligned_range, beam_valid: [B,K,H]

        # ---------------------------------------------------------------
        # 2) LiCS-style angular patch tokenization
        # ---------------------------------------------------------------
        tokens, patch_valid = self.tokenizer(
            aligned_range, beam_valid
        )
        # tokens:      [B,K,P,D]
        # patch_valid: [B,K,P]

        tokens = self.embedding(tokens)

        # ---------------------------------------------------------------
        # 3) Cross-frame matching
        # ---------------------------------------------------------------
        current = tokens[:, -1]     # [B,P,D]
        history = tokens[:, :-1]    # [B,K-1,P,D]

        matched = self.cross_frame(
            current=current,
            history=history,
        )                            # [B,K-1,P,D]

        # ---------------------------------------------------------------
        # 4) Residual motion token construction
        # ---------------------------------------------------------------
        residual_motion = self.motion_encoder(
            current=current,
            matched=matched,
        )                            # [B,K-1,P,D]

        # We append the current spatial token as the last temporal context.
        temporal_input = torch.cat(
            [residual_motion, current.unsqueeze(1)],
            dim=1,
        )                            # [B,K,P,D]

        temporal_valid = torch.cat(
            [patch_valid[:, :-1], patch_valid[:, -1:].clone()],
            dim=1,
        )                            # [B,K,P]

        # ---------------------------------------------------------------
        # 5) Temporal dynamics
        # ---------------------------------------------------------------
        dynamic_tokens = self.temporal(
            temporal_input
        )                            # [B,K,P,D]

        dynamic_features = self.pool(
            dynamic_tokens,
            frame_valid=temporal_valid,
        )                            # [B,P,D]

        # ---------------------------------------------------------------
        # 6) Geometric anchors in current frame
        # ---------------------------------------------------------------
        anchors, current_patch_valid = self.anchor_extractor(
            aligned_range[:, -1],
            beam_valid[:, -1],
        )                            # [B,P,2], [B,P]

        # ---------------------------------------------------------------
        # 7) Motion field
        # ---------------------------------------------------------------
        velocity, confidence = self.motion_head(
            dynamic_features,
            current_patch_valid=current_patch_valid,
        )

        motion_field = torch.cat(
            [anchors, velocity, confidence],
            dim=-1,
        )                            # [B,P,5]

        out: Dict[str, torch.Tensor] = {
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
                    "residual_motion_tokens": residual_motion,
                    "dynamic_tokens": dynamic_tokens,
                    "dynamic_features": dynamic_features,
                }
            )

        return out


# ---------------------------------------------------------------------------
# Minimal smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)

    cfg = ModelConfig(
        num_frames=6,
        num_beams=720,
        num_patches=36,
        d_model=128,
    )

    model = DynamicLiDARNetwork(cfg)

    B, K, H = 2, cfg.num_frames, cfg.num_beams

    # Synthetic scans.
    lidar = torch.rand(B, K, H) * 7.0 + 0.5

    # Synthetic odometry: robot moves forward slightly over the history.
    odom = torch.zeros(B, K, 3)
    odom[:, :, 0] = torch.linspace(0.0, 0.5, K)
    odom[:, :, 2] = torch.linspace(0.0, 0.10, K)

    with torch.no_grad():
        out = model(lidar, odom, return_aux=True)

    expected = {
        "anchors": (B, cfg.num_patches, 2),
        "velocity": (B, cfg.num_patches, 2),
        "confidence": (B, cfg.num_patches, 1),
        "motion_field": (B, cfg.num_patches, 5),
    }

    for key, shape in expected.items():
        assert tuple(out[key].shape) == shape, (
            key,
            out[key].shape,
            shape,
        )

    print("Smoke test passed.")
    for key in ("anchors", "velocity", "confidence", "motion_field"):
        print(f"{key:16s}: {tuple(out[key].shape)}")


if __name__ == "__main__":
    _smoke_test()
