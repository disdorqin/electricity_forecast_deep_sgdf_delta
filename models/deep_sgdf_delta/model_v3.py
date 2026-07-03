"""TrendKnight-X v3 — Multiscale decomposition + teacher residual fusion.

Phase 3.5 model building on V2 (day-level 24h decoder).  Adds:
  - Multiscale decomposition branch (trend / seasonal / shock)
  - Period-aware 2D branch for 1_8 / 9_16 / 17_24 segments
  - Teacher residual fusion gate (optional SGDFNet / RT916 / TimeMixer)
  - Confidence + shock sensitivity auxiliary heads
  - Parameters < 500k with default config (hidden_dim=96)

Architecture overview::

    features_24h ──► FeatureProjection ──► h ──► Backbone ──► h_bb
                                            │                      │
                                            ├──► MultiscaleDecomposer ──► trend / seasonal / shock
                                            │                              │
                                            │          multiscale_delta ◄──┘ (weighted sum)
                                            │
    segment_id ──► PeriodBranch ──► h_bb refined
                                            │
    teacher_features ──► TeacherFusionGate ──► h_fused
                                            │
                                            ├──► DayDecoderHead ──► delta_from_features
                                            ├──► ConfidenceHead ──► confidence (sigmoid)
                                            └──► ShockSensitivityHead ──► shock_sensitivity (sigmoid)

    delta_pred = delta_from_features + multiscale_delta
    rt_pred    = da_anchor + delta_pred
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Literal, Optional


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class TrendKnightV3Config:
    """Model hyperparameters for TrendKnight-X v3."""

    # Input / output
    input_dim: int = 40
    hidden_dim: int = 96
    num_layers: int = 2
    dropout: float = 0.1

    # Backbone
    backbone: Literal["tcn", "gru", "transformer_tiny"] = "tcn"

    # TCN-specific
    tcn_kernel_size: int = 3
    tcn_dilation_base: int = 2

    # Transformer-tiny
    transformer_nhead: int = 4
    transformer_dim_ff: int = 128

    # Multiscale decomposition
    multiscale: bool = True

    # Teacher fusion
    teacher_input_dim: int = 3          # number of teacher models (sgdfnet/rt916/timemixer)
    use_teacher_gate: bool = True

    # Embeddings
    hour_embed_dim: int = 8
    segment_embed_dim: int = 8
    num_hours: int = 25                 # indices 0..24; 0 unused (hours are 1-24)
    num_segments: int = 3

    # AMP
    amp_enabled: bool = False


# ── Reusable building blocks (from V2 with minor adjustments) ────────

class FeatureProjection(nn.Module):
    """Project raw per-hour features to hidden dimension via MLP."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 24, input_dim] -> [B, 24, hidden_dim]"""
        return self.proj(x)


class HourEmbedding(nn.Module):
    """Learnable embedding for hours 1-24 (index 0 is padding/unused)."""

    def __init__(self, num_hours: int = 25, embed_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(num_hours, embed_dim, padding_idx=0)

    def forward(self, hour_ids: torch.Tensor) -> torch.Tensor:
        """hour_ids: [B, 24] (values 1-24) -> [B, 24, embed_dim]"""
        return self.embed(hour_ids.long())


class SegmentEmbedding(nn.Module):
    """Learnable segment embedding (1_8=0, 9_16=1, 17_24=2)."""

    def __init__(self, num_segments: int = 3, embed_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(num_segments, embed_dim)

    def forward(self, segment_ids: torch.Tensor) -> torch.Tensor:
        """segment_ids: [B] -> [B, embed_dim]"""
        return self.embed(segment_ids.long())


# ── TCN blocks ────────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    """Single Temporal Convolutional Network block with causal padding."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=padding)
        self.chomp = padding
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_ch)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.chomp > 0:
            out = out[:, :, :-self.chomp]
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        out = out.permute(0, 2, 1)
        out = self.act(out)
        out = self.dropout(out)
        res = self.residual(x)
        return self.act(out + res)


class TCNBackbone(nn.Module):
    """Temporal Convolutional Network backbone for the 24-step sequence."""

    def __init__(self, hidden_dim: int, num_layers: int, kernel_size: int,
                 dilation_base: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = dilation_base ** i
            layers.append(TCNBlock(hidden_dim, hidden_dim, kernel_size,
                                   dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.network(x)
        return x.permute(0, 2, 1)


# ── GRU backbone ─────────────────────────────────────────────────────

class GRUBackbone(nn.Module):
    """GRU backbone for the 24-step sequence."""

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out


# ── Transformer-tiny backbone ────────────────────────────────────────

class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the 24-hour sequence."""

    def __init__(self, d_model: int, max_len: int = 32, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerTinyBackbone(nn.Module):
    """Lightweight TransformerEncoder backbone (Pre-LN, GELU)."""

    def __init__(self, hidden_dim: int, nhead: int, dim_feedforward: int,
                 num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.pos_enc = _PositionalEncoding(hidden_dim, max_len=32, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        return self.encoder(x)


# ── Multiscale Decomposer ────────────────────────────────────────────

class _ScaleBranch(nn.Module):
    """Single scale branch: Conv1d -> LayerNorm -> GELU -> Linear(1)."""

    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size,
            dilation=dilation, padding=padding,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 24, hidden_dim] -> [B, 24]"""
        out = self.conv(x.permute(0, 2, 1))           # [B, hidden_dim, 24]
        # Trim to original length if needed
        T = x.size(1)
        out = out[:, :, :T]
        out = out.permute(0, 2, 1)                     # [B, 24, hidden_dim]
        out = self.norm(out)
        out = self.act(out)
        return self.proj(out).squeeze(-1)              # [B, 24]


class MultiscaleDecomposer(nn.Module):
    """Decompose the 24h sequence into trend / seasonal / shock components.

    Three parallel Conv1d branches at different temporal scales:
      - trend:    large receptive field (kernel=7, dilation=2) — slow trend
      - seasonal: medium receptive field (kernel=3, dilation=2) — intraday patterns
      - shock:    point-wise (kernel=1, dilation=1) — sudden changes
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.trend_branch = _ScaleBranch(hidden_dim, kernel_size=7, dilation=2)
        self.seasonal_branch = _ScaleBranch(hidden_dim, kernel_size=3, dilation=2)
        self.shock_branch = _ScaleBranch(hidden_dim, kernel_size=1, dilation=1)

        # Learnable blending weights (softmax-normalised)
        self.blend_weights = nn.Parameter(torch.zeros(3))

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """h: [B, 24, hidden_dim]

        Returns dict with:
          trend:    [B, 24]
          seasonal: [B, 24]
          shock:    [B, 24]
          combined: [B, 24]  (softmax-weighted sum)
        """
        trend = self.trend_branch(h)
        seasonal = self.seasonal_branch(h)
        shock = self.shock_branch(h)

        w = F.softmax(self.blend_weights, dim=0)
        combined = w[0] * trend + w[1] * seasonal + w[2] * shock

        return {
            "trend": trend,
            "seasonal": seasonal,
            "shock": shock,
            "combined": combined,
        }


# ── Period Branch ────────────────────────────────────────────────────

class PeriodBranch(nn.Module):
    """Period-aware refinement conditioned on segment_id.

    Uses segment embedding to generate a per-hour gate (sigmoid) and
    a refinement vector that modulate the backbone hidden states.
    """

    def __init__(self, hidden_dim: int, segment_embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(segment_embed_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.refine_proj = nn.Sequential(
            nn.Linear(segment_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, seg_emb: torch.Tensor) -> torch.Tensor:
        """h: [B, 24, hidden_dim], seg_emb: [B, segment_embed_dim]

        Returns: [B, 24, hidden_dim]
        """
        seg_3d = seg_emb.unsqueeze(1)                      # [B, 1, seg_emb]
        gate = self.gate_proj(seg_3d)                      # [B, 1, hidden_dim]
        refine = self.refine_proj(seg_3d)                  # [B, 1, hidden_dim]
        return h * gate + refine


# ── Teacher Fusion Gate ──────────────────────────────────────────────

class TeacherFusionGate(nn.Module):
    """Gate teacher model predictions into the backbone hidden states.

    Each teacher's 24h prediction is projected independently, then a
    learned attention gate decides how much to fuse.  When teacher
    features are absent (all zeros) the gate naturally suppresses them.
    """

    def __init__(self, hidden_dim: int, num_teachers: int, dropout: float = 0.1):
        super().__init__()
        self.num_teachers = num_teachers

        # Per-teacher projection: [B, 24] -> [B, 24, hidden_dim]
        self.teacher_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
            )
            for _ in range(num_teachers)
        ])

        # Attention gate over teachers
        self.gate_attn = nn.Sequential(
            nn.Linear(hidden_dim, num_teachers),
            nn.Softmax(dim=-1),
        )

        # Final fusion with backbone hidden states
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h: torch.Tensor,
        teacher_features: torch.Tensor | None = None,
        teacher_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse teacher predictions into backbone hidden states.

        Args:
            h: [B, 24, hidden_dim] — backbone output
            teacher_features: [B, num_teachers, 24] — per-teacher delta predictions
            teacher_mask: [B, num_teachers] — 1 where teacher is available

        Returns: [B, 24, hidden_dim]
        """
        if teacher_features is None:
            return h

        if teacher_mask is not None and teacher_mask.sum() == 0:
            return h

        B, T, H = h.shape

        # Per-teacher projections
        teacher_reprs = []
        for t_idx in range(self.num_teachers):
            t_feat = teacher_features[:, t_idx, :].unsqueeze(-1)       # [B, 24, 1]
            t_repr = self.teacher_projs[t_idx](t_feat)                 # [B, 24, hidden_dim]
            teacher_reprs.append(t_repr)
        teacher_reprs = torch.stack(teacher_reprs, dim=2)              # [B, 24, num_teachers, H]

        # Attention gate: [B, 1, num_teachers] -> [B, 1, num_teachers, 1]
        h_query = h.mean(dim=1, keepdim=True)                          # [B, 1, H]
        gate = self.gate_attn(h_query).unsqueeze(-1)                    # [B, 1, num_teachers, 1]

        # Apply teacher_mask: zero out unavailable teachers
        if teacher_mask is not None:
            # teacher_mask: [B, num_teachers] -> [B, 1, num_teachers, 1]
            mask = teacher_mask.unsqueeze(1).unsqueeze(-1)             # [B, 1, num_teachers, 1]
            teacher_reprs = teacher_reprs * mask
            gate = gate * mask                                          # [B, 1, num_teachers, 1]
            # Re-normalise gate over available teachers
            gate_sum = gate.sum(dim=2, keepdim=True).clamp(min=1e-8)
            gate = gate / gate_sum

        # Weighted sum over teachers (dim=2 = num_teachers)
        teacher_fused = (teacher_reprs * gate).sum(dim=2)             # [B, 24, H]

        # Fuse with backbone
        fused = torch.cat([h, teacher_fused], dim=-1)                 # [B, 24, 2H]
        return self.fusion(fused)                                      # [B, 24, H]


# ── Decoder heads ────────────────────────────────────────────────────

class DayDecoderHead(nn.Module):
    """Per-hour MLP that maps backbone hidden states to delta predictions.

    Input:  [B, 24, hidden_dim]
    Output: [B, 24]
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        mid = max(hidden_dim // 2, 16)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, 24, hidden_dim] -> [B, 24]"""
        return self.head(h).squeeze(-1)


class ConfidenceHead(nn.Module):
    """Per-hour confidence score (sigmoid output in [0, 1]).

    High confidence = model is sure about its prediction.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        mid = max(hidden_dim // 2, 16)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, 24, hidden_dim] -> [B, 24]"""
        return self.head(h).squeeze(-1)


class ShockSensitivityHead(nn.Module):
    """Per-hour shock sensitivity score (sigmoid output in [0, 1]).

    High sensitivity = prediction is likely affected by a price shock.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        mid = max(hidden_dim // 2, 16)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, 24, hidden_dim] -> [B, 24]"""
        return self.head(h).squeeze(-1)


# ── Full V3 Model ────────────────────────────────────────────────────

class TrendKnightV3(nn.Module):
    """TrendKnight-X v3: multiscale decomposition + teacher fusion.

    Forward pass:
      1. Project per-hour features to hidden_dim
      2. Inject hour + segment embeddings
      3. Run backbone (TCN / GRU / transformer_tiny)
      4. Multiscale decomposition (trend / seasonal / shock)
      5. Period-aware refinement
      6. Teacher fusion gate (optional)
      7. Decode per-hour delta, confidence, shock sensitivity
      8. Combine decoder output with multiscale prediction
      9. rt_pred = da_anchor + delta_pred
    """

    def __init__(self, config: TrendKnightV3Config):
        super().__init__()
        self.config = config
        hd = config.hidden_dim

        # 1. Feature projection
        self.feature_proj = FeatureProjection(config.input_dim, hd, config.dropout)

        # 2. Embeddings
        self.hour_embed = HourEmbedding(config.num_hours, config.hour_embed_dim)
        self.seg_embed = SegmentEmbedding(config.num_segments, config.segment_embed_dim)

        # 3. Fusion: projected features + hour_embed + seg_embed -> hidden_dim
        fused_dim = hd + config.hour_embed_dim + config.segment_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hd),
            nn.LayerNorm(hd),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # 4. Backbone
        if config.backbone == "tcn":
            self.backbone = TCNBackbone(
                hd, config.num_layers,
                config.tcn_kernel_size, config.tcn_dilation_base, config.dropout,
            )
        elif config.backbone == "gru":
            self.backbone = GRUBackbone(hd, config.num_layers, config.dropout)
        elif config.backbone == "transformer_tiny":
            self.backbone = TransformerTinyBackbone(
                hd, config.transformer_nhead, config.transformer_dim_ff,
                config.num_layers, config.dropout,
            )
        else:
            raise ValueError(f"Unknown backbone: {config.backbone}")

        # 5. Multiscale decomposer (optional)
        self.multiscale_decomposer: MultiscaleDecomposer | None = None
        if config.multiscale:
            self.multiscale_decomposer = MultiscaleDecomposer(hd)
            # Decoder takes backbone + multiscale combined
            decoder_input_dim = hd + 1
        else:
            decoder_input_dim = hd

        # 6. Period branch
        self.period_branch = PeriodBranch(hd, config.segment_embed_dim, config.dropout)

        # 7. Teacher fusion gate (optional)
        self.teacher_gate: TeacherFusionGate | None = None
        if config.use_teacher_gate and config.teacher_input_dim > 0:
            self.teacher_gate = TeacherFusionGate(
                hd, config.teacher_input_dim, config.dropout,
            )

        # 8. Decoder head (takes backbone output + optional multiscale combined)
        self.decoder_head = DayDecoderHead(decoder_input_dim, config.dropout)

        # 9. Confidence head
        self.confidence_head = ConfidenceHead(hd, config.dropout)

        # 10. Shock sensitivity head
        self.shock_head = ShockSensitivityHead(hd, config.dropout)

        # Default hour indices: 1..24
        self.register_buffer("default_hours", torch.arange(1, 25, dtype=torch.long))

    def forward(
        self,
        features_24h: torch.Tensor,
        segment_id: torch.Tensor,
        da_anchor_24: torch.Tensor,
        hour_ids: torch.Tensor | None = None,
        teacher_features: torch.Tensor | None = None,
        teacher_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            features_24h:     [B, 24, input_dim] — per-hour feature vectors
            segment_id:       [B] — majority segment for the day (0/1/2)
            da_anchor_24:     [B, 24] — day-ahead anchor prices
            hour_ids:         [B, 24] — hour indices 1-24 (default: arange(1,25))
            teacher_features: [B, num_teachers, 24] — per-teacher delta predictions
            teacher_mask:     [B, num_teachers] — 1 where teacher is available

        Returns:
            dict with:
              delta_pred_24:        [B, 24]
              rt_pred_24:           [B, 24]
              confidence_24:        [B, 24]
              shock_sensitivity_24: [B, 24]
              multiscale_trend:     [B, 24]
              multiscale_seasonal:  [B, 24]
              multiscale_shock:     [B, 24]
        """
        B = features_24h.size(0)
        device = features_24h.device

        # 1. Project features
        h = self.feature_proj(features_24h)                       # [B, 24, hd]

        # 2. Hour embeddings
        if hour_ids is None:
            hour_ids = self.default_hours.unsqueeze(0).expand(B, -1)
        h_emb = self.hour_embed(hour_ids)                         # [B, 24, hour_emb]

        # 3. Segment embeddings (broadcast across 24 hours)
        s_emb = self.seg_embed(segment_id)                        # [B, seg_emb]
        s_emb_3d = s_emb.unsqueeze(1).expand(-1, 24, -1)          # [B, 24, seg_emb]

        # 4. Fuse
        fused = torch.cat([h, h_emb, s_emb_3d], dim=-1)
        h = self.fusion(fused)                                    # [B, 24, hd]

        # 5. Backbone
        h_bb = self.backbone(h)                                   # [B, 24, hd]

        # 6. Multiscale decomposition
        zero_24 = torch.zeros(B, 24, device=device)
        if self.multiscale_decomposer is not None:
            ms = self.multiscale_decomposer(h_bb)
            ms_trend = ms["trend"]
            ms_seasonal = ms["seasonal"]
            ms_shock = ms["shock"]
            ms_combined = ms["combined"]
        else:
            ms_trend = zero_24
            ms_seasonal = zero_24
            ms_shock = zero_24
            ms_combined = zero_24

        # 7. Period-aware refinement
        h_refined = self.period_branch(h_bb, s_emb)               # [B, 24, hd]

        # 8. Teacher fusion gate
        if self.teacher_gate is not None:
            h_fused = self.teacher_gate(h_refined, teacher_features, teacher_mask)
        else:
            h_fused = h_refined

        # 9. Decoder head
        if self.multiscale_decomposer is not None:
            decoder_input = torch.cat(
                [h_fused, ms_combined.unsqueeze(-1)], dim=-1
            )                                                     # [B, 24, hd+1]
        else:
            decoder_input = h_fused                               # [B, 24, hd]

        delta_from_features = self.decoder_head(decoder_input)     # [B, 24]

        # 10. Combine decoder output with multiscale prediction
        delta_pred_24 = delta_from_features + ms_combined

        # 11. Auxiliary heads
        confidence_24 = self.confidence_head(h_fused)              # [B, 24]
        shock_sensitivity_24 = self.shock_head(h_fused)            # [B, 24]

        # 12. Compute rt_pred
        rt_pred_24 = da_anchor_24 + delta_pred_24

        return {
            "delta_pred_24": delta_pred_24,
            "rt_pred_24": rt_pred_24,
            "confidence_24": confidence_24,
            "shock_sensitivity_24": shock_sensitivity_24,
            "multiscale_trend": ms_trend,
            "multiscale_seasonal": ms_seasonal,
            "multiscale_shock": ms_shock,
        }


# ── Factory ──────────────────────────────────────────────────────────

def build_model_v3(config: TrendKnightV3Config) -> TrendKnightV3:
    """Factory function for TrendKnight-X v3 model."""
    return TrendKnightV3(config)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
