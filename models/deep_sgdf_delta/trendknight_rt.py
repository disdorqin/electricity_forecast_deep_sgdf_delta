"""TrendKnight-RT — Realtime prediction with multi-head decoding and gated fusion.

Phase DeepFinal-1 model building on TrendKnightV3 architecture.  Adds:
  - Delta head: predicts rt - da_anchor
  - Residual head: predicts rt - sgdfnet_pred (optional)
  - Confidence head: per-hour confidence score in [0, 1]
  - Period bias head: period-specific bias correction (optional)
  - Three fusion modes for combining anchors:
      Mode A: trend_rt = da_anchor + delta_pred
      Mode B: trend_rt = sgdfnet_pred + residual_to_sgdfnet
      Mode C: gated blend of A and B via LearnedGate

Architecture overview::

    features_24h ──> FeatureProjection ──> h ──> Backbone ──> h_bb
                                                │
                                                ├──> MultiscaleDecomposer (optional)
                                                │
    segment_id ──> PeriodBranch ──> h_refined
                                                │
    teacher_features ──> TeacherFusionGate (optional) ──> h_fused
                                                │
                          ┌─────────────────────┤
                          │                     │
                   delta_head              residual_head
                   (DayDecoderHead)        (DayDecoderHead)
                          │                     │
                   delta_pred_24         residual_to_sgdfnet_24
                          │
                   confidence_head ──> confidence_24
                   period_bias_head ──> period_bias_24
                          │
                   ┌──────┴──────┐
                Mode A         Mode B
           da + delta     sgdfnet + resid
                   │              │
                   └── LearnedGate ──> trend_rt_pred_24  (Mode C)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from .model_v3 import (
    FeatureProjection,
    HourEmbedding,
    SegmentEmbedding,
    TCNBackbone,
    GRUBackbone,
    TransformerTinyBackbone,
    MultiscaleDecomposer,
    PeriodBranch,
    TeacherFusionGate,
    DayDecoderHead,
    ConfidenceHead,
)


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class TrendKnightRTConfig:
    """Model hyperparameters for TrendKnight-RT (Phase DeepFinal-1).

    Builds on TrendKnightV3Config with additional heads for residual and
    period-bias prediction, plus three fusion modes for combining the
    day-ahead anchor and SGDFNet predictions.
    """

    # Input / backbone
    input_dim: int = 40
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    backbone: Literal["tcn", "gru", "transformer"] = "tcn"

    # TCN-specific
    tcn_kernel_size: int = 3
    tcn_dilation_base: int = 2

    # Transformer-tiny
    transformer_nhead: int = 4
    transformer_dim_ff: int = 128

    # Heads
    use_sgdfnet_residual_head: bool = True
    use_delta_head: bool = True
    use_confidence_head: bool = True
    use_period_bias: bool = True

    # Fusion mode: A=delta, B=residual, C=gated
    fusion_mode: Literal["A", "B", "C"] = "C"

    # Embeddings
    hour_embed_dim: int = 8
    segment_embed_dim: int = 8
    num_hours: int = 25
    num_segments: int = 3

    # Multiscale decomposition (reuse from V3)
    multiscale: bool = True

    # Teacher (optional)
    teacher_input_dim: int = 0  # 0 = no teacher


# ── Learned Gate ──────────────────────────────────────────────────────

class LearnedGate(nn.Module):
    """Per-hour learned gate for blending two prediction modes.

    Maps backbone hidden states to a scalar weight in [0, 1] per hour
    via a small MLP with sigmoid activation.  Used in fusion Mode C to
    blend between Mode A (delta-based) and Mode B (residual-based).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, 24, hidden_dim] -> [B, 24]"""
        return self.gate(h).squeeze(-1)


# ── Full RT Model ─────────────────────────────────────────────────────

class TrendKnightRT(nn.Module):
    """TrendKnight-RT: realtime prediction with multi-head decoding and gated fusion.

    Extends the V3 architecture with:
      * A **delta head** predicting ``rt - da_anchor``.
      * A **residual head** predicting ``rt - sgdfnet_pred`` (optional).
      * A **confidence head** outputting per-hour confidence in [0, 1].
      * A **period bias head** for period-specific bias correction (optional).
      * Three **fusion modes** for combining anchor-based predictions:
          - Mode A: ``trend_rt = da_anchor + delta_pred``
          - Mode B: ``trend_rt = sgdfnet_pred + residual_to_sgdfnet``
          - Mode C: ``trend_rt = gate * modeB + (1 - gate) * modeA``
            where ``gate = sigmoid(Linear(h_fused))`` per hour.

    Forward pass:
      1. Project per-hour features to hidden_dim.
      2. Inject hour + segment embeddings.
      3. Run backbone (TCN / GRU / Transformer).
      4. Optional multiscale decomposition.
      5. Period-aware refinement.
      6. Optional teacher fusion gate.
      7. Decode per-hour delta, residual, confidence, period bias.
      8. Combine via selected fusion mode.
    """

    def __init__(self, config: TrendKnightRTConfig):
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
        elif config.backbone == "transformer":
            self.backbone = TransformerTinyBackbone(
                hd, config.transformer_nhead, config.transformer_dim_ff,
                config.num_layers, config.dropout,
            )
        else:
            raise ValueError(f"Unknown backbone: {config.backbone}")

        # 5. Multiscale decomposer (optional, reuse from V3)
        self.multiscale_decomposer: Optional[MultiscaleDecomposer] = None
        if config.multiscale:
            self.multiscale_decomposer = MultiscaleDecomposer(hd)
            decoder_input_dim = hd + 1
        else:
            decoder_input_dim = hd

        # 6. Period branch (reuse from V3)
        self.period_branch = PeriodBranch(hd, config.segment_embed_dim, config.dropout)

        # 7. Teacher fusion gate (optional, reuse from V3)
        self.teacher_gate: Optional[TeacherFusionGate] = None
        if config.teacher_input_dim > 0:
            self.teacher_gate = TeacherFusionGate(
                hd, config.teacher_input_dim, config.dropout,
            )

        # 8. Prediction heads
        # Delta head: predicts rt - da_anchor
        self.delta_head: Optional[DayDecoderHead] = None
        if config.use_delta_head:
            self.delta_head = DayDecoderHead(decoder_input_dim, config.dropout)

        # Residual head: predicts rt - sgdfnet_pred
        self.residual_head: Optional[DayDecoderHead] = None
        if config.use_sgdfnet_residual_head:
            self.residual_head = DayDecoderHead(decoder_input_dim, config.dropout)

        # Confidence head: per-hour confidence in [0, 1]
        self.confidence_head: Optional[ConfidenceHead] = None
        if config.use_confidence_head:
            self.confidence_head = ConfidenceHead(hd, config.dropout)

        # Period bias head: period-specific bias correction
        self.period_bias_head: Optional[DayDecoderHead] = None
        if config.use_period_bias:
            self.period_bias_head = DayDecoderHead(hd, config.dropout)

        # 9. Learned gate for fusion Mode C
        self.fusion_gate: Optional[LearnedGate] = None
        if config.fusion_mode == "C":
            self.fusion_gate = LearnedGate(hd)

        # Default hour indices: 1..24
        self.register_buffer("default_hours", torch.arange(1, 25, dtype=torch.long))

    def forward(
        self,
        features_24h: torch.Tensor,
        segment_id: torch.Tensor,
        da_anchor_24: torch.Tensor,
        sgdfnet_pred_24: torch.Tensor,
        hour_ids: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
        teacher_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for TrendKnight-RT.

        Args:
            features_24h:     [B, 24, input_dim] -- per-hour feature vectors.
            segment_id:       [B] -- majority segment for the day (0/1/2).
            da_anchor_24:     [B, 24] -- day-ahead anchor prices.
            sgdfnet_pred_24:  [B, 24] -- SGDFNet prediction (zeros if unavailable).
            hour_ids:         [B, 24] -- hour indices 1-24 (default: arange(1,25)).
            teacher_features: [B, num_teachers, 24] -- per-teacher predictions.
            teacher_mask:     [B, num_teachers] -- 1 where teacher is available.

        Returns:
            dict with:
              delta_pred_24:           [B, 24] -- delta = rt - da_anchor.
              residual_to_sgdfnet_24:  [B, 24] -- residual = rt - sgdfnet_pred.
              trend_rt_pred_24:        [B, 24] -- final prediction (fusion mode).
              confidence_24:           [B, 24] -- confidence score [0, 1].
              period_bias_24:          [B, 24] -- period-specific bias correction.

        Fusion modes:
          Mode A: ``trend_rt_pred = da_anchor + delta_pred``
          Mode B: ``trend_rt_pred = sgdfnet_pred + residual_to_sgdfnet``
          Mode C: ``trend_rt_pred = gate * modeB + (1 - gate) * modeA``
            where ``gate = sigmoid(Linear(h_fused))`` per hour.
        """
        B = features_24h.size(0)
        device = features_24h.device

        # 1. Project features
        h = self.feature_proj(features_24h)                           # [B, 24, hd]

        # 2. Hour embeddings
        if hour_ids is None:
            hour_ids = self.default_hours.unsqueeze(0).expand(B, -1)
        h_emb = self.hour_embed(hour_ids)                             # [B, 24, hour_emb]

        # 3. Segment embeddings (broadcast across 24 hours)
        s_emb = self.seg_embed(segment_id)                            # [B, seg_emb]
        s_emb_3d = s_emb.unsqueeze(1).expand(-1, 24, -1)             # [B, 24, seg_emb]

        # 4. Fuse projected features with embeddings
        fused = torch.cat([h, h_emb, s_emb_3d], dim=-1)
        h = self.fusion(fused)                                        # [B, 24, hd]

        # 5. Backbone
        h_bb = self.backbone(h)                                       # [B, 24, hd]

        # 6. Multiscale decomposition (optional)
        if self.multiscale_decomposer is not None:
            ms = self.multiscale_decomposer(h_bb)
            ms_combined = ms["combined"]                              # [B, 24]
        else:
            ms_combined = torch.zeros(B, 24, device=device)

        # 7. Period-aware refinement
        h_refined = self.period_branch(h_bb, s_emb)                   # [B, 24, hd]

        # 8. Teacher fusion gate (optional)
        if self.teacher_gate is not None:
            h_fused = self.teacher_gate(h_refined, teacher_features, teacher_mask)
        else:
            h_fused = h_refined

        # 9. Decoder heads
        # Build decoder input: h_fused + optional multiscale combined
        if self.multiscale_decomposer is not None:
            decoder_input = torch.cat(
                [h_fused, ms_combined.unsqueeze(-1)], dim=-1
            )                                                         # [B, 24, hd+1]
        else:
            decoder_input = h_fused                                   # [B, 24, hd]

        # Delta head
        if self.delta_head is not None:
            delta_pred_24 = self.delta_head(decoder_input) + ms_combined  # [B, 24]
        else:
            delta_pred_24 = torch.zeros(B, 24, device=device)

        # Residual head
        if self.residual_head is not None:
            residual_to_sgdfnet_24 = self.residual_head(decoder_input) + ms_combined  # [B, 24]
        else:
            residual_to_sgdfnet_24 = torch.zeros(B, 24, device=device)

        # Confidence head
        if self.confidence_head is not None:
            confidence_24 = self.confidence_head(h_fused)             # [B, 24]
        else:
            confidence_24 = torch.ones(B, 24, device=device)

        # Period bias head
        if self.period_bias_head is not None:
            period_bias_24 = self.period_bias_head(h_fused)           # [B, 24]
        else:
            period_bias_24 = torch.zeros(B, 24, device=device)

        # 10. Fusion mode
        mode_a = da_anchor_24 + delta_pred_24                         # [B, 24]
        mode_b = sgdfnet_pred_24 + residual_to_sgdfnet_24            # [B, 24]

        if self.config.fusion_mode == "A":
            trend_rt_pred_24 = mode_a
        elif self.config.fusion_mode == "B":
            trend_rt_pred_24 = mode_b
        elif self.config.fusion_mode == "C":
            gate = self.fusion_gate(h_fused)                          # [B, 24]
            trend_rt_pred_24 = gate * mode_b + (1 - gate) * mode_a   # [B, 24]
        else:
            raise ValueError(f"Unknown fusion mode: {self.config.fusion_mode}")

        return {
            "delta_pred_24": delta_pred_24,
            "residual_to_sgdfnet_24": residual_to_sgdfnet_24,
            "trend_rt_pred_24": trend_rt_pred_24,
            "confidence_24": confidence_24,
            "period_bias_24": period_bias_24,
        }


# ── Factory ───────────────────────────────────────────────────────────

def build_trendknight_rt(config: TrendKnightRTConfig) -> TrendKnightRT:
    """Factory function for TrendKnight-RT model."""
    return TrendKnightRT(config)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
