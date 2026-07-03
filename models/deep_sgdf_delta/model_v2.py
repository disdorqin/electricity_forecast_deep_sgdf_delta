"""DeepSGDFDeltaV2 — Day-level 24-hour decoder model.

Architecture:
  1. FeatureProjection: Linear + LayerNorm + GELU
  2. HourEmbedding (1-24) and SegmentEmbedding (0/1/2) injected into the sequence
  3. Backbone: TCN | GRU | transformer_tiny (configurable)
  4. 24-hour Decoder Head: per-hour MLP producing delta_pred_24 [B, 24]
  5. Optional residual head for SGDFNet baseline correction

Lightweight by design:
  hidden_dim <= 128, num_layers <= 2
  Total params < 300k with default config
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class DeepSGDFDeltaV2Config:
    """Model hyperparameters for the V2 day-level 24-hour decoder."""
    input_dim: int = 40
    hidden_dim: int = 64          # max 128
    num_layers: int = 2
    dropout: float = 0.1
    backbone: Literal["tcn", "gru", "transformer_tiny"] = "tcn"

    # TCN-specific
    tcn_kernel_size: int = 3
    tcn_dilation_base: int = 2

    # Transformer-tiny
    transformer_nhead: int = 4
    transformer_dim_ff: int = 128

    # Embeddings
    hour_embed_dim: int = 8
    segment_embed_dim: int = 8
    num_hours: int = 25           # indices 0..24; 0 unused (hours are 1-24)
    num_segments: int = 3

    # Residual head
    use_residual_head: bool = True
    residual_weight: float = 0.3

    # AMP
    amp_enabled: bool = False


# ── Building blocks ──────────────────────────────────────────────────

class FeatureProjection(nn.Module):
    """Project raw per-hour features to hidden dimension."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
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


# ── TCN blocks (reused from V1 with minor adjustments) ───────────────

class TCNBlock(nn.Module):
    """Single Temporal Convolutional Network block with causal padding."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.chomp = padding
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_ch)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv(x)
        if self.chomp > 0:
            out = out[:, :, :-self.chomp]
        out = out.permute(0, 2, 1)          # (B, T, C)
        out = self.norm(out)
        out = out.permute(0, 2, 1)          # (B, C, T)
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
            layers.append(TCNBlock(hidden_dim, hidden_dim, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H) -> (B, T, H)
        x = x.permute(0, 2, 1)              # (B, H, T)
        x = self.network(x)
        return x.permute(0, 2, 1)            # (B, T, H)


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
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerTinyBackbone(nn.Module):
    """Lightweight TransformerEncoder backbone.

    Uses nhead=4, dim_feedforward=128, num_layers=2 by default.
    """

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
            norm_first=True,          # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 24, H)
        x = self.pos_enc(x)
        return self.encoder(x)


# ── 24-hour Decoder Head ─────────────────────────────────────────────

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


# ── Residual Head ────────────────────────────────────────────────────

class ResidualHead(nn.Module):
    """Global residual head providing a baseline correction per hour.

    Pools the backbone output, then projects to 24 scalars.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        mid = max(hidden_dim // 2, 16)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 24),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, 24, hidden_dim] -> pool -> [B, hidden_dim] -> [B, 24]"""
        pooled = h.mean(dim=1)
        return self.head(pooled)


# ── Full V2 Model ────────────────────────────────────────────────────

class DeepSGDFDeltaV2(nn.Module):
    """Day-level 24-hour decoder model.

    Forward pass:
      1. Project per-hour features to hidden_dim
      2. Inject hour embeddings and segment embeddings
      3. Fuse via linear layer back to hidden_dim
      4. Run backbone (TCN / GRU / transformer_tiny)
      5. Decode per-hour delta predictions
      6. Blend with optional residual head
      7. rt_pred_24 = da_anchor_24 + delta_pred_24
    """

    def __init__(self, config: DeepSGDFDeltaV2Config):
        super().__init__()
        self.config = config
        hd = config.hidden_dim

        # 1. Feature projection
        self.feature_proj = FeatureProjection(config.input_dim, hd)

        # 2. Embeddings
        self.hour_embed = HourEmbedding(config.num_hours, config.hour_embed_dim)
        self.seg_embed = SegmentEmbedding(config.num_segments, config.segment_embed_dim)

        # 3. Fusion: combine projected features + hour_embed + seg_embed -> hidden_dim
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

        # 5. Decoder head
        self.decoder_head = DayDecoderHead(hd, config.dropout)

        # 6. Residual head (optional)
        self.residual_head = ResidualHead(hd, config.dropout) if config.use_residual_head else None

        # Default hour indices: 1..24
        self.register_buffer("default_hours", torch.arange(1, 25, dtype=torch.long))

    def forward(
        self,
        features_24h: torch.Tensor,
        segment_id: torch.Tensor,
        da_anchor_24: torch.Tensor,
        hour_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            features_24h: [B, 24, input_dim] — per-hour feature vectors for one day
            segment_id:   [B] — majority segment for the day (or any representative)
            da_anchor_24: [B, 24] — day-ahead anchor prices for all 24 hours
            hour_ids:     [B, 24] — hour indices 1-24 (default: arange(1,25))

        Returns:
            dict with:
              delta_pred_24: [B, 24]
              rt_pred_24:    [B, 24]
        """
        B = features_24h.size(0)

        # 1. Project features
        h = self.feature_proj(features_24h)           # [B, 24, hidden_dim]

        # 2. Hour embeddings
        if hour_ids is None:
            hour_ids = self.default_hours.unsqueeze(0).expand(B, -1)  # [B, 24]
        h_emb = self.hour_embed(hour_ids)              # [B, 24, hour_embed_dim]

        # 3. Segment embeddings (broadcast across 24 hours)
        s_emb = self.seg_embed(segment_id)             # [B, seg_embed_dim]
        s_emb = s_emb.unsqueeze(1).expand(-1, 24, -1)  # [B, 24, seg_embed_dim]

        # 4. Fuse: concat along feature dim then project back
        fused = torch.cat([h, h_emb, s_emb], dim=-1)   # [B, 24, hd+h_emb+s_emb]
        h = self.fusion(fused)                          # [B, 24, hidden_dim]

        # 5. Backbone
        h = self.backbone(h)                            # [B, 24, hidden_dim]

        # 6. Decoder head
        delta_pred_24 = self.decoder_head(h)            # [B, 24]

        # 7. Residual head blend
        if self.residual_head is not None:
            residual_pred = self.residual_head(h)       # [B, 24]
            alpha = self.config.residual_weight
            delta_pred_24 = alpha * residual_pred + (1 - alpha) * delta_pred_24

        # 8. Compute rt_pred
        rt_pred_24 = da_anchor_24 + delta_pred_24

        return {
            "delta_pred_24": delta_pred_24,
            "rt_pred_24": rt_pred_24,
        }


# ── Factory ──────────────────────────────────────────────────────────

def build_model_v2(config: DeepSGDFDeltaV2Config) -> DeepSGDFDeltaV2:
    """Factory function for V2 model."""
    return DeepSGDFDeltaV2(config)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
