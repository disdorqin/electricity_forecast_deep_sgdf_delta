"""DeepSGDFDelta model architecture.

Architecture:
  1. Feature projection MLP
  2. TCN or GRU backbone (configurable)
  3. Segment-conditioned heads (1_8 / 9_16 / 17_24)
  4. Global residual head

Lightweight by design:
  hidden_dim <= 128
  layers <= 2
  Total params < 100k typical
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DeepSGDFDeltaConfig:
    """Model hyperparameters — all lightweight by design."""
    input_dim: int = 40
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    backbone: Literal["tcn", "gru"] = "tcn"
    tcn_kernel_size: int = 3
    tcn_dilation_base: int = 2
    num_segments: int = 3
    segment_embed_dim: int = 8
    use_global_residual: bool = True
    global_residual_weight: float = 0.3
    amp_enabled: bool = False


class SegmentEmbedding(nn.Module):
    """Learnable segment embedding (1_8=0, 9_16=1, 17_24=2)."""

    def __init__(self, num_segments: int = 3, embed_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(num_segments, embed_dim)

    def forward(self, segment_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(segment_ids.long())


class FeatureProjection(nn.Module):
    """Project raw features to hidden dimension."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TCNBlock(nn.Module):
    """Single Temporal Convolutional Network block with causal padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding,
        )
        self.chomp = padding
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)

        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        out = self.conv(x)
        out = out[:, :, :-self.chomp] if self.chomp > 0 else out
        out = out.permute(0, 2, 1)  # -> (batch, seq_len, channels)
        out = self.norm(out)
        out = out.permute(0, 2, 1)  # -> (batch, channels, seq_len)
        out = self.relu(out)
        out = self.dropout(out)

        res = self.residual(x)
        return self.relu(out + res)


class TCNBackbone(nn.Module):
    """Temporal Convolutional Network backbone."""

    def __init__(self, hidden_dim: int, num_layers: int, kernel_size: int, dilation_base: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = dilation_base ** i
            layers.append(TCNBlock(hidden_dim, hidden_dim, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim) -> (batch, seq_len, hidden_dim)
        x = x.permute(0, 2, 1)  # -> (batch, hidden_dim, seq_len)
        x = self.network(x)
        return x.permute(0, 2, 1)  # -> (batch, seq_len, hidden_dim)


class GRUBackbone(nn.Module):
    """GRU backbone as alternative to TCN."""

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


class SegmentHead(nn.Module):
    """Per-segment prediction head for delta output."""

    def __init__(self, hidden_dim: int, segment_embed_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + segment_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled: torch.Tensor, seg_embed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([pooled, seg_embed], dim=-1)
        return self.head(combined).squeeze(-1)


class GlobalResidualHead(nn.Module):
    """Simple global head that provides a baseline delta prediction."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.head(pooled).squeeze(-1)


class DeepSGDFDeltaModel(nn.Module):
    """Full DeepSGDFDelta model.

    Forward pass:
      1. Project features to hidden dim
      2. Run TCN/GRU backbone
      3. Pool temporal dimension (mean pool)
      4. For each sample, select segment head based on segment_id
      5. Add global residual contribution
      6. Output delta_pred; rt_pred = da_anchor + delta_pred
    """

    def __init__(self, config: DeepSGDFDeltaConfig):
        super().__init__()
        self.config = config

        # Feature projection
        self.feature_proj = FeatureProjection(config.input_dim, config.hidden_dim, config.dropout)

        # Backbone
        if config.backbone == "tcn":
            self.backbone = TCNBackbone(
                config.hidden_dim, config.num_layers,
                config.tcn_kernel_size, config.tcn_dilation_base, config.dropout,
            )
        elif config.backbone == "gru":
            self.backbone = GRUBackbone(config.hidden_dim, config.num_layers, config.dropout)
        else:
            raise ValueError(f"Unknown backbone: {config.backbone}")

        # Segment embedding
        self.seg_embed = SegmentEmbedding(config.num_segments, config.segment_embed_dim)

        # Segment-conditioned heads
        self.segment_heads = nn.ModuleList([
            SegmentHead(config.hidden_dim, config.segment_embed_dim)
            for _ in range(config.num_segments)
        ])

        # Global residual head
        self.global_residual = GlobalResidualHead(config.hidden_dim) if config.use_global_residual else None

    def forward(
        self,
        features: torch.Tensor,
        segment_ids: torch.Tensor,
        da_anchor: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            features: (batch, seq_len, input_dim) — feature sequence
            segment_ids: (batch,) — target segment for each sample
            da_anchor: (batch,) — day-ahead anchor price, optional

        Returns:
            dict with delta_pred, rt_pred (if da_anchor given)
        """
        # 1. Feature projection
        h = self.feature_proj(features)  # (batch, seq_len, hidden_dim)

        # 2. Backbone
        h = self.backbone(h)  # (batch, seq_len, hidden_dim)

        # 3. Temporal pooling
        pooled = h.mean(dim=1)  # (batch, hidden_dim)

        # 4. Segment-conditioned prediction
        seg_emb = self.seg_embed(segment_ids)  # (batch, segment_embed_dim)
        seg_preds = []
        for i, head in enumerate(self.segment_heads):
            seg_preds.append(head(pooled, seg_emb))
        seg_pred_tensor = torch.stack(seg_preds, dim=1)  # (batch, num_segments)

        # Select the prediction for the target segment
        delta_pred = seg_pred_tensor.gather(1, segment_ids.long().unsqueeze(1)).squeeze(1)

        # 5. Global residual blend
        if self.global_residual is not None:
            global_pred = self.global_residual(pooled)
            alpha = self.config.global_residual_weight
            delta_pred = alpha * global_pred + (1 - alpha) * delta_pred

        result = {"delta_pred": delta_pred}

        # 6. Compute rt_pred if da_anchor provided
        if da_anchor is not None:
            result["rt_pred"] = da_anchor + delta_pred

        return result


def build_model(config: DeepSGDFDeltaConfig) -> DeepSGDFDeltaModel:
    """Factory function."""
    return DeepSGDFDeltaModel(config)
