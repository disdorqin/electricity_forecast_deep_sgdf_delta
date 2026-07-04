"""Model architecture for DeepRT-SOTA v2.

This module implements model architectures for the standalone realtime price deep model.

Supported model profiles:
- deep_rt_mlp: Multi-layer perceptron
- deep_rt_tcn: Temporal convolutional network
- deep_rt_gru: Gated recurrent unit
- deep_rt_transformer: Transformer encoder

Output:
- rt_pred: realtime price prediction (24-hour vector or hourly)
- confidence: prediction confidence [0, 1]
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DeepRTSOTAModelConfig:
    """Configuration for DeepRT-SOTA model."""

    def __init__(
        self,
        model_profile: str = "deep_rt_tcn",
        seq_len_days: int = 14,
        target_mode: str = "direct",
        n_features: int = 20,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        output_dim: int = 24,  # 24-hour prediction
    ):
        """Initialize model configuration.

        Args:
            model_profile: Model architecture ("deep_rt_mlp", "deep_rt_tcn",
                          "deep_rt_gru", "deep_rt_transformer").
            seq_len_days: Number of past days used as sequence.
            target_mode: "direct" or "residual_to_da".
            n_features: Number of input features.
            hidden_dim: Hidden dimension.
            num_layers: Number of layers.
            dropout: Dropout rate.
            output_dim: Output dimension (24 for 24-hour prediction).
        """
        self.model_profile = model_profile
        self.seq_len_days = seq_len_days
        self.target_mode = target_mode
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim


class DeepRTMLP(nn.Module):
    """Multi-layer perceptron for realtime price prediction."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        output_dim: int = 24,
    ):
        """Initialize MLP.

        Args:
            n_features: Number of input features.
            hidden_dim: Hidden dimension.
            num_layers: Number of layers.
            dropout: Dropout rate.
            output_dim: Output dimension.
        """
        super().__init__()

        layers = []
        input_dim = n_features

        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.output_head = nn.Linear(hidden_dim, output_dim)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, n_features).

        Returns:
            Tuple of (rt_pred, confidence).
        """
        features = self.feature_extractor(x)
        rt_pred = self.output_head(features)
        confidence = self.confidence_head(features)
        return rt_pred, confidence


class DeepRTTCN(nn.Module):
    """Temporal Convolutional Network for realtime price prediction.

    Uses causal dilated convolutions to capture temporal patterns.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        output_dim: int = 24,
    ):
        """Initialize TCN.

        Args:
            n_features: Number of input features.
            hidden_dim: Hidden dimension.
            num_layers: Number of TCN layers.
            kernel_size: Convolution kernel size.
            dropout: Dropout rate.
            output_dim: Output dimension.
        """
        super().__init__()

        self.input_proj = nn.Linear(n_features, hidden_dim)

        # TCN layers with dilated convolutions
        tcn_layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            tcn_layers.extend([
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) * dilation,
                    dilation=dilation,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.tcn_layers = nn.Sequential(*tcn_layers)

        self.output_head = nn.Linear(hidden_dim, output_dim)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x_seq: Input sequence tensor (batch_size, seq_len, n_features).

        Returns:
            Tuple of (rt_pred, confidence).
        """
        batch_size, seq_len, _ = x_seq.shape

        # Project input
        x = self.input_proj(x_seq)  # (batch_size, seq_len, hidden_dim)

        # Transpose for Conv1d: (batch_size, hidden_dim, seq_len)
        x = x.transpose(1, 2)

        # Apply TCN layers
        x = self.tcn_layers(x)

        # Transpose back: (batch_size, seq_len, hidden_dim)
        x = x.transpose(1, 2)

        # Use last time step
        x = x[:, -1, :]  # (batch_size, hidden_dim)

        rt_pred = self.output_head(x)
        confidence = self.confidence_head(x)

        return rt_pred, confidence


class DeepRTGRU(nn.Module):
    """GRU-based model for realtime price prediction."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        output_dim: int = 24,
    ):
        """Initialize GRU.

        Args:
            n_features: Number of input features.
            hidden_dim: Hidden dimension.
            num_layers: Number of GRU layers.
            dropout: Dropout rate.
            output_dim: Output dimension.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.output_head = nn.Linear(hidden_dim, output_dim)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x_seq: Input sequence tensor (batch_size, seq_len, n_features).

        Returns:
            Tuple of (rt_pred, confidence).
        """
        # GRU output
        output, _ = self.gru(x_seq)  # output: (batch_size, seq_len, hidden_dim)

        # Use last time step
        x = output[:, -1, :]  # (batch_size, hidden_dim)

        rt_pred = self.output_head(x)
        confidence = self.confidence_head(x)

        return rt_pred, confidence


class DeepRTTransformer(nn.Module):
    """Transformer-based model for realtime price prediction."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        nhead: int = 4,
        dropout: float = 0.1,
        output_dim: int = 24,
    ):
        """Initialize Transformer.

        Args:
            n_features: Number of input features.
            hidden_dim: Hidden dimension.
            num_layers: Number of Transformer layers.
            nhead: Number of attention heads.
            dropout: Dropout rate.
            output_dim: Output dimension.
        """
        super().__init__()

        self.input_proj = nn.Linear(n_features, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.output_head = nn.Linear(hidden_dim, output_dim)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x_seq: Input sequence tensor (batch_size, seq_len, n_features).

        Returns:
            Tuple of (rt_pred, confidence).
        """
        # Project input
        x = self.input_proj(x_seq)  # (batch_size, seq_len, hidden_dim)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_dim)

        # Use last time step
        x = x[:, -1, :]  # (batch_size, hidden_dim)

        rt_pred = self.output_head(x)
        confidence = self.confidence_head(x)

        return rt_pred, confidence


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize positional encoding.

        Args:
            d_model: Model dimension.
            dropout: Dropout rate.
            max_len: Maximum sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor (batch_size, seq_len, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[: x.size(1)].transpose(0, 1)
        return self.dropout(x)


def create_deep_rt_sota_model(
    config: DeepRTSOTAModelConfig,
) -> nn.Module:
    """Create DeepRT-SOTA model based on configuration.

    Args:
        config: Model configuration.

    Returns:
        PyTorch model.
    """
    if config.model_profile == "deep_rt_mlp":
        return DeepRTMLP(
            n_features=config.n_features,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )
    elif config.model_profile == "deep_rt_tcn":
        return DeepRTTCN(
            n_features=config.n_features,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )
    elif config.model_profile == "deep_rt_gru":
        return DeepRTGRU(
            n_features=config.n_features,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )
    elif config.model_profile == "deep_rt_transformer":
        return DeepRTTransformer(
            n_features=config.n_features,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )
    else:
        raise ValueError(f"Unknown model_profile: {config.model_profile}")


class DeepRTSOTAModel(nn.Module):
    """Unified model wrapper for DeepRT-SOTA v2.

    This wrapper handles both sequence and static inputs,
    and provides a unified interface for training and inference.
    """

    def __init__(self, config: DeepRTSOTAModelConfig):
        """Initialize model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.model = create_deep_rt_sota_model(config)

    def forward(
        self,
        x_seq: torch.Tensor,
        x_static: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x_seq: Sequence input (batch_size, seq_len, n_features).
            x_static: Static input (batch_size, n_static_features).

        Returns:
            Tuple of (rt_pred, confidence).
        """
        # For sequence models (TCN, GRU, Transformer)
        if self.config.model_profile in ["deep_rt_tcn", "deep_rt_gru", "deep_rt_transformer"]:
            return self.model(x_seq)
        # For MLP (uses static features)
        elif self.config.model_profile == "deep_rt_mlp":
            if x_static is None:
                # Flatten sequence if static not provided
                x_static = x_seq.reshape(x_seq.shape[0], -1)
            return self.model(x_static)
        else:
            raise ValueError(f"Unknown model_profile: {self.config.model_profile}")

    def predict(
        self,
        x_seq: torch.Tensor,
        x_static: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with model (numpy output).

        Args:
            x_seq: Sequence input.
            x_static: Static input.

        Returns:
            Tuple of (rt_pred, confidence) as numpy arrays.
        """
        self.eval()
        with torch.no_grad():
            rt_pred, confidence = self.forward(x_seq, x_static)
            return rt_pred.cpu().numpy(), confidence.cpu().numpy()
