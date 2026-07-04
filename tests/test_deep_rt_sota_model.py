"""Tests for DeepRT-SOTA v2 model module."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deep_sgdf_delta.deep_rt_sota_model import (
    DeepRTSOTAModelConfig,
    DeepRTMLP,
    DeepRTTCN,
    DeepRTGRU,
    DeepRTTransformer,
    DeepRTSOTAModel,
    create_deep_rt_sota_model,
)


class TestDeepRTSOTAModelConfig:
    """Tests for DeepRTSOTAModelConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DeepRTSOTAModelConfig()
        assert config.model_profile == "deep_rt_tcn"
        assert config.seq_len_days == 14
        assert config.n_features == 20
        assert config.hidden_dim == 128
        assert config.num_layers == 3
        assert config.dropout == 0.1
        assert config.output_dim == 24

    def test_custom_config(self):
        """Test custom configuration."""
        config = DeepRTSOTAModelConfig(
            model_profile="deep_rt_gru",
            seq_len_days=7,
            n_features=30,
            hidden_dim=256,
            num_layers=2,
            dropout=0.2,
            output_dim=24,
        )
        assert config.model_profile == "deep_rt_gru"
        assert config.seq_len_days == 7
        assert config.n_features == 30


class TestDeepRTMLP:
    """Tests for DeepRTMLP."""

    def test_init(self):
        """Test model initialization."""
        model = DeepRTMLP(n_features=20, hidden_dim=128, num_layers=3, output_dim=24)
        assert model is not None

    def test_forward(self):
        """Test forward pass."""
        model = DeepRTMLP(n_features=20, hidden_dim=128, num_layers=3, output_dim=24)

        batch_size = 32
        x = torch.randn(batch_size, 20)
        rt_pred, confidence = model(x)

        assert rt_pred.shape == (batch_size, 24)
        assert confidence.shape == (batch_size, 24)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)


class TestDeepRTTCN:
    """Tests for DeepRTTCN."""

    def test_init(self):
        """Test model initialization."""
        model = DeepRTTCN(n_features=20, hidden_dim=128, num_layers=3, output_dim=24)
        assert model is not None

    def test_forward(self):
        """Test forward pass."""
        model = DeepRTTCN(n_features=20, hidden_dim=128, num_layers=3, output_dim=24)

        batch_size = 32
        seq_len = 14 * 24  # 14 days
        x = torch.randn(batch_size, seq_len, 20)
        rt_pred, confidence = model(x)

        assert rt_pred.shape == (batch_size, 24)
        assert confidence.shape == (batch_size, 24)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)


class TestDeepRTGRU:
    """Tests for DeepRTGRU."""

    def test_init(self):
        """Test model initialization."""
        model = DeepRTGRU(n_features=20, hidden_dim=128, num_layers=3, output_dim=24)
        assert model is not None

    def test_forward(self):
        """Test forward pass."""
        model = DeepRTGRU(n_features=20, hidden_dim=128, num_layers=3, output_dim=24)

        batch_size = 32
        seq_len = 14 * 24  # 14 days
        x = torch.randn(batch_size, seq_len, 20)
        rt_pred, confidence = model(x)

        assert rt_pred.shape == (batch_size, 24)
        assert confidence.shape == (batch_size, 24)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)


class TestDeepRTTransformer:
    """Tests for DeepRTTransformer."""

    def test_init(self):
        """Test model initialization."""
        model = DeepRTTransformer(n_features=20, hidden_dim=128, num_layers=3, output_dim=24)
        assert model is not None

    def test_forward(self):
        """Test forward pass."""
        model = DeepRTTransformer(n_features=20, hidden_dim=128, num_layers=3, output_dim=24)

        batch_size = 32
        seq_len = 14 * 24  # 14 days
        x = torch.randn(batch_size, seq_len, 20)
        rt_pred, confidence = model(x)

        assert rt_pred.shape == (batch_size, 24)
        assert confidence.shape == (batch_size, 24)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)


class TestCreateDeepRTSOTAModel:
    """Tests for create_deep_rt_sota_model function."""

    def test_create_mlp(self):
        """Test creating MLP model."""
        config = DeepRTSOTAModelConfig(model_profile="deep_rt_mlp", n_features=20)
        model = create_deep_rt_sota_model(config)
        assert isinstance(model, DeepRTMLP)

    def test_create_tcn(self):
        """Test creating TCN model."""
        config = DeepRTSOTAModelConfig(model_profile="deep_rt_tcn", n_features=20)
        model = create_deep_rt_sota_model(config)
        assert isinstance(model, DeepRTTCN)

    def test_create_gru(self):
        """Test creating GRU model."""
        config = DeepRTSOTAModelConfig(model_profile="deep_rt_gru", n_features=20)
        model = create_deep_rt_sota_model(config)
        assert isinstance(model, DeepRTGRU)

    def test_create_transformer(self):
        """Test creating Transformer model."""
        config = DeepRTSOTAModelConfig(model_profile="deep_rt_transformer", n_features=20)
        model = create_deep_rt_sota_model(config)
        assert isinstance(model, DeepRTTransformer)

    def test_invalid_profile(self):
        """Test invalid model profile."""
        config = DeepRTSOTAModelConfig(model_profile="invalid_model")
        with pytest.raises(ValueError, match="Unknown model_profile"):
            create_deep_rt_sota_model(config)


class TestDeepRTSOTAModel:
    """Tests for DeepRTSOTAModel wrapper."""

    def test_init(self):
        """Test model wrapper initialization."""
        config = DeepRTSOTAModelConfig(model_profile="deep_rt_tcn", n_features=20)
        model = DeepRTSOTAModel(config)
        assert model is not None

    def test_forward_tcn(self):
        """Test forward pass for TCN."""
        config = DeepRTSOTAModelConfig(model_profile="deep_rt_tcn", n_features=20)
        model = DeepRTSOTAModel(config)

        batch_size = 32
        seq_len = 14 * 24
        x_seq = torch.randn(batch_size, seq_len, 20)
        rt_pred, confidence = model(x_seq)

        assert rt_pred.shape == (batch_size, 24)
        assert confidence.shape == (batch_size, 24)

    def test_forward_gru(self):
        """Test forward pass for GRU."""
        config = DeepRTSOTAModelConfig(model_profile="deep_rt_gru", n_features=20)
        model = DeepRTSOTAModel(config)

        batch_size = 32
        seq_len = 14 * 24
        x_seq = torch.randn(batch_size, seq_len, 20)
        rt_pred, confidence = model(x_seq)

        assert rt_pred.shape == (batch_size, 24)
        assert confidence.shape == (batch_size, 24)

    def test_predict(self):
        """Test predict method."""
        config = DeepRTSOTAModelConfig(model_profile="deep_rt_tcn", n_features=20)
        model = DeepRTSOTAModel(config)

        batch_size = 32
        seq_len = 14 * 24
        x_seq = torch.randn(batch_size, seq_len, 20)

        rt_pred, confidence = model.predict(x_seq)

        assert isinstance(rt_pred, np.ndarray)
        assert isinstance(confidence, np.ndarray)
        assert rt_pred.shape == (batch_size, 24)
        assert confidence.shape == (batch_size, 24)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
