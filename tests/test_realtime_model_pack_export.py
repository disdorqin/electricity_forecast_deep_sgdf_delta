"""Tests for realtime model pack export.

Covers:
  - Export creates all required files (config.yaml, model weights, README.md)
  - README.md exists and contains model name
  - config.yaml is loadable and contains expected fields
  - Model can be instantiated from the exported pack

Uses tmp_path fixtures for all file operations.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import yaml

from models.deep_sgdf_delta.trendknight_rt import (
    TrendKnightRTConfig,
    TrendKnightRT,
    build_trendknight_rt,
    count_parameters,
)


# ── Helpers ────────────────────────────────────────────────────────────

MODEL_NAME = "TrendKnightRT"


def _export_model_pack(
    export_dir: Path,
    config: TrendKnightRTConfig | None = None,
    model_name: str = MODEL_NAME,
) -> Path:
    """Simulate exporting a model pack to *export_dir*.

    Creates:
      - config.yaml: serialised TrendKnightRTConfig
      - model.pt: state dict
      - README.md: human-readable model card
      - metadata.json: lightweight run metadata

    Returns the export directory path.
    """
    if config is None:
        config = TrendKnightRTConfig(input_dim=40)

    model = build_trendknight_rt(config)
    model.eval()

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save config as YAML
    config_dict = asdict(config)
    config_dict["model_name"] = model_name
    config_path = export_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    # 2. Save model state dict
    model_path = export_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    # 3. Generate README
    readme_path = export_dir / "README.md"
    readme_content = (
        f"# {model_name}\n\n"
        f"Realtime electricity price trend prediction model.\n\n"
        f"## Configuration\n\n"
        f"- Backbone: {config.backbone}\n"
        f"- Hidden dim: {config.hidden_dim}\n"
        f"- Input dim: {config.input_dim}\n"
        f"- Fusion mode: {config.fusion_mode}\n"
        f"- Parameters: {count_parameters(model):,}\n"
    )
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    # 4. Save metadata
    metadata = {
        "model_name": model_name,
        "config_path": str(config_path),
        "model_path": str(model_path),
        "n_parameters": count_parameters(model),
        "feature_version": "v1.0",
    }
    metadata_path = export_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return export_dir


def _load_config_from_pack(export_dir: Path) -> TrendKnightRTConfig:
    """Load a TrendKnightRTConfig from an exported pack."""
    config_path = export_dir / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Remove non-config keys
    config_dict.pop("model_name", None)

    # Reconstruct config
    config = TrendKnightRTConfig(**config_dict)
    return config


# ── Export creates files ──────────────────────────────────────────────

class TestExportCreatesFiles:
    def test_export_creates_files(self, tmp_path):
        """All required files are created."""
        export_dir = _export_model_pack(tmp_path / "model_pack")

        required_files = ["config.yaml", "model.pt", "README.md", "metadata.json"]
        for fname in required_files:
            fpath = export_dir / fname
            assert fpath.exists(), f"Required file '{fname}' not found in {export_dir}"
            assert fpath.stat().st_size > 0, f"File '{fname}' is empty"

    def test_export_creates_directory(self, tmp_path):
        """Export creates the directory if it doesn't exist."""
        export_dir = tmp_path / "new_dir" / "model_pack"
        _export_model_pack(export_dir)
        assert export_dir.exists()
        assert export_dir.is_dir()


# ── README generated ──────────────────────────────────────────────────

class TestReadmeGenerated:
    def test_readme_generated(self, tmp_path):
        """README.md exists and contains model name."""
        export_dir = _export_model_pack(tmp_path / "model_pack")
        readme_path = export_dir / "README.md"

        assert readme_path.exists()
        content = readme_path.read_text(encoding="utf-8")
        assert MODEL_NAME in content, f"README should contain '{MODEL_NAME}'"

    def test_readme_contains_config_info(self, tmp_path):
        """README contains backbone and hidden_dim info."""
        config = TrendKnightRTConfig(input_dim=40, backbone="gru", hidden_dim=128)
        export_dir = _export_model_pack(tmp_path / "model_pack", config=config)
        readme_path = export_dir / "README.md"
        content = readme_path.read_text(encoding="utf-8")

        assert "gru" in content
        assert "128" in content


# ── Config loadable ───────────────────────────────────────────────────

class TestConfigLoadable:
    def test_config_loadable(self, tmp_path):
        """config.yaml is loadable and contains expected fields."""
        original_config = TrendKnightRTConfig(
            input_dim=40,
            hidden_dim=64,
            backbone="tcn",
            fusion_mode="C",
        )
        export_dir = _export_model_pack(
            tmp_path / "model_pack", config=original_config
        )

        config_path = export_dir / "config.yaml"
        assert config_path.exists()

        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)

        assert isinstance(loaded, dict)
        assert loaded["input_dim"] == 40
        assert loaded["hidden_dim"] == 64
        assert loaded["backbone"] == "tcn"
        assert loaded["fusion_mode"] == "C"
        assert loaded["model_name"] == MODEL_NAME

    def test_config_roundtrip(self, tmp_path):
        """Config survives a save/load roundtrip."""
        original = TrendKnightRTConfig(
            input_dim=32,
            hidden_dim=48,
            num_layers=3,
            dropout=0.2,
            backbone="gru",
            fusion_mode="B",
        )
        export_dir = _export_model_pack(
            tmp_path / "model_pack", config=original
        )
        loaded = _load_config_from_pack(export_dir)

        assert loaded.input_dim == original.input_dim
        assert loaded.hidden_dim == original.hidden_dim
        assert loaded.num_layers == original.num_layers
        assert loaded.dropout == original.dropout
        assert loaded.backbone == original.backbone
        assert loaded.fusion_mode == original.fusion_mode


# ── Model instantiable ────────────────────────────────────────────────

class TestModelInstantiable:
    def test_model_instantiable(self, tmp_path):
        """Model can be instantiated from pack."""
        original_config = TrendKnightRTConfig(input_dim=40, hidden_dim=64)
        export_dir = _export_model_pack(
            tmp_path / "model_pack", config=original_config
        )

        # Load config and instantiate model
        loaded_config = _load_config_from_pack(export_dir)
        model = build_trendknight_rt(loaded_config)

        assert isinstance(model, TrendKnightRT)
        assert model.config.input_dim == 40
        assert model.config.hidden_dim == 64

    def test_model_state_dict_loadable(self, tmp_path):
        """Saved state dict can be loaded into a fresh model."""
        original_config = TrendKnightRTConfig(input_dim=40, hidden_dim=64)
        export_dir = _export_model_pack(
            tmp_path / "model_pack", config=original_config
        )

        # Load config and create fresh model
        loaded_config = _load_config_from_pack(export_dir)
        model = build_trendknight_rt(loaded_config)

        # Load state dict
        model_path = export_dir / "model.pt"
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        # Verify model works with a forward pass
        model.eval()
        with torch.no_grad():
            out = model(
                features_24h=torch.randn(1, 24, 40),
                segment_id=torch.tensor([0]),
                da_anchor_24=torch.randn(1, 24) * 50 + 300,
                sgdfnet_pred_24=torch.randn(1, 24) * 50 + 300,
            )

        assert out["trend_rt_pred_24"].shape == (1, 24)
        assert torch.isfinite(out["trend_rt_pred_24"]).all()

    def test_model_produces_same_output(self, tmp_path):
        """Loaded model produces same output as original."""
        torch.manual_seed(42)
        original_config = TrendKnightRTConfig(input_dim=40, hidden_dim=64)
        original_model = build_trendknight_rt(original_config)
        original_model.eval()

        # Create test inputs
        inputs = {
            "features_24h": torch.randn(2, 24, 40),
            "segment_id": torch.randint(0, 3, (2,)),
            "da_anchor_24": torch.randn(2, 24) * 50 + 300,
            "sgdfnet_pred_24": torch.randn(2, 24) * 50 + 300,
        }

        # Get original output
        with torch.no_grad():
            original_out = original_model(**inputs)

        # Export and reload
        export_dir = tmp_path / "model_pack"
        export_dir.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(original_config)
        config_dict["model_name"] = MODEL_NAME
        with open(export_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f)
        torch.save(original_model.state_dict(), export_dir / "model.pt")

        # Load and compare
        loaded_config = _load_config_from_pack(export_dir)
        loaded_model = build_trendknight_rt(loaded_config)
        state_dict = torch.load(
            export_dir / "model.pt", map_location="cpu", weights_only=True
        )
        loaded_model.load_state_dict(state_dict)
        loaded_model.eval()

        with torch.no_grad():
            loaded_out = loaded_model(**inputs)

        torch.testing.assert_close(
            original_out["trend_rt_pred_24"],
            loaded_out["trend_rt_pred_24"],
            atol=1e-5,
            rtol=1e-5,
        )
