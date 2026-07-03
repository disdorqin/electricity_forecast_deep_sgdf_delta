"""Runtime profile definitions for TrendKnight-X v3.

Each profile defines a complete set of hyperparameters for training and
inference, allowing quick switching between debug, fast, and production
configurations.

Profiles:
  - debug_cpu:              tiny model on CPU for unit tests
  - v3_fast_tcn:            fast TCN baseline
  - v3_fast_gru:            fast GRU baseline
  - v3_multiscale_tcn:      full multiscale TCN (recommended)
  - v3_teacher_residual:    multiscale + teacher residual distillation
  - v3_teacher_moe:         multiscale + MoE teacher fusion
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ProfileConfig:
    """Complete runtime profile for TrendKnight-X v3."""

    # Model architecture
    hidden_dim: int = 96
    num_layers: int = 2
    backbone: str = "tcn"                       # tcn | gru | transformer_tiny
    multiscale: bool = True
    use_teacher_gate: bool = True
    teacher_input_dim: int = 3

    # Training
    epochs: int = 40
    patience: int = 6
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1

    # Data
    lookback_days: int = 7
    val_days: int = 30

    # Teacher usage
    teacher_usage: str = "optional"             # none | optional | required

    # AMP
    amp: bool = False

    # Runtime
    max_runtime_minutes: float = 30.0
    device: str = "auto"

    # Loss weights
    w_smape: float = 0.45
    w_delta_mae: float = 0.20
    w_period: float = 0.10
    w_smooth: float = 0.10
    w_teacher_distill: float = 0.10
    w_confidence_cal: float = 0.05

    # Reproducibility
    seed: int = 42


# ── Built-in profiles ────────────────────────────────────────────────

PROFILES: dict[str, ProfileConfig] = {

    "debug_cpu": ProfileConfig(
        hidden_dim=32,
        num_layers=1,
        backbone="tcn",
        multiscale=False,
        use_teacher_gate=False,
        teacher_input_dim=0,
        epochs=3,
        patience=2,
        batch_size=8,
        learning_rate=1e-3,
        lookback_days=3,
        val_days=7,
        teacher_usage="none",
        amp=False,
        max_runtime_minutes=5.0,
        device="cpu",
        seed=42,
    ),

    "v3_fast_tcn": ProfileConfig(
        hidden_dim=64,
        num_layers=2,
        backbone="tcn",
        multiscale=False,
        use_teacher_gate=False,
        teacher_input_dim=0,
        epochs=25,
        patience=5,
        batch_size=64,
        learning_rate=1e-3,
        lookback_days=7,
        val_days=30,
        teacher_usage="none",
        amp=False,
        max_runtime_minutes=15.0,
    ),

    "v3_fast_gru": ProfileConfig(
        hidden_dim=64,
        num_layers=2,
        backbone="gru",
        multiscale=False,
        use_teacher_gate=False,
        teacher_input_dim=0,
        epochs=25,
        patience=5,
        batch_size=64,
        learning_rate=1e-3,
        lookback_days=7,
        val_days=30,
        teacher_usage="none",
        amp=False,
        max_runtime_minutes=15.0,
    ),

    "v3_multiscale_tcn": ProfileConfig(
        hidden_dim=96,
        num_layers=2,
        backbone="tcn",
        multiscale=True,
        use_teacher_gate=False,
        teacher_input_dim=0,
        epochs=40,
        patience=6,
        batch_size=64,
        learning_rate=1e-3,
        lookback_days=7,
        val_days=30,
        teacher_usage="none",
        amp=False,
        max_runtime_minutes=25.0,
    ),

    "v3_teacher_residual": ProfileConfig(
        hidden_dim=96,
        num_layers=2,
        backbone="tcn",
        multiscale=True,
        use_teacher_gate=True,
        teacher_input_dim=3,
        epochs=50,
        patience=8,
        batch_size=64,
        learning_rate=8e-4,
        weight_decay=1e-4,
        dropout=0.15,
        lookback_days=7,
        val_days=30,
        teacher_usage="optional",
        amp=True,
        max_runtime_minutes=30.0,
        w_teacher_distill=0.15,
        w_smape=0.40,
    ),

    "v3_teacher_moe": ProfileConfig(
        hidden_dim=96,
        num_layers=2,
        backbone="tcn",
        multiscale=True,
        use_teacher_gate=True,
        teacher_input_dim=3,
        epochs=50,
        patience=8,
        batch_size=48,
        learning_rate=6e-4,
        weight_decay=1.5e-4,
        dropout=0.15,
        lookback_days=7,
        val_days=30,
        teacher_usage="optional",
        amp=True,
        max_runtime_minutes=35.0,
        w_teacher_distill=0.12,
        w_smape=0.40,
        w_confidence_cal=0.08,
    ),
}


# ── Access functions ─────────────────────────────────────────────────

def get_profile(name: str) -> ProfileConfig:
    """Get a built-in profile by name.

    Raises:
        KeyError: if profile name is not found.
    """
    if name not in PROFILES:
        available = ", ".join(sorted(PROFILES.keys()))
        raise KeyError(
            f"Unknown profile: {name!r}. Available profiles: {available}"
        )
    from copy import deepcopy
    return deepcopy(PROFILES[name])


def load_profiles_from_yaml(path: str | Path) -> dict[str, ProfileConfig]:
    """Load profile configurations from a YAML file.

    The YAML file should have a top-level ``profiles`` key mapping profile
    names to their configuration dictionaries.  Unknown keys are silently
    ignored.

    Args:
        path: Path to the YAML file.

    Returns:
        Dict mapping profile names to ProfileConfig instances.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required to load profiles from YAML. "
            "Install it with: pip install pyyaml"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Profile YAML not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "profiles" not in data:
        raise ValueError(
            f"YAML file must have a top-level 'profiles' key. Got keys: "
            f"{list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
        )

    profiles: dict[str, ProfileConfig] = {}
    valid_fields = set(ProfileConfig.__dataclass_fields__.keys())

    for name, cfg_dict in data["profiles"].items():
        if not isinstance(cfg_dict, dict):
            logger.warning("Profile %r is not a dict, skipping", name)
            continue

        # Filter to valid fields only
        filtered = {k: v for k, v in cfg_dict.items() if k in valid_fields}
        unknown = set(cfg_dict.keys()) - valid_fields
        if unknown:
            logger.warning(
                "Profile %r has unknown keys (ignored): %s", name, unknown,
            )

        profiles[name] = ProfileConfig(**filtered)
        logger.info("Loaded profile %r from %s", name, path)

    return profiles


def list_profiles() -> list[str]:
    """Return sorted list of available built-in profile names."""
    return sorted(PROFILES.keys())
