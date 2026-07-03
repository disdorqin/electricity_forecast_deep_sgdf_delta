"""Bridge module to find and import SGDFNet from the sibling project.

DeepSGDFDelta / TrendKnight depends on SGDFNet code from the sibling project.
This module locates the SGDFNet ``src`` directory and imports the required
sub-packages without copying any source code.

Resolution order for the SGDFNet root directory:
  1. Explicit ``--sgdfnet-root`` CLI argument
  2. ``SGDFNET_ROOT`` environment variable
  3. Sibling directory: ``../electricity_forecast_model2.0_exp/SGDFNet``

Usage from other modules::

    from models.deep_sgdf_delta.sgdfnet_bridge import (
        load_dataset,
        preprocess_dataframe,
        FeatureConfig,
        add_business_time_columns,
        run_protocol_b_cutoff_experiment,
        build_metrics_frame,
        capped_smape,
        DeltaRegressor,
        HGBModelConfig,
    )
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


# ── Discovery ──────────────────────────────────────────────────────────

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent  # deep_model_for_electricity/

_SIBLING_CANDIDATES = [
    # ../electricity_forecast_model2.0_exp/SGDFNet   (side-by-side repos)
    _REPO_ROOT.parent / "electricity_forecast_model2.0_exp" / "SGDFNet",
    # Inside this repo (unlikely, but kept for safety)
    _REPO_ROOT / "SGDFNet",
]


def _parse_cli_sgdfnet_root() -> Optional[Path]:
    """Extract ``--sgdfnet-root <path>`` from *sys.argv* without consuming it.

    We scan *sys.argv* manually so that the bridge can be used before any
    ``argparse`` parser runs (and without interfering with other parsers).
    """
    args = sys.argv[1:]
    for i, token in enumerate(args):
        if token == "--sgdfnet-root" and i + 1 < len(args):
            return Path(args[i + 1]).resolve()
        if token.startswith("--sgdfnet-root="):
            return Path(token.split("=", 1)[1]).resolve()
    return None


def find_sgdfnet_root(sgdfnet_root: Optional[str | Path] = None) -> Path:
    """Resolve the SGDFNet project root directory.

    Parameters
    ----------
    sgdfnet_root:
        An explicit path to the SGDFNet project root (the directory that
        *contains* the ``src/`` sub-directory).  When *None*, the function
        checks the ``--sgdfnet-root`` CLI flag, then the ``SGDFNET_ROOT``
        environment variable, and finally the well-known sibling directory.

    Returns
    -------
    Path
        The resolved, absolute path to the SGDFNet project root.

    Raises
    ------
    FileNotFoundError
        If no valid SGDFNet installation can be located.
    """
    # 1. Explicit parameter (highest priority)
    if sgdfnet_root is not None:
        candidate = Path(sgdfnet_root).resolve()
        if (candidate / "src" / "sgdfnet").is_dir():
            return candidate
        raise FileNotFoundError(
            f"Explicit sgdfnet_root={candidate} does not contain src/sgdfnet/."
        )

    # 2. CLI argument
    cli_root = _parse_cli_sgdfnet_root()
    if cli_root is not None:
        if (cli_root / "src" / "sgdfnet").is_dir():
            return cli_root
        raise FileNotFoundError(
            f"--sgdfnet-root={cli_root} does not contain src/sgdfnet/."
        )

    # 3. Environment variable
    env_root = os.environ.get("SGDFNET_ROOT")
    if env_root:
        env_path = Path(env_root).resolve()
        if (env_path / "src" / "sgdfnet").is_dir():
            return env_path
        raise FileNotFoundError(
            f"SGDFNET_ROOT={env_path} does not contain src/sgdfnet/."
        )

    # 4. Sibling directory fallback
    for sibling in _SIBLING_CANDIDATES:
        sibling = sibling.resolve()
        if (sibling / "src" / "sgdfnet").is_dir():
            return sibling

    raise FileNotFoundError(
        "Could not locate SGDFNet.  Tried:\n"
        + "\n".join(f"  - {c.resolve()}" for c in _SIBLING_CANDIDATES)
        + "\n\n"
        "Please use one of the following methods to specify the location:\n"
        "  1. Pass --sgdfnet-root <path> on the command line\n"
        "  2. Set the SGDFNET_ROOT environment variable\n"
        "  3. Place SGDFNet at one of the sibling paths listed above\n"
        "\n"
        "The path should point to the SGDFNet project root (the directory\n"
        "that contains src/sgdfnet/).\n"
    )


def _ensure_sgdfnet_on_path(sgdfnet_root: Optional[Path] = None) -> Path:
    """Make sure the SGDFNet ``src`` directory is on *sys.path*.

    Returns the resolved SGDFNet project root.
    """
    root = find_sgdfnet_root(sgdfnet_root)
    src_dir = root / "src"
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return root


# ── Bootstrap: resolve and inject at import time ──────────────────────

_SGDFNET_ROOT = _ensure_sgdfnet_on_path()
_SGDFNET_SRC = _SGDFNET_ROOT / "src"


# ── Re-exports from sgdfnet.data_contract ──────────────────────────────

from sgdfnet.data_contract import (  # noqa: E402
    ACTUAL_COLS,
    ACTUAL_TO_FORECAST_MAP,
    DA_COL,
    FORECAST_COLS,
    REQUIRED_COLUMNS,
    RT_COL,
    TIMESTAMP_COL,
    FeatureConfig,
    add_business_time_columns,
    build_feature_manifest,
    load_dataset,
    preprocess_dataframe,
)

# ── Re-exports from sgdfnet.protocol_b_cutoff ──────────────────────────

from sgdfnet.protocol_b_cutoff import (  # noqa: E402
    ProtocolBCutoffConfig,
    _build_inference_frame,
    _build_protocol_b_visible_frame,
    load_protocol_b_cutoff_config,
    run_protocol_b_cutoff_experiment,
)

# ── Re-exports from sgdfnet.metrics ────────────────────────────────────

from sgdfnet.metrics import (  # noqa: E402
    build_metrics_frame,
    build_segment_metrics,
    build_tail_metrics,
    capped_smape,
    direction_accuracy,
    mae,
    positive_direction_recall,
    rmse,
    smape,
)

# ── Re-exports from sgdfnet.models ─────────────────────────────────────

from sgdfnet.models import (  # noqa: E402
    DeltaRegressor,
    HGBModelConfig,
    SegmentConditionedDeltaRegressor,
)


# ── Public API ─────────────────────────────────────────────────────────

__all__ = [
    # Discovery
    "find_sgdfnet_root",
    # data_contract
    "ACTUAL_COLS",
    "ACTUAL_TO_FORECAST_MAP",
    "DA_COL",
    "FORECAST_COLS",
    "REQUIRED_COLUMNS",
    "RT_COL",
    "TIMESTAMP_COL",
    "FeatureConfig",
    "add_business_time_columns",
    "build_feature_manifest",
    "load_dataset",
    "preprocess_dataframe",
    # protocol_b_cutoff
    "ProtocolBCutoffConfig",
    "_build_inference_frame",
    "_build_protocol_b_visible_frame",
    "load_protocol_b_cutoff_config",
    "run_protocol_b_cutoff_experiment",
    # metrics
    "build_metrics_frame",
    "build_segment_metrics",
    "build_tail_metrics",
    "capped_smape",
    "direction_accuracy",
    "mae",
    "positive_direction_recall",
    "rmse",
    "smape",
    # models
    "DeltaRegressor",
    "HGBModelConfig",
    "SegmentConditionedDeltaRegressor",
]
