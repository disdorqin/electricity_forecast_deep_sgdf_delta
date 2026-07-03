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


# ── Lazy import system ──────────────────────────────────────────────
#
# SGDFNet is NOT imported at module load time.  Call ``lazy_import()``
# explicitly (or access any re-exported symbol) to trigger resolution.
# This ensures ``python scripts/*.py --help`` works even when SGDFNet
# is not present on the filesystem.

_SGDFNET_ROOT: Path | None = None
_SGDFNET_SRC: Path | None = None
_IMPORTED = False
_IMPORT_ERROR: str | None = None


def lazy_import(sgdfnet_root: Optional[str | Path] = None) -> Path:
    """Resolve SGDFNet, inject onto sys.path, and import all re-exports.

    Safe to call multiple times — subsequent calls are no-ops.

    Returns the resolved SGDFNet project root.
    """
    global _SGDFNET_ROOT, _SGDFNET_SRC, _IMPORTED, _IMPORT_ERROR

    if _IMPORTED:
        assert _SGDFNET_ROOT is not None
        return _SGDFNET_ROOT

    try:
        root = _ensure_sgdfnet_on_path(sgdfnet_root)
        _SGDFNET_ROOT = root
        _SGDFNET_SRC = root / "src"
        _IMPORTED = True
        return root
    except Exception as exc:
        _IMPORT_ERROR = str(exc)
        raise


def _ensure_imported() -> None:
    """Trigger lazy import if not yet done.  Raises on failure."""
    if not _IMPORTED:
        lazy_import()


class _LazyModule:
    """Module-level __getattr__ proxy for lazy sgdfnet.* imports."""

    # Map of symbol name → (sgdfnet submodule, attribute name)
    _SYMBOL_MAP: dict[str, tuple[str, str]] = {
        # data_contract
        "ACTUAL_COLS":              ("sgdfnet.data_contract", "ACTUAL_COLS"),
        "ACTUAL_TO_FORECAST_MAP":   ("sgdfnet.data_contract", "ACTUAL_TO_FORECAST_MAP"),
        "DA_COL":                   ("sgdfnet.data_contract", "DA_COL"),
        "FORECAST_COLS":            ("sgdfnet.data_contract", "FORECAST_COLS"),
        "REQUIRED_COLUMNS":         ("sgdfnet.data_contract", "REQUIRED_COLUMNS"),
        "RT_COL":                   ("sgdfnet.data_contract", "RT_COL"),
        "TIMESTAMP_COL":            ("sgdfnet.data_contract", "TIMESTAMP_COL"),
        "FeatureConfig":            ("sgdfnet.data_contract", "FeatureConfig"),
        "add_business_time_columns": ("sgdfnet.data_contract", "add_business_time_columns"),
        "build_feature_manifest":   ("sgdfnet.data_contract", "build_feature_manifest"),
        "load_dataset":             ("sgdfnet.data_contract", "load_dataset"),
        "preprocess_dataframe":     ("sgdfnet.data_contract", "preprocess_dataframe"),
        # protocol_b_cutoff
        "ProtocolBCutoffConfig":          ("sgdfnet.protocol_b_cutoff", "ProtocolBCutoffConfig"),
        "_build_inference_frame":         ("sgdfnet.protocol_b_cutoff", "_build_inference_frame"),
        "_build_protocol_b_visible_frame": ("sgdfnet.protocol_b_cutoff", "_build_protocol_b_visible_frame"),
        "load_protocol_b_cutoff_config":  ("sgdfnet.protocol_b_cutoff", "load_protocol_b_cutoff_config"),
        "run_protocol_b_cutoff_experiment": ("sgdfnet.protocol_b_cutoff", "run_protocol_b_cutoff_experiment"),
        # metrics
        "build_metrics_frame":      ("sgdfnet.metrics", "build_metrics_frame"),
        "build_segment_metrics":    ("sgdfnet.metrics", "build_segment_metrics"),
        "build_tail_metrics":       ("sgdfnet.metrics", "build_tail_metrics"),
        "capped_smape":             ("sgdfnet.metrics", "capped_smape"),
        "direction_accuracy":       ("sgdfnet.metrics", "direction_accuracy"),
        "mae":                      ("sgdfnet.metrics", "mae"),
        "positive_direction_recall": ("sgdfnet.metrics", "positive_direction_recall"),
        "rmse":                     ("sgdfnet.metrics", "rmse"),
        "smape":                    ("sgdfnet.metrics", "smape"),
        # models
        "DeltaRegressor":                   ("sgdfnet.models", "DeltaRegressor"),
        "HGBModelConfig":                   ("sgdfnet.models", "HGBModelConfig"),
        "SegmentConditionedDeltaRegressor":  ("sgdfnet.models", "SegmentConditionedDeltaRegressor"),
    }

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name: str):
        mapping = self._SYMBOL_MAP.get(name)
        if mapping is not None:
            _ensure_imported()

            import importlib
            mod = importlib.import_module(mapping[0])
            value = getattr(mod, mapping[1])
            # Cache on the real module so subsequent access is fast
            setattr(self._module, name, value)
            return value

        # Fall through to the real module for non-lazy attributes
        # (find_sgdfnet_root, lazy_import, __all__, etc.)
        try:
            return getattr(self._module, name)
        except AttributeError:
            raise AttributeError(
                f"module {self._module.__name__!r} has no attribute {name!r}"
            )


import sys as _sys
_sys.modules[__name__] = _LazyModule(_sys.modules[__name__])


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
