#!/usr/bin/env python
"""Validate the DeepSGDFDelta / TrendKnight runtime environment.

Checks SGDFNet path, data file, Python imports, CUDA/CPU availability,
output directory writability, and all key script --help outputs.

Usage:
    python scripts/validate_environment.py \
      --sgdfnet-root ../electricity_forecast_model2.0_exp/SGDFNet \
      --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.xlsx

    python scripts/validate_environment.py --help
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate DeepSGDFDelta / TrendKnight runtime environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Checks performed:
  1. SGDFNet path resolution
  2. Data file existence
  3. Python import availability
  4. CUDA / CPU detection
  5. Output directory writability
  6. Key script --help smoke tests

Output:
  environment_report.json in the output directory (default: reports/local/).
""",
    )
    parser.add_argument(
        "--sgdfnet-root", type=str, default=None,
        help="Path to SGDFNet project root (contains src/sgdfnet/)",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to raw data file (xlsx or csv)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="reports/local",
        help="Output directory for environment_report.json",
    )
    return parser.parse_args()


# ── Check functions ──────────────────────────────────────────────────

def check_sgdfnet_path(sgdfnet_root: str | None) -> dict[str, Any]:
    """Check SGDFNet path resolution."""
    result: dict[str, Any] = {"check": "sgdfnet_path", "status": "UNKNOWN"}

    from models.deep_sgdf_delta.sgdfnet_bridge import find_sgdfnet_root

    try:
        root = find_sgdfnet_root(sgdfnet_root)
        result["status"] = "OK"
        result["path"] = str(root)
        result["src_exists"] = (root / "src" / "sgdfnet").is_dir()

        # Try listing key files
        key_files = [
            "src/sgdfnet/data_contract.py",
            "src/sgdfnet/protocol_b_cutoff.py",
            "src/sgdfnet/metrics.py",
            "src/sgdfnet/models.py",
        ]
        result["key_files"] = {}
        for kf in key_files:
            result["key_files"][kf] = (root / kf).is_file()

    except FileNotFoundError as exc:
        result["status"] = "FAIL"
        result["error"] = str(exc)
    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = str(exc)

    return result


def check_data_file(data_path: str | None) -> dict[str, Any]:
    """Check data file existence and basic validity."""
    result: dict[str, Any] = {"check": "data_file", "status": "UNKNOWN"}

    candidates = []
    if data_path:
        candidates.append(Path(data_path))

    # Default locations
    candidates.extend([
        PROJECT_ROOT / "data" / "shandong_pmos_hourly.xlsx",
        PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp" / "data" / "shandong_pmos_hourly.xlsx",
        PROJECT_ROOT / "data" / "shandong_pmos_hourly.csv",
        PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp" / "data" / "shandong_pmos_hourly.csv",
    ])

    for c in candidates:
        if c.exists():
            result["status"] = "OK"
            result["path"] = str(c.resolve())
            result["size_mb"] = round(c.stat().st_size / 1024 / 1024, 2)
            result["suffix"] = c.suffix

            # Try reading first few rows
            try:
                import pandas as pd
                if c.suffix in (".xlsx", ".xls"):
                    df = pd.read_excel(c, nrows=5)
                else:
                    df = pd.read_csv(c, nrows=5, encoding="utf-8-sig")
                result["columns_sample"] = list(df.columns)[:10]
                result["n_rows_sample"] = len(df)
            except Exception as exc:
                result["read_error"] = str(exc)
            return result

    result["status"] = "FAIL"
    result["error"] = "Data file not found in any candidate location"
    result["tried"] = [str(c) for c in candidates]
    return result


def check_python_imports() -> dict[str, Any]:
    """Check that all required Python packages are importable."""
    result: dict[str, Any] = {"check": "python_imports", "status": "OK"}

    required = [
        "torch", "numpy", "pandas", "sklearn", "scipy",
        "yaml", "openpyxl", "pytest",
    ]

    results = {}
    all_ok = True
    for pkg in required:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            results[pkg] = {"status": "OK", "version": version}
        except ImportError as exc:
            results[pkg] = {"status": "FAIL", "error": str(exc)}
            all_ok = False

    # Check project-internal imports (without triggering SGDFNet)
    internal = [
        "models.deep_sgdf_delta.model",
        "models.deep_sgdf_delta.losses",
        "models.deep_sgdf_delta.metrics",
        "models.deep_sgdf_delta.output_contract",
        "models.deep_sgdf_delta.model_v2",
    ]
    for mod_name in internal:
        try:
            importlib.import_module(mod_name)
            results[mod_name] = {"status": "OK"}
        except Exception as exc:
            results[mod_name] = {"status": "FAIL", "error": str(exc)}
            all_ok = False

    # Check bridge lazy import (should work without SGDFNet)
    try:
        from models.deep_sgdf_delta.sgdfnet_bridge import find_sgdfnet_root, lazy_import
        results["sgdfnet_bridge"] = {"status": "OK"}
    except Exception as exc:
        results["sgdfnet_bridge"] = {"status": "FAIL", "error": str(exc)}
        all_ok = False

    result["imports"] = results
    if not all_ok:
        result["status"] = "FAIL"
    return result


def check_cuda_cpu() -> dict[str, Any]:
    """Check CUDA/CPU availability."""
    result: dict[str, Any] = {"check": "cuda_cpu", "status": "UNKNOWN"}

    try:
        import torch
        result["torch_version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            result["cuda_version"] = torch.version.cuda
            result["gpu_count"] = torch.cuda.device_count()
            result["gpu_names"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
            # Memory info for first GPU
            try:
                props = torch.cuda.get_device_properties(0)
                result["gpu_0_memory_gb"] = round(props.total_mem / 1024**3, 1)
            except Exception:
                pass
            result["recommended_device"] = "cuda"
        else:
            result["recommended_device"] = "cpu"

        result["status"] = "OK"
    except ImportError:
        result["status"] = "FAIL"
        result["error"] = "PyTorch not installed"
        result["recommended_device"] = "cpu"

    return result


def check_output_writable(out_dir: str) -> dict[str, Any]:
    """Check that the output directory is writable."""
    result: dict[str, Any] = {"check": "output_writable", "status": "UNKNOWN"}

    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    try:
        out_path.mkdir(parents=True, exist_ok=True)
        # Try writing a temp file
        with tempfile.NamedTemporaryFile(dir=out_path, delete=True, suffix=".tmp") as tmp:
            tmp.write(b"test")
        result["status"] = "OK"
        result["path"] = str(out_path.resolve())
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = str(exc)
        result["path"] = str(out_path)

    return result


def check_script_help() -> dict[str, Any]:
    """Check that all key scripts can display --help without errors."""
    result: dict[str, Any] = {"check": "script_help", "status": "OK"}

    scripts = [
        "scripts/validate_environment.py",
        "scripts/p0_reproduce_sgdfnet_baseline.py",
        "scripts/search_phase2_champion.py",
        "scripts/train_phase2_trendknight.py",
        "scripts/evaluate_phase2_trendknight.py",
        "scripts/run_phase2_monthly_backtest.py",
        "scripts/train_deep_sgdf_delta.py",
        "scripts/predict_deep_sgdf_delta.py",
        "scripts/evaluate_deep_sgdf_delta.py",
        "scripts/check_sgdfnet_bridge.py",
    ]

    results = {}
    all_ok = True
    python_exe = sys.executable

    for script in scripts:
        script_path = PROJECT_ROOT / script
        if not script_path.exists():
            results[script] = {"status": "SKIP", "reason": "file not found"}
            continue

        try:
            proc = subprocess.run(
                [python_exe, str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(PROJECT_ROOT),
            )
            if proc.returncode == 0:
                results[script] = {"status": "OK"}
            else:
                results[script] = {
                    "status": "FAIL",
                    "returncode": proc.returncode,
                    "stderr": proc.stderr[:500],
                }
                all_ok = False
        except subprocess.TimeoutExpired:
            results[script] = {"status": "TIMEOUT"}
            all_ok = False
        except Exception as exc:
            results[script] = {"status": "ERROR", "error": str(exc)}
            all_ok = False

    result["scripts"] = results
    if not all_ok:
        result["status"] = "FAIL"
    return result


def check_pytest_collection() -> dict[str, Any]:
    """Check that pytest can collect tests without errors."""
    result: dict[str, Any] = {"check": "pytest_collection", "status": "UNKNOWN"}

    python_exe = sys.executable
    try:
        proc = subprocess.run(
            [python_exe, "-m", "pytest", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        # pytest --collect-only returns 0 even if no tests
        output = proc.stdout.strip()
        lines = output.split("\n")
        # Last line usually has the count
        result["status"] = "OK" if proc.returncode == 0 else "WARN"
        result["output_tail"] = "\n".join(lines[-5:]) if lines else ""
        result["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = str(exc)

    return result


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("DeepSGDFDelta / TrendKnight Environment Validation")
    print("=" * 60)
    print(f"  Timestamp  : {datetime.now().isoformat()}")
    print(f"  Python     : {sys.executable}")
    print(f"  Project    : {PROJECT_ROOT}")
    print()

    checks = [
        check_sgdfnet_path(args.sgdfnet_root),
        check_data_file(args.data_path),
        check_python_imports(),
        check_cuda_cpu(),
        check_output_writable(args.out_dir),
        check_script_help(),
        check_pytest_collection(),
    ]

    # Print summary
    for chk in checks:
        status_icon = {"OK": "[OK]", "FAIL": "[FAIL]", "ERROR": "[ERR]",
                       "WARN": "[WARN]", "UNKNOWN": "[???]", "SKIP": "[--]"}.get(
            chk["status"], "[??]"
        )
        print(f"  {status_icon:8s} {chk['check']}")

    # Write report
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "python": sys.executable,
        "python_version": sys.version,
        "project_root": str(PROJECT_ROOT),
        "platform": sys.platform,
        "checks": {chk["check"]: chk for chk in checks},
    }

    report_path = out_dir / "environment_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print()
    print(f"Report saved to: {report_path}")

    # Overall verdict
    fails = [c for c in checks if c["status"] in ("FAIL", "ERROR")]
    if fails:
        print(f"\nVERDICT: {len(fails)} check(s) FAILED")
        for c in fails:
            print(f"  - {c['check']}: {c.get('error', 'see report')}")
        sys.exit(1)
    else:
        print("\nVERDICT: All checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
