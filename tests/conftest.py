"""Pytest conftest — ensure SGDFNet is importable from the sibling project."""
from __future__ import annotations

import sys
from pathlib import Path

# Add SGDFNet src to path so tests can import sgdfnet.*
_SGDFNET_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "SGDFNet" / "src",
    Path(__file__).resolve().parent.parent / "electricity_forecast_model2.0_exp" / "SGDFNet" / "src",
    Path(__file__).resolve().parent.parent.parent / "electricity_forecast_model2.0_exp" / "SGDFNet" / "src",
    Path(r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\SGDFNet\src"),
]

for _p in _SGDFNET_CANDIDATES:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
        break
