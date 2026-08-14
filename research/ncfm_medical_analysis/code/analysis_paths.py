"""Shared output locations for auditable mathematical diagnostics."""

from __future__ import annotations

import os
from pathlib import Path


def mathematical_root(root: Path) -> Path:
    """Return the explicit mathematical-analysis root for this benchmark."""
    configured = os.environ.get("NCFM_MATH_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (root / "research" / "ncfm_mathematical_analysis").resolve()
