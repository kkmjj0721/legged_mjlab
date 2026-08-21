"""Project-relative paths used by task asset adapters."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the source checkout root.

    The editable source layout is:

        <project>/legged_mjlab/utils/paths.py

    Therefore ``parents[2]`` is the repository root.  An explicit environment
    variable is supported for deployment or a packaged checkout.
    """

    override = os.environ.get("LEGGED_MJLAB_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[2]

    if not root.exists():
        raise FileNotFoundError(f"Project root does not exist: {root}")
    return root


PROJECT_ROOT = project_root()
RESOURCE_ROOT = PROJECT_ROOT / "resources"
ROBOT_RESOURCE_ROOT = RESOURCE_ROOT / "robots"


def resource_path(*parts: str) -> Path:
    """Resolve a project resource and fail early when it is missing."""

    path = RESOURCE_ROOT.joinpath(*parts).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resource does not exist: {path}")
    return path
