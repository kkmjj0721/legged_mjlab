import os
from pathlib import Path

def project_root() -> Path:
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
    path = RESOURCE_ROOT.joinpath(*parts).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resource does not exist: {path}")
    return path