from __future__ import annotations

import sys
from pathlib import Path


# Ensure local extension packages and the LeRobot submodule are on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"

for path in (SRC_DIR, LEROBOT_SRC):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

