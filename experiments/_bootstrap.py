from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = project_root.as_posix()
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

