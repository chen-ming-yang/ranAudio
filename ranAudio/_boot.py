"""Put this project and MiniCPM-o-Demo on sys.path. Import before other ranAudio modules."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT = str(Path(__file__).resolve().parents[1])
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from ranAudio.paths import bootstrap_sys_path  # noqa: E402

bootstrap_sys_path()
