"""Environment diagnostics. Same as `python -m exocortex --doctor`."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exocortex.config import load_config  # noqa: E402
from exocortex.doctor import run_doctor  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_doctor(load_config(sys.argv[1] if len(sys.argv) > 1 else None)))
