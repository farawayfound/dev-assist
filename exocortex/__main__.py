"""`python -m exocortex` entry point."""
from __future__ import annotations

import sys

from exocortex.app import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
