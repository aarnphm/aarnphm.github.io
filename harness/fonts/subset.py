#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["fonttools==4.68.0", "brotli>=1.1"]
# ///
"""Entry point for `pnpm fonts:subset`. Usage and rerun notes: harness/fonts/subset/__init__.py (also --help)."""

import sys

from subset.cli import main

if __name__ == '__main__':
  sys.exit(main())
