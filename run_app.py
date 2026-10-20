"""Development and packaged application entry point."""

from pathlib import Path
import sys

if not getattr(sys, "frozen", False):
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / ".vendor"))
    sys.path.insert(0, str(root.parent / ".deps"))

from backend.app import main


if __name__ == "__main__":
    main()
