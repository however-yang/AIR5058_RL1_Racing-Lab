"""Compatibility entry point for the student evaluator."""

from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent / "student"))
    from evaluate import main

    main()
