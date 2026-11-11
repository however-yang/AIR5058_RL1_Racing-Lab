"""Compatibility entry point for the student trainer."""

from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent / "student"))
    from train import parser, train

    train(parser().parse_args())
