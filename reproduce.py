"""Run the numerical reproduction commands for the current paper."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    package = Path(__file__).resolve().parent / "research/normalized_score"
    sys.path.insert(0, str(package))
    runpy.run_path(str(package / "reproduce.py"), run_name="__main__")
