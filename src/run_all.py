"""
Master pipeline: end-to-end reproducibility.
Run from project root:  python src/run_all.py

Steps:
  1. download.py    — Fetch all raw data from APIs
  2. fix_codes.py   — Convert ISO2->ISO3, build country classification
  3. build_panel.py — Merge all data into analysis panel
  4. run_analysis.py— GSC estimation + robustness checks
  5. visualize.py   — Generate all figures and tables
"""
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "src"


def run_step(name, script):
    print(f"\n{'#'*70}\n# STEP: {name}\n{'#'*70}")
    result = subprocess.run([sys.executable, str(SRC / script)],
                            cwd=str(PROJECT), capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: {name} failed (exit code {result.returncode})")
        return False
    return True


def main():
    steps = [
        ("Download raw data",         "download.py"),
        ("Fix country codes",         "fix_codes.py"),
        ("Build analysis panel",      "build_panel.py"),
        ("Run GSC analysis",          "run_analysis.py"),
        ("Generate visualizations",   "visualize.py"),
    ]

    for name, script in steps:
        if not run_step(name, script):
            print(f"\nPipeline stopped at: {name}")
            return 1

    print(f"\n{'#'*70}")
    print("# PIPELINE COMPLETE — All outputs in output/ and data/processed/")
    print(f"{'#'*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
