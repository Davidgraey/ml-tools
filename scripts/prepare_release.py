"""
Prepare polyergalio for PyPI publication.

Runs the test suite with coverage, builds the wheel and sdist, and
validates both with twine. Stops at the first failing step.

Requires the dev extras: pip install -e ".[dev]"

python scripts/prepare_release.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"


def run_step(title: str, command: list[str]) -> None:
    """
    Run a command from the repo root, stopping the script on failure.

    Parameters
    ----------
    title : printed as a section header before the command runs
    command : argv list passed to subprocess
    """
    print(f"\n=== {title} ===")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\n{title} failed (exit {result.returncode})")


def run_tests_with_coverage() -> None:
    run_step("tests", [sys.executable, "-m", "coverage", "run", "-m", "pytest"])
    run_step("coverage report", [sys.executable, "-m", "coverage", "report", "-m"])


def build_distributions() -> None:
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    run_step("build wheel + sdist", [sys.executable, "-m", "build"])


def check_distributions() -> None:
    dist_files = sorted(str(path) for path in DIST_DIR.glob("*"))
    if not dist_files:
        sys.exit("no distributions found in dist/")
    run_step("twine check", [sys.executable, "-m", "twine", "check", *dist_files])


if __name__ == "__main__":
    run_tests_with_coverage()
    build_distributions()
    check_distributions()
    print("\nall checks passed -- dist/ is ready for `twine upload`.")
