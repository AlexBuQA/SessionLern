"""Run every track's tests from the repository root without training models."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    for track in ["track1_acrobot", "track2_inventory", "track3_countdown_grpo"]:
        print(f"\n=== {track} ===", flush=True)
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root / track, check=True)
    print("\nAll project checks passed.")


if __name__ == "__main__":
    main()
