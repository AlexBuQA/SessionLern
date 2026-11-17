"""Create a clean source-and-results archive suitable for handoff."""
from __future__ import annotations

import zipfile
from pathlib import Path


def should_include(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    excluded_dirs = {".venv", ".git", "__pycache__", ".pytest_cache", "outputs", ".ipynb_checkpoints", "verification"}
    if any(part in excluded_dirs for part in relative.parts):
        return False
    if len(relative.parts) == 1 and (path.name.startswith("terminal-log") or path.name.startswith("colab-")):
        return False
    if path.suffix.lower() in {".zip", ".pyc", ".pt", ".pth", ".safetensors", ".bin", ".ckpt"}:
        return False
    return path.is_file()


def main() -> None:
    root = Path(__file__).resolve().parent
    destination = root / "rl-session-assignment.zip"
    files = [path for path in root.rglob("*") if should_include(path, root)]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, Path("rl-session-assignment") / path.relative_to(root))
    print(f"Created {destination} with {len(files)} files ({destination.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
