"""Install the same native bindings before fingerprinting either checkout."""

import argparse
import platform
import shutil
import sys

from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("roots", type=Path, nargs="+")
    args = parser.parse_args()
    system = {"win32": "windows", "darwin": "macos", "linux": "linux"}[sys.platform]
    architecture = "arm64" if platform.machine().lower() in {"arm64", "aarch64"} else "x64"
    relative = Path(system) / architecture / f"py{sys.version_info.major}{sys.version_info.minor}"
    source = args.bindings / relative
    if not source.is_dir():
        raise FileNotFoundError(f"Missing bindings for {relative}")
    for root in args.roots:
        shutil.copytree(source, root / "src/addons/character_dna/bindings" / relative, dirs_exist_ok=True)
        (root / "reports").mkdir(exist_ok=True)


if __name__ == "__main__":
    main()
