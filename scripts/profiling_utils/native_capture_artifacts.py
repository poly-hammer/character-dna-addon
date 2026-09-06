"""Preserve the exact native experiment source and build provenance without committing."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess

from pathlib import Path


def git(root: Path, *arguments: str) -> str:
    """Read git metadata without changing the index or working files."""
    executable = shutil.which("git")
    if executable is None:
        raise FileNotFoundError("Git executable not found")
    return subprocess.run(  # noqa: S603
        [executable, "-C", str(root), *arguments], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture(output: Path, addon: Path, blender: Path, sdk: Path, build: Path) -> None:
    """Archive only source diffs/new files and build configuration, not large assets."""
    output.mkdir(parents=True, exist_ok=False)
    repositories = {}
    for name, root in (("addon", addon), ("blender", blender), ("sdk", sdk)):
        patch = git(root, "diff", "HEAD", "--binary")
        patch_path = output / f"{name}.patch"
        patch_path.write_text(patch, encoding="utf-8")
        changed = set(git(root, "diff", "--name-only", "HEAD").splitlines())
        untracked = set(git(root, "ls-files", "--others", "--exclude-standard").splitlines())
        copied = {}
        for relative in sorted(changed | untracked):
            path = root / relative
            if not path.is_file() or path.suffix.lower() not in {
                ".py",
                ".ps1",
                ".cc",
                ".hh",
                ".h",
                ".cpp",
                ".cmake",
                ".md",
                ".txt",
            }:
                continue
            destination = output / "source" / name / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied[relative] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        repositories[name] = {
            "root": str(root),
            "revision": git(root, "rev-parse", "HEAD").strip(),
            "branch": git(root, "branch", "--show-current").strip(),
            "patch_sha256": file_hash(patch_path),
            "source_files": copied,
        }
    binaries = {}
    for relative in (
        "candidate-install/blender.exe",
        "baseline-install/blender.exe",
        "sdk-install/lib/riglogic413_2_7.lib",
    ):
        path = build / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        binaries[relative] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    for name in ("candidate-ninja", "baseline-ninja", "sdk-ninja"):
        source = build / name / "CMakeCache.txt"
        shutil.copy2(source, output / f"{name}-CMakeCache.txt")
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": 1, "repositories": repositories, "binaries": binaries}, stream, indent=2)
    print(f"ARTIFACTS CAPTURED: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--addon", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--blender", type=Path, default=Path("E:/repos/blender"))
    parser.add_argument("--sdk", type=Path, default=Path("E:/repos/OpenRigLogic"))
    parser.add_argument("--build", type=Path, default=Path("E:/repos/build_riglogic_native"))
    args = parser.parse_args()
    capture(args.output, args.addon, args.blender, args.sdk, args.build)
