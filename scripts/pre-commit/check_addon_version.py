"""Pre-commit hook to auto-bump and sync addon version across all version files.

This hook ensures that the version in:
- src/addons/<addon_name>/__init__.py (bl_info["version"])
- src/addons/<addon_name>/blender_manifest.toml
- pyproject.toml (at repo root)
- uv.lock (at repo root)

Are all in sync and auto-bumps the patch version when addon files are changed.

Usage:
    python check_addon_version.py <addon_name> [--watch <glob_pattern>...]
    python check_addon_version.py <addon_name> --set-version <version> [--stage]

Example:
    python check_addon_version.py character_dna --watch "src/addons/character_dna/**/*.py"
    python check_addon_version.py character_dna --set-version 1.0.0
    python check_addon_version.py character_dna --set-version 1.0.0 --stage
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


class Version(NamedTuple):
    """Semantic version representation."""

    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def to_tuple_str(self) -> str:
        """Return version as Python tuple string, e.g., '(0, 5, 4)'."""
        return f"({self.major}, {self.minor}, {self.patch})"

    def bumped_patch(self) -> Version:
        """Return a new Version with the patch number incremented."""
        return Version(self.major, self.minor, self.patch + 1)

    @classmethod
    def from_string(cls, version_str: str) -> Version:
        """Parse version from string like '0.5.4'."""
        parts = version_str.strip().split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version string: {version_str}")
        return cls(int(parts[0]), int(parts[1]), int(parts[2]))

    @classmethod
    def from_tuple_str(cls, tuple_str: str) -> Version:
        """Parse version from tuple string like '(0, 5, 4)'."""
        # Remove parentheses and split by comma
        clean = tuple_str.strip("() ")
        parts = [p.strip() for p in clean.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid version tuple: {tuple_str}")
        return cls(int(parts[0]), int(parts[1]), int(parts[2]))


def get_repo_root() -> Path:
    """Return the absolute path to the top-level (parent) git repository root.

    When called from inside a submodule, walks up to the superproject root.
    """
    # If inside a submodule, this returns the superproject working tree
    result = subprocess.run(
        ["git", "rev-parse", "--show-superproject-working-tree"],
        capture_output=True,
        text=True,
        check=True,
    )
    superproject = result.stdout.strip()
    if superproject:
        return Path(superproject)

    # Not a submodule — use the current repo root
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def get_staged_files() -> list[Path]:
    """Get list of files staged for commit."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(f) for f in result.stdout.strip().split("\n") if f]


def matches_any_pattern(file_path: Path, patterns: list[str]) -> bool:
    """Check if a file path matches any of the given glob patterns."""
    # Normalize path to use forward slashes for consistent matching
    path_str = str(file_path).replace("\\", "/")
    for pattern in patterns:
        # Normalize pattern too
        normalized_pattern = pattern.replace("\\", "/")
        if fnmatch.fnmatch(path_str, normalized_pattern):
            return True
    return False


def has_relevant_changes(
    staged_files: list[Path],
    watch_patterns: list[str],
    version_files: list[Path],
) -> bool:
    """Check if any staged files match watch patterns (excluding version files)."""
    version_file_names = {p.name for p in version_files}

    for file_path in staged_files:
        # Skip version files
        if file_path.name in version_file_names:
            continue

        # Check if file matches any watch pattern
        if matches_any_pattern(file_path, watch_patterns):
            return True

    return False


def read_version_from_init(file_path: Path) -> Version:
    """Read version from addon __init__.py bl_info dict."""
    content = file_path.read_text(encoding="utf-8")
    # Match: "version": (0, 5, 4),
    pattern = r'"version"\s*:\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)'
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"Could not find version in {file_path}")
    return Version(int(match.group(1)), int(match.group(2)), int(match.group(3)))


def read_version_from_toml(file_path: Path) -> Version:
    """Read version from a TOML file (blender_manifest.toml or pyproject.toml)."""
    content = file_path.read_text(encoding="utf-8")
    # Match: version = "0.5.4"
    pattern = r'^version\s*=\s*"(\d+\.\d+\.\d+)"'
    match = re.search(pattern, content, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find version in {file_path}")
    return Version.from_string(match.group(1))


def read_version_from_uv_lock(file_path: Path, package_name: str) -> Version | None:
    """Read version from uv.lock for a specific package.

    Returns None if the lock file doesn't exist or package not found.
    """
    if not file_path.exists():
        return None

    content = file_path.read_text(encoding="utf-8")

    # Match the package block: [[package]]\nname = "package-name"\nversion = "0.5.4"
    # Use a pattern that finds the package by name and extracts its version
    pattern = rf'\[\[package\]\]\s*\nname\s*=\s*"{re.escape(package_name)}"\s*\nversion\s*=\s*"(\d+\.\d+\.\d+)"'
    match = re.search(pattern, content, re.MULTILINE)
    if not match:
        return None
    return Version.from_string(match.group(1))


def write_version_to_init(file_path: Path, version: Version) -> None:
    """Write version to addon __init__.py bl_info dict."""
    content = file_path.read_text(encoding="utf-8")
    pattern = r'("version"\s*:\s*)\(\d+\s*,\s*\d+\s*,\s*\d+\)'
    replacement = rf"\g<1>{version.to_tuple_str()}"
    new_content = re.sub(pattern, replacement, content)
    # Preserve LF line endings
    file_path.write_text(new_content, encoding="utf-8", newline="\n")


def write_version_to_toml(file_path: Path, version: Version) -> None:
    """Write version to a TOML file."""
    content = file_path.read_text(encoding="utf-8")
    pattern = r'^(version\s*=\s*)"[\d.]+"'
    replacement = rf'\g<1>"{version}"'
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    # Preserve LF line endings
    file_path.write_text(new_content, encoding="utf-8", newline="\n")


def write_version_to_uv_lock(
    file_path: Path, package_name: str, version: Version
) -> bool:
    """Write version to uv.lock for a specific package.

    Returns True if the file was updated, False if file doesn't exist or package not found.
    """
    if not file_path.exists():
        return False

    content = file_path.read_text(encoding="utf-8")

    # Match the package block and replace version
    pattern = rf'(\[\[package\]\]\s*\nname\s*=\s*"{re.escape(package_name)}"\s*\nversion\s*=\s*)"[\d.]+"'
    replacement = rf'\g<1>"{version}"'
    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)

    if count == 0:
        return False

    # Preserve LF line endings
    file_path.write_text(new_content, encoding="utf-8", newline="\n")
    return True


def stage_file(file_path: Path) -> None:
    """Add a file to the git staging area."""
    subprocess.run(["git", "add", str(file_path)], check=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Auto-bump and sync addon version across version files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s character_dna
  %(prog)s character_dna --watch "src/addons/character_dna/**/*.py"
  %(prog)s character_dna --watch "src/addons/character_dna/**/*.py" --watch "src/addons/character_dna/**/*.toml"
  %(prog)s character_dna --set-version 1.0.0
  %(prog)s character_dna --set-version 1.0.0 --stage
        """,
    )
    parser.add_argument(
        "addon_name",
        help="Name of the addon (e.g., 'character_dna')",
    )
    parser.add_argument(
        "--watch",
        action="append",
        dest="watch_patterns",
        default=[],
        metavar="PATTERN",
        help="Glob pattern for files that trigger version bump (can be specified multiple times)",
    )
    parser.add_argument(
        "--set-version",
        dest="set_version",
        metavar="VERSION",
        help="Set a specific version (e.g., '1.0.0') and sync across all version files",
    )
    parser.add_argument(
        "--stage",
        action="store_true",
        help="Stage the updated version files after setting version (use with --set-version)",
    )
    return parser.parse_args()


def sync_version_to_all_files(
    version: Version,
    addon_init_path: Path,
    blender_manifest_path: Path,
    pyproject_path: Path,
    uv_lock_path: Path,
    package_name: str,
    stage_files: bool = False,
) -> list[Path]:
    """Sync version to all version files.

    Returns list of files that were updated.
    """
    updated_files: list[Path] = []

    # Update main version files
    write_version_to_init(addon_init_path, version)
    updated_files.append(addon_init_path)
    print(f"  Updated {addon_init_path}")

    write_version_to_toml(blender_manifest_path, version)
    updated_files.append(blender_manifest_path)
    print(f"  Updated {blender_manifest_path}")

    write_version_to_toml(pyproject_path, version)
    updated_files.append(pyproject_path)
    print(f"  Updated {pyproject_path}")

    # Update uv.lock if it exists
    if write_version_to_uv_lock(uv_lock_path, package_name, version):
        updated_files.append(uv_lock_path)
        print(f"  Updated {uv_lock_path} for package '{package_name}'")

    # Stage files if requested
    if stage_files:
        for file_path in updated_files:
            stage_file(file_path)
        print("\nVersion files staged for commit.")

    return updated_files


def main() -> int:
    """Main entry point for the pre-commit hook.

    Returns:
        0 if successful, 1 if there was an error.
    """
    args = parse_args()

    repo_root = get_repo_root()

    # Build paths based on addon name
    addon_base_path = repo_root / "src/addons" / args.addon_name
    addon_init_path = addon_base_path / "__init__.py"
    blender_manifest_path = addon_base_path / "blender_manifest.toml"
    pyproject_path = repo_root / "pyproject.toml"
    uv_lock_path = repo_root / "uv.lock"

    # Convert addon name to package name format (underscores to hyphens)
    package_name = args.addon_name.replace("_", "-") + "-addon"

    version_files = [addon_init_path, blender_manifest_path, pyproject_path]

    # Check if version files exist
    missing_files = [p for p in version_files if not p.exists()]
    if missing_files:
        print("ERROR: One or more version files not found:")
        for p in missing_files:
            print(f"  - {p}")
        return 1

    # Handle --set-version mode (manual version sync, not pre-commit hook)
    if args.set_version:
        try:
            new_version = Version.from_string(args.set_version)
        except ValueError:
            print(f"ERROR: Invalid version format '{args.set_version}'")
            print("Version must be in format 'major.minor.patch' (e.g., '1.0.0')")
            return 1

        print(f"Setting version to: {new_version}")
        print()

        sync_version_to_all_files(
            version=new_version,
            addon_init_path=addon_init_path,
            blender_manifest_path=blender_manifest_path,
            pyproject_path=pyproject_path,
            uv_lock_path=uv_lock_path,
            package_name=package_name,
            stage_files=args.stage,
        )

        print(f"\nAll version files updated to {new_version}")
        return 0

    # Pre-commit hook mode: check staged files and auto-bump if needed
    # Default watch patterns if none provided
    watch_patterns = args.watch_patterns or [f"src/addons/{args.addon_name}/**/*"]

    print(f"Addon: {args.addon_name}")
    print(f"Watch patterns: {watch_patterns}")

    # Get staged files
    staged_files = get_staged_files()
    if not staged_files:
        print("No staged files.")
        return 0

    # Read current versions from all files
    try:
        init_version = read_version_from_init(addon_init_path)
        manifest_version = read_version_from_toml(blender_manifest_path)
        pyproject_version = read_version_from_toml(pyproject_path)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # Read uv.lock version if it exists
    uv_lock_version = read_version_from_uv_lock(uv_lock_path, package_name)

    print("Current versions:")
    print(f"  __init__.py:           {init_version}")
    print(f"  blender_manifest.toml: {manifest_version}")
    print(f"  pyproject.toml:        {pyproject_version}")
    if uv_lock_version:
        print(f"  uv.lock:               {uv_lock_version}")

    # Check if relevant files have changed (matching watch patterns, excluding version files)
    relevant_changes = has_relevant_changes(staged_files, watch_patterns, version_files)

    # Determine if version files are already staged
    resolved_version_files = {path.resolve() for path in version_files}
    version_files_staged = any(path.resolve() in resolved_version_files for path in staged_files)

    if relevant_changes and not version_files_staged:
        # Auto-bump patch version
        new_version = init_version.bumped_patch()
        print(
            f"\nRelevant files changed. Auto-bumping patch version: {init_version} -> {new_version}"
        )

        sync_version_to_all_files(
            version=new_version,
            addon_init_path=addon_init_path,
            blender_manifest_path=blender_manifest_path,
            pyproject_path=pyproject_path,
            uv_lock_path=uv_lock_path,
            package_name=package_name,
            stage_files=True,
        )

        print("Version files updated and staged.")
        return 0

    # Check if versions are in sync (include uv.lock if it exists)
    all_versions = [init_version, manifest_version, pyproject_version]
    if uv_lock_version:
        all_versions.append(uv_lock_version)

    if len(set(all_versions)) > 1:
        print("\nERROR: Version mismatch detected!")
        print("All version files must have the same version.")
        print(f"  __init__.py:           {init_version}")
        print(f"  blender_manifest.toml: {manifest_version}")
        print(f"  pyproject.toml:        {pyproject_version}")
        if uv_lock_version:
            print(f"  uv.lock:               {uv_lock_version}")
        print("\nPlease sync the versions manually or let this hook auto-bump by")
        print("unstaging the version files and committing your addon changes again.")
        return 1

    print("\nAll versions are in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
