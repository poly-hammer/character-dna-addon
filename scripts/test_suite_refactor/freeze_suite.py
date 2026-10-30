"""Freeze current tracked/untracked sources and installed bindings for an isolated run."""

import argparse
import json
import shutil
import subprocess

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def freeze(destination: Path) -> None:
    destination = destination.resolve()
    allowed = (ROOT / "reports/test-suite-refactor").resolve()
    if not destination.is_relative_to(allowed) or destination == allowed or destination.exists():
        raise ValueError("Choose a new subdirectory inside reports/test-suite-refactor")
    destination.mkdir(parents=True)
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("Git executable was not found")
    revisions = {}
    for repo in (ROOT, ROOT / "src/addons/character_dna/editors"):
        prefix = repo.relative_to(ROOT)
        revisions[prefix.as_posix()] = subprocess.check_output(  # noqa: S603 - local Git, fixed read-only arguments
            [git, "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        raw = subprocess.check_output(  # noqa: S603 - local Git, fixed read-only arguments
            [git, "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=repo
        )
        for name in set(raw.decode().split("\0")) - {""}:
            source = repo / name
            if not source.is_file():
                continue
            target = destination / prefix / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    bindings = Path("src/addons/character_dna/bindings")
    for system in ("windows", "linux", "macos"):
        source = ROOT / bindings / system
        if source.is_dir():
            shutil.copytree(source, destination / bindings / system, dirs_exist_ok=True)
    (destination / "baseline-manifest.json").write_text(json.dumps(revisions, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    freeze(parser.parse_args().destination)
