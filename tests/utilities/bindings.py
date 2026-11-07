"""Keep local test bindings in sync with the checkout used by CI."""

import importlib.metadata
import shutil

from pathlib import Path


def _copy_changed(source: str, destination: str) -> str:
    target = Path(destination)
    if not target.is_file() or Path(source).read_bytes() != target.read_bytes():
        shutil.copy2(source, destination)
    return destination


def install_test_bindings(source: Path, destination: Path) -> None:
    """Refresh a sibling checkout, or use already bundled bindings if it is absent."""
    try:
        candidate = importlib.metadata.distribution("character-dna-core")
    except importlib.metadata.PackageNotFoundError:
        candidate = None
    if source.is_dir():
        # Skip identical binaries: another Blender process may have them loaded.
        # An installed candidate takes precedence over the released Core checkout.
        # Never replace it with older bytes first, especially when Blender holds it.
        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True,
            copy_function=_copy_changed,
            ignore=shutil.ignore_patterns("character_dna_core*") if candidate else None,
        )
    elif not destination.is_dir():
        raise FileNotFoundError(f'The bindings in "{destination}" are missing. Please add them to run the tests.')

    # Reusable CI installs the exact candidate wheel into this test environment.
    # Production imports are package-scoped, so stage its bytes without importing
    # a second, top-level copy of the native module.
    if candidate is None:
        return
    files = list(candidate.files or ())
    binaries = [path for path in files if Path(path).suffix in {".pyd", ".so"}]
    if len(binaries) != 1:
        raise RuntimeError("The installed Character DNA Core wheel must contain one native module")
    binary = binaries[0]
    for stale in destination.glob("character_dna_core*"):
        if stale.suffix in {".pyd", ".so"} and stale.name != Path(binary).name:
            raise RuntimeError("Competing converter binaries in the test bindings folder")
    _copy_changed(str(candidate.locate_file(binary)), str(destination / Path(binary).name))
    notices = next(path for path in files if str(path) == "THIRD_PARTY_NOTICES.txt")
    _copy_changed(str(candidate.locate_file(notices)), str(destination / "character_dna_core_NOTICES.txt"))
    stub = next(path for path in files if str(path) == "character_dna_core.pyi")
    _copy_changed(str(candidate.locate_file(stub)), str(destination / "character_dna_core.pyi"))
