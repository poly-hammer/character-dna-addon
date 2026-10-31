"""Blender probe shutdown must preserve both successes and assertion failures."""

import subprocess
import sys

from pathlib import Path

import pytest


@pytest.mark.parametrize("fails", [False, True])
def test_blender_process_preserves_result(fails: bool):
    script = 'import bpy\nprint("PROBE_STARTED", flush=True)\n'
    script += 'raise AssertionError("deliberate probe failure")' if fails else 'print("PROBE_PASSED")'
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(Path(__file__).parents[1] / "utilities" / "process.py"), script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == int(fails), result.stdout + result.stderr
    assert "PROBE_STARTED" in result.stdout
    if fails:
        assert "AssertionError: deliberate probe failure" in result.stderr
        assert "PROBE_PASSED" not in result.stdout
    else:
        assert "PROBE_PASSED" in result.stdout


def test_runtime_can_be_imported_before_addon_registration():
    root = Path(__file__).resolve().parents[2]
    script = (
        f"import sys; sys.path.insert(0, {str(root / 'src/addons')!r})\n"
        "from character_dna.runtime import engine, controller\n"
        "import character_dna\n"
        "assert character_dna.native_runtime is controller\n"
        "print('IMPORT_ORDER_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, str(root / "tests/utilities/process.py"), script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "IMPORT_ORDER_OK" in result.stdout
