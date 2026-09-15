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
        [sys.executable, str(Path(__file__).parent / "utilities" / "process.py"), script],
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
