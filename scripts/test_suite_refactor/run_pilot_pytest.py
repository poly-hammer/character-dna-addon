"""Run opt-in tests outside tests/ without the global Blender fixture bootstrap."""

import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "src/addons")]
import pytest  # noqa: E402 - bootstrap paths before loading the opt-in runner

from utilities.process import exit_blender  # noqa: E402


exit_blender(pytest.main(sys.argv[1:]))
