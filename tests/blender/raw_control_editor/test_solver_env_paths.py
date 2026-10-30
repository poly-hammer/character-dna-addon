"""Unit tests for the per-device solver-env path + readiness-marker helpers
(:mod:`venv_env`).

Pure filesystem logic -- no Blender, no torch, no network. These lock in the
model that powers switchable per-device solver environments: each device gets a
named subdir, and "ready" means both the venv interpreter and the readiness
marker exist.
"""

from __future__ import annotations

from pathlib import Path

from character_dna.editors.raw_control_editor.solver_worker import venv_env


def test_resolve_env_dir_is_device_subdir(tmp_path: Path) -> None:
    assert venv_env.resolve_env_dir(tmp_path, "cuda") == tmp_path / "cuda"
    assert venv_env.resolve_env_dir(tmp_path, "cpu") == tmp_path / "cpu"
    assert venv_env.resolve_env_dir(tmp_path, "auto") == tmp_path / "auto"


def test_env_is_ready_requires_both_python_and_marker(tmp_path: Path) -> None:
    venv_dir = tmp_path / "cpu"
    python = venv_env.env_python_path(venv_dir)
    python.parent.mkdir(parents=True, exist_ok=True)
    # Nothing yet.
    assert venv_env.env_is_ready(venv_dir) is False
    # Interpreter present but not marked ready (e.g. torch install failed).
    python.write_text("#!/fake/python")
    assert venv_env.env_is_ready(venv_dir) is False
    # Marker written -> ready.
    venv_env.write_ready_marker(venv_dir, {"torch": "2.0", "size_bytes": 1})
    assert venv_env.env_is_ready(venv_dir) is True


def test_marker_round_trip(tmp_path: Path) -> None:
    venv_dir = tmp_path / "cuda"
    venv_dir.mkdir()
    assert venv_env.read_ready_marker(venv_dir) is None
    info = {"torch": "2.9.0", "numpy": "2.1", "device": "cpu", "size_bytes": 42}
    venv_env.write_ready_marker(venv_dir, info)
    assert venv_env.read_ready_marker(venv_dir) == info


def test_read_ready_marker_bad_json_returns_none(tmp_path: Path) -> None:
    venv_dir = tmp_path / "cpu"
    venv_dir.mkdir()
    (venv_dir / venv_env._READY_MARKER).write_text("{not valid json")
    assert venv_env.read_ready_marker(venv_dir) is None


def test_dir_size_bytes_sums_all_files(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    sub = tmp_path / "lib"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"y" * 25)
    assert venv_env.dir_size_bytes(tmp_path) == 35


def test_dir_size_bytes_missing_path_is_zero(tmp_path: Path) -> None:
    assert venv_env.dir_size_bytes(tmp_path / "does-not-exist") == 0


def test_human_size_formats() -> None:
    from character_dna.editors.raw_control_editor import dependency_extraction as de

    assert de.human_size(0) == "0.0 B"
    assert de.human_size(None) == "0.0 B"
    assert de.human_size(1024) == "1.0 KB"
    assert de.human_size(5 * 1024**3) == "5.0 GB"
