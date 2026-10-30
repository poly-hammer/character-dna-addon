"""Unit tests for the PyTorch wheel-index selection in the solver venv
provisioner (:mod:`venv_env`).

Pure logic -- no Blender, no torch, no network. The NVIDIA-driver probe and the
platform are monkeypatched so the index math can be exercised for every driver /
device combination on any CI host.
"""

from __future__ import annotations

import pytest

from character_dna.editors.raw_control_editor.solver_worker import venv_env


def _cu(suffix: str) -> str:
    return f"{venv_env.TORCH_DOWNLOAD_BASE}/{suffix}"


@pytest.fixture
def linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a non-macOS platform so the CUDA / CPU selection branch runs."""
    monkeypatch.setattr(venv_env.sys, "platform", "linux")


def _patch_driver(monkeypatch: pytest.MonkeyPatch, version: tuple[int, int] | None) -> None:
    monkeypatch.setattr(venv_env, "detect_nvidia_cuda_version", lambda: version)


@pytest.mark.parametrize(
    ("driver", "expected"),
    [
        ((13, 3), "cu132"),  # newer-than-listed driver -> newest wheel
        ((13, 2), "cu132"),
        ((13, 0), "cu130"),
        ((12, 9), "cu126"),
        ((12, 6), "cu126"),
        ((12, 4), "cu121"),
        ((12, 1), "cu121"),
        ((11, 8), "cu118"),
    ],
)
def test_torch_index_url_picks_newest_supported(
    monkeypatch: pytest.MonkeyPatch, linux: None, driver: tuple[int, int], expected: str
) -> None:
    _patch_driver(monkeypatch, driver)
    assert venv_env.torch_index_url("auto") == _cu(expected)


def test_torch_index_url_driver_below_all_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch, linux: None) -> None:
    _patch_driver(monkeypatch, (11, 0))
    assert venv_env.torch_index_url("auto") == venv_env.TORCH_CPU_INDEX


def test_torch_index_url_no_driver_auto_is_cpu(monkeypatch: pytest.MonkeyPatch, linux: None) -> None:
    _patch_driver(monkeypatch, None)
    assert venv_env.torch_index_url("auto") == venv_env.TORCH_CPU_INDEX


def test_torch_index_url_no_driver_forced_cuda_uses_newest(monkeypatch: pytest.MonkeyPatch, linux: None) -> None:
    _patch_driver(monkeypatch, None)
    assert venv_env.torch_index_url("cuda") == _cu("cu132")


def test_torch_index_url_cpu_and_mps_use_cpu_index(monkeypatch: pytest.MonkeyPatch, linux: None) -> None:
    _patch_driver(monkeypatch, (13, 3))  # ignored for cpu / mps
    assert venv_env.torch_index_url("cpu") == venv_env.TORCH_CPU_INDEX
    assert venv_env.torch_index_url("mps") == venv_env.TORCH_CPU_INDEX


def test_torch_index_url_macos_always_pypi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(venv_env.sys, "platform", "darwin")
    for device in ("auto", "cuda", "mps", "cpu"):
        assert venv_env.torch_index_url(device) == venv_env.PYPI_INDEX


def test_candidates_auto_orders_supported_then_cpu(monkeypatch: pytest.MonkeyPatch, linux: None) -> None:
    _patch_driver(monkeypatch, (13, 3))
    assert venv_env.torch_index_candidates("auto") == [
        _cu("cu132"),
        _cu("cu130"),
        _cu("cu126"),
        _cu("cu121"),
        _cu("cu118"),
        venv_env.TORCH_CPU_INDEX,
    ]


def test_candidates_forced_cuda_omits_cpu_fallback(monkeypatch: pytest.MonkeyPatch, linux: None) -> None:
    _patch_driver(monkeypatch, (12, 6))
    assert venv_env.torch_index_candidates("cuda") == [_cu("cu126"), _cu("cu121"), _cu("cu118")]


def test_candidates_macos_pypi_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(venv_env.sys, "platform", "darwin")
    assert venv_env.torch_index_candidates("auto") == [venv_env.PYPI_INDEX]


def test_candidates_cpu_device_is_cpu_only(monkeypatch: pytest.MonkeyPatch, linux: None) -> None:
    _patch_driver(monkeypatch, (13, 3))
    assert venv_env.torch_index_candidates("cpu") == [venv_env.TORCH_CPU_INDEX]


def test_detect_nvidia_cuda_version_parses_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(venv_env.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    class _Proc:
        returncode = 0
        stdout = "+---------+\n| NVIDIA-SMI 999   Driver Version: 999   CUDA Version: 13.3 |\n"

    monkeypatch.setattr(venv_env.subprocess, "run", lambda *_a, **_k: _Proc())
    assert venv_env.detect_nvidia_cuda_version() == (13, 3)


def test_detect_nvidia_cuda_version_no_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(venv_env.shutil, "which", lambda _: None)
    assert venv_env.detect_nvidia_cuda_version() is None
