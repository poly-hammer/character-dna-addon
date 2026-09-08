"""
Export profiling results for CI snapshots and analysis.

This module provides exporters for profiling results in various formats:
- JSON: Complete results with all metadata for CI comparison
- CSV: Tabular format for spreadsheet analysis
- Markdown: Human-readable reports

Example:
    from scripts.profiling_utils import run_profiler
    from scripts.profiling_utils.exporters import export_snapshot

    results = run_profiler(iterations=100)
    export_snapshot(results, "reports/profiling", format="all")
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .profile_rig_evaluation import ProfileResults, TimingResult


@dataclass
class HardwareInfo:
    """Hardware specifications for the profiling machine."""

    cpu_name: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    cpu_frequency_mhz: float = 0.0
    ram_total_gb: float = 0.0
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0

    def to_fingerprint(self) -> str:
        """Generate a fingerprint string for hardware comparison."""
        return f"{self.cpu_name}|{self.cpu_cores_physical}|{self.ram_total_gb:.0f}GB|{self.gpu_name}"


def get_hardware_info() -> HardwareInfo:
    """Collect hardware information about the current machine."""
    info = HardwareInfo()

    # CPU info
    info.cpu_name = platform.processor() or "unknown"
    info.cpu_cores_physical = os.cpu_count() or 0
    info.cpu_cores_logical = os.cpu_count() or 0

    # Try to get more detailed CPU info on Windows
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            info.cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            info.cpu_frequency_mhz = winreg.QueryValueEx(key, "~MHz")[0]
            winreg.CloseKey(key)
        except (OSError, ImportError):
            pass

    # Try psutil for more detailed info (optional dependency)
    try:
        import psutil

        info.cpu_cores_physical = psutil.cpu_count(logical=False) or info.cpu_cores_physical
        info.cpu_cores_logical = psutil.cpu_count(logical=True) or info.cpu_cores_logical
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            info.cpu_frequency_mhz = cpu_freq.max or cpu_freq.current
        info.ram_total_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        # Fallback for RAM on Windows without psutil
        if sys.platform == "win32":
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32
                c_ulonglong = ctypes.c_ulonglong

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", c_ulonglong),
                        ("ullAvailPhys", c_ulonglong),
                        ("ullTotalPageFile", c_ulonglong),
                        ("ullAvailPageFile", c_ulonglong),
                        ("ullTotalVirtual", c_ulonglong),
                        ("ullAvailVirtual", c_ulonglong),
                        ("ullAvailExtendedVirtual", c_ulonglong),
                    ]

                mem_status = MEMORYSTATUSEX()
                mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))
                info.ram_total_gb = round(mem_status.ullTotalPhys / (1024**3), 1)
            except (OSError, AttributeError):
                pass

    # GPU info - try Blender first, then fallback methods
    try:
        import bpy

        # Get GPU info from Blender's system info
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons.get("cycles")
        if cycles_prefs:
            devices = cycles_prefs.preferences.get_devices()
            if devices:  # Check if devices is not None
                for device_type in devices:
                    for device in device_type:
                        if device.type in ("CUDA", "OPTIX", "HIP", "METAL", "ONEAPI"):
                            info.gpu_name = device.name
                            break
                    if info.gpu_name:
                        break

        # Fallback to system GPU info from Blender
        if not info.gpu_name:
            gpu_backend = getattr(bpy.context.preferences.system, "gpu_backend", None)
            if gpu_backend:
                info.gpu_name = gpu_backend
    except (ImportError, AttributeError):
        pass

    # Windows fallback for GPU
    if not info.gpu_name and sys.platform == "win32":
        try:
            import subprocess

            result = subprocess.check_output(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            lines = [line.strip() for line in result.split("\n") if line.strip() and line.strip() != "Name"]
            if lines:
                info.gpu_name = lines[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    return info


@dataclass
class SnapshotMetadata:
    """Metadata for a profiling snapshot."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    python_version: str = field(default_factory=lambda: sys.version)
    platform: str = field(default_factory=lambda: platform.platform())
    platform_system: str = field(default_factory=lambda: platform.system())
    platform_machine: str = field(default_factory=lambda: platform.machine())
    blender_version: str = ""
    git_commit: str = ""
    git_branch: str = ""
    iterations: int = 0
    warmup: int = 0
    # Hardware info
    hardware: dict = field(default_factory=dict)


def get_blender_version() -> str:
    """Get the current Blender version string."""
    try:
        import bpy

        return f"{bpy.app.version_string}"
    except ImportError:
        return "unknown"


def get_git_info() -> tuple[str, str]:
    """Get the current git commit hash and branch name."""
    import subprocess

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        branch = "unknown"

    return commit, branch


def timing_result_to_dict(timing: TimingResult) -> dict[str, Any]:
    """Convert a TimingResult to a dictionary for export."""
    return {
        "name": timing.name,
        "count": timing.count,
        "mean_ms": round(timing.mean_ms, 4),
        "median_ms": round(timing.median_ms, 4),
        "min_ms": round(timing.min_ms, 4),
        "max_ms": round(timing.max_ms, 4),
        "p95_ms": round(timing.p95_ms, 4),
    }


def riglogic_stats_to_dict(stats: Any) -> dict[str, Any]:
    """Convert RigLogicStats to a dictionary for export."""
    return {
        "calculation_type": stats.calculation_type,
        "floating_point_type": stats.floating_point_type,
        "rbf_solver_count": stats.rbf_solver_count,
        "neural_network_count": stats.neural_network_count,
        "psd_count": stats.psd_count,
        "blend_shape_channel_count": stats.blend_shape_channel_count,
        "animated_map_count": stats.animated_map_count,
        "joint_count": stats.joint_count,
        "joint_delta_value_count": stats.joint_delta_value_count,
    }


def results_to_dict(
    results: ProfileResults,
    iterations: int = 0,
    warmup: int = 0,
) -> dict[str, Any]:
    """Convert ProfileResults to a complete dictionary for export."""
    commit, branch = get_git_info()
    hardware = get_hardware_info()

    metadata = SnapshotMetadata(
        blender_version=get_blender_version(),
        git_commit=commit,
        git_branch=branch,
        iterations=iterations,
        warmup=warmup,
        hardware=asdict(hardware),
    )

    return {
        "metadata": {**asdict(metadata), "benchmark_schema": 2},
        "timings": {
            "head": {
                "native_evaluation": timing_result_to_dict(results.head_evaluation),
            },
            "body": {
                "native_evaluation": timing_result_to_dict(results.body_evaluation),
            },
            "full_evaluation": timing_result_to_dict(results.full_evaluation),
        },
        "riglogic_stats": {
            "head": riglogic_stats_to_dict(results.head_stats),
            "body": riglogic_stats_to_dict(results.body_stats),
        },
        "summary": {
            "native_head_ms": round(results.head_evaluation.mean_ms, 4),
            "native_body_ms": round(results.body_evaluation.mean_ms, 4),
            "full_evaluation_ms": round(results.full_evaluation.mean_ms, 4),
            "theoretical_fps": round(1000 / results.full_evaluation.mean_ms, 1)
            if results.full_evaluation.mean_ms > 0
            else 0.0,
        },
    }


def export_json(results: ProfileResults, output_path: Path, iterations: int = 0, warmup: int = 0) -> Path:
    """
    Export profiling results to JSON format.

    Args:
        results: The ProfileResults to export.
        output_path: Directory to write the output file.
        iterations: Number of iterations run.
        warmup: Number of warmup iterations.

    Returns:
        Path to the created JSON file.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    data = results_to_dict(results, iterations, warmup)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"profiling_snapshot_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return filename


def export_csv(results: ProfileResults, output_path: Path, iterations: int = 0, warmup: int = 0) -> Path:
    """
    Export profiling results to CSV format.

    Args:
        results: The ProfileResults to export.
        output_path: Directory to write the output file.
        iterations: Number of iterations run.
        warmup: Number of warmup iterations.

    Returns:
        Path to the created CSV file.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    data = results_to_dict(results, iterations, warmup)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"profiling_snapshot_{timestamp}.csv"

    rows = []

    # Flatten timings for CSV
    for category, timings in data["timings"].items():
        if isinstance(timings, dict) and "name" in timings:
            # Single timing (full_evaluation)
            rows.append(
                {
                    "category": category,
                    "operation": timings["name"],
                    "mean_ms": timings["mean_ms"],
                    "median_ms": timings["median_ms"],
                    "min_ms": timings["min_ms"],
                    "max_ms": timings["max_ms"],
                    "p95_ms": timings["p95_ms"],
                    "count": timings["count"],
                }
            )
        else:
            # Nested timings (head/body)
            for timing in timings.values():
                rows.append(
                    {
                        "category": category,
                        "operation": timing["name"],
                        "mean_ms": timing["mean_ms"],
                        "median_ms": timing["median_ms"],
                        "min_ms": timing["min_ms"],
                        "max_ms": timing["max_ms"],
                        "p95_ms": timing["p95_ms"],
                        "count": timing["count"],
                    }
                )

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "operation", "mean_ms", "median_ms", "min_ms", "max_ms", "p95_ms", "count"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return filename


def export_markdown(results: ProfileResults, output_path: Path, iterations: int = 0, warmup: int = 0) -> Path:
    """
    Export profiling results to Markdown format.

    Args:
        results: The ProfileResults to export.
        output_path: Directory to write the output file.
        iterations: Number of iterations run.
        warmup: Number of warmup iterations.

    Returns:
        Path to the created Markdown file.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    data = results_to_dict(results, iterations, warmup)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"profiling_snapshot_{timestamp}.md"

    lines = [
        "# Rig Evaluation Profiling Report",
        "",
        "## Metadata",
        "",
        f"- **Timestamp**: {data['metadata']['timestamp']}",
        f"- **Blender Version**: {data['metadata']['blender_version']}",
        f"- **Git Commit**: {data['metadata']['git_commit']}",
        f"- **Git Branch**: {data['metadata']['git_branch']}",
        f"- **Iterations**: {data['metadata']['iterations']} (+ {data['metadata']['warmup']} warmup)",
        "",
        "## Summary",
        "",
        f"- **Native Head Callback**: {data['summary']['native_head_ms']:.3f} ms",
        f"- **Native Body Callback**: {data['summary']['native_body_ms']:.3f} ms",
        f"- **Full Evaluation**: {data['summary']['full_evaluation_ms']:.3f} ms",
        f"- **Theoretical FPS**: {data['summary']['theoretical_fps']:.1f}",
        "",
        "## Head Timings",
        "",
        "| Operation | Mean (ms) | Median (ms) | P95 (ms) | Max (ms) |",
        "|-----------|-----------|-------------|----------|----------|",
    ]

    for _, timing in data["timings"]["head"].items():
        lines.append(
            f"| {timing['name']} | {timing['mean_ms']:.3f} | {timing['median_ms']:.3f} | {timing['p95_ms']:.3f} | {timing['max_ms']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Body Timings",
            "",
            "| Operation | Mean (ms) | Median (ms) | P95 (ms) | Max (ms) |",
            "|-----------|-----------|-------------|----------|----------|",
        ]
    )

    for _, timing in data["timings"]["body"].items():
        lines.append(
            f"| {timing['name']} | {timing['mean_ms']:.3f} | {timing['median_ms']:.3f} | {timing['p95_ms']:.3f} | {timing['max_ms']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## RigLogic Stats",
            "",
            "### Head",
            f"- Joints: {data['riglogic_stats']['head']['joint_count']}",
            f"- BlendShapes: {data['riglogic_stats']['head']['blend_shape_channel_count']}",
            f"- RBF Solvers: {data['riglogic_stats']['head']['rbf_solver_count']}",
            f"- Neural Networks: {data['riglogic_stats']['head']['neural_network_count']}",
            "",
            "### Body",
            f"- Joints: {data['riglogic_stats']['body']['joint_count']}",
            f"- BlendShapes: {data['riglogic_stats']['body']['blend_shape_channel_count']}",
            f"- RBF Solvers: {data['riglogic_stats']['body']['rbf_solver_count']}",
            f"- Neural Networks: {data['riglogic_stats']['body']['neural_network_count']}",
        ]
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filename


def export_snapshot(
    results: ProfileResults,
    output_path: str | Path,
    output_format: str = "all",
    iterations: int = 0,
    warmup: int = 0,
) -> list[Path]:
    """
    Export profiling snapshot in specified format(s).

    Args:
        results: The ProfileResults to export.
        output_path: Directory to write output files.
        output_format: Export format - "json", "csv", "markdown", or "all".
        iterations: Number of iterations run.
        warmup: Number of warmup iterations.

    Returns:
        List of paths to created files.
    """
    output_path = Path(output_path)
    created_files = []

    if output_format in ("json", "all"):
        created_files.append(export_json(results, output_path, iterations, warmup))
    if output_format in ("csv", "all"):
        created_files.append(export_csv(results, output_path, iterations, warmup))
    if output_format in ("markdown", "md", "all"):
        created_files.append(export_markdown(results, output_path, iterations, warmup))

    return created_files


def compare_snapshots(
    baseline_path: str | Path,
    current_path: str | Path,
    threshold_pct: float = 10.0,
    require_matching_hardware: bool = True,
) -> dict:
    """
    Compare two profiling snapshots and detect regressions.

    Args:
        baseline_path: Path to baseline JSON snapshot.
        current_path: Path to current JSON snapshot.
        threshold_pct: Percentage threshold for regression detection.
        require_matching_hardware: If True, only flag regressions when hardware matches.

    Returns:
        Dictionary with comparison results and detected regressions.
    """
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(current_path, encoding="utf-8") as f:
        current = json.load(f)

    if baseline.get("metadata", {}).get("benchmark_schema") != current.get("metadata", {}).get("benchmark_schema"):
        raise ValueError("Benchmark workloads differ; capture a new baseline for the native runtime")

    # Check hardware compatibility
    baseline_hw = baseline.get("metadata", {}).get("hardware", {})
    current_hw = current.get("metadata", {}).get("hardware", {})

    def get_hw_fingerprint(hw: dict) -> str:
        """Generate a fingerprint for hardware comparison."""
        return (
            f"{hw.get('cpu_name', 'unknown')}|"
            f"{hw.get('cpu_cores_physical', 0)}|"
            f"{hw.get('ram_total_gb', 0):.0f}GB|"
            f"{hw.get('gpu_name', 'unknown')}"
        )

    baseline_fingerprint = get_hw_fingerprint(baseline_hw)
    current_fingerprint = get_hw_fingerprint(current_hw)
    hardware_matches = baseline_fingerprint == current_fingerprint

    regressions = []
    improvements = []
    comparisons = {}

    def compare_timing(name: str, baseline_ms: float, current_ms: float) -> None:
        if baseline_ms == 0:
            return

        diff_pct = ((current_ms - baseline_ms) / baseline_ms) * 100
        comparisons[name] = {
            "baseline_ms": baseline_ms,
            "current_ms": current_ms,
            "diff_ms": current_ms - baseline_ms,
            "diff_pct": round(diff_pct, 2),
        }

        # Only flag regressions/improvements if hardware matches (when required)
        if require_matching_hardware and not hardware_matches:
            return

        if diff_pct > threshold_pct:
            regressions.append({"name": name, "diff_pct": round(diff_pct, 2)})
        elif diff_pct < -threshold_pct:
            improvements.append({"name": name, "diff_pct": round(diff_pct, 2)})

    # Compare summary metrics
    for key in ["native_head_ms", "native_body_ms", "full_evaluation_ms"]:
        compare_timing(key, baseline["summary"].get(key, 0), current["summary"].get(key, 0))

    return {
        "baseline_commit": baseline["metadata"].get("git_commit", "unknown"),
        "current_commit": current["metadata"].get("git_commit", "unknown"),
        "baseline_hardware": baseline_hw,
        "current_hardware": current_hw,
        "hardware_matches": hardware_matches,
        "hardware_mismatch_warning": (
            "Hardware does not match between snapshots. Regression detection skipped."
            if require_matching_hardware and not hardware_matches
            else None
        ),
        "threshold_pct": threshold_pct,
        "has_regressions": len(regressions) > 0,
        "regressions": regressions,
        "improvements": improvements,
        "comparisons": comparisons,
    }
