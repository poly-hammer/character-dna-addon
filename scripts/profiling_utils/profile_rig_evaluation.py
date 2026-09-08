"""Measure native rig callbacks and full dependency-graph evaluation for CI."""

from __future__ import annotations

import statistics
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bpy


@dataclass
class TimingResult:
    """Nanosecond samples exported in milliseconds."""

    name: str
    times_ns: list[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.times_ns)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ns) / 1e6 if self.times_ns else 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ns) / 1e6 if self.times_ns else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.times_ns) / 1e6 if self.times_ns else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.times_ns) / 1e6 if self.times_ns else 0.0

    @property
    def p95_ms(self) -> float:
        ordered = sorted(self.times_ns)
        return ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)] / 1e6 if ordered else 0.0

    def add(self, time_ns: int) -> None:
        self.times_ns.append(time_ns)


@dataclass
class RigLogicStats:
    """DNA workload dimensions recorded with each benchmark."""

    calculation_type: str = "native_frame"
    floating_point_type: str = "float32"
    rbf_solver_count: int = 0
    neural_network_count: int = 0
    psd_count: int = 0
    blend_shape_channel_count: int = 0
    animated_map_count: int = 0
    joint_count: int = 0
    joint_delta_value_count: int = 0


@dataclass
class ProfileResults:
    """Native callback times are subsets of the full graph evaluation time."""

    head_evaluation: TimingResult = field(default_factory=lambda: TimingResult("head_evaluation"))
    body_evaluation: TimingResult = field(default_factory=lambda: TimingResult("body_evaluation"))
    full_evaluation: TimingResult = field(default_factory=lambda: TimingResult("full_evaluation"))
    head_stats: RigLogicStats = field(default_factory=RigLogicStats)
    body_stats: RigLogicStats = field(default_factory=RigLogicStats)


def get_active_rig_instance() -> Any:
    """Resolve the benchmark character through the add-on's current selection API."""
    from character_dna.utilities import get_active_rig_instance as get_instance

    return get_instance()


def _stats(reader: Any) -> RigLogicStats:
    if reader is None:
        return RigLogicStats()
    return RigLogicStats(
        joint_count=reader.getJointCount(),
        blend_shape_channel_count=reader.getBlendShapeChannelCount(),
        animated_map_count=reader.getAnimatedMapCount(),
        rbf_solver_count=reader.getRBFSolverCount(),
        neural_network_count=reader.getNeuralNetworkCount(),
        psd_count=reader.getPSDCount(),
    )


class RigEvaluationProfiler:
    """Benchmark changed inputs, including Blender output drivers and deformation."""

    def __init__(self, rig_instance: Any):
        self.rig_instance = rig_instance
        self.results = ProfileResults()

    def run_benchmark(self, iterations: int = 100, warmup: int = 10) -> ProfileResults:  # noqa: PLR0912
        """Time the actual live path and restore all input poses and instrumentation."""
        from character_dna.runtime import engine

        if iterations < 1 or warmup < 0:
            raise ValueError("Use at least one iteration and nonnegative warmup")
        instance = self.rig_instance
        instance.evaluate()
        if not engine.active(instance):
            raise RuntimeError("The benchmark requires an active native runtime")
        controls = []
        if instance.face_board:
            for name in ("CTRL_C_jaw", "CTRL_L_brow_down", "CTRL_C_eye"):
                bone = instance.face_board.pose.bones.get(name)
                if bone:
                    controls.append((instance.face_board, bone, "location", bone.location.copy()))
        if instance.body_rig:
            for name in ("upperarm_l", "neck_01"):
                bone = instance.body_rig.pose.bones.get(name)
                if bone:
                    controls.append((instance.body_rig, bone, "rotation_quaternion", bone.rotation_quaternion.copy()))
        if not controls:
            raise RuntimeError("The benchmark fixture has no controllable rig inputs")
        self.results = ProfileResults(
            head_stats=_stats(instance.head_dna_reader), body_stats=_stats(instance.body_dna_reader)
        )
        original = bpy.app.driver_namespace[engine.NAMESPACE]
        identity = engine.instance_id(instance)
        collecting = False

        def measured_solve(owner: Any, graph: Any) -> float:
            start = time.perf_counter_ns()
            result = original(owner, graph)
            elapsed = time.perf_counter_ns() - start
            if collecting and owner.get("instance_id") == identity:
                metric = getattr(self.results, f"{owner['component']}_evaluation", None)
                if metric is not None:
                    metric.add(elapsed)
            return result

        bpy.app.driver_namespace[engine.NAMESPACE] = measured_solve
        try:
            for iteration in range(warmup + iterations):
                value = (iteration % 19 + 1) / 20.0
                for owner, bone, attribute, _saved in controls:
                    if attribute == "location":
                        bone.location.y = value
                    else:
                        bone.rotation_quaternion = (1.0, value * 0.2, 0.0, 0.0)
                    owner.update_tag()
                collecting = iteration >= warmup
                start = time.perf_counter_ns()
                bpy.context.view_layer.update()
                elapsed = time.perf_counter_ns() - start
                if collecting:
                    self.results.full_evaluation.add(elapsed)
            for component in ("head", "body"):
                if (
                    getattr(instance, f"{component}_rig")
                    and getattr(self.results, f"{component}_evaluation").count < iterations
                ):
                    raise RuntimeError(f"The {component} callback was not evaluated for every timed sample")
        finally:
            bpy.app.driver_namespace[engine.NAMESPACE] = original
            for owner, bone, attribute, saved in controls:
                setattr(bone, attribute, saved)
                owner.update_tag()
            bpy.context.view_layer.update()
        return self.results

    def print_report(self) -> None:
        """Print only measured runtime stages; callback times are not standalone C++ timings."""
        for metric in (self.results.head_evaluation, self.results.body_evaluation, self.results.full_evaluation):
            print(f"{metric.name}: mean={metric.mean_ms:.3f}ms p95={metric.p95_ms:.3f}ms samples={metric.count}")


def run_profiler(
    iterations: int = 100, warmup: int = 10, export_path: str | Path | None = None, export_format: str = "all"
) -> ProfileResults:
    """Run the CI workload on the active character and optionally export its snapshot."""
    instance = get_active_rig_instance()
    if instance is None:
        raise RuntimeError("Load a character before benchmarking")
    profiler = RigEvaluationProfiler(instance)
    results = profiler.run_benchmark(iterations, warmup)
    profiler.print_report()
    if export_path:
        from .exporters import export_snapshot

        export_snapshot(results, export_path, export_format, iterations, warmup)
    return results
