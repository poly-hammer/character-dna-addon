"""Validate one cached native Ada solve feeding simple-expression output drivers."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import uuid

from array import array
from pathlib import Path
from typing import Any

import bpy


sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_benchmark import ADDON_ROOT, _hash_file, make_raw_inputs
from stock_riglogic_probe import runtime_manifest


def add_variable(driver: Any, name: str, source: Any, path: str) -> None:
    """Declare an explicit scalar dependency on a source property."""
    variable = driver.variables.new()
    variable.name = name
    variable.type = "SINGLE_PROP"
    variable.targets[0].id = source
    variable.targets[0].data_path = path


def main() -> None:  # noqa: PLR0915
    """Assert input deduplication and complete output parity in multiple contexts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--full-outputs", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    manifest = runtime_manifest()
    sys.path.insert(0, str(args.module_dir))
    native = importlib.import_module("_riglogic_blender")
    scene = bpy.context.scene
    scene.frame_set(0)
    model = native.load_model(str(ADDON_ROOT / "tests/test_files/dna/ada/head.dna"))
    reference = native.create_session(model)
    info = native.describe(reference)
    output_count = sum(info["output_counts"])
    base = array("f", make_raw_inputs(info["raw_names"], 0))
    jaw_index = info["raw_names"].index("CTRL_expressions.jawOpen")
    records: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    drivers = []
    generation = 1

    def solve(owner: Any, graph: Any, signal: float) -> float:
        try:
            record = records[owner["token"]]
            if not owner.is_evaluated or owner == record["carrier"]:
                raise RuntimeError("Native publication requires an evaluated scratch carrier")
            key = (generation, graph.as_pointer(), owner.as_pointer(), graph.mode)
            context = record["contexts"].get(key)
            if context is None:
                context = {"session": native.create_session(model), "controls": array("f", base), "calls": 0}
                record["contexts"][key] = context
            context["controls"][jaw_index] = signal
            context["calls"] += 1
            native.evaluate_into(context["session"], context["controls"], owner["outputs"])
            return signal
        except Exception as error:
            failures.append(str(error))
            raise

    bpy.app.driver_namespace["rl_native_carrier"] = solve
    for instance_index in range(args.instances):
        token = uuid.uuid4().hex
        source = bpy.data.objects.new(f"Source{instance_index}", None)
        constrained = bpy.data.objects.new(f"ConstrainedInput{instance_index}", None)
        carrier = bpy.data.objects.new(f"Carrier{instance_index}", None)
        target = bpy.data.objects.new(f"Target{instance_index}", None)
        for obj in (source, constrained, carrier, target):
            scene.collection.objects.link(obj)
        source["signal"] = 0.0
        input_driver = source.driver_add("location", 0).driver
        add_variable(input_driver, "signal", source, '["signal"]')
        input_driver.expression = "signal"
        constraint = constrained.constraints.new("COPY_LOCATION")
        constraint.target = source
        for frame in range(1, 9):
            source["signal"] = (frame + instance_index) * 0.05
            source.keyframe_insert(data_path='["signal"]', frame=frame)
        source["signal"] = 0.0
        carrier["token"] = token
        carrier["epoch"] = 0.0
        carrier["outputs"] = array("d", [-1000.0]) * output_count
        selected = list(range(output_count)) if args.full_outputs else list(range(0, output_count, 31))
        target["values"] = array("d", [-2000.0]) * len(selected)
        records[token] = {
            "source": source,
            "constrained": constrained,
            "carrier": carrier,
            "target": target,
            "indices": selected,
            "contexts": {},
        }
        driver = carrier.driver_add('["epoch"]').driver
        driver.use_self = True
        variable = driver.variables.new()
        variable.name = "signal"
        variable.type = "TRANSFORMS"
        variable.targets[0].id = constrained
        variable.targets[0].transform_type = "LOC_X"
        variable.targets[0].transform_space = "WORLD_SPACE"
        driver.expression = "rl_native_carrier(self, depsgraph, signal)"
        drivers.append(driver)
        for target_index, output_index in enumerate(selected):
            curve = target.driver_add('["values"]', target_index)
            curve.keyframe_points.clear()
            for modifier in tuple(curve.modifiers):
                curve.modifiers.remove(modifier)
            driver = curve.driver
            add_variable(driver, "epoch", carrier, '["epoch"]')
            add_variable(driver, "output", carrier, f'["outputs"][{output_index}]')
            driver.expression = "output + 0 * epoch"
            assert driver.is_simple_expression
            drivers.append(driver)

    checks = []

    def verify(label: str, graph: Any) -> None:
        maximum = 0.0
        for record in records.values():
            controls = array("f", base)
            controls[jaw_index] = record["constrained"].evaluated_get(graph).matrix_world.translation.x
            expected = native.evaluate(reference, controls)
            flat = [value for family in ("joints", "blend_shapes", "animated_maps") for value in expected[family]]
            actual = record["target"].evaluated_get(graph)["values"]
            assert len(actual) == len(record["indices"])
            error = max(abs(value - flat[index]) for value, index in zip(actual, record["indices"], strict=True))
            published = record["carrier"].evaluated_get(graph)["outputs"]
            assert math.isfinite(error) and error < 1e-5, {
                "label": label,
                "error": error,
                "reference_input": controls[jaw_index],
                "native_inputs": [context["controls"][jaw_index] for context in record["contexts"].values()],
                "published_error": max(abs(value - flat[index]) for index, value in enumerate(published)),
                "worst": sorted(
                    (
                        (abs(value - flat[index]), index, value, flat[index], published[index])
                        for value, index in zip(actual, record["indices"], strict=True)
                    ),
                    reverse=True,
                )[:3],
            }
            assert all(value == -1000.0 for value in record["carrier"]["outputs"]), "Original carrier modified"
            maximum = max(maximum, error)
        assert all(driver.is_valid for driver in drivers), label
        assert not failures, failures
        checks.append({"label": label, "max_error": maximum})

    def totals() -> dict[str, int]:
        result = {"solves": 0, "cache_hits": 0, "calls": 0}
        for record in records.values():
            for context in record["contexts"].values():
                statistics = native.session_statistics(context["session"])
                for name in ("solves", "cache_hits"):
                    result[name] += statistics[name]
                result["calls"] += context["calls"]
        return result

    bpy.context.view_layer.update()
    for frame in (1, 2, 3, 3, 5, 8, 2):
        scene.frame_set(frame)
        verify(f"frame-{frame}", bpy.context.evaluated_depsgraph_get())
    before = totals()
    for record in records.values():
        record["carrier"].update_tag(refresh={"OBJECT"})
    bpy.context.view_layer.update()
    verify("same-input-recopy", bpy.context.evaluated_depsgraph_get())
    after = totals()
    assert after["solves"] == before["solves"], (before, after)
    assert after["cache_hits"] > before["cache_hits"], (before, after)
    first = next(iter(records.values()))
    first["source"].animation_data.action = None
    first["source"]["signal"] = 0.77
    first["source"].update_tag()
    bpy.context.view_layer.update()
    verify("same-frame-edit", bpy.context.evaluated_depsgraph_get())
    edited = totals()
    assert edited["solves"] == after["solves"] + 1, (after, edited)
    scene.frame_set(2, subframe=0.25)
    verify("subframe", bpy.context.evaluated_depsgraph_get())
    second_layer = scene.view_layers.new("SecondContext")
    with bpy.context.temp_override(view_layer=second_layer):
        second_layer.update()
        verify("second-view-layer", bpy.context.evaluated_depsgraph_get())
    generation += 1
    for record in records.values():
        record["carrier"].update_tag()
    bpy.context.view_layer.update()
    verify("explicit-generation-invalidation", bpy.context.evaluated_depsgraph_get())
    result = {
        "runtime": manifest,
        "module_sha256": _hash_file(Path(native.__file__)),
        "instances": args.instances,
        "output_count_per_model": output_count,
        "output_driver_count": len(drivers) - args.instances,
        "checks": checks,
        "same_input_before": before,
        "same_input_after": after,
        "same_frame_edit": edited,
        "final_counts": totals(),
        "contexts_per_instance": [len(record["contexts"]) for record in records.values()],
        "passed": True,
        "scope": "Native SDK outputs through drivers; not Blender rest-transform conversion or production lifecycle",
        "input_dependency": "Animated property -> driver -> COPY_LOCATION constraint -> native carrier",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
