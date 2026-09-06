"""Verify native SDK parity inside the experimental Blender executable.

The implemented stages verify SDK output parity and prepare an Ada fixture.
They do not measure scene or viewport FPS or register a realtime handler.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import sys

from pathlib import Path
from typing import Any


ADDON_ROOT = Path(__file__).resolve().parents[2]


def make_raw_inputs(names: list[str], seed: int) -> list[float]:
    """Generate bounded expressions and normalized, nonzero driver quaternions."""
    generator = random.Random(seed)  # noqa: S311
    values = [0.0] * len(names)
    quaternion_indices: dict[str, dict[str, int]] = {}
    for index, name in enumerate(names):
        control, _, axis = name.rpartition(".")
        if axis in {"qx", "qy", "qz", "qw"}:
            quaternion_indices.setdefault(control, {})[axis] = index
        else:
            values[index] = generator.uniform(0.0, 0.7) if seed else 0.0
    for indices in quaternion_indices.values():
        if set(indices) != {"qx", "qy", "qz", "qw"}:
            raise ValueError(f"Incomplete quaternion input: {indices}")
        quaternion = [generator.uniform(-0.25, 0.25) if seed else 0.0 for _ in range(3)] + [1.0]
        length = math.sqrt(sum(value * value for value in quaternion))
        for axis, value in zip(("qx", "qy", "qz", "qw"), quaternion, strict=True):
            values[indices[axis]] = value / length
    return values


def compare_outputs(reference: list[float], candidate: list[float], tolerance: float) -> dict[str, Any]:
    """Compare finite, equally sized arrays without hiding large individual errors."""
    if len(reference) != len(candidate):
        return {"passed": False, "reference_count": len(reference), "candidate_count": len(candidate)}
    if not all(math.isfinite(value) for value in reference + candidate):
        return {"passed": False, "non_finite": True}
    errors = [abs(expected - actual) for expected, actual in zip(reference, candidate, strict=True)]
    maximum = max(errors, default=0.0)
    return {
        "passed": maximum <= tolerance,
        "count": len(errors),
        "max_absolute_error": maximum,
        "rms_error": math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else 0.0,
    }


def _hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_component(native: Any, dna: Any, riglogic: Any, path: Path, quaternion: bool) -> dict[str, Any]:
    session = native.load(str(path), quaternion)
    info = native.info(session)
    stream = dna.FileStream.create(str(path), dna.AccessMode_Read, dna.OpenMode_Binary, None)
    reader = None
    results: list[dict[str, Any]] = []
    try:
        reader_config = dna.Configuration()
        reader_config.layer = dna.DataLayer_All
        reader = dna.BinaryStreamReader.create(stream, reader_config, None)
        reader.read()
        if not dna.Status.isOk():
            raise RuntimeError(dna.Status.get().message)
        names = [reader.getRawControlName(index) for index in range(reader.getRawControlCount())]
        if names != info["raw_names"]:
            raise AssertionError("Native and packaged SDK raw-control schemas differ")
        for precision in ("default", "float"):
            config = riglogic.Configuration()
            if precision == "float":
                config.floatingPointType = riglogic.FloatingPointType_Float
            if quaternion:
                config.rotationType = riglogic.RotationType_Quaternions
            manager = riglogic.RigLogic.create(reader, config, None)
            reference = None
            try:
                reference = riglogic.RigInstance.create(manager, None)
                for lod in sorted({0, min(2, info["lod_count"] - 1), info["lod_count"] - 1}):
                    for seed in (0, 101, 202, 303, 404, 505, 606, 707):
                        raw = make_raw_inputs(names, seed)
                        for index, value in enumerate(raw):
                            reference.setRawControl(index, value)
                        reference.setLOD(lod)
                        manager.calculate(reference)
                        actual = native.evaluate(session, raw, lod)
                        expected = {
                            "joints": list(reference.getJointOutputs()),
                            "blend_shapes": list(reference.getBlendShapeOutputs()),
                            "animated_maps": list(reference.getAnimatedMapOutputs()),
                        }
                        errors = {
                            key: compare_outputs([float(value) for value in values], actual[key], 1e-5)
                            for key, values in expected.items()
                        }
                        results.append(
                            {
                                "precision": precision,
                                "lod": lod,
                                "seed": seed,
                                "outputs": errors,
                                "passed": all(metric["passed"] for metric in errors.values()),
                            }
                        )
            finally:
                if reference is not None:
                    riglogic.RigInstance.destroy(reference)
                riglogic.RigLogic.destroy(manager)
        for bad_raw, bad_lod in (([], 0), (make_raw_inputs(names, 0), info["lod_count"])):
            try:
                native.evaluate(session, bad_raw, bad_lod)
            except ValueError:
                pass
            else:
                raise AssertionError("Native diagnostic API accepted an invalid input")
    finally:
        if reader is not None:
            dna.BinaryStreamReader.destroy(reader)
        dna.FileStream.destroy(stream)
    return {
        "path": str(path),
        "sha256": _hash_file(path),
        "input_schema": info,
        "cases": results,
        "passed": all(case["passed"] for case in results),
    }


def verify_sdk(output: Path, dna_root: Path) -> bool:
    """Compare shipped and compiled SDKs in the actual native Blender executable."""
    import bpy

    native = getattr(bpy.app, "riglogic", None)
    if native is None or not native.supported:
        raise RuntimeError("This executable was not built with WITH_RIGLOGIC=ON")
    if sys.version_info[:2] != (3, 13) or sys.platform != "win32":
        raise RuntimeError("This experiment requires the recorded Windows Python 3.13 bindings")
    bindings = ADDON_ROOT / "src/addons/character_dna/bindings/windows/x64/py313"
    sys.path.insert(0, str(bindings))
    dna = importlib.import_module("dna")
    riglogic = importlib.import_module("riglogic")
    components = {
        component: _verify_component(native, dna, riglogic, dna_root / f"{component}.dna", component == "body")
        for component in ("head", "body")
    }
    report = {
        "schema_version": 1,
        "scope": "sdk_output_parity_only",
        "blender_binary": bpy.app.binary_path,
        "blender_version": bpy.app.version_string,
        "blender_build_hash": bpy.app.build_hash.decode(),
        "script_sha256": _hash_file(Path(__file__)),
        "binding_hashes": {path.name: _hash_file(path) for path in bindings.glob("*.pyd")},
        "tolerance_absolute": 1e-5,
        "components": components,
        "passed": all(component["passed"] for component in components.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    for name, component in components.items():
        failures = sum(not case["passed"] for case in component["cases"])
        print(f"SDK PARITY {name}: {len(component['cases'])} cases, {failures} failures", flush=True)
    print(f"Report: {output}", flush=True)
    return report["passed"]


def prepare_scene(output: Path, dna_root: Path) -> bool:
    """Prepare the real LOD0 fixture and audit native input/output dependencies."""
    import bpy

    sys.path.insert(0, str(ADDON_ROOT / "scripts"))
    from profiling_utils.ci_benchmark import _import_shape_keys_synchronously, setup_environment

    bpy.ops.wm.read_factory_settings(use_empty=True)
    if not setup_environment():
        raise RuntimeError("Cannot initialize the reference add-on in the built executable")
    from character_dna.utilities import get_active_rig_instance

    bpy.ops.character_dna.import_dna(
        filepath=str(dna_root / "head.dna"),
        include_body=True,
        import_lod0=True,
        **{f"import_lod{lod}": False for lod in range(1, 8)},
    )
    _import_shape_keys_synchronously()
    instance = get_active_rig_instance()
    if instance is None:
        raise RuntimeError("No reference instance after Ada import")
    instance.view_options.active_lod = "lod0"
    instance.evaluate_bones = True
    instance.evaluate_shape_keys = True
    instance.evaluate_texture_masks = True
    instance.evaluate_rbfs = True
    instance.evaluate()
    components = {}
    for name in ("head", "body"):
        rig = getattr(instance, f"{name}_rig")
        drivers = set(getattr(instance, f"{name}_driver_bone_names"))
        plan = getattr(instance, f"{name}_bone_transform_plan")
        driven = {entry[1] for entry in plan}
        feedback = []
        for driver_name in sorted(drivers):
            bone = rig.pose.bones.get(driver_name)
            ancestors = []
            while bone and bone.parent:
                bone = bone.parent
                if bone.name in driven:
                    ancestors.append(bone.name)
            if ancestors:
                feedback.append({"driver": driver_name, "driven_ancestors": ancestors})
        constraints = [
            {
                "owner": bone.name,
                "type": constraint.type,
                "target": getattr(getattr(constraint, "target", None), "name", None),
                "subtarget": getattr(constraint, "subtarget", None),
            }
            for bone in rig.pose.bones
            for constraint in bone.constraints
            if not constraint.mute
        ]
        components[name] = {
            "object": rig.name,
            "drivers": sorted(drivers),
            "driven": sorted(driven),
            "overlap": sorted(drivers & driven),
            "feedback": feedback,
            "constraints": constraints,
        }
    shape_keys = [
        {"key": key_blocks.id_data.name, "mapped_blocks": len(positions)}
        for key_blocks, positions, _channels, _blocks, _buffer in instance.head_shape_key_apply_plan
    ]
    texture_node = instance.head_texture_masks_node
    if not shape_keys or texture_node is None:
        raise AssertionError("The headline fixture lacks shape keys or its texture logic node")
    if instance.face_board is None:
        raise AssertionError("The headline fixture lacks a face board")
    output.parent.mkdir(parents=True, exist_ok=True)
    blend_path = output.with_suffix(".blend")
    if blend_path.exists():
        raise FileExistsError(blend_path)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), relative_remap=False)
    report = {
        "schema_version": 1,
        "scope": "ada_lod0_preparation_and_dependency_audit",
        "components": components,
        "shape_keys": shape_keys,
        "texture_node": texture_node.name,
        "animated_map_bindings": instance.head_animated_map_plan,
        "face_board": instance.face_board.name,
        "eye_aim_enabled": instance.head_use_eye_aim,
        "armatures": [obj.name for obj in bpy.data.objects if obj.type == "ARMATURE"],
        "blend_file": str(blend_path),
        "blend_sha256": _hash_file(blend_path),
        "dna_hashes": {name: _hash_file(dna_root / f"{name}.dna") for name in ("head", "body")},
        "blender_binary": bpy.app.binary_path,
        "has_bone_feedback": any(part["overlap"] or part["feedback"] for part in components.values()),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f"ADA PREPARED: {output}; bone feedback={report['has_bone_feedback']}", flush=True)
    return True


def main() -> None:
    """Run the implemented benchmark gate; unsupported stages cannot silently pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["verify-sdk", "prepare"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dna-root", type=Path, default=ADDON_ROOT / "tests/test_files/dna/ada")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    operation = verify_sdk if args.stage == "verify-sdk" else prepare_scene
    if not operation(args.output, args.dna_root):
        raise AssertionError("SDK parity gate failed; inspect the recorded errors before benchmarking")


if __name__ == "__main__":
    main()
