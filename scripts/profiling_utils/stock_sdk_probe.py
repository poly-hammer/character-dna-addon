"""Verify portable buffer-module parity and ownership in stock Blender."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import sys

from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_benchmark import ADDON_ROOT, _hash_file, compare_outputs, make_raw_inputs
from stock_riglogic_probe import runtime_manifest


def verify_cached_publication(native: Any, model: Any, info: dict[str, Any]) -> dict[str, Any]:
    """Check duplicate solves, republishing, invalidation and transactional rejection."""
    session = native.create_session(model)
    controls = array("f", make_raw_inputs(info["raw_names"], 101))
    destination = array("d", [-1000.0]) * sum(info["output_counts"])
    native.evaluate_into(session, controls, destination)
    expected = array("d", destination)
    assert native.session_statistics(session) == {"solves": 1, "cache_hits": 0}
    for _repeat in range(20):
        destination[:] = array("d", [-1000.0]) * len(destination)
        native.evaluate_into(session, controls, destination)
        assert destination == expected
    assert native.session_statistics(session) == {"solves": 1, "cache_hits": 20}
    controls[0] += 0.125
    native.evaluate_into(session, controls, destination)
    assert native.session_statistics(session)["solves"] == 2
    native.evaluate_into(session, controls, destination, 1)
    assert native.session_statistics(session)["solves"] == 3
    if info["gui_names"]:
        gui = array("f", [0.0]) * len(info["gui_names"])
        native.evaluate_into(session, gui, destination, 1, True)
        assert native.session_statistics(session)["solves"] == 4
    original = native.session_statistics(session)
    invalid = (
        array("d", [0.0]),
        array("f", [0.0]) * len(destination),
        memoryview(destination).toreadonly(),
        memoryview(destination)[::2],
        memoryview(destination).cast("B").cast("d", shape=[1, len(destination)]),
    )
    for output in invalid:
        before = bytes(output)
        try:
            native.evaluate_into(session, controls, output)
        except (ValueError, BufferError):
            pass
        else:
            raise AssertionError("Invalid destination accepted")
        assert bytes(output) == before
        assert native.session_statistics(session) == original
    before = destination.tobytes()
    controls[0] = math.nan
    try:
        native.evaluate_into(session, controls, destination)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite controls accepted")
    assert destination.tobytes() == before
    assert native.session_statistics(session) == original
    return {"duplicate_requests": 20, "duplicate_solves": 0, "invalid_destinations": len(invalid), "passed": True}


def verify_component(  # noqa: PLR0912, PLR0915
    native: Any, dna: Any, riglogic: Any, path: Path, quaternion: bool
) -> dict[str, Any]:
    """Compare all output families across LODs and exercise buffer ownership."""
    model = native.load_model(str(path), quaternion)
    sessions = [native.create_session(model) for _index in range(8)]
    info = native.describe(sessions[0])
    cached_publication = verify_cached_publication(native, model, info)
    stream = dna.FileStream.create(str(path), dna.AccessMode_Read, dna.OpenMode_Binary, None)
    reader = manager = reference = None
    cases = []
    try:
        config = dna.Configuration()
        reader = dna.BinaryStreamReader.create(stream, config, None)
        reader.read()
        if not dna.Status.isOk():
            raise RuntimeError(dna.Status.get().message)
        names = [reader.getRawControlName(index) for index in range(reader.getRawControlCount())]
        assert names == info["raw_names"]
        config = riglogic.Configuration()
        config.floatingPointType = riglogic.FloatingPointType_Float
        if quaternion:
            config.rotationType = riglogic.RotationType_Quaternions
        manager = riglogic.RigLogic.create(reader, config, None)
        reference = riglogic.RigInstance.create(manager, None)
        output_counts = {
            "active_joint_attributes_lod0": len(manager.getJointVariableAttributeIndices(0)),
            "joints": reader.getJointCount(),
            "blend_shapes": reader.getBlendShapeChannelCount(),
            "animated_maps": reader.getAnimatedMapCount(),
        }
        for lod in sorted({0, min(2, info["lod_count"] - 1), info["lod_count"] - 1}):
            for seed in (0, 101, 202, 303, 404, 505, 606, 707):
                controls = array("f", make_raw_inputs(names, seed))
                for index, value in enumerate(controls):
                    reference.setRawControl(index, value)
                reference.setLOD(lod)
                manager.calculate(reference)
                outputs = native.evaluate(sessions[0], controls, lod)
                expected = {
                    "joints": reference.getJointOutputs(),
                    "blend_shapes": reference.getBlendShapeOutputs(),
                    "animated_maps": reference.getAnimatedMapOutputs(),
                }
                errors = {
                    key: compare_outputs([float(value) for value in values], list(outputs[key]), 1e-5)
                    for key, values in expected.items()
                }
                assert all(metric["passed"] for metric in errors.values()), (lod, seed, errors)
                assert all(value.format == "f" and value.readonly for value in outputs.values())
                cases.append({"lod": lod, "seed": seed, "errors": errors})
        if info["gui_names"]:
            for seed in (0, 1, 2):
                controls = array("f", [(index % 5 - 2) * 0.1 * seed for index in range(len(info["gui_names"]))])
                for index, value in enumerate(controls):
                    reference.setGUIControl(index, value)
                reference.setLOD(0)
                manager.mapGUIToRawControls(reference)
                manager.calculate(reference)
                actual = native.evaluate(sessions[0], controls, 0, True)
                for key, expected in (
                    ("joints", reference.getJointOutputs()),
                    ("blend_shapes", reference.getBlendShapeOutputs()),
                    ("animated_maps", reference.getAnimatedMapOutputs()),
                ):
                    error = compare_outputs([float(value) for value in expected], list(actual[key]), 1e-5)
                    assert error["passed"], ("gui", seed, key, error)
        controls = array("f", make_raw_inputs(names, 0))
        snapshot = native.evaluate(sessions[0], controls)
        retained = {key: value.tobytes() for key, value in snapshot.items()}
        invalid = (
            (array("f"), 0),
            (array("d", controls), 0),
            (memoryview(controls)[::2], 0),
            (memoryview(controls).cast("B").cast("f", shape=[1, len(controls)]), 0),
            (controls, -1),
            (controls, info["lod_count"]),
            (array("f", [math.nan]) + controls[1:], 0),
            (array("f", [math.inf]) + controls[1:], 0),
        )
        for values, lod in invalid:
            try:
                native.evaluate(sessions[0], values, lod)
            except (ValueError, BufferError):
                pass
            else:
                raise AssertionError("Invalid numeric input was accepted")
        del model
        gc.collect()

        def evaluate_independent(index: int) -> dict[str, bytes]:
            values = array("f", make_raw_inputs(names, 101 * (index + 1)))
            result = native.evaluate(sessions[index], values)
            return {key: value.tobytes() for key, value in result.items()}

        sequential = [evaluate_independent(index) for index in range(len(sessions))]
        with ThreadPoolExecutor(max_workers=len(sessions)) as executor:
            parallel = list(executor.map(evaluate_independent, range(len(sessions))))
        assert parallel == sequential
        assert any(result["joints"] != sequential[0]["joints"] for result in sequential[1:])
        sessions.clear()
        gc.collect()
        assert {key: value.tobytes() for key, value in snapshot.items()} == retained
    finally:
        if reference is not None:
            riglogic.RigInstance.destroy(reference)
        if manager is not None:
            riglogic.RigLogic.destroy(manager)
        if reader is not None:
            dna.BinaryStreamReader.destroy(reader)
        dna.FileStream.destroy(stream)
    return {
        "dna_sha256": _hash_file(path),
        "cached_publication": cached_publication,
        "output_counts": output_counts,
        "cases": cases,
        "gui_cases": 3 if info["gui_names"] else 0,
        "invalid_buffers": len(invalid),
        "independent_sessions": 8,
        "retained_snapshot": True,
    }


def main() -> None:
    """Run parity with unchanged packaged SDK bindings and record exact binaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dna-root", type=Path, default=ADDON_ROOT / "tests/test_files/dna/ada")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    manifest = runtime_manifest()
    sys.path.insert(0, str(args.module_dir))
    native = importlib.import_module("_riglogic_blender")
    bindings = ADDON_ROOT / "src/addons/character_dna/bindings/windows/x64/py313"
    sys.path.insert(0, str(bindings))
    dna = importlib.import_module("dna")
    riglogic = importlib.import_module("riglogic")
    result = {
        "runtime": manifest,
        "capabilities": native.capabilities(),
        "module_sha256": _hash_file(Path(native.__file__)),
        "binding_hashes": {path.name: _hash_file(path) for path in bindings.glob("*.pyd")},
        "components": {
            component: verify_component(native, dna, riglogic, args.dna_root / f"{component}.dna", component == "body")
            for component in ("head", "body")
        },
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Portable SDK parity passed; report: {args.output}")


if __name__ == "__main__":
    main()
