"""Compare native local channels against Blender pose matrix application."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys

from array import array
from pathlib import Path

from mathutils import Euler, Matrix, Quaternion, Vector


sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_riglogic_probe import make_rig, runtime_manifest


def main() -> None:  # noqa: PLR0915
    """Check both joint formats, signed scales, singular matrices and invalid buffers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    manifest = runtime_manifest()
    sys.path.insert(0, str(args.module_dir))
    native = importlib.import_module("_riglogic_blender")
    rig, _mesh = make_rig()
    bone = rig.pose.bones[0]
    bone.rotation_mode = "XYZ"
    generator = random.Random(9082)  # noqa: S311
    cases = []
    for quaternion in (False, True):
        for case in range(80):
            location = Vector([generator.uniform(-0.2, 0.2) for _axis in range(3)])
            rotation = Euler([generator.uniform(-2.5, 2.5) for _axis in range(3)])
            scale = Vector([1.0, 1.0, 1.0])
            rest = Matrix.LocRotScale(location, rotation, scale)
            inverse = rest.inverted_safe()
            children = not quaternion and case % 2 == 0
            binding = array(
                "f", [0, int(children), *location, *rotation, *scale, *(value for row in inverse for value in row)]
            )
            delta_location = [generator.uniform(-2.0, 2.0) for _axis in range(3)]
            delta_scale = [generator.uniform(-0.2, 0.2) for _axis in range(3)]
            if case % 8 == 0:
                delta_scale[0] = -2.0
            if case % 13 == 0:
                delta_scale = [-1.0, -1.0, -1.0]
            if quaternion:
                delta = Quaternion((0.4, 0.6, 0.8), generator.uniform(-1, 1))
                delta = Quaternion([value * (1.05 if case % 3 == 0 else 1.0) for value in delta])
                values = array("d", array("f", [*delta_location, delta.x, delta.y, delta.z, delta.w, *delta_scale]))
                expected_rotation = rotation.to_quaternion() @ Quaternion((values[6], *values[3:6]))
                expected_scale = Vector(values[7:10])
            else:
                values = array(
                    "d",
                    array("f", [*delta_location, *(generator.uniform(-90, 90) for _axis in range(3)), *delta_scale]),
                )
                delta = Euler([math.radians(value) for value in values[3:6]])
                expected_rotation = Euler([rotation[axis] + delta[axis] for axis in range(3)])
                expected_scale = Vector(values[6:9])
            translation = Vector([value / 100 for value in values[:3]])
            modified = Matrix.LocRotScale(location + translation, expected_rotation, scale + expected_scale)
            bone.matrix_basis = inverse @ modified
            if children:
                bone.rotation_euler = delta
            expected = [*bone.location, *bone.rotation_euler, *bone.scale]
            output = array("d", [99.0]) * 9
            native.transform_into(binding, values, output, quaternion)
            error = max(abs(actual - reference) for actual, reference in zip(output, expected, strict=True))
            assert math.isfinite(error) and error < 1e-5, (quaternion, case, error, list(output), expected)
            cases.append(error)
    original = output.tobytes()
    invalid = array("f", binding)
    invalid[0] = 9999999
    for plan, source, target in (
        (invalid, values, output),
        (array("f", [0.0]), values, output),
        (binding, array("f", values), output),
        (binding, values, memoryview(output).toreadonly()),
    ):
        try:
            native.transform_into(plan, source, target, True)
        except (ValueError, BufferError):
            pass
        else:
            raise AssertionError("Invalid transform buffers accepted")
        assert output.tobytes() == original
    result = {"runtime": manifest, "cases": len(cases), "max_error": max(cases), "invalid_cases": 4, "passed": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
