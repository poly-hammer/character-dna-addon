"""Describe DNA driver inputs; expected joint outputs are captured only in Unreal."""

import argparse
import hashlib
import json
import math
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src/addons"), str(ROOT / "tests")]


def prepare(dna_path: Path, output: Path) -> None:
    """Write neutral transforms and one frame per non-default solver target."""
    import bpy  # noqa: F401  # Initializes mathutils in the standalone bpy wheel.

    from mathutils import Euler, Quaternion, Vector

    from character_dna.dna_io import get_dna_reader

    reader = get_dna_reader(dna_path)
    joints = []
    for index in range(reader.getJointCount()):
        rotation = Euler([math.radians(v) for v in reader.getNeutralJointRotation(index)], "XYZ").to_quaternion()
        translation = Vector(reader.getNeutralJointTranslation(index))
        parent = reader.getJointParentIndex(index)
        if parent == index:
            # Body DNA's root carries -90 X. The displayed rig applies +90 X.
            bridge = Quaternion((1, 0, 0), math.pi / 2)
            rotation = bridge @ rotation
            translation = bridge @ translation
        joints.append(
            {
                "name": reader.getJointName(index),
                "parent": int(parent),
                "unreal_translation_cm": [translation.x, -translation.y, translation.z],
                "unreal_rotation_xyzw": [-rotation.x, rotation.y, -rotation.z, rotation.w],
            }
        )
    poses = []
    for solver in range(reader.getRBFSolverCount()):
        controls = [reader.getRawControlName(i) for i in reader.getRBFSolverRawControlIndices(solver)]
        values = list(reader.getRBFSolverRawControlValues(solver))
        for target, pose in enumerate(reader.getRBFSolverPoseIndices(solver)):
            name = reader.getRBFPoseName(pose)
            if name == "default":
                continue
            drivers = {}
            for control, value in zip(
                controls, values[target * len(controls) : (target + 1) * len(controls)], strict=True
            ):
                bone, channel = control.rsplit(".", 1)
                assert channel in ("qx", "qy", "qz", "qw"), control
                drivers.setdefault(bone, [0.0, 0.0, 0.0, 1.0])["xyzw".index(channel[-1])] = value
            poses.append({"solver": reader.getRBFSolverName(solver), "pose": name, "drivers": drivers})
    poses.sort(key=lambda pose: (pose["solver"].casefold(), pose["pose"].casefold()))
    frames = [{"solver": None, "pose": "neutral", "drivers": {}}, *poses]
    for index, frame in enumerate(frames):
        frame["frame"] = index
    result = {
        "schema_version": 1,
        "fps": 30,
        "dna_sha256": hashlib.sha256(dna_path.read_bytes()).hexdigest(),
        "driver_convention": "Maya Y-up, neutral-relative quaternion x,y,z,w; compose neutral * delta",
        "joints": joints,
        "frames": frames,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(frames)} frames and {len(joints)} joints to {output}")


if __name__ == "__main__":
    from utilities.process import exit_blender

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dna", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        prepare(args.dna, args.output)
    except BaseException:
        import traceback

        traceback.print_exc()
        exit_blender(1)
    exit_blender(0)
