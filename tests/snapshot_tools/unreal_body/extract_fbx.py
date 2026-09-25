"""Extract Unreal's baked FBX through UFBX, validating against Unreal's raw pose audit."""

import argparse
import hashlib
import json
import math
import shutil

from pathlib import Path

import numpy as np
import ufbx


def _inherit_mode(node):
    try:
        return int(node.inherit_mode)
    except ValueError as error:
        # Bundled pyufbx 0.0.0 omits ufbx's third enum member from its Python enum.
        if str(error) == "2 is not a valid InheritMode":
            return 2  # UFBX_INHERIT_MODE_COMPONENTWISE_SCALE
        raise


def _rotation_matrix(quaternion):
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def extract(manifest_path: Path, capture_directory: Path, destination: Path, fbx_destination: Path) -> None:
    """Write snapshots only after every sampled joint agrees with Unreal's audit."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture = json.loads((capture_directory / "capture.json").read_text(encoding="utf-8"))
    audit = json.loads((capture_directory / "unreal_world_audit.json").read_text(encoding="utf-8"))
    fbx = capture_directory / "ada_unreal_rbf.fbx"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == capture["inputs_sha256"]
    assert hashlib.sha256(fbx.read_bytes()).hexdigest() == capture["fbx_sha256"]
    assert len(audit) == len(manifest["frames"])
    scene = ufbx.load_file(str(fbx), ignore_geometry=True, ignore_embedded=True)
    snapshots = []
    max_error = 0.0
    try:
        axes = scene.settings.axes
        assert (int(axes.right), int(axes.up), int(axes.front)) == (0, 4, 3), "Expected Unreal FBX X / Z / -Y axes"
        assert math.isclose(scene.settings.frames_per_second, manifest["fps"])
        assert math.isclose(
            scene.anim_stacks[0].time_end, (len(manifest["frames"]) - 1) / manifest["fps"], abs_tol=1e-6
        )
        nodes = {joint["name"]: scene.find_node(joint["name"]) for joint in manifest["joints"]}
        assert all(nodes.values()), "Missing DNA joints in FBX"
        for frame in manifest["frames"]:
            worlds = {}

            def world(node, worlds=worlds, frame=frame):
                if node.typed_id in worlds:
                    return worlds[node.typed_id]
                transform = node.evaluate_transform(frame["frame"] / manifest["fps"])
                translation = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
                rotation = _rotation_matrix(transform.rotation)
                scale = np.array([transform.scale.x, transform.scale.y, transform.scale.z])
                mode = _inherit_mode(node)
                if node.parent:
                    parent_t, parent_r, parent_s = world(node.parent)
                    # Unreal FBX uses RrSs: accumulated scale is component-wise,
                    # not sheared through a child rotation. Animated scales matter.
                    assert mode == 2 or (node.name == "root" and mode == 0 and np.allclose(parent_s, 1))
                    translation = parent_t + parent_r @ (parent_s * translation)
                    rotation = parent_r @ rotation
                    scale = parent_s * scale
                else:
                    assert mode == 0
                worlds[node.typed_id] = (translation, rotation, scale)
                return worlds[node.typed_id]

            locations = {name: (world(node)[0] * scene.settings.unit_meters).tolist() for name, node in nodes.items()}
            for name, location in locations.items():
                error = math.dist(location, audit[frame["frame"]][name])
                assert error < 1e-5, f"FBX audit mismatch: frame {frame['frame']}, {name}, {error} m"
                max_error = max(max_error, error)
            snapshots.append((frame, locations))
    finally:
        scene.close()
    # No fixture files are written until the complete export passes its audit.
    for frame, locations in snapshots:
        path = destination / (frame["solver"] or "_neutral") / (frame["pose"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(locations, indent=2) + "\n", encoding="utf-8")
    if manifest_path.resolve() != (destination / "inputs.json").resolve():
        shutil.copyfile(manifest_path, destination / "inputs.json")
    fbx_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fbx, fbx_destination)
    capture.update(
        {
            "snapshot_space": "Blender world axes, meters; actor at origin; UFBX RrSs scale inheritance",
            "frames": len(snapshots),
            "joints_per_frame": len(nodes),
            "dna_sha256": manifest["dna_sha256"],
            "maximum_fbx_audit_error_m": max_error,
            "fbx_file": fbx_destination.name,
            "extractor": "pyufbx 0.0.0: Node.evaluate_transform at exact frame / fps, including animated scale",
        }
    )
    (destination / "provenance.json").write_text(json.dumps(capture, indent=2) + "\n", encoding="utf-8")
    print(f"Extracted {len(snapshots)} frames; maximum FBX/Unreal error: {max_error:.9g} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("fbx_destination", type=Path)
    args = parser.parse_args()
    extract(args.inputs, args.capture, args.destination, args.fbx_destination)
