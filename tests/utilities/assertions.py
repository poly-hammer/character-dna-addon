import re

from typing import Literal

import math
import pytest

from mathutils import Euler, Vector

from constants import DNA_BEHAVIOR_VERSION, DNA_DEFINITION_VERSION, DNA_GEOMETRY_VERSION, ROTATION_ANGLE_TOLERANCE


def assert_bone_definitions(
    expected_data: dict,
    current_data: dict,
    bone_name: str,
    attribute: str,
    axis_name: str,
    changed_bone_name: str,
    changed_bone_rotation: tuple[Euler, Euler],
    changed_bone_location: tuple[Vector, Vector],
    output_method: Literal["calibrate", "export"] = "calibrate",
    ignored_bones: list[str] | None = None,
    tolerance: float = 1e-3,
):
    if ignored_bones is None:
        ignored_bones = []
    if bone_name in ignored_bones:
        return

    expected_bone_index = expected_data[DNA_DEFINITION_VERSION]["jointNames"].index(bone_name)
    current_bone_index = current_data[DNA_DEFINITION_VERSION]["jointNames"].index(bone_name)
    assert (
        current_bone_index == expected_bone_index
    ), f"Bone index mismatch. {bone_name} should be at index {expected_bone_index} but is at {current_bone_index}"

    expected_hierarchy = expected_data[DNA_DEFINITION_VERSION]["jointHierarchy"][expected_bone_index]
    current_hierarchy = current_data[DNA_DEFINITION_VERSION]["jointHierarchy"][current_bone_index]
    assert (
        current_hierarchy == expected_hierarchy
    ), f"Bone hierarchy mismatch. {bone_name} should have hierarchy {expected_hierarchy} but has {current_hierarchy}"

    expected_value = expected_data[DNA_DEFINITION_VERSION][attribute][f"{axis_name}s"][expected_bone_index]
    current_value = current_data[DNA_DEFINITION_VERSION][attribute][f"{axis_name}s"][current_bone_index]

    # this ensures that we don't assert that the bone was moved in the dna if it was not moved in blender
    changed_location = False
    if attribute == "neutralJointTranslations":
        changed_location = getattr(changed_bone_location[0], axis_name) != 0.0

    changed_rotation = False
    if attribute == "neutralJointRotations":
        changed_rotation = getattr(changed_bone_rotation[0], axis_name) != 0.0

        # for exports, ignore cases were angle wrapping (e.g., 180 vs -180 degrees) occurs
        if output_method == "export" and abs(expected_value - current_value) == 360:
            current_value = expected_value

        # reduce the tolerance for joint rotations since they are in degrees
        tolerance = 1e-2

    if bone_name == changed_bone_name and (changed_rotation or changed_location):
        assert current_value != pytest.approx(
            expected_value, abs=tolerance
        ), f"{axis_name} bone {bone_name} {attribute} should not match, since it was moved in blender."
    elif attribute == "neutralJointRotations" and bone_name != changed_bone_name:
        # Compare the joint's full orientation as an angular difference rather than per-axis euler
        # components. Near gimbal lock individual axes can differ noticeably while the actual
        # orientation is effectively identical, so a rotation is valid when the angle between the
        # expected and exported orientations is within ROTATION_ANGLE_TOLERANCE degrees.
        expected_rotations = expected_data[DNA_DEFINITION_VERSION][attribute]
        current_rotations = current_data[DNA_DEFINITION_VERSION][attribute]
        expected_euler = Euler(
            (
                math.radians(expected_rotations["xs"][expected_bone_index]),
                math.radians(expected_rotations["ys"][expected_bone_index]),
                math.radians(expected_rotations["zs"][expected_bone_index]),
            ),
            "XYZ",
        )
        current_euler = Euler(
            (
                math.radians(current_rotations["xs"][current_bone_index]),
                math.radians(current_rotations["ys"][current_bone_index]),
                math.radians(current_rotations["zs"][current_bone_index]),
            ),
            "XYZ",
        )
        angle_difference = math.degrees(
            expected_euler.to_quaternion().rotation_difference(current_euler.to_quaternion()).angle
        )
        # fold to the shortest angular distance (the quaternion angle can wrap up to 360 degrees)
        angle_difference = min(angle_difference, 360.0 - angle_difference)
        assert angle_difference <= ROTATION_ANGLE_TOLERANCE, (
            f"{axis_name} bone {attribute} mismatch. {bone_name} should have {axis_name} {attribute} "
            f"{expected_value} but has {current_value} (orientation differs by {angle_difference:.4f} degrees, "
            f"tolerance {ROTATION_ANGLE_TOLERANCE} degrees)."
        )
    else:
        assert (
            current_value == pytest.approx(expected_value, abs=tolerance)
        ), f"{axis_name} bone {attribute} mismatch. {bone_name} should have {axis_name} {attribute} {expected_value} but has {current_value}."


def assert_bone_behaviors(expected_data: dict, current_data: dict, bone_name: str):
    # First get the bone index from its name
    expected_bone_index = expected_data[DNA_DEFINITION_VERSION]["jointNames"].index(bone_name)
    current_bone_index = current_data[DNA_DEFINITION_VERSION]["jointNames"].index(bone_name)
    assert (
        current_bone_index == expected_bone_index
    ), f"Bone index mismatch. {bone_name} should be at index {expected_bone_index} but is at {current_bone_index}"

    expected_row_count = expected_data[DNA_BEHAVIOR_VERSION]["joints"]["rowCount"]
    current_row_count = current_data[DNA_BEHAVIOR_VERSION]["joints"]["rowCount"]
    assert (
        current_row_count == expected_row_count
    ), f"Row count mismatch. {bone_name} should have row count {expected_row_count} but has {current_row_count}"

    expected_column_count = expected_data[DNA_BEHAVIOR_VERSION]["joints"]["colCount"]
    current_column_count = current_data[DNA_BEHAVIOR_VERSION]["joints"]["colCount"]
    assert (
        current_column_count == expected_column_count
    ), f"Column count mismatch. {bone_name} should have column count {expected_column_count} but has {current_column_count}"

    expected_joint_groups = expected_data[DNA_BEHAVIOR_VERSION]["joints"]["jointGroups"]
    current_joint_groups = current_data[DNA_BEHAVIOR_VERSION]["joints"]["jointGroups"]

    for _joint_group_index, (expected_joint_group_data, current_joint_group_data) in enumerate(
        zip(expected_joint_groups, current_joint_groups, strict=False)
    ):
        if expected_bone_index in expected_joint_group_data["jointIndices"]:
            assert current_bone_index in current_joint_group_data["jointIndices"]
            break


def assert_mesh_geometry(
    expected_data: dict,
    current_data: dict,
    mesh_name: str,
    attribute: str,
    axis_name: str,
    changed_mesh_name: int,
    changed_vertex_index: int,
    changed_vertex_location: tuple[Vector, Vector, Vector],
    changed_normal_index: int | None = None,
    changed_normal_vector: tuple[Vector, Vector, Vector] | None = None,
    changed_normal_neighbours: list[int] | None = None,
    lower_lod_vertices: list[dict] | None = None,
    assert_mesh_indices: bool = True,
    assert_index_order: bool = True,
    output_method: Literal["calibrate", "export"] = "calibrate",
    tolerance: float = 1e-3,
):
    expected_mesh_index = expected_data[DNA_DEFINITION_VERSION]["meshNames"].index(mesh_name)
    current_mesh_index = current_data[DNA_DEFINITION_VERSION]["meshNames"].index(mesh_name)

    if assert_mesh_indices:
        assert (
            expected_mesh_index == current_mesh_index
        ), f"Mesh index mismatch. {mesh_name} should be at index {expected_mesh_index} but is at {current_mesh_index}"

    # Lower-LOD meshes (lod1+) are not in the scene, so `auto_update_lods` resamples
    # their entire geometry from the calibrated LOD0 shape through the UV-barycentric
    # solver. Their positions therefore no longer match the original DNA, so instead
    # of comparing every vertex we spot-check one statically-captured representative
    # vertex per modeled mesh. Normals and texture coordinates are not propagated and
    # still compare against the original DNA in the logic below.
    if attribute == "positions" and re.search(r"_lod[1-9]\d*_mesh$", mesh_name):
        sample = next((s for s in (lower_lod_vertices or []) if s["mesh_name"] == mesh_name), None)
        if sample is None:
            pytest.skip(
                f"No static lower-LOD position sample defined for {mesh_name}; "
                "auto_update_lods propagation reshapes it away from the original DNA."
            )
        current_values = current_data[DNA_GEOMETRY_VERSION]["meshes"][current_mesh_index]["positions"][f"{axis_name}s"]
        current_value = current_values[sample["vertex_index"]]
        expected_value = getattr(sample["new_dna"], axis_name)
        assert current_value == pytest.approx(expected_value, abs=tolerance), (
            f"Mesh {mesh_name} propagated positions {axis_name} vertex index {sample['vertex_index']} mismatch. "
            f"Expected {expected_value} but has {current_value}."
        )
        return

    # this ensures that we don't assert that the vertex was moved in the dna if it was not moved in blender by
    # comparing the original and new dna vertex positions
    changed_position = False
    changed_index = changed_vertex_index
    tolerated: set[int] = set()
    if attribute == "positions":
        changed_position = (
            getattr(changed_vertex_location[1], axis_name) - getattr(changed_vertex_location[-1], axis_name) != 0.0
        )
    elif attribute == "normals" and changed_normal_vector is not None:
        # Normal index equals position index in the DNA, so this indexes the same arrays.
        changed_index = changed_normal_index
        changed_position = (
            getattr(changed_normal_vector[1], axis_name) - getattr(changed_normal_vector[-1], axis_name) != 0.0
        )
        tolerated = set(changed_normal_neighbours or ())

    expected_indices = expected_data[DNA_GEOMETRY_VERSION]["meshes"][expected_mesh_index]["layouts"][attribute]
    current_indices = current_data[DNA_GEOMETRY_VERSION]["meshes"][current_mesh_index]["layouts"][attribute]

    expected_values = expected_data[DNA_GEOMETRY_VERSION]["meshes"][expected_mesh_index][attribute][f"{axis_name}s"]
    current_values = current_data[DNA_GEOMETRY_VERSION]["meshes"][current_mesh_index][attribute][f"{axis_name}s"]

    # The mesh indices should be the same
    if assert_index_order:
        assert (
            len(expected_indices) == len(current_indices)
        ), f"Mesh {mesh_name} {attribute} indices length mismatch. Expected {len(expected_indices)} indices but has {len(current_indices)}."

        for index, (expected_index, current_index) in enumerate(zip(expected_indices, current_indices, strict=False)):
            assert (
                expected_index == current_index
            ), f"Mesh {mesh_name} {attribute} indices order mismatch at array index {index}. Expected {expected_index} but has {current_index}."

        for expected_index, current_index in zip(expected_indices, current_indices, strict=False):
            expected_value = expected_values[expected_index]
            current_value = current_values[current_index]

            # if this is the changed vertex and the axis value is not ignored
            if mesh_name == changed_mesh_name and expected_index == changed_index and changed_position:
                assert (
                    expected_value != pytest.approx(current_value, abs=tolerance)
                ), f"Mesh {mesh_name} {attribute} {axis_name} vertex index {changed_index} should not match, since it was changed in blender."
            elif mesh_name == changed_mesh_name and expected_index in tolerated:
                continue
            else:
                assert (
                    expected_value == pytest.approx(current_value, abs=tolerance)
                ), f"Mesh {mesh_name} {attribute} {axis_name} vertex index {expected_index} mismatch. Expected {expected_value} but has {current_value}."

    # Otherwise, the mesh indices can be in any order but still the number unique indices should be the same
    else:
        sorted_expected_values = expected_values
        sorted_current_values = current_values
        # remove duplicate values and sort the lists for UVs since there can be multiple UVs for the same vertex
        if attribute == "textureCoordinates":
            sorted_expected_values = sorted(set(expected_values))
            sorted_current_values = sorted(set(current_values))

        for expected_value, current_value in zip(sorted_expected_values, sorted_current_values, strict=False):
            if expected_value != pytest.approx(current_value, abs=tolerance) and changed_position:
                assert (
                    expected_values[changed_index] == expected_value
                ), f"Mesh {mesh_name} {attribute} {axis_name} value mismatch. The vertex index that was changed was {changed_index} but this is not that one."
            else:
                assert (
                    expected_value == pytest.approx(current_value, abs=tolerance)
                ), f"Mesh {mesh_name} {attribute} {axis_name} value mismatch. Expected {expected_value} but has {current_value}."


def get_corner_normals(data: dict, mesh_name: str) -> dict[tuple[int, int], Vector]:
    """The normal every face corner resolves to, keyed by ``(face index, position index)``.

    An export rebuilds the vertex layouts -- one slot per UV-split vertex rather than the
    source DNA's one per position -- so the normal array legitimately changes length and its
    indices mean different things on each side. What a corner actually resolves to does not.
    """
    mesh_index = data[DNA_DEFINITION_VERSION]["meshNames"].index(mesh_name)
    mesh = data[DNA_GEOMETRY_VERSION]["meshes"][mesh_index]
    normal_indices = mesh["layouts"]["normals"]
    position_indices = mesh["layouts"]["positions"]
    xs, ys, zs = mesh["normals"]["xs"], mesh["normals"]["ys"], mesh["normals"]["zs"]

    corners: dict[tuple[int, int], Vector] = {}
    for face_index, face in enumerate(mesh["faces"]):
        for layout_index in face["layoutIndices"]:
            normal_index = normal_indices[layout_index]
            corners[(face_index, position_indices[layout_index])] = Vector(
                (xs[normal_index], ys[normal_index], zs[normal_index])
            )
    return corners


def assert_mesh_corner_normals(
    expected_data: dict,
    current_data: dict,
    mesh_name: str,
    changed_mesh_name: str,
    changed_normal_index: int,
    changed_normal_neighbours: list[int],
    bounds: dict[str, float],
):
    """Every face corner keeps the normal it had, except the ones deliberately changed.

    The export rewrites every normal from the scene rather than only the ones that moved, so
    it carries Blender's custom-normal storage error on all of them. That error is a
    distribution, not a per-corner bound -- a handful of corners cannot be represented at all
    -- so the whole spread is asserted instead. A wrong rotation, a wrong index or the linear
    unit applied to a direction all move the mean, not just the tail.
    """
    expected = get_corner_normals(expected_data, mesh_name)
    current = get_corner_normals(current_data, mesh_name)

    assert set(expected) == set(current), (
        f"Mesh {mesh_name} face corners changed. Expected {len(expected)} corners but has {len(current)}, "
        f"so the exported layouts no longer address the same geometry."
    )

    tolerated = set(changed_normal_neighbours) if mesh_name == changed_mesh_name else set()
    errors = []
    changed_corners = 0

    for key, expected_normal in expected.items():
        _, position_index = key
        current_normal = current[key]

        assert current_normal.length == pytest.approx(1.0, abs=1e-3), (
            f"Mesh {mesh_name} normal at corner {key} has length {current_normal.length}, not 1. "
            f"A normal is a direction, so the DNA's linear unit must not be applied to it."
        )

        if mesh_name == changed_mesh_name and position_index == changed_normal_index:
            changed_corners += 1
            assert (expected_normal - current_normal).length > bounds["max"], (
                f"Mesh {mesh_name} normal at corner {key} should not match, since it was changed in blender."
            )
        elif position_index not in tolerated:
            errors.append((expected_normal - current_normal).length)

    if mesh_name == changed_mesh_name:
        assert changed_corners, f"Vertex {changed_normal_index} was never reached on mesh {mesh_name}."

    errors.sort()
    measured = {
        "mean": sum(errors) / len(errors),
        "p99": errors[int(len(errors) * 0.99)],
        "max": errors[-1],
    }
    for name, limit in bounds.items():
        assert measured[name] <= limit, (
            f"Mesh {mesh_name} normal round trip {name} error is {measured[name]:.6f}, over the {limit} limit. "
            f"Measured over {len(errors)} corners: {measured}."
        )


def assert_skin_weights(
    expected_data: dict,
    current_data: dict,
    mesh_name: str,
    mesh_vertex_count: int,
    attribute: str,
    changed_mesh_name: int,
    changed_vertex_group_name: str,
    changed_vertex_group_vertex_index: int,
    changed_vertex_group_weight: float,
    tolerance: float = 1e-3,
    assert_index_order: bool = True,
):
    expected_mesh_index = expected_data[DNA_DEFINITION_VERSION]["meshNames"].index(mesh_name)
    current_mesh_index = current_data[DNA_DEFINITION_VERSION]["meshNames"].index(mesh_name)

    for vertex_index in range(mesh_vertex_count):
        expected_joint_indices = expected_data[DNA_GEOMETRY_VERSION]["meshes"][expected_mesh_index][attribute][
            vertex_index
        ]["jointIndices"]
        current_joint_indices = current_data[DNA_GEOMETRY_VERSION]["meshes"][current_mesh_index][attribute][
            vertex_index
        ]["jointIndices"]

        expected_weights = expected_data[DNA_GEOMETRY_VERSION]["meshes"][expected_mesh_index][attribute][vertex_index][
            "weights"
        ]
        current_weights = current_data[DNA_GEOMETRY_VERSION]["meshes"][current_mesh_index][attribute][vertex_index][
            "weights"
        ]

        # The mesh indices should be the same
        if assert_index_order:
            assert (
                len(expected_joint_indices) == len(current_joint_indices)
            ), f"Mesh {mesh_name} vertex index {vertex_index} {attribute} joint indices length mismatch. Expected {len(expected_joint_indices)} indices but has {len(current_joint_indices)}."

            for index, (expected_joint_index, current_joint_index) in enumerate(
                zip(expected_joint_indices, current_joint_indices, strict=False)
            ):
                assert (
                    expected_joint_index == current_joint_index
                ), f"Mesh {mesh_name} vertex index {vertex_index} {attribute} joint indices order mismatch at array index {index}. Expected {expected_joint_index} but has {current_joint_index}."

            for index, (expected_joint_index, _current_joint_index) in enumerate(
                zip(expected_joint_indices, current_joint_indices, strict=False)
            ):
                expected_weight = expected_weights[index]
                current_weight = current_weights[index]
                joint_name = expected_data[DNA_DEFINITION_VERSION]["jointNames"][expected_joint_index]

                # if this is the changed vertex group index
                if (
                    mesh_name == changed_mesh_name
                    and joint_name == changed_vertex_group_name
                    and vertex_index == changed_vertex_group_vertex_index
                ):
                    assert (
                        current_weight == pytest.approx(changed_vertex_group_weight, abs=tolerance)
                    ), f"Mesh {mesh_name} {attribute} on vertex group index {changed_vertex_group_vertex_index} should have weight since it was changed in blender {changed_vertex_group_weight} but has {current_weight}."
                else:
                    assert (
                        expected_weight == pytest.approx(current_weight, abs=tolerance)
                    ), f"Mesh {mesh_name} {attribute} on vertex group index {expected_joint_index} mismatch. Expected {expected_weight} but has {current_weight}."

        # Otherwise, the mesh indices can be in any order but still the number unique indices should be the same
        else:
            sorted_expected_values = sorted(expected_weights)
            sorted_current_values = sorted(current_weights)

            for expected_weight, current_weight in zip(sorted_expected_values, sorted_current_values, strict=False):
                if (
                    expected_weight != pytest.approx(current_weight, abs=tolerance)
                    and vertex_index == changed_vertex_group_vertex_index
                ):
                    assert (
                        expected_weights[changed_vertex_group_vertex_index] == expected_weight
                    ), f"Mesh {mesh_name} {attribute} on vertex group index {changed_vertex_group_vertex_index} value mismatch. The vertex group index that was moved was {changed_vertex_group_vertex_index} but this is not that one."
                else:
                    assert (
                        expected_weight == pytest.approx(current_weight, abs=tolerance)
                    ), f"Mesh {mesh_name} {attribute} on vertex group index {changed_vertex_group_vertex_index} value mismatch. Expected {expected_weight} but has {current_weight}."
