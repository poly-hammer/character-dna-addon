import shutil

from pathlib import Path
from typing import TYPE_CHECKING

import bmesh
import bpy
import pytest

from mathutils import Euler, Vector

from constants import TEST_DNA_FOLDER, TEST_FBX_FOLDER


if TYPE_CHECKING:
    from character_dna.rig_instance import RigInstance


def load_dna(
    file_path: Path,
    import_lods: list,
    include_body: bool = True,
    import_shape_keys: bool = False,
    import_face_board: bool = True,
):
    # open default scene
    bpy.ops.wm.read_homefile(app_template="")

    # remove all default objects
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    lods_to_import = {}
    # Set all LODs to False by default
    for index in range(8):
        lods_to_import[f"import_lod{index}"] = False
    # Set the LODs to True that are in the import_lods list
    for lod_name in import_lods:
        lods_to_import[f"import_{lod_name}"] = True

    bpy.ops.character_dna.import_dna(  # type: ignore
        filepath=str(file_path),
        import_mesh=True,
        import_bones=True,
        import_shape_keys=import_shape_keys,
        import_vertex_groups=True,
        import_materials=True,
        import_face_board=import_face_board,
        include_body=include_body,
        **lods_to_import,
    )


def _load_temp_body_dna(
    file_name: str, temp_folder: Path, dna_folder_name: str, import_shape_keys: bool, import_lods: list
):
    destination_file_path = temp_folder / dna_folder_name / file_name

    # copy the dna file to the temp folder so we don't modify the original
    destination_file_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src=TEST_DNA_FOLDER / dna_folder_name / file_name, dst=destination_file_path)
    # copy the export manifest as well (This is used for naming the imported instance)
    shutil.copy(
        src=TEST_DNA_FOLDER / dna_folder_name / "ExportManifest.json",
        dst=temp_folder / dna_folder_name / "ExportManifest.json",
    )

    load_dna(
        file_path=destination_file_path,
        import_lods=import_lods,
        import_shape_keys=import_shape_keys,
        import_face_board=False,
        include_body=False,
    )


@pytest.fixture(scope="session")
def load_head_dna(
    addon,
    dna_folder_name: str,
    import_shape_keys: bool,
    import_lods: list,
):
    load_dna(
        file_path=TEST_DNA_FOLDER / dna_folder_name / "head.dna",
        import_lods=import_lods,
        import_shape_keys=import_shape_keys,
        import_face_board=True,
        include_body=True,
    )


@pytest.fixture(scope="session")
def load_body_dna(
    addon,
    dna_folder_name: str,
    import_shape_keys: bool,
    import_lods: list,
):
    load_dna(
        file_path=TEST_DNA_FOLDER / dna_folder_name / "body.dna",
        import_lods=import_lods,
        import_shape_keys=import_shape_keys,
        import_face_board=False,
        include_body=False,
    )


@pytest.fixture(scope="session")
def load_body_dna_for_pose_editing(
    addon,
    temp_folder,
    dna_folder_name: str,
    import_shape_keys: bool,
    import_lods: list,
):
    _load_temp_body_dna(
        file_name="body.dna",
        temp_folder=temp_folder,
        dna_folder_name=dna_folder_name,
        import_shape_keys=import_shape_keys,
        import_lods=import_lods,
    )


@pytest.fixture(scope="session")
def load_body_dna_for_pose_roundtrip(
    addon,
    temp_folder,
    dna_folder_name: str,
    import_shape_keys: bool,
    import_lods: list,
):
    _load_temp_body_dna(
        file_name="body.dna",
        temp_folder=temp_folder,
        dna_folder_name=dna_folder_name,
        import_shape_keys=import_shape_keys,
        import_lods=import_lods,
    )


@pytest.fixture(scope="session")
def load_full_dna_for_animation(
    addon,
    temp_folder,
    dna_folder_name: str,
    import_shape_keys: bool,
    import_lods: list,
):
    load_dna(
        file_path=TEST_DNA_FOLDER / dna_folder_name / "head.dna",
        import_lods=import_lods,
        import_shape_keys=import_shape_keys,
        import_face_board=True,
        include_body=True,
    )


@pytest.fixture(scope="session")
def load_dna_for_rig_instance_ops(addon):
    load_dna(
        file_path=TEST_DNA_FOLDER / "ada" / "head.dna",
        import_lods=["lod0"],
        import_shape_keys=False,
        import_face_board=True,
        include_body=True,
    )


@pytest.fixture
def load_head_only_dna(addon):
    """A head DNA imported with no body, so nothing constrains the head rig's neck bones."""
    load_dna(
        file_path=TEST_DNA_FOLDER / "ada" / "head.dna",
        import_lods=["lod0"],
        import_shape_keys=False,
        import_face_board=True,
        include_body=False,
    )


@pytest.fixture(scope="session")
def load_mhc_conformed_topology_meshes(addon):
    # open default scene
    bpy.ops.wm.read_homefile(app_template="")
    # import head and body wrapped meshes
    for component in ["head", "body"]:
        file_path = TEST_FBX_FOLDER / "mhc_conformed_topology" / f"{component}.fbx"
        bpy.ops.import_scene.fbx(filepath=str(file_path))
    # The FBX importer leaves the Y-up -> Z-up conversion as an unapplied object
    # rotation. The converter requires zeroed transforms, so apply them (this is
    # geometrically neutral: world-space geometry is preserved).
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.select_all(action="DESELECT")


@pytest.fixture(scope="session")
def setup_reference_blend_file(addon, temp_folder) -> Path:
    load_dna(
        file_path=TEST_DNA_FOLDER / "ada" / "head.dna",
        import_lods=["lod0"],
        import_shape_keys=False,
        import_face_board=True,
        include_body=True,
    )
    instance = bpy.context.scene.character_dna.rig_instance_list[0]
    face = instance.face_board
    face["authored_reference_control"] = True
    jaw = face.pose.bones["CTRL_C_jaw"]
    for frame, value in ((1, 0.0), (10, 0.8), (25, 0.2), (50, 1.0), (100, 0.4), (200, 0.0)):
        jaw.location.y = value
        jaw.keyframe_insert(data_path="location", index=1, frame=frame)
    root = bpy.data.collections.get((instance.name, None))
    nested = bpy.data.collections.new("reference_nested")
    root.children.link(nested)
    helper = bpy.data.objects.new("reference_shared_helper", None)
    root.objects.link(helper)
    nested.objects.link(helper)
    _add_reference_control_rig(instance, root)
    bpy.context.scene.frame_set(1)
    file_path = temp_folder / "reference_blend_file.blend"
    # Save the blend file
    bpy.ops.wm.save_as_mainfile(filepath=str(file_path))

    return file_path


def _add_reference_control_rig(instance: "RigInstance", root: bpy.types.Collection) -> None:
    """Author a one-bone control rig with rest-offset intermediary binding and saved body samples."""
    body = instance.body_rig
    assert body is not None and body.pose is not None
    assert "upperarm_l" in body.pose.bones and "hand_l" in body.pose.bones
    armature = bpy.data.armatures.new(f"{instance.name}_control_rig")
    control = bpy.data.objects.new(armature.name, armature)
    root.objects.link(control)
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    control.select_set(True)
    bpy.context.view_layer.objects.active = control
    bpy.ops.object.mode_set(mode="EDIT")
    bone = armature.edit_bones.new("DEF-upper_arm.L")
    bone.head = (0.3, 0.2, 1.4)
    bone.tail = (0.3, 0.4, 1.4)
    bone.roll = 0.3
    bpy.ops.object.mode_set(mode="OBJECT")
    instance.control_rig = control
    widget_mesh = bpy.data.meshes.new("Arbitrary control artwork")
    widget_mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1), (1, 2), (2, 0)], [])
    widget = bpy.data.objects.new("Unrelated display geometry", widget_mesh)
    widget_parent = bpy.data.objects.new("Unrelated display parent", None)
    widget.parent = widget_parent
    control.pose.bones["DEF-upper_arm.L"].custom_shape = widget
    constraints = bpy.data.collections.new("reference_control_constraints")
    root.children.link(constraints)
    constraints.hide_viewport = True
    parent = bpy.data.objects.new("reference_control_parent", None)
    child = bpy.data.objects.new("reference_control_child", None)
    constraints.objects.link(parent)
    constraints.objects.link(child)
    parent.matrix_world = control.matrix_world @ armature.bones["DEF-upper_arm.L"].matrix_local
    child.matrix_world = body.matrix_world @ body.data.bones["upperarm_l"].matrix_local
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()
    binding = body.pose.bones["upperarm_l"].constraints.new("COPY_TRANSFORMS")
    binding.name = "Reference Control Rig"
    binding.target = child
    following = parent.constraints.new("COPY_TRANSFORMS")
    following.target = control
    following.subtarget = "DEF-upper_arm.L"
    pose = control.pose.bones["DEF-upper_arm.L"]
    pose.rotation_mode = "XYZ"
    for frame, angle in ((1, 0.0), (10, 0.65), (25, -0.35), (200, 0.0)):
        pose.rotation_euler.z = angle
        pose.keyframe_insert(data_path="rotation_euler", index=2, frame=frame)
    samples = {}
    for frame in (1, 10, 25, 200):
        bpy.context.scene.frame_set(frame)
        evaluated = body.evaluated_get(bpy.context.view_layer.depsgraph)
        samples[str(frame)] = {
            name: [value for row in evaluated.matrix_world @ evaluated.pose.bones[name].matrix for value in row]
            for name in ("upperarm_l", "hand_l")
        }
    assert samples["1"]["hand_l"] != samples["10"]["hand_l"]
    assert samples["10"]["hand_l"] != samples["25"]["hand_l"]
    assert samples["1"]["hand_l"] == pytest.approx(samples["200"]["hand_l"], abs=1e-5)
    control["reference_body_samples"] = samples


@pytest.fixture(scope="session")
def head_bmesh(load_head_dna) -> bmesh.types.BMesh | None:
    from character_dna.dna_io.exporter import DNAExporter
    from character_dna.utilities import get_active_head

    head = get_active_head()
    if head and head.head_mesh_object:
        return DNAExporter.get_bmesh(head.head_mesh_object)
    return None


@pytest.fixture(scope="session")
def head_armature(load_head_dna) -> bpy.types.Object | None:
    from character_dna.utilities import get_active_head

    head = get_active_head()
    if head and head.head_rig_object:
        return head.head_rig_object
    return None


@pytest.fixture(scope="session")
def modify_head_scene(
    load_head_dna,
    dna_folder_name: str,
    changed_head_bone_name: str,
    changed_head_bone_location: tuple[Vector, Vector],
    changed_head_bone_rotation: tuple[Euler, Euler],
    changed_head_mesh_name: str,
    changed_head_vertex_index: int,
    changed_head_vertex_location: tuple[Vector, Vector, Vector],
    changed_head_normal_index: int,
    changed_head_normal_vector: tuple[Vector, Vector, Vector],
    changed_head_vertex_group_name: str,
    changed_head_vertex_group_vertex_index: int,
    changed_head_vertex_group_weight: float,
    temp_folder,
):
    from utilities.modify import (
        apply_bone_transform,
        apply_vertex_group_weight,
        apply_vertex_normal,
        apply_vertex_transform,
    )

    # Make some changes
    apply_vertex_transform(
        prefix=dna_folder_name,
        mesh_name=changed_head_mesh_name,
        vertex_index=changed_head_vertex_index,
        location=changed_head_vertex_location[0],
    )
    apply_vertex_normal(
        prefix=dna_folder_name,
        mesh_name=changed_head_mesh_name,
        vertex_index=changed_head_normal_index,
        normal=changed_head_normal_vector[0],
    )
    apply_bone_transform(
        prefix=dna_folder_name,
        component="head",
        bone_name=changed_head_bone_name,
        location=changed_head_bone_location[0],
        rotation=changed_head_bone_rotation[0],
    )
    apply_vertex_group_weight(
        prefix=dna_folder_name,
        mesh_name=changed_head_mesh_name,
        vertex_group_name=changed_head_vertex_group_name,
        vertex_index=changed_head_vertex_group_vertex_index,
        weight=changed_head_vertex_group_weight,
    )

    # Save the blend file
    bpy.ops.wm.save_as_mainfile(filepath=str(temp_folder / f"{dna_folder_name}_head_modified.blend"))
