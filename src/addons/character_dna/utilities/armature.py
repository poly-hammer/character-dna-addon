# standard library imports
import logging
import math

# third party imports
import bmesh
import bpy

from mathutils import Euler, Matrix, Quaternion, Vector

# local imports
from ..constants import (
    CUSTOM_BONE_SHAPE_NAME,
    CUSTOM_BONE_SHAPE_SCALE,
    INTERNAL_BONE_COLLECTION,
    VOLUME_BONE_COLLECTION,
    BodyBoneCollection,
)
from ..typing import *  # noqa: F403
from .misc import (
    exclude_rig_instance_evaluation,
    switch_to_pose_mode,
)


logger = logging.getLogger(__name__)


def get_bone_rest_transformations(
    bone: bpy.types.Bone, force_object_space: bool = False, rotation_mode: str = "XYZ"
) -> tuple[Vector, Euler, Vector, Matrix]:
    if force_object_space:
        rest_to_parent_matrix = bone.matrix_local
    elif bone.parent:
        rest_to_parent_matrix = bone.parent.matrix_local.inverted_safe() @ bone.matrix_local
    else:
        rest_to_parent_matrix = bone.matrix_local

    bone_matrix_parent_space = rest_to_parent_matrix @ Matrix.Identity(4)
    # get respective transforms in parent space
    rest_location, rest_rotation, rest_scale = bone_matrix_parent_space.decompose()

    if rotation_mode in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
        rest_rotation = rest_rotation.to_euler(rotation_mode)

    return rest_location, rest_rotation, rest_scale, rest_to_parent_matrix  # type: ignore[return-value]


def get_bone_shape(name: str = CUSTOM_BONE_SHAPE_NAME) -> bpy.types.Object | None:
    rotations = [
        [90, 0, 0],
        [0, 90, 0],
        [0, 0, 90],
    ]
    new_objects = []
    sphere_control = bpy.data.objects.get(name)
    if not sphere_control:
        for rotation in rotations:
            bpy.ops.mesh.primitive_circle_add(
                vertices=16,
                radius=1,
                enter_editmode=False,
                align="WORLD",
                location=[0, 0, 0],
                scale=[1, 1, 1],
                rotation=[math.radians(rotation[0]), math.radians(rotation[1]), math.radians(rotation[2])],
            )
            new_objects.append(bpy.context.active_object)

        for new_object in new_objects:
            new_object.select_set(True)

        bpy.ops.object.join()
        if bpy.context.active_object:
            bpy.context.active_object.name = name
        sphere_control = bpy.data.objects.get(name)
        if not sphere_control:
            return None
        sphere_control.use_fake_user = True

    if (
        bpy.context.collection
        and bpy.context.collection.objects
        and sphere_control in bpy.context.collection.objects.values()
    ):
        bpy.context.collection.objects.unlink(sphere_control)

    sphere_control.hide_viewport = True
    return sphere_control


def assign_body_bone_collections(
    rig_object: bpy.types.Object,
    swing_bone_names: tuple[str, ...] = (),
    twist_bone_names: tuple[str, ...] = (),
    driver_bone_names: tuple[str, ...] = (),
    driven_bone_names: tuple[str, ...] = (),
):
    if (not rig_object.data or not rig_object.pose) or not isinstance(rig_object.data, bpy.types.Armature):
        return

    items = (
        (BodyBoneCollection.DRIVERS, driver_bone_names, "THEME09", True),
        (BodyBoneCollection.DRIVEN, driven_bone_names, "THEME01", True),
        (BodyBoneCollection.TWISTS, twist_bone_names, "THEME03", False),
        (BodyBoneCollection.SWINGS, swing_bone_names, "THEME04", False),
    )
    for collection_name, bone_names, theme, visible in items:
        if not bone_names:
            continue

        # create the bone collection if it does not already exist
        collection = rig_object.data.collections.get(collection_name)
        if not collection:
            collection = rig_object.data.collections.new(name=collection_name)
        collection.is_visible = visible

        # remove bones from other collections
        for other_collection in rig_object.data.collections:
            if other_collection.name != collection_name:
                for bone_name in bone_names:
                    pose_bone = rig_object.pose.bones.get(bone_name)
                    if pose_bone:
                        other_collection.unassign(pose_bone)
                        if pose_bone.color:
                            pose_bone.color.palette = "DEFAULT"

        # add bones to the correct collection
        for bone_name in bone_names:
            pose_bone = rig_object.pose.bones.get(bone_name)
            if pose_bone:
                collection.assign(pose_bone)
                if theme and pose_bone.color:
                    pose_bone.color.palette = theme  # type: ignore[value-assign]


def set_pose_bone_custom_color(pose_bone: bpy.types.PoseBone, color: tuple[float, ...]):
    """Apply an explicit RGB color to a single pose bone via its custom palette.

    Blender bone collections cannot carry a color, so per-joint rig-definition
    colors are applied directly to each bone. The ``select`` and ``active``
    states are derived as progressively lighter tints of ``color``.
    """
    if not pose_bone.color or len(color) < 3:
        return

    normal = (float(color[0]), float(color[1]), float(color[2]))
    select = tuple(channel + (1.0 - channel) * 0.4 for channel in normal)
    active = tuple(channel + (1.0 - channel) * 0.7 for channel in normal)

    pose_bone.color.palette = "CUSTOM"  # type: ignore[assignment]
    pose_bone.color.custom.normal = normal  # type: ignore[assignment]
    pose_bone.color.custom.select = select  # type: ignore[assignment]
    pose_bone.color.custom.active = active  # type: ignore[assignment]


def assign_joint_group_bone_collections(
    rig_object: bpy.types.Object,
    joint_groups: "tuple[RigJointGroup, ...]",
    color_by_joint_name: "dict[str, tuple[float, ...]] | None" = None,
    exclude_joint_names: "set[str]" = frozenset(),  # type: ignore[assignment]
):
    """Create a bone collection per rig-definition joint group and color bones.

    Each joint group becomes a bone collection named after the group, with its
    member bones assigned. Bones in ``exclude_joint_names`` (the volume bones,
    which are owned by the Volume collection) are kept out of every group
    collection so the groups stay disjoint and can be hidden individually. When
    ``color_by_joint_name`` is provided, every matching bone is tinted with its
    rig-definition color. Bones and colors absent from the rig are skipped.
    """
    if (not rig_object.data or not rig_object.pose) or not isinstance(rig_object.data, bpy.types.Armature):
        return

    for joint_group in joint_groups:
        if not joint_group.joints:
            continue

        collection = rig_object.data.collections.get(joint_group.name)
        if not collection:
            collection = rig_object.data.collections.new(name=joint_group.name)

        for bone_name in joint_group.joints:
            if bone_name in exclude_joint_names:
                continue
            pose_bone = rig_object.pose.bones.get(bone_name)
            if pose_bone:
                collection.assign(pose_bone)

    if color_by_joint_name:
        for bone_name, color in color_by_joint_name.items():
            pose_bone = rig_object.pose.bones.get(bone_name)
            if pose_bone:
                set_pose_bone_custom_color(pose_bone, color)


def get_volume_joint_names(rig_object: bpy.types.Object, skinned_joint_names: "set[str]") -> "set[str]":
    """Return the volume bones of ``rig_object``: the non-skinned leaf joints.

    MetaHuman head rigs contain many volumetric helper joints that drive no
    mesh vertices. The ones that are also leaves (no children) are the volume
    bones -- ``skinned_joint_names`` is the set of joint names with at least
    one skin-weight influence. Returns an empty set when the rig has no
    armature data.
    """
    if not rig_object.data or not isinstance(rig_object.data, bpy.types.Armature):
        return set()
    return {bone.name for bone in rig_object.data.bones if not bone.children and bone.name not in skinned_joint_names}


def assign_volume_bone_collection(
    rig_object: bpy.types.Object,
    volume_joint_names: "set[str]",
):
    """Author the ``Volume`` bone collection: the non-skinned leaf joints.

    The volume bones (such as the volumetric face helpers) drive no mesh
    vertices and only add visual clutter, so they are gathered into their own
    collection that can be hidden out of the way. Skipped when the rig has no
    armature data.
    """
    if (not rig_object.data or not rig_object.pose) or not isinstance(rig_object.data, bpy.types.Armature):
        return

    collection = rig_object.data.collections.get(VOLUME_BONE_COLLECTION)
    if not collection:
        collection = rig_object.data.collections.new(name=VOLUME_BONE_COLLECTION)

    for bone_name in volume_joint_names:
        pose_bone = rig_object.pose.bones.get(bone_name)
        if pose_bone:
            collection.assign(pose_bone)


def assign_internal_bone_collection(
    rig_object: bpy.types.Object,
    skinned_joint_names: "set[str]",
    surface_skinned_joint_names: "set[str]",
):
    """Author the ``Internal`` bone collection: the internal-skeleton joints.

    A bone is internal when it has children, or it is a leaf joint that skins a
    non-surface mesh (such as the teeth or eyes) rather than the surface
    (``*_lod0_mesh``) mesh. This excludes both the volume bones (leaf joints that
    skin no LOD0 mesh) and the surface-skinning leaf bones, while still keeping
    internal anatomy such as the teeth joints. ``skinned_joint_names`` is the set
    of joints that skin any LOD0 mesh and ``surface_skinned_joint_names`` the
    subset that skins the surface mesh. Gathering them into their own collection
    lets the user solo it to get a clear view of the internal skeleton. Skipped
    when the rig has no armature data.
    """
    if (not rig_object.data or not rig_object.pose) or not isinstance(rig_object.data, bpy.types.Armature):
        return

    collection = rig_object.data.collections.get(INTERNAL_BONE_COLLECTION)
    if not collection:
        collection = rig_object.data.collections.new(name=INTERNAL_BONE_COLLECTION)

    for bone in rig_object.data.bones:
        is_internal = bool(bone.children) or (
            bone.name in skinned_joint_names and bone.name not in surface_skinned_joint_names
        )
        if not is_internal:
            continue
        pose_bone = rig_object.pose.bones.get(bone.name)
        if pose_bone:
            collection.assign(pose_bone)


def get_meshes_using_armature(armature_object: bpy.types.Object) -> list[bpy.types.Object]:
    # find the related mesh objects for the head rig
    mesh_objects = []
    for mesh_object in bpy.data.objects:
        if mesh_object.type == "MESH":
            for modifier in mesh_object.modifiers:
                if modifier.type == "ARMATURE" and modifier.object == armature_object:  # type: ignore[attr-defined]
                    mesh_objects.append(mesh_object)
                    break
    return mesh_objects


def get_closet_vertex_to_bone(
    mesh_object: bpy.types.Object, pose_bone: bpy.types.PoseBone, max_distance: float = 0.01
) -> bpy.types.MeshVertex | None:
    # get the bone applied position not the pose position
    bone = pose_bone.id_data.data.bones[pose_bone.name]  # type: ignore[attr-defined]
    position = mesh_object.matrix_world.inverted() @ bone.head_local
    vert = min(
        mesh_object.data.vertices,  # type: ignore[attr-defined]
        key=lambda vert: (position - vert.co).length_squared,
    )
    distance = (position - vert.co).length_squared
    # only return the vertex if it is within the max distance
    if distance < max_distance:
        return vert
    logger.warning(f'Vertex {vert.index} is too far from bone "{pose_bone.name}":\n{distance} > {max_distance}')
    return None


def get_ray_cast_normal(
    mesh_object: bpy.types.Object, pose_bone: bpy.types.PoseBone, max_distance: float = 0.01
) -> Vector | None:
    vertex = get_closet_vertex_to_bone(mesh_object, pose_bone, max_distance)
    if vertex:
        return mesh_object.matrix_world @ vertex.normal
    return None


def get_vertex_positions(mesh_object: bpy.types.Object, bone_to_vert_index: dict[str, int]) -> dict[str, Vector]:
    positions = {}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bmesh_object = bmesh.new()
    bmesh_object.from_object(mesh_object, depsgraph)
    bmesh_object.verts.ensure_lookup_table()

    for bone_name, index in bone_to_vert_index.items():
        positions[bone_name] = bmesh_object.verts[index].co

    bmesh_object.free()

    return positions


def get_closet_vertex_indices_to_bones(
    mesh_object: bpy.types.Object, pose_bones: list[bpy.types.PoseBone], max_distance: float = 0.01
) -> dict[str, int]:
    bone_to_vert_index = {}

    # initialize the bmesh object to evaluate against the current depsgraph so
    # we get the correct vertex positions with taking into account modifiers
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bmesh_object = bmesh.new()
    bmesh_object.from_object(mesh_object, depsgraph)
    bmesh_object.verts.ensure_lookup_table()

    for pose_bone in pose_bones:
        position = pose_bone.matrix.translation
        vert = min(
            bmesh_object.verts,
            key=lambda vert: (position - vert.co).length_squared,
        )
        # only return the vertex if it is within the max distance
        distance = (position - vert.co).length_squared
        if distance < max_distance:
            bone_to_vert_index[pose_bone.name] = vert.index
        else:
            logger.warning(
                'Vertex %d is too far from bone "%s":\\n%f > %f', vert.index, pose_bone.name, distance, max_distance
            )

    bmesh_object.free()

    return bone_to_vert_index


@exclude_rig_instance_evaluation
def copy_armature(armature_object: bpy.types.Object, new_armature_name: str) -> bpy.types.Object:
    # remove the object if it already exists
    armature_object_copy = bpy.data.objects.get(new_armature_name)
    if armature_object_copy:
        bpy.data.objects.remove(armature_object_copy)

    # remove the existing armature if it exists
    armature = bpy.data.meshes.get(new_armature_name)
    if armature:
        bpy.data.armatures.remove(armature)  # pyright: ignore[reportArgumentType]

    # copy the armature
    armature_data = armature_object.data.copy()  # pyright: ignore[reportOptionalMemberAccess]
    armature_data.name = new_armature_name
    armature_object_copy = bpy.data.objects.get(new_armature_name)
    armature_object_copy = bpy.data.objects.new(name=new_armature_name, object_data=armature_data)

    # make sure the mesh is in the scene collection
    if bpy.context.scene and armature_object_copy not in bpy.context.scene.collection.objects.values():
        bpy.context.scene.collection.objects.link(armature_object_copy)

    # set custom bone shape
    bones_shape_object = get_bone_shape()
    switch_to_pose_mode(armature_object_copy)
    for pose_bone in armature_object_copy.pose.bones:  # type: ignore[attr-defined]
        pose_bone.custom_shape = bones_shape_object
        pose_bone.custom_shape_scale_xyz = CUSTOM_BONE_SHAPE_SCALE

    return armature_object_copy


def get_body_constraint_name(bone_name: str) -> str:
    return f"MH_DNA {bone_name} to body"


def constrain_head_to_body(instance: "RigInstance"):
    if not instance.head_rig or not instance.body_rig:
        logger.warning("Head rig or body rig not found. Cannot constrain head rig to body rig.")
        return

    body_bone_names = [pose_bone.name for pose_bone in instance.body_rig.pose.bones]

    # add copy transforms constraint to the head rig
    for pose_bone in instance.head_rig.pose.bones:
        if pose_bone.name in body_bone_names:
            name = get_body_constraint_name(pose_bone.name)
            constraint = pose_bone.constraints.get(name)
            if not constraint:
                constraint = pose_bone.constraints.new(type="COPY_TRANSFORMS")
                constraint.name = name

            constraint.target = instance.body_rig
            constraint.subtarget = pose_bone.name
            constraint.target_space = "WORLD"
            constraint.owner_space = "WORLD"

    # Drop the cached constraint list so the influence setter rebuilds it
    # from the freshly created constraints below (and on later edits).
    instance.data.pop(instance.cache_key("head", "body_constraints"), None)
    instance.head_to_body_constraint_influence = 1.0


def reset_pose(rig_object: bpy.types.Object):
    if not rig_object.pose:
        return
    # show the rig and switch to pose mode
    is_hidden = rig_object.hide_get()
    rig_object.hide_set(False)
    switch_to_pose_mode(rig_object)

    # reset to rest pose
    for pose_bone in rig_object.pose.bones:
        pose_bone.rotation_quaternion = Quaternion((1, 0, 0, 0))
        pose_bone.rotation_euler = Euler((0, 0, 0))
        pose_bone.location = Vector((0, 0, 0))
        pose_bone.scale = Vector((1, 1, 1))

    # restore the rig's hidden state
    rig_object.hide_set(is_hidden)


def get_bone_local_axes(pose_bone: bpy.types.PoseBone) -> tuple[Vector, Vector, Vector]:
    """
    Get the local X, Y, Z axes of a pose bone in world space.

    Args:
        pose_bone: The pose bone to analyze

    Returns:
        Tuple of (x_axis, y_axis, z_axis) as world-space vectors
    """
    # Get the bone's world matrix
    world_matrix = pose_bone.id_data.matrix_world @ pose_bone.matrix  # type: ignore[attr-defined]

    # Extract the rotation component (3x3 part of 4x4 matrix)
    # Each column represents a local axis in world space
    x_axis = world_matrix.col[0].to_3d().normalized()
    y_axis = world_matrix.col[1].to_3d().normalized()
    z_axis = world_matrix.col[2].to_3d().normalized()

    return x_axis, y_axis, z_axis


def compare_bone_orientations(bone1: bpy.types.PoseBone, bone2: bpy.types.PoseBone) -> bool:
    """
    Compare if two bones have the same local orientations.

    Args:
        bone1: First pose bone
        bone2: Second pose bone

    Returns:
        True if orientations are similar
    """
    x1, y1, z1 = get_bone_local_axes(bone1)
    x2, y2, z2 = get_bone_local_axes(bone2)

    # Compare axes using dot product (1.0 = same direction, -1.0 = opposite)
    x_match = abs(x1.dot(x2)) > 0.999
    y_match = abs(y1.dot(y2)) > 0.999
    z_match = abs(z1.dot(z2)) > 0.999

    return x_match and y_match and z_match


def get_pose_bone_local_quaternion(pose_bone: bpy.types.PoseBone) -> Quaternion:
    """
    Calculate the local quaternion rotation of a pose bone using world space matrices.

    This method works even when the bone is constrained by calculating the rotation
    from the bone's evaluated world space direction vector. Note, this only works if
    the pose bone passed in is from an already evaluated armature object.
    (i.e., armature.evaluated_get(dependency_graph)).

    Args:
        pose_bone: The pose bone to get the local quaternion from

    Returns:
        The local quaternion rotation in the bone's parent space
    """
    # Solve for matrix_basis
    if pose_bone.parent:
        parent_world_matrix = pose_bone.parent.matrix
        parent_rest_local_matrix = pose_bone.parent.bone.matrix_local
        matrix_basis = (
            pose_bone.bone.matrix_local.inverted_safe()
            @ parent_rest_local_matrix
            @ parent_world_matrix.inverted_safe()
            @ pose_bone.matrix
        )
    else:
        matrix_basis = (
            pose_bone.bone.matrix_local.inverted_safe() @ pose_bone.id_data.matrix_world.inverted() @ pose_bone.matrix  # type: ignore[attr-defined]
        )

    # Extract and return the quaternion
    return matrix_basis.to_quaternion().normalized()


def get_pose_bone_local_transform(pose_bone: bpy.types.PoseBone) -> tuple[Vector, Quaternion, Vector]:
    """
    Calculate the local (parent space) location, rotation, and scale of a pose bone using
    world space matrices.

    Like :func:`get_pose_bone_local_quaternion`, this works even when the bone is constrained
    (e.g. driven by a control rig) because it derives the local transform from the bone's
    evaluated world space matrix rather than its ``matrix_basis``. The pose bone must come from
    an already evaluated armature object (i.e. ``armature.evaluated_get(dependency_graph)``).

    Args:
        pose_bone: The evaluated pose bone to read the local transform from.

    Returns:
        A tuple of (location, rotation_quaternion, scale) in the bone's parent space.
    """
    if pose_bone.parent:
        matrix_basis = (
            pose_bone.bone.matrix_local.inverted_safe()
            @ pose_bone.parent.bone.matrix_local
            @ pose_bone.parent.matrix.inverted_safe()
            @ pose_bone.matrix
        )
    else:
        matrix_basis = (
            pose_bone.bone.matrix_local.inverted_safe() @ pose_bone.id_data.matrix_world.inverted() @ pose_bone.matrix  # type: ignore[attr-defined]
        )

    location, rotation, scale = matrix_basis.decompose()
    return location, rotation.normalized(), scale
