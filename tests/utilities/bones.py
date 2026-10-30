import bpy

from mathutils import Vector

from character_dna.constants import EXTRA_BONES
from character_dna.utilities import deselect_all, switch_to_object_mode, switch_to_pose_mode


def get_bone_differences(
    source_rig_name: str,
    target_rig_name: str | None = None,
    target_bone_locations: dict | None = None,
    tolerance: float = 0.01,
    ignore_prefix: str | None = None,
    isolated_bones: list[str] | None = None,
) -> tuple[list, dict]:
    differences = []
    if not target_bone_locations:
        target_bone_locations = {}

    source_rig = bpy.data.objects[source_rig_name]
    # Keep the established pose context between numerical samples. Leaving and
    # re-entering it evaluates the same rig twice; a flush is sufficient when
    # the source is already the active pose object.
    if bpy.context.mode == "POSE" and bpy.context.object == source_rig:
        bpy.context.view_layer.update()
    else:
        switch_to_pose_mode(source_rig)

    bone_names = source_rig.pose.bones.keys()  # type: ignore
    if isolated_bones:
        bone_names = isolated_bones

    # get the bone differences against the passed in target bone locations
    # this is used to test against the saved json files for more speed
    if target_bone_locations:
        for bone_name in bone_names:
            # skip the extra bones
            if bone_name in [i for i, _ in EXTRA_BONES]:
                continue

            if ignore_prefix and bone_name.startswith(ignore_prefix):
                continue

            source_bone = source_rig.pose.bones[bone_name]  # type: ignore
            source_world_location = source_rig.matrix_world @ source_bone.head
            loc_diff = (source_world_location - Vector(target_bone_locations[bone_name])).length
            if loc_diff >= tolerance:
                differences.append((bone_name, loc_diff))
    # get the bone differences against the target rig in the scene
    elif target_rig_name:
        target_rig = bpy.data.objects[target_rig_name]
        for bone_name in bone_names:  # type: ignore
            if ignore_prefix and bone_name.startswith(ignore_prefix):
                continue

            source_bone = source_rig.pose.bones[bone_name]  # type: ignore
            target_bone = target_rig.pose.bones.get(bone_name)  # type: ignore

            if target_bone:
                source_world_location = source_rig.matrix_world @ source_bone.head
                target_world_location = target_rig.matrix_world @ target_bone.head
                target_bone_locations[bone_name] = target_world_location[:]

                loc_diff = (source_world_location - target_world_location).length
                if loc_diff >= tolerance:
                    differences.append((bone_name, loc_diff))

    return differences, target_bone_locations


def show_differences(source_rig_name: str, target_rig_name: str, differences: list[tuple[str, float]]):
    # hide all bones
    source_rig = bpy.data.objects[source_rig_name]
    source_rig.hide_set(False)
    for bone in source_rig.data.bones:  # type: ignore
        bone.hide = True
    target_rig = bpy.data.objects[target_rig_name]
    target_rig.hide_set(False)
    for bone in target_rig.data.bones:  # type: ignore
        bone.hide = True

    # switch to pose mode with both rigs selected
    deselect_all()
    switch_to_object_mode()
    source_rig.select_set(True)
    target_rig.select_set(True)
    bpy.context.view_layer.objects.active = target_rig  # type: ignore
    bpy.ops.object.mode_set(mode="POSE")

    # show the bones with differences
    for bone_name, _ in differences:
        source_bone = source_rig.data.bones[bone_name]  # type: ignore
        target_bone = target_rig.data.bones[bone_name]  # type: ignore
        source_bone.hide = False
        target_bone.hide = False
