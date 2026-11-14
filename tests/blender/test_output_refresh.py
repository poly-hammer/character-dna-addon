"""Output follows current scene and material dependencies, not retained IDs."""

import bpy
import pytest


def _material(kind):
    material = bpy.data.materials.new(f"{kind}_output_test")
    material.use_nodes = True
    tree = bpy.data.node_groups.new(f"{kind}_texture_logic", "ShaderNodeTree")
    sockets = ["Color_MAIN", "Normal_MAIN"]
    if kind == "head":
        sockets += ["Color_CM1", "wm1.head_wm1_jawOpen_msk"]
    for name in sockets:
        tree.interface.new_socket(name=name, in_out="INPUT", socket_type="NodeSocketColor")
    group = material.node_tree.nodes.new("ShaderNodeGroup")
    group.node_tree = tree
    image = bpy.data.images.new("current_output_image", 1, 1)
    nodes = []
    roles = ["Color_MAIN", "Color_CM1" if kind == "head" else "Normal_MAIN"]
    for role in roles:
        node = material.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        material.node_tree.links.new(node.outputs["Color"], group.inputs[role])
        nodes.append(node)
    return material, image, nodes


@pytest.mark.parametrize("kind", ["head", "body"])
def test_refresh_removes_disconnected_images_and_preserves_distinct_roles(minimal_character, kind):
    instance = minimal_character
    material, image, nodes = _material(kind)
    setattr(instance, f"{kind}_material", material)
    items = getattr(instance.output, f"{kind}_item_list")
    expected_names = (
        {"Head_Basecolor.png", "Head_Basecolor_Animated_CM1.png"}
        if kind == "head"
        else {"Body_Basecolor.png", "Body_Normal.png"}
    )
    assert {item.name for item in items if item.image_object} == expected_names
    kept = next(item for item in items if item.image_object and "Basecolor." in item.name)
    kept.include = False
    duplicate = items.add()
    duplicate.name, duplicate.image_object = kept.name, image
    stale = items.add()
    stale.name = kept.name
    stale.image_object = bpy.data.images.new("old_material_image", 1, 1)
    assert bpy.ops.character_dna.refresh_output_items() == {"FINISHED"}
    image_items = [item for item in items if item.image_object]
    assert len(image_items) == 2
    assert {item.name for item in image_items} == expected_names
    assert all(item.image_object == image for item in image_items)
    assert not next(item for item in image_items if "Basecolor." in item.name).include

    # An image datablock can still exist after its node is cleared/disconnected.
    nodes[1].image = None
    bpy.ops.character_dna.refresh_output_items()
    assert len([item for item in items if item.image_object]) == 1
    material.node_tree.links.clear()
    bpy.ops.character_dna.refresh_output_items()
    assert not any(item.image_object for item in items)

    material.node_tree.links.new(
        nodes[0].outputs["Color"],
        next(node for node in material.node_tree.nodes if node.type == "GROUP").inputs["Color_MAIN"],
    )
    bpy.ops.character_dna.refresh_output_items()
    assert any(item.image_object for item in items)
    setattr(instance, f"{kind}_material", None)
    bpy.ops.character_dna.refresh_output_items()
    assert not any(item.image_object for item in items)


@pytest.mark.parametrize("kind", ["head", "body"])
def test_refresh_drops_unlinked_meshes_and_cleared_rigs(minimal_character, kind):
    instance = minimal_character
    mesh = getattr(instance, f"{kind}_mesh")
    items = getattr(instance.output, f"{kind}_item_list")
    mesh_item = next(item for item in items if item.scene_object == mesh)
    mesh_item.name = "OldMeshName"
    mesh_item.editable_name = True
    mesh_item.include = False
    orphan = mesh.copy()
    orphan.name = "UnlinkedOldMesh"
    stale = items.add()
    stale.scene_object = orphan
    assert not orphan.users_scene
    bpy.ops.character_dna.refresh_output_items()
    assert orphan not in {item.scene_object for item in items}
    mesh_item = next(item for item in items if item.scene_object == mesh)
    assert mesh_item.name == f"{kind}_lod0_mesh"
    assert not mesh_item.editable_name
    assert not mesh_item.include
    setattr(instance, f"{kind}_rig", None)
    bpy.ops.character_dna.refresh_output_items()
    assert not items
