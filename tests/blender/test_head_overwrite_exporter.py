"""End-to-end coverage for the ``overwrite`` DNA export path (``DNAExporter``).

The overwrite exporter rewrites the mesh and joint tables from the Blender scene
"from scratch" while preserving the behavior layer (raw controls, PSDs, RBFs, GUI
controls, joint groups, blend-shape channel wiring) copied verbatim from the
source DNA. This locks two contracts that the geometry/bone assertions in
``test_head_exporter.py`` do not cover:

1. Round-tripping the head unchanged must preserve every behavior-layer entity
   count and every mesh's blend-shape targets -- even though the artist never
   imported shape keys into the scene (the common case), the source targets must
   be copied through rather than wiped by ``clearMeshes``.
2. Deleting a secondary mesh object (e.g. swapping in a custom teeth mesh) must
   drop exactly that mesh, renumber the remaining meshes, and keep the
   mesh -> blend-shape channel mapping pointing at valid mesh indices.

These tests are CI-safe: a single head LOD is loaded with no body and no shape
keys, so no full-vertex shape-key blocks are allocated.
"""

import pytest

from constants import TEST_DNA_FOLDER
from fixtures.rigs import RigProfile


def _open_reader(dna_file_path):
    from character_dna.dna_io import get_dna_reader

    return get_dna_reader(file_path=dna_file_path, file_format="binary", data_layer="All")


def _behavior_entity_counts(reader) -> dict:
    """Behavior/definition-layer entity counts that the overwrite path must
    preserve verbatim from the source DNA."""
    return {
        "guiControls": reader.getGUIControlCount(),
        "rawControls": reader.getRawControlCount(),
        "psds": reader.getPSDCount(),
        "joints": reader.getJointCount(),
        "jointGroups": reader.getJointGroupCount(),
        "blendShapeChannels": reader.getBlendShapeChannelCount(),
        "animatedMaps": reader.getAnimatedMapCount(),
        "rbfPoses": reader.getRBFPoseCount(),
    }


def _blend_shape_target_counts_by_mesh(reader) -> dict[str, int]:
    """Map each mesh name to its blend-shape target count."""
    return {
        reader.getMeshName(mesh_index): reader.getBlendShapeTargetCount(mesh_index)
        for mesh_index in range(reader.getMeshCount())
    }


def _lod0_mesh_names(reader) -> set[str]:
    """Names of the meshes belonging to LOD0 in ``reader``."""
    return {reader.getMeshName(index) for index in reader.getMeshIndicesForLOD(0)}


def _run_overwrite_export(export_folder):
    """Run the overwrite exporter against the currently loaded head scene and
    return the written DNA path."""
    from character_dna.dna_io import DNAExporter
    from character_dna.utilities import get_active_head

    head = get_active_head()
    assert head is not None and head.rig_instance is not None, "no active head after import"

    export_folder.mkdir(parents=True, exist_ok=True)
    head.rig_instance.output.folder_path = str(export_folder)
    head.rig_instance.output.method = "overwrite"

    DNAExporter(
        file_name="head.dna",
        instance=head.rig_instance,
        linear_modifier=head.linear_modifier,
    ).run()
    return export_folder / "head.dna"


@pytest.fixture(scope="module")
def overwritten_head_dna(rig_factory, temp_folder):
    """Import Ada's head (LOD0, no body, no shape keys), run the overwrite export
    unchanged, and yield ``(source_path, exported_path)``."""
    source_path = TEST_DNA_FOLDER / "ada" / "head.dna"
    profile = RigProfile(components=("head",), face_board=True, materials=True)
    with rig_factory.lease(profile, temp_folder / "overwrite_scene"):
        exported_path = _run_overwrite_export(temp_folder / "overwrite" / "ada")
        return source_path, exported_path


@pytest.fixture(scope="module")
def mesh_deleted_head_dna(rig_factory, temp_folder):
    """Import Ada's head (LOD0), delete a secondary LOD0 mesh object (mimicking a
    custom mesh swap), run the overwrite export, and yield
    ``(source_path, exported_path, deleted_mesh_name)``."""
    import bpy

    from character_dna.utilities import get_active_head

    source_path = TEST_DNA_FOLDER / "ada" / "head.dna"
    profile = RigProfile(components=("head",), face_board=True, materials=True)
    with rig_factory.lease(profile, temp_folder / "overwrite_delete_scene"):
        head = get_active_head()
        assert head is not None and head.rig_instance is not None
        prefix = head.rig_instance.name
        main_mesh = head.rig_instance.head_mesh

        # Pick a secondary LOD0 mesh output item to delete (not the main head mesh).
        deleted_object = None
        for output_item in head.rig_instance.output.head_item_list:
            scene_object = output_item.scene_object
            if (
                output_item.include
                and scene_object
                and scene_object is not main_mesh
                and scene_object.type == "MESH"
                and "_lod0_" in scene_object.name
            ):
                deleted_object = scene_object
                break

        assert deleted_object is not None, "no secondary LOD0 mesh found to delete"
        deleted_mesh_name = deleted_object.name.replace(f"{prefix}_", "")
        bpy.data.objects.remove(deleted_object, do_unlink=True)

        exported_path = _run_overwrite_export(temp_folder / "overwrite_delete" / "ada")
        return source_path, exported_path, deleted_mesh_name


def test_overwrite_preserves_behavior_entity_counts(overwritten_head_dna):
    """Every behavior-layer entity count must be unchanged after an unmodified
    overwrite export -- the overwrite path only rewrites geometry/joints."""
    source_path, exported_path = overwritten_head_dna
    source_counts = _behavior_entity_counts(_open_reader(source_path))
    exported_counts = _behavior_entity_counts(_open_reader(exported_path))
    assert exported_counts == source_counts


def test_overwrite_preserves_mesh_set(overwritten_head_dna):
    """An unmodified overwrite export (LOD0 only) keeps exactly the source's LOD0
    mesh set."""
    source_path, exported_path = overwritten_head_dna
    source_reader = _open_reader(source_path)
    exported_reader = _open_reader(exported_path)

    exported_meshes = {exported_reader.getMeshName(i) for i in range(exported_reader.getMeshCount())}
    assert exported_meshes == _lod0_mesh_names(source_reader)


def test_overwrite_preserves_blend_shape_targets(overwritten_head_dna):
    """With no scene shape keys, the source blend-shape targets must be copied
    through for every exported (LOD0) mesh (``clearMeshes`` wipes them, so this
    guards the source-copy path)."""
    source_path, exported_path = overwritten_head_dna
    source_targets = _blend_shape_target_counts_by_mesh(_open_reader(source_path))
    exported_targets = _blend_shape_target_counts_by_mesh(_open_reader(exported_path))

    # Guard against a vacuous pass: the exported meshes must actually carry blend shapes.
    assert sum(exported_targets.values()) > 0, "exported head DNA has no blend-shape targets"
    for mesh_name, exported_count in exported_targets.items():
        assert exported_count == source_targets.get(mesh_name), (
            f'blend-shape targets for mesh "{mesh_name}" changed: {source_targets.get(mesh_name)} -> {exported_count}'
        )


def test_overwrite_mapping_references_valid_meshes(overwritten_head_dna):
    """Every mesh -> blend-shape channel mapping must reference a valid mesh and a
    valid channel index in the exported DNA."""
    _, exported_path = overwritten_head_dna
    reader = _open_reader(exported_path)
    mesh_count = reader.getMeshCount()
    channel_count = reader.getBlendShapeChannelCount()

    for index in range(reader.getMeshBlendShapeChannelMappingCount()):
        mapping = reader.getMeshBlendShapeChannelMapping(index)
        assert 0 <= mapping.meshIndex < mesh_count, f"mapping {index} references invalid mesh {mapping.meshIndex}"
        assert 0 <= mapping.blendShapeChannelIndex < channel_count, (
            f"mapping {index} references invalid channel {mapping.blendShapeChannelIndex}"
        )


def test_overwrite_deleted_mesh_is_dropped(mesh_deleted_head_dna):
    """Deleting a secondary mesh object drops exactly that mesh from the DNA."""
    source_path, exported_path, deleted_mesh_name = mesh_deleted_head_dna
    source_reader = _open_reader(source_path)
    exported_reader = _open_reader(exported_path)

    exported_meshes = {exported_reader.getMeshName(i) for i in range(exported_reader.getMeshCount())}
    assert deleted_mesh_name not in exported_meshes
    # LOD0 export loses exactly the one deleted mesh from the source's LOD0 set.
    assert exported_meshes == _lod0_mesh_names(source_reader) - {deleted_mesh_name}
    assert exported_reader.getMeshCount() == len(_lod0_mesh_names(source_reader)) - 1


def test_overwrite_mapping_valid_after_mesh_delete(mesh_deleted_head_dna):
    """After a mesh is deleted (and remaining meshes renumbered), the mesh ->
    blend-shape channel mapping must still reference only valid mesh indices."""
    _, exported_path, _ = mesh_deleted_head_dna
    reader = _open_reader(exported_path)
    mesh_count = reader.getMeshCount()
    channel_count = reader.getBlendShapeChannelCount()

    for index in range(reader.getMeshBlendShapeChannelMappingCount()):
        mapping = reader.getMeshBlendShapeChannelMapping(index)
        assert 0 <= mapping.meshIndex < mesh_count, f"mapping {index} references invalid mesh {mapping.meshIndex}"
        assert 0 <= mapping.blendShapeChannelIndex < channel_count, (
            f"mapping {index} references invalid channel {mapping.blendShapeChannelIndex}"
        )
