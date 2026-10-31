"""End-to-end coverage for shape-key editing through the ``overwrite``
DNA export path.

Unlike the source-copy case in ``test_head_overwrite_exporter.py`` (where the
scene carries no shape keys), this exercises the scene-delta path: every
head shape-key block is built synchronously (the modal import operator does
nothing headless), one channel's block is edited by a known offset, the overwrite
export is run, and the exported DNA is re-read to confirm the edited delta landed
on the correct blend-shape target and the channel survives in the mesh ->
blend-shape channel mapping.

Building a full-vertex Blender shape-key block per blend-shape target is the
heaviest allocation in the suite, so this is skipped on CI (mirroring
``test_head_shape_keys.py``).
"""

import math
import queue

import pytest

from mathutils import Matrix, Vector

from constants import RUNNING_CI, TEST_DNA_FOLDER
from fixtures.scene import load_dna


# Shape-key creation lives in the Pro ``editors`` submodule; skip cleanly on a
# Free checkout where it is absent.
shape_key_utilities = pytest.importorskip("character_dna.editors.shape_key_editor.utilities")

pytestmark = pytest.mark.skipif(
    RUNNING_CI,
    reason="Shape-key import builds a full-vertex block per blend-shape target — too memory-heavy for CI runners.",
)

# A known Blender-space edit applied to a single vertex of one shape-key block.
# X is unchanged by the DNA Y-up rotation, so the expected DNA delta is trivial to
# derive while still exceeding SHAPE_KEY_DELTA_THRESHOLD (1e-6 m).
EDIT_VERTEX_INDEX = 0
EDIT_OFFSET = Vector((0.01, 0.0, 0.0))


def _create_head_shape_keys() -> None:
    """Build every head shape-key block synchronously, then rebuild the
    rig-instance cache like the import operator's ``finish()``."""
    from character_dna.utilities import get_active_head, get_active_rig_instance

    head = get_active_head()
    assert head is not None, "no active head component after import"

    commands_queue: queue.Queue = queue.Queue()
    shape_key_utilities.build_shape_key_import_commands(head, commands_queue)
    while not commands_queue.empty():
        index, mesh_index, _description, kwargs_callback, callback = commands_queue.get()
        callback(**kwargs_callback(index, mesh_index))

    instance = get_active_rig_instance()
    assert instance is not None
    instance.data.clear()
    instance.initialize()


def _open_reader(dna_file_path):
    from character_dna.dna_io import get_dna_reader

    return get_dna_reader(file_path=dna_file_path, file_format="binary", data_layer="All")


@pytest.fixture(scope="module")
def edited_shape_key_export(addon, temp_folder):
    """Import Ada's head (LOD0) with every shape-key block created, edit one head
    channel's block by ``EDIT_OFFSET``, run the overwrite export, and yield the
    facts needed to verify the exported blend-shape target."""
    from character_dna.dna_io import DNAExporter
    from character_dna.utilities import get_active_head

    load_dna(
        file_path=TEST_DNA_FOLDER / "ada" / "head.dna",
        import_lods=["lod0"],
        import_shape_keys=False,
        import_face_board=True,
        include_body=True,
    )
    _create_head_shape_keys()

    head = get_active_head()
    assert head is not None and head.rig_instance is not None
    reader = head.rig_instance.head_dna_reader
    mesh_object = head.rig_instance.head_mesh
    assert mesh_object is not None and mesh_object.data.shape_keys is not None
    mesh_name = mesh_object.name.replace(f"{head.rig_instance.name}_", "")

    # The head mesh is index 0. Find a blend-shape target whose scene block exists.
    head_mesh_index = 0
    edited_channel_index = None
    for target_index in range(reader.getBlendShapeTargetCount(head_mesh_index)):
        channel_index = reader.getBlendShapeChannelIndex(head_mesh_index, target_index)
        channel_name = reader.getBlendShapeChannelName(channel_index)
        block = mesh_object.data.shape_keys.key_blocks.get(f"{mesh_name}__{channel_name}")
        if block is not None:
            edited_channel_index = channel_index
            edited_block = block
            break

    assert edited_channel_index is not None, "no editable head shape-key block found"

    basis = mesh_object.data.shape_keys.key_blocks.get("Basis")
    assert basis is not None
    # Override the vertex to a known delta from basis.
    edited_block.data[EDIT_VERTEX_INDEX].co = basis.data[EDIT_VERTEX_INDEX].co + EDIT_OFFSET

    export_folder = temp_folder / "overwrite_shape_keys" / "ada"
    export_folder.mkdir(parents=True, exist_ok=True)
    head.rig_instance.output.folder_path = str(export_folder)
    head.rig_instance.output.method = "overwrite"
    linear_modifier = head.linear_modifier
    DNAExporter(
        file_name="head.dna",
        instance=head.rig_instance,
        linear_modifier=linear_modifier,
    ).run()

    # DNA is Y-up, Blender is Z-up: the exporter rotates deltas by -90° about X.
    rotation_matrix = Matrix.Rotation(math.radians(-90), 4, "X")
    expected_delta = (rotation_matrix @ EDIT_OFFSET) / linear_modifier

    return {
        "exported_path": export_folder / "head.dna",
        "mesh_index": head_mesh_index,
        "channel_index": edited_channel_index,
        "expected_delta": expected_delta,
    }


def test_edited_shape_key_delta_written_to_target(edited_shape_key_export):
    """The edited vertex's delta must appear on the matching blend-shape target in
    the exported DNA."""
    facts = edited_shape_key_export
    reader = _open_reader(facts["exported_path"])
    mesh_index = facts["mesh_index"]

    target_index = next(
        (
            index
            for index in range(reader.getBlendShapeTargetCount(mesh_index))
            if reader.getBlendShapeChannelIndex(mesh_index, index) == facts["channel_index"]
        ),
        None,
    )
    assert target_index is not None, "edited channel has no blend-shape target in exported DNA"

    vertex_indices = list(reader.getBlendShapeTargetVertexIndices(mesh_index, target_index))
    assert EDIT_VERTEX_INDEX in vertex_indices, "edited vertex missing from exported blend-shape target"

    position = vertex_indices.index(EDIT_VERTEX_INDEX)
    delta_x = reader.getBlendShapeTargetDeltaXs(mesh_index, target_index)[position]
    delta_y = reader.getBlendShapeTargetDeltaYs(mesh_index, target_index)[position]
    delta_z = reader.getBlendShapeTargetDeltaZs(mesh_index, target_index)[position]

    expected = facts["expected_delta"]
    assert delta_x == pytest.approx(expected.x, abs=1e-4)
    assert delta_y == pytest.approx(expected.y, abs=1e-4)
    assert delta_z == pytest.approx(expected.z, abs=1e-4)


def test_edited_channel_survives_in_mapping(edited_shape_key_export):
    """The edited channel must still be wired to its mesh in the exported mesh ->
    blend-shape channel mapping."""
    facts = edited_shape_key_export
    reader = _open_reader(facts["exported_path"])

    mapped = {
        (
            reader.getMeshBlendShapeChannelMapping(index).meshIndex,
            reader.getMeshBlendShapeChannelMapping(index).blendShapeChannelIndex,
        )
        for index in range(reader.getMeshBlendShapeChannelMappingCount())
    }
    assert (facts["mesh_index"], facts["channel_index"]) in mapped
