"""Permanent guard for ``RigInstance.update_head_shape_keys``.

This locks the runtime shape-key application contract independently of how it is
implemented: after RigLogic ``calculate()``, every cached shape-key block must
carry the value of its blend-shape channel from ``getBlendShapeOutputs()``, and
applying those values must actually refresh the evaluated head mesh geometry.

The interactive ``character_dna.import_shape_keys`` operator is modal and its
timer loop never runs to completion under the headless ``bpy`` module, so it
creates no shape keys. The blocks are therefore built here by draining the same
command queue the operator builds (``build_shape_key_import_commands``), then
rebuilding the rig-instance cache exactly like the operator's ``finish()`` does.

The controls are driven directly with deterministic seeded arrays (no bone poses
are read and no constraint / eye-aim feedback is triggered), so the assertions
are fully deterministic across processes.
"""

import queue

import numpy as np
import pytest

from constants import RUNNING_CI, TEST_DNA_FOLDER
from fixtures.scene import load_dna


# Shape-key creation lives in the Pro ``editors`` submodule; skip cleanly on a
# Free checkout where it is absent (the runtime path under test is core code).
shape_key_utilities = pytest.importorskip("character_dna.editors.shape_key_editor.utilities")

# Building every blend-shape target as a full-vertex Blender shape-key block is the
# heaviest single allocation in the suite. Skip it on CI where the runners (notably
# the 7 GB Linux Blender 4.5 box) run out of memory; the runtime contract is still
# verified on local runs.
pytestmark = pytest.mark.skipif(
    RUNNING_CI,
    reason="Shape-key import builds a full-vertex block per blend-shape target — too memory-heavy for CI runners.",
)


def _seeded_array(length: int, seed: int, amplitude: float) -> np.ndarray:
    """Deterministic float32 array in ``[-amplitude, amplitude]`` from a seed."""
    idx = np.arange(length, dtype=np.float64)
    raw = np.sin(idx * 12.9898 + seed * 78.233) * 43758.5453
    frac = raw - np.floor(raw)  # [0, 1)
    return ((frac - 0.5) * 2.0 * amplitude).astype(np.float32)


def _create_head_shape_keys() -> None:
    """Build every head shape-key block synchronously (the modal operator does
    nothing headless), then rebuild the rig-instance cache like ``finish()``."""
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
    # Rebuild the shape-key block cache, mirroring the import operator's finish().
    instance.data.clear()
    instance.initialize()


def _drive_head(instance, seed: int, amplitude: float) -> None:
    """Push deterministic GUI controls, map to raw controls, and calculate so the
    head blend-shape outputs are non-trivial and reproducible."""
    head_instance = instance.head_instance
    gui_count = instance.head_dna_reader.getGUIControlCount()
    for control_index, value in enumerate(_seeded_array(gui_count, seed, amplitude)):
        head_instance.setGUIControl(control_index, float(value))
    head_instance.setLOD(level=0)
    instance.head_manager.mapGUIToRawControls(head_instance)
    instance.head_manager.calculate(head_instance)


def _zero_head(instance) -> None:
    """Drive every GUI control to zero so all blend-shape outputs return to rest."""
    head_instance = instance.head_instance
    gui_count = instance.head_dna_reader.getGUIControlCount()
    for control_index in range(gui_count):
        head_instance.setGUIControl(control_index, 0.0)
    head_instance.setLOD(level=0)
    instance.head_manager.mapGUIToRawControls(head_instance)
    instance.head_manager.calculate(head_instance)


def _evaluated_vertices(mesh_object) -> np.ndarray:
    """Return the evaluated (deformed) vertex coordinates of ``mesh_object`` as a
    flat float32 array, forcing a dependency-graph re-evaluation first."""
    import bpy

    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated_mesh = mesh_object.evaluated_get(depsgraph).data
    coordinates = np.empty(len(evaluated_mesh.vertices) * 3, dtype=np.float32)
    evaluated_mesh.vertices.foreach_get("co", coordinates)
    return coordinates


@pytest.fixture(scope="module")
def head_shape_key_rig(addon):
    """Load Ada's head with output drivers released for explicit authoring checks."""
    from character_dna.runtime.controller import authoring_operation
    from character_dna.utilities import get_active_rig_instance

    load_dna(
        file_path=TEST_DNA_FOLDER / "ada" / "head.dna",
        import_lods=["lod0"],
        import_shape_keys=False,
        import_face_board=True,
        include_body=True,
    )
    _create_head_shape_keys()

    instance = get_active_rig_instance()
    assert instance is not None
    if not instance.head_initialized:
        instance.head_initialize()

    with authoring_operation(instance):
        yield instance


def test_update_head_shape_keys_writes_blend_shape_outputs(head_shape_key_rig):
    """Every cached block must carry its channel's ``getBlendShapeOutputs()`` value."""
    instance = head_shape_key_rig
    _drive_head(instance, seed=202, amplitude=0.5)

    instance.update_head_shape_keys()

    outputs = np.asarray(instance.head_instance.getBlendShapeOutputs(), dtype=np.float32)
    blocks_map = instance.head_shape_key_blocks
    assert blocks_map, "expected cached shape-key blocks after import"

    # Guard against a vacuous pass: the drive must actually move some channels.
    driven_channels = [
        channel for channel in blocks_map if channel < len(outputs) and abs(float(outputs[channel])) > 1e-4
    ]
    assert driven_channels, "seeded drive produced no non-zero blend-shape outputs"

    mismatches = []
    for channel_index, blocks in blocks_map.items():
        if channel_index >= len(outputs):
            continue  # channel outside the active LOD output range
        expected = float(outputs[channel_index])
        mismatches.extend(
            (block.name, float(block.value), expected) for block in blocks if abs(float(block.value) - expected) > 1e-5
        )

    assert not mismatches, f"shape-key value mismatches (first 10): {mismatches[:10]}"


def test_update_head_shape_keys_refreshes_evaluated_mesh(head_shape_key_rig):
    """Applying shape-key values must change the evaluated head mesh geometry.

    Bones are never driven between the two captures, so the only difference in the
    evaluated geometry is the shape-key contribution. This catches a bulk-write
    implementation that updates ``.value`` without tagging the mesh for refresh."""
    instance = head_shape_key_rig
    mesh_object = instance.head_mesh
    assert mesh_object is not None
    assert mesh_object.data.shape_keys is not None, "head mesh has no shape keys"

    # Rest: all channels zero -> shape keys at basis.
    _zero_head(instance)
    instance.update_head_shape_keys()
    rest_coordinates = _evaluated_vertices(mesh_object)

    # Driven: a strong expression -> non-trivial shape-key deltas.
    _drive_head(instance, seed=303, amplitude=0.8)
    instance.update_head_shape_keys()
    driven_coordinates = _evaluated_vertices(mesh_object)

    max_displacement = float(np.max(np.abs(driven_coordinates - rest_coordinates)))
    assert max_displacement > 1e-4, (
        f"evaluated head mesh did not change after applying shape keys (max displacement {max_displacement})"
    )


def test_zero_head_shape_keys_clears_driven_blocks(head_shape_key_rig):
    """``zero_head_shape_keys`` must reset every RigLogic-driven block to 0.0.

    The Raw Control Editor calls this on edit-mode entry so editing works on the
    basis mesh (bone transforms only, no blend-shape contribution)."""
    instance = head_shape_key_rig

    # Drive a strong expression and push the values so blocks are non-zero.
    _drive_head(instance, seed=404, amplitude=0.8)
    instance.update_head_shape_keys()

    outputs = np.asarray(instance.head_instance.getBlendShapeOutputs(), dtype=np.float32)
    blocks_map = instance.head_shape_key_blocks
    assert blocks_map, "expected cached shape-key blocks after import"
    driven_channels = [
        channel for channel in blocks_map if channel < len(outputs) and abs(float(outputs[channel])) > 1e-4
    ]
    assert driven_channels, "seeded drive produced no non-zero blend-shape outputs"
    # Guard against a vacuous pass: blocks really are non-zero before zeroing.
    assert any(abs(float(block.value)) > 1e-4 for channel in driven_channels for block in blocks_map[channel])

    instance.zero_head_shape_keys()

    nonzero = [
        (block.name, float(block.value))
        for blocks in blocks_map.values()
        for block in blocks
        if abs(float(block.value)) > 1e-6
    ]
    assert not nonzero, f"expected all driven shape-key blocks zeroed, got (first 10): {nonzero[:10]}"
