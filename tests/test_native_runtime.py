"""Native backend integration with the real add-on and head-only fixture."""

from __future__ import annotations

from array import array
from types import SimpleNamespace

import bpy
import pytest

from character_dna.runtime import controller, engine
from character_dna.utilities import get_active_rig_instance, get_addon_preferences


@pytest.fixture
def native_head(load_head_only_dna):
    """Opt in through the same operator used by the preferences UI."""
    from character_dna.bindings import load_native_runtime

    try:
        load_native_runtime()
    except ModuleNotFoundError:
        pytest.skip("Native platform binding is not installed")
    if bpy.app.version[:2] != (5, 2):
        pytest.skip("Native backend is gated to Blender 5.2")
    instance = get_active_rig_instance()
    preferences = get_addon_preferences()
    old_value = preferences.experimental_native_riglogic
    preferences.experimental_native_riglogic = True
    assert bpy.ops.character_dna.sync_native_runtime() == {"FINISHED"}
    assert engine.active(instance), controller.status()
    try:
        yield instance
    finally:
        preferences.experimental_native_riglogic = False
        bpy.ops.character_dna.sync_native_runtime()
        preferences.experimental_native_riglogic = old_value


def test_native_head_only(native_head):
    """Head-only rigs retain their driver inputs and update without a body."""
    assert native_head.body_rig is None
    head = native_head.head_rig
    native_head.face_board.pose.bones["CTRL_C_jaw"].location.y = 0.0
    native_head.face_board.update_tag()
    bpy.context.view_layer.update()
    before = head.evaluated_get(bpy.context.evaluated_depsgraph_get()).pose.bones["FACIAL_C_Jaw"].matrix.copy()
    native_head.face_board.pose.bones["CTRL_C_jaw"].location.y = 0.8
    native_head.face_board.update_tag()
    bpy.context.view_layer.update()
    after = head.evaluated_get(bpy.context.evaluated_depsgraph_get()).pose.bones["FACIAL_C_Jaw"].matrix.copy()
    assert before != after
    assert engine.status(native_head) == "Native"
    reader = native_head.head_dna_reader
    channel = next(
        index
        for index in range(reader.getRawControlCount())
        if reader.getRawControlName(index) == "CTRL_expressions.jawOpen"
    )
    assert native_head.head_instance.getRawControl(channel) > 0.5


def test_native_without_face_board(native_head):
    """No-face-board bindings can still consume constrained/raw joint inputs."""
    engine.discard(native_head)
    native_head.face_board = None
    native_head.evaluate()
    engine.install(native_head)
    native_head.head_rig.pose.bones["head"].rotation_quaternion = (0.9238795, 0, 0, 0.3826834)
    native_head.head_rig.update_tag()
    bpy.context.view_layer.update()
    assert engine.active(native_head)
    assert engine.status(native_head) == "Native"


@pytest.fixture
def pro_editors():
    """The native runtime remains testable in Free-only checkouts."""
    pytest.importorskip("character_dna.editors.raw_control_editor.callbacks")


@pytest.mark.usefixtures("pro_editors")
def test_native_raw_slider_preview(native_head):
    """A manual raw preview must survive evaluation of the native output drivers."""
    from character_dna.editors.raw_control_editor.callbacks import (
        get_raw_control_item_value,
        set_raw_control_item_value,
    )

    native_head.face_board.pose.bones["CTRL_C_jaw"].location.y = 0.0
    native_head.face_board.update_tag()
    bpy.context.view_layer.update()
    reader = native_head.head_dna_reader
    index = next(
        index
        for index in range(reader.getRawControlCount())
        if reader.getRawControlName(index) == "CTRL_expressions.jawOpen"
    )
    item = SimpleNamespace(name="CTRL_expressions.jawOpen", raw_control_index=index)
    rig = native_head.head_rig.evaluated_get(bpy.context.view_layer.depsgraph)
    before = rig.pose.bones["FACIAL_C_Jaw"].matrix.copy()
    set_raw_control_item_value(item, 0.8)
    bpy.context.view_layer.update()
    rig = native_head.head_rig.evaluated_get(bpy.context.view_layer.depsgraph)
    after = rig.pose.bones["FACIAL_C_Jaw"].matrix.copy()
    assert get_raw_control_item_value(item) == pytest.approx(0.8)
    assert before != after
    assert engine.active(native_head)


@pytest.mark.usefixtures("pro_editors")
@pytest.mark.parametrize("release", ["frame", "face_board"])
def test_native_raw_preview_returns_to_live_inputs(native_head, release):
    """Successive slider edits persist, but playback and face-board posing take over."""
    from character_dna.editors.raw_control_editor.callbacks import update_head_raw_control_list

    update_head_raw_control_list(native_head)
    item = native_head.raw_control_editor.raw_controls["CTRL_expressions.jawOpen"]
    carriers = {carrier.as_pointer() for carrier in engine.carriers(native_head)}
    for value in (0.8, 0.3, 0.6):
        item.value = value
        bpy.context.view_layer.update()
        assert item.value == pytest.approx(value)
    if release == "frame":
        bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
        assert item.value == pytest.approx(0.0)
    else:
        native_head.face_board.pose.bones["CTRL_C_jaw"].location.y = 0.2
        native_head.face_board.update_tag()
        bpy.context.view_layer.update()
        assert item.value == pytest.approx(0.2)
    assert {carrier.as_pointer() for carrier in engine.carriers(native_head)} == carriers
    assert engine.active(native_head)


def _head_matrices(instance):
    rig = instance.head_rig.evaluated_get(bpy.context.view_layer.depsgraph)
    return [value for bone in rig.pose.bones for row in bone.matrix for value in row]


@pytest.mark.usefixtures("pro_editors")
@pytest.mark.parametrize("control", ["jawOpen", "jawOpenExtreme"])
def test_native_raw_activate_on_click_matches_legacy(native_head, control):
    """Selection, prerequisite chains and the neutral row retain their legacy poses."""
    from character_dna.editors.raw_control_editor.callbacks import update_head_raw_control_list

    update_head_raw_control_list(native_head)
    editor = native_head.raw_control_editor
    editor.activate_on_click = True
    index = editor.raw_controls.find(f"CTRL_expressions.{control}")
    editor.raw_controls_active_index = index
    bpy.context.view_layer.update()
    assert editor.raw_controls[index].value == pytest.approx(1.0)
    native_pose = _head_matrices(native_head)
    editor.activate_on_click = False
    editor.raw_controls_active_index = 0
    bpy.context.view_layer.update()
    assert editor.raw_controls[index].value == pytest.approx(1.0)
    editor.activate_on_click = True
    editor.raw_controls_active_index = 0
    bpy.context.view_layer.update()
    assert editor.raw_controls[index].value == pytest.approx(0.0)
    get_addon_preferences().experimental_native_riglogic = False
    bpy.ops.character_dna.sync_native_runtime()
    editor.raw_controls_active_index = index
    bpy.context.view_layer.update()
    assert native_pose == pytest.approx(_head_matrices(native_head), abs=1e-5)


@pytest.fixture
def native_shape(native_head, request):
    """Use a real DNA shape channel with a small deterministic vertex displacement."""
    from character_dna.utilities import remove_instance_prefix

    pytest.importorskip("character_dna.editors.shape_key_editor.callbacks")
    engine.discard(native_head)
    mesh = native_head.head_mesh
    if mesh.data.shape_keys is None:
        mesh.shape_key_add(name="Basis")
    channel = getattr(request, "param", "brow_down_L")
    name = f"{remove_instance_prefix(mesh.name, native_head.name)}__{channel}"
    block = mesh.shape_key_add(name=name, from_mix=False)
    block.data[0].co.x += 0.025
    native_head.data.clear()
    native_head.initialize()
    bpy.ops.character_dna.sync_native_runtime()
    bpy.context.view_layer.update()
    assert engine.active(native_head), controller.status()
    return native_head, name


@pytest.mark.parametrize("evaluate_shapes", [True, False])
def test_native_shape_slider_matches_legacy(native_shape, evaluate_shapes):
    """Shape sliders deform the evaluated mesh and do not change the raw-control pose."""
    instance, name = native_shape
    instance.evaluate_shape_keys = evaluate_shapes
    bpy.context.view_layer.update()
    item = instance.shape_key_editor.shape_key_list[name]
    before = _head_matrices(instance)
    for value in (0.8, 0.25):
        item.value = value
        bpy.context.view_layer.update()
        assert item.value == pytest.approx(value)
        assert _head_matrices(instance) == pytest.approx(before, abs=1e-5)
    native_vertices = [
        value
        for vertex in instance.head_mesh.evaluated_get(bpy.context.view_layer.depsgraph).data.vertices
        for value in vertex.co
    ]
    get_addon_preferences().experimental_native_riglogic = False
    bpy.ops.character_dna.sync_native_runtime()
    item.value = 0.25
    bpy.context.view_layer.update()
    legacy_vertices = [
        value
        for vertex in instance.head_mesh.evaluated_get(bpy.context.view_layer.depsgraph).data.vertices
        for value in vertex.co
    ]
    assert native_vertices == pytest.approx(legacy_vertices, abs=1e-5)


@pytest.mark.parametrize("native_shape", ["brow_down_L", "head_turnUp_U"], indirect=True)
def test_native_shape_activate_on_click_matches_legacy(native_shape):
    """Selecting a shape activates its DNA-derived raw pose in either backend."""
    instance, name = native_shape
    editor = instance.shape_key_editor
    index = editor.shape_key_list.find(name)
    editor.activate_on_click = True
    editor.shape_key_list_active_index = index
    bpy.context.view_layer.update()
    native_value = editor.shape_key_list[index].value
    assert native_value > 0.5
    native_pose = _head_matrices(instance)
    get_addon_preferences().experimental_native_riglogic = False
    bpy.ops.character_dna.sync_native_runtime()
    editor.shape_key_list_active_index = index
    bpy.context.view_layer.update()
    assert editor.shape_key_list[index].value == pytest.approx(native_value, abs=1e-5)
    assert native_pose == pytest.approx(_head_matrices(instance), abs=1e-5)


@pytest.mark.usefixtures("pro_editors")
def test_native_previews_are_instance_local(native_head):
    """A preview on the selected character cannot overwrite another native rig."""
    from character_dna.editors.raw_control_editor.callbacks import update_head_raw_control_list
    from constants import TEST_DNA_FOLDER

    name = native_head.name
    assert bpy.ops.character_dna.import_dna(
        filepath=str(TEST_DNA_FOLDER / "default" / "head.dna"),
        import_mesh=True,
        import_bones=True,
        import_shape_keys=False,
        import_vertex_groups=True,
        import_materials=False,
        import_face_board=True,
        include_body=False,
        **{f"import_lod{index}": index == 0 for index in range(8)},
    ) == {"FINISHED"}
    bpy.ops.character_dna.sync_native_runtime()
    properties = bpy.context.scene.character_dna
    instances = properties.rig_instance_list
    original = instances[name]
    duplicate = next(instance for instance in instances if instance.name != name)
    assert engine.active(original) and engine.active(duplicate), controller.status()
    assert engine.instance_id(original) != engine.instance_id(duplicate)
    for instance, value in ((original, 0.75), (duplicate, 0.25)):
        properties.rig_instance_list_active_index = instances.find(instance.name)
        update_head_raw_control_list(instance)
        instance.raw_control_editor.raw_controls["CTRL_expressions.jawOpen"].value = value
        bpy.context.view_layer.update()
    for instance, value in ((original, 0.75), (duplicate, 0.25)):
        properties.rig_instance_list_active_index = instances.find(instance.name)
        assert instance.raw_control_editor.raw_controls["CTRL_expressions.jawOpen"].value == pytest.approx(value)


def test_native_operator_has_undo():
    """Native scene mutations participate in Blender's operator undo stack."""
    assert "UNDO" in controller.CHARACTER_DNA_OT_sync_native_runtime.bl_options


def test_native_auto_evaluate_off_releases_channels(native_head):
    """Manual posing remains possible after disabling automatic rig evaluation."""
    native_head.auto_evaluate = False
    bpy.ops.character_dna.sync_native_runtime()
    assert not engine.active(native_head)
    bone = native_head.head_rig.pose.bones["FACIAL_C_Jaw"]
    bone.location.x = 0.125
    bpy.context.view_layer.update()
    assert bone.location.x == pytest.approx(0.125)


def test_native_frame_rejects_invalid_buffers(native_head):
    """Frame validation cannot mutate published output or solver state on bad buffers."""
    from character_dna.runtime.frame import buffers, capture

    record = next(record for record in engine._records.values() if record["component"] == "head")
    native = engine._module
    context = buffers(record)
    capture(record, context, bpy.context.evaluated_depsgraph_get())
    session = native.create_session(record["model"])
    sdk = array("d", [19.0]) * sum(record["info"]["output_counts"])
    local = array("d", [23.0]) * (record["joint_count"] * 9)
    arguments = [
        session,
        record["frame_plan"],
        context["poses"],
        context["bases"],
        context["face"],
        context["spatial"],
        sdk,
        local,
        0,
        True,
        False,
    ]
    invalid_values = [
        (2, array("d", context["poses"])),
        (3, array("f")),
        (6, memoryview(sdk).toreadonly()),
        (7, memoryview(local)[::2]),
        (8, -1),
        (8, record["info"]["lod_count"]),
    ]
    nonfinite = array("f", context["poses"])
    nonfinite[0] = float("nan")
    invalid_values.append((2, nonfinite))
    for index, value in invalid_values:
        request = list(arguments)
        request[index] = value
        with pytest.raises((ValueError, BufferError)):
            native.evaluate_frame(*request)
        assert all(value == 19.0 for value in sdk)
        assert all(value == 23.0 for value in local)
        assert native.session_statistics(session)["solves"] == 0
    native.evaluate_frame(*arguments)
    first = (sdk.tobytes(), local.tobytes())
    native.evaluate_frame(*arguments)
    assert first == (sdk.tobytes(), local.tobytes())
    assert native.session_statistics(session) == {"solves": 1, "cache_hits": 1}


def test_native_frame_plan_rejects_invalid_indices(native_head):
    """Reject out-of-range controls and cyclic eye chains before publishing a plan."""
    record = next(record for record in engine._records.values() if record["component"] == "head")
    native = engine._module
    session = native.create_session(record["model"])
    rest = array("f", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    args = [session, array("f"), rest, array("i", [-1]), array("i"), array("i"), array("i"), array("i"), 0, False]
    for index, value in (
        (5, array("i", [0, 999, 0])),
        (6, array("i", [0, 0, -1, 0])),
        (4, array("i", [99999, -1, 0, -1])),
        (7, array("i", [99999, 0, 0, 1])),
    ):
        request = list(args)
        request[index] = value
        with pytest.raises(ValueError):
            native.create_frame_plan(*request)


def test_native_eye_cache_includes_spatial_inputs(native_head):
    """Skip repeated native eye solves, but invalidate when only the aim target moves."""
    from character_dna.runtime.frame import buffers, capture

    record = next(record for record in engine._records.values() if record["component"] == "head")
    native = engine._module
    native_head.face_board.pose.bones["CTRL_lookAtSwitch"].location.y = 1.0
    native_head.face_board.update_tag()
    bpy.context.view_layer.update()
    context = buffers(record)
    assert capture(record, context, bpy.context.evaluated_depsgraph_get())
    session = native.create_session(record["model"])
    sdk = array("d", [0.0]) * sum(record["info"]["output_counts"])
    local = array("d", [0.0]) * (record["joint_count"] * 9)
    args = [
        session,
        record["frame_plan"],
        context["poses"],
        context["bases"],
        context["face"],
        context["spatial"],
        sdk,
        local,
        0,
        True,
        True,
    ]
    native.evaluate_frame(*args)
    before = native.session_statistics(session)
    saved = sdk.tobytes()
    sdk[:] = array("d", [999.0]) * len(sdk)
    native.evaluate_frame(*args)
    assert sdk.tobytes() == saved
    assert native.session_statistics(session)["solves"] == before["solves"]
    assert native.session_statistics(session)["cache_hits"] == before["cache_hits"] + 1
    context["spatial"][32] += 0.005
    native.evaluate_frame(*args)
    assert native.session_statistics(session)["solves"] > before["solves"]
