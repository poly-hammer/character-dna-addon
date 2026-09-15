import hashlib

from pathlib import Path
from typing import TYPE_CHECKING, cast

import bpy
import pytest

from character_dna.runtime import engine
from character_dna.ui.callbacks import get_active_rig_instance
from character_dna.utilities.reference import validate_names
from constants import TEST_DNA_FOLDER


if TYPE_CHECKING:
    from typing import Any

    from mathutils import Matrix

    from character_dna.rig_instance import RigInstance


@pytest.mark.parametrize("selected", [["ada"], ["other", "ada"], ["other", "other"], [""]])
def test_reference_names_rejected_without_mutation(selected: list[str], monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    from character_dna import operators

    reports = []
    monkeypatch.setattr(
        operators.utilities,
        "get_addon_scene_properties",
        lambda _context: SimpleNamespace(rig_instance_list=[SimpleNamespace(name="ada")]),
    )
    monkeypatch.setattr(
        operators.utilities,
        "extract_rig_instance_data_from_blend_file",
        lambda *_args: pytest.fail("Name validation must precede metadata extraction"),
    )
    before = set(bpy.data.objects)
    operator = SimpleNamespace(
        filepath="not-loaded.blend",
        meta_human_names=",".join(selected),
        meta_human_list=[],
        report=lambda _level, message: reports.append(message),
    )
    assert operators.AppendOrLinkCharacter.execute(cast("Any", operator), cast("Any", bpy.context)) == {"CANCELLED"}
    assert set(bpy.data.objects) == before
    assert reports


def test_reference_names_valid_batch():
    assert validate_names([" ada ", "bruce"], ["other"]) == ["ada", "bruce"]


def test_face_widgets_reuse_matching_geometry_only():
    """Widget sources are shared by role and geometry without collapsing artist edits."""
    from types import SimpleNamespace

    from character_dna.utilities.reference import _share_face_widgets, _widget_objects

    before = set(bpy.data.user_map())
    try:
        mesh = bpy.data.meshes.new("SharedWidgetMesh")
        mesh.from_pydata([(0, 0, 0), (1, 0, 0)], [(0, 1)], [])
        original = bpy.data.objects.new("WidgetOriginal", mesh)
        duplicate = bpy.data.objects.new("WidgetCopy", mesh.copy())
        edited = bpy.data.objects.new("WidgetEdited", mesh.copy())
        assert isinstance(edited.data, bpy.types.Mesh)
        assert isinstance(duplicate.data, bpy.types.Mesh)
        edited.data.vertices[1].co.x = 2.0
        duplicate_name, duplicate_mesh = duplicate.name, duplicate.data.name
        parent = bpy.data.objects.new("WidgetGroup", None)
        duplicate.parent = parent

        def face(shape):
            return SimpleNamespace(
                library=None,
                override_library=None,
                pose=SimpleNamespace(
                    bones=[
                        SimpleNamespace(
                            name="CTRL_C_jaw",
                            custom_shape=shape,
                            path_from_id=lambda _field: 'pose.bones["CTRL_C_jaw"].custom_shape',
                        )
                    ]
                ),
            )

        previous, new, changed = face(original), face(duplicate), face(edited)
        assert _widget_objects(new, {duplicate, parent}) == {duplicate, parent}
        _share_face_widgets([new, changed], [previous], {duplicate, edited, parent})
        assert new.pose.bones[0].custom_shape == original
        assert changed.pose.bones[0].custom_shape == edited
        assert bpy.data.objects.get(duplicate_name) is None
        assert bpy.data.meshes.get(duplicate_mesh) is None
        assert not original.users_collection
    finally:
        bpy.data.batch_remove(ids=list(set(bpy.data.user_map()) - before))


@pytest.mark.parametrize("component", ["head", "body"])
@pytest.mark.parametrize("protection", ["LINK", "EDITABLE_LINK", "library", "system_override", "carrier"])
def test_reference_readonly_initialization(component: str, protection: str, monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    from character_dna.rig_instance import RigInstance

    def unexpected_write(*_args, **_kwargs):
        pytest.fail("Protected authoring initialization must reject before writes or teardown")

    rig = SimpleNamespace(
        library=object() if protection == "library" else None,
        override_library=SimpleNamespace(is_system_override=True) if protection == "system_override" else None,
    )
    instance = SimpleNamespace(
        name="Readonly",
        get=lambda key, default=None: protection if key == "reference_mode" else default,
        **{
            f"{component}_valid": True,
            f"{component}_rig": rig,
            f"destroy_{component}": unexpected_write,
            "_apply_head_rotation_modes": unexpected_write,
        },
    )
    carrier = SimpleNamespace(library=object(), override_library=None, get=lambda _key: component)
    monkeypatch.setattr(engine, "bound_carriers", lambda _instance: [carrier] if protection == "carrier" else [])
    monkeypatch.setattr(engine, "active", unexpected_write)

    with pytest.raises(RuntimeError, match=r"source.*Append"):
        getattr(RigInstance, f"{component}_initialize")(instance)


@pytest.mark.parametrize("mode", ["LINK", "EDITABLE_LINK"])
@pytest.mark.parametrize("adoption_fails", [False, True])
def test_reference_readonly_force_evaluate(mode: str, adoption_fails: bool, monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    from character_dna import operators

    calls = []
    reports = []
    instance = SimpleNamespace(get=lambda _key, _default=None: mode)
    monkeypatch.setattr(operators.callbacks, "get_active_rig_instance", lambda: instance)
    for name in ("teardown_scene", "setup_scene", "get_addon_window_manager_properties", "switch_to_pose_mode"):
        monkeypatch.setattr(operators.utilities, name, lambda *_args: pytest.fail("Reference refresh must not write"))
    monkeypatch.setattr(engine, "active", lambda _instance: pytest.fail("Discovery must not require active records"))
    monkeypatch.setattr(engine, "release_records", lambda target: calls.append(("release", target)))

    def adopt(target):
        calls.append(("adopt", target))
        if adoption_fails:
            raise ValueError("Saved bindings need source migration")

    monkeypatch.setattr(engine, "adopt", adopt)
    context = SimpleNamespace(view_layer=SimpleNamespace(update=lambda: calls.append(("update", instance))))
    operator = SimpleNamespace(report=lambda _level, message: reports.append(message))
    outcome = operators.ForceEvaluate.execute(cast("Any", operator), cast("Any", context))
    assert outcome == ({"CANCELLED"} if adoption_fails else {"FINISHED"})
    expected = ["release", "adopt"] if adoption_fails else ["release", "adopt", "update"]
    assert [name for name, _target in calls] == expected
    assert all(target is instance for _name, target in calls)
    assert bool(reports) == adoption_fails


@pytest.mark.parametrize("mode", ["LINK", "EDITABLE_LINK"])
def test_reference_readonly_reverse_mapping(mode: str, monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    from character_dna import operators

    reports = []
    instance = SimpleNamespace(
        get=lambda _key, _default=None: mode,
        face_board=SimpleNamespace(library=None, override_library=None),
        head_initialized=False,
        head_initialize=lambda: pytest.fail("Reference reverse mapping must reject before initialization"),
    )
    monkeypatch.setattr(operators.callbacks, "get_active_rig_instance", lambda: instance)
    operator = SimpleNamespace(report=lambda _level, message: reports.append(message))
    assert operators.MapRawToGuiControls.execute(cast("Any", operator), cast("Any", bpy.context)) == {"CANCELLED"}
    assert "Append" in reports[0]


@pytest.mark.parametrize("mode", ["LINK", "EDITABLE_LINK"])
def test_reference_readonly_metadata_callbacks(mode: str, monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    from character_dna import utilities
    from character_dna.ui import callbacks

    def unexpected_write(*_args, **_kwargs):
        pytest.fail("Reference callbacks must not mutate metadata or protected IDs")

    owner = SimpleNamespace(library=object(), override_library=None)
    instance = SimpleNamespace(
        get=lambda _key, _default=None: mode,
        head_rig=owner,
        body_rig=owner,
        face_board=owner,
        control_rig=owner,
    )
    view_options = {}
    monkeypatch.setattr(callbacks, "_get_view_options_owner", lambda _self: instance)
    monkeypatch.setattr(callbacks, "get_active_rig_instance", lambda: instance)
    monkeypatch.setattr(callbacks, "get_active_head", unexpected_write)
    monkeypatch.setattr(engine, "sync_settings", unexpected_write)
    monkeypatch.setattr(utilities, "set_hidden", unexpected_write)
    monkeypatch.setattr(utilities, "rename_rig_instance", unexpected_write)
    monkeypatch.setattr(
        utilities, "get_addon_scene_properties", lambda *_args: SimpleNamespace(rig_instance_list=[instance])
    )
    for setter in (
        callbacks.set_active_lod,
        callbacks.set_active_material_preview,
        callbacks.set_show_head_bones,
        callbacks.set_show_body_bones,
        callbacks.set_show_face_board,
        callbacks.set_show_control_rig,
        callbacks.set_hide_volume_bones,
        callbacks.set_solo_internal_bones,
    ):
        setter(cast("Any", view_options), True)
    assert view_options == {}
    for update in (
        callbacks.update_instance_name,
        callbacks.update_head_to_body_constraint_influence,
        callbacks.update_evaluate_rbfs_value,
        callbacks.update_head_output_items,
        callbacks.update_body_output_items,
    ):
        update(cast("Any", instance), cast("Any", bpy.context))


@pytest.mark.parametrize("mode", ["APPEND", "LINK", "EDITABLE_LINK"])
@pytest.mark.parametrize("owner_kind", ["local", "library", "system_override", "editable_override"])
def test_reference_readonly_face_switch(mode: str, owner_kind: str, monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    from character_dna.ui import callbacks

    calls = []
    bone = SimpleNamespace(location=SimpleNamespace(y=0.0))
    face = SimpleNamespace(
        library=object() if owner_kind == "library" else None,
        override_library=SimpleNamespace(is_system_override=owner_kind == "system_override")
        if owner_kind.endswith("override")
        else None,
        pose=SimpleNamespace(bones={"CTRL_lookAtSwitch": bone}),
    )
    instance = SimpleNamespace(
        get=lambda _key, _default=None: mode, face_board=face, evaluate=lambda: calls.append("evaluate")
    )
    monkeypatch.setattr(callbacks, "get_active_rig_instance", lambda: instance)
    callbacks.set_use_eye_aim(cast("Any", None), True)
    writable = mode != "LINK" and owner_kind in {"local", "editable_override"}
    assert bone.location.y == float(writable)
    assert calls == (["evaluate"] if writable else [])


@pytest.mark.parametrize("mode", ["LINK", "EDITABLE_LINK"])
@pytest.mark.parametrize("previous_evaluation", [False, True])
def test_reference_readonly_face_preset(mode: str, previous_evaluation: bool, monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    from character_dna import utilities
    from character_dna.constants import POSES_FOLDER
    from character_dna.ui import callbacks

    initial = set(bpy.data.user_map())
    calls = []
    face = bpy.data.objects.new("Readonly_face_controls", bpy.data.armatures.new("Readonly_face_controls"))
    bpy.context.scene.collection.objects.link(face)
    previous_context = utilities.get_current_context()
    try:
        utilities.switch_to_bone_edit_mode(face)
        assert isinstance(face.data, bpy.types.Armature)
        for name in ("CTRL_C_jaw", "CTRL_lookAtSwitch"):
            bone = face.data.edit_bones.new(name)
            bone.head = (0, 0, 0)
            bone.tail = (0, 0, 1)
        utilities.switch_to_object_mode()
        assert face.pose is not None
        jaw = face.pose.bones["CTRL_C_jaw"]
        jaw.location.y = 0.7
        switch = face.pose.bones["CTRL_lookAtSwitch"]
        switch.location.y = 1.0
        head = bpy.data.objects.new("Readonly_head", None)
        body = bpy.data.objects.new("Readonly_body", None)
        head.location.x = 2.0
        body.location.x = 3.0
        instance = SimpleNamespace(
            get=lambda _key, _default=None: mode,
            face_board=face,
            head_rig=head,
            body_rig=body,
            evaluate=lambda: calls.append("evaluate"),
        )
        settings = SimpleNamespace(evaluate_dependency_graph=previous_evaluation)
        monkeypatch.setattr(callbacks, "get_active_rig_instance", lambda: instance)
        monkeypatch.setattr(utilities, "get_addon_window_manager_properties", lambda: settings)
        monkeypatch.setattr(
            utilities, "get_addon_scene_properties", lambda: SimpleNamespace(rig_instance_list=[instance])
        )
        monkeypatch.setattr(utilities, "get_head", lambda *_args: pytest.fail("Must not reset protected head poses"))
        monkeypatch.setattr(utilities, "get_body", lambda *_args: pytest.fail("Must not reset protected body poses"))
        monkeypatch.setattr(utilities, "switch_to_pose_mode", lambda *owners: calls.append(tuple(owners)))
        preview = POSES_FOLDER / "face" / "scan_reference" / "Neutral" / "thumbnail-preview.png"
        properties = SimpleNamespace(face_pose_previews=str(preview))
        context = SimpleNamespace(selected_objects=[head, body])
        callbacks.update_face_pose(cast("Any", properties), cast("Any", context))
        assert jaw.location.y == pytest.approx(0.7 if mode == "LINK" else 0.0)
        assert switch.location.y == 1.0
        assert head.location.x == 2.0 and body.location.x == 3.0
        assert settings.evaluate_dependency_graph == previous_evaluation
        assert ("evaluate" in calls) == (mode == "EDITABLE_LINK")
    finally:
        utilities.switch_to_object_mode()
        bpy.data.batch_remove(ids=list(set(bpy.data.user_map()) - initial))
        utilities.set_context(previous_context)


class _ReadonlyUILayout:
    def __init__(self, parent=None):
        self.parent = parent
        self.enabled = True
        self.drawn = parent.drawn if parent else []

    def row(self, *_args, **_kwargs):
        return _ReadonlyUILayout(self)

    column = box = grid_flow = split = row

    def prop(self, _owner, name, **_kwargs):
        self.drawn.append((name, self))

    def operator(self, name, **_kwargs):
        self.drawn.append((name, self))
        return self

    def label(self, **_kwargs):
        pass

    def separator(self, **_kwargs):
        pass

    def popover(self, **_kwargs):
        pass

    template_icon_view = prop

    def writable(self, name):
        for identifier, layout in self.drawn:
            if identifier != name:
                continue
            current = layout
            while current:
                if not current.enabled:
                    return False
                current = current.parent
            return True
        pytest.fail(f"Control not drawn: {name}")


@pytest.mark.parametrize("mode", ["APPEND", "LINK", "EDITABLE_LINK"])
def test_reference_readonly_panel_states(mode: str, monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    from character_dna.ui import view_3d

    owner = SimpleNamespace(library=None, override_library=None, hide_get=lambda: False)
    view_options = SimpleNamespace(show_head_bones=True)
    instance = SimpleNamespace(
        get=lambda _key, _default=None: mode,
        head_rig=owner,
        body_rig=owner,
        face_board=owner,
        control_rig=owner,
        head_material=owner,
        head_mesh=owner,
        view_options=view_options,
        auto_evaluate=True,
        auto_evaluate_head=True,
        auto_evaluate_body=True,
        evaluate_bones=True,
        evaluate_shape_keys=True,
        evaluate_texture_masks=True,
        evaluate_rbfs=True,
        head_dna_file_path="",
        body_dna_file_path="",
        is_pro=False,
    )
    properties = SimpleNamespace(rig_instance_list=[instance], rig_instance_list_active_index=0, face_board=object())
    context = SimpleNamespace(scene=SimpleNamespace(character_dna=properties))
    monkeypatch.setattr(view_3d, "get_active_rig_instance", lambda: instance)
    monkeypatch.setattr(view_3d, "valid_rig_instance_exists", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(engine, "carriers", lambda _instance: [])
    layout = _ReadonlyUILayout()
    view_3d.CHARACTER_DNA_UL_rig_instances.draw_item(
        cast("Any", None),
        cast("Any", context),
        cast("Any", layout),
        cast("Any", properties),
        cast("Any", instance),
        0,
        cast("Any", properties),
        "active_index",
    )
    for name in (
        "name",
        "auto_evaluate",
        "evaluate_bones",
        "evaluate_shape_keys",
        "evaluate_texture_masks",
        "evaluate_rbfs",
    ):
        assert layout.writable(name) == (mode == "APPEND")
    for panel, names in (
        (
            view_3d.CHARACTER_DNA_PT_rig_instance_head_sub_panel,
            ("auto_evaluate_head", "head_dna_file_path", "head_rig", "head_mesh", "head_material", "face_board"),
        ),
        (
            view_3d.CHARACTER_DNA_PT_rig_instance_body_sub_panel,
            ("auto_evaluate_body", "body_dna_file_path", "body_rig", "body_mesh", "body_material", "control_rig"),
        ),
        (
            view_3d.CHARACTER_DNA_PT_view_options,
            (
                "active_lod",
                "active_material_preview",
                "show_head_bones",
                "show_body_bones",
                "hide_volume_bones",
                "solo_internal_bones",
            ),
        ),
        (view_3d.CHARACTER_DNA_PT_rig_instance_footer_sub_panel, ("head_to_body_constraint_influence",)),
    ):
        layout = _ReadonlyUILayout()
        panel_self = cast("Any", SimpleNamespace(layout=layout))
        if hasattr(panel, "draw_header"):
            panel.draw_header(panel_self, cast("Any", context))
        panel.draw(panel_self, cast("Any", context))
        for name in names:
            assert layout.writable(name) == (mode == "APPEND"), (panel.__name__, name, mode)
    assert layout.writable("character_dna.force_evaluate")
    layout = _ReadonlyUILayout()
    view_3d.CHARACTER_DNA_PT_face_board.draw(cast("Any", SimpleNamespace(layout=layout)), cast("Any", context))
    for name in ("category", "face_pose_previews", "use_eye_aim", "eyes_follow_head", "face_board_follow_head"):
        assert layout.writable(name) == (mode != "LINK")


@pytest.mark.parametrize("operation,editable", [("APPEND", False), ("LINK", False), ("LINK", True)])
@pytest.mark.parametrize(
    "descriptor",
    [
        {"schema_version": 1},
        {"schema_version": 2, "issues": ["Ada_head_native: obsolete runtime", "Ada_body_native: scene dependency"]},
        {"schema_version": 2, "carriers": [{"schema_version": 1}]},
    ],
)
def test_reference_legacy_report_is_concise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    editable: bool,
    descriptor: dict,
    caplog: pytest.LogCaptureFixture,
):
    from types import SimpleNamespace

    from character_dna import operators
    from character_dna.utilities import reference

    source = tmp_path / "legacy.blend"
    source.touch()
    reports = []
    monkeypatch.setattr(reference, "scene_names", lambda _scene: set())
    monkeypatch.setattr(
        operators.utilities, "get_addon_scene_properties", lambda _context: SimpleNamespace(rig_instance_list=[])
    )
    monkeypatch.setattr(
        operators.utilities, "extract_rig_instance_data_from_blend_file", lambda _path: ({"Ada": descriptor}, "")
    )
    monkeypatch.setattr(
        reference, "import_characters", lambda *_args: pytest.fail("Legacy validation must prevent import")
    )
    operator = SimpleNamespace(
        filepath=str(source),
        meta_human_names="Ada",
        operation_type=operation,
        editable_rig=editable,
        relative_path=False,
        report=lambda level, message: reports.append((level, message)),
    )
    assert operators.AppendOrLinkCharacter.execute(cast("Any", operator), cast("Any", bpy.context)) == {"CANCELLED"}
    assert all(record.levelname == "WARNING" and record.exc_info is None for record in caplog.records)
    assert reports == [
        (
            {"WARNING"},
            (
                "Legacy data detected. Please open the source file, run Migrate Legacy Data, "
                "save then retry appending/linking"
            ),
        )
    ]


def test_reference_saved_descriptor_rejects_legacy(setup_reference_blend_file: Path):
    from character_dna import utilities
    from character_dna.utilities import reference

    before = (setup_reference_blend_file.stat().st_mtime_ns, setup_reference_blend_file.stat().st_size)
    extracted, error = utilities.extract_rig_instance_data_from_blend_file(setup_reference_blend_file)
    assert not error, error
    assert isinstance(extracted, dict)
    data = cast("dict[str, Any]", extracted)
    descriptor = data["ada"]
    assert descriptor["pointers"]["face_board"] == "OBJECT::ada_face_gui"
    assert descriptor["pointers"]["control_rig"] == "OBJECT::ada_control_rig"
    assert len(descriptor["objects"]) == len({item["id"] for item in descriptor["objects"]})
    assert descriptor["collections"]["name"] == "ada"
    assert descriptor["carriers"]
    if any(carrier["schema_version"] != reference.SCHEMA_VERSION for carrier in descriptor["carriers"]):
        with pytest.raises(ValueError, match=reference.MIGRATION_MESSAGE):
            reference.validate_descriptors(data, ["ada"])
    else:
        assert reference.validate_descriptors(data, ["ada"]) == [descriptor]
    assert before == (setup_reference_blend_file.stat().st_mtime_ns, setup_reference_blend_file.stat().st_size)


@pytest.mark.parametrize("mode", ["APPEND", "LINK", "EDITABLE_LINK"])
def test_reference_rollback(tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch):
    from character_dna import utilities
    from character_dna.utilities import reference

    initial = set(bpy.data.user_map())
    root = bpy.data.collections.new("Rollback")
    bpy.context.scene.collection.children.link(root)
    root.objects.link(bpy.data.objects.new("Rollback_object", None))
    descriptor = reference.describe_instance(bpy.context.scene, "character_dna", {"name": root.name})
    source = tmp_path / "rollback.blend"
    bpy.data.libraries.write(str(source), {root})
    bpy.data.batch_remove(ids=list(set(bpy.data.user_map()) - initial))
    released = []
    notified = []
    monkeypatch.setattr(engine, "binding_issues", lambda _instance: [], raising=False)

    def fail_adoption(_instance) -> int:
        raise RuntimeError("deliberate adoption failure")

    monkeypatch.setattr(engine, "adopt", fail_adoption, raising=False)
    monkeypatch.setattr(engine, "release_records", lambda instance: released.append(instance.name), raising=False)
    monkeypatch.setattr(utilities, "notify_rig_instances_changed", lambda instance: notified.append(instance.name))
    properties = utilities.get_addon_scene_properties()
    names = [instance.name for instance in properties.rig_instance_list]
    active_index = properties.rig_instance_list_active_index
    before = set(bpy.data.user_map())
    with pytest.raises(RuntimeError, match="deliberate adoption failure"):
        reference.import_characters(bpy.context, str(source), [descriptor], mode, False)
    assert set(bpy.data.user_map()) == before
    assert [instance.name for instance in properties.rig_instance_list] == names
    assert properties.rig_instance_list_active_index == active_index
    assert released == ["Rollback"]
    assert notified == []


@pytest.mark.parametrize("mode", ["APPEND", "LINK", "EDITABLE_LINK"])
def test_reference_object_transport(tmp_path: Path, mode: str):  # noqa: PLR0915
    import subprocess
    import sys
    import textwrap

    from character_dna.utilities import reference

    # This subprocess tests transport independently of native rigs in prior fixtures.
    bpy.ops.wm.read_homefile(app_template="")
    before = set(bpy.data.user_map())
    try:
        root = bpy.data.collections.new("Transport")
        bpy.context.scene.collection.children.link(root)
        nested = bpy.data.collections.new("Transport_nested")
        root.children.link(nested)
        control = bpy.data.objects.new("Transport_authored_control", bpy.data.armatures.new("Transport_controls"))
        root.objects.link(control)
        control.location.x = 1.0
        control.keyframe_insert(data_path="location", index=0, frame=1)
        control.location.x = 3.0
        control.keyframe_insert(data_path="location", index=0, frame=10)
        animation = control.animation_data
        assert animation is not None and animation.action is not None
        action = animation.action
        animation.nla_tracks.new().strips.new("Authored clip", 20, action)
        helper = bpy.data.objects.new("Transport_helper", None)
        root.objects.link(helper)
        nested.objects.link(helper)
        constraint = helper.constraints.new("COPY_TRANSFORMS")
        assert isinstance(constraint, bpy.types.CopyTransformsConstraint)
        constraint.target = control
        helper["nested"] = {"targets": [{"owner": control}]}
        helper["signal"] = 0.0
        curve = helper.driver_add('["signal"]')
        assert isinstance(curve, bpy.types.FCurve) and curve.driver is not None
        variable = curve.driver.variables.new()
        variable.name = "control_x"
        variable.type = "TRANSFORMS"
        variable.targets[0].id = control
        curve.driver.expression = "control_x"
        mesh = bpy.data.meshes.new("Transport_geometry")
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        output = bpy.data.objects.new("Transport_mesh", mesh)
        nested.objects.link(output)
        output.parent = helper
        bpy.context.view_layer.update()
        output.matrix_parent_inverse = helper.matrix_world.inverted()
        parent_inverse = output.matrix_parent_inverse.copy()
        output["reference_parent_inverse"] = [value for row in parent_inverse for value in row]
        modifier = output.modifiers.new("Authored binding", "ARMATURE")
        assert isinstance(modifier, bpy.types.ArmatureModifier)
        modifier.object = control
        output.shape_key_add(name="Basis")
        shape = output.shape_key_add(name="Expression")
        material = bpy.data.materials.new("Transport_material")
        material.use_nodes = True
        mesh.materials.append(material)
        assert material.node_tree is not None
        value = material.node_tree.nodes.new("ShaderNodeValue")
        for driven in (shape.driver_add("value"), value.outputs[0].driver_add("default_value")):
            assert isinstance(driven, bpy.types.FCurve) and driven.driver is not None
            variable = driven.driver.variables.new()
            variable.name = "signal"
            variable.type = "SINGLE_PROP"
            variable.targets[0].id = helper
            variable.targets[0].data_path = '["signal"]'
            driven.driver.expression = "signal"
        descriptor = reference.describe_instance(
            bpy.context.scene, "character_dna", {"name": root.name, "face_board": control}
        )
        assert not descriptor["issues"]
        source = tmp_path / "transport.blend"
        bpy.data.libraries.write(str(source), {root})
        bpy.data.batch_remove(ids=list(set(bpy.data.user_map()) - before))
        loaded = reference._load_objects(str(source), [descriptor], mode != "APPEND", False)  # pyright: ignore[reportPrivateUsage]
        original = dict(loaded)
        if mode == "EDITABLE_LINK":
            loaded = reference._editable_objects(loaded, descriptor)  # pyright: ignore[reportPrivateUsage]
        destination = reference._local_collections(descriptor, loaded)  # pyright: ignore[reportPrivateUsage]
        bpy.context.scene.collection.children.link(destination)
        control = loaded[descriptor["pointers"]["face_board"]]
        helper = loaded["OBJECT::Transport_helper"]
        output = loaded["OBJECT::Transport_mesh"]
        assert helper.constraints[0].target == control
        assert helper["nested"]["targets"][0]["owner"] == control
        assert helper.animation_data.drivers[0].driver.variables[0].targets[0].id == control
        assert output.parent == helper
        assert output.matrix_parent_inverse == parent_inverse
        assert output.modifiers[0].object == control
        assert output.data.shape_keys.animation_data.drivers[0].driver.variables[0].targets[0].id == helper
        material = output.material_slots[0].material
        assert material.node_tree.animation_data.drivers[0].driver.variables[0].targets[0].id == helper
        assert helper in destination.objects.values()
        assert helper in destination.children[0].objects.values()
        assert len(bpy.data.scenes) == len([owner for owner in before if isinstance(owner, bpy.types.Scene)])
        assert destination.library is None
        assert destination.children[0].library is None
        assert len(control.animation_data.nla_tracks) == 1
        assert control.animation_data.nla_tracks[0].strips[0].action == control.animation_data.action
        assert bool(control.library) == (mode == "LINK")
        assert bool(control.data.library) == (mode == "LINK")
        assert bool(control.animation_data.action.library) == (mode == "LINK")
        if mode == "EDITABLE_LINK":
            assert helper.override_library.is_system_override
            assert output.override_library.is_system_override
            assert output.data.override_library.is_system_override
            assert material.override_library.is_system_override
            assert (
                original["OBJECT::Transport_helper"].constraints[0].target
                == original["OBJECT::Transport_authored_control"]
            )
            assert control.animation_data.action != original["OBJECT::Transport_authored_control"].animation_data.action
        positions = []
        for frame in (1, 10):
            bpy.context.scene.frame_set(frame)
            positions.append(control.evaluated_get(bpy.context.view_layer.depsgraph).location.x)
        assert positions == pytest.approx([1.0, 3.0])
        saved = tmp_path / f"transport_{mode}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(saved), copy=True)
        script = textwrap.dedent("""\
            import sys
            import bpy

            saved, mode = sys.argv[1:]
            for cycle in range(2):
                bpy.ops.wm.open_mainfile(filepath=saved)
                root = next(collection for collection in bpy.data.collections if collection.name == "Transport")
                control = next(owner for owner in root.all_objects if owner.name == "Transport_authored_control")
                helper = next(owner for owner in root.all_objects if owner.name == "Transport_helper")
                output = next(owner for owner in root.all_objects if owner.name == "Transport_mesh")
                assert helper.constraints[0].target == control
                assert all(item["owner"] == control for item in helper["nested"]["targets"])
                assert helper.animation_data.drivers[0].driver.variables[0].targets[0].id == control
                assert output.parent == helper
                assert list(output["reference_parent_inverse"]) == [
                    value for row in output.matrix_parent_inverse for value in row
                ]
                assert output.modifiers[0].object == control
                shape_keys = output.data.shape_keys
                assert shape_keys.animation_data.drivers[0].driver.variables[0].targets[0].id == helper
                material = output.material_slots[0].material
                assert material.node_tree.animation_data.drivers[0].driver.variables[0].targets[0].id == helper
                assert control.animation_data.nla_tracks[0].strips[0].action == control.animation_data.action
                assert bool(control.library) == (mode == "LINK")
                if mode == "EDITABLE_LINK":
                    protected = (helper, output, output.data, material)
                    assert all(owner.override_library.is_system_override for owner in protected)
                positions = []
                for frame in (1, 10):
                    bpy.context.scene.frame_set(frame)
                    positions.append(control.evaluated_get(bpy.context.view_layer.depsgraph).location.x)
                assert positions == [1.0, 3.0]
                if cycle == 0:
                    bpy.ops.wm.save_as_mainfile(filepath=saved)
            print("TRANSPORT_RELOAD_OK", flush=True)
            """)
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(Path(__file__).parent / "utilities" / "process.py"), script, str(saved), mode],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "TRANSPORT_RELOAD_OK" in result.stdout
    finally:
        bpy.data.batch_remove(ids=list(set(bpy.data.user_map()) - before))


@pytest.mark.parametrize(
    ("operation", "editable_rig", "metahuman_names", "current_metahuman_name"),
    [
        ("APPEND", False, ["ada"], "ada2"),
        ("LINK", False, ["ada"], "ada2"),
        ("LINK", True, ["ada"], "ada2"),
        ("APPEND", False, ["ada"], ""),
        ("LINK", False, ["ada"], ""),
        ("LINK", True, ["ada"], ""),
    ],
)
def test_reference_blend_file(  # noqa: PLR0912, PLR0915
    setup_reference_blend_file: Path,
    temp_folder: Path,
    operation: str,
    editable_rig: bool,
    metahuman_names: list[str],
    current_metahuman_name: str,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
):
    from fixtures.scene import load_dna

    protected_paths = (
        setup_reference_blend_file,
        *(TEST_DNA_FOLDER / "ada" / name for name in ("head.dna", "body.dna")),
    )
    fingerprints = {path: hashlib.sha256(path.read_bytes()).digest() for path in protected_paths}

    def verify_protected_files() -> None:
        """Verify the source blend and both DNA inputs even when a regression assertion fails."""
        assert fingerprints == {path: hashlib.sha256(path.read_bytes()).digest() for path in protected_paths}

    request.addfinalizer(verify_protected_files)
    load_dna(
        file_path=TEST_DNA_FOLDER / "ada" / "head.dna",
        import_lods=["lod0"],
        import_shape_keys=False,
        import_face_board=True,
        include_body=True,
    )
    instance = get_active_rig_instance()
    if not instance:
        pytest.fail("Rig instance should be created after loading DNA")

    # Rename the current instance to avoid name clashes
    if current_metahuman_name:
        instance.name = current_metahuman_name
    else:
        bpy.ops.wm.read_homefile(app_template="")

    monkeypatch.setattr(bpy.context.preferences.filepaths, "use_scripts_auto_execute", True)
    scene_names = {scene.name for scene in bpy.data.scenes}
    original_widget_count = len(
        {
            bone.custom_shape
            for item in bpy.context.scene.character_dna.rig_instance_list
            if item.face_board
            for bone in item.face_board.pose.bones
            if bone.custom_shape
        }
    )

    result = bpy.ops.character_dna.append_or_link_metahuman(  # type: ignore
        filepath=str(setup_reference_blend_file),
        operation_type=operation,
        editable_rig=editable_rig,
        meta_human_names=",".join(metahuman_names),
    )
    assert result == {"FINISHED"}

    instances = list(bpy.context.scene.character_dna.rig_instance_list)  # type: ignore
    from character_dna.utilities.reference import _widget_objects

    for imported_instance in instances:
        widgets = _widget_objects(imported_instance.face_board, set(bpy.data.objects))
        assert not widgets.intersection(bpy.context.scene.objects.values()), "Widget sources must not be scene objects"
    if current_metahuman_name and (operation == "APPEND" or editable_rig):
        existing_face = next(item.face_board for item in instances if item.name == current_metahuman_name)
        imported_face = next(item.face_board for item in instances if item.name in metahuman_names)
        for bone in imported_face.pose.bones:
            if bone.custom_shape:
                assert bone.custom_shape == existing_face.pose.bones[bone.name].custom_shape
        assert (
            len({bone.custom_shape for item in instances for bone in item.face_board.pose.bones if bone.custom_shape})
            == original_widget_count
        )
    instance_names = [instance.name for instance in instances]

    assert {scene.name for scene in bpy.data.scenes} == scene_names
    for name in [*metahuman_names, *([current_metahuman_name] if current_metahuman_name else [])]:
        assert name in instance_names, f"Rig instance {name} should be present in the scene"

    for instance in instances:
        assert instance.body_rig is not None, f"Body rig should be created for {name}"
        assert instance.body_mesh is not None, f"Body mesh should be created for {name}"
        assert instance.body_dna_file_path is not None, f"Body DNA file path should be set for {name}"
        assert instance.head_rig is not None, f"Head rig should be created for {name}"
        assert instance.head_mesh is not None, f"Head mesh should be created for {name}"
        assert instance.head_dna_file_path is not None, f"Head DNA file path should be set for {name}"

    # The face board must be grouped in the instance's collection for both operations, not
    # left loose in the scene root. See issue #341.
    for name in metahuman_names:
        instance = bpy.context.scene.character_dna.rig_instance_list.get(name)  # type: ignore
        assert instance and instance.face_board, f"Face board should be created for {name}"
        assert instance.face_board.get("authored_reference_control") is True
        assert instance.face_board.animation_data.action is not None
        root = instance["reference_root"]
        assert root.library is None
        assert all(collection.library is None for collection in root.children_recursive)
        helper = next(owner for owner in root.objects if owner.name.startswith("reference_shared_helper"))
        nested = next(collection for collection in root.children if collection.name.startswith("reference_nested"))
        assert helper in nested.objects.values()
        face_board_collections = [c.name for c in instance.face_board.users_collection]
        assert face_board_collections == [name], (
            f"Face board for {name} should only be in the {name} collection, got {face_board_collections}"
        )
        root_objects = [o.name for o in bpy.context.scene.collection.objects]
        assert instance.face_board.name not in root_objects, (
            f"Face board for {name} should not be loose in the scene root collection"
        )

    for reload in (False, True):
        if reload:
            saved_path = temp_folder / f"{operation}_{editable_rig}_{current_metahuman_name}_evaluated.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(saved_path))
            bpy.ops.wm.open_mainfile(filepath=str(saved_path))
        instances = list(bpy.context.scene.character_dna.rig_instance_list)  # type: ignore
        assert {instance.name for instance in instances} == set(instance_names)
        for imported_instance in instances:
            widgets = _widget_objects(imported_instance.face_board, set(bpy.data.objects))
            assert not widgets.intersection(bpy.context.scene.objects.values())
        if current_metahuman_name and (operation == "APPEND" or editable_rig):
            existing_face = next(item.face_board for item in instances if item.name == current_metahuman_name)
            imported_face = next(item.face_board for item in instances if item.name in metahuman_names)
            assert all(
                bone.custom_shape == existing_face.pose.bones[bone.name].custom_shape
                for bone in imported_face.pose.bones
                if bone.custom_shape
            )
        for instance in instances:
            assert engine.active(instance), {
                "reload": reload,
                "issues": engine.binding_issues(instance),
                "carriers": [
                    (carrier.name, carrier.get("rig"), carrier.get("face")) for carrier in engine.carriers(instance)
                ],
            }
            imported = instance.name in metahuman_names
            if imported:
                full_link = operation == "LINK" and not editable_rig
                assert bool(instance.face_board.library) == full_link
                assert bool(instance.face_board.data.library) == full_link
                assert bool(instance.face_board.animation_data.action.library) == full_link
                if full_link:
                    assert instance.head_rig.library is not None
                    assert instance.head_rig.override_library is None
                if operation == "LINK" and editable_rig:
                    assert instance.head_rig.override_library is not None
                    assert instance.head_rig.override_library.is_system_override
                    assert instance.head_rig.data.library is not None
                _assert_reference_control_rig(instance, instances, operation, editable_rig)
                matrices = []
                for frame in (1, 10, 25, 50, 100, 200):
                    bpy.context.scene.frame_set(frame)
                    matrices.append(_jaw_matrix(instance))
                assert matrices[0] != matrices[1]
                assert matrices[1] != matrices[2]
                assert matrices[0] == matrices[-1]
                if full_link:
                    continue
            face_board = instance.face_board
            animation = face_board.animation_data
            action = animation.action if animation else None
            slot_handle = animation.action_slot_handle if animation else None
            if action:
                animation.action = None
            face_board.pose.bones["CTRL_C_jaw"].location.y = 0.0
            face_board.update_tag()
            bpy.context.view_layer.update()
            before = {other.name: _jaw_matrix(other) for other in instances}
            face_board.pose.bones["CTRL_C_jaw"].location.y = 0.8
            face_board.update_tag()
            bpy.context.view_layer.update()
            assert before[instance.name] != _jaw_matrix(instance), (
                f"Face board should drive head bones for {instance.name} after {operation} (reload={reload})"
            )
            for other in instances:
                if other != instance:
                    assert _jaw_matrix(other) == before[other.name], "Face boards must evaluate independently"
            face_board.pose.bones["CTRL_C_jaw"].location.y = 0.0
            face_board.update_tag()
            bpy.context.view_layer.update()
            assert _jaw_matrix(instance) == before[instance.name]
            if action:
                animation.action = action
                animation.action_slot_handle = slot_handle


def _body_matrix(rig: bpy.types.Object, bone_name: str) -> "Matrix":
    """Read an evaluated body bone in world space without writing its protected pose."""
    evaluated = rig.evaluated_get(bpy.context.view_layer.depsgraph)
    assert evaluated.pose is not None
    return evaluated.matrix_world @ evaluated.pose.bones[bone_name].matrix


def _assert_reference_control_rig(  # noqa: PLR0915
    instance: "RigInstance", instances: list["RigInstance"], operation: str, editable_rig: bool
) -> None:
    """Check authored binding, sampled motion, and independent editable controls across reloads."""
    control = instance.control_rig
    body = instance.body_rig
    assert control is not None and body is not None
    widget = control.pose.bones["DEF-upper_arm.L"].custom_shape
    assert widget is not None, "Control-rig custom shapes must survive reference imports"
    assert widget not in bpy.context.scene.objects.values()
    assert widget.parent not in bpy.context.scene.objects.values()
    root = instance["reference_root"]
    assert control in root.objects.values()
    animation = control.animation_data
    assert animation is not None and animation.action is not None
    full_link = operation == "LINK" and not editable_rig
    for owner in (control, control.data, animation.action):
        assert bool(owner.library) == full_link
        assert owner.is_editable != full_link
        assert owner.override_library is None
    binding = body.pose.bones["upperarm_l"].constraints["Reference Control Rig"]
    assert binding.type == "COPY_TRANSFORMS" and binding.influence == 1.0 and not binding.mute
    assert binding.owner_space == binding.target_space == "WORLD"
    child = binding.target
    assert child is not None and child.name.startswith("reference_control_child")
    parent = child.parent
    assert parent is not None and parent.name.startswith("reference_control_parent")
    following = parent.constraints[0]
    assert following.type == "COPY_TRANSFORMS" and following.influence == 1.0 and not following.mute
    assert following.owner_space == following.target_space == "WORLD"
    assert following.target == control and following.subtarget == "DEF-upper_arm.L"
    nested = next(
        collection for collection in root.children if collection.name.startswith("reference_control_constraints")
    )
    assert child in nested.objects.values() and parent in nested.objects.values()
    source_body = source_control = None
    if operation == "LINK" and editable_rig:
        for owner in (body, child, parent):
            assert owner.library is None
            assert owner.override_library is not None and owner.override_library.is_system_override
        assert body.data.library is not None
        source_body = body.override_library.reference
        source_child = source_body.pose.bones["upperarm_l"].constraints["Reference Control Rig"].target
        source_control = source_child.parent.constraints[0].target
        assert source_control.library is not None and source_control != control
        assert source_child == child.override_library.reference
        assert source_child.parent == parent.override_library.reference
        assert source_control.data != control.data
        assert source_control.animation_data.action != animation.action
        assert source_control.animation_data.action.library is not None
        assert child.matrix_parent_inverse == source_child.matrix_parent_inverse, (
            "Editable link must preserve the authored control-rig rest offset when remapping the parent empty"
        )
    else:
        for owner in (body, child, parent):
            assert bool(owner.library) == full_link
            assert owner.override_library is None
    for frame in (1, 10, 25, 200):
        bpy.context.scene.frame_set(frame)
        for bone_name, expected in control["reference_body_samples"][str(frame)].items():
            actual = [value for row in _body_matrix(body, bone_name) for value in row]
            assert actual == pytest.approx(list(expected), abs=1e-5), {
                "instance": instance.name,
                "frame": frame,
                "bone": bone_name,
                "control": tuple(control.pose.bones["DEF-upper_arm.L"].rotation_euler),
                "child": [list(row) for row in child.matrix_world],
                "parent_inverse": [list(row) for row in child.matrix_parent_inverse],
                "source_inverse": [list(row) for row in source_child.matrix_parent_inverse] if source_body else None,
            }
    if full_link:
        return
    bpy.context.scene.frame_set(10)
    before = {other.name: _body_matrix(other.body_rig, "hand_l") for other in instances}
    source_before = _body_matrix(source_body, "hand_l") if source_body else None
    source_pose = source_control.pose.bones["DEF-upper_arm.L"] if source_control else None
    source_basis = source_pose.matrix_basis.copy() if source_pose else None
    pose = control.pose.bones["DEF-upper_arm.L"]
    angle = pose.rotation_euler.z
    try:
        pose.rotation_euler.z = angle + 0.4
        pose.keyframe_insert(data_path="rotation_euler", index=2, frame=10)
        control.update_tag()
        bpy.context.scene.frame_set(10)
        assert _body_matrix(body, "hand_l") != before[instance.name], "Local control must drive the bound body"
        for other in instances:
            if other != instance:
                assert _body_matrix(other.body_rig, "hand_l") == before[other.name]
        if source_body:
            assert _body_matrix(source_body, "hand_l") == source_before
            assert source_pose is not None
            assert source_pose.matrix_basis == source_basis
            assert source_child.parent.constraints[0].target == source_control
    finally:
        pose.rotation_euler.z = angle
        pose.keyframe_insert(data_path="rotation_euler", index=2, frame=10)
        control.update_tag()
        bpy.context.scene.frame_set(10)
    restored = [value for row in _body_matrix(body, "hand_l") for value in row]
    assert restored == pytest.approx(list(control["reference_body_samples"]["10"]["hand_l"]), abs=1e-5)


def _jaw_matrix(instance):
    return instance.head_rig.evaluated_get(bpy.context.view_layer.depsgraph).pose.bones["FACIAL_C_Jaw"].matrix.copy()
