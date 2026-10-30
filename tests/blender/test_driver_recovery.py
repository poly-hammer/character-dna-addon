"""Real persisted rigs recover without granting script execution to the blend file."""

from __future__ import annotations

import hashlib

from pathlib import Path

import bpy
import pytest

from character_dna.runtime import controller, engine
from character_dna.utilities import get_active_rig_instance
from constants import TEST_DNA_FOLDER


def _pose() -> tuple[float, ...]:
    instance = get_active_rig_instance()
    graph = bpy.context.evaluated_depsgraph_get()
    return tuple(
        value
        for rig in (instance.head_rig, instance.body_rig)
        for bone in rig.evaluated_get(graph).pose.bones
        for row in bone.matrix
        for value in row
    )


def _drivers() -> list:
    return [carrier.animation_data.drivers.find('["epoch"]').driver for carrier in engine.carriers()]


def _fail_drivers() -> None:
    bpy.app.driver_namespace.pop(engine.NAMESPACE, None)
    carriers = engine.carriers(get_active_rig_instance())
    for carrier in carriers:
        driver = carrier.animation_data.drivers.find('["epoch"]').driver
        driver.expression = driver.expression
        carrier.update_tag()
    bpy.context.view_layer.update()
    assert carriers and all(
        not carrier.animation_data.drivers.find('["epoch"]').driver.is_valid for carrier in carriers
    )


@pytest.fixture(scope="module")
def saved_native_rig(addon, tmp_path_factory):
    """Keep a small full head/body rig with a non-neutral saved pose."""
    bpy.ops.wm.read_homefile(app_template="")
    bpy.ops.character_dna.import_dna(
        filepath=str(TEST_DNA_FOLDER / "ada" / "head.dna"),
        include_body=True,
        import_face_board=True,
        import_mesh=False,
        import_bones=True,
        import_shape_keys=False,
        import_materials=False,
    )
    instance = get_active_rig_instance()
    instance.face_board.pose.bones["CTRL_C_jaw"].location.y = 0.8
    instance.face_board.update_tag()
    bpy.context.view_layer.update()
    expected = _pose()
    path = tmp_path_factory.mktemp("driver_recovery") / "rig.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    return path, expected, instance.name


@pytest.fixture
def untrusted_rig(saved_native_rig, monkeypatch):
    path, expected, name = saved_native_rig
    monkeypatch.setattr(bpy.context.preferences.filepaths, "use_scripts_auto_execute", False)
    bpy.ops.wm.open_mainfile(filepath=str(path), use_scripts=False)
    yield path, expected, name
    bpy.ops.wm.read_homefile(app_template="")


def _assert_recovered(expected) -> None:
    bpy.context.view_layer.update()
    assert _drivers() and all(driver.is_valid for driver in _drivers())
    assert engine.active(get_active_rig_instance())
    assert _pose() == pytest.approx(expected, abs=1e-6)
    assert not bpy.context.preferences.filepaths.use_scripts_auto_execute


def test_load_without_script_permission(untrusted_rig):
    """Ordinary initialization prevents a permission failure in the first place."""
    _path, expected, _name = untrusted_rig
    _assert_recovered(expected)
    assert not bpy.app.autoexec_fail


@pytest.mark.parametrize("trusted", [False, True])
def test_recovers_after_earlier_load_handler(untrusted_rig, trusted):
    """Another addon can evaluate before our load handler, with either trust setting."""
    path, expected, _name = untrusted_rig
    failures = []

    @bpy.app.handlers.persistent
    def early_evaluation(_):
        bpy.context.view_layer.update()
        failures.append([driver.is_valid for driver in _drivers()])

    bpy.app.handlers.load_post.insert(0, early_evaluation)
    try:
        bpy.ops.wm.open_mainfile(filepath=str(path), use_scripts=trusted)
        assert failures and not any(failures[0])
        _assert_recovered(expected)
    finally:
        bpy.app.handlers.load_post.remove(early_evaluation)


def test_enabling_addon_after_open_recovers_drivers(untrusted_rig):
    """The deferred startup path repairs a file opened before the addon was enabled."""
    path, expected, _name = untrusted_rig
    bpy.ops.preferences.addon_disable(module="character_dna")
    try:
        bpy.ops.wm.open_mainfile(filepath=str(path), use_scripts=False)
        bpy.context.view_layer.update()
        assert _drivers() and not any(driver.is_valid for driver in _drivers())
    finally:
        bpy.ops.preferences.addon_enable(module="character_dna")
        bpy.context.preferences.addons["character_dna"].preferences.backup_manager.enabled = False
    controller._startup()
    _assert_recovered(expected)


@pytest.mark.parametrize("recover", ["hydrate", "adopt", "force_evaluate"])
def test_recovery_of_existing_native_sessions(untrusted_rig, recover):
    """Both automatic and manual recovery repair a driver with an existing session."""
    _path, expected, _name = untrusted_rig
    records = dict(engine._records)
    _fail_drivers()
    if recover == "hydrate":
        engine.hydrate()
    elif recover == "adopt":
        engine.adopt(get_active_rig_instance())
    else:
        assert bpy.ops.character_dna.force_evaluate() == {"FINISHED"}
    _assert_recovered(expected)
    if recover != "force_evaluate":
        assert all(engine._records[key] is record for key, record in records.items())


def test_scene_setup_initializes_before_external_callbacks(untrusted_rig):
    """Setup callbacks from other addons must see a ready runtime before evaluating."""
    from character_dna import post_setup_scene_callbacks
    from character_dna.utilities.misc import setup_scene

    observed = []

    def evaluate(instance):
        observed.append(engine.active(instance))
        bpy.context.view_layer.update()

    post_setup_scene_callbacks.append(evaluate)
    try:
        setup_scene()
        assert observed and all(observed)
        _assert_recovered(untrusted_rig[1])
        assert not bpy.app.autoexec_fail
    finally:
        post_setup_scene_callbacks.remove(evaluate)


def test_hydration_tags_valid_drivers_evaluated_before_sessions(untrusted_rig):
    """A callback returning cached outputs before hydration must get another evaluation."""
    _path, expected, _name = untrusted_rig
    engine.invalidate()
    instance = get_active_rig_instance()
    instance.face_board.pose.bones["CTRL_C_jaw"].location.y = 0.2
    instance.face_board.update_tag()
    bpy.context.view_layer.update()
    assert all(driver.is_valid for driver in _drivers())
    engine.hydrate()
    bpy.context.view_layer.update()
    assert _pose() != pytest.approx(expected, abs=1e-6)
    pose = _pose()
    instance.face_board.update_tag()
    bpy.context.view_layer.update()
    assert _pose() == pytest.approx(pose, abs=1e-6)


def test_unrelated_scripts_remain_blocked(untrusted_rig, tmp_path):
    """Global autoexec_fail may refer to unrelated scripts; never grant those permission."""
    _path, expected, _name = untrusted_rig
    text = bpy.data.texts.new("UntrustedRecoveryTest.py")
    text.from_string("import bpy\nbpy.context.scene['untrusted_text_ran'] = True\n")
    text.use_module = True
    owner = bpy.data.objects.new("UnrelatedScriptedDriver", None)
    bpy.context.scene.collection.objects.link(owner)
    owner["value"] = 0.0
    owner.driver_add('["value"]').driver.expression = "__import__('math').sqrt(4)"
    path = tmp_path / "unrelated_scripts.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    bpy.ops.wm.open_mainfile(filepath=str(path), use_scripts=False)
    _assert_recovered(expected)
    assert bpy.app.autoexec_fail
    assert not bpy.context.scene.get("untrusted_text_ran", False)
    driver = bpy.data.objects["UnrelatedScriptedDriver"].animation_data.drivers[0].driver
    assert not driver.is_valid
    assert driver.expression == "__import__('math').sqrt(4)"


@pytest.mark.parametrize("expression", [engine.SOLVE_EXPRESSION, engine.PREVIEW_EXPRESSION])
def test_modified_solve_expression_is_not_repaired(untrusted_rig, expression):
    """A carrier marker alone does not authorize rewriting a custom expression."""
    carrier = engine.carriers()[0]
    driver = carrier.animation_data.drivers.find('["epoch"]').driver
    expression += " + missing_custom_function()"
    driver.expression = expression
    carrier.update_tag()
    bpy.context.view_layer.update()
    assert not driver.is_valid
    assert "missing native solve driver" in "; ".join(engine.carrier_issues(carrier))
    engine.hydrate()
    bpy.context.view_layer.update()
    assert driver.expression == expression
    assert not driver.is_valid


@pytest.mark.parametrize("recover", ["hydrate", "adopt", "reload"])
def test_preview_recovery_preserves_frame_dependency(untrusted_rig, tmp_path, recover):
    """Preview rigs remain current after recovery and return to live inputs on playback."""
    from character_dna.utilities import detect_runtime_migration

    pytest.importorskip("character_dna.editors.raw_control_editor.callbacks")
    instance = get_active_rig_instance()
    # File loading hydrates native sessions; editing initializes the SDK lazily.
    instance.evaluate()
    instance.raw_control_editor.raw_controls["CTRL_expressions.jawOpen"].value = 0.3
    bpy.context.view_layer.update()
    expected = _pose()
    live_pose = untrusted_rig[1]
    assert expected != pytest.approx(live_pose, abs=1e-6)
    assert engine.binding_issues(instance) == []
    assert not detect_runtime_migration(bpy.context.scene)
    if recover == "reload":
        path = tmp_path / "preview.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        bpy.ops.wm.open_mainfile(filepath=str(path), use_scripts=False)
    else:
        _fail_drivers()
        if recover == "hydrate":
            engine.hydrate()
        else:
            engine.adopt(instance)
    _assert_recovered(expected)
    instance = get_active_rig_instance()
    assert engine.binding_issues(instance) == []
    assert not detect_runtime_migration(bpy.context.scene)
    head = next(carrier for carrier in engine.carriers(instance) if carrier["component"] == "head")
    assert head.animation_data.drivers.find('["epoch"]').driver.expression == engine.PREVIEW_EXPRESSION
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
    _assert_recovered(live_pose)


@pytest.mark.parametrize("editable", [False, True], ids=["linked", "override"])
def test_linked_driver_recovery_preserves_source(untrusted_rig, editable):
    """Recovery works for linked rigs and overrides without changing the source file."""
    path, expected, name = untrusted_rig
    fingerprint = hashlib.sha256(Path(path).read_bytes()).digest()
    bpy.ops.wm.read_homefile(app_template="")
    assert bpy.ops.character_dna.append_or_link_metahuman(
        filepath=str(path), operation_type="LINK", editable_rig=editable, meta_human_names=name
    ) == {"FINISHED"}
    carriers = engine.carriers()
    identities = [(carrier.as_pointer(), carrier.library, carrier.override_library) for carrier in carriers]
    assert all(carrier.library or carrier.override_library for carrier in carriers)
    _fail_drivers()
    engine.hydrate()
    _assert_recovered(expected)
    assert identities == [(carrier.as_pointer(), carrier.library, carrier.override_library) for carrier in carriers]
    assert all(driver.expression == engine.SOLVE_EXPRESSION for driver in _drivers())
    assert hashlib.sha256(Path(path).read_bytes()).digest() == fingerprint
