import subprocess
import sys
import textwrap

from pathlib import Path

import bpy
import pytest

from character_dna import operators, utilities
from character_dna.ui import view_3d


def test_addons_are_enabled(addons):
    for addon_name, _ in addons:
        assert "character_dna" in bpy.context.preferences.addons, f"{addon_name} is not enabled"  # type: ignore


@pytest.mark.parametrize(
    "panel_class",
    [
        view_3d.CHARACTER_DNA_PT_output_panel,
        view_3d.CHARACTER_DNA_PT_rig_instance,
        view_3d.CHARACTER_DNA_PT_view_options,
        view_3d.CHARACTER_DNA_PT_face_board,
    ],
)
def test_view_3d(panel_class):
    assert panel_class.is_registered, f'The Panel in the 3D View "{panel_class.bl_label}" is not registered.'


@pytest.mark.parametrize(
    "operator_class",
    [
        operators.ForceEvaluate,
        operators.RefreshOutputItems,
        operators.ImportCharacterDna,
    ],
)
def test_operators(operator_class):
    assert operator_class.is_registered, f"Operator {operator_class.bl_idname} is not registered."


def test_edition_specific_registration():
    """The upsell panel is always registered; editor panels only exist in the Pro edition."""
    # The upsell panel is always registered in both editions; its poll controls visibility.
    assert view_3d.CHARACTER_DNA_PT_pro_upsell.is_registered, "The upsell panel should always be registered."

    editors = utilities.get_editors()
    if utilities.editors_available():
        assert editors is not None, "Editors submodule present but registry failed to import."
        raw_control_editor_ui = editors.raw_control_editor_ui  # type: ignore[attr-defined]
        assert raw_control_editor_ui.CHARACTER_DNA_PT_raw_control_editor.is_registered, (
            "Pro edition should register the editor panels."
        )
    else:
        assert editors is None, "Free edition should not import an editors registry."


def test_native_runtime_registration_is_repeatable():
    """Repeated or partial lifecycle calls must leave one operator and no stale callbacks."""
    from character_dna.runtime import controller, engine

    operator = controller.CHARACTER_DNA_OT_sync_native_runtime
    try:
        controller.register()
        controller.register()
        assert operator.is_registered
        assert bpy.app.handlers.blend_import_post.count(controller._after_import) == 1
        assert bpy.app.driver_namespace[engine.NAMESPACE] == engine.solve
        controller.request_sync()
        assert bpy.app.timers.is_registered(controller._apply_requested)
        controller.unregister()
        controller.unregister()
        assert not operator.is_registered
        assert controller._after_import not in bpy.app.handlers.blend_import_post
        assert not bpy.app.timers.is_registered(controller._apply_requested)
        assert not bpy.app.timers.is_registered(controller._startup)
        assert engine.NAMESPACE not in bpy.app.driver_namespace
    finally:
        controller.register()


@pytest.mark.parametrize("operation", ["register", "unregister"])
def test_native_runtime_resolves_registered_class(operation: str, monkeypatch: pytest.MonkeyPatch):
    """Resolve the live RNA class even when Python holds a replacement definition."""
    from character_dna.runtime import controller

    original = controller.CHARACTER_DNA_OT_sync_native_runtime
    replacement = type(
        original.__name__,
        (bpy.types.Operator,),
        {"bl_idname": original.bl_idname, "bl_label": original.bl_label, "execute": original.execute},
    )
    try:
        with monkeypatch.context() as patch:
            patch.setattr(controller, "CHARACTER_DNA_OT_sync_native_runtime", replacement)
            getattr(controller, operation)()
            assert not original.is_registered
            assert replacement.is_registered == (operation == "register")
            controller.unregister()
    finally:
        controller.register()


def test_reload_addon_source_code_in_isolated_process():
    """The development reload helper restores operators, properties and runtime services."""
    source = str(Path(utilities.__file__).resolve().parents[2])
    script = textwrap.dedent(
        f"""
        import sys
        import bpy
        import addon_utils
        from poly_hammer_utils.helpers import reload_addon_source_code

        sys.path.insert(0, {source!r})
        addon_utils.enable('character_dna', default_set=True)
        import character_dna

        character_dna.native_runtime.unregister()
        for cycle in range(2):
            previous = character_dna.native_runtime
            old_operator = previous.CHARACTER_DNA_OT_sync_native_runtime
            old_handler = previous._after_import
            old_startup = previous._startup
            old_requested = previous._apply_requested
            previous.request_sync()
            reload_addon_source_code(['character_dna'])
            current = character_dna.native_runtime
            assert current.CHARACTER_DNA_OT_sync_native_runtime.is_registered
            assert current.CHARACTER_DNA_OT_sync_native_runtime is not old_operator
            assert not old_operator.is_registered
            assert old_handler not in bpy.app.handlers.blend_import_post
            assert not bpy.app.timers.is_registered(old_startup)
            assert not bpy.app.timers.is_registered(old_requested)
            assert bpy.app.handlers.blend_import_post.count(current._after_import) == 1
            assert bpy.app.driver_namespace[current.engine.NAMESPACE] == current.engine.solve
            assert bpy.app.timers.is_registered(current._startup)
            assert bpy.context.scene.character_dna is not None
            assert bpy.context.window_manager.character_dna is not None
            assert character_dna.properties.RigInstance.is_registered
            assert character_dna.operators.ImportCharacterDna.is_registered

        addon_utils.disable('character_dna', default_set=False)
        assert not hasattr(bpy.types.Scene, 'character_dna')
        assert not hasattr(bpy.types.WindowManager, 'character_dna')
        print('RELOAD_CHECK_PASSED', flush=True)
        """
    )
    result = subprocess.run(
        [sys.executable, "-"], input=script, text=True, capture_output=True, timeout=120, check=False
    )
    output = result.stdout + result.stderr
    assert "RELOAD_CHECK_PASSED" in result.stdout, output
    assert "Traceback" not in result.stderr, output
