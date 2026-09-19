"""Face-board placement with Blender 4.5's native visibility bindings."""

import bpy
import pytest

from mathutils import Vector

from character_dna.components.head import CharacterComponentHead
from character_dna.runtime import controller, engine


@pytest.mark.parametrize("language", ["DEFAULT", "en_US", "zh_HANS"])
def test_reposition_bound_face_board(load_head_only_dna, language: str) -> None:
    instance = bpy.context.scene.character_dna.rig_instance_list[0]
    controller.rebuild(instance)
    face = instance.face_board
    armature = face.data
    assert engine.active(instance)
    if bpy.app.version < (5, 0, 0):
        assert armature.users > 1
        assert sum(obj.data == armature for obj in bpy.data.objects) == 1

    preferences = bpy.context.preferences.view
    previous_language = preferences.language
    previous_translation = preferences.use_translate_interface
    try:
        try:
            preferences.language = language
        except TypeError:
            pytest.skip("This Blender build does not bundle the requested translation")
        preferences.use_translate_interface = True
        component = CharacterComponentHead(rig_instance=instance)
        component.reposition_face_board(Vector((0, 0, 0)))
        assert face.data == armature
        assert engine.active(instance)
        assert not engine.binding_issues(instance)
    finally:
        preferences.language = previous_language
        preferences.use_translate_interface = previous_translation
