from pathlib import Path

import bpy
import pytest

from character_dna.runtime import engine
from character_dna.ui.callbacks import get_active_rig_instance
from constants import TEST_DNA_FOLDER


@pytest.mark.parametrize(
    ("operation", "metahuman_names", "current_metahuman_name"),
    [
        ("APPEND", ["ada"], "ada2"),
        ("LINK", ["ada"], "ada2"),
        ("APPEND", ["ada"], ""),
        ("LINK", ["ada"], ""),
    ],
)
def test_reference_blend_file(
    setup_reference_blend_file: Path,
    temp_folder: Path,
    operation: str,
    metahuman_names: list[str],
    current_metahuman_name: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from fixtures.scene import load_dna

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

    result = bpy.ops.character_dna.append_or_link_metahuman(  # type: ignore
        filepath=str(setup_reference_blend_file), operation_type=operation, meta_human_names=",".join(metahuman_names)
    )
    assert result == {"FINISHED"}

    instances = list(bpy.context.scene.character_dna.rig_instance_list)  # type: ignore
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
            saved_path = temp_folder / f"{operation}_{current_metahuman_name}_evaluated.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(saved_path))
            bpy.ops.wm.open_mainfile(filepath=str(saved_path))
        instances = list(bpy.context.scene.character_dna.rig_instance_list)  # type: ignore
        assert {instance.name for instance in instances} == set(instance_names)
        for instance in instances:
            assert engine.active(instance)
            if instance.name in metahuman_names and operation == "LINK":
                assert instance.head_rig.override_library is not None
                assert instance.body_rig.override_library is not None
            face_board = instance.face_board
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


def _jaw_matrix(instance):
    return instance.head_rig.evaluated_get(bpy.context.view_layer.depsgraph).pose.bones["FACIAL_C_Jaw"].matrix.copy()
