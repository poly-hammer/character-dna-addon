import shutil

import pytest

from character_dna.utilities import remove_instance_prefix, replace_instance_prefix
from constants import HEAD_DNA_FILE
from fixtures.scene import load_dna


@pytest.mark.parametrize(
    ("name", "instance_name", "expected"),
    [
        ("Ada_head_lod0_mesh", "Ada", "head_lod0_mesh"),
        ("head_head_lod0_mesh", "head", "head_lod0_mesh"),
        ("head_lod0_mesh", "Ada", "head_lod0_mesh"),
        ("_head_lod0_mesh", "", "_head_lod0_mesh"),
    ],
)
def test_remove_instance_prefix(name: str, instance_name: str, expected: str) -> None:
    assert remove_instance_prefix(name, instance_name) == expected


@pytest.mark.parametrize(
    ("name", "old_instance_name", "new_instance_name", "expected"),
    [
        ("head", "head", "Ada", "Ada"),
        ("head_head_lod0_mesh", "head", "Ada", "Ada_head_lod0_mesh"),
        ("custom_head_mesh", "head", "Ada", "custom_head_mesh"),
    ],
)
def test_replace_instance_prefix(name: str, old_instance_name: str, new_instance_name: str, expected: str) -> None:
    assert replace_instance_prefix(name, old_instance_name, new_instance_name) == expected


def test_standalone_head_dna_preserves_shape_key_mesh_name(addon, temp_folder) -> None:
    from character_dna.dna_io import get_dna_reader
    from character_dna.editors.shape_key_editor import callbacks
    from character_dna.utilities import get_active_rig_instance

    dna_path = temp_folder / "standalone" / "head.dna"
    dna_path.parent.mkdir(parents=True)
    shutil.copy(HEAD_DNA_FILE, dna_path)

    load_dna(
        file_path=dna_path,
        import_lods=["lod0"],
        include_body=False,
        import_shape_keys=False,
        import_face_board=False,
    )

    instance = get_active_rig_instance()
    assert instance is not None
    assert instance.name == "head"

    reader = get_dna_reader(dna_path)
    instance.data[instance.cache_key("head", "dna_reader")] = reader
    mesh_index = next(iter(reader.getMeshIndicesForLOD(0)))
    mesh_object = instance.head_mesh_index_lookup[mesh_index]
    assert mesh_object.name == "head_head_lod0_mesh"

    mesh_object.shape_key_add(name="Basis", from_mix=False)
    channel_index = reader.getBlendShapeChannelIndex(mesh_index, 0)
    channel_name = reader.getBlendShapeChannelName(channel_index)
    shape_key = mesh_object.shape_key_add(name=f"head_lod0_mesh__{channel_name}", from_mix=False)

    enum_items = callbacks.get_active_shape_key_mesh_names(None, None)
    assert enum_items[0][0] == "head_head_lod0_mesh"
    assert enum_items[0][1] == "head_lod0_mesh"

    instance.data.pop(instance.cache_key("head", "shape_key_blocks"), None)
    assert shape_key in instance.head_shape_key_blocks[channel_index]
