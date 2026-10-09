"""Face-board curve writing uses the same control selection as validation."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from character_dna.fbx import writer
from character_dna.utilities.action import FACE_BOARD_EXCLUDED_CONTROLS


@pytest.mark.parametrize("extra_exclusion", [frozenset(), frozenset({"CTRL_C_jaw"})])
def test_face_board_writer_only_writes_allowed_controls(monkeypatch, extra_exclusion):
    names = [
        "CTRL_C_jaw",
        "CTRL_L_brow_down",
        "FRM_C_tongue_move",
        "GRP_faceGUI",
        "headGui_grp",
        "headRig_grp",
        "CTRL_helper_grp",
        *sorted(FACE_BOARD_EXCLUDED_CONTROLS),
    ]
    clip = SimpleNamespace(
        node_indices={name: index for index, name in enumerate(names)},
        translations=np.ones((2, len(names), 3)),
        rotations=np.tile([1.0, 0.0, 0.0, 0.0], (2, len(names), 1)),
        rest_rotations=np.tile([1.0, 0.0, 0.0, 0.0], (len(names), 1)),
    )
    armature = SimpleNamespace(pose=SimpleNamespace(bones=[SimpleNamespace(name=name) for name in names]))
    container = object()
    write_curves = Mock()
    monkeypatch.setattr(writer, "get_channel_container", lambda *_args: container)
    monkeypatch.setattr(writer, "frame_range", lambda _clip: np.array([1.0, 2.0]))
    monkeypatch.setattr(writer, "write_bulk_fcurves", write_curves)

    written = writer.write_face_board_animation(
        clip, armature, object(), exclude_bones=FACE_BOARD_EXCLUDED_CONTROLS | extra_exclusion
    )

    expected = [name for name in ("CTRL_C_jaw", "CTRL_L_brow_down") if name not in extra_exclusion]
    assert written == expected
    assert [call.args[1] for call in write_curves.call_args_list] == [
        f'pose.bones["{name}"].location' for name in expected
    ]


def test_native_dna_skeletal_tracks_change_local_basis(monkeypatch):
    """A root-only rotation cannot perform this conjugation of local tracks."""
    from mathutils import Euler, Matrix

    rest = Euler((0.3, -0.4, 0.2)).to_quaternion()
    delta = Euler((-0.2, 0.5, 0.7)).to_quaternion()
    offset = np.array([0.02, -0.04, 0.07])
    clip = SimpleNamespace(
        node_indices={"joint": 0},
        rest_translations=np.array([[3, 4, 5]]),
        rest_rotations=np.array([rest]),
        rotations=np.array([[rest @ delta]]),
        translations=np.array([[np.array([3, 4, 5]) + np.asarray(rest.to_matrix()) @ offset / 0.01]]),
        unit_meters=0.01,
    )
    bone = SimpleNamespace(name="joint", rotation_mode="QUATERNION", path_from_id=lambda: 'pose.bones["joint"]')
    armature = SimpleNamespace(pose=SimpleNamespace(bones=[bone]), get=lambda _key, _default: 1)
    rotations = Mock()
    locations = Mock()
    monkeypatch.setattr(writer, "get_channel_container", lambda *_args: object())
    monkeypatch.setattr(writer, "frame_range", lambda _clip: np.array([1.0]))
    monkeypatch.setattr(writer, "_write_rotation", rotations)
    monkeypatch.setattr(writer, "write_bulk_fcurves", locations)
    writer.write_skeleton_animation(clip, armature, object())
    basis = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
    expected = (basis @ delta.to_matrix() @ basis.transposed()).to_quaternion()
    np.testing.assert_allclose(rotations.call_args.args[2][0], expected, atol=1e-6)
    np.testing.assert_allclose(locations.call_args.args[2][0], np.asarray(basis) @ offset, atol=1e-6)
