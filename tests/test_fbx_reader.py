from types import SimpleNamespace

import numpy as np
import pytest

from character_dna.fbx import load_fbx_animation, reader
from character_dna.fbx.maths import quat_multiply, quat_to_euler, quat_to_matrix
from constants import TEST_ANIMATION_FOLDER


ANIMATION_FILES = [
    ("body/MHC_BodyROM.fbx", "root"),
    ("head/MHC_HeadROM.fbx", "root"),
    ("head/MHC_FaceBoardROM.fbx", "Face_ControlBoard_CtrlRig"),
]


@pytest.fixture(scope="module", params=ANIMATION_FILES, ids=lambda value: value[0])
def clip(request):
    relative_path, root_name = request.param
    return load_fbx_animation(TEST_ANIMATION_FOLDER / relative_path), root_name


def test_clip_shapes_are_consistent(clip):
    animation, _ = clip

    assert animation.num_nodes == len(animation.node_names) > 0
    assert animation.num_frames > 0
    assert animation.rotations.shape == (animation.num_frames, animation.num_nodes, 4)
    assert animation.translations.shape == (animation.num_frames, animation.num_nodes, 3)
    assert animation.rest_rotations.shape == (animation.num_nodes, 4)
    assert animation.rest_translations.shape == (animation.num_nodes, 3)
    assert animation.parent_indices.shape == (animation.num_nodes,)


def test_unreal_export_metadata(clip):
    animation, _ = clip

    # These files are exported from Unreal, which is Z up and centimeters.
    assert animation.up_axis == "Z"
    assert animation.unit_meters == pytest.approx(0.01)
    assert animation.frame_rate > 0.0
    assert animation.take_name


def test_hierarchy_is_topologically_ordered(clip):
    animation, root_name = clip

    assert animation.node_names[0] == root_name
    assert animation.parent_indices[0] == -1
    # Every parent must appear before its child so hierarchy walks are a single pass.
    for index, parent in enumerate(animation.parent_indices):
        assert parent < index


def test_rotations_are_unit_quaternions(clip):
    animation, _ = clip

    norms = np.linalg.norm(animation.rotations, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(animation.rest_rotations, axis=-1), 1.0, atol=1e-6)


def test_animated_nodes_are_reported(clip):
    animation, _ = clip

    assert animation.animated_node_names
    assert animation.animated_node_names <= set(animation.node_names)


def test_node_indices_resolve_names(clip):
    animation, root_name = clip

    assert animation.node_indices[root_name] == 0
    for name in animation.node_names:
        assert animation.node_names[animation.node_indices[name]] == name


def test_explicit_frame_rate_changes_sample_count():
    file_path = TEST_ANIMATION_FOLDER / "body" / "MHC_BodyROM.fbx"

    single_rate = load_fbx_animation(file_path, frame_rate=30.0)
    double_rate = load_fbx_animation(file_path, frame_rate=60.0)

    assert double_rate.num_frames > single_rate.num_frames
    assert double_rate.frame_rate == 60.0


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_fbx_animation(TEST_ANIMATION_FOLDER / "does_not_exist.fbx")


def test_oversized_clip_is_rejected_before_frame_allocation(monkeypatch):
    node = SimpleNamespace(typed_id=1, parent=None, local_transform=np.eye(4), world_transform=np.eye(4))
    scene = SimpleNamespace(settings=SimpleNamespace(frames_per_second=30.0))
    baked = SimpleNamespace(playback_duration=2043224 / 30.0, key_time_max=2043224 / 30.0)
    monkeypatch.setattr(reader, "_order_nodes", lambda _scene: [node] * 2914)
    monkeypatch.setattr(reader, "_select_baked_anim", lambda _scene, _frame_rate: (baked, "Oversized take"))

    def reject_allocation(*args, **kwargs):
        pytest.fail("Oversized clip reached frame allocation")

    monkeypatch.setattr(reader.np, "arange", reject_allocation)
    monkeypatch.setattr(reader.np, "broadcast_to", reject_allocation)

    with pytest.raises(ValueError, match=r"2,043,225 frames.*2,914 nodes.*GiB"):
        reader._build_clip(scene, None)


@pytest.mark.parametrize("budget", [255, 256])
def test_clip_buffer_budget_counts_both_transforms_and_sample_times(monkeypatch, budget):
    node = SimpleNamespace(
        typed_id=1, name="root", parent=None, local_transform=np.eye(4), world_transform=np.eye(4)
    )
    scene = SimpleNamespace(
        settings=SimpleNamespace(frames_per_second=30.0, unit_meters=0.01, axes=SimpleNamespace(up=4))
    )
    baked = SimpleNamespace(playback_duration=0.1, key_time_max=0.1, nodes=[])
    monkeypatch.setattr(reader, "_order_nodes", lambda _scene: [node])
    monkeypatch.setattr(reader, "_select_baked_anim", lambda _scene, _frame_rate: (baked, "Short take"))
    monkeypatch.setattr(reader, "MAX_CLIP_BUFFER_BYTES", budget)

    if budget < 256:
        with pytest.raises(ValueError, match="4 frames across 1 nodes"):
            reader._build_clip(scene, None)
    else:
        animation = reader._build_clip(scene, None)
        assert animation.num_frames == 4
        assert animation.rotations.nbytes + animation.translations.nbytes + 4 * 8 == budget
        np.testing.assert_allclose(animation.rotations[:, 0], [[1.0, 0.0, 0.0, 0.0]] * 4)


@pytest.mark.parametrize("from_buffer", [False, True])
def test_rejected_clip_closes_scene(monkeypatch, tmp_path, from_buffer):
    closed = []
    scene = SimpleNamespace(close=lambda: closed.append(True))
    bindings = SimpleNamespace(load_file=lambda *_args, **_kwargs: scene, load_memory=lambda *_args, **_kwargs: scene)
    monkeypatch.setattr(reader, "_require_ufbx", lambda: bindings)

    def reject_clip(scene, frame_rate):
        raise ValueError("Clip exceeds animation buffer limit")

    monkeypatch.setattr(reader, "_build_clip", reject_clip)
    with pytest.raises(ValueError, match="buffer limit"):
        if from_buffer:
            reader.load_fbx_animation_buffer(b"")
        else:
            file_path = tmp_path / "oversized.fbx"
            file_path.touch()
            reader.load_fbx_animation(file_path)
    assert closed == [True]


def test_rejected_clip_is_reported_by_import_operator(monkeypatch, tmp_path):
    from character_dna import utilities
    from character_dna.operators import ImportAnimationBase

    reports = []
    operator = SimpleNamespace(match_frame_rate=True, report=lambda level, message: reports.append((level, message)))

    def reject_clip(*args, **kwargs):
        raise ValueError("Clip exceeds animation buffer limit")

    monkeypatch.setattr(utilities, "load_animation_clip", reject_clip)
    assert ImportAnimationBase.load_clip(operator, tmp_path / "oversized.fbx") is None
    assert reports == [({"ERROR"}, "Clip exceeds animation buffer limit")]


@pytest.mark.parametrize("order", ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"])
def test_quat_to_euler_round_trips_through_mathutils(order: str):
    from mathutils import Quaternion

    rng = np.random.default_rng(seed=7)
    quaternions = rng.normal(size=(32, 4))
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
    quaternions[quaternions[:, 0] < 0] *= -1.0

    ours = quat_to_euler(quaternions, order)

    for index, quaternion in enumerate(quaternions):
        expected = Quaternion(quaternion.tolist()).to_euler(order)
        # mathutils works in single precision, so this is its noise floor.
        assert ours[index] == pytest.approx(list(expected), abs=1e-5)


def test_quat_to_matrix_matches_mathutils():
    from mathutils import Quaternion

    rng = np.random.default_rng(seed=11)
    quaternions = rng.normal(size=(16, 4))
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)

    matrices = quat_to_matrix(quaternions)

    for index, quaternion in enumerate(quaternions):
        expected = Quaternion(quaternion.tolist()).to_matrix()
        assert np.allclose(matrices[index], np.asarray(expected), atol=1e-6)


def test_quat_multiply_matches_mathutils():
    from mathutils import Quaternion

    rng = np.random.default_rng(seed=13)
    left = rng.normal(size=(16, 4))
    right = rng.normal(size=(16, 4))
    left /= np.linalg.norm(left, axis=-1, keepdims=True)
    right /= np.linalg.norm(right, axis=-1, keepdims=True)

    products = quat_multiply(left, right)

    for index in range(left.shape[0]):
        expected = Quaternion(left[index].tolist()) @ Quaternion(right[index].tolist())
        assert products[index] == pytest.approx(list(expected), abs=1e-6)
