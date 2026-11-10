"""Paired UV repairs preserve seams and refuse ambiguous or stale inputs."""

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from character_dna.editors.converter import uv_repair, uv_repair_dialog
from character_dna.editors.converter.native import supplied
from character_dna.editors.converter.native.session import GeometryCopyError, GeometryJob


@pytest.fixture
def pair():
    points = np.array([[0, 0, 0], [1, 0, 0], [0.1, 1, 0], [0.2, 0.3, 1]], dtype=float)
    faces = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]
    values = np.array(
        [[0, 0], [0, 1], [1, 0], [2, 0], [3, 0], [2, 1], [4, 0], [5, 0], [4, 1], [6, 0], [7, 0], [6, 1]], dtype=float
    )
    objects = []
    for index in range(2):
        mesh = bpy.data.meshes.new(f"RepairTest{index}")
        vertices = points.copy()
        vertices[:, 0] *= -1 if index else 1
        mesh.from_pydata(vertices, [], faces)
        mesh.uv_layers.new(name="Primary").data.foreach_set("uv", values.ravel())
        mesh.uv_layers.new(name="Extra").data.foreach_set("uv", (values / 2).ravel())
        mesh.uv_layers["Primary"].active_render = True
        obj = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        objects.append(obj)
    yield objects
    uv_repair_dialog.clear()
    for obj in objects:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def test_mirror_transfer_keeps_per_corner_seams_and_other_uv_maps(pair):
    reference, target = map(uv_repair.MeshSnapshot.read, pair)
    target.obj.data.uv_layers["Primary"].data.foreach_set("uv", (target.uv + 0.5).ravel())
    target = uv_repair.MeshSnapshot.read(target.obj)
    before_extra = [tuple(loop.uv) for loop in target.mesh.uv_layers["Extra"].data]
    fixed = uv_repair.mirrored_uvs(reference, target)
    np.testing.assert_array_equal(fixed, reference.uv)
    assert len(set(map(tuple, fixed[reference.corners == 0]))) > 1
    assert [tuple(loop.uv) for loop in target.mesh.uv_layers["Extra"].data] == before_extra
    np.testing.assert_array_equal(uv_repair.MeshSnapshot.read(target.obj).uv, target.uv)


def test_mirror_transfer_accepts_reordered_vertices_and_faces(pair):
    reference, target = map(uv_repair.MeshSnapshot.read, pair)
    order = np.array([2, 0, 3, 1])
    inverse = np.argsort(order)
    target.points = target.points[order]
    target.corners = inverse[target.corners].reshape(-1, 3)[::-1, ::-1].ravel()
    expected = reference.uv.reshape(-1, 3, 2)[::-1, ::-1].reshape(-1, 2)
    np.testing.assert_array_equal(uv_repair.mirrored_uvs(reference, target), expected)


def test_mirror_transfer_refuses_ambiguous_geometry(pair):
    reference, target = map(uv_repair.MeshSnapshot.read, pair)
    reference.points[1] = reference.points[0]
    target.points[1] = target.points[0]
    with pytest.raises(ValueError, match="ambiguous"):
        uv_repair.mirrored_uvs(reference, target)


def test_mirror_transfer_refuses_changed_connectivity(pair):
    reference, target = map(uv_repair.MeshSnapshot.read, pair)
    target.corners[:3] = target.corners[3:6]
    with pytest.raises(ValueError, match="bijection"):
        uv_repair.mirrored_uvs(reference, target)


def test_validation_rejects_current_uvs_and_accepts_verified_candidate(pair):
    reference, target = pair
    snapshot = uv_repair.MeshSnapshot.read(target)
    target.data.uv_layers["Primary"].data.foreach_set("uv", (snapshot.uv + 0.2).ravel())
    capture = supplied.capture(
        "eye_right",
        target,
        target.data,
        np.arange(4, dtype=np.int32),
        reference.data,
        {},
        {"validate_uvs": True, "uv_tolerance": 0.001},
        "test",
        deferred=True,
    )
    assert not uv_repair.validates(capture)
    assert uv_repair.validates(capture, snapshot.uv)


def test_stale_second_repair_prevents_all_writes(pair, monkeypatch):
    reference, target = map(uv_repair.MeshSnapshot.read, pair)
    monkeypatch.setattr(uv_repair, "settings_stamp", lambda _props: "same")
    monkeypatch.setattr(uv_repair, "validates", lambda *_args: True)
    good = uv_repair.Repair(reference, target, target.uv + 1, None)
    stale = uv_repair.MeshSnapshot.read(target.obj)
    stale.uv = stale.uv + 0.1
    batch = uv_repair.RepairBatch(bpy.context.scene, "same", [good, uv_repair.Repair(reference, stale, stale.uv, None)])
    with pytest.raises(ValueError, match="Input meshes changed"):
        batch.apply(bpy.context)
    np.testing.assert_array_equal(uv_repair.MeshSnapshot.read(target.obj).uv, target.uv)


def test_batch_only_changes_primary_uvs(pair, monkeypatch):
    reference, target = map(uv_repair.MeshSnapshot.read, pair)
    monkeypatch.setattr(uv_repair, "settings_stamp", lambda _props: "same")
    monkeypatch.setattr(uv_repair, "validates", lambda *_args: True)
    extra = [tuple(loop.uv) for loop in target.mesh.uv_layers["Extra"].data]
    fixed = target.uv + 0.25
    batch = uv_repair.RepairBatch(bpy.context.scene, "same", [uv_repair.Repair(reference, target, fixed, None)])
    batch.apply(bpy.context)
    after = uv_repair.MeshSnapshot.read(target.obj)
    np.testing.assert_array_equal(after.uv, fixed)
    np.testing.assert_array_equal(after.points, target.points)
    assert [tuple(loop.uv) for loop in target.mesh.uv_layers["Extra"].data] == extra


def test_preview_has_fixed_left_current_right_and_shared_coordinates():
    before = np.array([[0, 0], [1, 0], [3, 1]], dtype=float)
    after = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    fixed, current = uv_repair_dialog.preview_pixels(before, after, np.array([3]))
    assert fixed.shape == current.shape == (320, 320, 4)
    assert np.any(np.all(np.isclose(fixed, [0.25, 0.9, 0.4, 1]), axis=2))
    assert np.any(np.all(np.isclose(current, [1, 0.3, 0.25, 1]), axis=2))
    np.testing.assert_array_equal(fixed[112, 64], current[112, 64])
    np.testing.assert_allclose(fixed[112, 64], [0.46, 0.46, 0.46, 1])
    assert np.isfinite(fixed).all() and np.isfinite(current).all()


def test_geometry_error_keeps_failed_role():
    def fail():
        raise ValueError('{"geometry_copy_error":{"key":"eye_right","error":"UV coverage missing"}}')

    task = SimpleNamespace(
        poll=lambda: {"events": [], "dropped_events": 0, "status": "failed"}, result=fail, close=lambda: None
    )
    job = GeometryJob(task, {"eye_right": SimpleNamespace(prefix="Right eye")})
    with pytest.raises(GeometryCopyError) as caught:
        job.poll()
    assert caught.value.key == "eye_right"
    assert str(caught.value) == "Right eye: UV coverage missing"
