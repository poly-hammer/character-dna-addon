"""Temporary posing restrictions must not leak into the next edit session."""

from __future__ import annotations

import subprocess
import sys
import textwrap

from pathlib import Path
from types import SimpleNamespace

import pytest

from character_dna.editors.raw_control_editor.constants import (
    DEFAULT_RAW_CONTROL_INDEX,
    CacheNamespace,
)


class _FakeReader:
    """4-joint chain ``root(0) -> mid(1) -> leafA(2), leafB(3)``. Joint
    group 0 is driven by raw control 5 and outputs the two leaves, so
    ``joints_driven_by_raw_control(5) == {2, 3}`` -- joints 0 and 1 are
    outside the group."""

    def getJointCount(self) -> int:
        return 4

    def getJointName(self, i: int) -> str:
        return ["FACIAL_C_FacialRoot", "mid", "leafA", "leafB"][i]

    def getJointGroupCount(self) -> int:
        return 1

    def getJointGroupInputIndices(self, jg: int) -> list[int]:
        return [5]

    def getJointGroupOutputIndices(self, jg: int) -> list[int]:
        return [2 * 9, 3 * 9]


class _FakePoseBone:
    def __init__(self, name: str) -> None:
        self.name = name
        self.lock_location = (False, False, False)
        self.lock_rotation = (False, False, False)
        self.lock_rotation_w = False
        self.lock_scale = (False, False, False)


class _FakeBones:
    def __init__(self, names: list[str]) -> None:
        self._bones = {name: _FakePoseBone(name) for name in names}

    def get(self, name: str) -> _FakePoseBone | None:
        return self._bones.get(name)


class _FakePose:
    def __init__(self, names: list[str]) -> None:
        self.bones = _FakeBones(names)


class _FakeRig(dict):
    def __init__(self, name: str, bone_names: list[str]) -> None:
        self.name = name
        self.pose = _FakePose(bone_names)


class _FakeCache:
    """Stand-in for :class:`EditorCache` -- one persistent dict per
    namespace key."""

    def __init__(self) -> None:
        self._namespaces: dict[str, dict] = {}

    def namespace(self, key: CacheNamespace) -> dict:
        return self._namespaces.setdefault(str(key), {})


class _FakeInstance:
    def __init__(self, rig: _FakeRig, reader: _FakeReader) -> None:
        self.head_rig = rig
        self.head_dna_reader = reader


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch):
    from character_dna.editors.raw_control_editor import utilities

    cache = _FakeCache()
    monkeypatch.setattr(utilities, "session_cache", lambda _instance: cache)
    reader = _FakeReader()
    rig = _FakeRig("Ada_head_rig", [reader.getJointName(i) for i in range(reader.getJointCount())])
    instance = _FakeInstance(rig, reader)
    return utilities, instance, reader, rig, cache


def test_lock_locks_only_bones_outside_joint_group(patched) -> None:
    utilities, instance, reader, rig, _cache = patched

    locked = utilities.lock_bones_outside_joint_group(instance, reader, 5)

    # joints 0 (root) and 1 (mid) are outside the group -> locked.
    assert locked == 2
    bones = rig.pose.bones
    for name in ("FACIAL_C_FacialRoot", "mid"):
        pose_bone = bones.get(name)
        assert tuple(pose_bone.lock_location) == (True, True, True)
        assert tuple(pose_bone.lock_rotation) == (True, True, True)
        assert pose_bone.lock_rotation_w is True
        assert tuple(pose_bone.lock_scale) == (True, True, True)
    # joints 2 (leafA) and 3 (leafB) are in the group -> untouched.
    for name in ("leafA", "leafB"):
        pose_bone = bones.get(name)
        assert tuple(pose_bone.lock_location) == (False, False, False)
        assert tuple(pose_bone.lock_rotation) == (False, False, False)
        assert pose_bone.lock_rotation_w is False
        assert tuple(pose_bone.lock_scale) == (False, False, False)


def test_restore_returns_prior_lock_state(patched) -> None:
    utilities, instance, reader, rig, _cache = patched

    # leafA carries a pre-existing user lock that must survive untouched
    # (it is in the group, so lock never touches it).
    rig.pose.bones.get("leafA").lock_location = (True, False, False)

    utilities.lock_bones_outside_joint_group(instance, reader, 5)
    restored = utilities.restore_locked_bones(instance)

    assert restored == 2
    for name in ("FACIAL_C_FacialRoot", "mid"):
        pose_bone = rig.pose.bones.get(name)
        assert tuple(pose_bone.lock_location) == (False, False, False)
        assert tuple(pose_bone.lock_rotation) == (False, False, False)
        assert pose_bone.lock_rotation_w is False
        assert tuple(pose_bone.lock_scale) == (False, False, False)
    # The in-group bone's pre-existing lock is never altered.
    assert tuple(rig.pose.bones.get("leafA").lock_location) == (True, False, False)


def test_restore_clears_the_persisted_snapshot(patched) -> None:
    utilities, instance, reader, rig, cache = patched

    utilities.lock_bones_outside_joint_group(instance, reader, 5)
    assert rig.get(utilities._BONE_LOCK_SNAPSHOT)
    assert not cache.namespace(CacheNamespace.LOCKED_BONES)

    utilities.restore_locked_bones(instance)
    assert utilities._BONE_LOCK_SNAPSHOT not in rig

    # A second restore with nothing cached is a harmless no-op.
    assert utilities.restore_locked_bones(instance) == 0


def test_bind_pose_sentinel_locks_nothing(patched) -> None:
    """The bind-pose ("default") sentinel drives no joint group and
    writes neutral joints directly, so no bone should be locked."""
    utilities, instance, reader, rig, _cache = patched

    assert utilities.lock_bones_outside_joint_group(instance, reader, DEFAULT_RAW_CONTROL_INDEX) == 0
    for name in ("FACIAL_C_FacialRoot", "mid", "leafA", "leafB"):
        assert tuple(rig.pose.bones.get(name).lock_location) == (False, False, False)


@pytest.mark.parametrize("lose_cache", [False, True])
def test_switching_controls_restores_newly_eligible_bones(patched, lose_cache) -> None:
    utilities, instance, reader, rig, cache = patched
    # An unrelated control locks all four joints. The next control drives
    # the leaves, just as blink drives eyelids previously locked by mouth.
    utilities.lock_bones_outside_joint_group(instance, reader, 6)
    if lose_cache:
        cache._namespaces.clear()

    utilities.lock_bones_outside_joint_group(instance, reader, 5)

    assert tuple(rig.pose.bones.get("leafA").lock_location) == (False, False, False)
    assert tuple(rig.pose.bones.get("mid").lock_location) == (True, True, True)
    utilities.restore_locked_bones(instance)
    assert tuple(rig.pose.bones.get("mid").lock_location) == (False, False, False)


def test_repeated_entry_preserves_original_user_locks_after_cache_loss(patched) -> None:
    utilities, instance, reader, rig, cache = patched
    bone = rig.pose.bones.get("mid")
    bone.lock_location = (True, False, True)
    bone.lock_rotation = (False, True, False)
    bone.lock_rotation_w = True
    bone.lock_scale = (False, False, True)

    for _ in range(3):
        utilities.lock_bones_outside_joint_group(instance, reader, 5)
        cache._namespaces.clear()
    # Recovery is associated with the rig ID, not its mutable name.
    rig.name = "Renamed_head_rig"
    utilities.restore_locked_bones(instance)

    assert tuple(bone.lock_location) == (True, False, True)
    assert tuple(bone.lock_rotation) == (False, True, False)
    assert bone.lock_rotation_w is True
    assert tuple(bone.lock_scale) == (False, False, True)


def test_bind_pose_releases_previous_expression_locks(patched) -> None:
    utilities, instance, reader, rig, _cache = patched
    utilities.lock_bones_outside_joint_group(instance, reader, 6)
    utilities.lock_bones_outside_joint_group(instance, reader, DEFAULT_RAW_CONTROL_INDEX)
    assert tuple(rig.pose.bones.get("leafA").lock_location) == (False, False, False)
    assert tuple(rig.pose.bones.get("mid").lock_location) == (False, False, False)


def test_recovers_legacy_session_once(patched) -> None:
    utilities, instance, reader, rig, cache = patched
    bone = rig.pose.bones.get("leafA")
    cache.namespace(CacheNamespace.LOCKED_BONES)[(rig.name, 6)] = {
        "leafA": {
            "location": (False, True, False),
            "rotation": (False, False, False),
            "rotation_w": False,
            "scale": (False, False, False),
        }
    }
    bone.lock_location = (True, True, True)
    utilities.lock_bones_outside_joint_group(instance, reader, 5)
    assert tuple(bone.lock_location) == (False, True, False)
    assert not cache.namespace(CacheNamespace.LOCKED_BONES)


def test_undoing_entry_does_not_restore_a_stale_python_snapshot(patched) -> None:
    utilities, instance, reader, rig, _cache = patched
    utilities.lock_bones_outside_joint_group(instance, reader, 5)
    # Undo restores both the bone state and the ID property, without undoing
    # Python dictionaries. A subsequent user lock must not be overwritten.
    rig.pop(utilities._BONE_LOCK_SNAPSHOT)
    bone = rig.pose.bones.get("mid")
    bone.lock_location = (False, True, False)
    assert utilities.restore_locked_bones(instance) == 0
    assert tuple(bone.lock_location) == (False, True, False)


def test_deleted_bone_does_not_prevent_other_locks_restoring(patched) -> None:
    utilities, instance, reader, rig, _cache = patched
    utilities.lock_bones_outside_joint_group(instance, reader, 5)
    del rig.pose.bones._bones["mid"]
    assert utilities.restore_locked_bones(instance) == 1
    assert tuple(rig.pose.bones.get("FACIAL_C_FacialRoot").lock_location) == (False, False, False)


def test_snapshot_survives_blend_roundtrip(patched, tmp_path) -> None:
    """Exercise real Blender ID-property serialization and pose lock storage."""
    import bpy

    utilities, _instance, reader, _rig, cache = patched
    armature = bpy.data.armatures.new("lock_test")
    rig = bpy.data.objects.new("lock_test", armature)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    loaded = None
    try:
        bpy.ops.object.mode_set(mode="EDIT")
        for i in range(reader.getJointCount()):
            bone = armature.edit_bones.new(reader.getJointName(i))
            bone.tail.y = 1.0
        bpy.ops.object.mode_set(mode="OBJECT")
        rig.pose.bones["mid"].lock_location = (True, False, False)
        instance = SimpleNamespace(head_rig=rig)
        utilities.lock_bones_outside_joint_group(instance, reader, 5)
        path = str(tmp_path / "locks.blend")
        bpy.data.libraries.write(path, {rig})
        cache._namespaces.clear()
        with bpy.data.libraries.load(path) as (source, target):
            target.objects = source.objects
        loaded = target.objects[0]
        assert tuple(loaded.pose.bones["mid"].lock_location) == (True, True, True)

        assert utilities.restore_locked_bones(SimpleNamespace(head_rig=loaded)) == 2
        assert tuple(loaded.pose.bones["mid"].lock_location) == (True, False, False)
        assert tuple(loaded.pose.bones["FACIAL_C_FacialRoot"].lock_location) == (False, False, False)
        assert utilities._BONE_LOCK_SNAPSHOT not in loaded
    finally:
        bpy.data.objects.remove(rig, do_unlink=True)
        bpy.data.armatures.remove(armature)
        if loaded is not None:
            loaded_armature = loaded.data
            bpy.data.objects.remove(loaded, do_unlink=True)
            bpy.data.armatures.remove(loaded_armature)


def test_snapshot_and_locks_follow_blender_undo_redo() -> None:
    """Undo Commit/entry in a separate Blender process, without reverting pytest's scene."""
    root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(f"""
        import runpy, sys
        from types import SimpleNamespace
        sys.path.insert(0, {str(root / "src" / "addons")!r})
        import bpy
        from character_dna.editors.raw_control_editor import utilities
        fixtures = runpy.run_path({str(Path(__file__).resolve())!r})
        reader = fixtures['_FakeReader']()
        cache = fixtures['_FakeCache']()
        utilities.session_cache = lambda instance: cache
        bpy.context.preferences.edit.use_global_undo = True
        bpy.ops.object.armature_add()
        rig = bpy.context.active_object
        rig.data.bones[0].name = 'mid'
        rig.pose.bones['mid'].lock_location = (True, False, False)
        instance = SimpleNamespace(head_rig=rig)
        key = utilities._BONE_LOCK_SNAPSHOT

        bpy.ops.ed.undo_push(message='before entry')
        utilities.lock_bones_outside_joint_group(instance, reader, 5)
        bpy.ops.ed.undo_push(message='editing')
        utilities.restore_locked_bones(instance)
        bpy.ops.ed.undo_push(message='committed')

        bpy.ops.ed.undo()
        rig = bpy.context.active_object
        assert key in rig
        assert tuple(rig.pose.bones['mid'].lock_location) == (True, True, True)
        bpy.ops.ed.undo()
        rig = bpy.context.active_object
        assert key not in rig
        assert tuple(rig.pose.bones['mid'].lock_location) == (True, False, False)
        bpy.ops.ed.redo()
        rig = bpy.context.active_object
        assert key in rig
        assert tuple(rig.pose.bones['mid'].lock_location) == (True, True, True)

        cache._namespaces.clear()
        instance.head_rig = rig
        assert utilities.restore_locked_bones(instance) == 1
        assert tuple(rig.pose.bones['mid'].lock_location) == (True, False, False)
        assert key not in rig
        print('LOCK_UNDO_REDO_PASSED')
        """)
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(root / "tests" / "utilities" / "process.py"), script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LOCK_UNDO_REDO_PASSED" in result.stdout
