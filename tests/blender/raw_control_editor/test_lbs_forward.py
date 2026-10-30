"""Unit tests for the pure-numpy LBS forward model and solver contract.

These run in CI with only numpy + a synthetic ``_FakeReader`` -- no
Blender, no DNA file, no torch. They lock down the forward kinematics,
skinning, and skin-weight extraction that every solver backend and the
parity bench depend on. If these break, no solver result can be trusted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from character_dna.editors.raw_control_editor.solver import lbs_math, metrics
from character_dna.editors.raw_control_editor.solver.interface import (
    SeedSolver,
    SolveProblem,
)


class _FakeReader:
    """A 3-joint chain ``root(0) -> mid(1) -> tip(2)`` with one LOD0 mesh of
    three vertices, each rigidly bound (weight 1) to one joint.

    Neutral local transforms are pure translations along +X (10 cm apart),
    so the neutral world translations are root=(0,0,0), mid=(10,0,0),
    tip=(20,0,0). Vertices sit on top of their joints.
    """

    def __init__(self) -> None:
        self._vertex_joint = [0, 1, 2]
        self._verts = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)]

    def getJointCount(self) -> int:
        return 3

    def getJointParentIndex(self, i: int) -> int:
        return [0, 0, 1][i]

    def getMeshIndicesForLOD(self, lod: int) -> list[int]:
        return [0]

    def getVertexPositionXs(self, mesh_index: int) -> list[float]:
        return [v[0] for v in self._verts]

    def getVertexPositionYs(self, mesh_index: int) -> list[float]:
        return [v[1] for v in self._verts]

    def getVertexPositionZs(self, mesh_index: int) -> list[float]:
        return [v[2] for v in self._verts]

    def getSkinWeightsJointIndices(self, mesh_index: int, v_index: int) -> list[int]:
        return [self._vertex_joint[v_index]]

    def getSkinWeightsValues(self, mesh_index: int, v_index: int) -> list[float]:
        return [1.0]

    # Neutral local-translation tables (10 cm steps along +X).
    def getNeutralJointTranslationXs(self) -> list[float]:
        return [0.0, 10.0, 10.0]

    def getNeutralJointTranslationYs(self) -> list[float]:
        return [0.0, 0.0, 0.0]

    def getNeutralJointTranslationZs(self) -> list[float]:
        return [0.0, 0.0, 0.0]

    def getNeutralJointRotationXs(self) -> list[float]:
        return [0.0, 0.0, 0.0]

    def getNeutralJointRotationYs(self) -> list[float]:
        return [0.0, 0.0, 0.0]

    def getNeutralJointRotationZs(self) -> list[float]:
        return [0.0, 0.0, 0.0]


def _neutral_world(reader: _FakeReader) -> np.ndarray:
    """Build neutral world matrices from the fake reader's translation
    tables (translation-only locals)."""
    parents = lbs_math.build_parent_indices(reader)
    tx = reader.getNeutralJointTranslationXs()
    ty = reader.getNeutralJointTranslationYs()
    tz = reader.getNeutralJointTranslationZs()
    local = np.array([np.eye(4) for _ in range(reader.getJointCount())])
    for i in range(reader.getJointCount()):
        local[i, :3, 3] = (tx[i], ty[i], tz[i])
    return lbs_math.forward_kinematics(local, parents)


def test_skin_weight_arrays_shapes() -> None:
    reader = _FakeReader()
    joints, weights = lbs_math.build_skin_weight_arrays(reader)
    assert joints.shape == (3, 1)
    assert weights.shape == (3, 1)
    assert joints[:, 0].tolist() == [0, 1, 2]
    assert weights[:, 0].tolist() == [1.0, 1.0, 1.0]


def test_neutral_vertices_concatenation() -> None:
    reader = _FakeReader()
    v = lbs_math.build_neutral_vertices(reader)
    assert v.shape == (3, 3)
    assert v[1].tolist() == [10.0, 0.0, 0.0]


def test_lbs_at_bind_is_identity() -> None:
    """LBS with ``world == bind`` must reproduce the neutral mesh exactly."""
    reader = _FakeReader()
    bind = _neutral_world(reader)
    v_bind = lbs_math.build_neutral_vertices(reader)
    joints, weights = lbs_math.build_skin_weight_arrays(reader)
    bind_inv = np.linalg.inv(bind)

    deformed = lbs_math.lbs_forward_from_world(v_bind, joints, weights, bind, bind_inv)
    assert np.allclose(deformed, v_bind, atol=1e-9)


def test_lbs_translates_single_joint() -> None:
    """Translating only the tip joint moves only the tip vertex by the same
    amount (each vertex is rigidly bound to one joint)."""
    reader = _FakeReader()
    bind = _neutral_world(reader)
    bind_inv = np.linalg.inv(bind)
    v_bind = lbs_math.build_neutral_vertices(reader)
    joints, weights = lbs_math.build_skin_weight_arrays(reader)

    world = bind.copy()
    world[2, :3, 3] += np.array([0.0, 5.0, 0.0])  # move tip +5cm in Y

    deformed = lbs_math.lbs_forward_from_world(v_bind, joints, weights, world, bind_inv)
    expected = v_bind.copy()
    expected[2] += np.array([0.0, 5.0, 0.0])
    assert np.allclose(deformed, expected, atol=1e-9)


def test_lbs_rotates_child_about_parent() -> None:
    """A 90 deg rotation of the mid joint about Z carries the tip vertex
    around the mid joint's pivot (rigid articulation sanity)."""
    reader = _FakeReader()
    bind = _neutral_world(reader)
    bind_inv = np.linalg.inv(bind)
    v_bind = lbs_math.build_neutral_vertices(reader)
    joints, weights = lbs_math.build_skin_weight_arrays(reader)
    parents = lbs_math.build_parent_indices(reader)

    # Rotate the mid joint's LOCAL frame by +90 deg about Z, recompose world.
    local = lbs_math.local_from_world(bind, parents)
    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    rot = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    local[1] = local[1] @ rot
    world = lbs_math.forward_kinematics(local, parents)

    deformed = lbs_math.lbs_forward_from_world(v_bind, joints, weights, world, bind_inv)
    # mid joint world pivot is (10,0,0); tip was at (20,0,0) -> offset (10,0,0)
    # rotated +90 about Z -> (0,10,0) -> world (10,10,0).
    assert np.allclose(deformed[2], np.array([10.0, 10.0, 0.0]), atol=1e-6)
    # root vertex unaffected, mid vertex unaffected (sits on the pivot).
    assert np.allclose(deformed[0], np.array([0.0, 0.0, 0.0]), atol=1e-9)
    assert np.allclose(deformed[1], np.array([10.0, 0.0, 0.0]), atol=1e-9)


def test_forward_kinematics_roundtrip() -> None:
    reader = _FakeReader()
    parents = lbs_math.build_parent_indices(reader)
    bind = _neutral_world(reader)
    local = lbs_math.local_from_world(bind, parents)
    world2 = lbs_math.forward_kinematics(local, parents)
    assert np.allclose(world2, bind, atol=1e-9)


def _make_problem(reader: _FakeReader) -> SolveProblem:
    bind = _neutral_world(reader)
    v_bind = lbs_math.build_neutral_vertices(reader)
    joints, weights = lbs_math.build_skin_weight_arrays(reader)
    return SolveProblem(
        v_bind=v_bind,
        influence_joints=joints,
        influence_weights=weights,
        parents=lbs_math.build_parent_indices(reader),
        bind_world=bind,
        seed_world=bind.copy(),
        target=v_bind.copy(),
        matcher_mask=np.array([False, True, True]),
    )


def test_seed_solver_returns_seed() -> None:
    reader = _FakeReader()
    problem = _make_problem(reader)
    result = SeedSolver().solve(problem)
    assert np.allclose(result.world, problem.seed_world)
    assert result.converged


def test_solve_problem_validate_rejects_mismatch() -> None:
    reader = _FakeReader()
    problem = _make_problem(reader)
    problem.target = np.zeros((2, 3))  # wrong vertex count
    with pytest.raises(ValueError, match="target has"):
        problem.validate()


def test_metrics_identity_is_zero() -> None:
    reader = _FakeReader()
    bind = _neutral_world(reader)
    v = lbs_math.build_neutral_vertices(reader)
    fit = metrics.mesh_fit(v, v)
    assert fit["rms"] == pytest.approx(0.0)
    jd = metrics.joint_diff(bind, bind)
    assert jd["translation_cm"]["max"] == pytest.approx(0.0)
    assert jd["rotation_deg"]["max"] == pytest.approx(0.0)


def test_influenced_vertex_mask() -> None:
    reader = _FakeReader()
    joints, weights = lbs_math.build_skin_weight_arrays(reader)
    joint_mask = np.array([False, True, False])
    vmask = metrics.influenced_vertex_mask(joints, weights, joint_mask)
    assert vmask.tolist() == [False, True, False]
