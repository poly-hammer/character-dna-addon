"""End-to-end test for the Match Bones to Mesh PyTorch solver.

This exercises the full *setup + run* path of the Bone Matching operator's
backend in a single, efficient test:

1. **Provision** the self-contained solver venv -- exactly what the
   ``Setup Solver Environment`` operator does: create a venv from the host
   Python and ``pip install torch + numpy`` for the CPU device. This is the
   guarantee that the install works on every supported platform.
2. **Gate** -- point the addon preferences at the provisioned venv and assert
   the operator's availability check (:func:`dependency_extraction.solver_available`)
   flips to ``True``.
3. **Run** -- drive a *tiny* synthetic solve problem (a 3-joint chain, mirroring
   :func:`utilities.solve_bone_match_v2`'s npz contract) through the real venv
   worker subprocess over IPC and assert the returned world matrices improve the
   fit to the target.

This runs as part of the normal test suite on every platform / Blender version.
The provisioned venv is cached at a stable, Python-version-keyed temp path, so
only the first run on a machine pays the torch-download cost; later runs reuse
it. CI runners are ephemeral, so they always provision (well within the job
timeout).

The solve is forced to ``device="cpu"`` for determinism (CI has no GPU; MPS /
CUDA variance is out of scope for this smoke test). The heavy real-DNA
870-joint operator solve is intentionally **not** exercised here -- it would
risk exhausting CI runner memory.
"""

from __future__ import annotations

import sys
import tempfile

from pathlib import Path

import numpy as np
import pytest


# Genuinely slow only on a cold cache (first torch download); the project
# convention marks such tests ``slow`` while still running them in the suite.
pytestmark = pytest.mark.slow


class _ChainReader:
    """Minimal DNA-reader stand-in for the solver npz builders.

    A 3-joint chain ``root(0) -> mid(1) -> tip(2)`` with one LOD0 mesh of three
    vertices, each rigidly bound (weight 1) to one joint. Neutral local
    transforms are pure +X translations 10 cm apart, so the neutral world
    translations are root=(0,0,0), mid=(10,0,0), tip=(20,0,0) and each vertex
    sits on top of its joint. Implements exactly the methods called by
    :func:`lbs_math.build_skin_weight_arrays`, :func:`lbs_math.build_neutral_vertices`,
    :func:`lbs_math.build_parent_indices` and
    :func:`shared.utilities.build_world_rest_pose`.
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


@pytest.fixture(scope="session")
def solver_venv():
    """Provision (or reuse) the per-device PyTorch solver venv and yield
    ``(root, device, venv_python)``.

    Mirrors ``Setup Solver Environment``: builds the ``cpu`` env under a stable,
    Python-version-keyed root and caches it (reused when it still validates), so
    only the first run pays the torch-download cost. Left on disk for reuse;
    every solver worker is shut down on teardown so nothing keeps files locked.
    """
    from character_dna.editors.raw_control_editor import dependency_extraction
    from character_dna.editors.raw_control_editor.solver_worker import (
        SolverEnvError,
        env_is_ready,
        provision,
        stop_all_solver_workers,
        validate_env,
    )
    from character_dna.utilities import get_addon_preferences

    version_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    root = Path(tempfile.gettempdir()) / f"character_dna_solver_env_e2e_{version_tag}"
    device = "cpu"

    # Point the addon preferences at this root + device so the venv is
    # provisioned into the exact (Python-version-scoped) directory the operator's
    # availability check will later resolve to. Using the production resolvers
    # keeps the fixture and the app in lockstep with the on-disk layout instead
    # of duplicating the ``<root>/python-<ver>/<device>`` path structure here.
    preferences = get_addon_preferences()
    assert preferences is not None, "addon preferences unavailable"
    rce = preferences.raw_control_editor
    rce.solver_env_root = str(root)
    rce.solver_device = device

    venv_dir = dependency_extraction.resolve_solver_env_dir(device)
    venv_python = dependency_extraction.resolve_solver_env_python(device)

    reuse = env_is_ready(venv_dir)
    if reuse:
        try:
            validate_env(venv_python)
        except SolverEnvError:
            reuse = False

    if not reuse:
        info = provision(venv_dir, device=device, blender_python=Path(sys.executable))
        # ``validate_env`` (inside ``provision``) raises if torch/numpy are
        # missing, so these keys are guaranteed present on success.
        assert "torch" in info, f"provision returned no torch version: {info}"
        assert "numpy" in info, f"provision returned no numpy version: {info}"

    assert venv_python.is_file(), f"venv interpreter not found at {venv_python}"

    try:
        yield root, device, venv_python
    finally:
        stop_all_solver_workers()


def test_solver_env_provisions_and_solves(solver_venv) -> None:
    """Provision the solver env, flip the operator gate, and run a tiny solve
    through the real venv worker."""
    from character_dna.editors.raw_control_editor import dependency_extraction
    from character_dna.editors.raw_control_editor.solver import lbs_math
    from character_dna.editors.raw_control_editor.solver_worker import get_or_start_solver_worker
    from character_dna.editors.shared.utilities import build_world_rest_pose
    from character_dna.utilities import get_addon_preferences

    root, device, venv_python = solver_venv

    # 1. Availability is derived from the env root + selected device; setting the
    #    prefs makes the operator gate flip on and resolve to this venv.
    preferences = get_addon_preferences()
    assert preferences is not None, "addon preferences unavailable"
    rce_prefs = preferences.raw_control_editor
    rce_prefs.solver_env_root = str(root)
    rce_prefs.solver_device = device
    assert dependency_extraction.solver_available() is True
    assert dependency_extraction.resolve_solver_env_python() == venv_python

    # 2. Build a tiny synthetic solve problem (mirrors solve_bone_match_v2's npz).
    reader = _ChainReader()
    influence_joints, influence_weights = lbs_math.build_skin_weight_arrays(reader)
    v_bind = lbs_math.build_neutral_vertices(reader)
    parents = lbs_math.build_parent_indices(reader)
    bind_world = lbs_math.matrices_to_numpy(build_world_rest_pose(reader))
    seed_world = bind_world.copy()  # seed from the neutral (bind) pose

    # Nudge the tip vertex 2 cm in +Y and unlock the mid + tip joints so the
    # solver has the freedom to fit it.
    target = v_bind.copy()
    target[2, 1] += 2.0
    matcher_mask = np.array([False, True, True], dtype=bool)

    # 3. Run the solve through the real venv worker subprocess over IPC.
    worker = get_or_start_solver_worker(venv_python)
    ping = worker.ping()
    assert ping.get("torch"), f"solver worker ping reported no torch: {ping}"

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        problem_path = Path(temp_dir) / "problem.npz"
        result_path = Path(temp_dir) / "result.npz"
        np.savez_compressed(
            problem_path,
            v_bind=v_bind,
            influence_joints=influence_joints,
            influence_weights=influence_weights,
            parents=parents,
            bind_world=bind_world,
            seed_world=seed_world,
            target=target,
            matcher_mask=matcher_mask,
            device=np.asarray("cpu", dtype="U"),
        )
        result = worker.solve(str(problem_path), str(result_path))
        with np.load(result_path) as result_npz:
            world = np.asarray(result_npz["world"])

    # 4. The worker returned well-formed matrices and improved the fit.
    assert world.shape == (3, 4, 4)
    assert np.all(np.isfinite(world))
    assert isinstance(result.get("converged"), bool)

    bind_world_inv = np.linalg.inv(bind_world)
    seed_fit = lbs_math.lbs_forward_from_world(v_bind, influence_joints, influence_weights, seed_world, bind_world_inv)
    solved_fit = lbs_math.lbs_forward_from_world(v_bind, influence_joints, influence_weights, world, bind_world_inv)
    seed_residual = float(np.linalg.norm(seed_fit - target))
    solved_residual = float(np.linalg.norm(solved_fit - target))
    assert solved_residual < seed_residual, (
        f"solver did not improve the fit: seed={seed_residual:.4f} cm, solved={solved_residual:.4f} cm"
    )
