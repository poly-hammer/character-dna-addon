"""The small transaction fixture must retain genuine native evaluation."""

from pathlib import Path

import numpy as np
import pytest

from character_dna.bindings import riglogic
from character_dna.dna_io import get_dna_reader, release_dna_handle
from character_dna.runtime import engine
from constants import TEST_DNA_FOLDER
from fixtures.rigs import RigProfile


@pytest.mark.parametrize("kind", ["head", "body"])
def test_compact_fixture_keeps_all_behavior_at_every_lod(compact_dna, kind):
    paths = [TEST_DNA_FOLDER / "ada" / f"{kind}.dna", compact_dna / f"{kind}.dna"]
    readers = [get_dna_reader(path) for path in paths]
    managers, states = [], []
    try:
        for reader in readers:
            managers.append(riglogic.RigLogic(reader, riglogic.Configuration(), None))
            states.append(riglogic.RigInstance(managers[-1], None))
        for method in ("getJointCount", "getRawControlCount", "getRBFSolverCount", "getLODCount", "getMeshCount"):
            assert getattr(readers[0], method)() == getattr(readers[1], method)()
        assert all(readers[1].getVertexPositionCount(index) == 3 for index in range(readers[1].getMeshCount()))
        for posed in (False, True):
            controls = [
                1.0 if readers[0].getRawControlName(index).endswith(".qw") else (0.2 if posed else 0.0)
                for index in range(readers[0].getRawControlCount())
            ]
            for state in states:
                for index, value in enumerate(controls):
                    state.setRawControl(index, value)
            for lod in range(readers[0].getLODCount()):
                for manager, state in zip(managers, states, strict=True):
                    state.setLOD(lod)
                    manager.calculate(state)
                for method in ("getJointOutputs", "getBlendShapeOutputs", "getAnimatedMapOutputs"):
                    np.testing.assert_array_equal(getattr(states[0], method)(), getattr(states[1], method)())
    finally:
        for handles in (states, managers, readers):
            for handle in handles:
                release_dna_handle(handle)


def test_compact_leases_preserve_complete_native_data_and_private_scene_state(compact_dna, rig_factory, tmp_path):
    profile = RigProfile(asset=str(compact_dna), pose_bones=("FACIAL_C_Jaw", "calf_l"))
    before = None
    for index in range(2):
        with rig_factory.lease(profile, tmp_path / str(index)) as instance:
            assert len(instance.head_mesh.data.vertices) == len(instance.body_mesh.data.vertices) == 3
            for kind in profile.components:
                component = instance.component(kind)
                assert 1 < len(component.rig.pose.bones) < component.dna_reader.getJointCount() // 2
                assert component.dna_reader.getJointCount() > 300
                assert Path(component.dna_file_path).parent == tmp_path / str(index)
                assert not engine.binding_issues(instance)
            bone = instance.body_rig.pose.bones["calf_l"]
            if before is None:
                before = np.array(bone.matrix_basis)
            else:
                np.testing.assert_array_equal(bone.matrix_basis, before)
            bone.rotation_mode = "XYZ"
            bone.rotation_euler.x = 0.7
