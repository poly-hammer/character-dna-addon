"""Compact previews match the full-DNA path, including existing and new shapes."""

from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest

from test_component_editing import component_reader, solver_data  # noqa: F401 - imported pytest fixture

from character_dna.editors.rbf_editor import core, preview


@pytest.mark.parametrize("new_shape", [False, True])
def test_evaluation_projection_preserves_every_output_and_target_identity(component_reader, new_shape):  # noqa: F811
    kind, reader = component_reader
    data = solver_data(reader)
    targets = []
    if new_shape:
        targets = [{"name": "projection_test", "mesh_index": 0, "vertices": [0], "deltas": [[0.1, 0, 0]]}]
        for number, pose in enumerate(data[0]["poses"]):
            item = asdict(core.initialize_rbf_driven_data("projection_test", pose["pose_index"], -1))
            item.update(data_type="SHAPE_KEY", scalar_value=0.37 if number else 0.0, joint_index=-1)
            pose["driven"].append(item)
    full = compact = None
    try:
        # Oracle uses the original complete source; only geometry projection is
        # bypassed, never RigLogic calculation or the production serializer.
        with patch.object(preview, "_evaluation_reader", side_effect=lambda source: (source, ())):
            full = preview.Preview(reader, data, quaternion_outputs=kind == "body", shape_targets=targets)
        compact = preview.Preview(reader, data, quaternion_outputs=kind == "body", shape_targets=targets)
        assert compact.reader.getVertexPositionCount(0) == 0
        assert full.reader.getVertexPositionCount(0) == reader.getVertexPositionCount(0) > 0
        for mesh in range(reader.getMeshCount()):
            assert compact.reader.getBlendShapeTargetCount(mesh) == full.reader.getBlendShapeTargetCount(mesh)
            assert [
                compact.reader.getBlendShapeChannelIndex(mesh, i)
                for i in range(compact.reader.getBlendShapeTargetCount(mesh))
            ] == [
                full.reader.getBlendShapeChannelIndex(mesh, i)
                for i in range(full.reader.getBlendShapeTargetCount(mesh))
            ]
        indices = list(reader.getRBFSolverRawControlIndices(0))
        values = list(reader.getRBFSolverRawControlValues(0))
        for lod in range(reader.getLODCount()):
            for pose in range(min(3, len(data[0]["poses"]))):
                for evaluator in (full, compact):
                    for index in range(reader.getRawControlCount()):
                        evaluator.state.setRawControl(index, float(reader.getRawControlName(index).endswith(".qw")))
                    for index, value in zip(
                        indices, values[pose * len(indices) : (pose + 1) * len(indices)], strict=True
                    ):
                        evaluator.state.setRawControl(index, value)
                    evaluator.state.setLOD(lod)
                    evaluator.manager.calculate(evaluator.state)
                for method in ("getJointOutputs", "getBlendShapeOutputs", "getAnimatedMapOutputs"):
                    actual, expected = getattr(compact.state, method)(), getattr(full.state, method)()
                    assert np.isfinite(actual).all()
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    finally:
        if compact:
            compact.close()
        if full:
            full.close()
