"""Coordinate policy round trips preserve geometry and animation semantics."""

import math

import numpy as np
import pytest

from mathutils import Euler, Matrix

from character_dna.bindings import dna, riglogic
from character_dna.dna_io import get_dna_reader, get_dna_writer, release_dna_handle
from constants import TEST_DNA_FOLDER


def source_reader(path):
    reader = dna.BinaryStreamReader(dna.FileStream(str(path), dna.AccessMode_Read, dna.OpenMode_Binary, None))
    reader.read()
    assert dna.Status.isOk(), dna.Status.get().message
    return reader


@pytest.mark.parametrize("component", ["head", "body"])
@pytest.mark.parametrize("file_format", ["binary", "json"])
def test_canonical_export_from_native_dna(tmp_path, component, file_format):
    path = TEST_DNA_FOLDER / "ada" / f"{component}.dna"
    original = source_reader(path)
    native = get_dna_reader(path)
    system = native.getCoordinateSystem()
    assert (system.x, system.y, system.z) == (dna.Direction_left, dna.Direction_back, dna.Direction_up)
    assert native.getRotationSequence() == dna.RotationSequence_xzy
    basis = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
    for joint in range(native.getJointCount()):
        old = Euler(tuple(math.radians(v) for v in original.getNeutralJointRotation(joint)), "XYZ").to_matrix()
        new = Euler(tuple(math.radians(v) for v in native.getNeutralJointRotation(joint)), "XZY").to_matrix()
        # The SDK chooses a canonical Euler branch near gimbal lock, with
        # float32 rotation-matrix error below 2e-5 (about 0.001 degrees).
        np.testing.assert_allclose(new, basis @ old @ basis.transposed(), atol=2e-5)

    target = tmp_path / f"{component}.{file_format}"
    writer = get_dna_writer(target, file_format)
    writer.setFrom(native, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
    writer.write()
    release_dna_handle(writer)
    if file_format == "binary":
        exported = source_reader(target)
        system = exported.getCoordinateSystem()
        assert (system.x, system.y, system.z) == (dna.Direction_left, dna.Direction_up, dna.Direction_front)
        assert exported.getRotationSequence() == dna.RotationSequence_xyz
    reread = get_dna_reader(target, file_format)
    if file_format == "json":
        # Compare behavior against the SDK's existing JSON serialization path.
        # Six-digit JSON rounding can amplify in interpolative RBF solvers;
        # the coordinate conversion must add no further discrepancy.
        baseline_path = tmp_path / f"{component}-baseline.json"
        baseline_writer = dna.JSONStreamWriter(
            dna.FileStream(str(baseline_path), dna.AccessMode_Write, dna.OpenMode_Binary, None)
        )
        baseline_writer.setFrom(original, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
        baseline_writer.write()
        release_dna_handle(baseline_writer)
        native = get_dna_reader(baseline_path, "json")
    for mesh in range(native.getMeshCount()):
        for axis in "XYZ":
            method = f"getVertexPosition{axis}s"
            # JSON uses the SDK's six significant-digit text serialization.
            np.testing.assert_allclose(
                getattr(reread, method)(mesh),
                getattr(native, method)(mesh),
                rtol=5e-6 if file_format == "json" else 1e-7,
                atol=1e-5,
            )

    # Evaluate a mixed expression/driver input: copying only the descriptor or
    # neutral geometry would pass static checks and corrupt behavior here.
    configuration = riglogic.Configuration()
    configuration.rotationType = (
        riglogic.RotationType_Quaternions if component == "body" else riglogic.RotationType_EulerAngles
    )
    managers = [riglogic.RigLogic(reader, configuration, None) for reader in (native, reread)]
    instances = [riglogic.RigInstance(manager, None) for manager in managers]
    for instance, manager in zip(instances, managers, strict=True):
        for index in range(native.getRawControlCount()):
            name = native.getRawControlName(index)
            value = 1.0 if name.endswith(".qw") else (0.0 if ".q" in name else 0.35)
            instance.setRawControl(index, value)
        if native.getRBFSolverCount():
            indices = native.getRBFSolverRawControlIndices(0)
            values = native.getRBFSolverRawControlValues(0)
            for index, value in zip(indices, values[-len(indices) :], strict=True):
                instance.setRawControl(index, float(value))
        manager.calculate(instance)
    np.testing.assert_allclose(
        instances[0].getJointOutputs(),
        instances[1].getJointOutputs(),
        atol=2e-5,
    )
