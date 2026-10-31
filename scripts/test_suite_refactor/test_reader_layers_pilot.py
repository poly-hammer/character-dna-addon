"""Benchmark actual native reads with all evaluation layers and no geometry."""

import json
import os
import time

from pathlib import Path

import numpy as np
import pytest

from character_dna.bindings import dna, riglogic
from character_dna.dna_io import get_dna_reader, release_dna_handle
from character_dna.dna_io.coordinates import configuration
from character_dna.editors.swing_twist_editor.core import read_setups


ROOT = Path(__file__).resolve().parents[2]
RESULTS = []


def read(path: Path, slim: bool) -> dna.BinaryStreamReader:
    if not slim:
        return get_dna_reader(path)
    config = configuration()
    config.layer = (
        dna.DataLayer_RBFBehavior
        | dna.DataLayer_MachineLearnedBehavior
        | dna.DataLayer_JointBehaviorMetadata
        | dna.DataLayer_TwistSwingBehavior
    )
    stream = dna.FileStream(str(path), dna.AccessMode_Read, dna.OpenMode_Binary, None)
    reader = dna.BinaryStreamReader(stream, config, None)
    reader.read()
    assert dna.Status.isOk(), dna.Status.get().message
    return reader


def outputs(reader: dna.BinaryStreamReader) -> list[list[np.ndarray]]:
    manager = riglogic.RigLogic(reader, riglogic.Configuration(), None)
    state = riglogic.RigInstance(rigLogic=manager, memRes=None)
    result = []
    try:
        indices = list(reader.getRBFSolverRawControlIndices(0)) if reader.getRBFSolverCount() else []
        values = list(reader.getRBFSolverRawControlValues(0)) if indices else []
        for lod in range(reader.getLODCount()):
            for pose in range(min(3, len(values) // len(indices)) if indices else 1):
                for i in range(reader.getRawControlCount()):
                    state.setRawControl(i, float(reader.getRawControlName(i).endswith(".qw")))
                for index, value in zip(indices, values[pose * len(indices) : (pose + 1) * len(indices)], strict=True):
                    state.setRawControl(index, value)
                state.setLOD(lod)
                manager.calculate(state)
                result.append(
                    [
                        np.array(state.getJointOutputs()),
                        np.array(state.getBlendShapeOutputs()),
                        np.array(state.getAnimatedMapOutputs()),
                    ]
                )
        return result
    finally:
        release_dna_handle(state)
        release_dna_handle(manager)


@pytest.mark.parametrize("kind", ["head", "body"])
def test_slim_reader_equivalence_and_timing(kind: str) -> None:
    path = ROOT / "tests/test_files/dna/ada" / f"{kind}.dna"
    readings = {False: [], True: []}
    # Alternate order to reduce warmed file-cache bias; every handle is owned.
    for repeat in range(5):
        for slim in [False, True] if repeat % 2 == 0 else [True, False]:
            start = time.perf_counter()
            reader = read(path, slim)
            readings[slim].append(time.perf_counter() - start)
            release_dna_handle(reader)
    full, slim = read(path, False), read(path, True)
    try:
        assert read_setups(full) == read_setups(slim)
        for name in (
            "JointCount",
            "RBFSolverCount",
            "RBFPoseCount",
            "RawControlCount",
            "BlendShapeChannelCount",
            "LODCount",
        ):
            assert getattr(full, "get" + name)() == getattr(slim, "get" + name)()
        assert full.getVertexPositionCount(0) > 0 and slim.getVertexPositionCount(0) == 0
        a, b = outputs(full), outputs(slim)
        assert len(a) == len(b)
        for full_values, slim_values in zip(a, b, strict=True):
            for x, y in zip(full_values, slim_values, strict=True):
                assert np.isfinite(x).all() and np.isfinite(y).all()
                np.testing.assert_allclose(x, y, atol=1e-6, rtol=0)
        RESULTS.append(
            {
                "kind": kind,
                "full_seconds": readings[False],
                "slim_seconds": readings[True],
                "evaluations": len(a),
                "equivalent": True,
            }
        )
        Path(os.environ["PILOT_REPORT"]).with_suffix(".readers.json").write_text(json.dumps(RESULTS, indent=2))
    finally:
        release_dna_handle(slim)
        release_dna_handle(full)
