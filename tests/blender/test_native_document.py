"""Exercise SDK projection validation and parity against the Python reference."""

import pytest

from test_native_merged_graph import combined as combined_fixture, first_difference, graphs as graphs_fixture

from character_dna.bindings import load_native_runtime
from character_dna.dna_io import merged_graph, native_projection, partitions
from fixtures.graphs import copy_graph


graphs = graphs_fixture
combined = combined_fixture


@pytest.fixture(scope="module")
def projection(graphs, combined):
    native, indices = combined
    records = partitions.describe(graphs, native, indices)
    definitions = partitions.project(native, indices, records, geometry=False)
    return merged_graph.to_bytes(native), native_projection.plans(definitions, indices, records)


def test_native_projector_matches_reference(graphs, projection):
    data, plans = projection
    for kind, content in zip(graphs, load_native_runtime().project_components(data, plans), strict=True):
        assert first_difference(merged_graph.from_bytes(content), graphs[kind]) is None


@pytest.mark.parametrize("field", ["vertex", "slice", "joint", "channel", "definition", "negative", "empty"])
def test_invalid_projection_is_rejected(projection, field):
    data, original = projection
    plans = copy_graph(original)
    mesh = plans[0]["meshes"][0]
    if field == "vertex":
        mesh["variants"][0]["vertices"][0] = 0xFFFFFFFF
    elif field == "slice":
        mesh["variants"][0]["counts"][0] = 0xFFFFFFFF
    elif field == "joint":
        plans[0]["joints"][0] = plans[0]["joints"][1]
    elif field == "channel":
        mesh["channels"][0] = 65535
    elif field == "definition":
        plans[0]["meshes"].pop()
    elif field == "negative":
        mesh["variants"][0]["mesh"] = -1
    else:
        mesh["variants"] = []
    with pytest.raises(ValueError):
        load_native_runtime().project_components(data, plans)


def test_shared_lod_conflicts_are_rejected(combined, projection):
    native, indices = combined
    _data, plans = projection
    duplicate = next(values for values in indices["meshes"]["body"].values() if len(values) > 1)
    edited = copy_graph(native)
    surface = next(item for item in indices["surfaces"] if item["mesh"] == duplicate[1])
    vertex = surface["vertices"]["body"][0]
    edited["geom1.1"]["meshes"][duplicate[1]]["positions"]["xs"][vertex] += 0.125
    with pytest.raises(ValueError, match="Conflicting native edits"):
        load_native_runtime().project_components(merged_graph.to_bytes(edited), plans)


def test_runtime_definition_excludes_only_geometry(graphs):
    sdk = load_native_runtime()
    original = merged_graph.to_bytes(graphs["head"])
    edited = copy_graph(graphs["head"])
    target = next(
        target for mesh in edited["geom1.1"]["meshes"] for target in mesh["blendShapeTargets"] if target["deltas"]["xs"]
    )
    target["deltas"]["xs"][0] += 0.125
    changed = merged_graph.to_bytes(edited)
    assert changed != original
    assert sdk.runtime_definition(changed) == sdk.runtime_definition(original)
    group = next(group for group in edited["bhvr1.1"]["joints"]["jointGroups"] if group["values"])
    group["values"][0] += 0.125
    assert sdk.runtime_definition(merged_graph.to_bytes(edited)) != sdk.runtime_definition(original)


@pytest.mark.parametrize("kind", ["head", "body"])
def test_runtime_without_geometry_matches_full_sdk_at_every_lod(kind):
    from array import array

    import numpy as np

    from character_dna.bindings import riglogic
    from character_dna.dna_io import get_dna_reader, release_dna_handle
    from constants import TEST_DNA_FOLDER

    path = TEST_DNA_FOLDER / "ada" / f"{kind}.dna"
    reader = get_dna_reader(path)
    config = riglogic.Configuration()
    config.rotationType = riglogic.RotationType_Quaternions if kind == "body" else riglogic.RotationType_EulerAngles
    manager = riglogic.RigLogic(reader, config, None)
    state = riglogic.RigInstance(manager, None)
    sdk = load_native_runtime()
    session = sdk.create_session(sdk.load_model(str(path), kind == "body", False, True))
    try:
        for posed in (False, True):
            controls = array(
                "f",
                (
                    1.0
                    if reader.getRawControlName(i).endswith(".qw")
                    else (0.0 if ".q" in reader.getRawControlName(i) or not posed else 0.35)
                    for i in range(reader.getRawControlCount())
                ),
            )
            if posed:
                for solver in range(reader.getRBFSolverCount()):
                    ids = reader.getRBFSolverRawControlIndices(solver)
                    values = reader.getRBFSolverRawControlValues(solver)
                    for index, value in zip(ids, values[-len(ids) :], strict=True):
                        controls[index] = float(value)
            for i, value in enumerate(controls):
                state.setRawControl(i, value)
            for lod in range(reader.getLODCount()):
                state.setLOD(lod)
                manager.calculate(state)
                actual = sdk.evaluate(session, controls, lod)
                for label, method in (
                    ("joints", "getJointOutputs"),
                    ("blend_shapes", "getBlendShapeOutputs"),
                    ("animated_maps", "getAnimatedMapOutputs"),
                ):
                    np.testing.assert_allclose(
                        actual[label],
                        getattr(state, method)(),
                        atol=2e-5,
                        err_msg=f"{kind}/{label}/LOD{lod}/posed={posed}",
                    )
    finally:
        for handle in (state, manager, reader):
            release_dna_handle(handle)
