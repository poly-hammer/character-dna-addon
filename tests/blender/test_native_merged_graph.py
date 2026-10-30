"""Native merged DNA retains every component and LOD without external files."""

import base64
import hashlib
import json

import numpy as np
import pytest

from character_dna.dna_io import merged_graph, partitions
from constants import TEST_DNA_FOLDER
from fixtures.graphs import copy_graph, first_difference, restore_graph, snapshot_graph


@pytest.fixture(scope="module")
def graphs(native_graph_archive):
    value = restore_graph(native_graph_archive[0])
    before = snapshot_graph(value)
    yield value
    unchanged = snapshot_graph(value) == before
    assert unchanged, "A test mutated the module's reference graphs"


@pytest.fixture(scope="module")
def combined(native_graph_archive):
    value = restore_graph(native_graph_archive[1])
    before = snapshot_graph(value)
    yield value
    # Compare this module's restored representation before/after its tests.
    # Pickle memoization can change across a serialization round trip. Keep
    # the assertion scalar so a failure cannot render gigabytes of byte diff.
    unchanged = snapshot_graph(value) == before
    assert unchanged, "A test mutated the module's reference composition"


def test_sdk_json_interchange(graphs):
    for graph in graphs.values():
        assert merged_graph.to_bytes(graph).startswith(b"DNA")


def test_graph_digest_preserves_scalars_and_filters_nested_archive_fields():
    from character_dna.dna_io.metadata import PREFIX

    graph = {
        "desc1.1": {"metadata": [["author", "Ada"], [PREFIX, "payload"], [PREFIX + ".0", "chunk"]]},
        "size": 40,
        "values": [1, 2.5, "text", None, True, [], {"offset": 12, "keep": [3, {"index": 4, "value": 5}]}],
    }
    expected = {
        "desc1.1": {"metadata": [["author", "Ada"]]},
        "values": [1, 2.5, "text", None, True, [], {"keep": [3, {"value": 5}]}],
    }
    before = snapshot_graph(graph)
    encoded = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    assert merged_graph.digest(graph) == hashlib.sha256(encoded).hexdigest()
    assert snapshot_graph(graph) == before
    graph["values"][0] += 1
    assert merged_graph.digest(graph) != hashlib.sha256(encoded).hexdigest()


def test_native_all_lods_and_domains(graphs, combined):
    native, info = combined
    assert native["desc1.1"]["lodCount"] == 8
    assert len(info["surfaces"]) == 8
    for field in merged_graph.NAME_DOMAINS:
        assert len(native["defn1.1"][field]) == sum(len(graph["defn1.1"][field]) for graph in graphs.values())
    assert len(native["rbfb1.0"]["solvers"]) == 73
    for surface in info["surfaces"]:
        expected = sum(
            len(graphs[kind]["geom1.1"]["meshes"][surface[kind]]["positions"]["xs"]) for kind in ("head", "body")
        ) - len(surface["seam"])
        assert len(native["geom1.1"]["meshes"][surface["mesh"]]["positions"]["xs"]) == expected


def test_native_inverse_without_edits(graphs, combined):
    native, info = combined
    projected = partitions.project(native, info, partitions.describe(graphs, native, info))
    for kind in graphs:
        assert merged_graph.digest(projected[kind]) == merged_graph.digest(graphs[kind]), first_difference(
            projected[kind], graphs[kind]
        )


def test_native_inverse_uses_current_edits(graphs, combined):
    native, info = combined
    edited = copy_graph(native)
    surface = info["surfaces"][0]
    vertex = next(index for index in range(len(surface["vertices"]["head"])) if index not in surface["seam"])
    native_vertex = surface["vertices"]["head"][vertex]
    edited["geom1.1"]["meshes"][surface["mesh"]]["positions"]["xs"][native_vertex] += 1.0
    edited["rbfb1.0"]["solvers"][-1]["radius"] += 0.25
    projected = partitions.project(edited, info, partitions.describe(graphs, native, info))
    assert projected["head"]["geom1.1"]["meshes"][surface["head"]]["positions"]["xs"][vertex] == pytest.approx(
        graphs["head"]["geom1.1"]["meshes"][surface["head"]]["positions"]["xs"][vertex] + 1
    )
    assert projected["body"]["rbfb1.0"]["solvers"][-1]["radius"] == pytest.approx(
        graphs["body"]["rbfb1.0"]["solvers"][-1]["radius"] + 0.25
    )
    np.testing.assert_array_equal(
        projected["body"]["geom1.1"]["meshes"][0]["positions"]["xs"],
        graphs["body"]["geom1.1"]["meshes"][0]["positions"]["xs"],
    )


def test_document_split_projects_edits_from_native_graph_without_cache(graphs, combined, tmp_path):
    from character_dna.bindings import dna
    from character_dna.dna_io import document, get_dna_reader, metadata, release_dna_handle
    from character_dna.dna_io.coordinates import configuration

    native, _indices = combined
    source = tmp_path / "source.dna"
    source.write_bytes(merged_graph.to_bytes(native))
    output = tmp_path / "edited.dna"
    document.merge_files(
        {kind: TEST_DNA_FOLDER / "ada" / f"{kind}.dna" for kind in graphs}, tmp_path / "compact.dna", "NativeRoundTrip"
    )
    source = tmp_path / "compact.dna"
    payload = document._payload(source)
    # Simulate an external SDK edit in the file's canonical basis. The addon's
    # Blender-space adapter would also re-express neutral Euler rotations.
    input_stream = dna.FileStream(str(source), dna.AccessMode_Read, dna.OpenMode_Binary, None)
    reader = dna.BinaryStreamReader(input_stream, configuration(blender=False), None)
    reader.read()
    output_stream = dna.FileStream(str(output), dna.AccessMode_Write, dna.OpenMode_Binary, None)
    writer = dna.BinaryStreamWriter(output_stream)
    try:
        writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
        metadata.write(writer, reader, payload)
        radius = reader.getRBFSolverRadius(reader.getRBFSolverCount() - 1) + 0.25
        writer.setRBFSolverRadius(reader.getRBFSolverCount() - 1, radius)
        writer.write()
        assert dna.Status.isOk()
    finally:
        for handle in (writer, reader, input_stream, output_stream):
            release_dna_handle(handle)
    document._read_cache.clear()
    outputs = document.split_files(output, tmp_path / "split")
    assert merged_graph.digest(merged_graph.from_bytes(outputs["head"].read_bytes())) == merged_graph.digest(
        graphs["head"]
    )
    reader = get_dna_reader(outputs["body"])
    try:
        assert reader.getRBFSolverRadius(reader.getRBFSolverCount() - 1) == pytest.approx(radius)
    finally:
        release_dna_handle(reader)


def test_native_evaluation_matches_components_at_every_lod(graphs, combined, tmp_path):
    from character_dna.bindings import dna, riglogic
    from character_dna.dna_io import get_dna_reader, release_dna_handle

    native, info = combined
    path = tmp_path / "evaluation.dna"
    path.write_bytes(merged_graph.to_bytes(native))
    readers = {kind: get_dna_reader(TEST_DNA_FOLDER / "ada" / f"{kind}.dna") for kind in graphs}
    readers["merged"] = get_dna_reader(path)
    configuration = riglogic.Configuration()
    configuration.rotationType = riglogic.RotationType_EulerAngles
    managers = {kind: riglogic.RigLogic(reader, configuration, None) for kind, reader in readers.items()}
    instances = {kind: riglogic.RigInstance(manager, None) for kind, manager in managers.items()}
    try:
        for posed in (False, True):
            for kind in graphs:
                reader = readers[kind]
                values = [
                    1.0
                    if reader.getRawControlName(index).endswith(".qw")
                    else (0.0 if ".q" in reader.getRawControlName(index) or not posed else 0.35)
                    for index in range(reader.getRawControlCount())
                ]
                if posed:
                    for solver in range(reader.getRBFSolverCount()):
                        indices = reader.getRBFSolverRawControlIndices(solver)
                        samples = reader.getRBFSolverRawControlValues(solver)
                        for index, value in zip(indices, samples[-len(indices) :], strict=True):
                            values[index] = float(value)
                for index, value in enumerate(values):
                    instances[kind].setRawControl(index, value)
                    instances["merged"].setRawControl(info["maps"][kind]["raw"][index], value)
            for lod in range(8):
                for kind, instance in instances.items():
                    instance.setLOD(lod if kind == "merged" else info["lods"][kind][lod])
                    managers[kind].calculate(instance)
                for kind in graphs:
                    for method, domain in (
                        ("getJointOutputs", "attribute"),
                        ("getBlendShapeOutputs", "blend"),
                        ("getAnimatedMapOutputs", "animated"),
                    ):
                        expected = np.asarray(getattr(instances[kind], method)())
                        current = np.asarray(getattr(instances["merged"], method)())[info["maps"][kind][domain]]
                        np.testing.assert_allclose(
                            current, expected, atol=2e-5, err_msg=f"{kind}/{method}/LOD{lod}/posed={posed}"
                        )
        assert dna.Status.isOk()
    finally:
        for handles in (instances, managers, readers):
            for handle in handles.values():
                release_dna_handle(handle)


def test_split_corrective_dependency_evaluates_without_the_other_component(graphs, tmp_path):
    from character_dna.bindings import riglogic
    from character_dna.dna_io import get_dna_reader, release_dna_handle, shape_dependencies

    count = len(graphs["body"]["geom1.1"]["meshes"][0]["positions"]["xs"])
    delta = np.zeros((count, 3), dtype="<f4")
    delta[0, 0] = 0.01
    bridge = {
        "kind": "body",
        "mesh": 0,
        "source": {"kind": "head", "name": "head_lod0_mesh__brow_down_L"},
        "deltas": base64.b64encode(delta.tobytes()).decode(),
        "coordinates": "",
    }
    recipient = shape_dependencies._extend(graphs, "body", "head", [bridge])
    path = tmp_path / "recipient.dna"
    path.write_bytes(merged_graph.to_bytes(recipient))
    readers = [get_dna_reader(TEST_DNA_FOLDER / "ada" / "head.dna"), get_dna_reader(path)]
    config = riglogic.Configuration()
    config.rotationType = riglogic.RotationType_EulerAngles
    managers = [riglogic.RigLogic(reader, config, None) for reader in readers]
    instances = [riglogic.RigInstance(manager, None) for manager in managers]
    try:
        for reader, manager, instance in zip(readers, managers, instances, strict=True):
            for index in range(reader.getRawControlCount()):
                name = reader.getRawControlName(index)
                instance.setRawControl(index, 1.0 if name.endswith(".qw") else 0 if ".q" in name else 0.35)
            manager.calculate(instance)
        assert instances[0].getBlendShapeOutputs()[0] > 0
        assert instances[1].getBlendShapeOutputs()[0] == pytest.approx(instances[0].getBlendShapeOutputs()[0])
        assert list(readers[1].getBlendShapeTargetVertexIndices(0, 0)) == [0]
        assert list(readers[1].getBlendShapeTargetDeltaXs(0, 0)) == pytest.approx([1.0])
    finally:
        for handles in (instances, managers, readers):
            for handle in handles:
                release_dna_handle(handle)


def test_split_rbf_corrective_copies_solver_closure(graphs, tmp_path):
    from character_dna.bindings import riglogic
    from character_dna.dna_io import get_dna_reader, release_dna_handle, shape_dependencies

    originals = copy_graph(graphs)
    source = originals["head"]
    solver = source["rbfb1.0"]["solvers"][0]
    pose = source["rbfe1.0"]["poses"][solver["poseIndices"][1]]
    channel = len(source["defn1.1"]["blendShapeChannelNames"])
    source["defn1.1"]["blendShapeChannelNames"].append("test_rbf_bridge")
    lods = merged_graph._lod_rows(source["defn1.1"]["lodBlendShapeMapping"], list(range(source["desc1.1"]["lodCount"])))
    source["defn1.1"]["lodBlendShapeMapping"] = merged_graph._lod_map([[*row, channel] for row in lods])
    table = source["bhvr1.1"]["blendShapeChannels"]
    table["inputIndices"].insert(0, pose["outputControlIndices"][0])
    table["outputIndices"].insert(0, channel)
    table["lods"] = [count + 1 for count in table["lods"]]
    count = len(graphs["body"]["geom1.1"]["meshes"][0]["positions"]["xs"])
    delta = np.zeros((count, 3), dtype="<f4")
    delta[0, 0] = 0.01
    bridge = {
        "kind": "body",
        "mesh": 0,
        "source": {"kind": "head", "name": "head_lod0_mesh__test_rbf_bridge"},
        "deltas": base64.b64encode(delta.tobytes()).decode(),
        "coordinates": "",
    }
    recipient = shape_dependencies._extend(originals, "body", "head", [bridge])
    readers, managers, states = [], [], []
    try:
        for kind, value in (("source", source), ("recipient", recipient)):
            path = tmp_path / f"{kind}.dna"
            path.write_bytes(merged_graph.to_bytes(value))
            reader = get_dna_reader(path)
            readers.append(reader)
            manager = riglogic.RigLogic(reader, riglogic.Configuration(), None)
            managers.append(manager)
            states.append(riglogic.RigInstance(manager, None))
        inputs = list(readers[0].getRBFSolverRawControlIndices(0))
        values = list(readers[0].getRBFSolverRawControlValues(0))[len(inputs) : 2 * len(inputs)]
        values_by_name = {
            readers[0].getRawControlName(index): value for index, value in zip(inputs, values, strict=True)
        }
        for reader, manager, state in zip(readers, managers, states, strict=True):
            for index in range(reader.getRawControlCount()):
                name = reader.getRawControlName(index)
                state.setRawControl(index, values_by_name.get(name, 1.0 if name.endswith(".qw") else 0.0))
            manager.calculate(state)
        expected = states[0].getBlendShapeOutputs()[channel]
        assert expected > 0.1
        assert states[1].getBlendShapeOutputs()[0] == pytest.approx(expected, abs=1e-6)
        assert readers[1].getRBFSolverCount() > len(graphs["body"]["rbfb1.0"]["solvers"])
    finally:
        for handles in (states, managers, readers):
            for handle in handles:
                release_dna_handle(handle)
