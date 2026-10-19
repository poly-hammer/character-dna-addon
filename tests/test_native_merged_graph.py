"""Native merged DNA retains every component and LOD without external files."""

import base64
import copy
import uuid

import numpy as np
import pytest

from character_dna.dna_io import merged_graph
from constants import TEST_DNA_FOLDER


@pytest.fixture(scope="module")
def graphs():
    return {
        kind: merged_graph.from_bytes((TEST_DNA_FOLDER / "ada" / f"{kind}.dna").read_bytes())
        for kind in ("head", "body")
    }


@pytest.fixture(scope="module")
def combined(graphs):
    graph, info = merged_graph.compose(graphs, "NativeRoundTrip")
    content = merged_graph.to_bytes(graph)
    assert content.startswith(b"DNA")
    return merged_graph.from_bytes(content), info


def test_sdk_json_interchange(graphs):
    for graph in graphs.values():
        assert merged_graph.to_bytes(graph).startswith(b"DNA")


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
    projected = merged_graph.project(native, graphs, info, "NativeRoundTrip")
    for kind in graphs:
        assert merged_graph.digest(projected[kind]) == merged_graph.digest(graphs[kind]), first_difference(
            projected[kind], graphs[kind]
        )


def first_difference(left, right, path=""):
    if isinstance(left, dict):
        for key in left:
            if key not in {"index", "size", "offset"} and (
                result := first_difference(left[key], right[key], path + "/" + key)
            ):
                return result
    elif isinstance(left, list):
        if len(left) != len(right):
            return path, len(left), len(right)
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            if result := first_difference(a, b, path + "/" + str(index)):
                return result
    elif left != right:
        return path, left, right
    return None


def test_native_inverse_uses_current_edits(graphs, combined):
    native, info = combined
    edited = copy.deepcopy(native)
    surface = info["surfaces"][0]
    vertex = next(index for index in range(len(surface["vertices"]["head"])) if index not in surface["seam"])
    native_vertex = surface["vertices"]["head"][vertex]
    edited["geom1.1"]["meshes"][surface["mesh"]]["positions"]["xs"][native_vertex] += 1.0
    edited["rbfb1.0"]["solvers"][-1]["radius"] += 0.25
    projected = merged_graph.project(edited, graphs, info, "NativeRoundTrip")
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

    native, indices = combined
    source = tmp_path / "source.dna"
    source.write_bytes(merged_graph.to_bytes(native))
    output = tmp_path / "edited.dna"
    parts = {kind: (TEST_DNA_FOLDER / "ada" / f"{kind}.dna").read_bytes() for kind in graphs}
    manifest = {
        "version": 1,
        "id": str(uuid.uuid4()),
        "name": "NativeRoundTrip",
        "partitions": {
            kind: {"id": kind, "db_name": graph["desc1.1"]["dbName"], "sha256": document.fingerprint(parts[kind])}
            for kind, graph in graphs.items()
        },
    }
    payload = {
        "schema": 2,
        "manifest": manifest,
        "indices": indices,
        "graph_digest": merged_graph.digest(native),
        "recovery": {kind: base64.b64encode(content).decode("ascii") for kind, content in parts.items()},
    }
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
    assert document.fingerprint(outputs["head"].read_bytes()) == document.fingerprint(parts["head"])
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
