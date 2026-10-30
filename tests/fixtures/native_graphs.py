"""Immutable native graph artifacts, independently restored by each module."""

import pytest

from constants import TEST_DNA_FOLDER
from fixtures.graphs import snapshot_graph


@pytest.fixture(scope="session")
def native_graph_archive(addon):
    from character_dna.dna_io import merged_graph

    graphs = {
        kind: merged_graph.from_bytes((TEST_DNA_FOLDER / "ada" / f"{kind}.dna").read_bytes())
        for kind in ("head", "body")
    }
    native, indices = merged_graph.compose(graphs, "NativeRoundTrip")
    content = merged_graph.to_bytes(native)
    assert content.startswith(b"DNA")
    # Only immutable bytes survive at session scope. Test modules own their
    # decoded containers, and native readers/writers remain operation-scoped.
    return snapshot_graph(graphs), snapshot_graph((merged_graph.from_bytes(content), indices))
