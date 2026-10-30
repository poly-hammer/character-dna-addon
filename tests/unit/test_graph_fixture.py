"""Mutation isolation and exact scalar preservation for numerical fixture copies."""

import math

from fixtures.graphs import copy_graph, first_difference, restore_graph, snapshot_graph


def test_graph_copy_owns_nested_data_and_preserves_float_bits():
    values = [0.0, -0.0, 1.2345678901234567, 1e-310, 1e308]
    source = {"meshes": [{"positions": {"xs": values}, "indices": [0, 65535]}], "name": "Ada"}
    copied = copy_graph(source)
    assert copied == source
    assert [value.hex() for value in copied["meshes"][0]["positions"]["xs"]] == [value.hex() for value in values]
    copied["meshes"][0]["positions"]["xs"][1] = 2.0
    copied["meshes"][0]["indices"].append(10)
    copied["meshes"].append({})
    assert source["meshes"][0]["positions"]["xs"][1].hex() == (-0.0).hex()
    assert source["meshes"][0]["indices"] == [0, 65535]
    assert len(source["meshes"]) == 1


def test_graph_comparison_checks_tail_values_and_retains_metadata_rules():
    source = {"values": [float(index) for index in range(1000)], "index": {"offset": 10}}
    candidate = copy_graph(source)
    candidate["index"]["offset"] = 20
    assert first_difference(source, candidate) is None
    candidate["values"][-1] += 1e-10
    assert first_difference(source, candidate) == ("/values/999", 999.0, candidate["values"][-1])
    assert first_difference([{"size": 1, "value": 3}], [{"size": 2, "value": 3}]) is None
    assert first_difference([1, 2], [1]) == ("", 2, 1)


def test_graph_comparison_rejects_nan_even_when_the_same_object_is_shared():
    nan = float("nan")
    result = first_difference({"values": [[nan]]}, {"values": [[nan]]})
    assert result[0] == "/values/0/0"
    assert math.isnan(result[1]) and math.isnan(result[2])


def test_graph_archive_restores_independent_mutable_owners():
    original = {"weights": [0.1, 0.2], "channels": {"name": "left"}}
    archive = snapshot_graph(original)
    assert isinstance(archive, bytes)
    first, second = restore_graph(archive), restore_graph(archive)
    first["weights"].append(0.8)
    first["channels"]["name"] = "right"
    assert second == original
    assert snapshot_graph(second) == archive
    assert restore_graph(archive) == original
