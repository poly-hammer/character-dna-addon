"""Tests for the converter's headless data plumbing.

These exercise only the pure parsing helpers (CSV vertex->bone maps and the
centroid JSON). End-to-end framing, UV-wrapping and bone fitting need a live
Blender session with the rig template meshes and are covered by manual probes
under ``scratches/``."""

from __future__ import annotations

import pytest

from character_dna.editors.converter import core
from character_dna.editors.shared import bone_mapping


# ---------------------------------------------------------------------------
# vertex_bone_mapping CSV parsing
# ---------------------------------------------------------------------------


def test_load_vertex_bone_mapping_missing_returns_empty() -> None:
    """Probing an unknown mesh name yields two empty maps so callers can
    safely query any mesh without a guard."""
    surface, volumetric = bone_mapping.load_vertex_bone_mapping("does_not_exist_mesh")
    assert surface == {}
    assert volumetric == {}


def test_load_vertex_bone_mapping_parses_surface_and_volumetric(tmp_path, monkeypatch) -> None:
    """Two-field lines are surface joints; five-field lines are volumetric
    (vertex id + dx dy dz). Blank lines, comments and malformed rows are
    skipped without raising."""
    csv_path = tmp_path / "probe_mesh.csv"
    csv_path.write_text(
        "# comment line\n"
        "\n"
        "FACIAL_C_Jaw 1234\n"
        "FACIAL_L_Eye 42 0.5 -1.5 2.0\n"
        "FACIAL_BAD 1 2 3\n",  # malformed: 4 fields -> ignored
        encoding="utf-8",
    )
    monkeypatch.setattr(bone_mapping, "_VERTEX_BONE_MAPPING_DIR", tmp_path)
    bone_mapping._vertex_bone_mapping_cache.pop("probe_mesh", None)

    surface, volumetric = bone_mapping.load_vertex_bone_mapping("probe_mesh")

    assert surface == {"FACIAL_C_Jaw": 1234}
    assert volumetric == {"FACIAL_L_Eye": (42, (0.5, -1.5, 2.0))}


def test_load_vertex_bone_mapping_is_cached(tmp_path, monkeypatch) -> None:
    """The parsed result is cached per mesh name; deleting the file after
    the first read still returns the cached maps."""
    csv_path = tmp_path / "cached_mesh.csv"
    csv_path.write_text("FACIAL_C_Jaw 7\n", encoding="utf-8")
    monkeypatch.setattr(bone_mapping, "_VERTEX_BONE_MAPPING_DIR", tmp_path)
    bone_mapping._vertex_bone_mapping_cache.pop("cached_mesh", None)

    first, _ = bone_mapping.load_vertex_bone_mapping("cached_mesh")
    csv_path.unlink()
    second, _ = bone_mapping.load_vertex_bone_mapping("cached_mesh")

    assert first == second == {"FACIAL_C_Jaw": 7}


def test_real_head_mapping_is_surface_dominant() -> None:
    """The shipped head CSV is the surface vertex->bone map; every value is
    a non-negative vertex id."""
    surface, _ = bone_mapping.load_vertex_bone_mapping("head_lod0_mesh")
    assert surface, "expected the shipped head_lod0_mesh.csv to be present"
    assert all(isinstance(v, int) and v >= 0 for v in surface.values())


# ---------------------------------------------------------------------------
# centroid_bones_mapping per-source-mesh files
# ---------------------------------------------------------------------------

# Source meshes that may appear in the shipped centroid candidates. Only the
# head and body JSON files ship today; the rest are recognized so a base DNA can
# add them later without a code change.
_EXPECTED_HEAD_SOURCE_MESHES = {
    "head_lod0_mesh",
    "teeth_lod0_mesh",
    "eyeLeft_lod0_mesh",
    "eyeRight_lod0_mesh",
    "body_lod0_mesh",
}


def test_default_centroid_source_precedence_chain() -> None:
    """The shipped default precedence puts specialized teeth / eye meshes first,
    then the body mesh, then the head fallback. The converter overrides this at
    runtime with the user's Extra Meshes order."""
    assert bone_mapping.DEFAULT_CENTROID_SOURCE_PRECEDENCE == (
        "teeth_lod0_mesh",
        "eyeLeft_lod0_mesh",
        "eyeRight_lod0_mesh",
        "body_lod0_mesh",
        "head_lod0_mesh",
    )


def test_load_centroid_bone_mapping_head_schema() -> None:
    """Every centroid joint maps to a non-empty list of candidate entries, each
    exposing the documented schema and drawing from one of the known source
    meshes."""
    mapping = core.load_centroid_bone_mapping("head")
    assert mapping, "expected head centroid entries"
    for joint_name, entries in mapping.items():
        assert isinstance(entries, list) and entries, joint_name
        for entry in entries:
            assert set(entry) == {"source_mesh", "vertices", "offset"}, joint_name
            assert entry["source_mesh"] in _EXPECTED_HEAD_SOURCE_MESHES, joint_name
            assert isinstance(entry["vertices"], list)
            assert len(entry["offset"]) == 3


def test_load_centroid_bone_mapping_candidates_have_distinct_sources() -> None:
    """A joint never lists the same source mesh twice; the placement precedence
    among them is applied at fit time, not at load time."""
    for joint_name, entries in core.load_centroid_bone_mapping().items():
        sources = [entry["source_mesh"] for entry in entries]
        assert len(sources) == len(set(sources)), joint_name


def test_load_centroid_bone_mapping_is_component_independent() -> None:
    """Centroid candidates are discovered from every shipped source-mesh JSON,
    so the mapping no longer depends on the component argument."""
    head = core.load_centroid_bone_mapping("head")
    body = core.load_centroid_bone_mapping("body")
    assert head is body


def test_load_centroid_bone_mapping_unknown_component_returns_global() -> None:
    """An unrecognized component argument is ignored; the global mapping is
    returned rather than an empty dict."""
    assert core.load_centroid_bone_mapping("not_a_component") is core.load_centroid_bone_mapping("head")  # type: ignore[arg-type]


def test_load_centroid_bone_mapping_is_cached() -> None:
    assert core.load_centroid_bone_mapping("head") is core.load_centroid_bone_mapping("head")


# ---------------------------------------------------------------------------
# joint_name_to_index
# ---------------------------------------------------------------------------


class _FakeReader:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def getJointCount(self) -> int:
        return len(self._names)

    def getJointName(self, i: int) -> str:
        return self._names[i]


def test_joint_name_to_index_maps_every_joint() -> None:
    reader = _FakeReader(["FACIAL_C_FacialRoot", "FACIAL_C_Jaw", "FACIAL_L_Eye"])
    assert bone_mapping.joint_name_to_index(reader) == {  # type: ignore[arg-type]
        "FACIAL_C_FacialRoot": 0,
        "FACIAL_C_Jaw": 1,
        "FACIAL_L_Eye": 2,
    }


# ---------------------------------------------------------------------------
# mapped_joint_names + no-mapping reporting
# ---------------------------------------------------------------------------


def test_mapped_joint_names_unions_csv_and_centroid() -> None:
    """A joint counts as mapped if it appears in any surface / volumetric CSV
    or has a centroid candidate."""
    names = bone_mapping.mapped_joint_names("head")
    assert names, "expected the shipped head mappings to yield mapped joints"

    surface, _ = bone_mapping.load_vertex_bone_mapping("head_lod0_mesh")
    assert surface, "expected the shipped head surface CSV to be present"
    assert set(surface).issubset(names)
    assert set(core.load_centroid_bone_mapping("head")).issubset(names)


def test_mapped_joint_names_is_component_independent() -> None:
    """Mapped joints are discovered from every shipped source-mesh file, so the
    result no longer depends on the component argument."""
    assert bone_mapping.mapped_joint_names("head") == bone_mapping.mapped_joint_names("not_a_component")  # type: ignore[arg-type]


def test_report_unmapped_joints_logs_only_unmapped(caplog) -> None:
    """Every DNA joint without a mapping is reported once; mapped joints stay
    silent."""
    mapped = bone_mapping.mapped_joint_names("head")
    a_mapped_joint = next(iter(mapped))

    with caplog.at_level("ERROR", logger=core.logger.name):
        core._report_unmapped_joints("head", [a_mapped_joint, "TOTALLY_FAKE_JOINT"])

    messages = [record.getMessage() for record in caplog.records]
    assert any("TOTALLY_FAKE_JOINT" in message for message in messages)
    assert not any(a_mapped_joint in message for message in messages)


# ---------------------------------------------------------------------------
# joints_mapped_by_meshes (head-vs-body reconcile authority)
# ---------------------------------------------------------------------------


def test_joints_mapped_by_meshes_selects_head_only_shared_bones() -> None:
    """The shared neck / clavicle_out / head joints are mapped on the head mesh
    only, so the head-vs-body difference selects exactly the joints the head fit
    is authoritative for during reconcile -- and excludes body-owned joints."""
    head_joints = bone_mapping.joints_mapped_by_meshes({"head_lod0_mesh"})
    body_joints = bone_mapping.joints_mapped_by_meshes({"body_lod0_mesh"})
    head_authoritative = head_joints - body_joints

    for joint in ("neck_01", "neck_02", "head", "clavicle_out_l", "clavicle_out_r"):
        assert joint in head_joints, f"{joint} should be mapped on head_lod0_mesh"
        assert joint not in body_joints, f"{joint} must not be mapped on body_lod0_mesh"
        assert joint in head_authoritative

    # Body-owned shared joints must stay body-authoritative (not in the set).
    assert "spine_04" in body_joints
    assert "spine_04" not in head_authoritative


# ---------------------------------------------------------------------------
# relative fit (surface-frame binding)
# ---------------------------------------------------------------------------


def _unit_quad():
    """A flat 2x2 quad in the XY plane (two triangles) as (positions, triangles)."""
    import numpy as np

    positions = np.array(
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)],
        dtype=np.float64,
    )
    triangles = np.array([(0, 1, 2), (0, 2, 3)], dtype=np.int64)
    return positions, triangles


def test_surface_frame_binding_identity_returns_input() -> None:
    """Reconstructing against the unchanged reference reproduces the dependent's
    bind positions exactly (including its off-surface offset)."""
    import numpy as np

    from character_dna.editors.converter import core

    positions, triangles = _unit_quad()
    # Two dependent points hovering 0.5 above the quad.
    dependent = np.array([(0.5, 0.5, 0.5), (1.5, 1.0, 0.5)], dtype=np.float64)

    binding = core.build_surface_frame_binding(positions, triangles, dependent)
    result = core.apply_surface_frame_binding(binding, positions)

    assert result == pytest.approx(dependent, abs=1e-9)
    assert not binding.missed.any()


def test_surface_frame_binding_follows_translation() -> None:
    """Translating the whole reference translates the dependent by the same
    amount while preserving its offset."""
    import numpy as np

    from character_dna.editors.converter import core

    positions, triangles = _unit_quad()
    dependent = np.array([(0.5, 0.5, 0.5), (1.5, 1.0, 0.5)], dtype=np.float64)
    binding = core.build_surface_frame_binding(positions, triangles, dependent)

    translation = np.array([3.0, -1.0, 2.0])
    result = core.apply_surface_frame_binding(binding, positions + translation)

    assert result == pytest.approx(dependent + translation, abs=1e-9)


def test_surface_frame_binding_follows_rotation() -> None:
    """Rotating the reference 90 degrees about X rotates the dependent (offset
    and all) with the surface frame."""
    import numpy as np

    from character_dna.editors.converter import core

    positions, triangles = _unit_quad()
    dependent = np.array([(0.5, 0.5, 0.5), (1.5, 1.0, 0.5)], dtype=np.float64)
    binding = core.build_surface_frame_binding(positions, triangles, dependent)

    # +90 deg about X: (x, y, z) -> (x, -z, y).
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    result = core.apply_surface_frame_binding(binding, positions @ rotation.T)

    assert result == pytest.approx(dependent @ rotation.T, abs=1e-9)


def test_surface_frame_binding_flags_far_vertices() -> None:
    """A dependent vertex beyond ``max_distance`` is flagged missed and keeps its
    bind position on reconstruction."""
    import numpy as np

    from character_dna.editors.converter import core

    positions, triangles = _unit_quad()
    dependent = np.array([(0.5, 0.5, 0.1), (0.5, 0.5, 5.0)], dtype=np.float64)
    binding = core.build_surface_frame_binding(positions, triangles, dependent, max_distance=1.0)

    assert not binding.missed[0]
    assert binding.missed[1]
    # Move the reference; the missed vertex stays put, the bound one follows.
    result = core.apply_surface_frame_binding(binding, positions + np.array([10.0, 0.0, 0.0]))
    assert result[1] == pytest.approx(dependent[1], abs=1e-9)
    assert result[0][0] == pytest.approx(dependent[0][0] + 10.0, abs=1e-9)


def test_surface_frame_binding_merges_multiple_references() -> None:
    """Two reference meshes merged into one surface bind each dependent vertex to
    its nearest reference; moving one reference moves only the vertices bound to
    it (e.g. an eyelash spanning both eyes following each eye independently)."""
    import numpy as np

    from character_dna.editors.converter import core

    # Two quads far apart in X, concatenated into one reference soup (as the
    # converter does for multiple reference meshes).
    quad, tris = _unit_quad()
    positions = np.concatenate([quad, quad + np.array([10.0, 0.0, 0.0])], axis=0)
    triangles = np.concatenate([tris, tris + len(quad)], axis=0)
    # One dependent vertex over each quad.
    dependent = np.array([(0.5, 0.5, 0.5), (10.5, 0.5, 0.5)], dtype=np.float64)

    binding = core.build_surface_frame_binding(positions, triangles, dependent)

    # Move only the second quad (+Y); the first dependent stays, the second follows.
    moved = positions.copy()
    moved[len(quad) :] += np.array([0.0, 3.0, 0.0])
    result = core.apply_surface_frame_binding(binding, moved)

    assert result[0] == pytest.approx(dependent[0], abs=1e-9)
    assert result[1] == pytest.approx(dependent[1] + np.array([0.0, 3.0, 0.0]), abs=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
