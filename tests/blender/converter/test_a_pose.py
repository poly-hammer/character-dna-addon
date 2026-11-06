"""Pose normalization contracts, anatomical profiles and worker isolation."""

from types import SimpleNamespace

import bpy
import numpy as np
import pytest

from character_dna.dna_io import get_dna_reader, release_dna_handle
from character_dna.editors.converter import pose_contacts, pose_profiles, pose_receipt
from character_dna.editors.converter.pose_fit import canonical, geometry, quality, samples
from character_dna.editors.converter.pose_fit.model import PoseModel
from character_dna.editors.converter.pose_fit.solver import fit
from character_dna.editors.converter.pose_job import PoseJob
from character_dna.editors.shared import pose_transfer, topology_profiles
from character_dna.editors.shared.utilities import build_world_rest_pose


def toy():
    rng = np.random.default_rng(71)
    points = rng.normal(size=(60, 3)) * 0.01
    points[30:, 0] += 0.1
    bind = np.tile(np.eye(4), (2, 1, 1))
    bind[1, 0, 3] = 0.1
    weights = np.zeros((60, 2))
    weights[:30, 0], weights[30:, 1] = 1, 1
    return {
        "reference": points,
        "target": points.copy(),
        "weights": weights,
        "bind": bind,
        "names": np.array(["root", "limb"]),
        "parents": np.array([-1, 0]),
        "edges": np.array([[i, i + 1] for i in range(59)]),
        "triangles": np.array([[0, 1, 2], [30, 31, 32]]),
    }, {"id": "test", "joints": {"root": {"solve": False}, "limb": {"solve": True}}, "helpers": {}}


def test_exact_inverse_preserves_subject_shape_and_asymmetry():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(100, 3))
    weights = geometry.normalized_weights(rng.random((100, 2)))
    bind = np.tile(np.eye(4), (2, 1, 1))
    bind[1, :3, 3] = [0.2, -0.1, 0.4]
    posed = bind.copy()
    posed[0, :3, :3] = geometry.rotation(np.array([0.2, 0.1, -0.1]))
    posed[1, :3, :3] = geometry.rotation(np.array([-0.3, 0.2, 0.15]))
    transforms = geometry.blend(weights, posed, bind)
    target = geometry.apply(transforms, points)
    recovered, condition = geometry.inverse_skin(target, transforms)
    np.testing.assert_allclose(recovered, points, atol=1e-12)
    assert condition < 2
    wrong = geometry.apply(geometry.blend(weights, bind, posed), target)
    assert np.max(np.linalg.norm(wrong - points, axis=1)) > 0.001


def test_template_is_exact_noop_and_input_is_immutable():
    arrays, profile = toy()
    originals = {key: value.copy() for key, value in arrays.items()}
    result, report = fit(arrays, profile)
    assert report["status"] == "already_template"
    assert np.array_equal(result["positions"], arrays["reference"])
    assert result["positions"] is not arrays["target"]
    for key in arrays:
        assert np.array_equal(arrays[key], originals[key])


def test_known_rigid_limb_pose_recovers_surface_and_roll():
    arrays, profile = toy()
    posed = arrays["bind"].copy()
    posed[1, :3, :3] = geometry.rotation(np.array([0.15, -0.2, 0.3]))
    arrays["target"] = geometry.apply(geometry.blend(arrays["weights"], posed, arrays["bind"]), arrays["reference"])
    result, _report = fit(arrays, profile)
    np.testing.assert_allclose(result["positions"], arrays["reference"], atol=1e-8)
    np.testing.assert_allclose(result["joints"][:, :3, :3], arrays["bind"][:, :3, :3], atol=1e-10)


@pytest.mark.parametrize("corruption", ["nan", "empty_weights", "missing_joint", "missing_owner"])
def test_reject_invalid_pose_inputs(corruption):
    arrays, profile = toy()
    if corruption == "nan":
        arrays["target"][0, 0] = np.nan
    elif corruption == "empty_weights":
        arrays["weights"][0] = 0
    elif corruption == "missing_joint":
        profile["joints"]["absent"] = {"solve": True}
    else:
        del profile["joints"]["limb"]
    with pytest.raises(ValueError):
        fit(arrays, profile)


def test_inverse_rejects_singular_and_reflected_matrices():
    for value in (0, -1):
        transforms = np.eye(4)[None].copy()
        transforms[0, 0, 0] = value
        with pytest.raises(ValueError, match="unstable"):
            geometry.inverse_skin(np.zeros((1, 3)), transforms)


def test_sparse_barycentric_samples_validate_domain():
    ids, weights = samples.pack({"triangles": [[10, 20, 40]], "weights": [[0.2, 0.3, 0.5]]}, np.array([40, 10, 20]))
    assert ids.tolist() == [[1, 2, 0]]
    assert weights.tolist() == [[0.2, 0.3, 0.5]]
    with pytest.raises(ValueError, match="unused"):
        samples.pack({"vertices": [10, 30]}, np.array([10, 20]))


@pytest.mark.parametrize(
    "identifier,dna", [("metahuman", "default"), ("3d_scan_store/full_body", "3d_scan_store/full_body")]
)
def test_bundled_profiles_match_dna_and_cover_every_joint(identifier, dna):
    profile = pose_profiles.load(identifier)
    reader = get_dna_reader(topology_profiles.ROOT.parent / "dna" / dna / "body.dna")
    try:
        index = next(i for i in reader.getMeshIndicesForLOD(0) if reader.getMeshName(i) == profile["mesh"])
        assert topology_profiles.fingerprint(reader, index) == profile["topology"]
        names = [reader.getJointName(i) for i in range(reader.getJointCount())]
        assert set(names) == profile["joints"].keys() | profile["helpers"].keys()
        assert set(profile["helpers"].values()) <= profile["joints"].keys()
        # Landmark IDs are canonical DNA positions, including sparse layouts.
        from character_dna.dna_io import mesh_data

        ids = np.array(sorted({v for face in mesh_data.reader_faces(reader, index) for v in face}))
        packed = samples.landmarks(profile, names, ids)
        assert packed["landmarks"].size > 0
        assert np.all(packed["landmarks"] < len(ids))
    finally:
        release_dna_handle(reader)


def test_profile_path_cannot_escape_resources():
    with pytest.raises(ValueError):
        pose_profiles.load("../../../../../../etc")


def test_receipt_requires_exact_geometry_reference_and_profile():
    arrays, profile = toy()
    names = arrays["names"].tolist()
    pipeline = SimpleNamespace(
        pose_joints=dict(zip(names, arrays["bind"], strict=True)),
        pose_names=names,
        pose_reference={"body": {name: (arrays["bind"][i], 1) for i, name in enumerate(names)}},
        targets={"body": arrays["target"]},
        pose_profile=profile,
    )
    source = {}
    pose_receipt.write(pipeline, [source])
    assert pose_receipt.read(source, arrays["target"], arrays["bind"], profile) is not None
    moved = arrays["target"].copy()
    moved[0, 0] += 1e-5
    assert pose_receipt.read(source, moved, arrays["bind"], profile) is None
    assert pose_receipt.read(source, arrays["target"], arrays["bind"], {**profile, "id": "other"}) is None


def test_worker_roundtrip_cleanup_and_cancel():
    arrays, profile = toy()
    job = PoseJob(arrays, profile)
    folder = job.folder
    try:
        job.commit()
        np.testing.assert_array_equal(job.result["positions"], arrays["reference"])
    finally:
        job.close()
    assert not folder.exists()
    job = PoseJob(arrays, profile)
    process, folder = job.process, job.folder
    job.close()
    assert process.poll() is not None and not folder.exists()
    job.close()


def test_contacts_reject_new_crossing_but_allow_existing():
    before = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0], [0, 0, 1], [0.2, 0, 2], [-0.2, 0, 2]], dtype=float)
    after = before.copy()
    after[3:, 2] -= 1.5
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    with pytest.raises(ValueError, match="contacts"):
        pose_contacts.validate(before, after, triangles)
    report = pose_contacts.validate(after, after + np.array([0, 0, 0.1]), triangles)
    assert report["new_contact_pairs"] == 0


def test_shape_guard_rejects_collapsed_faces():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="collapsed"):
        quality.validate_surface(source, np.zeros_like(source), np.array([[0, 1, 2]]), np.array([[0, 1]]))


def test_canonical_projection_proves_repeat_noop_without_receipt():
    arrays, profile = toy()
    arrays["target"][30:] *= [1.06, 0.94, 1.02]
    model = PoseModel(arrays, profile)
    candidate, report = canonical.project(model, arrays["target"], arrays["edges"])
    assert report["after_degrees"] < canonical.FRAME_TOLERANCE
    result, repeat_report = fit({**arrays, "target": candidate}, profile)
    assert repeat_report["status"] == "already_neutral"
    np.testing.assert_array_equal(result["positions"], candidate)


def test_pose_option_defaults_off_and_roundtrips():
    from character_dna.editors.converter import templates
    from character_dna.editors.converter.properties import get_converter_properties

    previous = bpy.context.window.scene
    scene = bpy.data.scenes.new("A-pose option test")
    bpy.context.window.scene = scene
    try:
        properties = get_converter_properties()
        templates.apply_template(properties, templates.default_template())
        assert not properties.fit_to_a_pose
        properties.fit_to_a_pose = True
        recipe = templates.from_properties(properties)
        templates.apply_template(properties, recipe)
        assert properties.fit_to_a_pose
        assert pose_profiles.available(properties)
        recipe["fit_to_a_pose"] = "yes"
        with pytest.raises(ValueError, match="boolean"):
            templates.validate_template(recipe)
    finally:
        bpy.context.window.scene = previous
        bpy.data.scenes.remove(scene)


def test_pose_landmark_generation_uses_target_position_domain(tmp_path):
    resources = topology_profiles.ROOT.parent
    report = pose_transfer.generate(
        resources / "dna/default/body.dna", resources / "dna/3d_scan_store/full_body/body.dna", tmp_path, "generated"
    )
    assert report["samples"] > 0
    assert report["maximum_distance_m"] < 0.005
    assert (tmp_path / "pose.json").is_file()


def test_orientation_guard_detects_local_fold():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    transforms = np.tile(np.eye(4), (3, 1, 1))
    with pytest.raises(ValueError, match="inverted"):
        quality.validate_orientation(source, source[[0, 2, 1]], np.array([[0, 1, 2]]), transforms)


def test_shared_seam_correction_has_fixed_boundary_and_falloff():
    points = np.array([[x, y, 0.0] for y in range(2) for x in range(14)])
    edges = np.array(
        [[i, i + 1] for start in (0, 14) for i in range(start, start + 13)] + [[i, i + 14] for i in range(14)]
    )
    ids = np.array([0, 14])
    targets = points[ids] + [0, 0, 0.001]
    result = quality.seam_patch(points, edges, ids, targets)
    np.testing.assert_array_equal(result[ids], targets)
    np.testing.assert_array_equal(result[[13, 27]], points[[13, 27]])
    assert 0 < result[4, 2] < 0.001


@pytest.mark.parametrize(
    "identifier,dna", [("metahuman", "default"), ("3d_scan_store/full_body", "3d_scan_store/full_body")]
)
def test_actual_template_known_pose_recovery(identifier, dna):
    from character_dna.dna_io import mesh_data

    profile = pose_profiles.load(identifier)
    reader = get_dna_reader(topology_profiles.ROOT.parent / "dna" / dna / "body.dna")
    try:
        mesh = next(i for i in reader.getMeshIndicesForLOD(0) if reader.getMeshName(i) == profile["mesh"])
        faces = mesh_data.reader_faces(reader, mesh)
        ids = sorted({v for face in faces for v in face})
        local = {v: i for i, v in enumerate(ids)}
        names = [reader.getJointName(i) for i in range(reader.getJointCount())]
        parents = np.array(
            [reader.getJointParentIndex(i) if reader.getJointParentIndex(i) != i else -1 for i in range(len(names))]
        )
        bind = np.array(build_world_rest_pose(reader))
        bind[:, :3, 3] *= 0.01
        reference = (
            np.column_stack(
                [reader.getVertexPositionXs(mesh), reader.getVertexPositionYs(mesh), reader.getVertexPositionZs(mesh)]
            )[ids]
            * 0.01
        )
        weights = np.zeros((len(ids), len(names)))
        for i, vertex in enumerate(ids):
            weights[i, list(reader.getSkinWeightsJointIndices(mesh, vertex))] = reader.getSkinWeightsValues(
                mesh, vertex
            )
        weights = geometry.normalized_weights(weights)
        triangles = np.array(
            [[local[face[0]], local[face[k]], local[face[k + 1]]] for face in faces for k in range(1, len(face) - 1)]
        )
        edges = np.unique(
            np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1), axis=0
        )
        posed = bind.copy()
        for j, name in enumerate(names):
            if parents[j] >= 0:
                posed[j] = posed[parents[j]] @ np.linalg.inv(bind[parents[j]]) @ bind[j]
            if name.startswith(("upperarm_", "thigh_")) and name in profile["joints"]:
                posed[j, :3, :3] = geometry.rotation(np.array([0.05, 0.12, -0.08])) @ posed[j, :3, :3]
            if name in {"index_01_l", "thumb_01_r", "foot_l", "hand_r"}:
                posed[j, :3, :3] = geometry.rotation(np.array([0.15, -0.05, 0.1])) @ posed[j, :3, :3]
        arrays = {
            "reference": reference,
            "target": geometry.apply(geometry.blend(weights, posed, bind), reference),
            "bind": bind,
            "names": np.array(names),
            "parents": parents,
            "weights": weights,
            "edges": edges,
            "triangles": triangles,
        }
        result, report = fit(arrays, profile)
        errors = np.linalg.norm(result["positions"] - reference, axis=1)
        assert report["status"] == "rigid_control"
        assert np.sqrt(np.mean(errors**2)) < 0.0001
        assert errors.max() < 0.0005
        np.testing.assert_allclose(result["joints"][:, :3, :3], bind[:, :3, :3], atol=1e-6)
    finally:
        release_dna_handle(reader)
