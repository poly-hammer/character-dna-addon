"""End-to-end coverage for the Mesh-to-DNA Converter operator.

Mirrors the legacy ``test_convert_to_dna`` round trip, updated for the converter
editor: the operator now takes the head/body mesh pointers directly and fits the
chosen base DNA onto them. Running headless (``bpy.app.background``) the operator
takes its synchronous path, so the settle animation is committed immediately."""

import json
import shutil

from pathlib import Path

import bpy
import pytest

from character_dna.editors.converter.properties import get_converter_properties
from character_dna.editors.converter.templates import apply_template, default_template
from constants import TEST_DNA_FOLDER


@pytest.fixture
def conversion_diagnostics(monkeypatch):
    """Observe export validation in memory without creating a report file."""
    from character_dna.editors.converter.lifecycle import ConversionLifecycle

    diagnostics = {}
    verify_export = ConversionLifecycle._verify_export

    def capture_validation(lifecycle, folder):
        verify_export(lifecycle, folder)
        diagnostics.update(lifecycle._pipeline.report)

    monkeypatch.setattr(ConversionLifecycle, "_verify_export", capture_validation)
    return diagnostics


@pytest.fixture
def base_dna_folder(temp_folder: Path):
    """Copy base DNA outside the addon to exercise custom template links."""
    folder = temp_folder / "converter_e2e"
    folder.mkdir(parents=True, exist_ok=True)
    for component in ("head", "body"):
        shutil.copy2(TEST_DNA_FOLDER / "ada" / f"{component}.dna", folder / f"{component}.dna")
    try:
        yield folder
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def configure_template(props, folder, preset="basic", components=("head", "body")):
    recipe = default_template(preset)
    recipe["dna_files"] = {kind: str(folder / f"{kind}.dna") for kind in components}
    recipe["meshes"] = [row for row in recipe["meshes"] if row["component"] in components]
    # Head-only fixtures exercise fitting/publication without a body pose model.
    recipe["normalize_to_a_pose"] = "body" in components
    apply_template(props, recipe)
    for item in props.extra_meshes:
        item.scene_object = bpy.data.objects.get(item.dna_mesh_name) if item.role in components else None
    return {item.role: item for item in props.extra_meshes}


def test_convert_meshes_to_dna(  # noqa: PLR0915 - complete conversion contract
    load_mhc_conformed_topology_meshes,
    base_dna_folder,
    temp_folder: Path,
    monkeypatch: pytest.MonkeyPatch,
    conversion_diagnostics,
):
    name = "TestMetaHuman01"
    output_folder = temp_folder / "converted_dna"
    output_folder.mkdir(parents=True, exist_ok=True)

    bpy.context.scene.unit_settings.scale_length = 1.0

    properties = get_converter_properties(bpy.context)
    rows = configure_template(properties, base_dna_folder)
    monkeypatch.setenv("CHARACTER_DNA_TEST_BASE", str(base_dna_folder))
    for component in properties.dna_components:
        component.filepath = "{CHARACTER_DNA_TEST_BASE}/" + component.component + ".dna"
    properties.new_name = name
    properties.new_folder = str(output_folder)
    properties.validate_uvs = False
    # This contract copies the already-conformed observations without changing
    # their pose. Pose normalization has its own end-to-end contracts below.
    properties.normalize_to_a_pose = False
    properties.constrain_head_to_body = True

    # Template data must stay native until the final scene is created.
    from character_dna.components.base import CharacterComponentBase
    from character_dna.editors.converter.native.direct import DirectPipeline
    from character_dna.editors.converter.pipeline import Transition, positions
    from character_dna.runtime import controller, engine

    commit_transition = Transition.commit
    input_positions = {obj: positions(obj) for obj in (rows["head"].scene_object, rows["body"].scene_object)}
    phases = []

    def unexpected_texture_loading(*_args):
        pytest.fail("Converter imports must not load texture maps from folders")

    monkeypatch.setattr(CharacterComponentBase, "_set_image_textures", unexpected_texture_loading)

    def commit_with_native_bindings(transition):
        controller.reconcile()
        instance = bpy.context.scene.character_dna.rig_instance_list.get(name)
        assert engine.active(instance)
        if bpy.app.version < (5, 0, 0):
            assert instance.face_board.data.users > 1
        phases.append(transition.label)
        commit_transition(transition)

    monkeypatch.setattr(Transition, "commit", commit_with_native_bindings)
    create_scene = DirectPipeline._create_scene
    scene_commits = []

    def create_final_scene(pipeline):
        assert pipeline.geometry_complete
        assert all((pipeline.operator._staging_folder / f"{kind}.dna").is_file() for kind in ("head", "body"))
        assert not any(obj.name.startswith(name + "_") for obj in bpy.data.objects)
        assert pipeline.native_joint_results["reconciled_joints"]
        create_scene(pipeline)
        scene_commits.append(True)

    monkeypatch.setattr(DirectPipeline, "_create_scene", create_final_scene)
    result = bpy.ops.character_dna.convert_to_dna()  # type: ignore[attr-defined]
    assert result == {"FINISHED"}, "Conversion operator should finish successfully"
    import numpy as np

    for obj, before in input_positions.items():
        np.testing.assert_array_equal(positions(obj), before)
        assert obj.hide_get() and obj.hide_render
    assert phases == []
    assert scene_commits == [True]
    assert not list(temp_folder.rglob("ConversionReport.json"))
    report = conversion_diagnostics
    assert report["models"] == []
    assert report["meshes"]["head"]["export_max_error_m"] < 2e-6
    assert report["meshes"]["body"]["export_max_error_m"] < 2e-6

    instance = bpy.context.scene.character_dna.rig_instance_list.get(name)  # type: ignore[attr-defined]
    assert instance is not None, "Rig instance should be created"
    assert instance.name == name, f"Instance name should be {name}"
    eye_edge = bpy.data.objects[f"{name}_eyeEdge_lod0_mesh"]
    assert eye_edge.hide_viewport and eye_edge.hide_get() and eye_edge.hide_render
    instance.view_options.active_lod = "lod0"
    assert eye_edge.hide_viewport and eye_edge.hide_get() and eye_edge.hide_render

    assert instance.head_rig is not None, "Head rig should be created"
    assert instance.head_mesh is not None, "Head mesh should be created"
    assert instance.head_dna_file_path, "Head DNA file path should be set"

    assert instance.body_rig is not None, "Body rig should be created"
    assert instance.body_mesh is not None, "Body mesh should be created"
    assert instance.body_dna_file_path, "Body DNA file path should be set"

    # The converter should write an ExportManifest.json alongside the DNA, like
    # the MetaHuman Creator DCC export.
    manifest_file = output_folder / "ExportManifest.json"
    assert manifest_file.exists(), "ExportManifest.json should be written alongside the DNA"
    manifest = json.loads(manifest_file.read_text())
    assert manifest["metaHumanName"] == name, "Manifest should record the converted MetaHuman name"

    # ``zero_shape_deltas`` defaults on, so the converted head DNA should carry no
    # blend shape deltas even though the base DNA does.
    from character_dna.dna_io import get_dna_reader

    def _blend_shape_delta_count(dna_path: Path) -> int:
        reader = get_dna_reader(dna_path)
        assert reader is not None, f"Should be able to read {dna_path}"
        return sum(
            len(reader.getBlendShapeTargetVertexIndices(mesh_index, target_index))
            for mesh_index in range(reader.getMeshCount())
            for target_index in range(reader.getBlendShapeTargetCount(mesh_index))
        )

    assert _blend_shape_delta_count(base_dna_folder / "head.dna") > 0, (
        "The base head DNA should ship blend shape deltas for this test to be meaningful"
    )
    assert _blend_shape_delta_count(output_folder / "head.dna") == 0, (
        "zero_shape_deltas should clear every blend shape delta in the converted head DNA"
    )


def test_template_pose_normalization_roundtrip(
    load_mhc_conformed_topology_meshes, base_dna_folder, tmp_path, conversion_diagnostics
):
    import numpy as np

    from character_dna.dna_io import get_dna_reader, release_dna_handle
    from character_dna.dna_io.mesh_data import reader_faces
    from character_dna.editors.converter.pipeline import positions, write_positions

    props = get_converter_properties()
    rows = configure_template(props, base_dna_folder, components=("body",))
    body = rows["body"].scene_object
    reader = get_dna_reader(base_dna_folder / "body.dna")
    try:
        index = next(i for i in reader.getMeshIndicesForLOD(0) if reader.getMeshName(i) == "body_lod0_mesh")
        ids = sorted({v for face in reader_faces(reader, index) for v in face})
        reference = (
            np.column_stack(
                (
                    reader.getVertexPositionXs(index),
                    reader.getVertexPositionYs(index),
                    reader.getVertexPositionZs(index),
                )
            )[ids]
            * 0.01
        )
        write_positions(body, reference)
    finally:
        release_dna_handle(reader)
    original = positions(body)
    props.new_name = "PoseNormalizationRoundtrip"
    props.new_folder = str(tmp_path)
    props.validate_uvs = False
    assert props.normalize_to_a_pose
    assert bpy.ops.character_dna.convert_to_dna() == {"FINISHED"}
    assert conversion_diagnostics["a_pose"]["status"] == "already_template"
    assert conversion_diagnostics["meshes"]["body"]["export_max_error_m"] < 2e-6
    np.testing.assert_array_equal(positions(body), original)
    assert (tmp_path / "body.dna").is_file()
    assert not list(tmp_path.rglob("ConversionReport.json"))


def test_known_limb_pose_full_operator(
    load_mhc_conformed_topology_meshes, base_dna_folder, tmp_path, conversion_diagnostics
):
    """Include the established nontrivial pose workload in click-to-ready timing."""
    import numpy as np

    from character_dna.bindings import load_converter_core
    from character_dna.editors.converter.pipeline import positions, write_positions
    from character_dna.editors.converter.pose import profiles
    from character_dna.editors.converter.pose.fit import geometry

    props = get_converter_properties()
    rows = configure_template(props, base_dna_folder, components=("body",))
    native = load_converter_core()
    template = native.DnaTemplate(str(base_dna_folder / "body.dna"))
    info = json.loads(template.info())
    name = rows["body"].dna_mesh_name
    names, parents = info["names"], info["parents"]
    bind = np.asarray([info["states"][name]["matrix"] for name in names]).reshape(-1, 4, 4)
    reference = np.frombuffer(template.mesh(name)["positions"], dtype=np.float32).reshape(-1, 3).astype(float)
    weights = np.frombuffer(template.weights(name, names), dtype=np.float64).reshape(-1, len(names))
    weights = geometry.normalized_weights(weights)
    profile = profiles.load("metahuman")
    posed = bind.copy()
    for index, name in enumerate(names):
        parent = parents[index]
        if parent >= 0:
            posed[index] = posed[parent] @ np.linalg.inv(bind[parent]) @ bind[index]
        if name.startswith(("upperarm_", "thigh_")) and name in profile["joints"]:
            posed[index, :3, :3] = geometry.rotation(np.array([0.05, 0.12, -0.08])) @ posed[index, :3, :3]
        if name in {"index_01_l", "thumb_01_r", "foot_l", "hand_r"}:
            posed[index, :3, :3] = geometry.rotation(np.array([0.15, -0.05, 0.1])) @ posed[index, :3, :3]
    body = rows["body"].scene_object
    write_positions(body, geometry.apply(geometry.blend(weights, posed, bind), reference))
    original = positions(body)
    props.new_name, props.new_folder = "KnownLimbPose", str(tmp_path)
    props.validate_uvs = False
    assert props.normalize_to_a_pose
    assert bpy.ops.character_dna.convert_to_dna() == {"FINISHED"}
    instance = bpy.context.scene.character_dna.rig_instance_list[props.new_name]
    error = np.linalg.norm(positions(instance.body_mesh) - reference, axis=1)
    assert conversion_diagnostics["a_pose"]["status"] == "rigid_control"
    assert np.sqrt(np.mean(error**2)) < 1e-4
    assert error.max() < 5e-4
    assert conversion_diagnostics["meshes"]["body"]["export_max_error_m"] < 2e-6
    np.testing.assert_array_equal(positions(body), original)
    assert (tmp_path / "body.dna").is_file()


def test_joined_input_conversion_preserves_each_component_domain(
    load_mhc_conformed_topology_meshes, base_dna_folder, tmp_path, conversion_diagnostics
):
    import numpy as np

    from character_dna.dna_io import get_dna_reader, release_dna_handle
    from character_dna.editors.converter import assembly
    from character_dna.editors.converter.pipeline import positions

    props = get_converter_properties()
    rows = configure_template(props, base_dna_folder)
    sources = {kind: rows[kind].scene_object for kind in ("head", "body")}
    original = {kind: positions(obj) for kind, obj in sources.items()}
    readers = {kind: get_dna_reader(base_dna_folder / f"{kind}.dna") for kind in sources}
    try:
        domain = assembly.prepare(sources, readers, {kind: rows[kind].dna_mesh_name for kind in sources})
    finally:
        for reader in readers.values():
            release_dna_handle(reader)
    joined = bpy.data.objects.new("Joined supplied character", domain.mesh)
    bpy.context.scene.collection.objects.link(joined)
    before = positions(joined)
    expected = {kind: before[ids].copy() for kind, ids in domain.component_to_assembly.items()}
    try:
        props.input_layout = "JOINED"
        props.joined_mesh = joined
        for kind in sources:
            rows[kind].scene_object = None
        props.normalize_to_a_pose = False
        props.adjust_seam = False
        props.validate_uvs = False
        props.new_name = "JoinedInputRoundtrip"
        props.new_folder = str(tmp_path)
        assert bpy.ops.character_dna.convert_to_dna() == {"FINISHED"}
        instance = bpy.context.scene.character_dna.rig_instance_list.get(props.new_name)
        assert instance is not None
        for kind, source in sources.items():
            np.testing.assert_allclose(positions(getattr(instance, f"{kind}_mesh")), expected[kind], atol=2e-6, rtol=0)
            np.testing.assert_array_equal(positions(source), original[kind])
            assert conversion_diagnostics["meshes"][kind]["binding"] == "full_body"
            assert conversion_diagnostics["meshes"][kind]["export_max_error_m"] < 2e-6
            assert (tmp_path / f"{kind}.dna").is_file()
        np.testing.assert_array_equal(positions(joined), before)
        assert joined.hide_get() and joined.hide_render
        assert conversion_diagnostics["assembly"]["seam_pairs"] == len(domain.seam)
    finally:
        props.joined_mesh = None
        bpy.data.objects.remove(joined, do_unlink=True)
        domain.close()


def test_pose_quality_failure_rolls_back_conversion(load_mhc_conformed_topology_meshes, base_dna_folder, tmp_path):
    import numpy as np

    from character_dna.editors.converter.pipeline import positions

    props = get_converter_properties()
    rows = configure_template(props, base_dna_folder)
    props.new_name = "RejectedPose"
    props.new_folder = str(tmp_path)
    props.validate_uvs = False
    assert props.normalize_to_a_pose
    before = {row.scene_object: positions(row.scene_object) for role, row in rows.items() if role in {"head", "body"}}
    (tmp_path / "head.dna").write_bytes(b"previous output")
    # These conformed observations cannot be normalized by the current solver
    # without a local fold. Keep the guard, and prove it fails transactionally.
    with pytest.raises(RuntimeError, match="inverted a local surface triangle"):
        bpy.ops.character_dna.convert_to_dna()
    for obj, original in before.items():
        np.testing.assert_array_equal(positions(obj), original)
        assert not obj.hide_get()
    assert (tmp_path / "head.dna").read_bytes() == b"previous output"
    assert not bpy.context.scene.character_dna.rig_instance_list.get("RejectedPose")


def test_runtime_initialization_failure_restores_scene_and_published_files(
    load_mhc_conformed_topology_meshes, base_dna_folder, tmp_path, monkeypatch
):
    from character_dna.rig_instance import RigInstance

    props = get_converter_properties()
    configure_template(props, base_dna_folder, components=("head",))
    props.new_name = "RejectedRuntime"
    props.new_folder = str(tmp_path)
    output = tmp_path / "head.dna"
    output.write_bytes(b"previous accepted output")
    evaluate = RigInstance.evaluate

    def reject(instance, *args, **kwargs):
        if instance.name == props.new_name:
            assert output.read_bytes() != b"previous accepted output"
            raise ValueError("injected runtime initialization failure")
        return evaluate(instance, *args, **kwargs)

    monkeypatch.setattr(RigInstance, "evaluate", reject)
    with pytest.raises(RuntimeError, match="injected runtime initialization failure"):
        bpy.ops.character_dna.convert_to_dna()
    assert output.read_bytes() == b"previous accepted output"
    assert not (tmp_path / "ExportManifest.json").exists()
    assert not bpy.context.scene.character_dna.rig_instance_list.get(props.new_name)
    assert not any(obj.name.startswith(props.new_name + "_") for obj in bpy.data.objects)


def test_failure_restores_owned_data_without_removing_unrelated_objects(
    load_mhc_conformed_topology_meshes, base_dna_folder, tmp_path, monkeypatch
):
    from character_dna.editors.converter.native.direct import DirectPipeline
    from character_dna.editors.converter.pipeline import positions
    from character_dna.utilities import get_addon_window_manager_properties

    props = get_converter_properties()
    rows = configure_template(props, base_dna_folder, components=("head",))
    head = rows["head"].scene_object
    material = bpy.data.materials.new("Cancellation source")
    material.use_fake_user = True
    head.data.materials.clear()
    head.data.materials.append(material)
    rows["head"].material_source = head
    props.new_name = "CancelledConversion"
    props.new_folder = str(tmp_path)
    original = positions(head)
    was_hidden = head.hide_get()
    was_render_disabled = head.hide_render
    wm = get_addon_window_manager_properties()
    previous_evaluation = wm.evaluate_dependency_graph
    wm.evaluate_dependency_graph = False
    (tmp_path / "head.dna").write_bytes(b"original output")
    unrelated = []

    create_scene = DirectPipeline._create_scene

    def interrupted(pipeline):
        create_scene(pipeline)
        copied = bpy.data.materials.get("CancelledConversion_head_shader")
        assert copied is not None and copied != material
        obj = bpy.data.objects.new("UnrelatedDuringConversion", None)
        bpy.context.scene.collection.objects.link(obj)
        unrelated.append(obj)
        raise ValueError("injected conversion failure")

    monkeypatch.setattr(DirectPipeline, "_create_scene", interrupted)
    try:
        with pytest.raises(RuntimeError, match="injected conversion failure"):
            bpy.ops.character_dna.convert_to_dna()
        import numpy as np

        np.testing.assert_array_equal(positions(head), original)
        assert head.hide_get() == was_hidden
        assert head.hide_render == was_render_disabled
        assert wm.evaluate_dependency_graph is False
        assert bpy.data.objects.get("UnrelatedDuringConversion") is unrelated[0]
        assert not any(obj.name.startswith("CancelledConversion_") for obj in bpy.data.objects)
        assert not bpy.data.materials.get("CancelledConversion_head_shader")
        assert head.data.materials[0] == material
        assert material.use_fake_user
        assert not bpy.context.scene.character_dna.rig_instance_list.get("CancelledConversion")
        assert (tmp_path / "head.dna").read_bytes() == b"original output"
    finally:
        wm.evaluate_dependency_graph = previous_evaluation
        bpy.data.materials.remove(material, do_unlink=True)
        for obj in unrelated:
            bpy.data.objects.remove(obj, do_unlink=True)


@pytest.mark.parametrize("preset", ["advanced", "exact"])
def test_custom_mesh_names_and_all_supplied_anatomy(  # noqa: PLR0915
    load_mhc_conformed_topology_meshes, base_dna_folder, tmp_path, preset, conversion_diagnostics
):
    import numpy as np

    from character_dna.bindings import dna
    from character_dna.dna_io import get_dna_reader, get_dna_writer, release_dna_handle
    from character_dna.editors.converter.pipeline import positions
    from character_dna.editors.shared.utilities import build_world_rest_pose

    path = base_dna_folder / "head.dna"
    reader = get_dna_reader(path)
    rest = build_world_rest_pose(reader)
    joint_indices = {str(reader.getJointName(i)): i for i in range(reader.getJointCount())}
    expected_eye_frames = {}
    names = {str(reader.getMeshName(i)): f"surface_{i}" for i in range(reader.getMeshCount())}
    writer = get_dna_writer(base_dna_folder / "renamed.dna")
    writer.setFrom(reader, dna.DataLayer_All, dna.UnknownLayerPolicy_Preserve, None)
    for index in range(reader.getMeshCount()):
        writer.setMeshName(index, names[str(reader.getMeshName(index))])
    writer.write()
    release_dna_handle(writer)
    created = []
    for index in reader.getMeshIndicesForLOD(0):
        old_name = str(reader.getMeshName(index))
        if old_name == "head_lod0_mesh" or (
            preset == "advanced" and old_name not in {"teeth_lod0_mesh", "eyeLeft_lod0_mesh", "eyeRight_lod0_mesh"}
        ):
            continue
        pos = np.array(reader.getVertexLayoutPositionIndices(index))
        tex = np.array(reader.getVertexLayoutTextureCoordinateIndices(index))
        faces = [np.array(reader.getFaceVertexLayoutIndices(index, f)) for f in range(reader.getFaceCount(index))]
        points = (
            np.column_stack(
                (
                    reader.getVertexPositionXs(index),
                    reader.getVertexPositionYs(index),
                    reader.getVertexPositionZs(index),
                )
            )
            * 0.01
        )
        uv = np.column_stack((reader.getVertexTextureCoordinateUs(index), reader.getVertexTextureCoordinateVs(index)))
        mesh = bpy.data.meshes.new("Input_" + old_name)
        target = points * 1.03 + (0.001, -0.002, 0.003)
        if old_name in {"eyeLeft_lod0_mesh", "eyeRight_lod0_mesh"}:
            from mathutils import Matrix

            angle = 0.15 if "Left" in old_name else -0.1
            rotation = np.array(Matrix.Rotation(angle, 3, "Z"))
            center = target.mean(axis=0)
            target = (target - center) @ rotation.T + center
            joint = "FACIAL_L_Eye" if "Left" in old_name else "FACIAL_R_Eye"
            expected_eye_frames[joint] = rotation @ np.array(rest[joint_indices[joint]].to_3x3())
        mesh.from_pydata(target, [], [pos[f].tolist() for f in faces])
        mesh.uv_layers.new().data.foreach_set("uv", uv[tex[np.concatenate(faces)]].astype(np.float32).ravel())
        obj = bpy.data.objects.new("Input_" + old_name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        created.append((old_name, obj, positions(obj)))
    release_dna_handle(reader)
    (base_dna_folder / "renamed.dna").replace(path)
    # The bundled Exact preset was retired. Keep the all-supplied-anatomy case
    # as an explicit custom recipe, including every COPY/export assertion below.
    recipe = default_template("advanced")
    if preset == "exact":
        for row in recipe["meshes"]:
            row["method"] = "COPY"
    recipe["normalize_to_a_pose"] = False
    recipe["meshes"] = [row for row in recipe["meshes"] if row["component"] == "head"]
    recipe["dna_files"] = {"head": str(path)}
    for row in recipe["meshes"]:
        if row["component"] == "head":
            row["mesh"] = names[row["mesh"]]
            row["lods"] = [names[name] for name in row["lods"] if name in names]
    props = get_converter_properties()
    apply_template(props, recipe)
    name = "CustomNames" + preset.title()
    props.new_name = name
    props.new_folder = str(tmp_path)
    for item in props.extra_meshes:
        item.scene_object = (
            bpy.data.objects["head_lod0_mesh"]
            if item.role == "head"
            else next((obj for name, obj, _positions in created if item.dna_mesh_name == names[name]), None)
        )
    try:
        assert bpy.ops.character_dna.convert_to_dna() == {"FINISHED"}
        assert not list(tmp_path.rglob("ConversionReport.json"))
        report = conversion_diagnostics
        for role in (row["id"] for row in recipe["meshes"] if row["method"] == "COPY"):
            assert report["meshes"][role]["status"] == "provided"
            assert report["meshes"][role]["export_max_error_m"] < 2e-6
        exported = get_dna_reader(tmp_path / "head.dna")
        assert str(exported.getMeshName(0)) == "surface_0"
        rig = bpy.context.scene.character_dna.rig_instance_list[name].head_rig
        for joint, expected in expected_eye_frames.items():
            np.testing.assert_allclose(rig.data.bones[joint].matrix_local.to_3x3(), expected, atol=2e-4)
        for _name, obj, original in created:
            np.testing.assert_array_equal(positions(obj), original)
    finally:
        for _name, obj, _original in created:
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.meshes.remove(data)
        apply_template(props, default_template())
