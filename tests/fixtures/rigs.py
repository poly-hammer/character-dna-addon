"""Pristine rig artifacts with independently owned scene and DNA state per test."""

import hashlib
import json
import platform
import shutil
import sys
import tempfile

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import bpy
import pytest

from constants import TEST_ANIMATION_FOLDER, TEST_DNA_FOLDER


def reset_scene():
    """Release Python ownership as well as Blender IDs, including failed edits."""
    from character_dna.editors.shared.editor import Editor
    from character_dna.rig_instance import RigInstance
    from character_dna.runtime import controller, engine

    for scene in bpy.data.scenes:
        properties = getattr(scene, "character_dna", None)
        for instance in getattr(properties, "rig_instance_list", ()):
            Editor.clear_instance_editors(instance)
    controller.unregister()
    try:
        bpy.ops.wm.read_homefile(use_empty=True)
        # load_pre released SDK handles; discard remaining Python-only caches.
        RigInstance.data.clear()
        assert not engine._records and not engine._models
        assert not controller._suspended
    finally:
        controller.register()


def release_partitions(previous):
    """Remove only SDK working directories created inside the current lease."""
    from character_dna.dna_io import document

    generated = set(document._leases) - previous
    temporary_root = Path(tempfile.gettempdir()).resolve()
    roots = {path.parents[2].resolve() for path in generated}
    for root in roots:
        assert root.parent == temporary_root and root.name.startswith("character-dna-"), root
    for path in generated:
        del document._leases[path]
    for root in roots:
        if root.is_dir():
            shutil.rmtree(root)


@dataclass(frozen=True)
class RigProfile:
    components: tuple[str, ...] = ("head", "body")
    asset: str = "ada"
    lods: tuple[int, ...] = (0,)
    meshes: bool = True
    bones: bool = True
    face_board: bool = False
    shape_keys: bool = False
    materials: bool = False
    face_board_animation: bool = False
    document_mode: Literal["separate", "scene", "combined"] = "separate"
    head_targets: tuple[int, ...] = ()
    pose_bones: tuple[str, ...] = ()


class RigFactory:
    """Cache files, never borrowed Blender objects or mutable SDK handles.

    Artifacts are scoped to this pytest process, binding build and checkout.
    Every lease reopens a private copy and rebinds all persisted runtime paths.
    Importer tests can request ``fresh=True`` to exercise the original import.
    """

    def __init__(self, directory: Path, *, fresh: bool = False):
        self.directory = directory
        self.fresh = fresh
        self.artifacts: dict[RigProfile, tuple[Path, dict[str, str]]] = {}
        self.build_count = 0

    @staticmethod
    def _copy_inputs(profile: RigProfile, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        source = TEST_DNA_FOLDER / profile.asset
        for name in (*[f"{kind}.dna" for kind in profile.components], "ExportManifest.json"):
            if (source / name).is_file():
                shutil.copyfile(source / name, folder / name)

    @staticmethod
    def _import(profile: RigProfile, folder: Path):
        from character_dna.runtime import engine
        from character_dna.utilities import get_active_rig_instance

        bpy.ops.wm.read_homefile(use_empty=True)
        RigFactory._copy_inputs(profile, folder)

        # These profiles discard unobserved scene bones before use. Defer native
        # output construction until that final skeleton and its keys exist,
        # instead of installing thousands of drivers only to delete them.
        # Ordinary profiles and every importer contract keep the normal path.
        deferred = patch.object(engine, "install", autospec=True) if profile.pose_bones else nullcontext()
        with deferred:
            result = bpy.ops.character_dna.import_dna(
                filepath=str(folder / f"{profile.components[0]}.dna"),
                include_body=len(profile.components) > 1,
                import_face_board=profile.face_board,
                import_mesh=profile.meshes,
                import_bones=profile.bones,
                import_shape_keys=profile.shape_keys,
                import_materials=profile.materials,
                import_vertex_groups=True,
                **{f"import_lod{index}": index in profile.lods for index in range(8)},
            )
            assert result == {"FINISHED"}
            if profile.face_board_animation:
                result = bpy.ops.character_dna.import_face_board_animation(
                    filepath=str(TEST_ANIMATION_FOLDER / "head/MHC_FaceBoardROM.fbx")
                )
                assert result == {"FINISHED"}
            instance = get_active_rig_instance()
            if profile.pose_bones:
                from character_dna.runtime import controller
                from character_dna.utilities import switch_to_bone_edit_mode

                # Lifecycle and transaction contracts only need their observed
                # bones. Retain native input bones and all ancestors so local/world
                # transforms have the same meaning; DNA/evaluator outputs stay complete.
                with controller.authoring_operation(instance):
                    for kind in profile.components:
                        rig = instance.component(kind).rig
                        reader = instance.component(kind).dna_reader
                        inputs = {
                            reader.getRawControlName(index).rsplit(".", 1)[0]
                            for index in range(reader.getRawControlCount())
                            if reader.getRawControlName(index).endswith((".qx", ".qy", ".qz", ".qw"))
                        }
                        keep = set()
                        for name in (*profile.pose_bones, *inputs, "head"):
                            bone = rig.data.bones.get(name)
                            while bone is not None:
                                keep.add(bone.name)
                                bone = bone.parent
                        assert keep, (kind, profile.pose_bones)
                        switch_to_bone_edit_mode(rig)
                        for bone in tuple(rig.data.edit_bones):
                            if bone.name not in keep:
                                rig.data.edit_bones.remove(bone)
                        bpy.ops.object.mode_set(mode="OBJECT")
                    instance.destroy()
                if "head" in profile.components:
                    instance.head_initialize(update_raw_control_list=False)
                if "body" in profile.components:
                    instance.body_initialize(update_rbf_solver_list=False)
            if profile.head_targets:
                from character_dna.dna_io.misc import create_shape_key
                from character_dna.editors.shape_key_editor.core import linear_modifier_for
                from character_dna.runtime import controller

                assert "head" in profile.components and profile.meshes and not profile.shape_keys
                with controller.authoring_operation(instance):
                    head = instance.component("head")
                    reader = head.dna_reader
                    for target in profile.head_targets:
                        channel = reader.getBlendShapeChannelName(reader.getBlendShapeChannelIndex(0, target))
                        create_shape_key(
                            target,
                            0,
                            head.mesh,
                            reader,
                            channel,
                            prefix=reader.getMeshName(0) + "__",
                            linear_modifier=linear_modifier_for(reader),
                        )
                    instance.destroy_references()
                    instance.sync_shape_key_list()
        if profile.pose_bones:
            instance.evaluate()
        if profile.document_mode != "separate":
            assert profile.components == ("head", "body")
            instance.converter_merge_mode = "DNA" if profile.document_mode == "combined" else "SCENE"
            instance.output.folder_path = str(folder)
            assert bpy.ops.character_dna.merge_components() == {"FINISHED"}
        return instance

    @staticmethod
    def _hashes(folder: Path, profile: RigProfile) -> dict[str, str]:
        names = (*profile.components, "combined") if profile.document_mode == "combined" else profile.components
        return {kind: hashlib.sha256((folder / f"{kind}.dna").read_bytes()).hexdigest() for kind in names}

    def _artifact(self, profile: RigProfile) -> tuple[Path, dict[str, str]]:
        if profile not in self.artifacts:
            from character_dna.runtime import engine

            folder = self.directory / str(len(self.artifacts))
            instance = self._import(profile, folder)
            if profile.document_mode == "combined":
                from character_dna.dna_io import document
                from character_dna.editors.converter import scene_merge, source_views
                from character_dna.runtime import controller

                combined = Path(instance["combined_dna_path"])
                # A newly written document can cache its original component
                # bytes; a cold projection serializes equivalent SDK views.
                # Build the pristine surface against those cold views so its
                # DNA revision guards remain valid after cache eviction.
                document._read_cache.pop(document.fingerprint(combined.read_bytes()), None)
                with controller.authoring_operation(instance):
                    document.bind_instance(instance, combined)
                    scene_merge.split(instance, apply=False)
                    scene_merge.merge(instance)
                    source_views.archive(instance)
                shutil.copyfile(instance["combined_dna_path"], folder / "combined.dna")
                instance["combined_dna_path"] = "//combined.dna"
            # A copied blend resolves these against its private directory on
            # load, so native hydration never opens another test's DNA.
            for kind in profile.components:
                instance[f"{kind}_dna_file_path"] = f"//{kind}.dna"
            for carrier in engine.carriers(instance):
                for kind in profile.components:
                    carrier[f"{kind}_dna_file_path"] = f"//{kind}.dna"
            path = folder / "pristine.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False)
            self.artifacts[profile] = path, self._hashes(folder, profile)
            native = Path(engine.native_module().__file__)
            (folder / "profile.json").write_text(
                json.dumps(
                    {
                        "schema": 2,
                        "profile": asdict(profile),
                        "dna_sha256": self.artifacts[profile][1],
                        "blender": bpy.app.version_string,
                        "python_abi": sys.implementation.cache_tag,
                        "platform": platform.platform(),
                        "binding_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.build_count += 1
        return self.artifacts[profile]

    @contextmanager
    def lease(self, profile: RigProfile, folder: Path, *, fresh: bool = False):
        from character_dna.dna_io import document
        from character_dna.runtime import controller, engine
        from character_dna.utilities import get_active_rig_instance

        folder = folder.resolve()
        previous_partitions = set(document._leases)
        try:
            if fresh or self.fresh:
                instance = self._import(profile, folder)
            else:
                artifact, hashes = self._artifact(profile)
                assert self._hashes(artifact.parent, profile) == hashes, "A test modified pristine DNA"
                self._copy_inputs(profile, folder)
                if profile.document_mode == "combined":
                    shutil.copyfile(artifact.parent / "combined.dna", folder / "combined.dna")
                local_blend = folder / "fixture.blend"
                shutil.copyfile(artifact, local_blend)
                bpy.ops.wm.open_mainfile(filepath=str(local_blend))
                instance = get_active_rig_instance()
                # Hydration already used private paths. Normalize them for
                # callers that use pathlib directly, then initialize authoring
                # readers without rebuilding the portable native bindings.
                with controller.preserve_bindings():
                    if profile.document_mode != "combined":
                        for kind in profile.components:
                            instance[f"{kind}_dna_file_path"] = str(folder / f"{kind}.dna")
                    for carrier in engine.carriers(instance):
                        for kind in ("head", "body"):
                            carrier[f"{kind}_dna_file_path"] = getattr(instance, f"{kind}_dna_file_path")
                    # The pristine blend already owns the serialized editor
                    # lists. Recreate native handles and RNA lookup plans only;
                    # rebuilding the same thousands of UI records is redundant.
                    # Combined-document recovery already initializes both
                    # private SDK partitions through bind_instance().
                    if not instance.head_initialized:
                        instance.head_initialize(update_raw_control_list=False)
                    if not instance.body_initialized:
                        instance.body_initialize(update_rbf_solver_list=False)
                    instance.enforce_merged_constraints()
                bpy.context.view_layer.update()
            instance.output.folder_path = str(folder / "output")
            for kind in profile.components:
                path = Path(instance.component(kind).dna_file_path).resolve()
                if profile.document_mode == "combined":
                    # Public blend-load recovery materializes disposable SDK
                    # partitions. Each must belong to this lease's document.
                    combined = Path(instance["combined_dna_path"]).resolve()
                    assert combined.parent == folder
                    assert document._leases[path][0] == combined
                else:
                    assert path.parent == folder
                assert instance.component(kind).dna_reader is not None
            assert not controller.is_suspended(instance)
            issues = engine.binding_issues(instance)
            assert not issues, issues
            if profile.bones:
                assert engine.active(instance)
            yield instance
        finally:
            # Existing load handlers release editor caches, timers and native
            # records before Blender IDs disappear. No RNA escapes this lease.
            reset_scene()
            # Materialized SDK partitions belong to this lease even though the
            # public loader places them in the system temporary directory.
            # Release all native handles before deleting only newly owned roots.
            release_partitions(previous_partitions)


@contextmanager
def saved_scene(path):
    """Lease a saved scenario while building immutable exported test data."""
    from character_dna.runtime import controller
    from character_dna.utilities import get_active_rig_instance

    reset_scene()
    try:
        bpy.ops.wm.open_mainfile(filepath=str(path))
        instance = get_active_rig_instance()
        assert instance is not None
        with controller.preserve_bindings():
            instance.head_initialize(update_raw_control_list=False)
            instance.body_initialize(update_rbf_solver_list=False)
        yield instance
    finally:
        reset_scene()


@pytest.fixture(scope="session")
def rig_factory(addon, tmp_path_factory, pytestconfig):
    return RigFactory(tmp_path_factory.mktemp("rig_profiles"), fresh=pytestconfig.getoption("--fresh-rigs"))


@pytest.fixture
def empty_character_scene(addon):
    reset_scene()
    try:
        yield bpy.context.scene
    finally:
        reset_scene()


@pytest.fixture
def minimal_character(addon):
    """Real RNA, callbacks and skinned geometry for tests unrelated to DNA data."""
    bpy.ops.wm.read_homefile(use_empty=True)
    try:
        instance = bpy.context.scene.character_dna.rig_instance_list.add()
        instance["name"] = "MinimalComponentRig"
        instance["auto_evaluate"] = False
        for kind in ("head", "body"):
            rig = bpy.data.objects.new(f"{instance.name}_{kind}_rig", bpy.data.armatures.new(f"{kind}Rig"))
            bpy.context.scene.collection.objects.link(rig)
            mesh = bpy.data.meshes.new(f"{kind}Mesh")
            mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
            obj = bpy.data.objects.new(f"{instance.name}_{kind}_lod0_mesh", mesh)
            bpy.context.scene.collection.objects.link(obj)
            obj.modifiers.new("Rig", "ARMATURE").object = rig
            setattr(instance, f"{kind}_rig", rig)
            setattr(instance, f"{kind}_mesh", obj)
        instance.editor_component = "head"
        yield instance
    finally:
        reset_scene()
