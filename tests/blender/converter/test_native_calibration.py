"""Direct SDK calibration must preserve the established saved DNA contract."""

import numpy as np
import pytest

from character_dna.dna_io import DNACalibrator, get_dna_reader, release_dna_handle
from character_dna.editors.converter.native_calibration import NativeCalibrator
from fixtures.rigs import RigProfile


@pytest.mark.parametrize("component,zero_shapes", [("head", False), ("head", True), ("body", False)])
def test_native_calibration_matches_python_saved_dna(rig_factory, tmp_path, component, zero_shapes):
    profile = RigProfile(components=(component,), lods=(0,), shape_keys=False, face_board=False, materials=False)
    with rig_factory.lease(profile, tmp_path / "scene") as instance:
        instance.output.folder_path = str(tmp_path)
        instance.output.auto_update_lods = True
        instance.output.align_head_and_body = False
        mesh = getattr(instance, f"{component}_mesh")
        mesh.data.vertices[42].co.x += 0.007
        mesh.data.update()
        source = get_dna_reader(getattr(instance, f"{component}_dna_file_path"))
        try:
            primary = str(source.getMeshName(0))
            family = [
                str(source.getMeshName(i))
                for i in range(1, source.getMeshCount())
                if str(source.getMeshName(i)).startswith(component + "_")
            ]
        finally:
            release_dna_handle(source)
        arguments = {
            "instance": instance,
            "linear_modifier": 1.0,
            "component_type": component,
            "shape_keys": False,
            "vertex_groups": False,
            "normals": True,
            "textures": False,
            "zero_shape_deltas": zero_shapes,
            "seam_follower": None,
            "lod_families": {primary: family},
            "mesh_objects_by_name": {primary: mesh},
        }
        reference = DNACalibrator(**arguments, file_name="reference.dna")
        native = NativeCalibrator(**arguments, file_name="native.dna")
        try:
            assert reference.run()[0]
            assert native.run()[0]
            assert native.native_report["transport"] == "direct-openriglogic"
            assert native.native_report["propagated"] == len(family)
        finally:
            release_dna_handle(reference._dna_reader)
            reference._dna_writer.close()
            release_dna_handle(native._dna_reader)
        expected = get_dna_reader(tmp_path / "reference.dna")
        actual = get_dna_reader(tmp_path / "native.dna")
        try:
            assert expected.getMeshCount() == actual.getMeshCount()
            for mesh_id in range(expected.getMeshCount()):
                for field in (
                    "VertexPositionXs",
                    "VertexPositionYs",
                    "VertexPositionZs",
                    "VertexNormalXs",
                    "VertexNormalYs",
                    "VertexNormalZs",
                ):
                    np.testing.assert_array_equal(
                        getattr(actual, "get" + field)(mesh_id),
                        getattr(expected, "get" + field)(mesh_id),
                        err_msg=f"{mesh_id}: {field}",
                    )
                assert actual.getBlendShapeTargetCount(mesh_id) == expected.getBlendShapeTargetCount(mesh_id)
                for target in range(expected.getBlendShapeTargetCount(mesh_id)):
                    for field in (
                        "BlendShapeTargetVertexIndices",
                        "BlendShapeTargetDeltaXs",
                        "BlendShapeTargetDeltaYs",
                        "BlendShapeTargetDeltaZs",
                    ):
                        np.testing.assert_array_equal(
                            getattr(actual, "get" + field)(mesh_id, target),
                            getattr(expected, "get" + field)(mesh_id, target),
                        )
            for field in (
                "NeutralJointTranslationXs",
                "NeutralJointTranslationYs",
                "NeutralJointTranslationZs",
                "NeutralJointRotationXs",
                "NeutralJointRotationYs",
                "NeutralJointRotationZs",
            ):
                np.testing.assert_array_equal(getattr(actual, "get" + field)(), getattr(expected, "get" + field)())
            for reader in (expected, actual):
                assert reader.getMetaDataCount() > 0

            def metadata(reader):
                return {
                    reader.getMetaDataKey(i): reader.getMetaDataValue(reader.getMetaDataKey(i))
                    for i in range(reader.getMetaDataCount())
                }

            assert metadata(actual) == metadata(expected)
        finally:
            release_dna_handle(expected)
            release_dna_handle(actual)
