"""Bulk geometry transport through the SDK, without decimal JSON intermediates."""

from typing import Any

import numpy as np


def _vectors(reader: Any, api: str, *indices: int, axes: tuple[str, ...] = ("xs", "ys", "zs")) -> dict:
    return {axis: getattr(reader, "get" + api + axis.capitalize())(*indices) for axis in axes}


def read(reader: Any) -> dict:
    """Read native arrays in the reader's coordinate system, retaining float32 bits."""
    face_indices = reader.getFaceVertexLayoutIndices
    skin_values = reader.getSkinWeightsValues
    skin_joints = reader.getSkinWeightsJointIndices
    meshes = [
        {
            "positions": _vectors(reader, "VertexPosition", index),
            "normals": _vectors(reader, "VertexNormal", index),
            "textureCoordinates": _vectors(reader, "VertexTextureCoordinate", index, axes=("us", "vs")),
            "layouts": {
                field: getattr(reader, "getVertexLayout" + api + "Indices")(index)
                for field, api in (
                    ("positions", "Position"),
                    ("textureCoordinates", "TextureCoordinate"),
                    ("normals", "Normal"),
                )
            },
            "faces": [{"layoutIndices": face_indices(index, face)} for face in range(reader.getFaceCount(index))],
            "maximumInfluencePerVertex": reader.getMaximumInfluencePerVertex(index),
            "skinWeights": [
                {
                    "weights": skin_values(index, vertex),
                    "jointIndices": skin_joints(index, vertex),
                }
                for vertex in range(reader.getSkinWeightsCount(index))
            ],
            "blendShapeTargets": [
                {
                    "deltas": _vectors(reader, "BlendShapeTargetDelta", index, target),
                    "vertexIndices": reader.getBlendShapeTargetVertexIndices(index, target),
                    "blendShapeChannelIndex": reader.getBlendShapeChannelIndex(index, target),
                }
                for target in range(reader.getBlendShapeTargetCount(index))
            ],
        }
        for index in range(reader.getMeshCount())
    ]
    return {"meshes": meshes}


def write(writer: Any, geometry: dict) -> None:
    """Write geometry using typed SDK setters, including sparse shape targets."""
    writer.clearMeshes()
    face_indices = writer.setFaceVertexLayoutIndices
    skin_values = writer.setSkinWeightsValues
    skin_joints = writer.setSkinWeightsJointIndices
    # Pre-size indexed collections: growing them one element per SDK call
    # repeatedly reallocates geometry and dominates structural write time.
    if geometry["meshes"]:
        writer.setVertexPositions(len(geometry["meshes"]) - 1, [])
    for index, mesh in enumerate(geometry["meshes"]):
        for field, api in (
            ("positions", "VertexPositions"),
            ("normals", "VertexNormals"),
            ("textureCoordinates", "VertexTextureCoordinates"),
            ("layouts", "VertexLayouts"),
        ):
            getattr(writer, "set" + api)(index, np.asarray(list(mesh[field].values())).T.tolist())
        if mesh["faces"]:
            face_indices(index, len(mesh["faces"]) - 1, [])
        for face, values in enumerate(mesh["faces"]):
            face_indices(index, face, values["layoutIndices"])
        writer.setMaximumInfluencePerVertex(index, mesh["maximumInfluencePerVertex"])
        if mesh["skinWeights"]:
            skin_values(index, len(mesh["skinWeights"]) - 1, [])
        for vertex, values in enumerate(mesh["skinWeights"]):
            skin_values(index, vertex, values["weights"])
            skin_joints(index, vertex, values["jointIndices"])
        if mesh["blendShapeTargets"]:
            writer.setBlendShapeChannelIndex(index, len(mesh["blendShapeTargets"]) - 1, 0)
        for target, values in enumerate(mesh["blendShapeTargets"]):
            writer.setBlendShapeChannelIndex(index, target, values["blendShapeChannelIndex"])
            writer.setBlendShapeTargetVertexIndices(index, target, values["vertexIndices"])
            writer.setBlendShapeTargetDeltas(index, target, np.asarray(list(values["deltas"].values())).T.tolist())
