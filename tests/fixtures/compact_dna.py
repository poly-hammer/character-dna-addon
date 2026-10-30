"""Real native behavior with small geometry for transaction-focused tests.

These assets retain every joint, control, solver and LOD from Ada. They are
not suitable as geometry or deformation oracles: those tests use the complete
original DNA. Each mesh has one triangle and at most one native shape target.
"""

import json
import shutil

import pytest

from constants import TEST_DNA_FOLDER


# Real observed outputs plus all native input bones/ancestors, retained by the
# rig factory. Full skeleton and deformation matrices use ordinary RigProfiles.
STATE_BONES = (
    "FACIAL_C_Jaw",
    "FACIAL_L_LipCorner",
    "FACIAL_L_EyelidUpperB1",
    "calf_knee_l",
    "calf_correctiveRoot_l",
)


def triangle(kind, channel=None, *, joints=(0, 0, 0)):
    """Create a nondegenerate surface with vertex two shared across components."""
    side = 1.0 if kind == "head" else -1.0
    return {
        "positions": {"xs": [side, side, 0.0], "ys": [150.0, 151.0, 150.0], "zs": [0.0, side, 0.0]},
        "normals": {"xs": [0.0] * 3, "ys": [1.0] * 3, "zs": [0.0] * 3},
        "textureCoordinates": {"us": [0.0, 1.0, 0.0], "vs": [0.0, 0.0, 1.0]},
        "layouts": {field: [0, 1, 2] for field in ("positions", "textureCoordinates", "normals")},
        "faces": [{"layoutIndices": [0, 1, 2]}],
        "maximumInfluencePerVertex": 1,
        "skinWeights": [{"weights": [1.0], "jointIndices": [joint]} for joint in joints],
        "blendShapeTargets": []
        if channel is None
        else [
            {
                "deltas": {"xs": [0.1], "ys": [0.0], "zs": [0.0]},
                "vertexIndices": [1],
                "blendShapeChannelIndex": channel,
            }
        ],
    }


@pytest.fixture(scope="session")
def compact_dna_asset(addon, tmp_path_factory):
    """Serialize genuine SDK files once; the rig factory leases private copies."""
    from character_dna.dna_io import merged_graph

    folder = tmp_path_factory.mktemp("compact_dna")
    for kind in ("head", "body"):
        graph = merged_graph.from_bytes((TEST_DNA_FOLDER / "ada" / f"{kind}.dna").read_bytes(), geometry=False)
        mapping = graph["defn1.1"]["meshBlendShapeChannelMapping"]
        channels = {}
        for mesh, channel in zip(mapping["from"], mapping["to"], strict=True):
            channels.setdefault(mesh, channel)
        joints = (
            (
                graph["defn1.1"]["jointNames"].index("FACIAL_C_Jaw"),
                graph["defn1.1"]["jointNames"].index("FACIAL_L_LipCorner"),
                0,
            )
            if kind == "head"
            else (0, 0, 0)
        )
        graph["geom1.1"]["meshes"] = [
            triangle(kind, channels.get(index), joints=joints) for index in range(len(graph["defn1.1"]["meshNames"]))
        ]
        graph["defn1.1"]["meshBlendShapeChannelMapping"] = {
            "from": list(channels),
            "to": list(channels.values()),
        }
        (folder / f"{kind}.dna").write_bytes(merged_graph.to_bytes(graph))
    shutil.copyfile(TEST_DNA_FOLDER / "ada/ExportManifest.json", folder / "ExportManifest.json")
    (folder / "necklines.json").write_text(json.dumps({str(lod): {"2": 2} for lod in range(8)}))
    return folder


@pytest.fixture
def compact_dna(compact_dna_asset, monkeypatch):
    """Use the matching seam only within a test leasing the compact asset."""
    from character_dna.dna_io import merged_graph
    from character_dna.editors.converter import scene_merge

    for module in (merged_graph, scene_merge):
        monkeypatch.setattr(module, "HEAD_TO_BODY_EDGE_LOOP_FILE_PATH", compact_dna_asset / "necklines.json")
    return compact_dna_asset
