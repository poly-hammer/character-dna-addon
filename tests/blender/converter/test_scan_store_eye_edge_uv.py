"""The head template's diagonal Eye Wet island must retain anatomical correspondence."""

import numpy as np
import pytest

from character_dna.editors.converter import templates
from character_dna.editors.converter.native.template import TemplateReader


@pytest.mark.parametrize("side", ["L", "R"])
def test_head_eye_wet_uv_matches_skinned_anatomy(side):
    path = templates.TEMPLATE_FOLDER / "../../../shared/resources/dna/3d_scan_store/head/head.dna"
    mesh = TemplateReader(str(path.resolve()), "head").meshes["eyeEdge_lod0_mesh"]
    uv = np.zeros((len(mesh.positions), 2))
    count = np.zeros(len(mesh.positions))
    np.add.at(uv, mesh.corners, mesh.uv)
    np.add.at(count, mesh.corners, 1)
    assert np.all(count > 0)
    uv /= count[:, None]
    np.testing.assert_allclose(uv[mesh.corners], mesh.uv, atol=1e-7)

    def center(region):
        weights = np.zeros(len(mesh.positions))
        for name, values in mesh.groups.items():
            if name.startswith(f"FACIAL_{side}_{region}"):
                for vertex, weight in values:
                    weights[vertex] += weight
        assert weights.sum() > 0, region
        return np.average(uv, axis=0, weights=weights)

    # Island coverage alone cannot detect a 180-degree rotation. In the Scan Store
    # head layout the tear duct has high U+V, and the upper lid has high U-V.
    upper, lower = center("EyelidUpper"), center("EyelidLower")
    inner, outer = center("EyeCornerInner"), center("EyeCornerOuter")
    assert (upper - lower) @ np.array([1.0, -1.0]) > 0.1
    assert (inner - outer).sum() > 0.5

    # Each eye is an annulus without UV folds; opposite eyes may have opposite winding.
    points = uv[mesh.triangles]
    a, b = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    areas = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    positive_x = mesh.positions[mesh.triangles, 0].mean(axis=1) > 0
    areas = areas[positive_x if side == "L" else ~positive_x]
    assert np.all(areas > 1e-10) or np.all(areas < -1e-10)
