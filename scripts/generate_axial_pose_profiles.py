"""Author axial landmarks on the bundled reference geometry, never on a scan."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/addons"))
import numpy as np
from character_dna.bindings import load_converter_core

root = Path(__file__).resolve().parents[1] / "src/addons/character_dna/editors/shared/resources"
core = load_converter_core()
axial = [f"spine_{i:02}" for i in range(1, 6)] + ["neck_01", "neck_02", "head"]
for identifier in ["metahuman/full_body", "3d_scan_store/full_body"]:
    path = root / "topologies" / identifier / "pose.json"
    profile = json.loads(path.read_text())
    templates = {k: core.DnaTemplate(str(root / "dna" / identifier / (k + ".dna"))) for k in ["body", "head"]}
    info = {k: json.loads(t.info()) for k, t in templates.items()}
    names = info["body"]["names"]
    bind = {n: np.array(info["body"]["states"][n]["matrix"]).reshape(4, 4) for n in names}
    for n, row in profile["joints"].items():
        row["region"] = "SPINE" if n in axial else "LIMBS" if row["solve"] else "ANCHOR"
        if n in axial:
            row["max_angle"] = 45 if n.startswith("spine") else 65
            if n != "head":
                row["aim"] = axial[axial.index(n) + 1]
    profile["axial_landmarks"] = {"body": {}, "head": {}}
    for key in ["body", "head"]:
        mesh = templates[key].mesh(key + "_lod0_mesh")
        pts = np.frombuffer(mesh["positions"], np.float32).reshape(-1, 3).astype(float)
        ids = np.frombuffer(mesh["position_ids"], np.int32)
        ns = info[key]["names"]
        parents = info[key]["parents"]
        w = np.frombuffer(templates[key].weights(key + "_lod0_mesh", ns), float).reshape(-1, len(ns))
        w = w / w.sum(axis=1)[:, None]
        support = {n: np.zeros(len(pts)) for n in axial}
        for j, n in enumerate(ns):
            owner = j
            while ns[owner] not in names and parents[owner] >= 0:
                owner = parents[owner]
            n = profile["helpers"].get(ns[owner], ns[owner])
            if n in support:
                support[n] += w[:, j]
        for n in axial[:5] if key == "body" else axial[5:]:
            candidates = np.flatnonzero(support[n] > 0.15)
            local = (pts - bind[n][:3, 3]) @ bind[n][:3, :3]
            if n == "head":
                candidates = candidates[(local[candidates, 0] > 0.025) & (local[candidates, 2] > -0.035)]
                selected = candidates[np.linspace(0, len(candidates) - 1, min(512, len(candidates)), dtype=int)]
                profile["cranium_samples"] = {"vertices": ids[selected].tolist()}
            assert len(candidates) > 16, (identifier, n)
            # Anatomical bind Y is the lateral axis on both shipped rigs. Take
            # compact bilateral patches, weighted toward the joint's elevation.
            patches = {}
            for sign, kind in [(1, "positive"), (-1, "negative")]:
                side = candidates[sign * local[candidates, 1] > np.quantile(sign * local[candidates, 1], 0.85)]
                order = np.argsort(np.abs(local[side, 0] - np.median(local[candidates, 0])))
                patches[kind] = {"vertices": ids[side[order[:8]]].tolist()}
            if n == "head":
                top = candidates[np.argsort(local[candidates, 0])[-12:]]
                patches["tip"] = {"vertices": ids[top].tolist()}
            profile["axial_landmarks"][key][n] = patches
        if key == "head":
            profile["head_topology"] = next(m["topology"] for m in info[key]["meshes"] if m["name"] == "head_lod0_mesh")
    path.write_text(json.dumps(profile, indent=2) + "\n", newline="\n")
    print(identifier, "cranium samples", len(profile["cranium_samples"]["vertices"]))
