"""Validate Big Guy or Male 1 without saving the supplied scene.

Run Blender --background --disable-autoexec <scan.blend> --python this_file -- MODE.
MODE is OFF, LIMBS, SPINE, or ALL. DNA, renders and numerical checks are retained
under reports/pose-normalization/<subject>/<mode>.
"""

import hashlib
import json
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/addons"))
mode = sys.argv[sys.argv.index("--") + 1]
if mode not in {"OFF", "LIMBS", "SPINE", "ALL"}:
    raise ValueError("Choose OFF, LIMBS, SPINE or ALL")
if not any(name in bpy.data.filepath for name in ("big_guy", "male01")):
    raise ValueError("This validation fixture expects the Big Guy or Male 1 scan")
subject = "big_guy" if "big_guy" in bpy.data.filepath else "male01"
folder = ROOT / "reports/pose-normalization" / subject / mode
folder.mkdir(parents=True, exist_ok=True)
record = {"subject": subject, "mode": mode, "source": bpy.data.filepath}
with Path(bpy.data.filepath).open("rb") as source_file:
    record["source_sha256"] = hashlib.file_digest(source_file, "sha256").hexdigest()
try:
    bpy.ops.preferences.addon_enable(module="character_dna")
    bpy.context.preferences.addons["character_dna"].preferences.backup_manager.enabled = False
    from character_dna.editors.converter import templates
    from character_dna.editors.converter.properties import get_converter_properties
    from character_dna.editors.converter.native import session
    from character_dna.editors.converter.lifecycle import ConversionLifecycle
    from character_dna.editors.converter.pipeline import positions
    from character_dna.bindings import load_converter_core

    props = get_converter_properties()
    templates.load_template(props, templates.TEMPLATE_FOLDER / "3d_scan_store_full_body.json")
    props.joined_mesh = bpy.data.objects["Body"]
    bindings = {
        "eye_left": "Eye_Balls",
        "eye_right": "Eye_Balls",
        "lens_left": "Lens",
        "lens_right": "Lens",
        "teeth": "Teeth",
        "brows": "Brows",
        "eyelashes": "Lashes",
        "eye_edge": "Eye_Wet",
    }
    for item in props.extra_meshes:
        if item.role in bindings:
            item.scene_object = bpy.data.objects[bindings[item.role]]
    props.pose_normalization = mode
    props.new_name = f"{subject}_{mode}"
    props.new_folder = str(folder)
    props.scene_layout = "SPLIT"
    originals = {o.name: positions(o) for o in bpy.data.objects if o.type == "MESH"}
    original_accept = session._accept
    arrays = {}

    def accept(pipeline, job, *args, **kwargs):
        output = job.result[0]
        if mode != "OFF":
            assert "joints" in output and "posed" in output
        if "joints" in output:
            arrays["joints"] = np.frombuffer(output["joints"], float).reshape(-1, 4, 4).copy()
            arrays["posed"] = np.frombuffer(output["posed"], float).reshape(-1, 4, 4).copy()
            arrays["names"] = np.array(pipeline.pose_names)
            arrays["parents"] = np.array(pipeline.readers["body"].info["parents"])
            record["profile"] = pipeline.pose_profile
        result = original_accept(pipeline, job, *args, **kwargs)
        for key, value in pipeline.targets.items():
            arrays[key] = value.copy()
        return result

    session._accept = accept
    original_verify = ConversionLifecycle._verify_export

    def verify(lifecycle, path):
        original_verify(lifecycle, path)
        record["conversion"] = lifecycle._pipeline.report

    ConversionLifecycle._verify_export = verify
    record["build"] = load_converter_core().build_info()
    result = bpy.ops.character_dna.convert_to_dna()
    assert result == {"FINISHED"}, str(result)
    record["source_unchanged"] = all(np.array_equal(positions(bpy.data.objects[n]), p) for n, p in originals.items())
    assert record["source_unchanged"]
    with Path(bpy.data.filepath).open("rb") as source_file:
        assert hashlib.file_digest(source_file, "sha256").hexdigest() == record["source_sha256"]
    np.savez_compressed(folder / "geometry.npz", **arrays)
    native = load_converter_core()
    exported = {key: native.DnaTemplate(str(folder / (key + ".dna"))) for key in ["head", "body"]}
    states = {
        key: {
            name: np.array(row["matrix"]).reshape(4, 4) for name, row in json.loads(template.info())["states"].items()
        }
        for key, template in exported.items()
    }
    common = states["head"].keys() & states["body"].keys()
    record["shared_joint_translation_error_m"] = max(
        float(np.linalg.norm(states["head"][n][:3, 3] - states["body"][n][:3, 3])) for n in common
    )
    record["shared_joint_rotation_error"] = max(
        float(np.abs(states["head"][n][:3, :3] - states["body"][n][:3, :3]).max()) for n in common
    )
    assert record["shared_joint_translation_error_m"] < 5e-5
    assert record["shared_joint_rotation_error"] < 5e-4
    if "joints" in arrays:
        c, p, names = arrays["joints"], arrays["posed"], arrays["names"].tolist()
        carried = record["conversion"]["a_pose"].get("carried_joints", [])
        errors = []
        for name in carried:
            j = names.index(name)
            parent = arrays["parents"][j]
            parent_name = names[parent]
            owner = record["profile"]["helpers"].get(parent_name, parent_name)
            parent = names.index(owner)
            errors.append(float(np.abs(np.linalg.inv(c[parent]) @ c[j] - np.linalg.inv(p[parent]) @ p[j]).max()))
        record["max_carried_local_error"] = max(errors, default=0.0)
        assert record["max_carried_local_error"] < 1e-7
        selected = record["conversion"]["a_pose"].get("normalized_joints", []) + carried
        record["exported_pose_translation_error_m"] = max(
            float(np.linalg.norm(states["body"][n][:3, 3] - c[names.index(n), :3, 3])) for n in selected
        )
        record["exported_pose_rotation_error"] = max(
            float(np.abs(states["body"][n][:3, :3] - c[names.index(n), :3, :3]).max()) for n in selected
        )
        assert record["exported_pose_translation_error_m"] < 5e-5
        assert record["exported_pose_rotation_error"] < 5e-4
    record["status"] = "passed"
    # Keep a neutral clay side view with consistent framing for each character.
    scene = bpy.context.scene
    record["rendered_meshes"] = []
    for obj in scene.objects:
        if obj.type == "LIGHT":
            obj.hide_render = True
        if obj.type == "MESH" and "_lod" in obj.name and "_lod0_" not in obj.name:
            obj.hide_render = True
        if obj.type == "MESH" and not obj.hide_render:
            record["rendered_meshes"].append(obj.name)
    camera_data = bpy.data.cameras.new("Pose validation camera")
    camera = bpy.data.objects.new("Pose validation camera", camera_data)
    scene.collection.objects.link(camera)
    center = Vector((0, 0, 1.12 if subject == "big_guy" else 0.95))
    camera.location = center + Vector((5, 0, 0))
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.5 if subject == "big_guy" else 2.12
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (0.65, 0.65, 0.65)
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.background_type = "VIEWPORT"
    scene.display.shading.background_color = (0.08, 0.08, 0.08)
    scene.render.resolution_x = 700
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(folder / "side.png")
    bpy.ops.render.render(write_still=True)
    camera.location = center + Vector((0, -5, 0))
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(folder / "front.png")
    bpy.ops.render.render(write_still=True)
except Exception:
    record["status"] = "failed"
    record["error"] = traceback.format_exc()
    print(record["error"])
finally:
    (folder / "report.json").write_text(json.dumps(record, indent=2))
    print("VALIDATION_RESULT=" + json.dumps({k: record[k] for k in ["subject", "mode", "status"]}))
