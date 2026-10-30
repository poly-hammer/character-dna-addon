import pytest

from constants import TEST_DNA_FOLDER
from fixtures.rigs import saved_scene


@pytest.fixture(scope="session")
def original_head_dna_json_data(temp_folder, dna_folder_name: str) -> dict:
    from utilities.dna_data import get_dna_json_data

    dna_file_path = TEST_DNA_FOLDER / dna_folder_name / "head.dna"
    json_file_path = temp_folder / dna_folder_name / "head.json"
    return get_dna_json_data(dna_file_path, json_file_path)


@pytest.fixture(scope="session")
def exported_head_dna_json_data(modify_head_scene, temp_folder, dna_folder_name: str) -> dict:
    from character_dna.dna_io import DNAExporter
    from character_dna.utilities import get_active_head
    from utilities.dna_data import get_dna_json_data

    with saved_scene(modify_head_scene):
        head = get_active_head()
        export_folder = temp_folder / "export" / dna_folder_name
        dna_file_path = export_folder / "head.dna"
        json_file_path = export_folder / "head.json"
        export_folder.mkdir(parents=True, exist_ok=True)

        if head and head.rig_instance:
            head.rig_instance.output.folder_path = str(export_folder)
            DNAExporter(file_name="head.dna", instance=head.rig_instance, linear_modifier=head.linear_modifier).run()
            return get_dna_json_data(dna_file_path, json_file_path)

        return {}


@pytest.fixture(scope="session")
def calibrated_head_dna_json_data(modify_head_scene, temp_folder, dna_folder_name: str) -> dict:
    from character_dna.dna_io import DNACalibrator
    from character_dna.utilities import get_active_head
    from utilities.dna_data import get_dna_json_data

    with saved_scene(modify_head_scene):
        head = get_active_head()
        calibrate_folder = temp_folder / "calibrate" / dna_folder_name
        dna_file_path = calibrate_folder / "head.dna"
        json_file_path = calibrate_folder / "head.json"
        calibrate_folder.mkdir(parents=True, exist_ok=True)

        if head and head.rig_instance:
            head.rig_instance.output.folder_path = str(calibrate_folder)
            DNACalibrator(file_name="head.dna", instance=head.rig_instance, linear_modifier=head.linear_modifier).run()

            return get_dna_json_data(dna_file_path, json_file_path)

        return {}


@pytest.fixture(scope="session")
def calibrated_head_and_body_dna_json_data(modify_head_scene, temp_folder, dna_folder_name: str) -> dict:
    """Export both the body and the head through the calibrator into a single
    output folder, mirroring the real ``Send to MetaHuman Creator`` flow: the body
    is written first, then the head conforms its neck edge loop onto that
    just-written body DNA. ``auto_update_lods`` regenerates every lower-LOD mesh
    via the UV-barycentric solver, so the exported head and body seams only line up
    when the head snaps onto the exported (propagated) body rather than the
    imported template. Returns ``{"head": <json>, "body": <json>}``."""
    from character_dna.dna_io import DNACalibrator
    from character_dna.utilities import get_active_body, get_active_head
    from utilities.dna_data import get_dna_json_data

    with saved_scene(modify_head_scene):
        head = get_active_head()
        body = get_active_body()
        calibrate_folder = temp_folder / "calibrate_seam" / dna_folder_name
        calibrate_folder.mkdir(parents=True, exist_ok=True)

        if not (head and body and head.rig_instance):
            return {}

        instance = head.rig_instance
        instance.output.folder_path = str(calibrate_folder)
        instance.output.method = "calibrate"
        instance.output.align_head_and_body = True
        instance.output.auto_update_lods = True

        # Body first, then the head conforms to the freshly-written body DNA.
        DNACalibrator(
            file_name="body.dna",
            instance=instance,
            linear_modifier=body.linear_modifier,
            component_type="body",
        ).run()
        DNACalibrator(
            file_name="head.dna",
            instance=instance,
            linear_modifier=head.linear_modifier,
            component_type="head",
            seam_reference_dna_path=str(calibrate_folder / "body.dna"),
        ).run()

        return {
            "head": get_dna_json_data(calibrate_folder / "head.dna", calibrate_folder / "head.json"),
            "body": get_dna_json_data(calibrate_folder / "body.dna", calibrate_folder / "body.json"),
        }
