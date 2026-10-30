"""Live DNA drift guards for the Python-only reference catalogs."""

from unit.test_mirror_rbf_name import EXPECTED_ALL_POSES, EXPECTED_ALL_SOLVERS


def test_solver_catalog_matches_live_dna() -> None:
    from character_dna.dna_io import get_dna_reader
    from constants import BODY_DNA_FILE

    reader = get_dna_reader(file_path=BODY_DNA_FILE, file_format="binary", data_layer="All")
    live = {str(reader.getRBFSolverName(i)) for i in range(int(reader.getRBFSolverCount()))}
    missing = live - EXPECTED_ALL_SOLVERS
    extra = EXPECTED_ALL_SOLVERS - live
    assert not missing, f"present in DNA but missing from catalog: {sorted(missing)}"
    assert not extra, f"listed in catalog but absent from DNA: {sorted(extra)}"


def test_pose_catalog_matches_live_dna() -> None:
    from character_dna.dna_io import get_dna_reader
    from constants import BODY_DNA_FILE

    reader = get_dna_reader(file_path=BODY_DNA_FILE, file_format="binary", data_layer="All")
    live = {str(reader.getRBFPoseName(i)) for i in range(int(reader.getRBFPoseCount()))}
    missing = live - EXPECTED_ALL_POSES
    extra = EXPECTED_ALL_POSES - live
    assert not missing, f"present in DNA but missing from catalog: {sorted(missing)}"
    assert not extra, f"listed in catalog but absent from DNA: {sorted(extra)}"
