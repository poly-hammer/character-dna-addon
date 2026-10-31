"""Live DNA drift guards for the Python-only reference catalogs."""

from unit.test_shape_key_side_classification import EXPECTED_ALL_NAMES


def test_catalog_matches_live_dna() -> None:
    from character_dna.dna_io import get_dna_reader
    from constants import HEAD_DNA_FILE

    reader = get_dna_reader(file_path=HEAD_DNA_FILE, file_format="binary", data_layer="All")
    live_names = {str(reader.getBlendShapeChannelName(i)) for i in range(int(reader.getBlendShapeChannelCount()))}

    missing = live_names - EXPECTED_ALL_NAMES
    extra = EXPECTED_ALL_NAMES - live_names
    assert not missing, f"present in DNA but missing from catalog: {sorted(missing)}"
    assert not extra, f"listed in catalog but absent from DNA: {sorted(extra)}"
