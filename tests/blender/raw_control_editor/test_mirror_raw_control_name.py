"""Live DNA drift guards for the Python-only reference catalogs."""

from unit.test_mirror_raw_control_name import EXPECTED_ALL_SHORT_NAMES, RAW_CONTROL_PREFIX


def test_catalog_matches_live_dna() -> None:
    """The hardcoded reference catalog must match every
    ``CTRL_expressions.*`` raw control on the Ada head DNA. If this
    fails, either the DNA changed or the catalog drifted -- update
    :data:`EXPECTED_PAIRS` / :data:`EXPECTED_CENTERS` accordingly."""
    from character_dna.dna_io import get_dna_reader
    from constants import HEAD_DNA_FILE

    reader = get_dna_reader(file_path=HEAD_DNA_FILE, file_format="binary", data_layer="Definition")
    live_short_names: set[str] = set()
    for i in range(int(reader.getRawControlCount())):
        name = str(reader.getRawControlName(i))
        if name.startswith(RAW_CONTROL_PREFIX):
            live_short_names.add(name.removeprefix(RAW_CONTROL_PREFIX))

    missing_from_catalog = live_short_names - EXPECTED_ALL_SHORT_NAMES
    extra_in_catalog = EXPECTED_ALL_SHORT_NAMES - live_short_names
    assert not missing_from_catalog, f"present in DNA but missing from test catalog: {sorted(missing_from_catalog)}"
    assert not extra_in_catalog, f"listed in test catalog but absent from DNA: {sorted(extra_in_catalog)}"
