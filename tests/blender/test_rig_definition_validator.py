# standard library imports
import gzip
import json

# third party imports
import pytest

# local imports
from character_dna.rig_definition import (
    BodyRigDefinition,
    HeadRigDefinition,
    RigDefinition,
    full_path,
    get_rig_definition,
)
from character_dna.validators.rig_definition import (
    RigDefinitionValidator,
    validate_dna_compatibility,
)


HEAD_DB_NAME = "MH.6"
BODY_DB_NAME = "MHB.1"


class FakeDnaReader:
    """A minimal stand-in for ``riglogic.BinaryStreamReader``.

    It exposes only the getters the validator uses, derived from a rig
    definition so a pristine instance validates cleanly.
    """

    def __init__(self, definition: RigDefinition) -> None:
        self._lod_count = definition.lod_count
        self._joint_count = definition.joint_count
        self._joint_group_count = definition.joint_group_count
        self._mesh_names = [mesh.name for mesh in definition.meshes]
        self._vertex_counts = {mesh.name: mesh.vertex_count for mesh in definition.meshes}

    def getLODCount(self) -> int:
        return self._lod_count

    def getJointCount(self) -> int:
        return self._joint_count

    def getJointGroupCount(self) -> int:
        return self._joint_group_count

    def getMeshCount(self) -> int:
        return len(self._mesh_names)

    def getMeshName(self, index: int) -> str:
        return self._mesh_names[index]

    def getVertexPositionCount(self, index: int) -> int:
        return self._vertex_counts[self._mesh_names[index]]


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------
@pytest.fixture
def head_definition() -> HeadRigDefinition:
    return HeadRigDefinition.load()


@pytest.fixture
def body_definition() -> BodyRigDefinition:
    return BodyRigDefinition.load()


def test_head_definition_defaults_to_mh6(head_definition: HeadRigDefinition):
    assert head_definition.db_name == HEAD_DB_NAME
    assert head_definition.component == "head"
    assert head_definition.schema_version == 1
    assert head_definition.lod_count > 0
    assert head_definition.joint_count == len(head_definition.joint_names)
    assert head_definition.joint_group_count == len(head_definition.joint_group_names)
    assert head_definition.region_names
    assert head_definition.meshes, "Expected at least one mesh in the rig definition."


def test_body_definition_defaults_to_mhb1(body_definition: BodyRigDefinition):
    assert body_definition.db_name == BODY_DB_NAME
    assert body_definition.component == "body"
    assert body_definition.schema_version == 1
    assert body_definition.lod_count > 0
    assert body_definition.joint_count == len(body_definition.joint_names)
    assert body_definition.joint_group_count == len(body_definition.joint_group_names)
    assert body_definition.meshes, "Expected at least one mesh in the body rig definition."


def test_get_rig_definition_is_cached_per_session():
    first = get_rig_definition("head")
    assert get_rig_definition("head") is first
    assert get_rig_definition("head", HEAD_DB_NAME) is first


# ----------------------------------------------------------------------------------------------
# Validator
# ----------------------------------------------------------------------------------------------
def test_matching_head_dna_is_valid(head_definition: HeadRigDefinition):
    report = RigDefinitionValidator(head_definition, FakeDnaReader(head_definition)).validate()
    assert report.is_valid
    assert report.errors == []


def test_matching_body_dna_is_valid(body_definition: BodyRigDefinition):
    report = RigDefinitionValidator(body_definition, FakeDnaReader(body_definition)).validate()
    assert report.is_valid
    assert report.errors == []


def test_lod_count_mismatch_is_detected(head_definition: HeadRigDefinition):
    reader = FakeDnaReader(head_definition)
    reader._lod_count = head_definition.lod_count + 1
    report = RigDefinitionValidator(head_definition, reader).validate()
    assert not report.is_valid
    assert any(issue.code == "lod_count_mismatch" for issue in report.errors)


def test_joint_count_mismatch_is_detected(head_definition: HeadRigDefinition):
    reader = FakeDnaReader(head_definition)
    reader._joint_count = head_definition.joint_count - 1
    report = RigDefinitionValidator(head_definition, reader).validate()
    assert not report.is_valid
    assert any(issue.code == "joint_count_mismatch" for issue in report.errors)


def test_joint_group_count_mismatch_is_detected(head_definition: HeadRigDefinition):
    reader = FakeDnaReader(head_definition)
    reader._joint_group_count = head_definition.joint_group_count + 2
    report = RigDefinitionValidator(head_definition, reader).validate()
    assert not report.is_valid
    assert any(issue.code == "joint_group_count_mismatch" for issue in report.errors)


def test_vertex_count_mismatch_is_detected(head_definition: HeadRigDefinition):
    reader = FakeDnaReader(head_definition)
    first_mesh = head_definition.meshes[0].name
    reader._vertex_counts[first_mesh] += 10
    report = RigDefinitionValidator(head_definition, reader).validate()
    assert not report.is_valid
    assert any(issue.code == "vertex_count_mismatch" for issue in report.errors)


def test_missing_mesh_is_detected(head_definition: HeadRigDefinition):
    reader = FakeDnaReader(head_definition)
    removed = reader._mesh_names.pop()
    del reader._vertex_counts[removed]
    report = RigDefinitionValidator(head_definition, reader).validate()
    assert not report.is_valid
    assert any(issue.code == "mesh_missing" for issue in report.errors)


def test_unexpected_mesh_is_a_warning(head_definition: HeadRigDefinition):
    reader = FakeDnaReader(head_definition)
    reader._mesh_names.append("not_a_real_mesh")
    reader._vertex_counts["not_a_real_mesh"] = 42
    report = RigDefinitionValidator(head_definition, reader).validate()
    # An unexpected mesh alone should not invalidate the DNA.
    assert report.is_valid
    assert any(issue.code == "mesh_unexpected" for issue in report.warnings)


def test_validate_dna_compatibility_head(head_definition: HeadRigDefinition):
    report = validate_dna_compatibility(FakeDnaReader(head_definition), component="head")
    assert report.db_name == HEAD_DB_NAME
    assert report.is_valid


def test_validate_dna_compatibility_body(body_definition: BodyRigDefinition):
    report = validate_dna_compatibility(FakeDnaReader(body_definition), component="body")
    assert report.db_name == BODY_DB_NAME
    assert report.is_valid


def test_validate_dna_compatibility_unknown_component(head_definition: HeadRigDefinition):
    with pytest.raises(ValueError, match="Unknown rig-definition component"):
        validate_dna_compatibility(FakeDnaReader(head_definition), component="nope")


# ----------------------------------------------------------------------------------------------
# Rich (lazy) data
# ----------------------------------------------------------------------------------------------
def test_head_definition_lazily_loads_rich_data():
    definition = HeadRigDefinition.load()
    assert definition.db_name == HEAD_DB_NAME

    # Joint colors for bone import.
    assert definition.joints
    assert len(definition.joints[0].color) == 3

    # Joint-group membership for bone collections.
    assert definition.joint_groups
    assert definition.joint_groups[0].joints

    # Region membership for vertex groups.
    assert definition.regions
    assert definition.regions[0].joints

    # PSD nets / definitions for future feature work.
    assert definition.psd_definitions
    assert definition.psd_nets
    assert definition.expressions


def test_body_definition_lazily_loads_rich_data():
    definition = BodyRigDefinition.load()
    assert definition.db_name == BODY_DB_NAME

    # Joint hierarchy is present; DNA carries no joint colors/radii.
    assert definition.joints
    assert definition.joints[0].color is None
    assert definition.joints[0].radius is None
    # The root joint has no parent.
    assert any(joint.parent is None for joint in definition.joints)

    # Joint-group membership is available.
    assert definition.joint_groups


def test_head_full_file_is_gzip_compressed():
    file_path = full_path(HEAD_DB_NAME, "head")
    with gzip.open(file_path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["db_name"] == HEAD_DB_NAME


def test_body_full_file_is_gzip_compressed():
    file_path = full_path(BODY_DB_NAME, "body")
    with gzip.open(file_path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["db_name"] == BODY_DB_NAME
