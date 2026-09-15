import os

from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent

# When running on CI (``RUNNING_CI`` is exported by the test workflow) the
# high-cardinality, DNA-driven parametrized tests are aggressively subsampled to
# cut runtime and memory. Local runs leave the env var unset and stay fully
# parametrized.
RUNNING_CI = bool(os.environ.get("RUNNING_CI"))

# CI subsample sizes for the heaviest generators.
CI_BONE_SAMPLE_SIZE = 15
CI_POSE_SAMPLE_SIZE = 5
CI_BODY_POSE_SAMPLE_SIZE = 3  # RBF pose roundtrip is the heaviest per-case test

# Entities the assertions explicitly verify as "changed" must always survive the
# CI subsample, otherwise the change-applied coverage is silently lost. These
# mirror the ``changed_head_*`` fixtures in conftest.py.
CI_REQUIRED_BONE_NAMES = ["FACIAL_C_12IPV_Chin3"]
# The calibrator keeps every head LOD: the lower-LOD propagation assertions only
# run when each ``head_lod{1..7}_mesh`` param is present (see assertions.py).
CI_CALIBRATE_MESH_NAMES = [f"head_lod{index}_mesh" for index in range(8)]
CI_EXPORT_MESH_NAMES = ["head_lod0_mesh"]
CI_SKIN_WEIGHT_MESH_NAMES = ["head_lod0_mesh"]

ADDON_NAME = "character_dna"

TEST_FILES_FOLDER = REPO_ROOT / "tests" / "test_files"
EXTRA_TEST_FILES_FOLDER = REPO_ROOT / "tests" / "extra_test_files"
TEST_FBX_POSES_FOLDER = EXTRA_TEST_FILES_FOLDER / "fbx" / "poses"
TEST_JSON_POSES_FOLDER = TEST_FILES_FOLDER / "json" / "poses"
TEST_ANIMATION_FOLDER = TEST_FILES_FOLDER / "animation"
TEST_DNA_FOLDER = TEST_FILES_FOLDER / "dna"
TEST_FBX_FOLDER = TEST_FILES_FOLDER / "fbx"

HEAD_DNA_FILE = TEST_DNA_FOLDER / "ada" / "head.dna"

BODY_DNA_FILE = TEST_DNA_FOLDER / "ada" / "body.dna"

TOLERANCE = {
    "neutralJointRotations": 1e-3,
    "neutralJointTranslations": 1e-3,
    "normals": 1e-3,
    "positions": 1e-2,  # these assertions are in centimeters
    "textureCoordinates": 1e-3,
    "skinWeights": 1e-5,
}

NORMAL_ROUND_TRIP_BOUNDS = {"mean": 1e-3, "p99": 5e-3, "max": 0.1}
"""How far an exported normal may drift, which is set by Blender's storage rather than by us.

``normals_split_custom_set`` does not hold a normal exactly, and the export rewrites every
normal from the scene rather than only the ones that moved. Measured over Ada's 96008 head
corners the drift is a mean of 5e-5 and a p99 of 3.2e-4, with a worst case of 0.0709 on the
few it cannot represent, so these leave roughly an order of magnitude of headroom.
"""

# Maximum allowed angular difference (in degrees) between an expected and an exported/calibrated
# joint orientation. Joint rotations are compared as a whole-orientation angular difference rather
# than per-axis euler components, since near gimbal lock individual axes can differ noticeably while
# the actual orientation is effectively identical.
ROTATION_ANGLE_TOLERANCE = 0.31

DNA_DEFINITION_VERSION = "defn1.1"

DNA_BEHAVIOR_VERSION = "bhvr1.1"

DNA_GEOMETRY_VERSION = "geom1.1"

DNA_RBF_BEHAVIOR_VERSION = "rbfb1.0"

DNA_RBF_EXTENSION_VERSION = "rbfe1.0"

# TODO: Investigate edge case where only these bone rotation values are always slightly rotated by a few degrees on the x and z.
IGNORED_BONE_ROTATIONS_ON_CALIBRATE = ["FACIAL_C_FacialRoot", "FACIAL_C_Neck1Root", "FACIAL_C_Neck2Root"]
IGNORED_BONE_ROTATIONS_ON_EXPORT = ["FACIAL_C_FacialRoot", "FACIAL_C_Neck1Root", "FACIAL_C_Neck2Root"]

FINGER_NAMES = ["index", "middle", "ring", "pinky", "thumb"]

EXCLUDE_FINGER_POSES = True
