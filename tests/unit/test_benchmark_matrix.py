"""A quick pair or a missing platform must not satisfy whole-matrix acceptance."""

import json

import pytest

from scripts.test_suite_refactor.combine_benchmarks import LANES, combine


@pytest.fixture
def matrix(tmp_path):
    for system, blender in LANES:
        for repeat in (1, 2, 3):
            folder = tmp_path / f"test-refactor-{system}-{blender}-{repeat}"
            folder.mkdir()
            pair = {
                "repeat": repeat,
                "accepted": True,
                "baseline_seconds": 100 + repeat,
                "candidate_seconds": 20 + repeat,
                "baseline_source": "original",
                "candidate_source": "refactor",
                "cache_policy": "cold",
                "environment": {
                    "python": "3.11.9" if blender == "4.5" else "3.13.11",
                    "blender": blender,
                    "coverage": "7.13.5",
                    "pytest": "9.0.2",
                    "ci_corpus": True,
                },
            }
            (folder / "summary.json").write_text(json.dumps({"accepted": True, "pairs": [pair]}))
            (folder / f"comparison-{repeat}.json").write_text(json.dumps({"accepted": True}))
    return tmp_path


def test_all_five_lanes_require_three_complete_comparisons(matrix):
    result = combine(matrix)
    assert result["accepted"]
    assert len(result["lanes"]) == 5
    assert not result["errors"]
    assert result["total_process_ratio"] == pytest.approx(22 / 102)
    assert result["critical_path_process_ratio"] == pytest.approx(22 / 102)
    assert result["lanes"][0]["candidate"] == {"median_seconds": 22, "min_seconds": 21, "max_seconds": 23}


@pytest.mark.parametrize("fault", ["missing", "regression", "ratio", "cache", "source", "environment", "wrong_lane"])
def test_one_invalid_repetition_rejects_the_complete_matrix(matrix, fault):
    folder = matrix / "test-refactor-windows-latest-4.5-2"
    path = folder / "summary.json"
    summary = json.loads(path.read_text())
    pair = summary["pairs"][0]
    if fault == "missing":
        path.unlink()
    elif fault == "regression":
        (folder / "comparison-2.json").write_text(json.dumps({"accepted": False}))
    else:
        if fault == "ratio":
            pair["candidate_seconds"] = 30
        elif fault == "cache":
            pair["cache_policy"] = "shared"
        elif fault == "source":
            pair["candidate_source"] = "changed"
        elif fault == "wrong_lane":
            # Even three consistently misconfigured runs cannot stand in for
            # the requested Blender/Python lane.
            for repeat in (1, 2, 3):
                target = matrix / f"test-refactor-windows-latest-4.5-{repeat}" / "summary.json"
                data = json.loads(target.read_text())
                data["pairs"][0]["environment"]["blender"] = "5.2"
                target.write_text(json.dumps(data))
            pair["environment"]["blender"] = "5.2"
        else:
            pair["environment"]["coverage"] = None
        path.write_text(json.dumps(summary))
    result = combine(matrix)
    assert not result["accepted"]
    assert result["errors"]
