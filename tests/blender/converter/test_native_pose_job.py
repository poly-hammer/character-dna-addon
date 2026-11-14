"""The in-process optimization preserves the process worker's result contract."""

import time

import numpy as np
import pytest

from character_dna.editors.converter.native.pose.job import NativePoseJob
from character_dna.editors.converter.pose.job import PoseJob

from .test_a_pose import toy


def test_native_pose_job_matches_process_result_and_keeps_exact_endpoint():
    arrays, profile = toy()
    native, process = NativePoseJob(arrays, profile), PoseJob(arrays, profile)
    try:
        native.commit()
        process.commit()
        assert native.report["native_worker"]["transport"] == "owned-rust-thread"
        assert native.task is None
        assert native.poll()
        assert native.previews
        for key in process.result:
            np.testing.assert_array_equal(native.result[key], process.result[key])
        np.testing.assert_array_equal(native.previews[-1][2], native.result["positions"].astype(np.float32))
    finally:
        native.close()
        process.close()


def test_native_pose_cancel_joins_and_rejects_late_poll():
    arrays, profile = toy()
    job = NativePoseJob(arrays, profile)
    started = time.monotonic()
    job.close()
    assert time.monotonic() - started < 2
    assert job.task is None
    with pytest.raises(ValueError, match="cancelled"):
        job.poll()
    job.close()
