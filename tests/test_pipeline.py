"""Integration test for full end-to-end InspectionPipeline."""

import pytest
from visionqc.pipeline import InspectionPipeline
from data.synthetic_generator import generate_base_gear, inject_crack


def test_pipeline_inspect(tmp_path):
    """Verifies complete inspection execution on PASS and DEFECT_CRACK components."""
    pipe = InspectionPipeline()

    # 1. Inspect nominal gear
    nominal = generate_base_gear()
    report_pass = pipe.inspect(nominal, save_dir=str(tmp_path), prefix="pass_test")

    assert "status" in report_pass
    assert "defect_type" in report_pass
    assert "confidence" in report_pass
    assert "metrics" in report_pass
    assert report_pass["metrics"]["defect_pixel_count"] < 100

    # 2. Inspect cracked gear
    flawed = inject_crack(nominal, length=80, thickness=3)
    report_fail = pipe.inspect(flawed, save_dir=str(tmp_path), prefix="fail_test")

    assert report_fail["status"] == "FAIL"
    assert report_fail["defect_type"] == "DEFECT_CRACK"
    assert report_fail["metrics"]["defect_pixel_count"] > 100
