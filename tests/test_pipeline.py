"""Unit tests for InspectionPipeline."""

import pytest
import numpy as np
import cv2
from visionqc.pipeline import InspectionPipeline
from data.synthetic_generator import generate_base_gear, inject_crack


def test_pipeline_inspect(tmp_path):
    pipeline = InspectionPipeline()

    # Test normal gear
    gear = generate_base_gear()
    report_pass = pipeline.inspect(gear, save_dir=str(tmp_path), prefix="pass_test")

    assert "status" in report_pass
    assert "defect_type" in report_pass
    assert "confidence" in report_pass
    assert "metrics" in report_pass
    assert report_pass["metrics"]["defect_pixel_count"] < 100

    # Test cracked gear
    cracked = inject_crack(gear, length=80, thickness=3)
    report_fail = pipeline.inspect(cracked, save_dir=str(tmp_path), prefix="fail_test")

    assert report_fail["status"] == "FAIL"
    assert report_fail["defect_type"] == "DEFECT_CRACK"
    assert report_fail["metrics"]["defect_pixel_count"] > 100
