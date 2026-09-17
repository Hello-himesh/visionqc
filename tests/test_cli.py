"""Unit tests for VisionQC CLI."""

import subprocess
import sys
import os
import pytest


def test_cli_help():
    cmd = [sys.executable, "-m", "visionqc.cli", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "VisionQC" in res.stdout
    assert "--demo" in res.stdout
    assert "--input" in res.stdout


def test_cli_demo(tmp_path):
    out_dir = str(tmp_path / "results")
    cmd = [sys.executable, "-m", "visionqc.cli", "--demo", "--save-dir", out_dir]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Inspection Summary" in res.stdout
    assert os.path.exists(out_dir)
