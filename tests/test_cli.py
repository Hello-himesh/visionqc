"""Unit test suite for VisionQC command-line execution."""

import subprocess
import sys
import os
import pytest


def test_cli_help():
    """Verifies CLI flag definitions and help menu."""
    cmd = [sys.executable, "-m", "visionqc.cli", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "VisionQC" in res.stdout
    assert "--demo" in res.stdout
    assert "--input" in res.stdout


def test_cli_demo(tmp_path):
    """Verifies end-to-end execution of self-contained demo."""
    target_dir = str(tmp_path / "results")
    cmd = [sys.executable, "-m", "visionqc.cli", "--demo", "--save-dir", target_dir]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Inspection Summary" in res.stdout
    assert os.path.exists(target_dir)
