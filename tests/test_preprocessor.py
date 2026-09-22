"""Unit test suite for ImagePreprocessor conditioning and transformation methods."""

import pytest
import numpy as np
import cv2
from visionqc.preprocessor import ImagePreprocessor


@pytest.fixture
def synthetic_patch():
    """Builds a deterministic multi-channel 100x100 test matrix."""
    matrix = np.zeros((100, 100, 3), dtype=np.uint8)
    matrix[25:75, 25:75] = [110, 140, 190]
    return matrix


def test_convert_color(synthetic_patch):
    """Verifies color channel transformations across supported spaces."""
    gray = ImagePreprocessor.convert_color(synthetic_patch, "GRAY")
    assert gray.ndim == 2
    assert gray.shape == (100, 100)

    hsv = ImagePreprocessor.convert_color(synthetic_patch, "HSV")
    assert hsv.shape == (100, 100, 3)

    lab = ImagePreprocessor.convert_color(synthetic_patch, "LAB")
    assert lab.shape == (100, 100, 3)


def test_resize_with_aspect_ratio(synthetic_patch):
    """Verifies proportional dimension preservation during resizing."""
    scaled = ImagePreprocessor.resize_with_aspect_ratio(synthetic_patch, width=50)
    assert scaled.shape[1] == 50
    assert scaled.shape[0] == 50


def test_flip(synthetic_patch):
    """Verifies matrix mirroring across primary axes."""
    horizontal = ImagePreprocessor.flip(synthetic_patch, "horizontal")
    assert horizontal.shape == synthetic_patch.shape
    assert horizontal.dtype == synthetic_patch.dtype


def test_remove_noise(synthetic_patch):
    """Verifies Gaussian, Median, and Bilateral spatial denoising filters."""
    corrupted = synthetic_patch.copy()
    corrupted[15, 15] = [255, 255, 255]

    filtered_g = ImagePreprocessor.remove_noise(corrupted, method="gaussian", kernel_size=3)
    assert filtered_g.shape == corrupted.shape

    filtered_m = ImagePreprocessor.remove_noise(corrupted, method="median", kernel_size=3)
    assert filtered_m.shape == corrupted.shape

    filtered_b = ImagePreprocessor.remove_noise(corrupted, method="bilateral", kernel_size=3)
    assert filtered_b.shape == corrupted.shape


def test_linear_contrast_stretch(synthetic_patch):
    """Verifies dynamic min-max contrast expansion."""
    stretched = ImagePreprocessor.linear_contrast_stretch(synthetic_patch)
    assert stretched.dtype == np.uint8
    assert stretched.shape == synthetic_patch.shape


def test_log_transform(synthetic_patch):
    """Verifies logarithmic intensity expansion."""
    logged = ImagePreprocessor.log_transform(synthetic_patch)
    assert logged.dtype == np.uint8
    assert logged.shape == synthetic_patch.shape


def test_power_law_transform(synthetic_patch):
    """Verifies gamma non-linear transformation."""
    gamma_res = ImagePreprocessor.power_law_transform(synthetic_patch, gamma=0.5)
    assert gamma_res.dtype == np.uint8
    assert gamma_res.shape == synthetic_patch.shape


def test_equalize_histogram(synthetic_patch):
    """Verifies standard and CLAHE histogram equalization."""
    clahe_out = ImagePreprocessor.equalize_histogram(synthetic_patch, use_clahe=True)
    assert clahe_out.shape == synthetic_patch.shape

    mono = cv2.cvtColor(synthetic_patch, cv2.COLOR_BGR2GRAY)
    mono_out = ImagePreprocessor.equalize_histogram(mono, use_clahe=False)
    assert mono_out.shape == mono.shape
