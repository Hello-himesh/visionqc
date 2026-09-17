"""Unit tests for ImagePreprocessor (Modules 1 & 2)."""

import pytest
import numpy as np
import cv2
from visionqc.preprocessor import ImagePreprocessor


@pytest.fixture
def sample_image():
    """Generates a synthetic 100x100 3-channel test image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = [120, 150, 200]
    return img


def test_convert_color(sample_image):
    gray = ImagePreprocessor.convert_color(sample_image, "GRAY")
    assert len(gray.shape) == 2
    assert gray.shape == (100, 100)

    hsv = ImagePreprocessor.convert_color(sample_image, "HSV")
    assert hsv.shape == (100, 100, 3)

    lab = ImagePreprocessor.convert_color(sample_image, "LAB")
    assert lab.shape == (100, 100, 3)


def test_resize_with_aspect_ratio(sample_image):
    resized = ImagePreprocessor.resize_with_aspect_ratio(sample_image, width=50)
    assert resized.shape[1] == 50
    assert resized.shape[0] == 50


def test_flip(sample_image):
    flipped = ImagePreprocessor.flip(sample_image, "horizontal")
    assert flipped.shape == sample_image.shape


def test_remove_noise(sample_image):
    noisy = sample_image.copy()
    noisy[10, 10] = [255, 255, 255]

    denoised_g = ImagePreprocessor.remove_noise(noisy, method="gaussian", kernel_size=3)
    assert denoised_g.shape == noisy.shape

    denoised_m = ImagePreprocessor.remove_noise(noisy, method="median", kernel_size=3)
    assert denoised_m.shape == noisy.shape

    denoised_b = ImagePreprocessor.remove_noise(noisy, method="bilateral", kernel_size=3)
    assert denoised_b.shape == noisy.shape


def test_linear_contrast_stretch(sample_image):
    stretched = ImagePreprocessor.linear_contrast_stretch(sample_image)
    assert stretched.dtype == np.uint8
    assert stretched.shape == sample_image.shape


def test_log_transform(sample_image):
    transformed = ImagePreprocessor.log_transform(sample_image)
    assert transformed.dtype == np.uint8
    assert transformed.shape == sample_image.shape


def test_power_law_transform(sample_image):
    gamma_corr = ImagePreprocessor.power_law_transform(sample_image, gamma=0.5)
    assert gamma_corr.dtype == np.uint8
    assert gamma_corr.shape == sample_image.shape


def test_equalize_histogram(sample_image):
    eq = ImagePreprocessor.equalize_histogram(sample_image, use_clahe=True)
    assert eq.shape == sample_image.shape

    gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    eq_gray = ImagePreprocessor.equalize_histogram(gray, use_clahe=False)
    assert eq_gray.shape == gray.shape
