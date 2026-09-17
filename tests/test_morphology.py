"""Unit tests for MorphologicalAnalyzer (Module 2)."""

import pytest
import numpy as np
import cv2
from visionqc.morphology import MorphologicalAnalyzer


@pytest.fixture
def binary_test_mask():
    mask = np.zeros((80, 80), dtype=np.uint8)
    # Draw a central rectangle
    mask[20:60, 20:60] = 255
    # Add a small hole inside
    mask[35:45, 35:45] = 0
    # Add small speckle noise outside
    mask[5, 5] = 255
    mask[75, 75] = 255
    return mask


def test_structuring_elements():
    rect = MorphologicalAnalyzer.get_kernel("rect", (5, 5))
    assert rect.shape == (5, 5)
    cross = MorphologicalAnalyzer.get_kernel("cross", (3, 3))
    assert cross.shape == (3, 3)
    ellipse = MorphologicalAnalyzer.get_kernel("ellipse", (7, 7))
    assert ellipse.shape == (7, 7)

    with pytest.raises(ValueError):
        MorphologicalAnalyzer.get_kernel("invalid_shape")


def test_erosion_dilation(binary_test_mask):
    eroded = MorphologicalAnalyzer.erode(binary_test_mask)
    assert np.sum(eroded) < np.sum(binary_test_mask)

    dilated = MorphologicalAnalyzer.dilate(binary_test_mask)
    assert np.sum(dilated) > np.sum(binary_test_mask)


def test_opening_closing(binary_test_mask):
    opened = MorphologicalAnalyzer.open(binary_test_mask)
    # Opening should remove small 1-pixel speckles
    assert opened[5, 5] == 0
    assert opened[75, 75] == 0

    closed = MorphologicalAnalyzer.close(binary_test_mask, kernel=MorphologicalAnalyzer.get_kernel("rect", (15, 15)))
    # Closing should fill the 10x10 hole inside
    assert closed[40, 40] == 255


def test_transforms(binary_test_mask):
    grad = MorphologicalAnalyzer.gradient(binary_test_mask)
    assert grad.shape == binary_test_mask.shape

    tophat = MorphologicalAnalyzer.top_hat(binary_test_mask)
    assert tophat.shape == binary_test_mask.shape

    blackhat = MorphologicalAnalyzer.black_hat(binary_test_mask)
    assert blackhat.shape == binary_test_mask.shape


def test_clean_defect_mask(binary_test_mask):
    cleaned = MorphologicalAnalyzer.clean_defect_mask(binary_test_mask, min_area=50, fill_holes=True)
    assert cleaned[5, 5] == 0
    assert cleaned[75, 75] == 0
    # Main object retained
    assert cleaned[30, 30] == 255
