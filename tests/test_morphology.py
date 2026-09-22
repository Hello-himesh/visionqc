"""Unit test suite for MorphologicalAnalyzer operators and defect filtering."""

import pytest
import numpy as np
from visionqc.morphology import MorphologicalAnalyzer


@pytest.fixture
def binary_test_mask():
    """Generates an 80x80 binary fixture featuring an outer box, internal hole, and speckles."""
    canvas = np.zeros((80, 80), dtype=np.uint8)
    canvas[20:60, 20:60] = 255
    canvas[35:45, 35:45] = 0
    canvas[5, 5] = 255
    canvas[75, 75] = 255
    return canvas


def test_structuring_elements():
    """Verifies generation of rectangular, cross, and elliptical footprints."""
    rect = MorphologicalAnalyzer.get_kernel("rect", (5, 5))
    assert rect.shape == (5, 5)

    cross = MorphologicalAnalyzer.get_kernel("cross", (3, 3))
    assert cross.shape == (3, 3)

    ellipse = MorphologicalAnalyzer.get_kernel("ellipse", (7, 7))
    assert ellipse.shape == (7, 7)

    with pytest.raises(ValueError):
        MorphologicalAnalyzer.get_kernel("invalid_shape")


def test_erosion_dilation(binary_test_mask):
    """Verifies foreground contraction via erosion and expansion via dilation."""
    eroded = MorphologicalAnalyzer.erode(binary_test_mask)
    assert np.sum(eroded) < np.sum(binary_test_mask)

    dilated = MorphologicalAnalyzer.dilate(binary_test_mask)
    assert np.sum(dilated) > np.sum(binary_test_mask)


def test_opening_closing(binary_test_mask):
    """Verifies opening eliminates fine speckles and closing bridges internal gaps."""
    opened = MorphologicalAnalyzer.open(binary_test_mask)
    assert opened[5, 5] == 0
    assert opened[75, 75] == 0

    large_k = MorphologicalAnalyzer.get_kernel("rect", (15, 15))
    closed = MorphologicalAnalyzer.close(binary_test_mask, kernel=large_k)
    assert closed[40, 40] == 255


def test_transforms(binary_test_mask):
    """Verifies gradient, top-hat, and black-hat non-linear morphological operations."""
    grad = MorphologicalAnalyzer.gradient(binary_test_mask)
    assert grad.shape == binary_test_mask.shape

    tophat = MorphologicalAnalyzer.top_hat(binary_test_mask)
    assert tophat.shape == binary_test_mask.shape

    blackhat = MorphologicalAnalyzer.black_hat(binary_test_mask)
    assert blackhat.shape == binary_test_mask.shape


def test_clean_defect_mask(binary_test_mask):
    """Verifies removal of noise particles while preserving genuine defect structures."""
    cleaned = MorphologicalAnalyzer.clean_defect_mask(binary_test_mask, min_area=50, fill_holes=True)
    assert cleaned[5, 5] == 0
    assert cleaned[75, 75] == 0
    assert cleaned[30, 30] == 255
