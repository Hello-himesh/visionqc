"""Unit test suite for FeatureDetector geometric extraction methods."""

import pytest
import numpy as np
import cv2
from visionqc.edge_corner import FeatureDetector


@pytest.fixture
def geometric_image():
    """Generates a 120x120 test image with a solid square and interior circle."""
    matrix = np.zeros((120, 120), dtype=np.uint8)
    cv2.rectangle(matrix, (20, 20), (80, 80), 255, -1)
    cv2.circle(matrix, (50, 50), 15, 0, -1)
    return matrix


def test_sobel_edges(geometric_image):
    """Verifies gradient magnitude, component derivatives, and angular orientation."""
    res = FeatureDetector.sobel_edges(geometric_image)
    assert "gx" in res
    assert "gy" in res
    assert "magnitude" in res
    assert "orientation" in res
    assert res["magnitude"].shape == geometric_image.shape


def test_canny_edges(geometric_image):
    """Verifies Canny hysteresis edge map generation."""
    edges = FeatureDetector.canny_edges(geometric_image)
    assert edges.shape == geometric_image.shape
    assert np.max(edges) == 255


def test_harris_corners(geometric_image):
    """Verifies Harris corner detector keypoint localization."""
    dst, corners = FeatureDetector.harris_corners(geometric_image)
    assert dst.shape == geometric_image.shape
    assert len(corners) > 0
    # Corner coordinates should be around the square perimeter vertices (e.g., around 20, 20)
    assert any(15 <= x <= 25 and 15 <= y <= 25 for x, y in corners)


def test_shi_tomasi_corners(geometric_image):
    """Verifies Shi-Tomasi good features to track extraction."""
    corners = FeatureDetector.shi_tomasi_corners(geometric_image, max_corners=10)
    assert len(corners) > 0


def test_hough_lines(geometric_image):
    """Verifies probabilistic Hough transform line segment identification."""
    lines = FeatureDetector.detect_hough_lines(geometric_image, threshold=20, min_line_length=15)
    assert isinstance(lines, list)


def test_hough_circles():
    """Verifies circular bore detection via Hough gradient transform."""
    canvas = np.zeros((150, 150), dtype=np.uint8)
    cv2.circle(canvas, (75, 75), 30, 255, 3)
    circles = FeatureDetector.detect_hough_circles(canvas, min_radius=20, max_radius=40)
    assert len(circles) > 0
    found = circles[0]
    assert abs(found["center"][0] - 75) <= 5
    assert abs(found["center"][1] - 75) <= 5
