"""Unit tests for FeatureDetector (Module 3)."""

import pytest
import numpy as np
import cv2
from visionqc.edge_corner import FeatureDetector


@pytest.fixture
def geometric_image():
    img = np.zeros((120, 120), dtype=np.uint8)
    # Draw rectangle (edges and corners)
    cv2.rectangle(img, (20, 20), (80, 80), 255, -1)
    # Draw circle inside
    cv2.circle(img, (50, 50), 15, 0, -1)
    return img


def test_sobel_edges(geometric_image):
    res = FeatureDetector.sobel_edges(geometric_image)
    assert "gx" in res
    assert "gy" in res
    assert "magnitude" in res
    assert "orientation" in res
    assert res["magnitude"].shape == geometric_image.shape


def test_canny_edges(geometric_image):
    edges = FeatureDetector.canny_edges(geometric_image)
    assert edges.shape == geometric_image.shape
    assert np.max(edges) == 255


def test_harris_corners(geometric_image):
    dst, corners = FeatureDetector.harris_corners(geometric_image)
    assert dst.shape == geometric_image.shape
    assert len(corners) > 0
    # Corners should include vertices around (20,20), (80,20), etc.
    assert any(15 <= x <= 25 and 15 <= y <= 25 for x, y in corners)


def test_shi_tomasi_corners(geometric_image):
    corners = FeatureDetector.shi_tomasi_corners(geometric_image, max_corners=10)
    assert len(corners) > 0


def test_hough_lines(geometric_image):
    lines = FeatureDetector.detect_hough_lines(geometric_image, threshold=20, min_line_length=15)
    assert isinstance(lines, list)


def test_hough_circles():
    # Synthetic image with a distinct circle
    circle_img = np.zeros((150, 150), dtype=np.uint8)
    cv2.circle(circle_img, (75, 75), 30, 255, 3)
    circles = FeatureDetector.detect_hough_circles(circle_img, min_radius=20, max_radius=40)
    assert len(circles) > 0
    c = circles[0]
    assert abs(c["center"][0] - 75) <= 5
    assert abs(c["center"][1] - 75) <= 5
