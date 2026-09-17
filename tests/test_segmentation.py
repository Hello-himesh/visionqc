"""Unit tests for ImageSegmenter (Module 4)."""

import pytest
import numpy as np
import cv2
from visionqc.segmentation import ImageSegmenter


@pytest.fixture
def touching_circles_image():
    # Synthetic image with two touching circles
    img = np.zeros((150, 150, 3), dtype=np.uint8)
    cv2.circle(img, (50, 75), 30, (200, 200, 200), -1)
    cv2.circle(img, (95, 75), 30, (200, 200, 200), -1)
    return img


def test_watershed_segmentation(touching_circles_image):
    result = ImageSegmenter.watershed_segmentation(touching_circles_image)
    assert "markers" in result
    assert "boundary_mask" in result
    assert "num_segments" in result
    assert "segmented_overlay" in result
    assert result["boundary_mask"].shape == touching_circles_image.shape[:2]


def test_kmeans_clustering(touching_circles_image):
    segmented, labels = ImageSegmenter.kmeans_clustering(touching_circles_image, k=2)
    assert segmented.shape == touching_circles_image.shape
    assert labels.shape == touching_circles_image.shape[:2]
    assert len(np.unique(labels)) <= 2


def test_kmedoids_clustering():
    data = np.array([
        [1.0, 1.0],
        [1.2, 0.9],
        [0.8, 1.1],
        [10.0, 10.0],
        [9.8, 10.2],
        [10.1, 9.9],
    ])
    medoids, assignments = ImageSegmenter.kmedoids_clustering(data, k=2)
    assert len(medoids) == 2
    assert len(assignments) == 6
    # Points 0, 1, 2 should belong to the same cluster
    assert assignments[0] == assignments[1] == assignments[2]
    # Points 3, 4, 5 should belong to another cluster
    assert assignments[3] == assignments[4] == assignments[5]
    assert assignments[0] != assignments[3]
