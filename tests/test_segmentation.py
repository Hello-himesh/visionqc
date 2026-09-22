"""Unit test suite for ImageSegmenter partitioning and clustering methods."""

import pytest
import numpy as np
import cv2
from visionqc.segmentation import ImageSegmenter


@pytest.fixture
def touching_circles_image():
    """Generates an image of two adjacent touching disks on dark background."""
    canvas = np.zeros((150, 150, 3), dtype=np.uint8)
    cv2.circle(canvas, (50, 75), 30, (200, 200, 200), -1)
    cv2.circle(canvas, (95, 75), 30, (200, 200, 200), -1)
    return canvas


def test_watershed_segmentation(touching_circles_image):
    """Verifies marker-controlled watershed segmentation outputs and boundaries."""
    result = ImageSegmenter.watershed_segmentation(touching_circles_image)
    assert "markers" in result
    assert "boundary_mask" in result
    assert "num_segments" in result
    assert "segmented_overlay" in result
    assert result["boundary_mask"].shape == touching_circles_image.shape[:2]


def test_kmeans_clustering(touching_circles_image):
    """Verifies color quantization in Lab space using K-Means."""
    quantized, label_map = ImageSegmenter.kmeans_clustering(touching_circles_image, k=2)
    assert quantized.shape == touching_circles_image.shape
    assert label_map.shape == touching_circles_image.shape[:2]
    assert len(np.unique(label_map)) <= 2


def test_kmedoids_clustering():
    """Verifies K-Medoids centroid selection and pairwise distance partitioning."""
    points = np.array([
        [1.0, 1.0],
        [1.2, 0.9],
        [0.8, 1.1],
        [10.0, 10.0],
        [9.8, 10.2],
        [10.1, 9.9],
    ])
    medoids, cluster_labels = ImageSegmenter.kmedoids_clustering(points, k=2)
    assert len(medoids) == 2
    assert len(cluster_labels) == 6
    # Cluster 1: first three points
    assert cluster_labels[0] == cluster_labels[1] == cluster_labels[2]
    # Cluster 2: second three points
    assert cluster_labels[3] == cluster_labels[4] == cluster_labels[5]
    assert cluster_labels[0] != cluster_labels[3]
