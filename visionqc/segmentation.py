"""Segmentation and Clustering Module (Module 4).

Implements Marker-Controlled Watershed Segmentation and K-Means / K-Medoids
clustering for component separation and surface anomaly grouping.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


class ImageSegmenter:
    """Provides watershed segmentation and color/texture clustering."""

    @staticmethod
    def watershed_segmentation(
        image: np.ndarray,
        distance_threshold_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        """Performs marker-controlled Watershed segmentation to separate overlapping parts.

        Workflow:
        1. Convert image to grayscale and binary threshold (Otsu).
        2. Morphological opening to remove noise.
        3. Dilation to obtain sure background.
        4. Distance transform and thresholding to obtain sure foreground.
        5. Subtract sure foreground from sure background to find unknown region.
        6. Label sure foreground markers with connected components.
        7. Apply cv2.watershed with markers on original color image.

        Args:
            image: BGR color image.
            distance_threshold_ratio: Multiplier of max distance transform to identify foreground.

        Returns:
            Dictionary containing:
            - 'markers': Integer matrix with segment IDs (-1 for boundaries)
            - 'boundary_mask': Binary mask showing watershed boundaries
            - 'num_segments': Number of segmented distinct objects
            - 'segmented_overlay': Color overlay of segmented regions
        """
        if len(image.shape) == 2:
            color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            gray = image.copy()
        else:
            color_img = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Otsu thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological opening to remove background noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area using distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        max_dist = dist_transform.max()
        if max_dist > 0:
            _, sure_fg = cv2.threshold(
                dist_transform, distance_threshold_ratio * max_dist, 255, cv2.THRESH_BINARY
            )
        else:
            sure_fg = np.zeros_like(opening)
        sure_fg = np.uint8(sure_fg)

        # Finding unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labeling
        _, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Mark the region of unknown with zero
        markers[unknown == 255] = 0

        # Apply watershed
        markers = cv2.watershed(color_img, markers)

        boundary_mask = np.uint8(markers == -1) * 255
        num_segments = int(np.max(markers) - 1) if np.max(markers) > 1 else 0

        # Create colored overlay for visualization
        overlay = color_img.copy()
        overlay[markers == -1] = [0, 0, 255]  # Red boundaries

        return {
            "markers": markers,
            "boundary_mask": boundary_mask,
            "num_segments": max(num_segments, 0),
            "segmented_overlay": overlay,
            "distance_map": dist_transform,
        }

    @staticmethod
    def kmeans_clustering(
        image: np.ndarray,
        k: int = 3,
        use_spatial: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Clusters image pixels using K-Means into K groups (e.g. background, metal, defect).

        Args:
            image: BGR color image.
            k: Number of clusters.
            use_spatial: If True, incorporates normalized (x, y) coordinates into feature vectors.

        Returns:
            Tuple of (segmented_image, labels_matrix).
        """
        # Convert to Lab color space for perceptual uniformity
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        h, w, c = lab.shape
        pixel_features = lab.reshape((-1, 3)).astype(np.float32)

        if use_spatial:
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            coords = np.stack((x_coords.flatten() / float(w) * 255.0, y_coords.flatten() / float(h) * 255.0), axis=1)
            pixel_features = np.hstack((pixel_features, coords.astype(np.float32)))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixel_features,
            k,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS,
        )

        # Convert back to uint8 color centers
        color_centers = np.uint8(centers[:, :3])
        segmented_lab = color_centers[labels.flatten()].reshape((h, w, 3))
        segmented_bgr = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2BGR)
        labels_map = labels.reshape((h, w))

        return segmented_bgr, labels_map

    @staticmethod
    def kmedoids_clustering(
        data_points: np.ndarray,
        k: int = 3,
        max_iters: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes K-Medoids clustering on feature vectors.

        Medoids are actual data points minimizing total pairwise distance.

        Args:
            data_points: 2D array of shape (N, D).
            k: Number of medoids.
            max_iters: Maximum optimization iterations.

        Returns:
            Tuple of (medoid_indices, cluster_assignments).
        """
        n_samples = data_points.shape[0]
        if n_samples <= k:
            return np.arange(n_samples), np.arange(n_samples)

        # Initialize medoids randomly
        np.random.seed(42)
        medoid_indices = np.random.choice(n_samples, size=k, replace=False)

        assignments = np.zeros(n_samples, dtype=int)

        for _ in range(max_iters):
            # Compute distances from all samples to current medoids: shape (N, k)
            medoids = data_points[medoid_indices]  # shape (k, D)
            distances = np.linalg.norm(data_points[:, np.newaxis, :] - medoids[np.newaxis, :, :], axis=2)
            new_assignments = np.argmin(distances, axis=1)

            # Update medoids
            new_medoids = medoid_indices.copy()
            for cluster_idx in range(k):
                cluster_members = np.where(new_assignments == cluster_idx)[0]
                if len(cluster_members) == 0:
                    continue
                # Find point that minimizes intra-cluster distance sum
                sub_data = data_points[cluster_members]
                sub_distances = np.linalg.norm(sub_data[:, np.newaxis, :] - sub_data[np.newaxis, :, :], axis=2)
                best_sub_idx = np.argmin(np.sum(sub_distances, axis=1))
                new_medoids[cluster_idx] = cluster_members[best_sub_idx]

            if np.array_equal(medoid_indices, new_medoids):
                break
            medoid_indices = new_medoids

        # Compute final assignments with converged medoids
        final_medoids = data_points[medoid_indices]
        final_distances = np.linalg.norm(data_points[:, np.newaxis, :] - final_medoids[np.newaxis, :, :], axis=2)
        assignments = np.argmin(final_distances, axis=1)

        return medoid_indices, assignments
