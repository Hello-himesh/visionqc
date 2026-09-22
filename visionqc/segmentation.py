"""Image Segmentation and Cluster Grouping Subsystem (Module 4).

Implements marker-controlled Watershed segmentation driven by distance transforms,
K-Means color-texture vector quantization, and K-Medoids centroid clustering.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


class ImageSegmenter:
    """Partitions touching industrial components and clusters surface appearance regions."""

    @staticmethod
    def watershed_segmentation(
        image: np.ndarray,
        distance_threshold_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        """Separates touching parts using distance-transform-seeded marker watershed."""
        bgr = image if (image.ndim == 3 and image.shape[2] == 3) else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Otsu thresholding
        _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological opening to suppress background artifacts
        morph_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleared = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, morph_k, iterations=2)

        # Sure background region
        background_sure = cv2.dilate(cleared, morph_k, iterations=3)

        # Distance transform to find sure foreground object centers
        distance_field = cv2.distanceTransform(cleared, cv2.DIST_L2, 5)
        peak_dist = float(distance_field.max())

        if peak_dist > 0.0:
            _, foreground_sure = cv2.threshold(
                distance_field,
                distance_threshold_ratio * peak_dist,
                255,
                cv2.THRESH_BINARY,
            )
        else:
            foreground_sure = np.zeros_like(cleared)

        foreground_uint8 = np.uint8(foreground_sure)
        unknown_band = cv2.subtract(background_sure, foreground_uint8)

        # Marker labeling
        _, markers = cv2.connectedComponents(foreground_uint8)
        markers = markers + 1
        markers[unknown_band == 255] = 0

        # Execute watershed
        markers = cv2.watershed(bgr, markers)

        boundaries = np.uint8(markers == -1) * 255
        distinct_count = int(np.max(markers) - 1) if np.max(markers) > 1 else 0

        overlay = bgr.copy()
        overlay[markers == -1] = [0, 0, 255]

        return {
            "markers": markers,
            "boundary_mask": boundaries,
            "num_segments": max(0, distinct_count),
            "segmented_overlay": overlay,
            "distance_map": distance_field,
        }

    @staticmethod
    def kmeans_clustering(
        image: np.ndarray,
        k: int = 3,
        use_spatial: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Clusters image pixels in perceptual Lab space into k dominant components."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        rows, cols = lab.shape[:2]
        descriptors = lab.reshape((-1, 3)).astype(np.float32)

        if use_spatial:
            y_mesh, x_mesh = np.mgrid[0:rows, 0:cols]
            coords = np.column_stack((
                (x_mesh.ravel() / float(cols)) * 255.0,
                (y_mesh.ravel() / float(rows)) * 255.0,
            )).astype(np.float32)
            descriptors = np.hstack((descriptors, coords))

        term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, label_vec, centroids = cv2.kmeans(
            descriptors,
            k,
            None,
            term_crit,
            10,
            cv2.KMEANS_RANDOM_CENTERS,
        )

        cluster_colors = np.uint8(centroids[:, :3])
        quantized_lab = cluster_colors[label_vec.flatten()].reshape((rows, cols, 3))
        quantized_bgr = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2BGR)
        label_matrix = label_vec.reshape((rows, cols))

        return quantized_bgr, label_matrix

    @staticmethod
    def kmedoids_clustering(
        data_points: np.ndarray,
        k: int = 3,
        max_iters: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs K-Medoids clustering, selecting representative sample points as cluster centers."""
        data = np.asarray(data_points, dtype=np.float64)
        n_samples = data.shape[0]

        if n_samples <= k:
            return np.arange(n_samples), np.arange(n_samples)

        # Deterministic medoid seed initialization
        # Pick points furthest apart
        rng = np.random.RandomState(42)
        medoid_indices = rng.choice(n_samples, size=k, replace=False)

        # Precompute pairwise distance matrix: shape (N, N)
        diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
        pairwise_dist = np.sqrt(np.sum(diff ** 2, axis=2))

        assignments = np.zeros(n_samples, dtype=int)

        for _ in range(max_iters):
            # Assign each point to closest medoid
            dist_to_medoids = pairwise_dist[:, medoid_indices]
            new_assignments = np.argmin(dist_to_medoids, axis=1)

            # Update medoids
            new_medoids = medoid_indices.copy()
            for cluster_id in range(k):
                members = np.where(new_assignments == cluster_id)[0]
                if len(members) == 0:
                    continue
                # Medoid is the member with minimum sum of distances to other members
                cluster_sub_matrix = pairwise_dist[np.ix_(members, members)]
                intra_dists = np.sum(cluster_sub_matrix, axis=1)
                best_member = members[np.argmin(intra_dists)]
                new_medoids[cluster_id] = best_member

            if np.array_equal(medoid_indices, new_medoids):
                break
            medoid_indices = new_medoids

        final_dist_to_medoids = pairwise_dist[:, medoid_indices]
        assignments = np.argmin(final_dist_to_medoids, axis=1)

        return medoid_indices, assignments
