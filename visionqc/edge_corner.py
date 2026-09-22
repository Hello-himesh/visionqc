"""Geometric Edge, Corner, and Parametric Transform Module (Module 3).

Implements spatial gradient computation via Sobel operators, Canny hysteresis edge
tracing, Harris structure tensor and Shi-Tomasi corner extraction, and Hough
transforms for mechanical lines and circular bore inspection.
"""

from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np


class FeatureDetector:
    """Detects geometric primitives, structural boundaries, and mechanical alignments."""

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Helper to ensure single-channel grayscale 8-bit input."""
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    @staticmethod
    def sobel_edges(image: np.ndarray, ksize: int = 3) -> Dict[str, np.ndarray]:
        """Calculates directional Sobel derivatives, gradient magnitude, and orientation."""
        gray = FeatureDetector._to_gray(image)

        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

        mag = np.hypot(dx, dy)
        peak_mag = np.max(mag)
        scaled_mag = (mag / (peak_mag + 1e-7)) * 255.0
        magnitude = np.clip(scaled_mag, 0, 255).astype(np.uint8)

        # Orientation in degrees [-180, 180]
        orientation = np.rad2deg(np.arctan2(dy, dx))

        return {
            "gx": dx,
            "gy": dy,
            "magnitude": magnitude,
            "orientation": orientation,
        }

    @staticmethod
    def canny_edges(
        image: np.ndarray,
        lower_threshold: Optional[float] = None,
        upper_threshold: Optional[float] = None,
        sigma: float = 0.33,
    ) -> np.ndarray:
        """Extracts thin edge contours using Canny with automatic Otsu/median thresholds."""
        gray = FeatureDetector._to_gray(image)

        if lower_threshold is None or upper_threshold is None:
            med = float(np.median(gray))
            low_val = int(max(0, (1.0 - sigma) * med))
            high_val = int(min(255, (1.0 + sigma) * med))
        else:
            low_val = int(lower_threshold)
            high_val = int(upper_threshold)

        return cv2.Canny(gray, low_val, high_val)

    @staticmethod
    def harris_corners(
        image: np.ndarray,
        block_size: int = 2,
        ksize: int = 3,
        k: float = 0.04,
        threshold_ratio: float = 0.01,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Identifies corner keypoints using the Harris structure tensor: R = det(M) - k*(tr(M))^2."""
        gray = FeatureDetector._to_gray(image)
        gray_32f = np.float32(gray)

        harris_response = cv2.cornerHarris(gray_32f, block_size, ksize, k)
        dilated_response = cv2.dilate(harris_response, None)

        threshold = threshold_ratio * float(harris_response.max())
        y_locs, x_locs = np.where(harris_response > threshold)
        corners = [(int(x), int(y)) for x, y in zip(x_locs, y_locs)]

        return dilated_response, corners

    @staticmethod
    def shi_tomasi_corners(
        image: np.ndarray,
        max_corners: int = 50,
        quality_level: float = 0.01,
        min_distance: float = 10.0,
    ) -> List[Tuple[int, int]]:
        """Finds prominent corners via Shi-Tomasi minimum eigenvalue criterion."""
        gray = FeatureDetector._to_gray(image)
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
        )
        if pts is None:
            return []

        return [(int(p[0][0]), int(p[0][1])) for p in pts]

    @staticmethod
    def detect_hough_lines(
        image: np.ndarray,
        threshold: int = 50,
        min_line_length: int = 30,
        max_line_gap: int = 10,
    ) -> List[Tuple[int, int, int, int]]:
        """Detects linear segments and crack pathways using Probabilistic Hough Transform."""
        if image.ndim == 3:
            edges = FeatureDetector.canny_edges(image)
        elif float(np.max(image)) <= 1.0:
            edges = (image * 255).astype(np.uint8)
        else:
            edges = image.copy()

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        if lines is None:
            return []

        return [tuple(int(val) for val in segment[0]) for segment in lines]

    @staticmethod
    def detect_hough_circles(
        image: np.ndarray,
        dp: float = 1.2,
        min_dist: float = 30.0,
        param1: float = 50.0,
        param2: float = 30.0,
        min_radius: int = 10,
        max_radius: int = 150,
    ) -> List[Dict[str, Any]]:
        """Identifies circular bores and holes using Hough Circle Transform."""
        gray = FeatureDetector._to_gray(image)
        smoothed = cv2.GaussianBlur(gray, (9, 9), sigmaX=2.0, sigmaY=2.0)

        candidates = cv2.HoughCircles(
            smoothed,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        detected_circles = []
        if candidates is not None:
            rounded = np.rint(candidates[0, :]).astype(int)
            for cx, cy, rad in rounded:
                detected_circles.append({
                    "center": (int(cx), int(cy)),
                    "radius": int(rad),
                })

        return detected_circles
