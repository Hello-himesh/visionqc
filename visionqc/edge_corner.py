"""Edge, Corner, and Hough Transform Module (Module 3).

Implements Sobel and Canny edge detection, Harris and Shi-Tomasi corner detection,
and Hough line and circle transforms for industrial component inspection.
"""

from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np


class FeatureDetector:
    """Detects geometric features including edges, corners, lines, and circular bores."""

    @staticmethod
    def sobel_edges(image: np.ndarray, ksize: int = 3) -> Dict[str, np.ndarray]:
        """Computes Sobel gradient magnitude, x-gradient, y-gradient, and orientation.

        Args:
            image: Grayscale or BGR image.
            ksize: Sobel kernel size (1, 3, 5, or 7).

        Returns:
            Dictionary containing 'gx', 'gy', 'magnitude', and 'orientation'.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = np.clip((magnitude / (np.max(magnitude) + 1e-6)) * 255.0, 0, 255).astype(np.uint8)

        orientation = np.arctan2(gy, gx) * (180.0 / np.pi)  # in degrees [-180, 180]

        return {
            "gx": gx,
            "gy": gy,
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
        """Applies Canny edge detection with automatic median-based thresholding.

        If thresholds are not provided, calculates dynamic thresholds using:
        lower = max(0, (1.0 - sigma) * median)
        upper = min(255, (1.0 + sigma) * median)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if lower_threshold is None or upper_threshold is None:
            v = np.median(gray)
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
        else:
            lower = int(lower_threshold)
            upper = int(upper_threshold)

        return cv2.Canny(gray, lower, upper)

    @staticmethod
    def harris_corners(
        image: np.ndarray,
        block_size: int = 2,
        ksize: int = 3,
        k: float = 0.04,
        threshold_ratio: float = 0.01,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Detects corners using the Harris Corner Detector.

        Response: R = det(M) - k * (trace(M))^2.

        Args:
            image: Grayscale or BGR image.
            block_size: Neighborhood size for structure tensor.
            ksize: Aperture parameter for Sobel operator.
            k: Harris detector free parameter in [0.04, 0.06].
            threshold_ratio: Minimum corner response relative to maximum.

        Returns:
            Tuple of (dilated corner response map, list of (x, y) corner coordinates).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray_f = np.float32(gray)
        dst = cv2.cornerHarris(gray_f, block_size, ksize, k)

        # Dilate corner response to mark corners visibly
        dst_dilated = cv2.dilate(dst, None)
        threshold = threshold_ratio * dst.max()

        corners_mask = dst > threshold
        y_coords, x_coords = np.where(corners_mask)
        corner_points = list(zip(x_coords.tolist(), y_coords.tolist()))

        return dst_dilated, corner_points

    @staticmethod
    def shi_tomasi_corners(
        image: np.ndarray,
        max_corners: int = 50,
        quality_level: float = 0.01,
        min_distance: float = 10.0,
    ) -> List[Tuple[int, int]]:
        """Detects corners using the Shi-Tomasi (Good Features to Track) criterion."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
        )
        if corners is None:
            return []

        pts = []
        for c in corners:
            x, y = c.ravel()
            pts.append((int(x), int(y)))
        return pts

    @staticmethod
    def detect_hough_lines(
        image: np.ndarray,
        threshold: int = 50,
        min_line_length: int = 30,
        max_line_gap: int = 10,
    ) -> List[Tuple[int, int, int, int]]:
        """Detects straight lines (cracks, straight part boundaries) using Probabilistic Hough Transform.

        Returns:
            List of lines defined as (x1, y1, x2, y2).
        """
        if len(image.shape) == 3:
            edges = FeatureDetector.canny_edges(image)
        elif np.max(image) <= 1:
            edges = (image * 255).astype(np.uint8)
        else:
            edges = image.copy()

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        if lines is None:
            return []

        return [tuple(line[0]) for line in lines]

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
        """Detects circular holes, washers, and central bores using Hough Circle Transform.

        Returns:
            List of detected circle dicts: {'center': (x, y), 'radius': r}.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        detected = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for x, y, r in circles:
                detected.append({
                    "center": (int(x), int(y)),
                    "radius": int(r),
                })
        return detected
