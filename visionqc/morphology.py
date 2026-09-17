"""Morphology Module (Module 2).

Implements mathematical morphology operations: erosion, dilation, opening,
closing, gradient, top-hat, black-hat transforms, and defect mask refinement.
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np


class MorphologicalAnalyzer:
    """Performs mathematical morphology for defect detection and mask cleaning."""

    @staticmethod
    def get_kernel(shape: str = "rect", ksize: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """Creates a structuring element.

        Args:
            shape: 'rect', 'cross', or 'ellipse'.
            ksize: Kernel dimensions (width, height).

        Returns:
            Structuring element array.
        """
        shape_map = {
            "rect": cv2.MORPH_RECT,
            "cross": cv2.MORPH_CROSS,
            "ellipse": cv2.MORPH_ELLIPSE,
        }
        shape_code = shape_map.get(shape.lower())
        if shape_code is None:
            raise ValueError(f"Unknown shape '{shape}'. Choose 'rect', 'cross', or 'ellipse'.")
        return cv2.getStructuringElement(shape_code, ksize)

    @staticmethod
    def erode(image: np.ndarray, kernel: Optional[np.ndarray] = None, iterations: int = 1) -> np.ndarray:
        """Erosion: shrinks foreground and removes tiny background noise."""
        if kernel is None:
            kernel = MorphologicalAnalyzer.get_kernel("rect", (3, 3))
        return cv2.erode(image, kernel, iterations=iterations)

    @staticmethod
    def dilate(image: np.ndarray, kernel: Optional[np.ndarray] = None, iterations: int = 1) -> np.ndarray:
        """Dilation: expands foreground and fills small holes."""
        if kernel is None:
            kernel = MorphologicalAnalyzer.get_kernel("rect", (3, 3))
        return cv2.dilate(image, kernel, iterations=iterations)

    @staticmethod
    def open(image: np.ndarray, kernel: Optional[np.ndarray] = None, iterations: int = 1) -> np.ndarray:
        """Opening (Erosion followed by Dilation): removes small noise specs."""
        if kernel is None:
            kernel = MorphologicalAnalyzer.get_kernel("rect", (5, 5))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)

    @staticmethod
    def close(image: np.ndarray, kernel: Optional[np.ndarray] = None, iterations: int = 1) -> np.ndarray:
        """Closing (Dilation followed by Erosion): bridges narrow breaks and fills holes."""
        if kernel is None:
            kernel = MorphologicalAnalyzer.get_kernel("rect", (5, 5))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    @staticmethod
    def gradient(image: np.ndarray, kernel: Optional[np.ndarray] = None) -> np.ndarray:
        """Morphological Gradient (Dilation - Erosion): yields object structural outlines."""
        if kernel is None:
            kernel = MorphologicalAnalyzer.get_kernel("rect", (3, 3))
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

    @staticmethod
    def top_hat(image: np.ndarray, kernel: Optional[np.ndarray] = None) -> np.ndarray:
        """Top-Hat (Image - Opening): isolates elements brighter than surroundings."""
        if kernel is None:
            kernel = MorphologicalAnalyzer.get_kernel("rect", (9, 9))
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    @staticmethod
    def black_hat(image: np.ndarray, kernel: Optional[np.ndarray] = None) -> np.ndarray:
        """Black-Hat (Closing - Image): isolates dark defects (cracks, fissures, voids)."""
        if kernel is None:
            kernel = MorphologicalAnalyzer.get_kernel("rect", (9, 9))
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    @staticmethod
    def clean_defect_mask(
        binary_mask: np.ndarray,
        min_area: int = 20,
        max_area: Optional[int] = None,
        fill_holes: bool = True,
    ) -> np.ndarray:
        """Filters out spurious noise particles and retains genuine defect regions.

        Args:
            binary_mask: 8-bit single channel binary mask (0 or 255).
            min_area: Minimum connected contour area to keep.
            max_area: Optional maximum area limit.
            fill_holes: Whether to close and fill inner holes in defect regions.

        Returns:
            Cleaned binary mask.
        """
        mask = binary_mask.copy()
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cleaned = np.zeros_like(thresh)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
            cv2.drawContours(cleaned, [cnt], -1, 255, -1)

        if fill_holes:
            kernel = MorphologicalAnalyzer.get_kernel("ellipse", (3, 3))
            cleaned = MorphologicalAnalyzer.close(cleaned, kernel)

        return cleaned
