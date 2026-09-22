"""Mathematical Morphology Subsystem (Module 2).

Provides algebraic set operations, structuring element generation, non-linear
topological filters (opening, closing, gradient, top-hat, black-hat), and
defect blob spatial filtering.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


class MorphologicalAnalyzer:
    """Applies mathematical morphology for defect isolation and region profiling."""

    _SHAPE_LOOKUP = {
        "rect": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
        "ellipse": cv2.MORPH_ELLIPSE,
    }

    @staticmethod
    def get_kernel(shape: str = "rect", ksize: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """Constructs a binary structuring element footprint.

        Args:
            shape: 'rect', 'cross', or 'ellipse'.
            ksize: Dimensions (width, height) of the kernel.

        Returns:
            Kernel mask matrix.
        """
        key = shape.strip().lower()
        if key not in MorphologicalAnalyzer._SHAPE_LOOKUP:
            raise ValueError(f"Unknown shape '{shape}'. Choose 'rect', 'cross', or 'ellipse'.")
        return cv2.getStructuringElement(MorphologicalAnalyzer._SHAPE_LOOKUP[key], ksize)

    @staticmethod
    def erode(image: np.ndarray, kernel: Optional[np.ndarray] = None, iterations: int = 1) -> np.ndarray:
        """Erosion operator: reduces foreground boundaries and suppresses tiny background noise."""
        elem = kernel if kernel is not None else MorphologicalAnalyzer.get_kernel("rect", (3, 3))
        return cv2.erode(image, elem, iterations=iterations)

    @staticmethod
    def dilate(image: np.ndarray, kernel: Optional[np.ndarray] = None, iterations: int = 1) -> np.ndarray:
        """Dilation operator: expands foreground regions and bridges small discontinuities."""
        elem = kernel if kernel is not None else MorphologicalAnalyzer.get_kernel("rect", (3, 3))
        return cv2.dilate(image, elem, iterations=iterations)

    @staticmethod
    def open(image: np.ndarray, kernel: Optional[np.ndarray] = None, iterations: int = 1) -> np.ndarray:
        """Morphological Opening (Erosion followed by Dilation): eliminates isolated speckle noise."""
        elem = kernel if kernel is not None else MorphologicalAnalyzer.get_kernel("rect", (5, 5))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, elem, iterations=iterations)

    @staticmethod
    def close(image: np.ndarray, kernel: Optional[np.ndarray] = None, iterations: int = 1) -> np.ndarray:
        """Morphological Closing (Dilation followed by Erosion): fills internal voids and narrow fractures."""
        elem = kernel if kernel is not None else MorphologicalAnalyzer.get_kernel("rect", (5, 5))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, elem, iterations=iterations)

    @staticmethod
    def gradient(image: np.ndarray, kernel: Optional[np.ndarray] = None) -> np.ndarray:
        """Morphological Gradient (Dilation - Erosion): highlights structural boundary perimeters."""
        elem = kernel if kernel is not None else MorphologicalAnalyzer.get_kernel("rect", (3, 3))
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, elem)

    @staticmethod
    def top_hat(image: np.ndarray, kernel: Optional[np.ndarray] = None) -> np.ndarray:
        """White Top-Hat (Image - Opening): isolates localized elements brighter than their surroundings."""
        elem = kernel if kernel is not None else MorphologicalAnalyzer.get_kernel("rect", (9, 9))
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, elem)

    @staticmethod
    def black_hat(image: np.ndarray, kernel: Optional[np.ndarray] = None) -> np.ndarray:
        """Black-Hat (Closing - Image): extracts dark fissures, hairline cracks, and surface pores."""
        elem = kernel if kernel is not None else MorphologicalAnalyzer.get_kernel("rect", (9, 9))
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, elem)

    @staticmethod
    def clean_defect_mask(
        binary_mask: np.ndarray,
        min_area: int = 20,
        max_area: Optional[int] = None,
        fill_holes: bool = True,
    ) -> np.ndarray:
        """Filters spurious noise blobs, preserving genuine industrial defect regions."""
        gray = (
            cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
            if binary_mask.ndim == 3
            else binary_mask
        )
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered = np.zeros_like(binary)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
            cv2.drawContours(filtered, [contour], -1, 255, -1)

        if fill_holes:
            closing_elem = MorphologicalAnalyzer.get_kernel("ellipse", (3, 3))
            filtered = MorphologicalAnalyzer.close(filtered, closing_elem)

        return filtered
