"""Image Preprocessor Module (Modules 1 & 2).

Provides image loading, color space transformations, noise reduction filters,
linear/logarithmic/power-law transformations, and histogram equalization.
"""

from typing import Tuple, Union, Optional
import os
import cv2
import numpy as np


class ImagePreprocessor:
    """Handles image acquisition, color adjustments, transformations, and enhancement."""

    @staticmethod
    def load_image(image_path: str, as_grayscale: bool = False) -> np.ndarray:
        """Loads an image from disk with validation.

        Args:
            image_path: Path to the image file.
            as_grayscale: Whether to load the image as single-channel grayscale.

        Returns:
            Loaded image as a NumPy ndarray.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be decoded as an image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        flag = cv2.IMREAD_GRAYSCALE if as_grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(image_path, flag)
        if image is None:
            raise ValueError(f"Failed to decode image from path: {image_path}")

        return image

    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> str:
        """Saves an image to disk ensuring parent directories exist.

        Args:
            image: Image array to write.
            output_path: Destination path.

        Returns:
            The normalized output path.
        """
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        success = cv2.imwrite(output_path, image)
        if not success:
            raise IOError(f"Failed to write image to {output_path}")
        return output_path

    @staticmethod
    def convert_color(image: np.ndarray, target_space: str = "GRAY") -> np.ndarray:
        """Converts an image between common color representations.

        Args:
            image: Input image (assumed BGR if 3-channel).
            target_space: One of 'GRAY', 'RGB', 'HSV', 'LAB'.

        Returns:
            Color-converted image.
        """
        target = target_space.upper()
        if target == "GRAY":
            if len(image.shape) == 2:
                return image.copy()
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif target == "RGB":
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif target == "HSV":
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif target == "LAB":
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        else:
            raise ValueError(f"Unsupported color space: {target_space}. Use GRAY, RGB, HSV, or LAB.")

    @staticmethod
    def resize_with_aspect_ratio(
        image: np.ndarray,
        width: Optional[int] = None,
        height: Optional[int] = None,
        inter: int = cv2.INTER_AREA,
    ) -> np.ndarray:
        """Resizes an image while preserving its original aspect ratio."""
        h, w = image.shape[:2]
        if width is None and height is None:
            return image.copy()

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

    @staticmethod
    def flip(image: np.ndarray, mode: str = "horizontal") -> np.ndarray:
        """Flips an image horizontally, vertically, or both."""
        mode_map = {"horizontal": 1, "vertical": 0, "both": -1}
        if mode not in mode_map:
            raise ValueError(f"Invalid flip mode: {mode}. Choose 'horizontal', 'vertical', or 'both'.")
        return cv2.flip(image, mode_map[mode])

    @staticmethod
    def remove_noise(
        image: np.ndarray,
        method: str = "gaussian",
        kernel_size: int = 5,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """Removes noise using Gaussian, Median, or Bilateral filters.

        Args:
            image: Input image.
            method: 'gaussian', 'median', or 'bilateral'.
            kernel_size: Size of the filter kernel (must be odd).
            sigma: Standard deviation for Gaussian / Bilateral filtering.

        Returns:
            Filtered denoised image.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1

        if method == "gaussian":
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        elif method == "median":
            return cv2.medianBlur(image, kernel_size)
        elif method == "bilateral":
            return cv2.bilateralFilter(image, d=kernel_size, sigmaColor=75, sigmaSpace=sigma * 50)
        else:
            raise ValueError(f"Unknown filter method '{method}'. Choose 'gaussian', 'median', or 'bilateral'.")

    @staticmethod
    def linear_contrast_stretch(image: np.ndarray) -> np.ndarray:
        """Applies min-max linear contrast stretching: s = (r - min)/(max - min) * 255."""
        if len(image.shape) == 3:
            channels = cv2.split(image)
            stretched = [ImagePreprocessor.linear_contrast_stretch(c) for c in channels]
            return cv2.merge(stretched)

        min_val = float(np.min(image))
        max_val = float(np.max(image))
        if max_val - min_val == 0:
            return image.copy()

        stretched = (image.astype(np.float32) - min_val) * (255.0 / (max_val - min_val))
        return np.clip(stretched, 0, 255).astype(np.uint8)

    @staticmethod
    def log_transform(image: np.ndarray, c: Optional[float] = None) -> np.ndarray:
        """Applies logarithmic transformation: s = c * log(1 + r).

        Enhances low-intensity details in dark defect regions.
        """
        is_color = len(image.shape) == 3
        work_img = image.astype(np.float32)

        if c is None:
            max_val = np.max(work_img)
            c = 255.0 / np.log(1.0 + max_val) if max_val > 0 else 1.0

        transformed = c * np.log(1.0 + work_img)
        return np.clip(transformed, 0, 255).astype(np.uint8)

    @staticmethod
    def power_law_transform(image: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
        """Applies power-law (Gamma) transformation: s = c * (r / 255)^gamma * 255.

        Gamma < 1 makes dark regions brighter; Gamma > 1 increases contrast in bright regions.
        """
        normalized = image.astype(np.float32) / 255.0
        transformed = c * (normalized ** gamma) * 255.0
        return np.clip(transformed, 0, 255).astype(np.uint8)

    @staticmethod
    def equalize_histogram(image: np.ndarray, use_clahe: bool = True, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Applies standard histogram equalization or CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        if len(image.shape) == 3:
            # Convert to LAB and equalize L-channel to preserve chromaticity
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            if use_clahe:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
                l_eq = clahe.apply(l)
            else:
                l_eq = cv2.equalizeHist(l)
            merged = cv2.merge((l_eq, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        else:
            if use_clahe:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
                return clahe.apply(image)
            return cv2.equalizeHist(image)
