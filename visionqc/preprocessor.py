"""Industrial Image Preprocessing and Restoration Subsystem (Modules 1 & 2).

Provides image ingestion, multi-space color transformations, spatial denoising,
dynamic radiometric intensity scaling, and adaptive histogram equalization.
"""

from typing import Tuple, Optional
import os
import cv2
import numpy as np


class ImagePreprocessor:
    """Acquires, normalizes, conditions, and enhances industrial component imagery."""

    @staticmethod
    def load_image(image_path: str, as_grayscale: bool = False) -> np.ndarray:
        """Loads an image from the filesystem with path and format validation.

        Args:
            image_path: Target image file path.
            as_grayscale: When True, loads image in 8-bit single-channel mode.

        Returns:
            Decoded image as a NumPy array.

        Raises:
            FileNotFoundError: If the file path does not point to an existing file.
            ValueError: If the file payload cannot be parsed by OpenCV.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        read_mode = cv2.IMREAD_GRAYSCALE if as_grayscale else cv2.IMREAD_COLOR
        matrix = cv2.imread(image_path, read_mode)
        if matrix is None:
            raise ValueError(f"Failed to decode image from path: {image_path}")

        return matrix

    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> str:
        """Writes an image to disk, creating enclosing directories if absent."""
        folder = os.path.dirname(output_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        if not cv2.imwrite(output_path, image):
            raise IOError(f"Failed to write image to {output_path}")
        return output_path

    @staticmethod
    def convert_color(image: np.ndarray, target_space: str = "GRAY") -> np.ndarray:
        """Transfers an image between color representations (GRAY, RGB, HSV, LAB)."""
        mode = target_space.strip().upper()
        is_single_channel = (image.ndim == 2) or (image.ndim == 3 and image.shape[2] == 1)

        if mode == "GRAY":
            if is_single_channel:
                return image.copy()
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # For 3-channel representations from grayscale
        src = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if is_single_channel else image

        conversion_table = {
            "RGB": cv2.COLOR_BGR2RGB,
            "HSV": cv2.COLOR_BGR2HSV,
            "LAB": cv2.COLOR_BGR2LAB,
        }
        if mode not in conversion_table:
            raise ValueError(f"Unsupported color space: {target_space}. Use GRAY, RGB, HSV, or LAB.")

        return cv2.cvtColor(src, conversion_table[mode])

    @staticmethod
    def resize_with_aspect_ratio(
        image: np.ndarray,
        width: Optional[int] = None,
        height: Optional[int] = None,
        inter: int = cv2.INTER_AREA,
    ) -> np.ndarray:
        """Scales image dimensions proportionally according to target width or height."""
        orig_h, orig_w = image.shape[:2]
        if width is None and height is None:
            return image.copy()

        if width is None and height is not None:
            scale = float(height) / float(orig_h)
            new_size = (int(round(orig_w * scale)), height)
        elif height is None and width is not None:
            scale = float(width) / float(orig_w)
            new_size = (width, int(round(orig_h * scale)))
        else:
            new_size = (int(width), int(height))

        return cv2.resize(image, new_size, interpolation=inter)

    @staticmethod
    def flip(image: np.ndarray, mode: str = "horizontal") -> np.ndarray:
        """Inverts image pixels across horizontal, vertical, or combined axes."""
        direction_codes = {"horizontal": 1, "vertical": 0, "both": -1}
        if mode not in direction_codes:
            raise ValueError(f"Invalid flip mode: {mode}. Choose 'horizontal', 'vertical', or 'both'.")
        return cv2.flip(image, direction_codes[mode])

    @staticmethod
    def remove_noise(
        image: np.ndarray,
        method: str = "gaussian",
        kernel_size: int = 5,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """Attenuates high-frequency sensor noise using Gaussian, Median, or Bilateral filters."""
        k = kernel_size if (kernel_size % 2 == 1) else (kernel_size + 1)
        filt = method.lower()

        if filt == "gaussian":
            return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)
        if filt == "median":
            return cv2.medianBlur(image, k)
        if filt == "bilateral":
            return cv2.bilateralFilter(image, d=k, sigmaColor=75, sigmaSpace=sigma * 50)

        raise ValueError(f"Unknown filter method '{method}'. Choose 'gaussian', 'median', or 'bilateral'.")

    @staticmethod
    def linear_contrast_stretch(image: np.ndarray) -> np.ndarray:
        """Applies min-max radiometric normalization: s = (r - r_min) / (r_max - r_min) * 255."""
        if image.ndim == 3:
            splits = cv2.split(image)
            return cv2.merge([ImagePreprocessor.linear_contrast_stretch(ch) for ch in splits])

        lo, hi = float(np.min(image)), float(np.max(image))
        delta = hi - lo
        if delta <= 1e-5:
            return image.copy()

        stretched = (image.astype(np.float32) - lo) * (255.0 / delta)
        return np.clip(stretched, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def log_transform(image: np.ndarray, c: Optional[float] = None) -> np.ndarray:
        """Applies logarithmic expansion s = c * log(1 + r) to boost low-intensity details."""
        float_buf = image.astype(np.float32)
        if c is None:
            max_intensity = float(np.max(float_buf))
            c = (255.0 / np.log(1.0 + max_intensity)) if max_intensity > 0.0 else 1.0

        out = c * np.log1p(float_buf)
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def power_law_transform(image: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
        """Applies power-law (gamma) illumination adjustment: s = c * (r / 255)^gamma * 255."""
        norm = image.astype(np.float32) / 255.0
        adjusted = c * np.power(norm, gamma) * 255.0
        return np.clip(adjusted, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def equalize_histogram(
        image: np.ndarray,
        use_clahe: bool = True,
        clip_limit: float = 2.0,
        tile_grid: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """Executes global or Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        equalizer = (
            cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
            if use_clahe
            else None
        )

        if image.ndim == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            l_out = equalizer.apply(l_chan) if use_clahe else cv2.equalizeHist(l_chan)
            return cv2.cvtColor(cv2.merge([l_out, a_chan, b_chan]), cv2.COLOR_LAB2BGR)

        return equalizer.apply(image) if use_clahe else cv2.equalizeHist(image)
