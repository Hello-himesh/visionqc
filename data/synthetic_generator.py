"""Synthetic Industrial Component and Defect Generator.

Generates realistic mechanical components (gears, bearings, brackets) with
controlled defect injections for testing and benchmark verification.
"""

from typing import Tuple, Optional
import os
import random
import cv2
import numpy as np


def generate_base_gear(size: int = 256, num_teeth: int = 12, inner_radius: int = 28, outer_radius: int = 75) -> np.ndarray:
    """Generates a clean mechanical spur gear on a dark background."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, size // 2

    # Conveyor belt subtle background texture
    noise = np.random.normal(30, 5, (size, size, 3)).astype(np.uint8)
    img = cv2.add(img, noise)

    # Base gear body
    cv2.circle(img, (cx, cy), outer_radius, (170, 175, 180), -1)

    # Gear teeth
    for i in range(num_teeth):
        angle = i * (2.0 * np.pi / num_teeth)
        tooth_len = 16
        tx = int(cx + (outer_radius + tooth_len / 2.0) * np.cos(angle))
        ty = int(cy + (outer_radius + tooth_len / 2.0) * np.sin(angle))
        # Draw tooth rect
        rect = ((tx, ty), (14, 18), np.degrees(angle))
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(img, [box], 0, (175, 180, 185), -1)

    # Center circular bore
    cv2.circle(img, (cx, cy), inner_radius, (30, 32, 35), -1)
    # Keyway notch in bore
    cv2.rectangle(img, (cx - 4, cy - inner_radius - 6), (cx + 4, cy - inner_radius), (30, 32, 35), -1)

    # Add metallic brushed texture
    brushed = np.random.normal(0, 8, (size, size, 3)).astype(np.float32)
    img_f = img.astype(np.float32) + brushed
    return np.clip(img_f, 0, 255).astype(np.uint8)


def inject_crack(image: np.ndarray, length: int = 40, thickness: int = 2) -> np.ndarray:
    """Injects a realistic wandering fracture/crack onto the component surface."""
    result = image.copy()
    h, w = result.shape[:2]
    cx, cy = w // 2, h // 2

    # Start near middle radius
    angle = random.uniform(0, 2 * np.pi)
    r = random.uniform(35, 60)
    curr_x = int(cx + r * np.cos(angle))
    curr_y = int(cy + r * np.sin(angle))

    for _ in range(length // 4):
        step_x = random.randint(-5, 5)
        step_y = random.randint(-5, 5)
        next_x = int(np.clip(curr_x + step_x, 10, w - 10))
        next_y = int(np.clip(curr_y + step_y, 10, h - 10))
        # Dark fracture line
        cv2.line(result, (curr_x, curr_y), (next_x, next_y), (25, 25, 30), thickness)
        curr_x, curr_y = next_x, next_y

    return result


def inject_surface_defect(image: np.ndarray, num_spots: int = 25) -> np.ndarray:
    """Injects oxidation, corrosion pitting, or surface scratches."""
    result = image.copy()
    h, w = result.shape[:2]
    cx, cy = w // 2, h // 2

    for _ in range(num_spots):
        angle = random.uniform(0, 2 * np.pi)
        r = random.uniform(32, 70)
        px = int(cx + r * np.cos(angle))
        py = int(cy + r * np.sin(angle))
        radius = random.randint(2, 5)
        # Brownish-orange rust/corrosion color or dark pit
        color = (random.randint(15, 35), random.randint(45, 90), random.randint(120, 190))
        cv2.circle(result, (px, py), radius, color, -1)

    return result


def inject_bore_defect(image: np.ndarray) -> np.ndarray:
    """Injects an eccentricity or missing bore defect."""
    result = image.copy()
    cx, cy = result.shape[1] // 2, result.shape[0] // 2
    # Fill in the central bore or make it severely elliptical / off-center
    cv2.circle(result, (cx, cy), 32, (170, 175, 180), -1)
    # Off-center deformed bore
    cv2.ellipse(result, (cx + 20, cy - 15), (20, 10), 45, 0, 360, (30, 32, 35), -1)
    return result


def generate_benchmark_dataset(output_dir: str = "data/samples", count_per_class: int = 8) -> None:
    """Generates train and test benchmark partitions."""
    classes = ["PASS", "DEFECT_CRACK", "DEFECT_SURFACE", "DEFECT_BORE"]

    for split in ["train", "test"]:
        split_count = count_per_class if split == "train" else max(3, count_per_class // 2)
        for cls in classes:
            dir_path = os.path.join(output_dir, split, cls)
            os.makedirs(dir_path, exist_ok=True)

            for i in range(split_count):
                img = generate_base_gear()
                if cls == "DEFECT_CRACK":
                    img = inject_crack(img)
                elif cls == "DEFECT_SURFACE":
                    img = inject_surface_defect(img)
                elif cls == "DEFECT_BORE":
                    img = inject_bore_defect(img)

                filename = f"{cls.lower()}_{i+1:03d}.png"
                cv2.imwrite(os.path.join(dir_path, filename), img)


if __name__ == "__main__":
    generate_benchmark_dataset()
    print("Benchmark dataset generated successfully in data/samples/")
