"""Synthetic Industrial Component and Defect Generator.

Generates realistic mechanical components (gears, bearings, brackets) with
controlled defect injections for testing and benchmark verification.
Optimized for storage efficiency and high visual inspection fidelity.
"""

from typing import Tuple, Optional
import os
import random
import cv2
import numpy as np


def generate_base_gear(
    size: int = 256,
    num_teeth: int = 12,
    inner_radius: int = 28,
    outer_radius: int = 75,
) -> np.ndarray:
    """Generates a clean mechanical spur gear on an industrial conveyor surface.

    Uses smooth gradients and anti-aliased rendering for high compression efficiency.
    """
    canvas = np.full((size, size, 3), 32, dtype=np.uint8)
    cx, cy = size // 2, size // 2

    # Smooth subtle conveyor lighting vignette
    y_coords, x_coords = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    vignette = np.clip(32 - (dist_from_center / size) * 12, 18, 34).astype(np.uint8)
    for c in range(3):
        canvas[:, :, c] = vignette

    # Gear teeth (drawn with anti-aliasing)
    tooth_len = 16
    tooth_w = 14
    tooth_color = (175, 180, 185)
    for i in range(num_teeth):
        theta = i * (2.0 * np.pi / num_teeth)
        tx = cx + (outer_radius + tooth_len / 2.0) * np.cos(theta)
        ty = cy + (outer_radius + tooth_len / 2.0) * np.sin(theta)
        rect = ((float(tx), float(ty)), (tooth_w, tooth_len + 4), np.degrees(theta))
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(canvas, [box], tooth_color, lineType=cv2.LINE_AA)

    # Base gear outer circular disk
    cv2.circle(canvas, (cx, cy), outer_radius, (170, 175, 180), -1, lineType=cv2.LINE_AA)

    # Center circular bore
    bore_color = (30, 32, 35)
    cv2.circle(canvas, (cx, cy), inner_radius, bore_color, -1, lineType=cv2.LINE_AA)

    # Keyway notch in bore
    notch_half_w = 4
    notch_h = 6
    cv2.rectangle(
        canvas,
        (cx - notch_half_w, cy - inner_radius - notch_h),
        (cx + notch_half_w, cy - inner_radius),
        bore_color,
        -1,
    )

    return canvas


def inject_crack(image: np.ndarray, length: int = 40, thickness: int = 2) -> np.ndarray:
    """Injects a realistic wandering fracture/crack onto the component surface."""
    result = image.copy()
    h, w = result.shape[:2]
    cx, cy = w // 2, h // 2

    # Start crack on the gear body between bore and teeth
    init_angle = random.uniform(0, 2.0 * np.pi)
    r = random.uniform(40, 52)
    curr_x = int(cx + r * np.cos(init_angle))
    curr_y = int(cy + r * np.sin(init_angle))

    # Propagate along tangent or radial direction
    prop_dir = init_angle + random.choice([np.pi / 2, -np.pi / 2])
    step_size = 3
    n_steps = max(8, int(length // step_size))
    crack_color = (20, 20, 25)

    for _ in range(n_steps):
        jitter = random.uniform(-0.35, 0.35)
        curr_dir = prop_dir + jitter
        next_x = int(np.clip(curr_x + step_size * np.cos(curr_dir), 20, w - 20))
        next_y = int(np.clip(curr_y + step_size * np.sin(curr_dir), 20, h - 20))
        cv2.line(result, (curr_x, curr_y), (next_x, next_y), crack_color, thickness)
        curr_x, curr_y = next_x, next_y

    return result


def inject_surface_defect(image: np.ndarray, num_spots: int = 25) -> np.ndarray:
    """Injects oxidation, corrosion pitting, or surface scratches."""
    result = image.copy()
    h, w = result.shape[:2]
    cx, cy = w // 2, h // 2

    for _ in range(num_spots):
        phi = random.uniform(0, 2.0 * np.pi)
        rho = random.uniform(34, 66)
        px = int(cx + rho * np.cos(phi))
        py = int(cy + rho * np.sin(phi))
        rad = random.randint(2, 4)
        # Brownish-orange rust/corrosion tone
        tone = (random.randint(18, 30), random.randint(55, 95), random.randint(140, 185))
        cv2.circle(result, (px, py), rad, tone, -1, lineType=cv2.LINE_AA)

    return result


def inject_bore_defect(image: np.ndarray) -> np.ndarray:
    """Injects an eccentricity or deformed bore defect."""
    result = image.copy()
    cx, cy = result.shape[1] // 2, result.shape[0] // 2

    # Fill original central bore with solid metal
    cv2.circle(result, (cx, cy), 32, (170, 175, 180), -1, lineType=cv2.LINE_AA)
    # Cut out an eccentric, distorted elliptical bore
    cv2.ellipse(
        result,
        (cx + 20, cy - 14),
        (22, 10),
        40,
        0,
        360,
        (30, 32, 35),
        -1,
        lineType=cv2.LINE_AA,
    )
    return result


def generate_benchmark_dataset(
    output_dir: str = "data/samples",
    count_per_class: int = 8,
) -> None:
    """Generates train and test benchmark partitions with maximum PNG compression."""
    classes = ["PASS", "DEFECT_CRACK", "DEFECT_SURFACE", "DEFECT_BORE"]
    encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]

    for split in ["train", "test"]:
        target_count = count_per_class if split == "train" else max(3, count_per_class // 2)
        for cls in classes:
            folder = os.path.join(output_dir, split, cls)
            os.makedirs(folder, exist_ok=True)

            for idx in range(target_count):
                img = generate_base_gear()
                if cls == "DEFECT_CRACK":
                    img = inject_crack(img, length=50, thickness=2)
                elif cls == "DEFECT_SURFACE":
                    img = inject_surface_defect(img, num_spots=28)
                elif cls == "DEFECT_BORE":
                    img = inject_bore_defect(img)

                filepath = os.path.join(folder, f"{cls.lower()}_{idx + 1:03d}.png")
                cv2.imwrite(filepath, img, encode_params)


if __name__ == "__main__":
    generate_benchmark_dataset()
    print("Benchmark dataset generated successfully in data/samples/")
