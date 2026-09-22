"""End-to-End Quality Inspection Pipeline Orchestrator.

Integrates preprocessing, morphology, geometric feature extraction, segmentation,
and classification into a unified inspection pipeline for manufacturing quality control.
"""

from typing import Dict, Any, Optional
import time
import os
import cv2
import numpy as np

from visionqc.preprocessor import ImagePreprocessor
from visionqc.morphology import MorphologicalAnalyzer
from visionqc.edge_corner import FeatureDetector
from visionqc.segmentation import ImageSegmenter
from visionqc.classifier import DefectClassifier


class InspectionPipeline:
    """Industrial vision inspection orchestrator."""

    def __init__(self, classifier: Optional[DefectClassifier] = None):
        self.preprocessor = ImagePreprocessor()
        self.morphology = MorphologicalAnalyzer()
        self.feature_detector = FeatureDetector()
        self.segmenter = ImageSegmenter()
        self.classifier = classifier

    def inspect(
        self,
        image_input: Any,
        save_dir: Optional[str] = None,
        prefix: str = "inspect",
    ) -> Dict[str, Any]:
        """Runs the complete inspection sequence on an input image."""
        t_zero = time.perf_counter()

        # 1. Ingest image input
        if isinstance(image_input, str):
            source_image = self.preprocessor.load_image(image_input)
            filename = os.path.basename(image_input)
        else:
            source_image = np.array(image_input, copy=True)
            filename = "memory_buffer.png"

        img_h, img_w = source_image.shape[:2]
        cx, cy = img_w // 2, img_h // 2

        # 2. Denoising & Enhancement
        smoothed = self.preprocessor.remove_noise(source_image, method="gaussian", kernel_size=5)
        gray_image = self.preprocessor.convert_color(smoothed, "GRAY")
        clahe_enhanced = self.preprocessor.equalize_histogram(smoothed, use_clahe=True, clip_limit=2.0)

        # 3. Morphological Defect Mask Extraction
        # Build donut inspection ROI for gear body (excluding teeth periphery and inner bore/keyway)
        gear_body_roi = np.zeros_like(gray_image)
        cv2.circle(gear_body_roi, (cx, cy), 72, 255, -1)
        cv2.circle(gear_body_roi, (cx, cy), 31, 0, -1)

        # Black-Hat isolates dark cracks and fissures
        bh_elem = self.morphology.get_kernel("rect", (9, 9))
        blackhat = self.morphology.black_hat(gray_image, bh_elem)
        blackhat_masked = cv2.bitwise_and(blackhat, gear_body_roi)

        _, raw_mask = cv2.threshold(blackhat_masked, 30, 255, cv2.THRESH_BINARY)
        defect_mask = self.morphology.clean_defect_mask(raw_mask, min_area=15)
        defect_pixels = int(np.count_nonzero(defect_mask))

        # 4. Geometric & Structural Features
        edges = self.feature_detector.canny_edges(clahe_enhanced)
        _, corners = self.feature_detector.harris_corners(gray_image, threshold_ratio=0.02)
        hough_lines = self.feature_detector.detect_hough_lines(edges, threshold=40, min_line_length=25)
        hough_circles = self.feature_detector.detect_hough_circles(
            gray_image,
            min_radius=15,
            max_radius=min(img_h, img_w) // 2,
        )

        # Check for concentric central bore circle (radius ~ 25..36 centered near cx, cy)
        has_centered_bore = any(
            (abs(c["center"][0] - cx) <= 6)
            and (abs(c["center"][1] - cy) <= 6)
            and (22 <= c["radius"] <= 36)
            for c in hough_circles
        )

        # Color variance in Lab space within gear body ROI
        lab_img = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB)
        _, _, b_chan = cv2.split(lab_img)
        b_body = b_chan[gear_body_roi == 255]
        b_std = float(np.std(b_body)) if len(b_body) > 0 else 0.0

        # 5. Segmentation & Color Grouping
        watershed_data = self.segmenter.watershed_segmentation(source_image)
        clustered_view, _ = self.segmenter.kmeans_clustering(source_image, k=3)

        # 6. Classification Logic
        if self.classifier is not None and self.classifier.is_fitted:
            features = self.classifier.extract_image_features(source_image)
            predicted_class, class_conf = self.classifier.predict(features)
            status = "PASS" if predicted_class == "PASS" else "FAIL"
        else:
            # Deterministic domain heuristics fallback
            if defect_pixels >= 80:
                status = "FAIL"
                predicted_class = "DEFECT_CRACK"
                class_conf = min(0.99, 0.75 + (defect_pixels / 800.0))
            elif not has_centered_bore:
                status = "FAIL"
                predicted_class = "DEFECT_BORE"
                class_conf = 0.92
            elif b_std > 2.5:
                status = "FAIL"
                predicted_class = "DEFECT_SURFACE"
                class_conf = min(0.98, 0.70 + (b_std / 20.0))
            else:
                status = "PASS"
                predicted_class = "PASS"
                class_conf = 0.96

        # 7. Diagnostic Annotations
        annotated = source_image.copy()

        # Green circles for circular bore detection
        for c in hough_circles:
            pt = c["center"]
            rad = c["radius"]
            cv2.circle(annotated, pt, rad, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.circle(annotated, pt, 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        # Yellow markers for Harris corners
        for pt in corners[:80]:
            cv2.circle(annotated, pt, 2, (0, 255, 255), -1, lineType=cv2.LINE_AA)

        # Cyan segments for Hough lines
        for x1, y1, x2, y2 in hough_lines[:20]:
            cv2.line(annotated, (x1, y1), (x2, y2), (255, 255, 0), 1, lineType=cv2.LINE_AA)

        # Red outlines for detected defect contours
        defect_cnts, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, defect_cnts, -1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Header status banner
        banner_bgr = (0, 180, 0) if status == "PASS" else (0, 0, 220)
        cv2.rectangle(annotated, (0, 0), (img_w, 34), banner_bgr, -1)
        header_text = f"Status: {status} | Class: {predicted_class} | Conf: {class_conf * 100:.1f}%"
        cv2.putText(
            annotated,
            header_text,
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        duration_ms = (time.perf_counter() - t_zero) * 1000.0

        # 8. Persist Artifacts if requested
        saved_paths = {}
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            base_stem = os.path.splitext(filename)[0]
            compress_flag = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]

            annotated_file = os.path.join(save_dir, f"{prefix}_{base_stem}_annotated.png")
            cv2.imwrite(annotated_file, annotated, compress_flag)
            saved_paths["annotated"] = annotated_file

            edges_file = os.path.join(save_dir, f"{prefix}_{base_stem}_edges.png")
            cv2.imwrite(edges_file, edges, compress_flag)
            saved_paths["edges"] = edges_file

            defect_file = os.path.join(save_dir, f"{prefix}_{base_stem}_defect_mask.png")
            cv2.imwrite(defect_file, defect_mask, compress_flag)
            saved_paths["defect_mask"] = defect_file

            watershed_file = os.path.join(save_dir, f"{prefix}_{base_stem}_watershed.png")
            cv2.imwrite(watershed_file, watershed_data["segmented_overlay"], compress_flag)
            saved_paths["watershed"] = watershed_file

            kmeans_file = os.path.join(save_dir, f"{prefix}_{base_stem}_kmeans.png")
            cv2.imwrite(kmeans_file, clustered_view, compress_flag)
            saved_paths["kmeans"] = kmeans_file

        return {
            "source": filename,
            "status": status,
            "defect_type": predicted_class,
            "confidence": round(float(class_conf), 4),
            "execution_time_ms": round(duration_ms, 2),
            "metrics": {
                "image_resolution": [img_w, img_h],
                "defect_pixel_count": defect_pixels,
                "num_corners_detected": len(corners),
                "num_hough_lines": len(hough_lines),
                "num_hough_circles": len(hough_circles),
                "num_watershed_segments": watershed_data["num_segments"],
            },
            "artifacts": saved_paths,
        }
