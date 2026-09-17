"""End-to-end Quality Inspection Pipeline.

Orchestrates preprocessing, morphological analysis, edge/corner/Hough detection,
watershed/clustering segmentation, and classification for industrial parts.
"""

from typing import Dict, Any, Optional, List
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
        """Initializes the inspection pipeline.

        Args:
            classifier: Optional pretrained DefectClassifier instance.
        """
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
        """Runs the complete inspection pipeline on an input image.

        Args:
            image_input: Image file path or NumPy ndarray.
            save_dir: Optional directory to save intermediate and annotated outputs.
            prefix: Filename prefix for saved images.

        Returns:
            Inspection report dictionary with defect classification and quantitative metrics.
        """
        start_time = time.time()

        # 1. Load and validate
        if isinstance(image_input, str):
            image = self.preprocessor.load_image(image_input)
            source_name = os.path.basename(image_input)
        else:
            image = image_input.copy()
            source_name = "memory_buffer.png"

        h, w = image.shape[:2]

        # 2. Preprocessing: Enhancement and Denoising
        denoised = self.preprocessor.remove_noise(image, method="gaussian", kernel_size=5)
        gray = self.preprocessor.convert_color(denoised, "GRAY")
        enhanced = self.preprocessor.equalize_histogram(denoised, use_clahe=True, clip_limit=2.0)

        # 3. Morphological Surface Defect Detection
        # Isolate component body to prevent outer tooth/boundary edges from triggering false defects
        _, comp_thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(comp_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            solid_body = np.zeros_like(gray)
            cv2.drawContours(solid_body, [largest_cnt], -1, 255, -1)
            # Erode slightly to avoid tooth edges
            inspect_roi = self.morphology.erode(solid_body, self.morphology.get_kernel("ellipse", (15, 15)), iterations=1)
        else:
            inspect_roi = np.ones_like(gray) * 255

        kernel_bh = self.morphology.get_kernel("rect", (7, 7))
        blackhat = self.morphology.black_hat(gray, kernel_bh)
        blackhat_roi = cv2.bitwise_and(blackhat, inspect_roi)

        _, raw_defect_mask = cv2.threshold(blackhat_roi, 30, 255, cv2.THRESH_BINARY)
        cleaned_defect_mask = self.morphology.clean_defect_mask(raw_defect_mask, min_area=20)
        defect_pixel_count = int(np.count_nonzero(cleaned_defect_mask))

        # 4. Feature Extraction: Edges, Corners, Hough
        edges = self.feature_detector.canny_edges(enhanced)
        _, corners = self.feature_detector.harris_corners(gray, threshold_ratio=0.02)
        hough_lines = self.feature_detector.detect_hough_lines(edges, threshold=40, min_line_length=25)
        hough_circles = self.feature_detector.detect_hough_circles(gray, min_radius=15, max_radius=min(h, w) // 2)

        # 5. Segmentation & Clustering
        watershed_res = self.segmenter.watershed_segmentation(image)
        clustered_img, cluster_labels = self.segmenter.kmeans_clustering(image, k=3)

        # 6. Classification & Decision Logic
        detected_status = "PASS"
        detected_defect = "PASS"
        confidence = 0.95

        # If trained ML classifier is available, run inference
        if self.classifier is not None and self.classifier.is_fitted:
            features = self.classifier.extract_image_features(image)
            detected_defect, confidence = self.classifier.predict(features)
            detected_status = "PASS" if detected_defect == "PASS" else "FAIL"
        else:
            # Domain heuristics fallback:
            # Excessive crack pixels indicate surface/fracture defect
            if defect_pixel_count > 50:
                detected_status = "FAIL"
                detected_defect = "DEFECT_CRACK"
                confidence = min(0.99, 0.70 + (defect_pixel_count / 1000.0))
            # Missing or multiple unexpected circular bores
            elif len(hough_circles) == 0:
                detected_status = "FAIL"
                detected_defect = "DEFECT_BORE"
                confidence = 0.88
            elif len(corners) > 150:
                # Highly jagged/corroded boundaries
                detected_status = "FAIL"
                detected_defect = "DEFECT_SURFACE"
                confidence = 0.85

        # 7. Diagnostic Image Annotation
        annotated = image.copy()

        # Draw detected circular bore
        for c in hough_circles:
            center = c["center"]
            radius = c["radius"]
            cv2.circle(annotated, center, radius, (0, 255, 0), 2)
            cv2.circle(annotated, center, 3, (0, 0, 255), -1)

        # Draw Harris corners (yellow dots)
        for pt in corners[:100]:
            cv2.circle(annotated, pt, 2, (0, 255, 255), -1)

        # Draw Hough lines (cyan)
        for x1, y1, x2, y2 in hough_lines[:20]:
            cv2.line(annotated, (x1, y1), (x2, y2), (255, 255, 0), 1)

        # Highlight defects with red contours
        defect_contours, _ = cv2.findContours(cleaned_defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, defect_contours, -1, (0, 0, 255), 2)

        # Status Banner
        banner_color = (0, 180, 0) if detected_status == "PASS" else (0, 0, 220)
        cv2.rectangle(annotated, (0, 0), (w, 36), banner_color, -1)
        banner_text = f"Status: {detected_status} | Class: {detected_defect} | Conf: {confidence*100:.1f}%"
        cv2.putText(
            annotated,
            banner_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        elapsed_ms = (time.time() - start_time) * 1000.0

        # Save artifacts if requested
        saved_files = {}
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            base = os.path.splitext(source_name)[0]

            annotated_path = os.path.join(save_dir, f"{prefix}_{base}_annotated.png")
            cv2.imwrite(annotated_path, annotated)
            saved_files["annotated"] = annotated_path

            edges_path = os.path.join(save_dir, f"{prefix}_{base}_edges.png")
            cv2.imwrite(edges_path, edges)
            saved_files["edges"] = edges_path

            defect_path = os.path.join(save_dir, f"{prefix}_{base}_defect_mask.png")
            cv2.imwrite(defect_path, cleaned_defect_mask)
            saved_files["defect_mask"] = defect_path

            watershed_path = os.path.join(save_dir, f"{prefix}_{base}_watershed.png")
            cv2.imwrite(watershed_path, watershed_res["segmented_overlay"])
            saved_files["watershed"] = watershed_path

            cluster_path = os.path.join(save_dir, f"{prefix}_{base}_kmeans.png")
            cv2.imwrite(cluster_path, clustered_img)
            saved_files["kmeans"] = cluster_path

        report = {
            "source": source_name,
            "status": detected_status,
            "defect_type": detected_defect,
            "confidence": round(float(confidence), 4),
            "execution_time_ms": round(elapsed_ms, 2),
            "metrics": {
                "image_resolution": [w, h],
                "defect_pixel_count": defect_pixel_count,
                "num_corners_detected": len(corners),
                "num_hough_lines": len(hough_lines),
                "num_hough_circles": len(hough_circles),
                "num_watershed_segments": watershed_res["num_segments"],
            },
            "artifacts": saved_files,
        }

        return report
