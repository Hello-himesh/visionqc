# VisionQC: Automated Industrial Defect Detection & Quality Inspection Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-30%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-Headless%204.10-orange.svg)](https://opencv.org/)

An automated, end-to-end computer vision quality inspection system designed for manufacturing assembly lines and conveyor systems. Built specifically for the **VITyarthi Computer Vision** flipped course project evaluation.

VisionQC provides comprehensive image restoration, morphological defect extraction, multi-scale edge and corner detection, Hough geometric transform verification, marker-controlled watershed segmentation, K-Means/K-Medoids color clustering, PCA dimensionality reduction, and machine-learning defect classification (`PASS`, `DEFECT_CRACK`, `DEFECT_SURFACE`, `DEFECT_BORE`).

---

## 📑 Table of Contents
- [Overview](#overview)
- [Key Features & Syllabus Alignment](#key-features--syllabus-alignment)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Execution & Usage Guide](#execution--usage-guide)
  - [1. One-Command Self-Contained Demo](#1-one-command-self-contained-demo)
  - [2. Inspecting Custom Images](#2-inspecting-custom-images)
  - [3. Structured JSON Output](#3-structured-json-output)
  - [4. Training Custom Classifier](#4-training-custom-classifier)
- [Running Automated Unit Tests](#running-automated-unit-tests)
- [Output Artifacts & Diagnostic Overlays](#output-artifacts--diagnostic-overlays)
- [Course Syllabus Concept Mapping](#course-syllabus-concept-mapping)
- [License](#license)

---

## Overview

In high-throughput industrial production, manual visual inspection is slow, subjective, and prone to operator fatigue. **VisionQC** replaces manual inspection with a deterministic, mathematically rigorous computer vision pipeline.

The system processes industrial mechanical parts (spur gears, washers, bushings, brackets) in real-time, extracts multi-modal features, isolates micro-fractures and surface oxidation, evaluates geometric concentricity, and classifies parts with quantifiable statistical confidence.

Importantly, **VisionQC is 100% headless and executable via the command line**. It has zero GUI/desktop dependencies, making it suitable for automated evaluation environments and edge devices.

---

## Key Features & Syllabus Alignment

1. **Adaptive Image Restoration (Modules 1 & 2)**
   - Color space conversions (`BGR`, `Grayscale`, `HSV`, `LAB`).
   - Gaussian, Median, and Bilateral edge-preserving noise filters.
   - Dynamic illumination correction via Power-Law (Gamma: $s = c \cdot r^\gamma$) and Logarithmic transformations ($s = c \cdot \log(1 + r)$).
   - Contrast Limited Adaptive Histogram Equalization (CLAHE).
2. **Morphological Defect Profiler (Module 2)**
   - Structuring elements (`rect`, `cross`, `ellipse`).
   - Erosion, Dilation, Morphological Opening (dust suppression), and Closing (gap bridging).
   - High-contrast Black-Hat ($T_{BH} = \text{Closing}(I) - I$) and Top-Hat transforms to isolate fissures and surface pitting.
   - Dynamic solid-body contour masking to eliminate false positives at part boundaries.
3. **Geometric & Structural Analysis (Module 3)**
   - Sobel operator for gradient magnitude and spatial orientation calculation.
   - Canny edge detector with median-adaptive threshold estimation.
   - Harris Corner Detector ($R = \det(M) - k \cdot \text{trace}(M)^2$) and Shi-Tomasi Good Features to Track for mechanical alignment.
   - Standard & Probabilistic Hough Line Transform (`cv2.HoughLinesP`) for crack orientation and edge straightness.
   - Hough Circle Transform (`cv2.HoughCircles`) for bore diameter, eccentricity, and circularity verification.
4. **Segmentation & Clustering (Module 4)**
   - Marker-controlled Watershed segmentation using distance transform peaks to segment touching parts.
   - K-Means clustering in Lab color-texture space to isolate oxidation and surface discoloration.
   - K-Medoids clustering for outlier-resistant representative cluster centers.
5. **Machine Learning Classification & Tracking (Module 5)**
   - Multi-modal feature extraction: Cell-based Histogram of Oriented Gradients (HOG), color/intensity moments, and contour shape descriptors (circularity, aspect ratio, solidity, extent).
   - Principal Component Analysis (PCA) dimensionality reduction retaining $\ge 95\%$ variance.
   - Distance-weighted K-Nearest Neighbors (KNN) and Gaussian Naive Bayes (GNB) classifiers.
   - Centroid Tracker with Euclidean distance matching, velocity estimation, and lifecycle management for moving conveyor belts.

---

## System Architecture

```text
                                [Input Image / Conveyor Stream]
                                              │
                                              ▼
                             ┌───────────────────────────────────┐
                             │    1. Image Preprocessing         │
                             │  - Bilateral / Gaussian Filtering │
                             │  - Gamma & Log Illumination Corr  │
                             │  - CLAHE Histogram Equalization   │
                             └─────────────────┬─────────────────┘
                                               │
                      ┌────────────────────────┴────────────────────────┐
                      ▼                                                 ▼
        ┌───────────────────────────┐                     ┌───────────────────────────┐
        │  2. Morphological Defect  │                     │  3. Feature Detection     │
        │     Extraction            │                     │  - Canny & Sobel Edges    │
        │  - Black-Hat & Top-Hat    │                     │  - Harris & Shi-Tomasi    │
        │  - Solid Body ROI Masking │                     │  - Hough Lines & Circles  │
        │  - Area Filter Cleaning   │                     └─────────────┬─────────────┘
        └─────────────┬─────────────┘                                   │
                      │                                                 │
                      └────────────────────────┬────────────────────────┘
                                               │
                                               ▼
                             ┌───────────────────────────────────┐
                             │  4. Segmentation & Clustering     │
                             │  - Marker-Controlled Watershed    │
                             │  - Distance Transform Sure FG/BG  │
                             │  - K-Means Lab Color Clustering   │
                             └─────────────────┬─────────────────┘
                                               │
                                               ▼
                             ┌───────────────────────────────────┐
                             │  5. Classification & Inference    │
                             │  - HOG + Shape + Moment Extraction│
                             │  - PCA Dimensionality Reduction   │
                             │  - Distance-Weighted KNN / GNB    │
                             └─────────────────┬─────────────────┘
                                               │
                                               ▼
                             ┌───────────────────────────────────┐
                             │  6. Reporting & Visual Overlays   │
                             │  - Diagnostic Overlay Generation  │
                             │  - Quantitative JSON Metrics      │
                             │  - Terminal Table & Status Code   │
                             └───────────────────────────────────┘
```

---

## Repository Structure

```text
visionqc/
├── README.md                 # Complete setup, execution, and evaluation guide
├── statement.md              # Problem statement, scope, target users, high-level features
├── requirements.txt          # Python package dependencies
├── setup.py                  # Package installation configuration
├── visionqc/                 # Core Python source package
│   ├── __init__.py           # Package exports and versioning
│   ├── preprocessor.py       # Filters, gamma/log transforms, CLAHE (Modules 1 & 2)
│   ├── morphology.py         # Erosion, dilation, opening/closing, black-hat (Module 2)
│   ├── edge_corner.py        # Canny, Sobel, Harris corners, Hough transforms (Module 3)
│   ├── segmentation.py       # Watershed segmentation, K-Means, K-Medoids (Module 4)
│   ├── classifier.py         # PCA, KNN, Gaussian Naive Bayes, HOG (Module 5)
│   ├── tracker.py            # Centroid object detection & conveyor tracking (Module 5)
│   ├── pipeline.py           # Inspection orchestrator and diagnostic generator
│   └── cli.py                # Command-line interface with argparse and JSON output
├── tests/                    # Comprehensive unit test suite (30 tests, 100% pass)
│   ├── __init__.py
│   ├── test_preprocessor.py  # Tests noise filters, color spaces, transformations
│   ├── test_morphology.py    # Tests structuring elements, open/close, black-hat
│   ├── test_edge_corner.py   # Tests Canny, Sobel, Harris corners, Hough lines/circles
│   ├── test_segmentation.py  # Tests Watershed and K-Means/K-Medoids
│   ├── test_classifier.py    # Tests PCA dimensionality reduction, KNN, Naive Bayes
│   ├── test_tracker.py       # Tests CentroidTracker registration, tracking, lifecycle
│   ├── test_pipeline.py      # Tests end-to-end inspection pipeline
│   └── test_cli.py           # Tests command-line interface arguments and demo
├── data/                     # Synthetic test generator and sample components
│   ├── __init__.py
│   ├── synthetic_generator.py# Generates mechanical components with controlled defects
│   └── samples/              # Train & test partitions (PASS, CRACK, SURFACE, BORE)
└── docs/
    └── PROJECT_REPORT.md     # Full 15-section project report
```

---

## Installation & Setup

### Prerequisites
- Python 3.9 or newer (tested on Python 3.9, 3.10, 3.11, 3.12)
- Virtual environment recommended (`venv`)

### 1. Clone the Repository
```bash
git clone https://github.com/Hello-himesh/visionqc.git
cd visionqc
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python3 -m venv .venv

# Activate environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Execution & Usage Guide

### 1. One-Command Self-Contained Demo
The demo automatically creates benchmark mechanical parts, trains the PCA + KNN classifier, inspects the test set, displays a clean summary table in the terminal, and saves diagnostic overlays:

```bash
python -m visionqc.cli --demo --save-dir results_demo
```

**Sample Terminal Output:**
```text
================================================================
 VisionQC: Automated Industrial Defect Inspection System Demo   
================================================================
[*] Generating synthetic industrial component dataset in 'data/samples'...
[*] Training PCA + KNN Defect Classifier on synthetic samples...
    -> Trained on 20 samples across 4 classes.
    -> PCA reduced feature vector to 1 components.
    -> Retained 98.5% explained variance.

[*] Running Inspection Pipeline on test components (saving to 'results_demo')...
    [✘ FAIL] defect_bore_001.png            Class: DEFECT_BORE     Conf: 100.0% (109.3 ms)
    [✘ FAIL] defect_bore_002.png            Class: DEFECT_BORE     Conf: 100.0% (99.3 ms)
    [✘ FAIL] defect_bore_003.png            Class: DEFECT_BORE     Conf: 100.0% (109.9 ms)
    [✘ FAIL] defect_crack_001.png           Class: DEFECT_CRACK    Conf: 100.0% (107.1 ms)
    [✘ FAIL] defect_crack_002.png           Class: DEFECT_CRACK    Conf: 100.0% (96.3 ms)
    [✘ FAIL] defect_crack_003.png           Class: DEFECT_CRACK    Conf: 100.0% (99.1 ms)
    [✘ FAIL] defect_surface_001.png         Class: DEFECT_SURFACE  Conf: 100.0% (89.8 ms)
    [✘ FAIL] defect_surface_002.png         Class: DEFECT_SURFACE  Conf: 100.0% (102.2 ms)
    [✘ FAIL] defect_surface_003.png         Class: DEFECT_SURFACE  Conf: 100.0% (138.0 ms)
    [✔ PASS] pass_001.png                   Class: PASS            Conf: 100.0% (98.9 ms)
    [✔ PASS] pass_002.png                   Class: PASS            Conf: 100.0% (108.9 ms)
    [✔ PASS] pass_003.png                   Class: PASS            Conf: 66.2% (105.7 ms)

----------------------------------------------------------------
Inspection Summary: 12 parts inspected | 3 PASSED | 9 DEFECTIVE
Diagnostic overlays and masks saved to: /absolute/path/results_demo
================================================================
```

### 2. Inspecting Custom Images
Inspect a single image or an entire folder:
```bash
# Single image
python -m visionqc.cli --input data/samples/test/DEFECT_CRACK/defect_crack_001.png --save-dir output/

# Batch folder inspection
python -m visionqc.cli --input data/samples/test/PASS/ --save-dir output/
```

### 3. Structured JSON Output
For automated grading pipelines or machine-to-machine integrations:
```bash
python -m visionqc.cli --input data/samples/test/DEFECT_CRACK/defect_crack_001.png --json
```

**Output:**
```json
{
  "source": "defect_crack_001.png",
  "status": "FAIL",
  "defect_type": "DEFECT_CRACK",
  "confidence": 0.849,
  "execution_time_ms": 105.84,
  "metrics": {
    "image_resolution": [256, 256],
    "defect_pixel_count": 149,
    "num_corners_detected": 434,
    "num_hough_lines": 421,
    "num_hough_circles": 2,
    "num_watershed_segments": 1
  },
  "artifacts": {
    "annotated": "results_demo/inspect_defect_crack_001_annotated.png",
    "edges": "results_demo/inspect_defect_crack_001_edges.png",
    "defect_mask": "results_demo/inspect_defect_crack_001_defect_mask.png",
    "watershed": "results_demo/inspect_defect_crack_001_watershed.png",
    "kmeans": "results_demo/inspect_defect_crack_001_kmeans.png"
  }
}
```

### 4. Training Custom Classifier
```bash
python -m visionqc.cli --train --model-path models/defect_classifier.pkl
```

---

## Running Automated Unit Tests

VisionQC features a comprehensive `pytest` test suite covering all mathematical algorithms, transformations, edge detectors, segmentation engines, and tracking logic:

```bash
# Run entire test suite with verbose output
pytest tests/ -v
```

**Results:**
```text
tests/test_classifier.py::test_shape_and_hog_features PASSED             [  3%]
tests/test_classifier.py::test_knn_fit_and_predict PASSED                [  6%]
tests/test_classifier.py::test_naive_bayes_fit_and_predict PASSED        [ 10%]
tests/test_classifier.py::test_save_and_load PASSED                      [ 13%]
tests/test_cli.py::test_cli_help PASSED                                  [ 16%]
tests/test_cli.py::test_cli_demo PASSED                                  [ 20%]
tests/test_edge_corner.py::test_sobel_edges PASSED                       [ 23%]
tests/test_edge_corner.py::test_canny_edges PASSED                       [ 26%]
tests/test_edge_corner.py::test_harris_corners PASSED                    [ 30%]
tests/test_edge_corner.py::test_shi_tomasi_corners PASSED                [ 33%]
tests/test_edge_corner.py::test_hough_lines PASSED                       [ 36%]
tests/test_edge_corner.py::test_hough_circles PASSED                     [ 40%]
tests/test_morphology.py::test_structuring_elements PASSED               [ 43%]
tests/test_morphology.py::test_erosion_dilation PASSED                   [ 46%]
tests/test_morphology.py::test_opening_closing PASSED                    [ 50%]
tests/test_morphology.py::test_transforms PASSED                         [ 53%]
tests/test_morphology.py::test_clean_defect_mask PASSED                  [ 56%]
tests/test_pipeline.py::test_pipeline_inspect PASSED                     [ 60%]
tests/test_preprocessor.py::test_convert_color PASSED                    [ 63%]
tests/test_preprocessor.py::test_resize_with_aspect_ratio PASSED         [ 66%]
tests/test_preprocessor.py::test_flip PASSED                             [ 70%]
tests/test_preprocessor.py::test_remove_noise PASSED                     [ 73%]
tests/test_preprocessor.py::test_linear_contrast_stretch PASSED          [ 76%]
tests/test_preprocessor.py::test_log_transform PASSED                    [ 80%]
tests/test_preprocessor.py::test_power_law_transform PASSED              [ 83%]
tests/test_preprocessor.py::test_equalize_histogram PASSED               [ 86%]
tests/test_segmentation.py::test_watershed_segmentation PASSED           [ 90%]
tests/test_segmentation.py::test_kmeans_clustering PASSED                [ 93%]
tests/test_segmentation.py::test_kmedoids_clustering PASSED              [ 96%]
tests/test_tracker.py::test_tracker_lifecycle PASSED                     [100%]

============================== 30 passed in 3.43s ==============================
```

---

## Output Artifacts & Diagnostic Overlays

For every inspected part, VisionQC generates 5 diagnostic images:
1. `*_annotated.png`: Full diagnostic overlay featuring Pass/Fail status banner, classification label, confidence score, detected circular bores (green circles), Harris corner keypoints (yellow dots), Hough lines (cyan), and defect contours (red).
2. `*_edges.png`: Canny edge boundary map.
3. `*_defect_mask.png`: Binary mask of cleaned surface defects.
4. `*_watershed.png`: Multi-color watershed segment boundaries separating foreground from background.
5. `*_kmeans.png`: Perceptual Lab color clustering isolating surface textures.

---

## Course Syllabus Concept Mapping

| Module | Curriculum Topic | VisionQC Implementation & File Location |
| :--- | :--- | :--- |
| **Module 1** | Image Formation & OpenCV Basics | [`ImagePreprocessor.load_image`](file:///visionqc/preprocessor.py), [`convert_color`](file:///visionqc/preprocessor.py), [`flip`](file:///visionqc/preprocessor.py) |
| **Module 2** | Image Enhancement & Noise Removal | [`remove_noise`](file:///visionqc/preprocessor.py) (Gaussian, Median, Bilateral) |
| | Transformations (Linear, Log, Power-law) | [`linear_contrast_stretch`](file:///visionqc/preprocessor.py), [`log_transform`](file:///visionqc/preprocessor.py), [`power_law_transform`](file:///visionqc/preprocessor.py) |
| | Morphological Operations | [`MorphologicalAnalyzer`](file:///visionqc/morphology.py) (Erode, Dilate, Open, Close, Gradient, Top-Hat, Black-Hat) |
| | Histogram Equalization | [`equalize_histogram`](file:///visionqc/preprocessor.py) (Global & Adaptive CLAHE) |
| **Module 3** | Edge Detection | [`FeatureDetector.sobel_edges`](file:///visionqc/edge_corner.py), [`canny_edges`](file:///visionqc/edge_corner.py) |
| | Corner Detection | [`harris_corners`](file:///visionqc/edge_corner.py), [`shi_tomasi_corners`](file:///visionqc/edge_corner.py) |
| | Hough Transform | [`detect_hough_lines`](file:///visionqc/edge_corner.py), [`detect_hough_circles`](file:///visionqc/edge_corner.py) |
| **Module 4** | Watershed Segmentation | [`ImageSegmenter.watershed_segmentation`](file:///visionqc/segmentation.py) |
| | Clustering (K-Means & K-Medoids) | [`kmeans_clustering`](file:///visionqc/segmentation.py), [`kmedoids_clustering`](file:///visionqc/segmentation.py) |
| **Module 5** | Dimensionality Reduction (PCA) | [`PrincipalComponentAnalysis`](file:///visionqc/classifier.py) (SVD-based decomposition) |
| | Classification (KNN & Naive Bayes) | [`KNearestNeighbors`](file:///visionqc/classifier.py), [`GaussianNaiveBayes`](file:///visionqc/classifier.py) |
| | Object Detection & Tracking | [`CentroidTracker`](file:///visionqc/tracker.py) with Euclidean trajectory matching |

---

## License
MIT License. Created for the VITyarthi Computer Vision course evaluation.
