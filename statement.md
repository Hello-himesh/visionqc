# Project Statement: VisionQC

## 1. Problem Statement
High-throughput assembly and manufacturing operations—such as precision gear manufacturing, bearing fabrication, and mechanical stamping—depend on rigorous quality assurance to intercept defective components before assembly. Subtle mechanical flaws, including hairline fracture cracks, surface pitting from corrosion/oxidation, bore diameter deformations, and missing gear teeth, compromise mechanical durability and can cause catastrophic machinery failure.

Conventional manual visual inspection is subjective, unscalable, prone to inspector fatigue, and introduces prohibitive latency on high-speed conveyor lines. Commercial computer vision inspection systems, conversely, are typically closed-source, cost-prohibitive, tied to proprietary camera hardware, and difficult to adapt to evolving defect taxonomies.

VisionQC provides an open, deterministic, and modular automated visual inspection pipeline capable of real-time image conditioning, morphological defect isolation, geometric validation, multi-part segmentation, statistical machine learning classification, and conveyor object tracking—without requiring desktop graphical user interfaces.

## 2. Scope of the Project
The functional scope of **VisionQC** includes:
- **Image Acquisition & Preprocessing**: Ingestion and validation of industrial component imagery across diverse color representations (`BGR`, `GRAY`, `HSV`, `LAB`), noise attenuation (Gaussian, Median, Bilateral), dynamic illumination adjustments (logarithmic and power-law gamma transforms), and contrast enhancement via CLAHE.
- **Surface Anomaly Extraction**: Mathematical morphological operations (structuring elements, erosion, dilation, opening, closing, morphological gradient, Top-Hat, and Black-Hat) to isolate cracks and porous surface anomalies from component metal substrates.
- **Geometric & Structural Validation**: Structural edge extraction using Canny and Sobel operators, keypoint identification via Harris and Shi-Tomasi detectors, and concentricity/dimension verification using Hough Line and Hough Circle Transforms.
- **Region-of-Interest Segmentation**: Disentangling touching parts via marker-controlled Watershed segmentation driven by distance transforms, alongside texture/color grouping via K-Means and K-Medoids in Lab space.
- **Defect Classification & Tracking**: High-dimensional feature engineering (HOG, intensity moments, contour geometry), variance-preserving dimensionality reduction via Principal Component Analysis (PCA), supervised classification (`PASS`, `DEFECT_CRACK`, `DEFECT_SURFACE`, `DEFECT_BORE`) using distance-weighted K-Nearest Neighbors (KNN) and Gaussian Naive Bayes, and Euclidean centroid object tracking across conveyor frames.
- **Headless Execution & Reporting**: Fully automated CLI supporting batch evaluation, structured JSON metrics, diagnostic visual overlay generation, and standard exit codes.

**Out of Scope**: Direct hardware bus integration (e.g., Modbus/EtherCAT PLC protocols) and mechanical sorting actuators.

## 3. Target Users
- **Quality Assurance Engineers**: Automating inspection audits, measuring yield percentages, and tracking defect taxonomy trends.
- **Automation & Robotics Developers**: Deploying edge computer vision inference nodes directly on conveyor sorting lines.
- **Plant Operations Managers**: Reviewing machine-readable JSON telemetry and defect logging for root-cause analysis.
- **Computer Vision Researchers & Students**: Studying foundational image processing, morphological filtering, and classical machine learning workflows.

## 4. High-Level Features
1. **Adaptive Image Restoration**: Edge-preserving noise filtration combined with dynamic power-law gamma illumination correction ($s = c \cdot r^\gamma$) and CLAHE.
2. **Morphological Defect Profiler**: High-contrast Black-Hat and Top-Hat filtering coupled with solid-body ROI masking to eliminate false alarms at part boundaries.
3. **Multi-Scale Geometric Inspection**: Automated Canny edge boundary tracing and Hough Circle bore measurement to verify internal concentricity and outer profile circularity.
4. **Marker-Controlled Watershed Separation**: Automated distance-transform-guided watershed segmentation to cleanly partition touching or clustered mechanical parts.
5. **PCA-Powered Machine Learning Classification**: High-dimensional HOG and shape feature compression via Principal Component Analysis (retaining $\ge 95\%$ variance) followed by distance-weighted KNN and probabilistic Naive Bayes classification.
6. **Centroid Conveyor Tracking Engine**: Real-time object identification, trajectory calculation, velocity tracking, and automated lifecycle handling for components passing along a production conveyor.
7. **Production-Ready Headless CLI**: Self-contained `--demo` execution, batch inspection, standardized exit codes, and JSON reporting optimized for automated CI/CD and evaluation pipelines.
