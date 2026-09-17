# Project Statement: VisionQC

## 1. Problem Statement
In modern manufacturing and high-throughput assembly lines (such as automotive transmission systems, mechanical powertrains, and precision machinery), manufactured components such as spur gears, bushings, bearings, and machined brackets frequently suffer from surface defects, fatigue fractures, micro-cracks, bore eccentricity, and missing gear teeth. 

Traditional manual quality inspection is slow, labor-intensive, subjective, prone to human visual fatigue, and cannot scale to high-speed conveyor environments where hundreds of parts pass per minute. Existing commercial computer vision systems are often proprietary, expensive, tightly bound to proprietary cameras, and lack modular adaptability to custom defect taxonomies. 

There is an urgent engineering need for a modular, transparent, and computationally efficient automated visual inspection pipeline that can process component images, detect geometric and structural anomalies, classify defect categories, and track components on moving conveyor belts with zero dependency on desktop graphical user interfaces.

## 2. Scope of the Project
The scope of **VisionQC** encompasses:
- **Image Acquisition & Preprocessing**: Ingesting high-resolution grayscale and color images of industrial mechanical components, mitigating sensor noise, and correcting uneven factory illumination through dynamic linear/log/power-law transformations and adaptive histogram equalization (CLAHE).
- **Surface Anomaly Isolation**: Utilizing mathematical morphology (erosion, dilation, opening, closing, morphological gradient, and top-hat/black-hat filters) to extract micro-cracks, scratches, and porous surface anomalies.
- **Geometric & Structural Validation**: Detecting structural boundaries using Canny and Sobel edge detection, keypoint integrity via Harris and Shi-Tomasi corner detectors, and dimension verification (bore diameter, concentricity, circularity) via Hough Line and Hough Circle Transforms.
- **Region-of-Interest Segmentation**: Disentangling overlapping parts and fixtures using marker-controlled Watershed segmentation and segmenting surface texture discolorations using K-Means and K-Medoids clustering.
- **Defect Categorization**: Extracting scale/rotation-invariant shape descriptors, intensity moments, and Histogram of Oriented Gradients (HOG), applying Principal Component Analysis (PCA) for dimensionality reduction, and classifying components into quality classes (`PASS`, `DEFECT_CRACK`, `DEFECT_SURFACE`, `DEFECT_BORE`) using K-Nearest Neighbors (KNN) and Gaussian Naive Bayes classifiers.
- **Conveyor Object Tracking**: Multi-object tracking across simulated video streams using Euclidean distance centroid association to monitor velocity vectors, conveyor throughput, and part departure.
- **Headless Execution & Reporting**: Full command-line interface execution (CLI) generating structured JSON reports and saving diagnostic annotated visual overlays without requiring GUI display servers.

**Out of Scope**: Real-time PLC hardware interfacing (Modbus/Profinet) and physical pneumatic actuator sorting mechanics.

## 3. Target Users
- **Quality Assurance (QA) Engineers**: Manufacturing and plant quality engineers monitoring assembly line defect rates and part tolerances.
- **Industrial Automation Developers**: Engineers deploying edge-computing computer vision nodes on conveyor belts and sorting stations.
- **Plant Operations Managers**: Supervisors requiring quantitative yield statistics, defect frequency reports, and automated audit trails.
- **Academic Evaluators & Researchers**: Students and instructors studying classical computer vision, mathematical image processing, and machine learning integration.

## 4. High-Level Features
1. **Adaptive Image Restoration**: Multi-scale Gaussian, Median, and Bilateral filtering coupled with dynamic gamma ($s = c \cdot r^\gamma$) illumination correction and CLAHE contrast enhancement.
2. **Morphological Defect Profiler**: High-contrast Black-Hat and Top-Hat filtering to isolate sub-millimeter cracks and surface pitting from high-reflectance metal backgrounds.
3. **Multi-Scale Geometric Inspection**: Automated Canny edge boundary tracing and Hough Circle bore measurement to verify internal concentricity and outer profile circularity.
4. **Marker-Controlled Watershed Separation**: Automated distance-transform-guided watershed segmentation to cleanly partition touching or clustered mechanical parts.
5. **PCA-Powered Machine Learning Classification**: High-dimensional HOG and shape feature compression via Principal Component Analysis (retaining $\ge 95\%$ variance) followed by distance-weighted KNN and probabilistic Naive Bayes classification.
6. **Centroid Conveyor Tracking Engine**: Real-time object identification, trajectory calculation, velocity tracking, and automated lifecycle handling for components passing along a production conveyor.
7. **Production-Ready Headless CLI**: Self-contained `--demo` execution, batch inspection, standardized exit codes, and JSON reporting optimized for automated CI/CD and evaluation pipelines.
