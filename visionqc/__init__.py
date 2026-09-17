"""VisionQC: Automated Industrial Defect Detection & Quality Inspection Pipeline.

A complete computer vision system implementing end-to-end image processing,
feature extraction, morphological inspection, segmentation, classification,
and tracking for manufacturing quality control.
"""

__version__ = "1.0.0"
__author__ = "Student / VisionQC Team"

from visionqc.preprocessor import ImagePreprocessor
from visionqc.morphology import MorphologicalAnalyzer
from visionqc.edge_corner import FeatureDetector
from visionqc.segmentation import ImageSegmenter
from visionqc.classifier import DefectClassifier
from visionqc.tracker import CentroidTracker
from visionqc.pipeline import InspectionPipeline

__all__ = [
    "ImagePreprocessor",
    "MorphologicalAnalyzer",
    "FeatureDetector",
    "ImageSegmenter",
    "DefectClassifier",
    "CentroidTracker",
    "InspectionPipeline",
]
