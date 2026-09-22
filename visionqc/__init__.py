"""VisionQC: Automated Industrial Defect Detection & Quality Inspection Pipeline.

High-throughput, headless computer vision quality inspection platform for mechanical
components, featuring restoration, morphology, edge/corner analytics, segmentation,
machine-learning classification, and conveyor object tracking.
"""

__version__ = "1.1.0"
__author__ = "Nagahimesh Vuppala"

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
