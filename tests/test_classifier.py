"""Unit test suite for DefectClassifier, PCA, KNN, and Naive Bayes."""

import pytest
import numpy as np
import cv2
from visionqc.classifier import DefectClassifier


@pytest.fixture
def sample_feature_dataset():
    """Generates synthetic 100-dimensional feature arrays across two separable classes."""
    np.random.seed(42)
    class_a = np.random.normal(0.0, 1.0, (10, 100))
    class_b = np.random.normal(5.0, 1.0, (10, 100))
    data_x = np.vstack([class_a, class_b]).astype(np.float32)
    data_y = np.array(["PASS"] * 10 + ["DEFECT_CRACK"] * 10)
    return data_x, data_y


def test_shape_and_hog_features():
    """Verifies HOG and shape feature extraction from test geometry."""
    circle_mat = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(circle_mat, (64, 64), 30, 255, -1)

    hog_vec = DefectClassifier.extract_hog_features(circle_mat)
    assert len(hog_vec) > 0

    composite = DefectClassifier.extract_image_features(circle_mat)
    assert len(composite) > 0


def test_knn_fit_and_predict(sample_feature_dataset):
    """Verifies PCA projection and KNN classification."""
    features, labels = sample_feature_dataset
    classifier = DefectClassifier(classifier_type="knn", n_neighbors=3, n_components=0.95)
    training_metrics = classifier.fit(features, labels)

    assert training_metrics["accuracy"] > 0.8
    assert classifier.is_fitted

    pred_label, confidence = classifier.predict(features[0])
    assert pred_label in ["PASS", "DEFECT_CRACK"]
    assert 0.0 <= confidence <= 1.0


def test_naive_bayes_fit_and_predict(sample_feature_dataset):
    """Verifies Gaussian Naive Bayes training and prediction."""
    features, labels = sample_feature_dataset
    classifier = DefectClassifier(classifier_type="naive_bayes", n_components=5)
    training_metrics = classifier.fit(features, labels)

    assert training_metrics["accuracy"] > 0.8
    pred_label, confidence = classifier.predict(features[-1])
    assert pred_label in ["PASS", "DEFECT_CRACK"]


def test_save_and_load(tmp_path, sample_feature_dataset):
    """Verifies model serialization and deserialization."""
    features, labels = sample_feature_dataset
    model = DefectClassifier(classifier_type="knn", n_neighbors=3, n_components=5)
    model.fit(features, labels)

    save_target = str(tmp_path / "model.pkl")
    model.save(save_target)

    restored = DefectClassifier()
    restored.load(save_target)
    assert restored.is_fitted

    label, conf = restored.predict(features[0])
    assert label in ["PASS", "DEFECT_CRACK"]
