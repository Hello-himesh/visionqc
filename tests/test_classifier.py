"""Unit tests for DefectClassifier (Module 5)."""

import pytest
import numpy as np
import cv2
from visionqc.classifier import DefectClassifier


@pytest.fixture
def sample_feature_dataset():
    np.random.seed(42)
    # Generate 20 samples of 100-dim features across 2 classes
    X_class0 = np.random.normal(0.0, 1.0, (10, 100))
    X_class1 = np.random.normal(5.0, 1.0, (10, 100))
    X = np.vstack([X_class0, X_class1]).astype(np.float32)
    y = np.array(["PASS"] * 10 + ["DEFECT_CRACK"] * 10)
    return X, y


def test_shape_and_hog_features():
    img = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(img, (64, 64), 30, 255, -1)

    hog_feat = DefectClassifier.extract_hog_features(img)
    assert len(hog_feat) > 0

    features = DefectClassifier.extract_image_features(img)
    assert len(features) > 0


def test_knn_fit_and_predict(sample_feature_dataset):
    X, y = sample_feature_dataset
    classifier = DefectClassifier(classifier_type="knn", n_neighbors=3, n_components=0.95)
    train_res = classifier.fit(X, y)

    assert train_res["accuracy"] > 0.8
    assert classifier.is_fitted

    label, conf = classifier.predict(X[0])
    assert label in ["PASS", "DEFECT_CRACK"]
    assert 0.0 <= conf <= 1.0


def test_naive_bayes_fit_and_predict(sample_feature_dataset):
    X, y = sample_feature_dataset
    classifier = DefectClassifier(classifier_type="naive_bayes", n_components=5)
    train_res = classifier.fit(X, y)

    assert train_res["accuracy"] > 0.8
    label, conf = classifier.predict(X[-1])
    assert label in ["PASS", "DEFECT_CRACK"]


def test_save_and_load(tmp_path, sample_feature_dataset):
    X, y = sample_feature_dataset
    classifier = DefectClassifier(classifier_type="knn", n_neighbors=3, n_components=5)
    classifier.fit(X, y)

    save_path = str(tmp_path / "model.pkl")
    classifier.save(save_path)

    loaded = DefectClassifier()
    loaded.load(save_path)
    assert loaded.is_fitted
    label, conf = loaded.predict(X[0])
    assert label in ["PASS", "DEFECT_CRACK"]
