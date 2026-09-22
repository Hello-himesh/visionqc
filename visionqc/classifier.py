"""Defect Classification and Statistical Learning Subsystem (Module 5).

Implements mathematical foundations for:
1. Multi-modal feature extraction (HOG, color/intensity statistics, contour shape geometry).
2. Dimensionality reduction via SVD-based Principal Component Analysis (PCA).
3. Distance-weighted K-Nearest Neighbors (KNN) and Gaussian Naive Bayes (GNB).
4. Serialization, training orchestrators, and diagnostic metric evaluation.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import pickle
import os
import cv2
import numpy as np


class PrincipalComponentAnalysis:
    """Computes Principal Component Analysis (PCA) using Singular Value Decomposition (SVD)."""

    def __init__(self, n_components: Union[int, float] = 0.95):
        self.n_components = n_components
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.n_components_: int = 0

    def fit(self, X: np.ndarray) -> "PrincipalComponentAnalysis":
        """Fits PCA on observation matrix X with shape (N, D)."""
        data = np.asarray(X, dtype=np.float64)
        n_obs, n_features = data.shape

        self.mean_ = np.mean(data, axis=0)
        centered = data - self.mean_

        # SVD: centered = U @ diag(S) @ Vt
        _, singular_values, v_t = np.linalg.svd(centered, full_matrices=False)

        # Variances = S^2 / (N - 1)
        variances = (singular_values ** 2) / float(max(1, n_obs - 1))
        var_sum = float(np.sum(variances))
        variance_ratios = variances / (var_sum + 1e-12)

        if isinstance(self.n_components, float) and 0.0 < self.n_components < 1.0:
            cumulative_ratios = np.cumsum(variance_ratios)
            component_count = int(np.searchsorted(cumulative_ratios, self.n_components) + 1)
            component_count = min(component_count, len(singular_values))
        else:
            component_count = int(self.n_components)
            component_count = min(component_count, n_obs, n_features)

        self.n_components_ = max(1, component_count)
        self.components_ = v_t[: self.n_components_]
        self.explained_variance_ratio_ = variance_ratios[: self.n_components_]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projects observations X onto the orthogonal principal subspace."""
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PrincipalComponentAnalysis model must be fitted before transform.")
        data = np.asarray(X, dtype=np.float64)
        return np.dot(data - self.mean_, self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits PCA and projects the input data."""
        return self.fit(X).transform(X)


class KNearestNeighbors:
    """K-Nearest Neighbors classifier supporting distance-weighted inverse voting."""

    def __init__(self, k: int = 3, weights: str = "distance"):
        self.k = max(1, int(k))
        self.weights = weights
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighbors":
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None or self.classes_ is None:
            raise RuntimeError("KNearestNeighbors classifier is not fitted.")

        queries = np.asarray(X, dtype=np.float64)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        # Compute pairwise Euclidean distances
        delta = queries[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
        distances = np.sqrt(np.sum(delta ** 2, axis=2))

        n_queries = queries.shape[0]
        n_classes = len(self.classes_)
        class_index = {cls_name: i for i, cls_name in enumerate(self.classes_)}
        probs = np.zeros((n_queries, n_classes), dtype=np.float64)

        k_neighbors = min(self.k, self.X_train.shape[0])

        for q_idx in range(n_queries):
            nearest = np.argsort(distances[q_idx])[:k_neighbors]
            for neighbor_idx in nearest:
                label = self.y_train[neighbor_idx]
                dist = distances[q_idx, neighbor_idx]
                w = (1.0 / (dist + 1e-5)) if self.weights == "distance" else 1.0
                probs[q_idx, class_index[label]] += w

            row_total = np.sum(probs[q_idx])
            if row_total > 0.0:
                probs[q_idx] /= row_total

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        posteriors = self.predict_proba(X)
        return self.classes_[np.argmax(posteriors, axis=1)]


class GaussianNaiveBayes:
    """Gaussian Naive Bayes with regularized variances and log-posterior inference."""

    def __init__(self):
        self.classes_: Optional[np.ndarray] = None
        self.priors_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.variances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        data = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)

        n_classes = len(self.classes_)
        n_features = data.shape[1]

        self.priors_ = np.zeros(n_classes, dtype=np.float64)
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.variances_ = np.zeros((n_classes, n_features), dtype=np.float64)

        for c_idx, cls in enumerate(self.classes_):
            subset = data[labels == cls]
            self.priors_[c_idx] = subset.shape[0] / float(data.shape[0])
            self.means_[c_idx] = np.mean(subset, axis=0)
            self.variances_[c_idx] = np.var(subset, axis=0) + 1e-4

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("GaussianNaiveBayes is not fitted.")

        data = np.asarray(X, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_samples = data.shape[0]
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes), dtype=np.float64)

        for c_idx in range(n_classes):
            mu = self.means_[c_idx]
            sigma_sq = self.variances_[c_idx]
            prior = self.priors_[c_idx]

            log_lik = -0.5 * np.sum(
                np.log(2.0 * np.pi * sigma_sq) + ((data - mu) ** 2) / sigma_sq,
                axis=1,
            )
            log_probs[:, c_idx] = np.log(prior + 1e-10) + log_lik

        max_logs = np.max(log_probs, axis=1, keepdims=True)
        exp_vals = np.exp(log_probs - max_logs)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class DefectClassifier:
    """Integrated pipeline feature extractor and defect classification engine."""

    DEFECT_CLASSES = ["PASS", "DEFECT_CRACK", "DEFECT_SURFACE", "DEFECT_BORE"]

    def __init__(
        self,
        classifier_type: str = "knn",
        n_neighbors: int = 3,
        n_components: Union[int, float] = 0.95,
    ):
        self.classifier_type = classifier_type.lower()
        self.n_neighbors = n_neighbors
        self.n_components = n_components

        self.pca = PrincipalComponentAnalysis(n_components=n_components)
        self.model = (
            KNearestNeighbors(k=n_neighbors)
            if self.classifier_type == "knn"
            else GaussianNaiveBayes()
        )
        self.is_fitted = False

    @staticmethod
    def extract_shape_features(contour: np.ndarray) -> np.ndarray:
        """Computes geometric invariant descriptors: area, perimeter, circularity, aspect ratio, solidity, extent."""
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, closed=True))
        circularity = (4.0 * np.pi * area) / (perimeter ** 2 + 1e-6)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(h + 1e-6)
        extent = area / float(w * h + 1e-6)

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / float(hull_area + 1e-6)

        return np.array([area, perimeter, circularity, aspect_ratio, solidity, extent], dtype=np.float32)

    @staticmethod
    def extract_hog_features(
        image: np.ndarray,
        cell_size: Tuple[int, int] = (16, 16),
        n_bins: int = 8,
    ) -> np.ndarray:
        """Extracts spatial Histogram of Oriented Gradients (HOG) feature vector."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        resized = cv2.resize(gray, (128, 128))

        dx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
        angle = np.mod(angle, 180.0)

        bin_width = 180.0 / n_bins
        h, w = resized.shape
        cx, cy = cell_size
        descriptor_list = []

        for y in range(0, h, cy):
            for x in range(0, w, cx):
                cell_mag = magnitude[y : y + cy, x : x + cx]
                cell_ang = angle[y : y + cy, x : x + cx]

                bins = np.clip((cell_ang / bin_width).astype(int), 0, n_bins - 1)
                hist = np.zeros(n_bins, dtype=np.float32)

                for b in range(n_bins):
                    hist[b] = np.sum(cell_mag[bins == b])

                hist_norm = hist / (np.linalg.norm(hist) + 1e-6)
                descriptor_list.extend(hist_norm)

        return np.array(descriptor_list, dtype=np.float32)

    @staticmethod
    def extract_image_features(image: np.ndarray) -> np.ndarray:
        """Assembles unified descriptor: HOG + Color/Intensity Moments + Dominant Shape."""
        hog_vec = DefectClassifier.extract_hog_features(image)

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_vals, std_vals = cv2.meanStdDev(image)
            moments = np.concatenate([mean_vals.ravel(), std_vals.ravel()]).astype(np.float32)
        else:
            gray = image.copy()
            mean_val, std_val = cv2.meanStdDev(gray)
            moments = np.array([mean_val[0][0], std_val[0][0]], dtype=np.float32)

        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            primary_contour = max(contours, key=cv2.contourArea)
            shape_vec = DefectClassifier.extract_shape_features(primary_contour)
        else:
            shape_vec = np.zeros(6, dtype=np.float32)

        return np.concatenate([hog_vec, moments, shape_vec])

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fits PCA space and model parameters."""
        data = np.asarray(X, dtype=np.float64)
        targets = np.asarray(y)

        self.pca.fit(data)
        reduced = self.pca.transform(data)

        self.model.fit(reduced, targets)
        self.is_fitted = True

        predictions = self.model.predict(reduced)
        acc = float(np.mean(targets == predictions))
        variance_ratio = float(np.sum(self.pca.explained_variance_ratio_))

        return {
            "accuracy": acc,
            "pca_components": self.pca.n_components_,
            "explained_variance_ratio": variance_ratio,
        }

    def predict(self, feature_vector: np.ndarray) -> Tuple[str, float]:
        """Infers class label and confidence score for a single feature vector."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before calling predict.")

        vector_2d = feature_vector.reshape(1, -1)
        subspace_vec = self.pca.transform(vector_2d)

        class_probabilities = self.model.predict_proba(subspace_vec)[0]
        max_idx = int(np.argmax(class_probabilities))
        pred_label = str(self.model.classes_[max_idx])
        conf = float(class_probabilities[max_idx])

        return pred_label, conf

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Calculates evaluation metrics and confusion matrix."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before calling evaluate.")

        test_proj = self.pca.transform(X_test)
        preds = self.model.predict(test_proj)

        accuracy = float(np.mean(y_test == preds))
        classes = self.model.classes_
        cls_to_idx = {name: i for i, name in enumerate(classes)}

        confusion = np.zeros((len(classes), len(classes)), dtype=int)
        for ground_truth, prediction in zip(y_test, preds):
            confusion[cls_to_idx[ground_truth], cls_to_idx[prediction]] += 1

        return {
            "accuracy": accuracy,
            "classes": classes.tolist(),
            "confusion_matrix": confusion.tolist(),
        }

    def save(self, file_path: str) -> None:
        """Serializes trained PCA and model states."""
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "wb") as handle:
            pickle.dump({
                "pca": self.pca,
                "model": self.model,
                "classifier_type": self.classifier_type,
                "is_fitted": self.is_fitted,
            }, handle)

    def load(self, file_path: str) -> None:
        """Deserializes classifier states from disk."""
        with open(file_path, "rb") as handle:
            state = pickle.load(handle)
        self.pca = state["pca"]
        self.model = state["model"]
        self.classifier_type = state["classifier_type"]
        self.is_fitted = state["is_fitted"]
