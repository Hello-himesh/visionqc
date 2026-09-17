"""Defect Classification and Dimensionality Reduction Module (Module 5).

Implements mathematical foundations of:
1. Feature extraction: Histogram of Oriented Gradients (HOG), color moments, shape descriptors.
2. Dimensionality Reduction: Principal Component Analysis (PCA via SVD).
3. Machine Learning Classifiers: K-Nearest Neighbors (KNN) & Gaussian Naive Bayes (GNB).
4. Model evaluation: Confusion matrix, precision, recall, and accuracy metrics.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import pickle
import os
import cv2
import numpy as np


class PrincipalComponentAnalysis:
    """Principal Component Analysis (PCA) implemented via Singular Value Decomposition (SVD).

    X_centered = X - mu
    U, S, Vt = SVD(X_centered)
    Components = Vt[:n_components]
    """

    def __init__(self, n_components: Union[int, float] = 0.95):
        """Args:
            n_components: If float in (0, 1), retains components explaining that variance ratio.
                          If int, retains that exact number of components.
        """
        self.n_components = n_components
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.n_components_: int = 0

    def fit(self, X: np.ndarray) -> "PrincipalComponentAnalysis":
        """Fits PCA on data matrix X of shape (N, D)."""
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # SVD: X_centered = U @ diag(S) @ Vt
        _, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Variance explained: s^2 / (n_samples - 1)
        var_explained = (s ** 2) / max(1, (n_samples - 1))
        total_var = np.sum(var_explained)
        var_ratio = var_explained / (total_var + 1e-12)

        if isinstance(self.n_components, float) and 0.0 < self.n_components < 1.0:
            cumulative_var = np.cumsum(var_ratio)
            k = int(np.searchsorted(cumulative_var, self.n_components) + 1)
            k = min(k, len(s))
        else:
            k = int(self.n_components)
            k = min(k, n_samples, n_features)

        self.n_components_ = max(1, k)
        self.components_ = Vt[: self.n_components_]
        self.explained_variance_ratio_ = var_ratio[: self.n_components_]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projects data X onto principal components."""
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class KNearestNeighbors:
    """K-Nearest Neighbors classifier with Euclidean distance and distance weighting."""

    def __init__(self, k: int = 3, weights: str = "distance"):
        self.k = k
        self.weights = weights
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighbors":
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        best_indices = np.argmax(probs, axis=1)
        return self.classes_[best_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None or self.classes_ is None:
            raise RuntimeError("KNN is not fitted.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Pairwise Euclidean distances: (N_test, N_train)
        diff = X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)

        n_queries = X.shape[0]
        n_classes = len(self.classes_)
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        probabilities = np.zeros((n_queries, n_classes), dtype=np.float64)

        k = min(self.k, self.X_train.shape[0])

        for i in range(n_queries):
            nearest_indices = np.argsort(distances[i])[:k]
            nearest_labels = self.y_train[nearest_indices]
            nearest_dists = distances[i][nearest_indices]

            for label, d in zip(nearest_labels, nearest_dists):
                weight = 1.0 / (d + 1e-5) if self.weights == "distance" else 1.0
                probabilities[i, class_to_idx[label]] += weight

            row_sum = np.sum(probabilities[i])
            if row_sum > 0:
                probabilities[i] /= row_sum

        return probabilities


class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier with log-likelihood estimation."""

    def __init__(self):
        self.classes_: Optional[np.ndarray] = None
        self.priors_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.variances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.priors_ = np.zeros(n_classes, dtype=np.float64)
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.variances_ = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            self.priors_[idx] = X_c.shape[0] / float(X.shape[0])
            self.means_[idx] = np.mean(X_c, axis=0)
            self.variances_[idx] = np.var(X_c, axis=0) + 1e-4

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Classifier not fitted.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_posteriors = np.zeros((n_samples, n_classes), dtype=np.float64)

        for idx in range(n_classes):
            mean = self.means_[idx]
            var = self.variances_[idx]
            prior = self.priors_[idx]

            # log P(x | C) = -0.5 * sum(log(2*pi*var) + (x - mean)^2 / var)
            log_likelihood = -0.5 * np.sum(np.log(2.0 * np.pi * var) + ((X - mean) ** 2) / var, axis=1)
            log_posteriors[:, idx] = np.log(prior + 1e-9) + log_likelihood

        # Softmax for probabilities
        max_log = np.max(log_posteriors, axis=1, keepdims=True)
        exp_post = np.exp(log_posteriors - max_log)
        return exp_post / np.sum(exp_post, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class DefectClassifier:
    """Composite feature extractor and classifier combining PCA and KNN/Naive Bayes."""

    DEFECT_CLASSES = ["PASS", "DEFECT_CRACK", "DEFECT_SURFACE", "DEFECT_BORE"]

    def __init__(self, classifier_type: str = "knn", n_neighbors: int = 3, n_components: Union[int, float] = 0.95):
        self.classifier_type = classifier_type.lower()
        self.n_neighbors = n_neighbors
        self.n_components = n_components

        self.pca = PrincipalComponentAnalysis(n_components=n_components)
        self.model = KNearestNeighbors(k=n_neighbors) if self.classifier_type == "knn" else GaussianNaiveBayes()
        self.is_fitted = False

    @staticmethod
    def extract_shape_features(contour: np.ndarray) -> np.ndarray:
        """Extracts geometric shape descriptors from a contour."""
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, closed=True))
        circularity = (4.0 * np.pi * area) / (perimeter**2 + 1e-6)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(h + 1e-6)
        extent = area / float(w * h + 1e-6)

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / float(hull_area + 1e-6)

        return np.array([area, perimeter, circularity, aspect_ratio, solidity, extent], dtype=np.float32)

    @staticmethod
    def extract_hog_features(image: np.ndarray, cell_size: Tuple[int, int] = (16, 16), n_bins: int = 8) -> np.ndarray:
        """Computes Histogram of Oriented Gradients (HOG) descriptor."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        resized = cv2.resize(gray, (128, 128))
        gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        angle = angle % 180.0
        bin_width = 180.0 / n_bins

        h, w = resized.shape
        cx, cy = cell_size
        hist_list = []

        for y in range(0, h, cy):
            for x in range(0, w, cx):
                cell_mag = mag[y : y + cy, x : x + cx]
                cell_ang = angle[y : y + cy, x : x + cx]
                cell_bins = (cell_ang / bin_width).astype(int)
                cell_bins = np.clip(cell_bins, 0, n_bins - 1)

                hist = np.zeros(n_bins, dtype=np.float32)
                for b in range(n_bins):
                    hist[b] = np.sum(cell_mag[cell_bins == b])

                norm = np.linalg.norm(hist) + 1e-6
                hist_list.extend(hist / norm)

        return np.array(hist_list, dtype=np.float32)

    @staticmethod
    def extract_image_features(image: np.ndarray) -> np.ndarray:
        """Extracts composite feature vector: HOG + Color Statistics + Dominant Contour shape."""
        hog_feat = DefectClassifier.extract_hog_features(image)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_c, std_c = cv2.meanStdDev(image)
            color_stats = np.concatenate([mean_c.flatten(), std_c.flatten()]).astype(np.float32)
        else:
            gray = image.copy()
            mean_g, std_g = cv2.meanStdDev(gray)
            color_stats = np.array([mean_g[0][0], std_g[0][0]], dtype=np.float32)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            shape_feat = DefectClassifier.extract_shape_features(largest)
        else:
            shape_feat = np.zeros(6, dtype=np.float32)

        return np.concatenate([hog_feat, color_stats, shape_feat])

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fits PCA and classifier."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.pca.fit(X)
        X_reduced = self.pca.transform(X)

        self.model.fit(X_reduced, y)
        self.is_fitted = True

        y_pred = self.model.predict(X_reduced)
        acc = float(np.mean(y == y_pred))
        explained_var = float(np.sum(self.pca.explained_variance_ratio_))

        return {
            "accuracy": acc,
            "pca_components": self.pca.n_components_,
            "explained_variance_ratio": explained_var,
        }

    def predict(self, feature_vector: np.ndarray) -> Tuple[str, float]:
        """Predicts label and confidence score for a feature vector."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict.")

        x_2d = feature_vector.reshape(1, -1)
        x_reduced = self.pca.transform(x_2d)

        probs = self.model.predict_proba(x_reduced)[0]
        best_idx = int(np.argmax(probs))
        label = str(self.model.classes_[best_idx])
        confidence = float(probs[best_idx])

        return label, confidence

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluates model performance metrics."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        X_test_red = self.pca.transform(X_test)
        preds = self.model.predict(X_test_red)

        acc = float(np.mean(y_test == preds))
        classes = self.model.classes_

        # Confusion matrix
        c_mat = np.zeros((len(classes), len(classes)), dtype=int)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for yt, yp in zip(y_test, preds):
            c_mat[class_to_idx[yt], class_to_idx[yp]] += 1

        return {
            "accuracy": acc,
            "classes": classes.tolist(),
            "confusion_matrix": c_mat.tolist(),
        }

    def save(self, file_path: str) -> None:
        """Saves model to disk."""
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({
                "pca": self.pca,
                "model": self.model,
                "classifier_type": self.classifier_type,
                "is_fitted": self.is_fitted,
            }, f)

    def load(self, file_path: str) -> None:
        """Loads model from disk."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.pca = data["pca"]
        self.model = data["model"]
        self.classifier_type = data["classifier_type"]
        self.is_fitted = data["is_fitted"]
