"""Conveyor Object Tracking Module (Module 5).

Implements Centroid Tracking for moving components on industrial conveyor systems,
tracking object lifecycles, bounding boxes, and velocity vectors across frames.
"""

from typing import List, Tuple, Dict, Any
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    """Tracks objects across sequential frames by minimizing centroid Euclidean distance."""

    def __init__(self, max_disappeared: int = 15, max_distance: float = 60.0):
        """Initializes the tracker.

        Args:
            max_disappeared: Consecutive frames an object can be missing before deregistration.
            max_distance: Maximum distance between centroids to consider them the same object.
        """
        self.next_object_id = 0
        self.objects: OrderedDict[int, Tuple[int, int]] = OrderedDict()
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        self.bboxes: OrderedDict[int, Tuple[int, int, int, int]] = OrderedDict()
        self.trajectories: OrderedDict[int, List[Tuple[int, int]]] = OrderedDict()
        self.velocities: OrderedDict[int, Tuple[float, float]] = OrderedDict()

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> int:
        """Registers a newly detected object."""
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.bboxes[obj_id] = bbox
        self.disappeared[obj_id] = 0
        self.trajectories[obj_id] = [centroid]
        self.velocities[obj_id] = (0.0, 0.0)
        self.next_object_id += 1
        return obj_id

    def deregister(self, object_id: int) -> None:
        """Removes an object that has left the camera field of view."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bboxes[object_id]
        del self.trajectories[object_id]
        del self.velocities[object_id]

    def update(self, rects: List[Tuple[int, int, int, int]]) -> Dict[int, Dict[str, Any]]:
        """Updates tracking state with detected bounding boxes in current frame.

        Args:
            rects: List of bounding boxes (startX, startY, width, height).

        Returns:
            Dictionary mapping object_id to tracking details.
        """
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self._format_results()

        input_centroids = np.zeros((len(rects), 2), dtype=int)
        for i, (x, y, w, h) in enumerate(rects):
            input_centroids[i] = (int(x + w / 2.0), int(y + h / 2.0))

        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(tuple(input_centroids[i]), rects[i])
            return self._format_results()

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # Compute Euclidean distance between existing and new centroids
        distances = np.linalg.norm(
            object_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :],
            axis=2,
        )

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if distances[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            old_centroid = self.objects[obj_id]
            new_centroid = tuple(input_centroids[col])

            # Update velocity
            vx = float(new_centroid[0] - old_centroid[0])
            vy = float(new_centroid[1] - old_centroid[1])
            self.velocities[obj_id] = (vx, vy)

            self.objects[obj_id] = new_centroid
            self.bboxes[obj_id] = rects[col]
            self.disappeared[obj_id] = 0
            self.trajectories[obj_id].append(new_centroid)

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(distances.shape[0])).difference(used_rows)
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        unused_cols = set(range(distances.shape[1])).difference(used_cols)
        for col in unused_cols:
            self.register(tuple(input_centroids[col]), rects[col])

        return self._format_results()

    def _format_results(self) -> Dict[int, Dict[str, Any]]:
        results = {}
        for obj_id in self.objects:
            results[obj_id] = {
                "id": obj_id,
                "centroid": self.objects[obj_id],
                "bbox": self.bboxes[obj_id],
                "velocity": self.velocities[obj_id],
                "history": list(self.trajectories[obj_id]),
                "disappeared": self.disappeared[obj_id],
            }
        return results
