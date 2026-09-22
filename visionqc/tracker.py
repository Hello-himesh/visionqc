"""Conveyor Motion Tracking and Centroid Kinematics Subsystem (Module 5).

Implements Euclidean distance association to track mechanical parts passing along
conveyor belts, estimating instantaneous velocity vectors, trajectory histories,
and component departure lifecycles.
"""

from typing import List, Tuple, Dict, Any
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    """Associates component bounding boxes across frames via Euclidean centroid proximity."""

    def __init__(self, max_disappeared: int = 15, max_distance: float = 60.0):
        self.next_object_id = 0
        self.objects: OrderedDict[int, Tuple[int, int]] = OrderedDict()
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        self.bboxes: OrderedDict[int, Tuple[int, int, int, int]] = OrderedDict()
        self.trajectories: OrderedDict[int, List[Tuple[int, int]]] = OrderedDict()
        self.velocities: OrderedDict[int, Tuple[float, float]] = OrderedDict()

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> int:
        """Registers a newly discovered part on the conveyor line."""
        oid = self.next_object_id
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.disappeared[oid] = 0
        self.trajectories[oid] = [centroid]
        self.velocities[oid] = (0.0, 0.0)
        self.next_object_id += 1
        return oid

    def deregister(self, object_id: int) -> None:
        """Removes an object that has exited the conveyor field of view."""
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)
        self.bboxes.pop(object_id, None)
        self.trajectories.pop(object_id, None)
        self.velocities.pop(object_id, None)

    def update(self, rects: List[Tuple[int, int, int, int]]) -> Dict[int, Dict[str, Any]]:
        """Processes current frame bounding boxes and matches them with existing tracks."""
        if not rects:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self._format_results()

        input_centroids = np.zeros((len(rects), 2), dtype=int)
        for i, (bx, by, bw, bh) in enumerate(rects):
            input_centroids[i] = (int(bx + bw / 2.0), int(by + bh / 2.0))

        if not self.objects:
            for i, rect in enumerate(rects):
                self.register(tuple(input_centroids[i]), rect)
            return self._format_results()

        tracked_ids = list(self.objects.keys())
        tracked_centroids = np.array(list(self.objects.values()))

        # Distance matrix (M tracked x N new detections)
        diff = tracked_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

        row_order = distance_matrix.min(axis=1).argsort()
        col_order = distance_matrix.argmin(axis=1)[row_order]

        matched_rows = set()
        matched_cols = set()

        for r, c in zip(row_order, col_order):
            if r in matched_rows or c in matched_cols:
                continue
            if distance_matrix[r, c] > self.max_distance:
                continue

            oid = tracked_ids[r]
            prior_pos = self.objects[oid]
            curr_pos = tuple(input_centroids[c])

            vx = float(curr_pos[0] - prior_pos[0])
            vy = float(curr_pos[1] - prior_pos[1])
            self.velocities[oid] = (vx, vy)

            self.objects[oid] = curr_pos
            self.bboxes[oid] = rects[c]
            self.disappeared[oid] = 0
            self.trajectories[oid].append(curr_pos)

            matched_rows.add(r)
            matched_cols.add(c)

        # Handle unmatched existing tracks
        for r, oid in enumerate(tracked_ids):
            if r not in matched_rows:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        # Handle new incoming tracks
        for c in range(len(rects)):
            if c not in matched_cols:
                self.register(tuple(input_centroids[c]), rects[c])

        return self._format_results()

    def _format_results(self) -> Dict[int, Dict[str, Any]]:
        """Formats current active tracking states into structured dictionary."""
        records = {}
        for oid in self.objects:
            records[oid] = {
                "id": oid,
                "centroid": self.objects[oid],
                "bbox": self.bboxes[oid],
                "velocity": self.velocities[oid],
                "history": list(self.trajectories[oid]),
                "disappeared": self.disappeared[oid],
            }
        return records
