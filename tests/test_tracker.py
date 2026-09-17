"""Unit tests for CentroidTracker (Module 5)."""

import pytest
from visionqc.tracker import CentroidTracker


def test_tracker_lifecycle():
    tracker = CentroidTracker(max_disappeared=3, max_distance=50.0)

    # Frame 1: One object appears at (100, 100) with size (40, 40)
    rects_f1 = [(80, 80, 40, 40)]
    tracked_f1 = tracker.update(rects_f1)
    assert len(tracked_f1) == 1
    assert 0 in tracked_f1
    assert tracked_f1[0]["centroid"] == (100, 100)

    # Frame 2: Object moves slightly along conveyor belt to (110, 100)
    rects_f2 = [(90, 80, 40, 40)]
    tracked_f2 = tracker.update(rects_f2)
    assert len(tracked_f2) == 1
    assert 0 in tracked_f2
    assert tracked_f2[0]["centroid"] == (110, 100)
    assert tracked_f2[0]["velocity"] == (10.0, 0.0)

    # Frame 3: Object disappears
    tracker.update([])
    assert len(tracker.objects) == 1
    assert tracker.disappeared[0] == 1

    # Frame 4, 5, 6: Exceeds max_disappeared (3) -> should deregister
    tracker.update([])
    tracker.update([])
    tracker.update([])
    assert len(tracker.objects) == 0
