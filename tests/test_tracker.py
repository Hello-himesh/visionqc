"""Unit test suite for CentroidTracker kinematics and lifecycle tracking."""

import pytest
from visionqc.tracker import CentroidTracker


def test_tracker_lifecycle():
    """Verifies track creation, position/velocity update, and disappearance deregistration."""
    tracker = CentroidTracker(max_disappeared=3, max_distance=50.0)

    # Frame 1: Object detected at (100, 100) with dimensions 40x40
    frame1_boxes = [(80, 80, 40, 40)]
    tracks_f1 = tracker.update(frame1_boxes)
    assert len(tracks_f1) == 1
    assert 0 in tracks_f1
    assert tracks_f1[0]["centroid"] == (100, 100)

    # Frame 2: Object shifts along the conveyor to (110, 100)
    frame2_boxes = [(90, 80, 40, 40)]
    tracks_f2 = tracker.update(frame2_boxes)
    assert len(tracks_f2) == 1
    assert 0 in tracks_f2
    assert tracks_f2[0]["centroid"] == (110, 100)
    assert tracks_f2[0]["velocity"] == (10.0, 0.0)

    # Frame 3: Object momentarily not detected
    tracker.update([])
    assert len(tracker.objects) == 1
    assert tracker.disappeared[0] == 1

    # Frames 4, 5, 6: Exceeds disappearance budget of 3 frames -> deregisters
    tracker.update([])
    tracker.update([])
    tracker.update([])
    assert len(tracker.objects) == 0
