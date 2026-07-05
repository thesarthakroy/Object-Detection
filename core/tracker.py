"""
Tracker Module.
Implements dependency-free centroid-IoU object tracking.
"""
from typing import List, Dict, Tuple, Any
import numpy as np

def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Computes Intersection over Union (IoU) of two boxes in [x, y, w, h] format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


class Track:
    """Represents a single tracked target instance."""

    def __init__(self, global_id: int, class_id: int, class_specific_id: int, bbox: List[int], confidence: float):
        self.global_id = global_id
        self.class_id = class_id
        self.class_specific_id = class_specific_id  # category ID tracker (e.g., Person #3)
        self.bbox = bbox  # [x, y, w, h]
        self.confidence = confidence
        self.age = 0
        self.centroid = self._get_centroid(bbox)

    def _get_centroid(self, bbox: List[int]) -> Tuple[float, float]:
        x, y, w, h = bbox
        return x + w / 2.0, y + h / 2.0

    def update(self, bbox: List[int], confidence: float) -> None:
        """Updates internal target coordinates and sets active age to 0."""
        self.bbox = bbox
        self.confidence = confidence
        self.centroid = self._get_centroid(bbox)
        self.age = 0


class ObjectTracker:
    """Updates, matches, and prunes active targets over frame sequences."""

    def __init__(self, max_age: int = 15, min_iou: float = 0.15):
        self.max_age = max_age
        self.min_iou = min_iou
        self.next_global_id = 1
        self.class_id_counters: Dict[int, int] = {}
        self.tracks: Dict[int, Track] = {}

    def update(self, bboxes: List[List[int]], class_ids: List[int], confidences: List[float]) -> List[Track]:
        """Runs match pairing over frame detection results."""
        # Age existing tracks
        for track in self.tracks.values():
            track.age += 1

        active_track_ids = list(self.tracks.keys())
        detections_to_match = list(range(len(bboxes)))

        matches: List[Tuple[int, int]] = []

        if active_track_ids and detections_to_match:
            # Build cost matrix
            iou_matrix = np.zeros((len(active_track_ids), len(bboxes)))
            for i, t_id in enumerate(active_track_ids):
                for j in range(len(bboxes)):
                    if self.tracks[t_id].class_id == class_ids[j]:
                        iou_matrix[i, j] = compute_iou(self.tracks[t_id].bbox, bboxes[j])

            # Match greedily
            while True:
                max_idx = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
                max_iou = iou_matrix[max_idx]

                if max_iou < self.min_iou:
                    break

                i, j = max_idx
                t_id = active_track_ids[i]
                
                matches.append((t_id, j))
                
                iou_matrix[i, :] = -1.0
                iou_matrix[:, j] = -1.0

        # Update matched tracks
        matched_track_ids = set()
        matched_detection_ids = set()
        for t_id, d_idx in matches:
            self.tracks[t_id].update(bboxes[d_idx], confidences[d_idx])
            matched_track_ids.add(t_id)
            matched_detection_ids.add(d_idx)

        # Spawn new tracks
        for d_idx in range(len(bboxes)):
            if d_idx not in matched_detection_ids:
                class_id = class_ids[d_idx]
                
                if class_id not in self.class_id_counters:
                    self.class_id_counters[class_id] = 1
                class_specific_id = self.class_id_counters[class_id]
                self.class_id_counters[class_id] += 1

                new_track = Track(
                    global_id=self.next_global_id,
                    class_id=class_id,
                    class_specific_id=class_specific_id,
                    bbox=bboxes[d_idx],
                    confidence=confidences[d_idx]
                )
                self.tracks[self.next_global_id] = new_track
                self.next_global_id += 1

        # Prune expired tracks
        expired_track_ids = [t_id for t_id, track in self.tracks.items() if track.age > self.max_age]
        for t_id in expired_track_ids:
            del self.tracks[t_id]

        return [track for track in self.tracks.values() if track.age == 0]
