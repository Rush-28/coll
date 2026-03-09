"""
vehicle_detector.py
===================
Vehicle-ONLY detector for bike blind-spot collision avoidance.

Runs a YOLOv8n TFLite model and filters strictly to vehicle classes only:
  COCO id  2  -> car
  COCO id  3  -> motorcycle
  COCO id  5  -> bus
  COCO id  7  -> truck
  COCO id  1  -> bicycle  (included for safety – cyclists are a hazard too)

Key improvements over the original:
  - Non-Maximum Suppression (NMS) removes duplicate boxes
  - Returns rich DetectionResult objects (box, confidence, class, zone)
  - Zone classification: CRITICAL / WARNING / CLEAR based on box area
  - Supports both float32 and uint8 (quantized) TFLite models
  - Thread-safe: one interpreter, mutex-protected inference call
  - Handles both YOLOv8 "transposed" and classic output formats
"""

import cv2
import numpy as np
import threading
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# COCO class IDs that represent vehicles we care about
VEHICLE_CLASS_IDS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Confidence threshold: detections below this are discarded
CONFIDENCE_THRESHOLD = 0.45

# NMS IoU threshold: overlapping boxes above this are suppressed
NMS_IOU_THRESHOLD = 0.45

# Zone thresholds based on bounding-box area as fraction of frame area
# A larger box = closer vehicle = higher danger
ZONE_CRITICAL_AREA_RATIO = 0.18   # box covers >18% of frame → CRITICAL
ZONE_WARNING_AREA_RATIO  = 0.05   # box covers >5%  of frame → WARNING
                                   # below 5%                 → MONITOR


class ThreatZone(Enum):
    CLEAR    = 0   # No vehicle detected
    MONITOR  = 1   # Far away, just track
    WARNING  = 2   # Getting close
    CRITICAL = 3   # Imminent collision risk


@dataclass
class Detection:
    """A single confirmed vehicle detection."""
    class_id:   int
    class_name: str
    confidence: float
    # Bounding box in pixel coords [x1, y1, x2, y2]
    box: List[int]
    # Box centre as fraction of frame [0–1]
    cx: float
    cy: float
    # Area of box as fraction of frame area
    area_ratio: float
    # Danger zone derived from area
    zone: ThreatZone


@dataclass
class DetectionResult:
    """Complete result for one frame."""
    vehicle_detected: bool
    detections: List[Detection] = field(default_factory=list)
    highest_zone: ThreatZone = ThreatZone.CLEAR
    # Highest single confidence among all detections
    max_confidence: float = 0.0
    frame_width: int = 0
    frame_height: int = 0


# ---------------------------------------------------------------------------
# VehicleDetector
# ---------------------------------------------------------------------------

class VehicleDetector:
    """
    Loads a YOLOv8 TFLite model and runs vehicle-only inference.

    Usage
    -----
    detector = VehicleDetector("yolov8n.tflite")
    result   = detector.detect(frame)   # frame is an OpenCV BGR numpy array
    """

    def __init__(self, model_path: str = "yolov8n.tflite"):
        self._lock = threading.Lock()
        self._load_model(model_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str):
        """Load TFLite interpreter and cache tensor details."""
        try:
            # Try tflite_runtime first (lighter, for Raspberry Pi)
            from tflite_runtime.interpreter import Interpreter
            logger.info("Using tflite_runtime Interpreter")
        except ImportError:
            # Fall back to full TensorFlow
            from tensorflow.lite.python.interpreter import Interpreter
            logger.info("Using tensorflow.lite Interpreter")

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        inp = self.input_details[0]
        self.input_h = inp['shape'][1]
        self.input_w = inp['shape'][2]
        self.is_quantized = (inp['dtype'] == np.uint8)

        logger.info(
            f"Model loaded | input=({self.input_h}x{self.input_w}) "
            f"| quantized={self.is_quantized}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run vehicle detection on a BGR OpenCV frame.

        Returns
        -------
        DetectionResult with all confirmed vehicle detections.
        """
        if frame is None or frame.size == 0:
            return DetectionResult(vehicle_detected=False)

        h, w = frame.shape[:2]
        blob = self._preprocess(frame)

        with self._lock:
            self.interpreter.set_tensor(self.input_details[0]['index'], blob)
            self.interpreter.invoke()
            raw_output = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )

        detections = self._postprocess(raw_output, w, h)

        if not detections:
            return DetectionResult(vehicle_detected=False, frame_width=w, frame_height=h)

        max_conf  = max(d.confidence for d in detections)
        highest_z = max(detections, key=lambda d: d.zone.value).zone

        return DetectionResult(
            vehicle_detected=True,
            detections=detections,
            highest_zone=highest_z,
            max_confidence=max_conf,
            frame_width=w,
            frame_height=h,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalise frame to model input format."""
        resized = cv2.resize(frame, (self.input_w, self.input_h))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if self.is_quantized:
            blob = np.expand_dims(rgb, axis=0).astype(np.uint8)
        else:
            blob = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)

        return blob

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _postprocess(
        self, raw: np.ndarray, frame_w: int, frame_h: int
    ) -> List[Detection]:
        """
        Parse model output, apply NMS, filter to vehicle classes only.

        Handles two common YOLOv8 TFLite output shapes:
          - (1, num_boxes, 85+)   — classic [cx, cy, w, h, obj, cls...]
          - (1, 85+, num_boxes)   — transposed (YOLOv8 default export)
        """
        output = raw[0]  # Remove batch dim

        # YOLOv8 exports in transposed format: (85, 8400) → transpose to (8400, 85)
        if output.ndim == 2 and output.shape[0] < output.shape[1]:
            output = output.T  # → (num_boxes, 85+)

        num_classes_total = output.shape[1] - 4  # subtract cx,cy,w,h

        boxes_xyxy   = []
        confidences  = []
        class_ids    = []

        for row in output:
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            class_scores   = row[4:]

            best_cls  = int(np.argmax(class_scores))
            best_conf = float(class_scores[best_cls])

            # Vehicle-only filter
            if best_cls not in VEHICLE_CLASS_IDS:
                continue
            if best_conf < CONFIDENCE_THRESHOLD:
                continue

            # Convert cx/cy/w/h (normalised [0-1]) → pixel xyxy
            x1 = int((cx - bw / 2) * frame_w)
            y1 = int((cy - bh / 2) * frame_h)
            x2 = int((cx + bw / 2) * frame_w)
            y2 = int((cy + bh / 2) * frame_h)

            # Clamp to frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)

            boxes_xyxy.append([x1, y1, x2 - x1, y2 - y1])  # cv2 NMS needs xywh
            confidences.append(best_conf)
            class_ids.append(best_cls)

        if not boxes_xyxy:
            return []

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy, confidences, CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD
        )
        if len(indices) == 0:
            return []

        frame_area = frame_w * frame_h
        results: List[Detection] = []

        for i in indices.flatten():
            x, y, bw, bh = boxes_xyxy[i]
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            conf     = confidences[i]
            cls_id   = class_ids[i]
            cls_name = VEHICLE_CLASS_IDS[cls_id]

            area_ratio = (bw * bh) / frame_area
            zone       = self._classify_zone(area_ratio)

            results.append(Detection(
                class_id   = cls_id,
                class_name = cls_name,
                confidence = conf,
                box        = [x1, y1, x2, y2],
                cx         = (x1 + x2) / 2 / frame_w,
                cy         = (y1 + y2) / 2 / frame_h,
                area_ratio = area_ratio,
                zone       = zone,
            ))

        # Sort by danger (zone desc, then confidence desc)
        results.sort(key=lambda d: (d.zone.value, d.confidence), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_zone(area_ratio: float) -> ThreatZone:
        if area_ratio >= ZONE_CRITICAL_AREA_RATIO:
            return ThreatZone.CRITICAL
        elif area_ratio >= ZONE_WARNING_AREA_RATIO:
            return ThreatZone.WARNING
        else:
            return ThreatZone.MONITOR

    def annotate_frame(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of the frame (for debug)."""
        out = frame.copy()
        colours = {
            ThreatZone.CRITICAL: (0, 0, 255),
            ThreatZone.WARNING:  (0, 165, 255),
            ThreatZone.MONITOR:  (0, 255, 0),
        }
        for det in result.detections:
            x1, y1, x2, y2 = det.box
            colour = colours.get(det.zone, (200, 200, 200))
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            label = f"{det.class_name} {det.confidence:.2f} [{det.zone.name}]"
            cv2.putText(out, label, (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        return out
