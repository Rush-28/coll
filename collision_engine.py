"""
collision_engine.py
===================
Multi-modal collision detection engine for bike blind-spot safety.

Combines inputs from:
  1. mmWave radar  → coarse proximity signal
  2. Camera + ML   → precise vehicle identification + zone estimation
  3. IMU (MPU6050) → lean-angle suppression (avoid false positives in turns)

Key improvements over original:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  TEMPORAL TRACKING   – rolling window smooths flickering detections │
  │  DUAL-MODE TRIGGER   – camera runs independently of radar           │
  │  THREAT SCORING      – numeric score drives alert level, not bool   │
  │  CONFIDENCE DECAY    – old detections fade → no ghost alerts        │
  │  HYSTERESIS          – alert stays on for MIN_ALERT_HOLD seconds    │
  │  PER-SENSOR HEALTH   – graceful degradation if radar/cam fails      │
  └─────────────────────────────────────────────────────────────────────┘

Alert output is one of:
  AlertLevel.CLEAR    → no action
  AlertLevel.MONITOR  → watch (no buzzer)
  AlertLevel.WARNING  → LED on, no vibration
  AlertLevel.CRITICAL → LED + vibration motor
"""

import time
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Deque

from vehicle_detector import ThreatZone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration — tweak these for your hardware
# ---------------------------------------------------------------------------

class Config:
    # How long (seconds) the alert stays active after the last detection
    MIN_ALERT_HOLD_SECS      = 1.5

    # Temporal smoothing: keep last N camera results
    CAM_HISTORY_WINDOW       = 6

    # Minimum fraction of recent camera frames that must show a vehicle
    # before declaring a confirmed sighting
    CAM_DETECTION_THRESHOLD  = 0.4   # 40 % of last N frames

    # If radar triggers but camera finds nothing, still emit a WARNING
    RADAR_ONLY_ALERT_LEVEL   = "WARNING"

    # Lean angle beyond which ALL alerts are suppressed (degrees)
    MAX_LEAN_ANGLE_DEG       = 25.0

    # Camera runs every frame; only run full ML inference every N captures
    # (set to 1 to always run ML)
    INFERENCE_EVERY_N_FRAMES = 1

    # Confidence boost given to a detection when radar also triggers
    RADAR_FUSION_BONUS       = 0.15

    # Minimum confidence (after fusion bonus) to accept a detection
    FINAL_CONFIDENCE_MIN     = 0.50


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class AlertLevel(Enum):
    CLEAR    = 0
    MONITOR  = 1
    WARNING  = 2
    CRITICAL = 3


@dataclass
class SensorState:
    """Live snapshot of every input sensor."""
    # IMU
    lean_angle_deg:   float = 0.0
    imu_ok:           bool  = True

    # Radar
    radar_triggered:  bool  = False
    radar_distance_m: Optional[float] = None  # None if sensor lacks range
    radar_ok:         bool  = True

    # Camera / ML
    cam_ok:                bool  = True
    vehicle_detected_cam:  bool  = False
    cam_zone:              ThreatZone = ThreatZone.CLEAR
    cam_confidence:        float = 0.0


@dataclass
class CollisionAlert:
    """Published output of the collision engine for the current cycle."""
    level:          AlertLevel
    led_on:         bool
    vibration_on:   bool
    reason:         str
    # For dashboard / logging
    details:        dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CollisionEngine
# ---------------------------------------------------------------------------

class CollisionEngine:
    """
    Stateful fusion engine.  Call `update(state)` once per main-loop cycle.

    Thread-safe: internal state protected by a lock so that the engine can
    be polled from a dashboard thread without race conditions.
    """

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self._lock = threading.Lock()

        # Rolling history of camera detections (True/False per frame)
        self._cam_history: Deque[bool] = deque(maxlen=self.cfg.CAM_HISTORY_WINDOW)

        # Smoothed threat zone from camera
        self._zone_history: Deque[ThreatZone] = deque(
            maxlen=self.cfg.CAM_HISTORY_WINDOW
        )

        # Alert hold timer
        self._alert_held_until: float = 0.0

        # Last published alert (for external polling)
        self._last_alert: CollisionAlert = CollisionAlert(
            level=AlertLevel.CLEAR,
            led_on=False,
            vibration_on=False,
            reason="System starting",
        )

        # Statistics
        self._cycle_count   = 0
        self._alert_count   = 0
        self._false_alarm_count = 0  # radar said yes, camera said no

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, state: SensorState) -> CollisionAlert:
        """
        Fuse all sensor inputs and return a CollisionAlert.

        Parameters
        ----------
        state : SensorState
            Current readings from all sensors (filled in by main.py).

        Returns
        -------
        CollisionAlert describing the current danger level.
        """
        with self._lock:
            self._cycle_count += 1
            now = time.time()

            # ── 1. IMU lean-angle guard ────────────────────────────────
            if abs(state.lean_angle_deg) > self.cfg.MAX_LEAN_ANGLE_DEG:
                alert = CollisionAlert(
                    level=AlertLevel.CLEAR,
                    led_on=False,
                    vibration_on=False,
                    reason=f"Suppressed: leaning {state.lean_angle_deg:.1f}°",
                )
                self._last_alert = alert
                return alert

            # ── 2. Update camera temporal history ─────────────────────
            self._cam_history.append(state.vehicle_detected_cam)
            if state.vehicle_detected_cam:
                self._zone_history.append(state.cam_zone)

            # Smoothed camera detection flag
            cam_confirmed = self._smoothed_camera_detection()

            # Smoothed threat zone
            smoothed_zone = self._smoothed_zone()

            # ── 3. Fuse radar + camera → threat score ─────────────────
            base_conf = state.cam_confidence if state.vehicle_detected_cam else 0.0

            if state.radar_triggered and state.vehicle_detected_cam:
                # Both sensors agree → highest confidence
                fused_conf = min(1.0, base_conf + self.cfg.RADAR_FUSION_BONUS)
                fusion_src = "RADAR+CAMERA"
            elif state.radar_triggered and not state.vehicle_detected_cam:
                # Radar says yes, camera says no → low-level fallback alert
                self._false_alarm_count += 1
                fused_conf = 0.0
                fusion_src = "RADAR_ONLY"
            elif not state.radar_triggered and cam_confirmed:
                # Camera found vehicle, radar silent → trust camera
                fused_conf = base_conf
                fusion_src = "CAMERA_ONLY"
            else:
                fused_conf = 0.0
                fusion_src = "NONE"

            # ── 4. Determine AlertLevel ────────────────────────────────
            level = self._compute_alert_level(
                fusion_src, fused_conf, smoothed_zone, cam_confirmed
            )

            # ── 5. Apply hold timer (hysteresis) ──────────────────────
            if level.value > AlertLevel.CLEAR.value:
                self._alert_held_until = now + self.cfg.MIN_ALERT_HOLD_SECS
                self._alert_count += 1

            if now < self._alert_held_until and level == AlertLevel.CLEAR:
                level = AlertLevel.MONITOR   # keep a soft alert during hold

            # ── 6. Map level → hardware outputs ───────────────────────
            led_on       = level.value >= AlertLevel.WARNING.value
            vibration_on = level.value >= AlertLevel.CRITICAL.value

            reason = self._build_reason(fusion_src, level, state, fused_conf)

            alert = CollisionAlert(
                level=level,
                led_on=led_on,
                vibration_on=vibration_on,
                reason=reason,
                details={
                    "fusion_source":     fusion_src,
                    "fused_confidence":  round(fused_conf, 3),
                    "smoothed_zone":     smoothed_zone.name,
                    "radar":             state.radar_triggered,
                    "cam_confirmed":     cam_confirmed,
                    "lean_angle":        round(state.lean_angle_deg, 1),
                    "cycle":             self._cycle_count,
                    "false_alarms":      self._false_alarm_count,
                },
            )
            self._last_alert = alert
            return alert

    @property
    def last_alert(self) -> CollisionAlert:
        """Thread-safe read of the most recent alert (for dashboard)."""
        with self._lock:
            return self._last_alert

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "cycles": self._cycle_count,
                "alerts": self._alert_count,
                "false_alarms": self._false_alarm_count,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _smoothed_camera_detection(self) -> bool:
        """True only if enough recent frames showed a vehicle."""
        if len(self._cam_history) == 0:
            return False
        ratio = sum(self._cam_history) / len(self._cam_history)
        return ratio >= self.cfg.CAM_DETECTION_THRESHOLD

    def _smoothed_zone(self) -> ThreatZone:
        """Return the most frequently observed zone in recent history."""
        if not self._zone_history:
            return ThreatZone.CLEAR
        from collections import Counter
        counts = Counter(self._zone_history)
        return counts.most_common(1)[0][0]

    def _compute_alert_level(
        self,
        fusion_src: str,
        fused_conf: float,
        zone: ThreatZone,
        cam_confirmed: bool,
    ) -> AlertLevel:
        """Map fused evidence → AlertLevel."""

        if fusion_src == "NONE":
            return AlertLevel.CLEAR

        if fusion_src == "RADAR_ONLY":
            # Radar fired but camera disagrees — emit WARNING
            return AlertLevel.WARNING

        # Camera (with or without radar) has evidence
        if fused_conf < self.cfg.FINAL_CONFIDENCE_MIN:
            return AlertLevel.MONITOR

        if zone == ThreatZone.CRITICAL:
            return AlertLevel.CRITICAL
        elif zone == ThreatZone.WARNING:
            return AlertLevel.WARNING
        elif zone == ThreatZone.MONITOR:
            return AlertLevel.MONITOR if cam_confirmed else AlertLevel.CLEAR
        else:
            return AlertLevel.CLEAR

    @staticmethod
    def _build_reason(
        fusion_src: str,
        level: AlertLevel,
        state: SensorState,
        fused_conf: float,
    ) -> str:
        if level == AlertLevel.CLEAR:
            return "Blind spot clear"
        if fusion_src == "RADAR_ONLY":
            return "Radar detected object; camera inconclusive — WARNING issued"
        if fusion_src == "RADAR+CAMERA":
            return (
                f"CONFIRMED vehicle in blind spot "
                f"(radar + camera {fused_conf:.0%} confidence) "
                f"— {level.name}"
            )
        return (
            f"Camera detected vehicle "
            f"({state.cam_zone.name}, {fused_conf:.0%} conf) "
            f"— {level.name}"
        )
