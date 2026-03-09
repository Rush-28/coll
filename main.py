"""
main.py  — BlindGuard Bike Blind-Spot System
============================================
  • Three live MJPEG camera streams: /stream/left  /stream/right  /stream/rear
  • Vehicle detection overlay drawn in real-time on each stream frame
  • Collision engine fuses right-camera detections + radar + IMU
  • Flask-SocketIO pushes live status to dashboard.html every cycle
  • GPIO: LED (WARNING+), Vibration motor (CRITICAL only)

Architecture
────────────
  MultiCameraManager (left / right / rear)
       │
       ├─ VehicleDetector (TFLite YOLOv8n, vehicle-only)
       │        │
       │        └──► CollisionEngine (fusion + threat scoring)
       │                      │
       │             GPIO LED + Vibration Motor
       │
  /stream/<role>  →  MJPEG generator (annotates frame, yields JPEG)
  SocketIO        →  status_update event every loop cycle
"""

import time
import threading
import logging
import io
import serial
import numpy as np
import cv2

# ── Hardware (Pi only) ────────────────────────────────────────────────────
try:
    from gpiozero import LED, OutputDevice
    GPIO_AVAILABLE = True
except (ImportError, Exception):
    GPIO_AVAILABLE = False

try:
    from mpu6050 import mpu6050 as MPU6050Sensor
    IMU_AVAILABLE = True
except (ImportError, Exception):
    IMU_AVAILABLE = False

# ── Web ───────────────────────────────────────────────────────────────────
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO

# ── Our modules ───────────────────────────────────────────────────────────
from camera import MultiCameraManager, CameraScanner, assign_roles
from vehicle_detector import VehicleDetector, ThreatZone, DetectionResult
from collision_engine import CollisionEngine, SensorState, AlertLevel, Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


# ==========================================================================
# CONFIGURATION
# ==========================================================================

class HWConfig:
    # GPIO pins (BCM)
    RIGHT_LED_PIN   = 22
    RIGHT_MOTOR_PIN = 25

    # mmWave radar UART
    RADAR_PORT      = "/dev/ttyAMA2"
    RADAR_BAUD      = 115200
    RADAR_TIMEOUT   = 0.1

    # Role assignment order — cameras are assigned left→right→rear
    # in the order they are discovered (ascending /dev/video* index).
    # Change this list to reorder which physical camera maps to which role.
    CAMERA_ROLE_ORDER = ["left", "right", "rear"]
    CAMERA_WIDTH      = 640
    CAMERA_HEIGHT     = 480
    CAMERA_FPS        = 30

    # TFLite model
    MODEL_PATH      = "yolov8n.tflite"

    # Main loop rate
    LOOP_INTERVAL   = 0.04      # 25 Hz

    # MJPEG stream quality (1–100)
    JPEG_QUALITY    = 80

    # Run ML on every N-th frame to save CPU
    DETECT_EVERY_N  = 1

    # IMU
    IMU_I2C_ADDR    = 0x68


# ==========================================================================
# FLASK + SOCKETIO
# ==========================================================================

app      = Flask(__name__, template_folder=".")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Shared annotated frames for MJPEG streaming (one per camera role)
_stream_frames: dict[str, bytes | None] = {"left": None, "right": None, "rear": None}
_stream_lock = threading.Lock()

# Live dashboard state
_dashboard_state: dict = {}
_dash_lock = threading.Lock()


# ==========================================================================
# ROUTES
# ==========================================================================

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/status")
def api_status():
    with _dash_lock:
        return jsonify(_dashboard_state)


def _mjpeg_generator(role: str):
    """Yield MJPEG boundary frames for the given camera role."""
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        with _stream_lock:
            jpeg = _stream_frames.get(role)

        if jpeg is None:
            # Send a blank dark frame while camera warms up
            blank = np.zeros((HWConfig.CAMERA_HEIGHT, HWConfig.CAMERA_WIDTH, 3), dtype=np.uint8)
            _draw_offline_overlay(blank, role)
            _, enc = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, HWConfig.JPEG_QUALITY])
            jpeg = enc.tobytes()

        yield boundary + jpeg + b"\r\n"
        time.sleep(0.033)  # ~30 fps cap for the HTTP stream


@app.route("/stream/left")
def stream_left():
    return Response(
        _mjpeg_generator("left"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stream/right")
def stream_right():
    return Response(
        _mjpeg_generator("right"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stream/rear")
def stream_rear():
    return Response(
        _mjpeg_generator("rear"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ==========================================================================
# FRAME ANNOTATION HELPERS
# ==========================================================================

# Colour palette per zone
_ZONE_COLOUR = {
    ThreatZone.CRITICAL: (0,   0,   255),   # red
    ThreatZone.WARNING:  (0,   140, 255),   # orange
    ThreatZone.MONITOR:  (0,   220, 60),    # green
    ThreatZone.CLEAR:    (80,  80,  80),
}

# Label colour per alert level (used for the HUD bar)
_ALERT_COLOUR = {
    AlertLevel.CRITICAL: (0,   0,   255),
    AlertLevel.WARNING:  (0,   120, 255),
    AlertLevel.MONITOR:  (0,   200, 60),
    AlertLevel.CLEAR:    (60,  60,  60),
}

_FONT    = cv2.FONT_HERSHEY_SIMPLEX
_MONO    = cv2.FONT_HERSHEY_PLAIN


def _draw_detections(frame: np.ndarray, result: DetectionResult | None):
    """Draw bounding boxes, labels, and confidence on the frame (in-place)."""
    if result is None or not result.vehicle_detected:
        return
    for det in result.detections:
        x1, y1, x2, y2 = det.box
        colour = _ZONE_COLOUR[det.zone]

        # Box outline — thicker for higher threat
        thickness = 3 if det.zone == ThreatZone.CRITICAL else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

        # Glow effect — second, slightly larger, semi-transparent rect
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1-2, y1-2), (x2+2, y2+2), colour, 1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Label background
        label = f"{det.class_name.upper()} {det.confidence:.0%}  [{det.zone.name}]"
        (tw, th), baseline = cv2.getTextSize(label, _FONT, 0.46, 1)
        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(frame,
                      (x1, label_y - th - baseline - 2),
                      (x1 + tw + 4, label_y + 2),
                      colour, cv2.FILLED)
        cv2.putText(frame, label,
                    (x1 + 2, label_y - baseline),
                    _FONT, 0.46, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_hud(
    frame:      np.ndarray,
    role:       str,
    alert:      AlertLevel,
    result:     DetectionResult | None,
    cam_index:  int,
    fps:        float,
):
    """Draw the HUD overlay: role label, timestamp, FPS, alert status bar."""
    h, w = frame.shape[:2]
    ts = time.strftime("%H:%M:%S")
    colour = _ALERT_COLOUR[alert]

    # ── Top-left: role badge ─────────────────────────────────────────
    role_label = role.upper()
    cv2.rectangle(frame, (0, 0), (len(role_label) * 10 + 16, 22), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, role_label, (8, 15), _FONT, 0.52, (0, 212, 255), 1, cv2.LINE_AA)

    # ── Top-right: CAM index + FPS ───────────────────────────────────
    top_right = f"CAM-{cam_index:02d}  {fps:.0f}fps"
    (tw, _), _ = cv2.getTextSize(top_right, _MONO, 1.0, 1)
    cv2.rectangle(frame, (w - tw - 14, 0), (w, 18), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, top_right, (w - tw - 8, 13), _MONO, 1.0, (100, 100, 100), 1, cv2.LINE_AA)

    # ── Bottom-left: timestamp ───────────────────────────────────────
    cv2.rectangle(frame, (0, h - 20), (200, h), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, ts, (6, h - 5), _MONO, 1.0, (0, 170, 200), 1, cv2.LINE_AA)

    # ── Bottom-right: REC dot ────────────────────────────────────────
    rec_col = (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (60, 60, 60)
    cv2.circle(frame, (w - 14, h - 10), 5, rec_col, cv2.FILLED)
    cv2.putText(frame, "REC", (w - 42, h - 5), _MONO, 1.0, rec_col, 1, cv2.LINE_AA)

    # ── Bottom alert bar (only when WARNING/CRITICAL) ────────────────
    if alert.value >= AlertLevel.WARNING.value:
        n_det = len(result.detections) if result else 0
        bar_text = f"  ⚠  {alert.name} — {n_det} VEHICLE(S) DETECTED"
        (bw, bh), _ = cv2.getTextSize(bar_text, _FONT, 0.5, 1)
        bar_y = h - 40
        cv2.rectangle(frame, (0, bar_y - bh - 6), (w, bar_y + 8), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(frame, (0, bar_y - bh - 6), (4, bar_y + 8), colour, cv2.FILLED)
        cv2.putText(frame, bar_text, (10, bar_y), _FONT, 0.5, colour, 1, cv2.LINE_AA)

    # ── Corner brackets ──────────────────────────────────────────────
    _draw_corner_brackets(frame, colour if alert.value >= AlertLevel.WARNING.value else (0, 100, 140))


def _draw_corner_brackets(frame: np.ndarray, colour: tuple, size: int = 18, t: int = 2):
    h, w = frame.shape[:2]
    corners = [(0, 0, 1, 1), (w, 0, -1, 1), (0, h, 1, -1), (w, h, -1, -1)]
    for cx, cy, sx, sy in corners:
        cv2.line(frame, (cx, cy), (cx + sx * size, cy), colour, t)
        cv2.line(frame, (cx, cy), (cx, cy + sy * size), colour, t)


def _draw_offline_overlay(frame: np.ndarray, role: str):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"{role.upper()} CAMERA", (w//2 - 80, h//2 - 14),
                _FONT, 0.7, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(frame, "NO SIGNAL", (w//2 - 60, h//2 + 14),
                _FONT, 0.6, (40, 40, 40), 1, cv2.LINE_AA)


def _encode_jpeg(frame: np.ndarray) -> bytes:
    _, enc = cv2.imencode(".jpg", frame,
                          [cv2.IMWRITE_JPEG_QUALITY, HWConfig.JPEG_QUALITY])
    return enc.tobytes()


# ==========================================================================
# HARDWARE HELPERS
# ==========================================================================

def _init_gpio():
    if GPIO_AVAILABLE:
        return LED(HWConfig.RIGHT_LED_PIN), OutputDevice(HWConfig.RIGHT_MOTOR_PIN)
    class _Stub:
        def on(self): pass
        def off(self): pass
    return _Stub(), _Stub()


def _init_imu():
    if IMU_AVAILABLE:
        try:
            return MPU6050Sensor(HWConfig.IMU_I2C_ADDR)
        except Exception as e:
            logger.warning(f"IMU init failed: {e}")
    return None


def _init_radar():
    try:
        port = serial.Serial(HWConfig.RADAR_PORT, HWConfig.RADAR_BAUD,
                             timeout=HWConfig.RADAR_TIMEOUT)
        logger.info(f"Radar on {HWConfig.RADAR_PORT}")
        return port
    except Exception as e:
        logger.warning(f"Radar unavailable ({e}) — camera-only mode")
        return None


def _read_lean(imu) -> tuple[float, bool]:
    if imu is None:
        return 0.0, False
    try:
        a = imu.get_accel_data()
        return float(np.degrees(np.arctan2(a["y"], a["z"]))), True
    except Exception:
        return 0.0, False


def _read_radar(port) -> tuple[bool, float | None]:
    if port is None:
        return False, None
    try:
        if port.in_waiting == 0:
            return False, None
        line = port.readline().decode("utf-8", errors="ignore").strip()
        parts = line.split()
        if "DETECT" in parts:
            dist = None
            try:
                dist = float(parts[parts.index("DETECT") + 1])
            except (IndexError, ValueError):
                pass
            return True, dist
        return False, None
    except Exception:
        return False, None


# ==========================================================================
# SENSOR LOOP (background thread)
# ==========================================================================

def sensor_loop():
    """
    25 Hz loop:
      1. Read all sensors
      2. Run ML on each camera frame
      3. Fuse with CollisionEngine (using right-camera + radar + IMU)
      4. Drive GPIO
      5. Update MJPEG stream frames for all three cameras
      6. Push SocketIO status event
    """
    logger.info("Sensor loop starting…")

    led, motor = _init_gpio()
    imu        = _init_imu()
    radar      = _init_radar()

    cam_mgr  = MultiCameraManager(
        # No `indices` arg → runs CameraScanner automatically at startup
        role_order=HWConfig.CAMERA_ROLE_ORDER,
        width=HWConfig.CAMERA_WIDTH,
        height=HWConfig.CAMERA_HEIGHT,
        fps=HWConfig.CAMERA_FPS,
    )

    # Log the discovered mapping so it appears in the Pi terminal
    discovered = cam_mgr.cam_map()
    if discovered:
        logger.info("Camera role assignment:")
        for role, idx in discovered.items():
            logger.info(f"  {role:>5s} → /dev/video{idx} (cv2 index {idx})")
    else:
        logger.warning("No cameras discovered — all streams will show offline placeholder")

    detector = VehicleDetector(model_path=HWConfig.MODEL_PATH)
    engine   = CollisionEngine(cfg=Config())

    frame_count = 0

    # Use discovered roles; fall back to all three role names so the MJPEG
    # generators still serve offline placeholders even with 0 cameras.
    _roles = cam_mgr.roles() or HWConfig.CAMERA_ROLE_ORDER

    # Per-camera state
    results:    dict[str, DetectionResult | None] = {r: None  for r in _roles}
    fps_counts: dict[str, int]                    = {r: 0     for r in _roles}
    fps_timers: dict[str, float]                  = {r: time.time() for r in _roles}
    fps_vals:   dict[str, float]                  = {r: 0.0   for r in _roles}

    # Camera index map for HUD — safe fallback to 0 if role not assigned
    cam_idx = discovered   # role → int index

    # Last published alert (to annotate all frames with coherent status)
    last_alert = AlertLevel.CLEAR

    try:
        while True:
            t0 = time.time()
            frame_count += 1

            # ── 1. IMU ───────────────────────────────────────────────
            lean_angle, imu_ok = _read_lean(imu)

            # ── 2. Radar ─────────────────────────────────────────────
            radar_triggered, radar_dist = _read_radar(radar)

            # ── 3. Camera + ML for ALL three cameras ─────────────────
            run_ml = (frame_count % HWConfig.DETECT_EVERY_N == 0)

            frames: dict[str, tuple[bool, object]] = {}
            for role in cam_mgr.roles():
                ok, frame = cam_mgr.get_frame(role)
                frames[role] = (ok, frame)

                if ok and run_ml:
                    results[role] = detector.detect(frame)
                elif not ok:
                    results[role] = None

                # FPS counter
                fps_counts[role] += 1
                now = time.time()
                if now - fps_timers[role] >= 1.0:
                    fps_vals[role]   = fps_counts[role]
                    fps_counts[role] = 0
                    fps_timers[role] = now

            # ── 4. Collision engine (uses right camera) ───────────────
            right_result = results.get("right")
            state = SensorState(
                lean_angle_deg       = lean_angle,
                imu_ok               = imu_ok,
                radar_triggered      = radar_triggered,
                radar_distance_m     = radar_dist,
                radar_ok             = radar is not None,
                cam_ok               = frames.get("right", (False, None))[0],
                vehicle_detected_cam = bool(right_result and right_result.vehicle_detected),
                cam_zone             = (right_result.highest_zone
                                        if right_result and right_result.vehicle_detected
                                        else ThreatZone.CLEAR),
                cam_confidence       = (right_result.max_confidence
                                        if right_result and right_result.vehicle_detected
                                        else 0.0),
            )
            alert = engine.update(state)
            last_alert = alert.level

            # ── 5. GPIO outputs ───────────────────────────────────────
            led.on() if alert.led_on else led.off()
            motor.on() if alert.vibration_on else motor.off()

            # ── 6. Annotate frames + push to MJPEG buffer ─────────────
            annotated_jpegs: dict[str, bytes] = {}
            for role in _roles:
                ok, raw_frame = frames.get(role, (False, None))

                if not ok or raw_frame is None:
                    blank = np.zeros((HWConfig.CAMERA_HEIGHT, HWConfig.CAMERA_WIDTH, 3),
                                     dtype=np.uint8)
                    _draw_offline_overlay(blank, role)
                    annotated_jpegs[role] = _encode_jpeg(blank)
                    continue

                frame = raw_frame.copy()
                _draw_detections(frame, results[role])
                _draw_hud(
                    frame,
                    role=role,
                    alert=last_alert if role == "right" else AlertLevel.CLEAR,
                    result=results[role],
                    cam_index=cam_idx.get(role, 0),
                    fps=fps_vals[role],
                )
                annotated_jpegs[role] = _encode_jpeg(frame)

            with _stream_lock:
                for role, jpeg in annotated_jpegs.items():
                    _stream_frames[role] = jpeg

            # ── 7. SocketIO status push ───────────────────────────────
            right_r = results.get("right")
            payload = {
                "level":        alert.level.name,
                "led":          alert.led_on,
                "vibration":    alert.vibration_on,
                "reason":       alert.reason,
                "lean_angle":   round(lean_angle, 1),
                "radar":        radar_triggered,
                "radar_dist_m": radar_dist,
                "cam_status":   cam_mgr.status(),
                "cam_map":  {role: f"/dev/video{idx}" for role, idx in discovered.items()},
                "cam_zone":     (right_r.highest_zone.name
                                 if right_r and right_r.vehicle_detected else "CLEAR"),
                "cam_conf":     round(right_r.max_confidence if right_r and right_r.vehicle_detected else 0.0, 3),
                "fps": {role: round(fps_vals[role], 1) for role in cam_mgr.roles()},
                "stats":        engine.stats,
                "timestamp":    t0,
                "detections": {
                    role: [
                        {
                            "class":      d.class_name,
                            "confidence": round(d.confidence, 3),
                            "zone":       d.zone.name,
                            "box":        d.box,
                        }
                        for d in (results[role].detections if results[role] else [])
                    ]
                    for role in cam_mgr.roles()
                },
            }

            with _dash_lock:
                _dashboard_state.update(payload)

            socketio.emit("status_update", payload)

            # Log significant events
            if alert.level.value >= AlertLevel.WARNING.value:
                logger.warning(f"[{alert.level.name}] {alert.reason}")

            # ── 8. Pace loop ──────────────────────────────────────────
            elapsed = time.time() - t0
            time.sleep(max(0.0, HWConfig.LOOP_INTERVAL - elapsed))

    except Exception as e:
        logger.exception(f"Sensor loop crashed: {e}")
    finally:
        led.off()
        motor.off()
        cam_mgr.release_all()
        if radar:
            radar.close()
        logger.info("Sensor loop shut down cleanly")


# ==========================================================================
# ENTRY POINT
# ==========================================================================

if __name__ == "__main__":
    logger.info("BlindGuard starting…")

    bg = threading.Thread(target=sensor_loop, daemon=True, name="sensor-loop")
    bg.start()

    logger.info("Dashboard  → http://0.0.0.0:5000")
    logger.info("Left  stream → http://0.0.0.0:5000/stream/left")
    logger.info("Right stream → http://0.0.0.0:5000/stream/right")
    logger.info("Rear  stream → http://0.0.0.0:5000/stream/rear")

    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)