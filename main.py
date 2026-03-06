"""
main.py  — BlindGuard Bike Blind-Spot Collision Detection System
================================================================
Orchestrates:
  • Camera  (camera.py)
  • Vehicle detector / ML  (vehicle_detector.py)
  • Collision engine / fusion  (collision_engine.py)
  • Hardware outputs  (GPIO LED + vibration motor)
  • MPU6050 IMU  (lean-angle guard)
  • mmWave radar  (UART)
  • Flask + SocketIO dashboard  (serves dashboard.html + real-time events)

Architecture
------------
  ┌─────────────┐   ┌──────────────────┐   ┌──────────────────┐
  │  MPU6050    │   │   mmWave Radar   │   │   USB Camera     │
  │  (I²C)      │   │   (UART serial)  │   │   (threaded)     │
  └──────┬──────┘   └────────┬─────────┘   └────────┬─────────┘
         │                   │                       │
         │                   │              ┌────────▼──────────┐
         │                   │              │ VehicleDetector   │
         │                   │              │ (TFLite YOLOv8n)  │
         │                   │              └────────┬──────────┘
         │                   │                       │
         └───────────────────┴───────────┐           │
                                         ▼           ▼
                                   ┌─────────────────────┐
                                   │   CollisionEngine    │
                                   │   (fusion + alerts)  │
                                   └──────────┬──────────┘
                                              │
                       ┌──────────────────────┼──────────────────────┐
                       ▼                      ▼                      ▼
                  GPIO LED            Vibration Motor        SocketIO push
                (WARNING+)           (CRITICAL only)       (dashboard.html)
"""

import time
import logging
import threading
import serial
import numpy as np

# ── Hardware imports (Raspberry Pi only) ──────────────────────────────────
# Wrapped in try/except so the code can be developed/tested on a PC too.
try:
    from gpiozero import LED, OutputDevice
    GPIO_AVAILABLE = True
except (ImportError, Exception):
    GPIO_AVAILABLE = False
    logging.warning("gpiozero not available — GPIO outputs will be simulated")

try:
    from mpu6050 import mpu6050 as MPU6050Sensor
    IMU_AVAILABLE = True
except (ImportError, Exception):
    IMU_AVAILABLE = False
    logging.warning("mpu6050 library not available — lean angle will be 0")

# ── Flask / SocketIO dashboard ─────────────────────────────────────────────
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

# ── Our own modules ───────────────────────────────────────────────────────
from camera import Camera
from vehicle_detector import VehicleDetector, ThreatZone
from collision_engine import CollisionEngine, SensorState, AlertLevel, Config

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


# ==========================================================================
# CONFIGURATION
# ==========================================================================

class HWConfig:
    # GPIO pin numbers (BCM numbering)
    RIGHT_LED_PIN     = 22
    RIGHT_MOTOR_PIN   = 25

    # Serial port for mmWave radar — adjust for your Pi UART assignment
    RADAR_PORT        = "/dev/ttyAMA2"
    RADAR_BAUD        = 115200
    RADAR_TIMEOUT     = 0.1          # seconds

    # Camera
    CAMERA_INDEX      = 0
    CAMERA_WIDTH      = 640
    CAMERA_HEIGHT     = 480
    CAMERA_FPS        = 30

    # TFLite model file (place in same directory)
    MODEL_PATH        = "yolov8n.tflite"

    # How often (seconds) the main loop runs
    LOOP_INTERVAL     = 0.04         # ~25 Hz

    # IMU I²C address
    IMU_I2C_ADDR      = 0x68


# ==========================================================================
# HARDWARE INITIALISATION
# ==========================================================================

def init_gpio() -> tuple:
    """Return (led, motor) — real GPIO or safe stubs."""
    if GPIO_AVAILABLE:
        led   = LED(HWConfig.RIGHT_LED_PIN)
        motor = OutputDevice(HWConfig.RIGHT_MOTOR_PIN)
        logger.info("GPIO: LED and motor initialised")
        return led, motor
    # Stub objects for development / testing
    class _Stub:
        def on(self):  pass
        def off(self): pass
    return _Stub(), _Stub()


def init_imu():
    """Return MPU6050 sensor or None."""
    if IMU_AVAILABLE:
        try:
            sensor = MPU6050Sensor(HWConfig.IMU_I2C_ADDR)
            logger.info("IMU: MPU6050 initialised")
            return sensor
        except Exception as e:
            logger.error(f"IMU init failed: {e}")
    return None


def init_radar() -> serial.Serial | None:
    """Return open Serial port for radar or None."""
    try:
        port = serial.Serial(
            HWConfig.RADAR_PORT,
            baudrate=HWConfig.RADAR_BAUD,
            timeout=HWConfig.RADAR_TIMEOUT,
        )
        logger.info(f"Radar: connected on {HWConfig.RADAR_PORT}")
        return port
    except Exception as e:
        logger.warning(f"Radar not available ({e}) — will use camera-only mode")
        return None


# ==========================================================================
# FLASK / SOCKETIO SETUP
# ==========================================================================

app     = Flask(__name__, template_folder=".")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Shared state pushed to the dashboard
_dashboard_state: dict = {}
_dashboard_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/status")
def api_status():
    with _dashboard_lock:
        return jsonify(_dashboard_state)


# ==========================================================================
# LEAN ANGLE HELPER
# ==========================================================================

def read_lean_angle(imu) -> tuple[float, bool]:
    """
    Returns (roll_degrees, imu_ok).
    Roll is the left-right tilt of the bike.
    """
    if imu is None:
        return 0.0, False
    try:
        accel = imu.get_accel_data()
        roll  = float(np.degrees(np.arctan2(accel['y'], accel['z'])))
        return roll, True
    except Exception as e:
        logger.debug(f"IMU read error: {e}")
        return 0.0, False


# ==========================================================================
# RADAR HELPER
# ==========================================================================

def read_radar(port: serial.Serial | None) -> tuple[bool, float | None]:
    """
    Returns (triggered, distance_m | None).

    Supports two common mmWave message formats:
      1. "DETECT <distance_m>"  — e.g. "DETECT 1.23"
      2. Plain "DETECT"         — no distance data
    Extend this function to match your sensor's actual protocol.
    """
    if port is None:
        return False, None
    try:
        if port.in_waiting == 0:
            return False, None
        line = port.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            return False, None

        parts = line.split()
        if "DETECT" in parts:
            triggered = True
            dist = None
            try:
                dist = float(parts[parts.index("DETECT") + 1])
            except (IndexError, ValueError):
                pass
            return triggered, dist

        return False, None
    except Exception as e:
        logger.debug(f"Radar read error: {e}")
        return False, None


# ==========================================================================
# MAIN SENSOR LOOP (runs in background thread)
# ==========================================================================

def sensor_loop():
    """
    Background thread:
    1. Read all sensors → build SensorState
    2. Run ML inference on camera frame
    3. Fuse with CollisionEngine → CollisionAlert
    4. Drive GPIO outputs
    5. Push update to dashboard via SocketIO
    """
    logger.info("Sensor loop starting …")

    # ── Hardware setup ──────────────────────────────────────────────
    led, motor = init_gpio()
    imu        = init_imu()
    radar      = init_radar()
    camera     = Camera(
        index=HWConfig.CAMERA_INDEX,
        width=HWConfig.CAMERA_WIDTH,
        height=HWConfig.CAMERA_HEIGHT,
        fps=HWConfig.CAMERA_FPS,
    )
    detector = VehicleDetector(model_path=HWConfig.MODEL_PATH)
    engine   = CollisionEngine(cfg=Config())

    frame_count = 0

    try:
        while True:
            t_start = time.time()
            frame_count += 1

            # ── 1. IMU ───────────────────────────────────────────────
            lean_angle, imu_ok = read_lean_angle(imu)

            # ── 2. Radar ─────────────────────────────────────────────
            radar_triggered, radar_dist = read_radar(radar)

            # ── 3. Camera + ML ───────────────────────────────────────
            cam_ok_flag        = camera.is_ok
            vehicle_detected   = False
            cam_zone           = ThreatZone.CLEAR
            cam_conf           = 0.0
            detection_result   = None

            if cam_ok_flag:
                ret, frame = camera.get_latest_frame()
                if ret and frame is not None:
                    # Run ML on every frame (or every N frames — see Config)
                    if frame_count % Config.INFERENCE_EVERY_N_FRAMES == 0:
                        detection_result = detector.detect(frame)
                        if detection_result.vehicle_detected:
                            vehicle_detected = True
                            cam_zone         = detection_result.highest_zone
                            cam_conf         = detection_result.max_confidence

            # ── 4. Build SensorState & fuse ──────────────────────────
            state = SensorState(
                lean_angle_deg       = lean_angle,
                imu_ok               = imu_ok,
                radar_triggered      = radar_triggered,
                radar_distance_m     = radar_dist,
                radar_ok             = radar is not None,
                cam_ok               = cam_ok_flag,
                vehicle_detected_cam = vehicle_detected,
                cam_zone             = cam_zone,
                cam_confidence       = cam_conf,
            )

            alert = engine.update(state)

            # ── 5. Drive GPIO ─────────────────────────────────────────
            if alert.led_on:
                led.on()
            else:
                led.off()

            if alert.vibration_on:
                motor.on()
            else:
                motor.off()

            # ── 6. Log significant events ─────────────────────────────
            if alert.level.value >= AlertLevel.WARNING.value:
                logger.warning(
                    f"[{alert.level.name}] {alert.reason} | "
                    f"details={alert.details}"
                )

            # ── 7. Push to dashboard ──────────────────────────────────
            payload = {
                "level":        alert.level.name,
                "led":          alert.led_on,
                "vibration":    alert.vibration_on,
                "reason":       alert.reason,
                "lean_angle":   lean_angle,
                "radar":        radar_triggered,
                "radar_dist_m": radar_dist,
                "cam_ok":       cam_ok_flag,
                "cam_zone":     cam_zone.name,
                "cam_conf":     round(cam_conf, 3),
                "stats":        engine.stats,
                "timestamp":    t_start,
                # Pretty detection list for the overlay panel
                "detections": [
                    {
                        "class":      d.class_name,
                        "confidence": round(d.confidence, 3),
                        "zone":       d.zone.name,
                        "box":        d.box,
                    }
                    for d in (detection_result.detections if detection_result else [])
                ],
            }

            with _dashboard_lock:
                _dashboard_state.update(payload)

            socketio.emit("status_update", payload)

            # ── 8. Pace the loop ──────────────────────────────────────
            elapsed = time.time() - t_start
            sleep   = max(0.0, HWConfig.LOOP_INTERVAL - elapsed)
            time.sleep(sleep)

    except Exception as e:
        logger.exception(f"Sensor loop crashed: {e}")
    finally:
        led.off()
        motor.off()
        camera.release()
        if radar:
            radar.close()
        logger.info("Sensor loop shut down cleanly")


# ==========================================================================
# ENTRY POINT
# ==========================================================================

if __name__ == "__main__":
    logger.info("BlindGuard starting …")

    # Start the sensor + fusion background thread
    bg = threading.Thread(target=sensor_loop, daemon=True, name="sensor-loop")
    bg.start()

    # Run the Flask-SocketIO server
    logger.info("Dashboard available at http://0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)