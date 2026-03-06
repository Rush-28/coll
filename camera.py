"""
camera.py  (v2 — Multi-Camera)
================================
Provides:
  • Camera     — single threaded camera with auto-reconnect
  • MultiCameraManager — manages left / right / rear cameras by name
"""

import cv2
import threading
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera discovery
# ---------------------------------------------------------------------------

def find_cameras(limit: int = 10) -> list[int]:
    """Scan indices 0..limit-1 and return those with a working camera."""
    available = []
    for i in range(limit):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
                logger.info(f"Camera found at index {i}")
        cap.release()
    return available


# ---------------------------------------------------------------------------
# Single Camera (threaded)
# ---------------------------------------------------------------------------

class Camera:
    """
    Threaded camera wrapper.

    A daemon thread continuously reads frames so `get_latest_frame()`
    always returns the newest image without stalling the main loop.
    Auto-reconnects if the device disconnects.
    """

    def __init__(
        self,
        index:           int   = 0,
        width:           int   = 640,
        height:          int   = 480,
        fps:             int   = 30,
        reconnect_delay: float = 2.0,
        name:            str   = "",
    ):
        self.index           = index
        self.width           = width
        self.height          = height
        self.fps             = fps
        self.reconnect_delay = reconnect_delay
        self.name            = name or f"cam-{index}"

        self._cap:   Optional[cv2.VideoCapture] = None
        self._frame: Optional[cv2.typing.MatLike] = None
        self._fresh: bool = False
        self._ok:    bool = False
        self._lock   = threading.Lock()
        self._stop   = threading.Event()

        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name=self.name
        )
        self._thread.start()
        self._wait_for_first_frame(timeout=5.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_latest_frame(self) -> tuple[bool, Optional[cv2.typing.MatLike]]:
        """Return (ok, frame). Marks frame as consumed."""
        with self._lock:
            if self._frame is None:
                return False, None
            frame = self._frame.copy()
            self._fresh = False
        return True, frame

    def is_fresh(self) -> bool:
        with self._lock:
            return self._fresh

    @property
    def is_ok(self) -> bool:
        return self._ok

    def release(self):
        self._stop.set()
        self._thread.join(timeout=3.0)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        logger.info(f"{self.name}: released")

    # ------------------------------------------------------------------
    # Background capture loop
    # ------------------------------------------------------------------

    def _open_device(self) -> bool:
        if self._cap:
            self._cap.release()
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        self._cap = cap
        logger.info(f"{self.name}: opened ({self.width}x{self.height}@{self.fps}fps)")
        return True

    def _capture_loop(self):
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._ok = False
                if not self._open_device():
                    logger.warning(f"{self.name}: device unavailable, retry in {self.reconnect_delay}s")
                    time.sleep(self.reconnect_delay)
                    continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning(f"{self.name}: read failed, reconnecting…")
                self._ok = False
                self._cap.release()
                time.sleep(self.reconnect_delay)
                continue

            with self._lock:
                self._frame = frame
                self._fresh = True
                self._ok    = True

    def _wait_for_first_frame(self, timeout: float = 5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    return
            time.sleep(0.05)
        logger.warning(f"{self.name}: timed out waiting for first frame")


# ---------------------------------------------------------------------------
# Multi-Camera Manager
# ---------------------------------------------------------------------------

# Logical camera roles → default USB device indices
# Adjust these to match how your cameras appear on the Pi
DEFAULT_CAMERA_INDICES = {
    "left":  0,
    "right": 1,
    "rear":  2,
}

# Resolution for all cameras
STREAM_WIDTH  = 640
STREAM_HEIGHT = 480
STREAM_FPS    = 30


class MultiCameraManager:
    """
    Manages left, right, and rear cameras by logical name.

    Usage
    -----
    mgr = MultiCameraManager()
    ok, frame = mgr.get_frame("right")
    mgr.release_all()
    """

    def __init__(
        self,
        indices: dict[str, int] | None = None,
        width:  int   = STREAM_WIDTH,
        height: int   = STREAM_HEIGHT,
        fps:    int   = STREAM_FPS,
    ):
        cfg = indices or DEFAULT_CAMERA_INDICES
        self._cameras: dict[str, Camera] = {}
        for role, idx in cfg.items():
            self._cameras[role] = Camera(
                index=idx,
                width=width,
                height=height,
                fps=fps,
                name=f"cam-{role}",
            )
            logger.info(f"MultiCameraManager: {role} → index {idx}")

    def get_frame(self, role: str) -> tuple[bool, Optional[cv2.typing.MatLike]]:
        """Return (ok, frame) for the named camera role."""
        cam = self._cameras.get(role)
        if cam is None:
            logger.warning(f"Unknown camera role: {role}")
            return False, None
        return cam.get_latest_frame()

    def is_ok(self, role: str) -> bool:
        cam = self._cameras.get(role)
        return cam.is_ok if cam else False

    def status(self) -> dict[str, bool]:
        return {role: cam.is_ok for role, cam in self._cameras.items()}

    def roles(self) -> list[str]:
        return list(self._cameras.keys())

    def release_all(self):
        for cam in self._cameras.values():
            cam.release()
        logger.info("MultiCameraManager: all cameras released")


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print("Scanning for cameras…")
    cams = find_cameras()
    if not cams:
        print("No cameras detected.")
        sys.exit(1)
    print(f"Available indices: {cams}")
    mgr = MultiCameraManager(
        indices={role: idx for role, idx in zip(["left", "right", "rear"], cams)}
    )
    print("Press Q to quit")
    while True:
        for role in mgr.roles():
            ok, frame = mgr.get_frame(role)
            if ok:
                cv2.imshow(f"Camera · {role}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    mgr.release_all()
    cv2.destroyAllWindows()
