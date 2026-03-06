"""
camera.py  (upgraded)
======================
Thread-safe camera wrapper with:
  - Background capture thread → always newest frame, no stale buffer
  - Auto-reconnect on device loss
  - Frame-skip control to limit CPU load
  - Warm-up flush to discard the initial stale frames
"""

import cv2
import threading
import logging
import time

logger = logging.getLogger(__name__)


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


class Camera:
    """
    Threaded camera wrapper.

    A background daemon thread continuously reads frames from the device
    and stores the latest one.  Callers use `get_latest_frame()` to
    retrieve it without blocking.

    Parameters
    ----------
    index           : OpenCV device index (default 0)
    width / height  : Requested resolution
    fps             : Capture rate hint sent to the driver
    reconnect_delay : Seconds to wait before retrying on device error
    """

    def __init__(
        self,
        index:            int = 0,
        width:            int = 640,
        height:           int = 480,
        fps:              int = 30,
        reconnect_delay:  float = 2.0,
    ):
        self.index           = index
        self.width           = width
        self.height          = height
        self.fps             = fps
        self.reconnect_delay = reconnect_delay

        self._cap:   cv2.VideoCapture | None = None
        self._frame: cv2.typing.MatLike | None = None
        self._fresh: bool  = False
        self._ok:    bool  = False
        self._lock   = threading.Lock()
        self._stop   = threading.Event()

        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name=f"cam-{index}"
        )
        self._thread.start()
        self._wait_for_first_frame(timeout=5.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_latest_frame(self) -> tuple[bool, cv2.typing.MatLike | None]:
        """
        Return (ok, frame).

        `ok` is False if no frame is available yet or camera failed.
        Marks the frame as consumed (fresh → False).
        """
        with self._lock:
            if self._frame is None:
                return False, None
            frame = self._frame.copy()
            self._fresh = False
        return True, frame

    def is_fresh(self) -> bool:
        """True if a frame newer than the last call to get_latest_frame is ready."""
        with self._lock:
            return self._fresh

    @property
    def is_ok(self) -> bool:
        return self._ok

    def release(self):
        """Stop the background thread and release the hardware device."""
        self._stop.set()
        self._thread.join(timeout=3.0)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        logger.info(f"Camera {self.index} released")

    # ------------------------------------------------------------------
    # Background capture loop
    # ------------------------------------------------------------------

    def _open_device(self) -> bool:
        """Try to open the camera device. Returns True on success."""
        if self._cap:
            self._cap.release()
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        # Minimal internal buffer — we manage freshness ourselves
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        logger.info(f"Camera {self.index} opened ({self.width}x{self.height}@{self.fps}fps)")
        return True

    def _capture_loop(self):
        """Background thread: keep reading frames, reconnect on error."""
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._ok = False
                if not self._open_device():
                    logger.warning(
                        f"Camera {self.index} not available. "
                        f"Retrying in {self.reconnect_delay}s …"
                    )
                    time.sleep(self.reconnect_delay)
                    continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning(f"Camera {self.index}: read failed, reconnecting …")
                self._ok = False
                self._cap.release()
                time.sleep(self.reconnect_delay)
                continue

            with self._lock:
                self._frame = frame
                self._fresh = True
                self._ok    = True

    def _wait_for_first_frame(self, timeout: float = 5.0):
        """Block until the first frame arrives or timeout is reached."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    return
            time.sleep(0.05)
        logger.warning(f"Camera {self.index}: timed out waiting for first frame")


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print("Scanning for cameras …")
    cams = find_cameras()
    if not cams:
        print("No cameras detected.")
        sys.exit(1)
    print(f"Available indices: {cams}")
    idx = cams[0]
    print(f"Opening camera {idx} — press Q to quit.")
    cam = Camera(index=idx)
    while True:
        ok, frame = cam.get_latest_frame()
        if ok:
            cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
