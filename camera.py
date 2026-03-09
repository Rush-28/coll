"""
camera.py  (v3 — Auto-Discovery)
==================================
Provides:
  • CameraScanner       — scans all V4L2/USB camera ports at startup
  • Camera              — single threaded camera with auto-reconnect
  • MultiCameraManager  — auto-assigns left / right / rear from scan results
"""

import cv2
import threading
import logging
import time
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ==========================================================================
# CameraScanner
# ==========================================================================

@dataclass
class CameraInfo:
    """Details discovered for a single camera port."""
    index:       int
    path:        str            # e.g. /dev/video0  OR  "cv2:0"
    width:       int = 0
    height:      int = 0
    fps:         float = 0.0
    backend:     str = ""
    ok:          bool = False
    note:        str = ""


class CameraScanner:
    """
    Scans every possible camera port at startup and returns an ordered list
    of working cameras with their actual capabilities.

    Strategy (Raspberry Pi + Linux):
      1. Enumerate /dev/video* device nodes (v4l2)
      2. Also try plain integer indices 0-15 (covers Windows + Pi fallback)
      3. De-duplicate (same physical device can appear under multiple indices)
      4. For each candidate: open → grab a test frame → record resolution/FPS
      5. Return sorted by index, working cameras first

    Thread-safe: all operations are synchronous (call from main thread only).
    """

    # Maximum integer index to probe when /dev/video* is not available
    MAX_INDEX = 15

    # Seconds to wait for a frame before declaring a port dead
    PROBE_TIMEOUT = 2.0

    # Number of test frames to grab (discards stale buffered frames)
    WARMUP_FRAMES = 3

    def __init__(self):
        self._results: list[CameraInfo] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, verbose: bool = True) -> list[CameraInfo]:
        """
        Run a full port scan.  Prints a summary table to stdout.
        Returns list of CameraInfo for working cameras only, sorted by index.
        """
        candidates = self._discover_candidates()

        if verbose:
            print("\n" + "═" * 58)
            print("  BlindGuard Camera Scanner — probing ports…")
            print("═" * 58)

        results: list[CameraInfo] = []
        seen_paths: set[str] = set()

        for idx, path in candidates:
            if path in seen_paths:
                continue
            seen_paths.add(path)

            info = self._probe(idx, path, verbose=verbose)
            results.append(info)

        working = [r for r in results if r.ok]
        failed  = [r for r in results if not r.ok]

        if verbose:
            print("─" * 58)
            print(f"  ✔  {len(working)} camera(s) ready  |  "
                  f"✘  {len(failed)} port(s) failed / empty")
            print("═" * 58 + "\n")

        self._results = working
        return working

    @property
    def results(self) -> list[CameraInfo]:
        """Last scan results (working cameras only)."""
        return self._results

    def working_indices(self) -> list[int]:
        return [c.index for c in self._results]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_candidates(self) -> list[tuple[int, str]]:
        """
        Build a list of (index, friendly_path) tuples to probe.
        On Linux/Pi: reads /dev/video* symlinks.
        On all platforms: also tries integer indices 0‥MAX_INDEX.
        """
        candidates: list[tuple[int, str]] = []

        # ── Linux / Raspberry Pi: /dev/video* ─────────────────────────
        if sys.platform.startswith("linux"):
            try:
                video_devs = sorted(
                    f for f in os.listdir("/dev")
                    if f.startswith("video")
                )
                for dev in video_devs:
                    try:
                        idx = int(dev.replace("video", ""))
                        candidates.append((idx, f"/dev/{dev}"))
                    except ValueError:
                        pass
            except (PermissionError, FileNotFoundError):
                pass

        # ── Fallback / Windows: plain integer indices ──────────────────
        existing_indices = {c[0] for c in candidates}
        for i in range(self.MAX_INDEX + 1):
            if i not in existing_indices:
                candidates.append((i, f"cv2:{i}"))

        # Sort by index
        candidates.sort(key=lambda x: x[0])
        return candidates

    def _probe(self, index: int, path: str, verbose: bool) -> CameraInfo:
        """Open one port, capture test frames, record capabilities."""
        info = CameraInfo(index=index, path=path)

        # Choose the right backend
        backend = cv2.CAP_V4L2 if path.startswith("/dev/") else cv2.CAP_ANY

        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            info.note = "could not open"
            if verbose:
                print(f"  [{index:>2}]  {path:<18}  ✘  {info.note}")
            return info

        # Record what the driver reports
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Grab & discard warmup frames (flush internal buffer)
        ok = False
        for _ in range(self.WARMUP_FRAMES):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                ok = True

        cap.release()

        if not ok:
            info.note = "opened but no frames"
            if verbose:
                print(f"  [{index:>2}]  {path:<18}  ✘  {info.note}")
            return info

        info.width   = w
        info.height  = h
        info.fps     = round(fps, 1) if fps > 0 else 0.0
        info.backend = "V4L2" if backend == cv2.CAP_V4L2 else "AUTO"
        info.ok      = True
        info.note    = f"{w}×{h} @ {info.fps}fps"

        if verbose:
            print(f"  [{index:>2}]  {path:<18}  ✔  {info.note}")

        return info


# ==========================================================================
# Role assigner
# ==========================================================================

def assign_roles(
    working: list[CameraInfo],
    role_order: list[str] | None = None,
) -> dict[str, int]:
    """
    Map camera roles to discovered device indices.

    Default role_order (leftmost = lowest index):
        ["left", "right", "rear"]

    If fewer cameras exist than roles, missing roles are omitted.
    If more cameras exist, extras are ignored (can be changed by passing
    a longer role_order list).
    """
    roles = role_order or ["left", "right", "rear"]
    mapping: dict[str, int] = {}
    for role, cam in zip(roles, working):
        mapping[role] = cam.index
    return mapping


# ==========================================================================
# Single Camera (threaded, auto-reconnect)
# ==========================================================================

class Camera:
    """
    Threaded camera wrapper.

    A daemon thread continuously reads frames so `get_latest_frame()`
    always returns the newest image instantly.
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
            frame       = self._frame.copy()
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
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        self._cap = cap
        logger.info(f"{self.name}: opened (index={self.index} "
                    f"{self.width}×{self.height}@{self.fps}fps)")
        return True

    def _capture_loop(self):
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._ok = False
                if not self._open_device():
                    logger.warning(f"{self.name}: device unavailable, "
                                   f"retry in {self.reconnect_delay}s")
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


# ==========================================================================
# Multi-Camera Manager  (auto-discovering)
# ==========================================================================

# Resolution for all streams
STREAM_WIDTH  = 640
STREAM_HEIGHT = 480
STREAM_FPS    = 30


class MultiCameraManager:
    """
    Manages left, right, and rear cameras by logical name.

    When `indices` is None (default) it runs `CameraScanner` automatically
    to discover which USB camera indices are live, then assigns them to roles
    in ascending order: left → right → rear.

    Pass an explicit `indices` dict to override auto-detection.
    """

    def __init__(
        self,
        indices: dict[str, int] | None = None,
        role_order: list[str] | None   = None,
        width:  int   = STREAM_WIDTH,
        height: int   = STREAM_HEIGHT,
        fps:    int   = STREAM_FPS,
    ):
        self._width  = width
        self._height = height
        self._fps    = fps
        self._cameras: dict[str, Camera] = {}

        if indices:
            # Manual override
            cam_map = indices
            logger.info("MultiCameraManager: using manual index map")
        else:
            # Auto-discover
            scanner = CameraScanner()
            working = scanner.scan(verbose=True)
            cam_map = assign_roles(working, role_order)

        if not cam_map:
            logger.error("MultiCameraManager: NO cameras found — streams will be offline")

        self._cam_map = cam_map   # role → index, for reference

        for role, idx in cam_map.items():
            self._cameras[role] = Camera(
                index=idx,
                width=width,
                height=height,
                fps=fps,
                name=f"cam-{role}",
            )
            logger.info(f"  {role:>5s} → /dev/video{idx}  (index {idx})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self, role: str) -> tuple[bool, Optional[cv2.typing.MatLike]]:
        cam = self._cameras.get(role)
        if cam is None:
            return False, None
        return cam.get_latest_frame()

    def is_ok(self, role: str) -> bool:
        cam = self._cameras.get(role)
        return cam.is_ok if cam else False

    def status(self) -> dict[str, bool]:
        return {role: cam.is_ok for role, cam in self._cameras.items()}

    def roles(self) -> list[str]:
        return list(self._cameras.keys())

    def cam_map(self) -> dict[str, int]:
        """Returns the role → device index mapping that was used."""
        return dict(self._cam_map)

    def release_all(self):
        for cam in self._cameras.values():
            cam.release()
        logger.info("MultiCameraManager: all cameras released")


# ==========================================================================
# CLI — run as standalone script to scan cameras
# ==========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    scanner = CameraScanner()
    working = scanner.scan(verbose=True)

    if not working:
        print("No working cameras found.")
        sys.exit(1)

    print("Assigned roles:")
    mapping = assign_roles(working)
    for role, idx in mapping.items():
        print(f"  {role:>5s} → index {idx}")

    print("\nOpening streams — press Q to quit")
    mgr = MultiCameraManager(indices=mapping)
    while True:
        for role in mgr.roles():
            ok, frame = mgr.get_frame(role)
            if ok:
                cv2.imshow(f"{role.upper()} (index {mgr.cam_map()[role]})", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    mgr.release_all()
    cv2.destroyAllWindows()
