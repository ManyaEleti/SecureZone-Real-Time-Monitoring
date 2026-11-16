"""
SecureZone Backend — backend.py
Real-Time Restricted Area Monitoring
--------------------------------------
Responsibilities:
  - Load YOLOv8-seg model (person class only)
  - Run inference on each camera frame
  - Check if any detected person overlaps a user-defined polygon zone
  - Apply persistence threshold (N consecutive frames before alert fires)
  - Send Telegram alert (photo + caption) with cooldown to avoid spam
  - Expose clean API: BackendEngine class consumed by frontend.py

Dependencies (inside yolo_gpu conda env):
  pip install ultralytics opencv-python-headless requests numpy
"""

import time
import threading
import requests
import numpy as np
import cv2
from ultralytics import YOLO

# ──────────────────────────────────────────────
# CONFIGURATION  — edit these before running
# ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"       # e.g. "7123456789:AAFxyz..."
TELEGRAM_CHAT_ID   = "YOUR_CHAT_ID_HERE"          # e.g. "123456789"

MODEL_PATH          = "/home/jetson_labs/Lecture_3/classwork_2_segmentation_inference/yolov8n-seg.pt"            # will auto-download if internet present;
                                                   # or place the .pt file in same folder
CAMERA_INDEX        = 0                            # 0 = first USB webcam
CONFIDENCE_THRESHOLD = 0.50                        # min detection confidence (0–1)
PERSISTENCE_FRAMES  = 10                           # consecutive frames before alert fires
ALERT_COOLDOWN_SEC  = 30                           # seconds between Telegram messages
PERSON_CLASS_ID     = 0                            # COCO class 0 = person


# ──────────────────────────────────────────────
# TELEGRAM HELPER
# ──────────────────────────────────────────────
class TelegramAlerter:
    """Sends photo + text alerts via Telegram Bot API."""

    def __init__(self, token: str, chat_id: str, cooldown: float):
        self.token    = token
        self.chat_id  = chat_id
        self.cooldown = cooldown
        self._last_sent = 0.0
        self._lock = threading.Lock()
        self._enabled = bool(token and token != "YOUR_BOT_TOKEN_HERE"
                             and chat_id and chat_id != "YOUR_CHAT_ID_HERE")

    def send(self, frame: np.ndarray, caption: str = "⚠️ INTRUSION DETECTED"):
        """Non-blocking: sends in a background thread, respects cooldown."""
        if not self._enabled:
            return
        with self._lock:
            now = time.time()
            if now - self._last_sent < self.cooldown:
                return
            self._last_sent = now
        t = threading.Thread(target=self._send_photo, args=(frame.copy(), caption), daemon=True)
        t.start()

    def _send_photo(self, frame: np.ndarray, caption: str):
        try:
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                return
            url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
            files   = {"photo": ("alert.jpg", buf.tobytes(), "image/jpeg")}
            payload = {"chat_id": self.chat_id, "caption": caption}
            resp = requests.post(url, data=payload, files=files, timeout=10)
            if resp.status_code != 200:
                print(f"[Telegram] Error {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[Telegram] Exception: {e}")


# ──────────────────────────────────────────────
# ZONE GEOMETRY HELPER
# ──────────────────────────────────────────────
def point_in_polygon(px: float, py: float, polygon: list) -> bool:
    """Ray-casting algorithm — returns True if (px, py) is inside polygon."""
    n = len(polygon)
    inside = False
    x, y = px, py
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def bbox_overlaps_zone(x1: int, y1: int, x2: int, y2: int, zone: list) -> bool:
    """
    Returns True if the bounding box bottom-center point is inside zone,
    OR if any corner of the bbox is inside zone.
    Bottom-center is the most reliable foot-position indicator for a standing person.
    """
    if len(zone) < 3:
        return False

    # Bottom-center foot point
    foot_x = (x1 + x2) / 2
    foot_y = float(y2)
    if point_in_polygon(foot_x, foot_y, zone):
        return True

    # All four corners
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return any(point_in_polygon(cx, cy, zone) for cx, cy in corners)


def mask_overlaps_zone(mask_points: np.ndarray, zone: list) -> bool:
    """
    Returns True if any point in the segmentation mask contour is inside zone.
    Falls back to False if mask is unavailable.
    """
    if mask_points is None or len(zone) < 3:
        return False
    for pt in mask_points:
        if point_in_polygon(float(pt[0]), float(pt[1]), zone):
            return True
    return False


# ──────────────────────────────────────────────
# DETECTION RESULT DATA CLASS
# ──────────────────────────────────────────────
class Detection:
    """Holds one detected person."""
    __slots__ = ("x1", "y1", "x2", "y2", "conf", "in_zone", "mask_pts")

    def __init__(self, x1, y1, x2, y2, conf, in_zone, mask_pts=None):
        self.x1       = int(x1)
        self.y1       = int(y1)
        self.x2       = int(x2)
        self.y2       = int(y2)
        self.conf     = float(conf)
        self.in_zone  = bool(in_zone)
        self.mask_pts = mask_pts   # np.ndarray of (N,2) contour points or None


# ──────────────────────────────────────────────
# MAIN BACKEND ENGINE
# ──────────────────────────────────────────────
class BackendEngine:
    """
    Public API consumed by frontend.py:

        engine = BackendEngine()
        engine.start()

        # each frame:
        result = engine.process_frame(frame, zone_polygon)
        # result.detections   — list[Detection]
        # result.alert_active — bool (True if intrusion confirmed)
        # result.alert_count  — int (total alerts fired this session)

        engine.stop()
    """

    def __init__(self):
        print("[Backend] Loading YOLOv8-seg model...")
        self.model      = YOLO(MODEL_PATH)
        self.alerter    = TelegramAlerter(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                                          ALERT_COOLDOWN_SEC)
        self._persist   = 0          # consecutive frames with intrusion
        self._alert_count = 0
        self._alert_active = False
        print("[Backend] Ready.")

    # ── public ──────────────────────────────
    def process_frame(self, frame: np.ndarray, zone: list) -> "FrameResult":
        """
        Run detection on frame, check zone overlap, update persistence counter.
        Returns FrameResult.
        """
        detections = self._detect(frame, zone)
        intruding  = any(d.in_zone for d in detections)

        if intruding:
            self._persist += 1
        else:
            self._persist = 0
            self._alert_active = False

        if self._persist >= PERSISTENCE_FRAMES and not self._alert_active:
            self._alert_active = True
            self._alert_count += 1
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            caption = f"⚠️ INTRUSION DETECTED\n🕐 {ts}\n📍 Restricted zone breached"
            self.alerter.send(frame, caption)

        return FrameResult(detections, self._alert_active, self._alert_count,
                           self._persist, intruding)

    def stop(self):
        pass  # placeholder for cleanup if needed

    # ── private ─────────────────────────────
    def _detect(self, frame: np.ndarray, zone: list) -> list:
        results = self.model(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )
        detections = []
        if not results or results[0].boxes is None:
            return detections

        r     = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()      # (N,4)
        confs = r.boxes.conf.cpu().numpy()      # (N,)
        masks = None

        # Extract segmentation mask contours if available
        if r.masks is not None:
            try:
                # masks.xy is list of (K,2) arrays, one per detection
                masks = r.masks.xy
            except Exception:
                masks = None

        for i, (box, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = box
            mask_pts = masks[i] if (masks is not None and i < len(masks)) else None

            # Prefer mask-based overlap; fall back to bbox
            if mask_pts is not None and len(mask_pts) > 0:
                in_zone = mask_overlaps_zone(mask_pts, zone)
            else:
                in_zone = bbox_overlaps_zone(x1, y1, x2, y2, zone)

            detections.append(Detection(x1, y1, x2, y2, conf, in_zone, mask_pts))

        return detections


# ──────────────────────────────────────────────
# FRAME RESULT
# ──────────────────────────────────────────────
class FrameResult:
    __slots__ = ("detections", "alert_active", "alert_count",
                 "persist_frames", "intruding_now")

    def __init__(self, detections, alert_active, alert_count,
                 persist_frames, intruding_now):
        self.detections     = detections
        self.alert_active   = alert_active
        self.alert_count    = alert_count
        self.persist_frames = persist_frames
        self.intruding_now  = intruding_now


# ──────────────────────────────────────────────
# STANDALONE TEST (run backend.py directly)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    engine = BackendEngine()
    cap    = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[Error] Cannot open camera index {CAMERA_INDEX}")
        sys.exit(1)

    # Hard-coded test zone (centre of frame) — in real use, frontend passes this
    ret, test_frame = cap.read()
    if ret:
        h, w = test_frame.shape[:2]
        test_zone = [
            (w * 0.3, h * 0.3),
            (w * 0.7, h * 0.3),
            (w * 0.7, h * 0.8),
            (w * 0.3, h * 0.8),
        ]
    else:
        test_zone = [(200, 150), (500, 150), (500, 400), (200, 400)]

    print("[Standalone test] Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = engine.process_frame(frame, test_zone)
        status = "ALERT" if result.alert_active else ("PERSON IN ZONE" if result.intruding_now else "CLEAR")
        cv2.putText(frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 255) if result.alert_active else (0, 255, 0), 2)
        cv2.imshow("Backend Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    engine.stop()
