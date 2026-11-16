"""
SecureZone Frontend — frontend.py
Real-Time Restricted Area Monitoring
--------------------------------------
Streams live video + zone overlay to your browser via MJPEG.
No display / X11 / XQuartz required.

Open in browser:  http://localhost:8891
  (or http://192.168.3.2:8891 from Mac if tunnel not set up)

Controls (via browser buttons):
  - Click on video to add zone polygon points
  - "Close Zone" button — finalises the polygon (need ≥3 points)
  - "Reset Zone" button — clears the polygon
  - "Save Screenshot" button — saves frame to disk

Dependencies (yolo_gpu env):
  pip install flask   (everything else already installed)
"""

import time
import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify

try:
    from backend import BackendEngine, CAMERA_INDEX, PERSISTENCE_FRAMES
except ImportError as e:
    print(f"[Error] Could not import backend.py: {e}")
    print("Make sure backend.py is in the same folder as frontend.py")
    import sys; sys.exit(1)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
WEB_PORT      = 8891
JPEG_QUALITY  = 80      # lower = faster stream on slow USB link
FPS_SMOOTH    = 0.9

# ──────────────────────────────────────────────
# COLORS (BGR)
# ──────────────────────────────────────────────
COLOR_ZONE_DRAWING = (255, 200, 0)
COLOR_ZONE_ACTIVE  = (0, 200, 255)
COLOR_ZONE_ALERT   = (0, 0, 255)
COLOR_BOX_SAFE     = (0, 220, 0)
COLOR_BOX_DANGER   = (0, 0, 255)
COLOR_MASK_SAFE    = (0, 255, 0)
COLOR_MASK_DANGER  = (0, 0, 255)

# ──────────────────────────────────────────────
# SHARED STATE  (thread-safe with a lock)
# ──────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.lock         = threading.Lock()
        self.zone_points  = []      # list of [x, y]
        self.zone_closed  = False
        self.alert_active = False
        self.alert_count  = 0
        self.persist      = 0
        self.fps          = 0.0
        self.alert_log    = []      # list of str
        self.latest_frame = None    # JPEG bytes for MJPEG stream
        self.screenshot_req = False

    def add_point(self, x, y):
        with self.lock:
            if not self.zone_closed:
                self.zone_points.append([x, y])

    def close_zone(self):
        with self.lock:
            if len(self.zone_points) >= 3:
                self.zone_closed = True
                return True
            return False

    def reset_zone(self):
        with self.lock:
            self.zone_points = []
            self.zone_closed = False
            self.alert_active = False
            self.persist = 0

    def get_polygon(self):
        with self.lock:
            if self.zone_closed and len(self.zone_points) >= 3:
                return [tuple(p) for p in self.zone_points]
            return []

    def log(self, msg):
        with self.lock:
            ts = time.strftime("%H:%M:%S")
            self.alert_log.append(f"{ts}  {msg}")
            if len(self.alert_log) > 50:
                self.alert_log.pop(0)


state = AppState()

# ──────────────────────────────────────────────
# DRAWING HELPERS
# ──────────────────────────────────────────────
def draw_zone(canvas, points, closed, alert):
    if not points:
        return
    color = COLOR_ZONE_ALERT if alert else (COLOR_ZONE_ACTIVE if closed else COLOR_ZONE_DRAWING)
    pts = [tuple(p) for p in points]
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i+1], color, 2)
    if closed and len(pts) >= 3:
        cv2.line(canvas, pts[-1], pts[0], color, 2)
        poly = np.array(pts, dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
    for p in pts:
        cv2.circle(canvas, p, 5, color, -1)


def draw_detections(canvas, detections):
    for d in detections:
        box_color  = COLOR_BOX_DANGER  if d.in_zone else COLOR_BOX_SAFE
        mask_color = COLOR_MASK_DANGER if d.in_zone else COLOR_MASK_SAFE
        if d.mask_pts is not None and len(d.mask_pts) > 0:
            pts_int = d.mask_pts.astype(np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts_int], mask_color)
            cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0, canvas)
            cv2.polylines(canvas, [pts_int], True, mask_color, 1)
        cv2.rectangle(canvas, (d.x1, d.y1), (d.x2, d.y2), box_color, 2)
        label = ("VIOLATION  " if d.in_zone else "") + f"Person {d.conf:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(canvas, (d.x1, d.y1 - lh - 6), (d.x1 + lw + 4, d.y1), box_color, -1)
        cv2.putText(canvas, label, (d.x1 + 2, d.y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def draw_alert_banner(canvas):
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w-1, h-1), COLOR_ZONE_ALERT, 10)
    text = "!! INTRUSION ALERT !!"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
    tx = (w - tw) // 2
    cv2.rectangle(canvas, (tx-10, 8), (tx+tw+10, th+20), (0, 0, 180), -1)
    cv2.putText(canvas, text, (tx, th+12),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)


def draw_hud(canvas, fps, alert, persist, alert_count):
    h, w = canvas.shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"Alerts: {alert_count}",
        f"Persist: {persist}/{PERSISTENCE_FRAMES}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (8, h - 10 - (len(lines)-1-i)*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)


# ──────────────────────────────────────────────
# CAMERA + INFERENCE THREAD
# ──────────────────────────────────────────────
def camera_loop():
    engine = BackendEngine()
    cap    = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[Error] Cannot open camera {CAMERA_INDEX}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps    = 0.0
    t_prev = time.time()
    prev_alert = False

    print(f"[Frontend] Camera open. Browse to http://0.0.0.0:{WEB_PORT}")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        t_now = time.time()
        inst  = 1.0 / max(t_now - t_prev, 1e-6)
        fps   = FPS_SMOOTH * fps + (1 - FPS_SMOOTH) * inst
        t_prev = t_now

        zone_poly = state.get_polygon()
        result    = engine.process_frame(frame, zone_poly)

        with state.lock:
            state.alert_active = result.alert_active
            state.alert_count  = result.alert_count
            state.persist      = result.persist_frames
            state.fps          = fps

        if result.alert_active and not prev_alert:
            state.log("ALERT — intrusion confirmed")
        prev_alert = result.alert_active

        # Draw
        canvas = frame.copy()
        with state.lock:
            pts    = list(state.zone_points)
            closed = state.zone_closed
            alert  = state.alert_active
            ac     = state.alert_count
            pf     = state.persist

        draw_zone(canvas, pts, closed, alert)
        if zone_poly:
            draw_detections(canvas, result.detections)
        if alert:
            draw_alert_banner(canvas)
        draw_hud(canvas, fps, alert, pf, ac)

        # Screenshot
        with state.lock:
            do_shot = state.screenshot_req
            if do_shot:
                state.screenshot_req = False
        if do_shot:
            fname = f"securezone_{int(time.time())}.jpg"
            cv2.imwrite(fname, canvas)
            state.log(f"Screenshot: {fname}")
            print(f"[Frontend] Screenshot saved: {fname}")

        # Encode JPEG
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with state.lock:
                state.latest_frame = buf.tobytes()

    cap.release()


# ──────────────────────────────────────────────
# FLASK APP
# ──────────────────────────────────────────────
app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SecureZone</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family:sans-serif; display:flex; flex-direction:column; align-items:center; }
  h2   { margin:10px 0 4px; color:#f0a500; }
  #wrap { position:relative; display:inline-block; cursor:crosshair; }
  #feed { display:block; max-width:640px; width:100%; border:2px solid #333; }
  #overlay { position:absolute; top:0; left:0; width:100%; height:100%; }
  .controls { margin:8px 0; display:flex; gap:8px; flex-wrap:wrap; justify-content:center; }
  button { padding:8px 18px; border:none; border-radius:4px; cursor:pointer; font-size:14px; font-weight:bold; }
  .btn-close  { background:#0a9; color:#fff; }
  .btn-reset  { background:#c33; color:#fff; }
  .btn-shot   { background:#36a; color:#fff; }
  #status { font-size:15px; margin:4px 0; }
  #log { width:640px; max-width:98vw; height:120px; overflow-y:auto; background:#1a1a1a;
         border:1px solid #333; padding:6px; font-size:12px; font-family:monospace; }
  .alert-on  { color:#f44; font-weight:bold; }
  .alert-off { color:#4c4; }
  p.hint { font-size:12px; color:#888; margin:2px 0; }
</style>
</head>
<body>
<h2>SecureZone — Real-Time Monitoring</h2>
<p class="hint">Click on the video to add zone points &nbsp;|&nbsp; Press <b>Close Zone</b> when done</p>
<div id="wrap">
  <img id="feed" src="/video_feed" alt="live feed">
</div>
<div class="controls">
  <button class="btn-close" onclick="closeZone()">Close Zone</button>
  <button class="btn-reset" onclick="resetZone()">Reset Zone</button>
  <button class="btn-shot"  onclick="saveShot()">Save Screenshot</button>
</div>
<div id="status" class="alert-off">Status: loading...</div>
<div id="log"></div>

<script>
const feed = document.getElementById('feed');

feed.addEventListener('click', function(e) {
  const rect = feed.getBoundingClientRect();
  const scaleX = feed.naturalWidth  / rect.width;
  const scaleY = feed.naturalHeight / rect.height;
  const x = Math.round((e.clientX - rect.left) * scaleX);
  const y = Math.round((e.clientY - rect.top)  * scaleY);
  fetch('/add_point', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({x, y})
  });
});

function closeZone() {
  fetch('/close_zone', {method:'POST'}).then(r=>r.json()).then(d=>{
    if(!d.ok) alert('Need at least 3 points first!');
  });
}
function resetZone() { fetch('/reset_zone', {method:'POST'}); }
function saveShot()  { fetch('/screenshot',  {method:'POST'}); }

// Poll status every 500ms
setInterval(() => {
  fetch('/status').then(r=>r.json()).then(d=>{
    const el = document.getElementById('status');
    if(d.alert) {
      el.className = 'alert-on';
      el.textContent = '⚠️  INTRUSION ALERT  |  Alerts: ' + d.alert_count + '  |  FPS: ' + d.fps.toFixed(1);
    } else if(d.intruding) {
      el.className = 'alert-on';
      el.textContent = '🔶 Person in zone (' + d.persist + '/' + d.threshold + ' frames)  |  FPS: ' + d.fps.toFixed(1);
    } else {
      el.className = 'alert-off';
      el.textContent = '✅ CLEAR  |  Alerts: ' + d.alert_count + '  |  FPS: ' + d.fps.toFixed(1);
    }
    const log = document.getElementById('log');
    log.innerHTML = d.log.slice().reverse().map(l =>
      '<div style="color:' + (l.includes('ALERT') ? '#f66' : '#aaa') + '">' + l + '</div>'
    ).join('');
  });
}, 500);
</script>
</body>
</html>
"""

def gen_frames():
    while True:
        with state.lock:
            frame = state.latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)   # ~30fps cap on stream

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_point', methods=['POST'])
def add_point():
    data = request.get_json()
    state.add_point(int(data['x']), int(data['y']))
    return jsonify(ok=True)

@app.route('/close_zone', methods=['POST'])
def close_zone():
    ok = state.close_zone()
    if ok:
        state.log("Zone closed by user")
    return jsonify(ok=ok)

@app.route('/reset_zone', methods=['POST'])
def reset_zone():
    state.reset_zone()
    state.log("Zone reset")
    return jsonify(ok=True)

@app.route('/screenshot', methods=['POST'])
def screenshot():
    with state.lock:
        state.screenshot_req = True
    return jsonify(ok=True)

@app.route('/status')
def status():
    with state.lock:
        return jsonify(
            alert       = state.alert_active,
            alert_count = state.alert_count,
            persist     = state.persist,
            threshold   = PERSISTENCE_FRAMES,
            intruding   = state.persist > 0,
            fps         = round(state.fps, 1),
            log         = list(state.alert_log[-20:])
        )

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == '__main__':
    print("[Frontend] Starting backend + camera thread...")
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    # Give camera a moment to init before Flask starts
    time.sleep(2)

    print(f"[Frontend] Web server on http://0.0.0.0:{WEB_PORT}")
    app.run(host='0.0.0.0', port=WEB_PORT, threaded=True)
