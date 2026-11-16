cat > /home/jetson_labs/project1/README.md << 'EOF'
# SecureZone — Real-Time Restricted Area Monitoring

> **Course Project · Computer Vision · NVIDIA Jetson Orin**  
> Proposal Score: 100/100

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-seg-orange)
![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Orin-green?logo=nvidia)
![Framework](https://img.shields.io/badge/UI-Flask%20MJPEG-lightgrey)
![Alerts](https://img.shields.io/badge/Alerts-Telegram%20Bot-blue?logo=telegram)

---

## Overview

**SecureZone** is a real-time restricted area monitoring system that uses **YOLOv8 instance segmentation** to detect when a person enters a user-defined polygon zone. When an intrusion is confirmed across a persistence threshold, the system fires a **Telegram push alert** with a photo snapshot.

The entire pipeline runs on an **NVIDIA Jetson Orin** developer board inside a Docker container, with a **browser-based UI** (no display required) served over Flask MJPEG streaming.

---

## Key Features

| Feature | Detail |
|---|---|
| **Instance Segmentation** | YOLOv8n-seg — person class only (COCO class 0) |
| **Zone Definition** | Click-to-draw polygon in the browser — any shape, any size |
| **Overlap Detection** | Mask-based ray-casting (falls back to bbox foot-point) |
| **Persistence Threshold** | Alert fires only after N consecutive frames (default: 10) |
| **Telegram Alerts** | Photo snapshot + timestamp sent via Bot API with 30s cooldown |
| **Browser UI** | Flask MJPEG stream — works over SSH tunnel, no X11/display needed |
| **Edge Deployment** | Runs fully on Jetson Orin via Docker + conda (yolo_gpu env) |

---

## System Architecture

┌─────────────────────────────────────────────────────────┐
│ NVIDIA Jetson Orin │
│ │
│ ┌──────────┐ ┌──────────────────────────────────┐ │
│ │ Camera │───▶│ backend.py │ │
│ │ /dev/ │ │ YOLOv8-seg inference │ │
│ │ video0 │ │ Zone overlap (ray-casting) │ │
│ └──────────┘ │ Persistence counter │ │
│ │ Telegram alerter (threaded) │ │
│ └──────────────┬───────────────────-┘ │
│ │ FrameResult │
│ ┌──────────────▼───────────────────-┐ │
│ │ frontend.py │ │
│ │ Flask MJPEG stream (:8891) │ │
│ │ Zone drawing (click-to-polygon) │ │
│ │ Status polling (500ms) │ │
│ └──────────────┬────────────────────┘ │
└─────────────────────────────────│──────────────────────┘
│ SSH Tunnel
┌─────────▼──────────┐
│ Mac / Browser │
│ localhost:8891 │
└────────────────────┘
│
┌─────────▼──────────┐
│ Telegram Bot │
│ Push Alert + │
│ Photo Snapshot │
└────────────────────┘


---

## Project Structure

SecureZone-Real-Time-Monitoring/
├── backend.py # YOLOv8-seg inference, zone logic, Telegram alerts
├── frontend.py # Flask web UI, MJPEG stream, zone drawing
└── README.md


---

## Algorithm Details

### Zone Overlap Detection

Uses pixel-precise instance segmentation masks from YOLOv8-seg. For each detected person, checks whether any point of their segmentation contour lies inside the user-defined polygon using the **ray-casting algorithm**. Falls back to bbox foot-point check if mask unavailable.

### Persistence Threshold (Anti-False-Positive)

frame 1: person in zone → persist = 1
frame 2: person in zone → persist = 2
...
frame 10: person in zone → persist = 10 → ALERT FIRES
frame 11: person leaves → persist = 0, alert resets


Eliminates false positives from momentary occlusions or misdetections.

---

## Hardware & Software Stack

| Component | Detail |
|---|---|
| **Board** | NVIDIA Jetson Orin Developer Kit |
| **Container** | nvcr.io/nvidia/l4t-ml:r36.2.0-py3 |
| **Python** | 3.10 (conda env: yolo_gpu) |
| **Model** | YOLOv8n-seg (ultralytics 8.4.x) |
| **Web Framework** | Flask |
| **Alert** | Telegram Bot API |

---

## Setup & Running

### Configuration

Edit `backend.py`:

```python
TELEGRAM_BOT_TOKEN   = "your_bot_token"
TELEGRAM_CHAT_ID     = "your_chat_id"
MODEL_PATH           = "/path/to/yolov8n-seg.pt"
CAMERA_INDEX         = 0
CONFIDENCE_THRESHOLD = 0.50
PERSISTENCE_FRAMES   = 10
ALERT_COOLDOWN_SEC   = 30
```

### Run

```bash
# Inside Docker container, yolo_gpu env active
python frontend.py
```

### Access UI

```bash
# SSH tunnel from Mac
ssh -L 8891:localhost:8891 jetson@192.168.3.2
# Open browser at http://localhost:8891
```

### Draw a Zone

1. Click on the video feed to place polygon points (min 3)
2. Click **Close Zone** to finalize
3. Person detected inside zone triggers alert after 10 frames
4. Click **Reset Zone** to clear and redraw

---

## Results

- **Inference speed**: ~24 FPS on Jetson Orin (YOLOv8n-seg)
- **Latency**: <100ms end-to-end
- **False positive rate**: Near-zero with persistence threshold
- **Alert delivery**: Telegram photo alert within ~2 seconds of threshold crossing

---

## References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [NVIDIA Jetson L4T ML Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-ml)
- [Telegram Bot API](https://core.telegram.org/bots/api)
EOF
