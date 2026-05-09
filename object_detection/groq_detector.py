"""
groq_detector.py — Real-time Shape Detection via Groq Vision API
-----------------------------------------------------------------
Uses llama-4-scout (or llama-3.2-vision) on Groq to identify 3D shapes,
their color, and geometry info — no local model training required.

API calls run in a background thread so the camera feed is always smooth.
The overlay label updates whenever a new Groq response arrives.

Usage:
    python groq_detector.py                  # auto-picks built-in camera
    python groq_detector.py --list-cameras   # show available cameras
    python groq_detector.py --camera 1       # force a specific camera index
    python groq_detector.py --interval 2.0   # seconds between API calls (default: 1.5)

API key — set one of these (checked in order):
    export GROQ_API_KEY="gsk_..."
    or pass --api-key "gsk_..."

Install dependencies:
    pip install groq opencv-python pillow numpy
"""

import argparse
import base64
import os
import platform
import subprocess
import threading
import time

PLATFORM = platform.system()   # "Darwin" | "Windows" | "Linux"

import cv2
import numpy as np
from PIL import Image

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Shape properties — Groq returns a shape name; we look up geometry here.
# Add any shape you want Groq to recognise in the prompt below.
SHAPE_INFO = {
    "cube": {
        "faces": 6, "vertices": 8, "edges": 12,
        "desc": "6 equal square faces",
    },
    "sphere": {
        "faces": 1, "vertices": 0, "edges": 0,
        "desc": "Perfectly round curved surface",
    },
    "cylinder": {
        "faces": 3, "vertices": 0, "edges": 2,
        "desc": "2 circular faces + 1 curved surface",
    },
    "rectangular prism": {
        "faces": 6, "vertices": 8, "edges": 12,
        "desc": "6 rectangular faces",
    },
    "cone": {
        "faces": 2, "vertices": 1, "edges": 1,
        "desc": "1 circular base + 1 curved surface",
    },
    "pyramid": {
        "faces": 5, "vertices": 5, "edges": 8,
        "desc": "Square base + 4 triangular faces",
    },
    "unknown": {
        "faces": "?", "vertices": "?", "edges": "?",
        "desc": "",
    },
}

PALETTE = {
    "cube":              (0, 174, 255),   # sky blue
    "sphere":            (0, 229, 160),   # mint green
    "cylinder":          (80, 200, 120),  # green
    "rectangular prism": (255, 160, 40),  # orange
    "cone":              (200, 80, 255),  # purple
    "pyramid":           (255, 220, 40),  # yellow
    "unknown":           (100, 100, 100), # grey
}

FONT     = cv2.FONT_HERSHEY_DUPLEX
BG_ALPHA = 0.55

# llama-4-scout is Groq's recommended fast vision model
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

GROQ_PROMPT = """\
Look at the image and identify the primary 3D geometric shape.
Reply with ONLY valid JSON, no markdown, no extra text:
{"shape":"<cube|sphere|cylinder|cone|pyramid|rectangular prism|unknown>","color":"<one word color>","confidence":<0.0-1.0>}
If no clear 3D shape is visible use "unknown". Color must be one of: red, orange, yellow, green, blue, purple, pink, brown, black, white, gray."""

# ArUco setup
try:
    _ARUCO_DICT     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    _ARUCO_PARAMS   = cv2.aruco.DetectorParameters()
    _ARUCO_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)
    ARUCO_AVAILABLE = True
except AttributeError:
    ARUCO_AVAILABLE = False


# Camera helpers (same logic as mac_detector / clip_detector)

def _cam_backend() -> int:
    if PLATFORM == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if PLATFORM == "Windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY

def _list_camera_names() -> list[str]:
    if PLATFORM == "Darwin":
        try:
            raw = subprocess.check_output(
                ["system_profiler", "SPCameraDataType"],
                stderr=subprocess.DEVNULL, timeout=5,
            ).decode()
            names = []
            for line in raw.splitlines():
                stripped = line.strip()
                if line.startswith("      ") and stripped.endswith(":") and "Model" not in stripped:
                    names.append(stripped.rstrip(":"))
            return names
        except Exception:
            return []
    if PLATFORM == "Windows":
        try:
            raw = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "Get-PnpDevice -Class Camera -Status OK "
                 "| Select-Object -ExpandProperty FriendlyName"],
                stderr=subprocess.DEVNULL, timeout=5,
            ).decode(errors="ignore")
            return [ln.strip() for ln in raw.splitlines() if ln.strip()]
        except Exception:
            return []
    return []

def scan_cameras(max_index: int = 5) -> list[dict]:
    backend = _cam_backend()
    names   = _list_camera_names()
    found   = []
    print(f"\nScanning cameras (indices 0-{max_index - 1})...")
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                name = names[i] if i < len(names) else f"Camera {i}"
                found.append({"index": i, "name": name})
    return found

def pick_builtin_camera(cameras: list[dict]) -> int:
    if PLATFORM == "Darwin":
        keywords = ["facetime", "built-in", "built in", "isight"]
    else:
        keywords = ["integrated", "internal", "built-in", "hd camera", "webcam"]
    for cam in cameras:
        if any(kw in cam["name"].lower() for kw in keywords):
            return cam["index"]
    return cameras[-1]["index"] if cameras else 0


# ArUco detection

def detect_aruco_markers(frame):
    if not ARUCO_AVAILABLE:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _ARUCO_DETECTOR.detectMarkers(gray)
    centers = []
    if ids is not None:
        for corner in corners:
            pts = corner[0]
            centers.append((int(pts[:, 0].mean()), int(pts[:, 1].mean())))
    return centers


# Frame → base64 JPEG (for Groq API)

def frame_to_b64(frame_bgr, quality: int = 55) -> str:
    """Encode a BGR OpenCV frame as a base64 JPEG string."""
    import io
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil = pil.resize((320, 180), Image.LANCZOS)   # small = fast upload + fast inference
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Groq inference

def call_groq(client: "Groq", b64_image: str) -> dict:
    """
    Send one frame to Groq and parse the JSON response.
    Returns a dict with keys: shape, color, confidence, notes.
    On any error returns {"shape": "unknown", "color": "unknown", "confidence": 0.0, "notes": ""}.
    """
    import json

    fallback = {"shape": "unknown", "color": "unknown", "confidence": 0.0}
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                            },
                        },
                        {
                            "type": "text",
                            "text": GROQ_PROMPT,
                        },
                    ],
                }
            ],
            max_tokens=80,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if the model wraps its JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        return {
            "shape":      str(parsed.get("shape", "unknown")).lower().strip(),
            "color":      str(parsed.get("color", "unknown")).lower().strip(),
            "confidence": float(parsed.get("confidence", 0.5)),
        }
    except Exception as exc:
        print(f"  [Groq error] {exc}")
        return fallback


# Overlay drawing

def draw_overlay(frame, detection: dict, fps: float, api_ms: float, marker_centers: list):
    """
    detection keys: shape, color, confidence, notes
    api_ms        : last API round-trip in milliseconds (0 if not yet called)
    """
    h, w = frame.shape[:2]

    shape      = detection.get("shape", "unknown")
    color_name = detection.get("color", "unknown")
    conf       = detection.get("confidence", 0.0)
    info       = SHAPE_INFO.get(shape, SHAPE_INFO["unknown"])
    shape_col  = PALETTE.get(shape, PALETTE["unknown"])

    # Top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 48), (15, 15, 20), -1)
    cv2.addWeighted(bar, BG_ALPHA, frame, 1 - BG_ALPHA, 0, frame)
    cv2.putText(frame, "Shape Detector  [Groq]", (12, 32), FONT, 0.72, (220, 220, 220), 1)
    fps_txt = f"FPS: {fps:.0f}  |  API: {api_ms:.0f}ms" if api_ms > 0 else f"FPS: {fps:.0f}"
    cv2.putText(frame, fps_txt, (w - 260, 32), FONT, 0.52, (160, 160, 160), 1)

    # ArUco markers
    for mx, my in marker_centers:
        cv2.circle(frame, (mx, my), 28, (0, 255, 255), 2)
        cv2.line(frame, (mx - 18, my), (mx + 18, my), (0, 255, 255), 1)
        cv2.line(frame, (mx, my - 18), (mx, my + 18), (0, 255, 255), 1)
        cv2.putText(frame, "MARKER", (mx - 28, my - 34), FONT, 0.42, (0, 255, 255), 1)

    # Scanning state (no result yet or low confidence)
    if shape == "unknown" or conf < 0.3:
        bot = frame.copy()
        cv2.rectangle(bot, (0, h - 70), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(bot, BG_ALPHA + 0.1, frame, 1 - (BG_ALPHA + 0.1), 0, frame)
        cv2.putText(frame, "Waiting for Groq response..." if api_ms == 0 else "Scanning...",
                    (18, h - 40), FONT, 0.9, (120, 120, 120), 1)
        cv2.putText(frame, "Point camera at a 3D shape",
                    (18, h - 14), FONT, 0.48, (90, 90, 90), 1)
        return frame

    # Result panel — bottom bar
    bot = frame.copy()
    cv2.rectangle(bot, (0, h - 140), (w, h), (15, 15, 20), -1)
    cv2.addWeighted(bot, BG_ALPHA + 0.1, frame, 1 - (BG_ALPHA + 0.1), 0, frame)

    cv2.putText(frame, shape.title(), (18, h - 108), FONT, 1.1, shape_col, 2)
    cv2.putText(frame, f"Color: {color_name.title()}",
                (18, h - 80), FONT, 0.52, (200, 200, 200), 1)
    cv2.putText(frame,
                f"Faces: {info['faces']}   Vertices: {info['vertices']}   Edges: {info['edges']}",
                (18, h - 58), FONT, 0.46, (170, 170, 170), 1)

    bar_w = int((w - 36) * conf)
    cv2.rectangle(frame, (18, h - 38), (18 + bar_w, h - 22), shape_col, -1)
    cv2.rectangle(frame, (18, h - 38), (w - 18, h - 22), (80, 80, 80), 1)
    cv2.putText(frame, f"{conf:.0%}", (w - 65, h - 24), FONT, 0.45, (200, 200, 200), 1)

    # Crosshair
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 22, cy), (cx + 22, cy), shape_col, 1)
    cv2.line(frame, (cx, cy - 22), (cx, cy + 22), shape_col, 1)
    cv2.circle(frame, (cx, cy), 40, shape_col, 1)

    return frame


# Main loop

def run_detector(camera_index: int, client: "Groq", interval: float):
    cap = cv2.VideoCapture(camera_index, _cam_backend())
    if not cap.isOpened():
        hint = (
            "System Settings → Privacy & Security → Camera"
            if PLATFORM == "Darwin"
            else "Settings → Privacy & Security → Camera" if PLATFORM == "Windows"
            else "check /dev/video* permissions"
        )
        raise RuntimeError(f"Cannot open camera index {camera_index}. Check: {hint}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"\nCamera {camera_index} opened — press Q or Esc to quit")
    print(f"Groq API call every {interval:.1f}s  |  model: {GROQ_MODEL}\n")

    # Shared state updated by the background inference thread
    result_lock    = threading.Lock()
    last_detection = {"shape": "unknown", "color": "unknown", "confidence": 0.0, "notes": ""}
    last_api_ms    = 0.0
    api_thread     = None
    last_api_call  = 0.0   # wall-clock time the last thread was spawned

    last_fps    = 0.0
    frame_count = 0
    t_fps       = time.time()

    def _bg_call(frame_snapshot):
        nonlocal last_detection, last_api_ms
        b64    = frame_to_b64(frame_snapshot)
        t0     = time.time()
        result = call_groq(client, b64)
        ms     = (time.time() - t0) * 1000
        with result_lock:
            last_detection = result
            last_api_ms    = ms
        print(f"  [{result['shape']}]  color={result['color']}  "
              f"conf={result['confidence']:.0%}  api={ms:.0f}ms")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed, retrying...")
            continue

        frame_count += 1
        now = time.time()

        # Spawn a new inference thread only when the previous one has finished
        # and the interval has elapsed — never blocks the camera loop
        if (api_thread is None or not api_thread.is_alive()) and (now - last_api_call >= interval):
            last_api_call = now
            api_thread = threading.Thread(target=_bg_call, args=(frame.copy(),), daemon=True)
            api_thread.start()

        if frame_count % 15 == 0:
            elapsed  = time.time() - t_fps
            last_fps = 15 / max(elapsed, 1e-6)
            t_fps    = time.time()

        marker_centers = detect_aruco_markers(frame)
        with result_lock:
            det    = dict(last_detection)
            api_ms = last_api_ms
        frame = draw_overlay(frame, det, last_fps, api_ms, marker_centers)
        cv2.imshow("Shape Detector [Groq]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDetector closed.\n")


def main():
    parser = argparse.ArgumentParser(description="Groq Vision Shape Detector")
    parser.add_argument("--camera",      type=int,   default=None)
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--interval",    type=float, default=1.5,
                        help="Seconds between Groq API calls (default: 1.5)")
    parser.add_argument("--api-key",     type=str,   default=None,
                        help="Groq API key (or set GROQ_API_KEY env var)")
    args = parser.parse_args()

    print("=" * 54)
    print("  Shape Detector  —  Groq Vision Mode")
    print("=" * 54)

    if not GROQ_AVAILABLE:
        print("\nGroq SDK not installed. Run:")
        print("   pip install groq")
        return

    api_key = args.api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("\nNo API key found. Set GROQ_API_KEY or pass --api-key.")
        print("Get a free key at: console.groq.com")
        return

    client = Groq(api_key=api_key)

    cameras = scan_cameras(max_index=6)
    if not cameras:
        print("No cameras found.")
        return

    print("\nAvailable cameras:")
    for cam in cameras:
        print(f"   [{cam['index']}] {cam['name']}")

    if args.list_cameras:
        return

    cam_index = args.camera if args.camera is not None else pick_builtin_camera(cameras)
    cam_name  = next((c["name"] for c in cameras if c["index"] == cam_index), f"Camera {cam_index}")
    print(f"\nUsing: [{cam_index}] {cam_name}")

    run_detector(cam_index, client, args.interval)


if __name__ == "__main__":
    main()
