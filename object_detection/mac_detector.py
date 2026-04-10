"""
mac_detector.py — Real-time Geometric Shape Detection via MacBook Camera
-------------------------------------------------------------------------
Loads the custom-trained model (shape_model.pth) and runs live inference
through your MacBook's built-in camera. Displays a stylish overlay window
showing the detected shape name and confidence.

Usage:
    python mac_detector.py                  # auto-picks built-in camera
    python mac_detector.py --list-cameras   # show all cameras + their index
    python mac_detector.py --camera 1       # force a specific camera index

Prerequisites:
    1. Run train_model.py first to generate shape_model.pth
    2. pip install torch torchvision opencv-python pillow
"""

import argparse
import json
import os
import subprocess
import time
import collections

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "shape_model.pth")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "class_labels.json")
IMG_SIZE    = 224
CONF_THRESH = 0.55     # Show label only if confidence > this
SMOOTH_N    = 8        # Smooth predictions over last N frames (reduces jitter)

# ─── Colours & Style ──────────────────────────────────────────────────────────

PALETTE = {
    "cylinder":          (0, 229, 160),   # mint green
    "rectangular_prism": (0, 174, 255),   # sky blue
    "unknown":           (100, 100, 100), # grey
}
FONT       = cv2.FONT_HERSHEY_DUPLEX
BG_ALPHA   = 0.55   # overlay transparency

# ─── Device ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ─── Camera Scanner ───────────────────────────────────────────────────────────

def get_macos_camera_names() -> list[str]:
    """Use system_profiler to get ordered camera names on macOS."""
    try:
        raw = subprocess.check_output(
            ["system_profiler", "SPCameraDataType"],
            stderr=subprocess.DEVNULL, timeout=5
        ).decode()
        # Each camera block starts with a name line (no leading spaces)
        names = []
        for line in raw.splitlines():
            stripped = line.strip()
            # Camera names are indented 6 spaces followed by a display name and colon
            if line.startswith("      ") and stripped.endswith(":") and "Model" not in stripped:
                names.append(stripped.rstrip(":"))
        return names
    except Exception:
        return []

def scan_cameras(max_index: int = 5) -> list[dict]:
    """Probe OpenCV indices 0..max_index-1 and return working ones."""
    names = get_macos_camera_names()
    found = []
    print(f"\n🔍 Scanning cameras (indices 0–{max_index - 1})...")
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                name = names[i] if i < len(names) else f"Camera {i}"
                found.append({"index": i, "name": name})
    return found

def pick_builtin_camera(cameras: list[dict]) -> int:
    """Return the index most likely to be the built-in FaceTime camera."""
    keywords = ["facetime", "built-in", "built in", "isight"]
    for cam in cameras:
        if any(kw in cam["name"].lower() for kw in keywords):
            return cam["index"]
    # Fallback: last available camera (Continuity Camera usually grabs index 0)
    return cameras[-1]["index"] if cameras else 0

# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(device):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Please run 'python train_model.py' first!"
        )
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")

    checkpoint   = torch.load(MODEL_PATH, map_location=device)
    num_classes  = checkpoint["num_classes"]

    with open(LABELS_PATH) as f:
        label_map = json.load(f)   # {"0": "cylinder", "1": "rectangular_prism"}

    # Rebuild model architecture
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"✅ Model loaded  —  {num_classes} classes: {list(label_map.values())}")
    return model, label_map

# ─── Preprocessing ────────────────────────────────────────────────────────────

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def frame_to_tensor(bgr_frame, device):
    """Convert OpenCV BGR frame → model input tensor."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = preprocess(pil).unsqueeze(0).to(device)
    return tensor

# ─── Inference ────────────────────────────────────────────────────────────────

def predict(model, tensor, label_map):
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(dim=0)
    label = label_map[str(idx.item())]
    return label, conf.item()

# ─── Drawing ──────────────────────────────────────────────────────────────────

def draw_overlay(frame, label, confidence, fps, smoothed_label):
    h, w = frame.shape[:2]

    # ── Top bar (FPS + title) ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (15, 15, 20), -1)
    cv2.addWeighted(overlay, BG_ALPHA, frame, 1 - BG_ALPHA, 0, frame)
    cv2.putText(frame, "Shape Detector", (12, 32), FONT, 0.75, (220, 220, 220), 1)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 32), FONT, 0.65, (160, 160, 160), 1)

    # ── Bottom result bar ──
    bar_h = 90
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - bar_h), (w, h), (15, 15, 20), -1)
    cv2.addWeighted(overlay2, BG_ALPHA + 0.1, frame, 1 - (BG_ALPHA + 0.1), 0, frame)

    color = PALETTE.get(smoothed_label, PALETTE["unknown"])

    if confidence >= CONF_THRESH:
        # Shape name (large)
        display_name = smoothed_label.replace("_", " ").title()
        cv2.putText(frame, display_name, (18, h - 52), FONT, 1.05, color, 2)

        # Confidence bar
        bar_w = int((w - 36) * confidence)
        cv2.rectangle(frame, (18, h - 30), (18 + bar_w, h - 14), color, -1)
        cv2.rectangle(frame, (18, h - 30), (w - 18, h - 14), (80, 80, 80), 1)
        cv2.putText(frame, f"{confidence:.0%}", (w - 65, h - 16),
                    FONT, 0.55, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "Scanning...", (18, h - 44), FONT, 0.9, (120, 120, 120), 1)
        cv2.putText(frame, "Point camera at a shape", (18, h - 18),
                    FONT, 0.52, (90, 90, 90), 1)

    # ── Crosshair guide ──
    cx, cy = w // 2, h // 2
    size = 22
    thick_color = (*color, 180)
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1)
    cv2.circle(frame, (cx, cy), 40, color, 1)

    return frame

# ─── Main Loop ────────────────────────────────────────────────────────────────

def run_detector(camera_index: int, model, label_map, device):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(
            f"❌ Cannot open camera index {camera_index}! "
            "Check camera permissions in System Settings → Privacy & Security → Camera."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"\n📷 Camera {camera_index} opened — press Q or Esc to quit\n")

    pred_history = collections.deque(maxlen=SMOOTH_N)
    label, confidence = "unknown", 0.0
    frame_count = 0
    t_start = time.time()
    last_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Camera read failed, retrying...")
            continue

        frame_count += 1

        if frame_count % 2 == 0:
            tensor            = frame_to_tensor(frame, device)
            label, confidence = predict(model, tensor, label_map)
            pred_history.append(label)

        smoothed = collections.Counter(pred_history).most_common(1)[0][0] if pred_history else "unknown"

        if frame_count % 15 == 0:
            elapsed  = time.time() - t_start
            last_fps = 15 / max(elapsed, 1e-6)
            t_start  = time.time()

        frame = draw_overlay(frame, label, confidence, last_fps, smoothed)
        cv2.imshow("Shape Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Detector closed.\n")


def main():
    parser = argparse.ArgumentParser(description="Shape Detector — MacBook Camera")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index to use (default: auto-detect built-in)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="List all available cameras and exit")
    args = parser.parse_args()

    print("=" * 52)
    print("  🔷 Shape Detector  —  MacBook Camera Mode")
    print("=" * 52)

    # ── Camera scan ──
    cameras = scan_cameras(max_index=6)

    if not cameras:
        print("❌ No cameras found! Check System Settings → Privacy & Security → Camera.")
        return

    print("\n📹 Available cameras:")
    for cam in cameras:
        print(f"   [{cam['index']}] {cam['name']}")

    if args.list_cameras:
        print("\nRun with --camera INDEX to choose one. Example:")
        print(f"   python mac_detector.py --camera {cameras[0]['index']}")
        return

    # ── Pick camera ──
    if args.camera is not None:
        cam_index = args.camera
        cam_name  = next((c["name"] for c in cameras if c["index"] == cam_index), f"Camera {cam_index}")
    else:
        cam_index = pick_builtin_camera(cameras)
        cam_name  = next((c["name"] for c in cameras if c["index"] == cam_index), f"Camera {cam_index}")

    print(f"\n✅ Using: [{cam_index}] {cam_name}")
    if any(kw in cam_name.lower() for kw in ["iphone", "phone", "continuity"]):
        print("⚠️  This looks like an iPhone camera. If wrong, run:")
        others = [c for c in cameras if c["index"] != cam_index]
        if others:
            print(f"   python mac_detector.py --camera {others[0]['index']}")

    # ── Load model & run ──
    device = get_device()
    print(f"🖥️  Device: {device}")
    model, label_map = load_model(device)
    run_detector(cam_index, model, label_map, device)


if __name__ == "__main__":
    main()
