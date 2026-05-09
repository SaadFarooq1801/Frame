"""
mac_detector.py — Real-time Geometric Shape Detection via MacBook Camera
-------------------------------------------------------------------------
Loads the custom-trained model (shape_model.pth) and runs live inference
through your MacBook's built-in camera. Displays detected shape name,
dominant color, and geometry info (faces, vertices, edges). When 2 shapes
are visible and an ArUco marker is present, labels the shape closest to it.

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
import platform
import subprocess
import time
import collections

PLATFORM = platform.system()   # "Darwin" | "Windows" | "Linux"

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Config

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "shape_model.pth")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "class_labels.json")
IMG_SIZE    = 224
CONF_THRESH = 0.35
SMOOTH_N    = 8

# Shape Properties

SHAPE_INFO = {
    "cylinder": {
        "faces":    3,
        "vertices": 0,
        "edges":    2,
        "desc":     "2 circular faces + 1 curved surface",
    },
    "rectangular_prism": {
        "faces":    6,
        "vertices": 8,
        "edges":    12,
        "desc":     "6 rectangular faces",
    },
    "unknown": {
        "faces":    "?",
        "vertices": "?",
        "edges":    "?",
        "desc":     "",
    },
}

# Colours & Style

PALETTE = {
    "cylinder":          (0, 229, 160),   # mint green
    "rectangular_prism": (0, 174, 255),   # sky blue
    "unknown":           (100, 100, 100), # grey
}
FONT     = cv2.FONT_HERSHEY_DUPLEX
BG_ALPHA = 0.55

# ArUco Setup

try:
    _ARUCO_DICT     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    _ARUCO_PARAMS   = cv2.aruco.DetectorParameters()
    _ARUCO_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)
    ARUCO_AVAILABLE = True
except AttributeError:
    ARUCO_AVAILABLE = False

# Device

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# Camera Scanner

def _cam_backend() -> int:
    """Return the best OpenCV camera backend for the current OS."""
    if PLATFORM == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if PLATFORM == "Windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY   # Linux / other

def _list_camera_names() -> list[str]:
    """Best-effort camera name list, platform-specific."""
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
    return []   # Linux: OpenCV has no portable name API

def scan_cameras(max_index: int = 5) -> list[dict]:
    """Probe OpenCV indices 0..max_index-1 and return working ones."""
    backend = _cam_backend()
    names   = _list_camera_names()
    found   = []
    print(f"\n🔍 Scanning cameras (indices 0–{max_index - 1})...")
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
    """Return the index most likely to be the built-in camera."""
    if PLATFORM == "Darwin":
        keywords = ["facetime", "built-in", "built in", "isight"]
    else:
        keywords = ["integrated", "internal", "built-in", "hd camera", "webcam"]
    for cam in cameras:
        if any(kw in cam["name"].lower() for kw in keywords):
            return cam["index"]
    return cameras[-1]["index"] if cameras else 0

# Model Loading 

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

# Preprocessing

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

# Inference
def predict(model, tensor, label_map):
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(dim=0)
    label = label_map[str(idx.item())]
    return label, conf.item()

# Color Detection

def get_dominant_color(frame_bgr, bbox=None, contour=None):
    """
    Detect dominant object color via HSV pixel analysis.
    When a contour is supplied, samples only pixels inside its convex hull —
    this prevents background bleed entirely. Falls back to center-50% crop.
    Detects: black, white, gray, red, orange, yellow, green, blue, purple, pink, brown.
    """
    fh, fw = frame_bgr.shape[:2]

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        roi_bgr = frame_bgr[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            return "unknown"
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        rh, rw  = roi_bgr.shape[:2]

        valid = None
        if contour is not None:
            hull     = cv2.convexHull(contour)
            hull_roi = (hull - np.array([x1, y1])).astype(np.int32)
            mask     = np.zeros((rh, rw), dtype=np.uint8)
            cv2.fillPoly(mask, [hull_roi], 255)
            mask  = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=2)
            valid = mask > 0
            if int(valid.sum()) < 100:
                valid = None   # hull too small after erosion — fallback

        if valid is not None:
            H = roi_hsv[:, :, 0][valid].astype(np.int32)
            S = roi_hsv[:, :, 1][valid].astype(np.int32)
            V = roi_hsv[:, :, 2][valid].astype(np.int32)
        else:
            # Center 50% of the roi
            ch, cw = rh // 2, rw // 2
            qh, qw = max(4, ch // 2), max(4, cw // 2)
            crop   = roi_hsv[ch - qh: ch + qh, cw - qw: cw + qw]
            if crop.size == 0:
                return "unknown"
            H = crop[:, :, 0].ravel().astype(np.int32)
            S = crop[:, :, 1].ravel().astype(np.int32)
            V = crop[:, :, 2].ravel().astype(np.int32)
    else:
        center = frame_bgr[fh // 4: 3 * fh // 4, fw // 4: 3 * fw // 4]
        if center.size == 0:
            return "unknown"
        hsv_c = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        H = hsv_c[:, :, 0].ravel().astype(np.int32)
        S = hsv_c[:, :, 1].ravel().astype(np.int32)
        V = hsv_c[:, :, 2].ravel().astype(np.int32)

    if len(H) == 0:
        return "unknown"
    total = len(H)

    # Pixel category masks
    is_black  = V < 55                            # dark regardless of saturation
    is_white  = (V > 190) & (S < 50)             # bright and desaturated
    is_gray   = (S < 50) & ~is_black & ~is_white  # mid-brightness, desaturated
    is_chroma = (S >= 35) & ~is_black             # any hue (includes pastels)

    chroma_frac = int(is_chroma.sum()) / total
    if chroma_frac < 0.12:
        # Mostly achromatic — return whichever category dominates
        achro = [
            ("black", int(is_black.sum())),
            ("white", int(is_white.sum())),
            ("gray",  int(is_gray.sum())),
        ]
        return max(achro, key=lambda t: t[1])[0]

    # Work with chromatic pixels only
    cH = H[is_chroma]
    cS = S[is_chroma]
    cV = V[is_chroma]
    n  = len(cH)

    # Pink vs light-red: red-adjacent hue + moderate saturation + high brightness
    # (pastel / light pink has S too low to be "vivid red")
    is_red_hue = (cH < 12) | (cH > 160)   # near H=0 or H=180 in OpenCV units
    pink_cands = is_red_hue & (cS < 160) & (cV > 150)
    vivid_red  = is_red_hue & (cS >= 120)
    if int(pink_cands.sum()) > n * 0.15 and int(pink_cands.sum()) > int(vivid_red.sum()):
        return "pink"

    # Hue histogram — 18 bins × 10 OCv units = 20° standard each
    hist, _ = np.histogram(cH, bins=18, range=(0, 180))
    peak_bin = int(np.argmax(hist))
    peak_ocv = peak_bin * 10 + 5   # OCv H at bin centre
    peak_deg = peak_ocv * 2        # standard 0-360°

    # Brown: orange-range hue but low brightness (not vivid enough to be orange)
    if 20 <= peak_deg < 60:
        orange_mask = (cH >= 10) & (cH <= 30)
        if orange_mask.any() and float(cV[orange_mask].mean()) < 130:
            return "brown"

    # Standard hue mapping
    if peak_deg < 20 or peak_deg >= 345:
        return "red"
    if peak_deg < 50:
        return "orange"
    if peak_deg < 75:
        return "yellow"
    if peak_deg < 165:
        return "green"
    if peak_deg < 255:
        return "blue"
    if peak_deg < 300:
        return "purple"
    return "pink"   # 300-345°: magenta / deep pink

# ArUco Marker Detection

def detect_aruco_markers(frame):
    """Return list of (cx, cy) centers for detected ArUco markers."""
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

# Shape Region Detection

def find_shape_bboxes(frame, min_area_frac=0.04):
    """
    Find 3D shape regions via Canny + contour analysis with quality scoring.
    Each candidate is scored by area × solidity × centrality so that compact,
    centrally-placed shapes beat sprawling background clutter.
    Returns list of (bbox, contour) tuples, best-scoring first.
    """
    fh, fw  = frame.shape[:2]
    cx_f, cy_f = fw // 2, fh // 2
    max_dist   = ((fw ** 2 + fh ** 2) ** 0.5) / 2

    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edges   = cv2.Canny(blurred, 30, 90)
    # Smaller kernel than before — avoids merging the shape with nearby clutter
    dilated = cv2.dilate(edges, np.ones((15, 15), np.uint8), iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = fw * fh * min_area_frac
    max_area = fw * fh * 0.85

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Solidity: 3D geometric shapes have compact, convex silhouettes
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 0
        if solidity < 0.45:          # irregular clutter (furniture, text, cables)
            continue

        # Aspect ratio: reject very wide or very tall blobs (desk edges, window frames)
        aspect = bw / max(bh, 1)
        if aspect < 0.25 or aspect > 4.0:
            continue

        # Centrality: user points camera at the object, so it's near the centre
        cnt_cx = x + bw // 2
        cnt_cy = y + bh // 2
        dist   = ((cnt_cx - cx_f) ** 2 + (cnt_cy - cy_f) ** 2) ** 0.5
        centrality = 1.0 - dist / max_dist

        score = area * solidity * (0.5 + 0.5 * centrality)
        candidates.append((score, area, (x, y, x + bw, y + bh), cnt))

    candidates.sort(key=lambda c: c[0], reverse=True)

    if not candidates:
        return []

    results = [(candidates[0][2], candidates[0][3])]

    # Second shape only if it is a substantial fraction of the first
    if len(candidates) > 1 and candidates[1][1] >= candidates[0][1] * 0.30:
        results.append((candidates[1][2], candidates[1][3]))

    return results

# Drawing

def draw_overlay(frame, detections, fps, marker_centers):
    """
    detections    : list of dicts {bbox, label, confidence, color_name}
                    bbox=None → single-shape bottom-bar fallback mode
    marker_centers: list of (cx, cy) for detected ArUco markers
    """
    h, w = frame.shape[:2]

    # Top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 48), (15, 15, 20), -1)
    cv2.addWeighted(bar, BG_ALPHA, frame, 1 - BG_ALPHA, 0, frame)
    cv2.putText(frame, "Shape Detector", (12, 32), FONT, 0.75, (220, 220, 220), 1)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 32), FONT, 0.65, (160, 160, 160), 1)

    # ArUco markers
    for mx, my in marker_centers:
        cv2.circle(frame, (mx, my), 28, (0, 255, 255), 2)
        cv2.line(frame, (mx - 18, my), (mx + 18, my), (0, 255, 255), 1)
        cv2.line(frame, (mx, my - 18), (mx, my + 18), (0, 255, 255), 1)
        cv2.putText(frame, "MARKER", (mx - 28, my - 34), FONT, 0.42, (0, 255, 255), 1)

    confident = [d for d in detections if d["confidence"] >= CONF_THRESH]

    if not confident:
        bot = frame.copy()
        cv2.rectangle(bot, (0, h - 70), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(bot, BG_ALPHA + 0.1, frame, 1 - (BG_ALPHA + 0.1), 0, frame)
        cv2.putText(frame, "Scanning...", (18, h - 40), FONT, 0.9, (120, 120, 120), 1)
        cv2.putText(frame, "Point camera at a shape", (18, h - 14),
                    FONT, 0.52, (90, 90, 90), 1)
        return frame

    # Single shape with no bbox → bottom-bar display
    if len(confident) == 1 and confident[0]["bbox"] is None:
        det        = confident[0]
        label      = det["label"]
        conf       = det["confidence"]
        color_name = det["color_name"]
        shape_col  = PALETTE.get(label, PALETTE["unknown"])
        info       = SHAPE_INFO.get(label, SHAPE_INFO["unknown"])

        bot = frame.copy()
        cv2.rectangle(bot, (0, h - 120), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(bot, BG_ALPHA + 0.1, frame, 1 - (BG_ALPHA + 0.1), 0, frame)

        cv2.putText(frame, label.replace("_", " ").title(),
                    (18, h - 88), FONT, 1.05, shape_col, 2)
        cv2.putText(frame, f"Color: {color_name.title()}",
                    (18, h - 62), FONT, 0.52, (200, 200, 200), 1)
        cv2.putText(frame,
                    f"Faces: {info['faces']}   Vertices: {info['vertices']}   Edges: {info['edges']}",
                    (18, h - 40), FONT, 0.46, (170, 170, 170), 1)

        bar_w = int((w - 36) * conf)
        cv2.rectangle(frame, (18, h - 22), (18 + bar_w, h - 8), shape_col, -1)
        cv2.rectangle(frame, (18, h - 22), (w - 18, h - 8), (80, 80, 80), 1)
        cv2.putText(frame, f"{conf:.0%}", (w - 65, h - 10),
                    FONT, 0.45, (200, 200, 200), 1)

        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 22, cy), (cx + 22, cy), shape_col, 1)
        cv2.line(frame, (cx, cy - 22), (cx, cy + 22), shape_col, 1)
        cv2.circle(frame, (cx, cy), 40, shape_col, 1)
        return frame

    # Multi-shape per-bbox display
    nearest_idx = -1
    if marker_centers and len(confident) >= 2:
        mk_x = sum(m[0] for m in marker_centers) / len(marker_centers)
        mk_y = sum(m[1] for m in marker_centers) / len(marker_centers)
        nearest_idx = min(
            range(len(confident)),
            key=lambda i: (
                ((confident[i]["bbox"][0] + confident[i]["bbox"][2]) / 2 - mk_x) ** 2 +
                ((confident[i]["bbox"][1] + confident[i]["bbox"][3]) / 2 - mk_y) ** 2
            ),
        )

    for i, det in enumerate(confident):
        x1, y1, x2, y2 = det["bbox"]
        label      = det["label"]
        conf       = det["confidence"]
        color_name = det["color_name"]
        shape_col  = PALETTE.get(label, PALETTE["unknown"])
        info       = SHAPE_INFO.get(label, SHAPE_INFO["unknown"])
        is_nearest = (i == nearest_idx)

        cv2.rectangle(frame, (x1, y1), (x2, y2), shape_col, 3 if is_nearest else 2)

        lines = [
            (label.replace("_", " ").title(),                          shape_col,       0.65, 2),
            (f"Color: {color_name.title()}",                           (200, 200, 200), 0.46, 1),
            (f"Faces: {info['faces']}  Verts: {info['vertices']}  Edges: {info['edges']}",
             (170, 170, 170), 0.40, 1),
        ]
        if is_nearest:
            lines.append(("NEAREST TO MARKER", (0, 255, 255), 0.44, 1))

        line_h  = 20
        panel_h = len(lines) * line_h + 8
        py1     = max(50, y1 - panel_h)

        panel = frame.copy()
        cv2.rectangle(panel, (x1, py1), (min(x2 + 10, w - 1), y1), (15, 15, 20), -1)
        cv2.addWeighted(panel, BG_ALPHA, frame, 1 - BG_ALPHA, 0, frame)

        ty = py1 + line_h
        for text, col, scale, thick in lines:
            cv2.putText(frame, text, (x1 + 6, ty), FONT, scale, col, thick)
            ty += line_h

        bar_y = y2 - 10
        bar_w = int((x2 - x1) * conf)
        cv2.rectangle(frame, (x1, bar_y), (x1 + bar_w, y2), shape_col, -1)
        cv2.rectangle(frame, (x1, bar_y), (x2, y2), (80, 80, 80), 1)
        cv2.putText(frame, f"{conf:.0%}", (x2 - 52, y2 - 1),
                    FONT, 0.40, (220, 220, 220), 1)

    return frame

# Main Loop
def run_detector(camera_index: int, model, label_map, device):
    cap = cv2.VideoCapture(camera_index, _cam_backend())
    if not cap.isOpened():
        hint = (
            "System Settings → Privacy & Security → Camera"
            if PLATFORM == "Darwin"
            else "Settings → Privacy & Security → Camera" if PLATFORM == "Windows"
            else "check /dev/video* permissions"
        )
        raise RuntimeError(f"❌ Cannot open camera index {camera_index}. Check: {hint}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"\n📷 Camera {camera_index} opened — press Q or Esc to quit\n")

    frame_count     = 0
    t_start         = time.time()
    last_fps        = 0.0
    last_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Camera read failed, retrying...")
            continue

        frame_count += 1

        if frame_count % 2 == 0:
            regions = find_shape_bboxes(frame)
            if regions:
                new_dets = []
                for bbox, cnt in regions:
                    x1, y1, x2, y2 = bbox
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    tensor = frame_to_tensor(roi, device)
                    lbl, conf = predict(model, tensor, label_map)
                    color_name = get_dominant_color(frame, bbox, cnt)
                    new_dets.append({
                        "bbox":       bbox,
                        "label":      lbl,
                        "confidence": conf,
                        "color_name": color_name,
                    })
                last_detections = new_dets
            else:
                # Fallback: classify the whole frame
                tensor = frame_to_tensor(frame, device)
                lbl, conf = predict(model, tensor, label_map)
                last_detections = [{
                    "bbox":       None,
                    "label":      lbl,
                    "confidence": conf,
                    "color_name": get_dominant_color(frame),
                }]

        marker_centers = detect_aruco_markers(frame)

        if frame_count % 15 == 0:
            elapsed  = time.time() - t_start
            last_fps = 15 / max(elapsed, 1e-6)
            t_start  = time.time()

        frame = draw_overlay(frame, last_detections, last_fps, marker_centers)
        cv2.imshow("Shape Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Detector closed.\n")


def main():
    parser = argparse.ArgumentParser(description="Shape Detector — Camera")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index to use (default: auto-detect built-in)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="List all available cameras and exit")
    args = parser.parse_args()

    print("=" * 52)
    print(f"  🔷 Shape Detector  —  {PLATFORM} Camera Mode")
    print("=" * 52)

    # Camera scan
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

    # Pick camera
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

    # Load model & run
    device = get_device()
    print(f"🖥️  Device: {device}")
    model, label_map = load_model(device)
    run_detector(cam_index, model, label_map, device)


if __name__ == "__main__":
    main()
