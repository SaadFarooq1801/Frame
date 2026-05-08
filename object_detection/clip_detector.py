"""
Uses OpenAI CLIP to classify shapes with no training and no API key.
Runs fully local on your Mac (Apple Silicon MPS accelerated).
Displays detected shape name, dominant color, and geometry info.
When 2 shapes are visible and an ArUco marker is present, labels the
shape closest to the marker.

Handles all 8 test cases:
  1. Cube  / solid bg   / single object
  2. Cube  / solid bg   / multiple objects
  3. Cube  / colored bg / single object
  4. Cube  / colored bg / multiple objects
  5. Sphere / solid bg  / single object
  6. Sphere / solid bg  / multiple objects
  7. Sphere / colored bg / single object
  8. Sphere / colored bg / multiple objects

Usage:
    python clip_detector.py                  # auto-picks built-in camera
    python clip_detector.py --list-cameras   # show available cameras
    python clip_detector.py --camera 1       # force a specific camera index

Install dependencies:
    pip install torch torchvision opencv-python pillow
    pip install git+https://github.com/openai/CLIP.git
"""

import argparse
import os
import subprocess
import time

import clip
import cv2
import numpy as np
import torch
from PIL import Image


CONF_THRESH = 0.55
SMOOTH_N    = 6

# Shape Properties

SHAPE_INFO = {
    "cube": {
        "faces":    6,
        "vertices": 8,
        "edges":    12,
        "desc":     "6 equal square faces",
    },
    "sphere": {
        "faces":    1,
        "vertices": 0,
        "edges":    0,
        "desc":     "Perfectly round curved surface",
    },
    "unknown": {
        "faces":    "?",
        "vertices": "?",
        "edges":    "?",
        "desc":     "",
    },
}

SHAPE_PROMPTS = {
    "cube": [
        "a photo of a cube",
        "a cube with flat square faces",
        "a 3D cube shape",
        "a wooden cube",
        "a blurry photo of a cube",
        "a cube shaped object on a table",
        "a box shaped like a cube",
    ],
    "sphere": [
        "a photo of a sphere",
        "a perfectly round sphere",
        "a 3D sphere shape",
        "a ball or sphere object",
        "a blurry photo of a sphere",
        "a sphere shaped object on a table",
        "a round ball",
    ],
}

# Colours & Style
PALETTE = {
    "cube":    (0, 174, 255),   # sky blue
    "sphere":  (0, 229, 160),   # mint green
    "unknown": (100, 100, 100), # grey
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

# Camera Scanner (same as mac_detector.py)

def get_macos_camera_names() -> list[str]:
    try:
        raw = subprocess.check_output(
            ["system_profiler", "SPCameraDataType"],
            stderr=subprocess.DEVNULL, timeout=5
        ).decode()
        names = []
        for line in raw.splitlines():
            stripped = line.strip()
            if line.startswith("      ") and stripped.endswith(":") and "Model" not in stripped:
                names.append(stripped.rstrip(":"))
        return names
    except Exception:
        return []

def scan_cameras(max_index: int = 5) -> list[dict]:
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
    keywords = ["facetime", "built-in", "built in", "isight"]
    for cam in cameras:
        if any(kw in cam["name"].lower() for kw in keywords):
            return cam["index"]
    return cameras[-1]["index"] if cameras else 0

# Color Detection

def get_dominant_color(frame_bgr, bbox=None):
    """
    Detect dominant object color via HSV pixel analysis.
    Detects: black, white, gray, red, orange, yellow, green, blue, purple, pink, brown.
    Samples the inner 50% of the bbox to avoid background contamination.
    """
    fh, fw = frame_bgr.shape[:2]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        qw = max(4, (x2 - x1) // 4)
        qh = max(4, (y2 - y1) // 4)
        roi = frame_bgr[max(0, cy - qh): min(fh, cy + qh),
                        max(0, cx - qw): min(fw, cx + qw)]
    else:
        roi = frame_bgr[fh // 4: 3 * fh // 4, fw // 4: 3 * fw // 4]

    if roi.size == 0:
        return "unknown"

    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H     = hsv[:, :, 0].ravel().astype(np.int32)
    S     = hsv[:, :, 1].ravel().astype(np.int32)
    V     = hsv[:, :, 2].ravel().astype(np.int32)
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
    Find bounding boxes of foreground objects via Canny + contour analysis.
    Returns boxes sorted largest-first (largest ≈ closest to camera).
    A second box is only included when its area is ≥ 30 % of the first,
    preventing background clutter from registering as a second shape.
    """
    fh, fw  = frame.shape[:2]
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edges   = cv2.Canny(blurred, 30, 90)
    kernel  = np.ones((25, 25), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = fw * fh * min_area_frac
    max_area = fw * fh * 0.85

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            x, y, bw, bh = cv2.boundingRect(cnt)
            candidates.append((area, (x, y, x + bw, y + bh)))

    # Largest first — closest object wins
    candidates.sort(key=lambda c: c[0], reverse=True)

    if not candidates:
        return []

    bboxes = [candidates[0][1]]

    # Only add a second shape if it is large enough relative to the first
    if len(candidates) > 1 and candidates[1][0] >= candidates[0][0] * 0.30:
        bboxes.append(candidates[1][1])

    return bboxes

# Blur Preprocessing

def enhance_frame(pil_img: Image.Image) -> Image.Image:
    """
    Apply CLAHE + unsharp mask to improve blurry low-res images.
    Designed to recover detail from Frame glasses 256x256 JPEG output,
    but also improves MacBook camera frames at distance.
    """
    img = np.array(pil_img.convert("RGB"))

    # CLAHE: improve local contrast in LAB colour space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Unsharp mask: sharpen edges
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    img = np.clip(img, 0, 255).astype(np.uint8)

    return Image.fromarray(img)

# CLIP Model Loading 

def load_clip(device):
    print("📦 Loading CLIP model (ViT-B/32)...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    print("📝 Encoding text prompts...")
    class_text_features = {}
    with torch.no_grad():
        for label, prompts in SHAPE_PROMPTS.items():
            tokens = clip.tokenize(prompts).to(device)
            features = model.encode_text(tokens)          # (num_prompts, 512)
            features = features / features.norm(dim=-1, keepdim=True)
            class_text_features[label] = features.mean(dim=0)  # average prompts

    print(f"✅ CLIP ready — {len(class_text_features)} classes: {list(class_text_features.keys())}")
    return model, preprocess, class_text_features

# Inference

def predict(model, preprocess, class_text_features, bgr_frame, device):
    """
    Classify a BGR OpenCV frame as cube or sphere using CLIP.
    Uses center crop to focus on the primary object and ignore edge distractors.
    """
    h, w = bgr_frame.shape[:2]

    # Center crop (60% of frame)
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.20)
    cropped = bgr_frame[margin_y:h - margin_y, margin_x:w - margin_x]

    # BGR → PIL → enhance
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil = enhance_frame(pil)

    # CLIP preprocessing + encode image
    tensor = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Cosine similarity against each class's averaged text features
    scores = {}
    for label, text_feat in class_text_features.items():
        sim = (image_features @ text_feat.unsqueeze(-1)).item()
        scores[label] = sim

    # Softmax over raw similarities for a confidence value
    labels = list(scores.keys())
    raw = torch.tensor([scores[l] for l in labels])
    probs = torch.softmax(raw * 100, dim=0)   # scale factor sharpens distribution

    best_idx = probs.argmax().item()
    best_label = labels[best_idx]
    confidence = probs[best_idx].item()

    return best_label, confidence, scores

# Drawing

def draw_overlay(frame, detections, fps, marker_centers):
    """
    detections    : list of dicts {bbox, label, confidence, color_name, scores}
                    bbox=None → single-shape bottom-bar fallback mode
    marker_centers: list of (cx, cy) for detected ArUco markers
    """
    h, w = frame.shape[:2]

    # Top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 48), (15, 15, 20), -1)
    cv2.addWeighted(bar, BG_ALPHA, frame, 1 - BG_ALPHA, 0, frame)
    cv2.putText(frame, "Shape Detector  [CLIP]", (12, 32), FONT, 0.72, (220, 220, 220), 1)
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
        cv2.putText(frame, "Point camera at a cube or sphere", (18, h - 14),
                    FONT, 0.48, (90, 90, 90), 1)
        return frame

    # Single shape with no bbox → bottom-bar display
    if len(confident) == 1 and confident[0]["bbox"] is None:
        det        = confident[0]
        label      = det["label"]
        conf       = det["confidence"]
        color_name = det["color_name"]
        scores     = det.get("scores", {})
        shape_col  = PALETTE.get(label, PALETTE["unknown"])
        info       = SHAPE_INFO.get(label, SHAPE_INFO["unknown"])

        margin_x = int(w * 0.20)
        margin_y = int(h * 0.20)
        cv2.rectangle(frame, (margin_x, margin_y), (w - margin_x, h - margin_y), shape_col, 1)

        bot = frame.copy()
        cv2.rectangle(bot, (0, h - 130), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(bot, BG_ALPHA + 0.1, frame, 1 - (BG_ALPHA + 0.1), 0, frame)

        cv2.putText(frame, label.upper(), (18, h - 98), FONT, 1.1, shape_col, 2)
        cv2.putText(frame, f"Color: {color_name.title()}",
                    (18, h - 72), FONT, 0.52, (200, 200, 200), 1)
        cv2.putText(frame,
                    f"Faces: {info['faces']}   Vertices: {info['vertices']}   Edges: {info['edges']}",
                    (18, h - 50), FONT, 0.46, (170, 170, 170), 1)

        bar_w = int((w - 36) * conf)
        cv2.rectangle(frame, (18, h - 32), (18 + bar_w, h - 18), shape_col, -1)
        cv2.rectangle(frame, (18, h - 32), (w - 18, h - 18), (80, 80, 80), 1)
        cv2.putText(frame, f"{conf:.0%}", (w - 65, h - 20),
                    FONT, 0.45, (200, 200, 200), 1)

        if scores:
            score_txt = "  ".join(f"{k}: {v:.2f}" for k, v in scores.items())
            cv2.putText(frame, score_txt, (18, h - 4), FONT, 0.38, (100, 100, 100), 1)

        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), shape_col, 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), shape_col, 1)
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
            (label.upper(),                                            shape_col,       0.65, 2),
            (f"Color: {color_name.title()}",                          (200, 200, 200), 0.46, 1),
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

def run_detector(camera_index, model, preprocess, class_text_features, device):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(
            f"❌ Cannot open camera index {camera_index}. "
            "Check System Settings → Privacy & Security → Camera."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"\n📷 Camera {camera_index} opened — press Q or Esc to quit\n")
    print("📋 Test cases covered:")
    print("   Cube  : solid bg single/multi, colored bg single/multi")
    print("   Sphere: solid bg single/multi, colored bg single/multi\n")

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

        # Run inference every 3rd frame (CLIP is slower than a custom model)
        if frame_count % 3 == 0:
            bboxes = find_shape_bboxes(frame)
            if bboxes:
                new_dets = []
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    lbl, conf, scores = predict(model, preprocess, class_text_features, roi, device)
                    color_name = get_dominant_color(frame, bbox)
                    new_dets.append({
                        "bbox":       bbox,
                        "label":      lbl,
                        "confidence": conf,
                        "color_name": color_name,
                        "scores":     scores,
                    })
                last_detections = new_dets
            else:
                lbl, conf, scores = predict(model, preprocess, class_text_features, frame, device)
                last_detections = [{
                    "bbox":       None,
                    "label":      lbl,
                    "confidence": conf,
                    "color_name": get_dominant_color(frame),
                    "scores":     scores,
                }]

        marker_centers = detect_aruco_markers(frame)

        if frame_count % 15 == 0:
            elapsed  = time.time() - t_start
            last_fps = 15 / max(elapsed, 1e-6)
            t_start  = time.time()

        frame = draw_overlay(frame, last_detections, last_fps, marker_centers)
        cv2.imshow("Shape Detector [CLIP]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Detector closed.\n")


def main():
    parser = argparse.ArgumentParser(description="CLIP Shape Detector — MacBook Camera")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--list-cameras", action="store_true")
    args = parser.parse_args()

    print("=" * 54)
    print("  🔷 Shape Detector  —  CLIP Zero-Shot Mode")
    print("=" * 54)

    cameras = scan_cameras(max_index=6)
    if not cameras:
        print("❌ No cameras found.")
        return

    print("\n📹 Available cameras:")
    for cam in cameras:
        print(f"   [{cam['index']}] {cam['name']}")

    if args.list_cameras:
        return

    if args.camera is not None:
        cam_index = args.camera
    else:
        cam_index = pick_builtin_camera(cameras)

    cam_name = next((c["name"] for c in cameras if c["index"] == cam_index), f"Camera {cam_index}")
    print(f"\n✅ Using: [{cam_index}] {cam_name}")

    device = get_device()
    print(f"🖥️  Device: {device}\n")

    model, preprocess, class_text_features = load_clip(device)
    run_detector(cam_index, model, preprocess, class_text_features, device)


if __name__ == "__main__":
    main()