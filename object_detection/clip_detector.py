"""
Uses OpenAI CLIP to classify shapes with no training and no API key.
Runs fully local on your Mac (Apple Silicon MPS accelerated).

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
import collections
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

#  Colours & Style 
PALETTE = {
    "cube":    (0, 174, 255),   # sky blue
    "sphere":  (0, 229, 160),   # mint green
    "unknown": (100, 100, 100), # grey
}
FONT     = cv2.FONT_HERSHEY_DUPLEX
BG_ALPHA = 0.55

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

def draw_overlay(frame, label, confidence, fps, smoothed_label, scores):
    h, w = frame.shape[:2]

    # Draw center crop guide
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.20)
    color = PALETTE.get(smoothed_label, PALETTE["unknown"])
    cv2.rectangle(frame, (margin_x, margin_y), (w - margin_x, h - margin_y), color, 1)

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (15, 15, 20), -1)
    cv2.addWeighted(overlay, BG_ALPHA, frame, 1 - BG_ALPHA, 0, frame)
    cv2.putText(frame, "Shape Detector  [CLIP]", (12, 32), FONT, 0.72, (220, 220, 220), 1)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 32), FONT, 0.65, (160, 160, 160), 1)

    # Bottom result bar
    bar_h = 110
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - bar_h), (w, h), (15, 15, 20), -1)
    cv2.addWeighted(overlay2, BG_ALPHA + 0.1, frame, 1 - (BG_ALPHA + 0.1), 0, frame)

    if confidence >= CONF_THRESH:
        display_name = smoothed_label.upper()
        cv2.putText(frame, display_name, (18, h - 64), FONT, 1.1, color, 2)

        # Confidence bar
        bar_w = int((w - 36) * confidence)
        cv2.rectangle(frame, (18, h - 40), (18 + bar_w, h - 24), color, -1)
        cv2.rectangle(frame, (18, h - 40), (w - 18, h - 24), (80, 80, 80), 1)
        cv2.putText(frame, f"{confidence:.0%}", (w - 65, h - 26), FONT, 0.55, (200, 200, 200), 1)

        # Per-class scores (small debug line)
        score_txt = "  ".join(f"{k}: {v:.2f}" for k, v in scores.items())
        cv2.putText(frame, score_txt, (18, h - 8), FONT, 0.42, (100, 100, 100), 1)
    else:
        cv2.putText(frame, "Scanning...", (18, h - 58), FONT, 0.9, (120, 120, 120), 1)
        cv2.putText(frame, "Point camera at a cube or sphere", (18, h - 28),
                    FONT, 0.48, (90, 90, 90), 1)

    # Crosshair
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), color, 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), color, 1)
    cv2.circle(frame, (cx, cy), 40, color, 1)

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

    pred_history = collections.deque(maxlen=SMOOTH_N)
    label, confidence = "unknown", 0.0
    scores = {}
    frame_count = 0
    t_start = time.time()
    last_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Camera read failed, retrying...")
            continue

        frame_count += 1

        # Run inference every 3rd frame (CLIP is slower than a custom model)
        if frame_count % 3 == 0:
            label, confidence, scores = predict(
                model, preprocess, class_text_features, frame, device
            )
            pred_history.append(label)

        smoothed = (
            collections.Counter(pred_history).most_common(1)[0][0]
            if pred_history else "unknown"
        )

        if frame_count % 15 == 0:
            elapsed  = time.time() - t_start
            last_fps = 15 / max(elapsed, 1e-6)
            t_start  = time.time()

        frame = draw_overlay(frame, label, confidence, last_fps, smoothed, scores)
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