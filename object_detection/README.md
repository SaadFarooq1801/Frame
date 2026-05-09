# Shape Detector — Setup & Usage

Real-time geometric shape detection using a camera. Two detectors are available:

| File | How it works | Training required? |
|------|-------------|-------------------|
| `mac_detector.py` | Custom MobileNetV2 model trained on your own photos | Yes — run `train_model.py` first |
| `clip_detector.py` | OpenAI CLIP zero-shot (no model training) | No — works straight away |

> `detector.py` is the original Frame smart-glasses version and is unrelated to camera detection.

---

## 1 — Clone the repo

```bash
git clone <repo-url>
cd Frame/object_detection
```

---

## 2 — Create a virtual environment

**Windows**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3 — Install dependencies

### PyTorch (install this first)

**Windows — NVIDIA GPU (recommended for speed)**
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Windows — CPU only**
```cmd
pip install torch torchvision
```

**macOS (Apple Silicon — GPU auto-detected)**
```bash
pip install torch torchvision
```

### Everything else

```bash
pip install opencv-python pillow numpy
```

For `clip_detector.py` only:
```bash
pip install git+https://github.com/openai/CLIP.git
```

> `frame-msg` and `frame-ble` in `requirements.txt` are only needed for the Frame smart-glasses version (`detector.py`). Skip them for camera detection.

---

## Option A — clip_detector.py (easiest, no training)

Detects **cubes** and **spheres** using CLIP. No training needed.

```bash
python clip_detector.py
```

Flags:
```
python clip_detector.py --list-cameras   # see available cameras
python clip_detector.py --camera 1       # force a specific camera index
```

---

## Option B — mac_detector.py (custom shapes, needs training)

Detects whatever shapes you train it on (default: cylinder, rectangular prism).

### Step 1 — Add training images

Drop photos into the correct folders:

| Folder | Contents |
|--------|----------|
| `training_data/cylinder/` | Photos of cylinders (cans, bottles, cups…) |
| `training_data/rectangular_prism/` | Photos of rectangular prisms (boxes, books…) |

- Formats: `.jpg`, `.jpeg`, `.png`
- **4–5 images per class** is enough — the script uses heavy augmentation

### Step 2 — Train

```bash
python train_model.py
```

Produces `shape_model.pth` and `class_labels.json`. Takes 2–5 minutes.

### Step 3 — Run

```bash
python mac_detector.py
```

Flags:
```
python mac_detector.py --list-cameras   # see available cameras
python mac_detector.py --camera 1       # force a specific camera index
```

---

## Controls

| Key | Action |
|-----|--------|
| `Q` or `Esc` | Quit |

---

## What the overlay shows

- **Shape name** and **confidence bar**
- **Dominant color** of the detected object
- **Faces / vertices / edges** for the shape geometry
- **NEAREST TO MARKER** label when 2 shapes are visible and an ArUco marker is in frame

---

## Troubleshooting

**Camera not opening**
- Windows: Settings → Privacy & Security → Camera → allow the app
- macOS: System Settings → Privacy & Security → Camera
- Try `--list-cameras` to see which index to use

**Wrong camera selected**
```bash
python mac_detector.py --list-cameras
python mac_detector.py --camera 1
```

**CLIP install fails**
Make sure Git is installed (`git --version`), then retry the pip install command.

**Low accuracy on custom shapes**
Add more photos (10–20 per class), vary the angle and background, then retrain.
