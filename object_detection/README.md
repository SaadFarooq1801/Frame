# Shape Detector — Custom Model Training & Detection

Real-time geometric shape detection using a custom-trained MobileNetV2 model and your MacBook's camera.

---

## Project Structure

```
object_detection/
├── training_data/
│   ├── cylinder/             ← Add your cylinder photos here
│   └── rectangular_prism/   ← Add your rectangular prism photos here
├── train_model.py            ← Step 1: Train the model on your photos
├── mac_detector.py           ← Step 2: Run live detection via MacBook camera
├── detector.py               ← Frame glasses version (kept intact)
├── shape_model.pth           ← Generated after training
├── class_labels.json         ← Generated after training
└── requirements.txt
```

---

## Setup

Activate the virtual environment:
```bash
cd object_detection
source test_venv/bin/activate
```

---

## Step 1 — Add Your Training Images

Drop your photos into the correct folders:

| Folder | What to put in it |
|--------|-------------------|
| `training_data/cylinder/` | Photos of cylinders (cans, bottles, cups…) |
| `training_data/rectangular_prism/` | Photos of rectangular prisms (boxes, books…) |

- Supported formats: `.jpg`, `.jpeg`, `.png`
- Any resolution is fine — they get resized automatically
- **4–5 images per class works** for prototype testing (heavy augmentation compensates)

---

## Step 2 — Train the Model

```bash
source test_venv/bin/activate
python train_model.py
```

Training runs for **60 epochs** using Apple Silicon MPS (GPU). With 4–5 images it takes ~2–5 minutes.

On completion you'll get:
- `shape_model.pth` — trained model weights
- `class_labels.json` — class index mapping

---

## Step 3 — Run Live Detection

```bash
source test_venv/bin/activate
python mac_detector.py
```

- Opens your MacBook's camera in a window
- Overlays the detected shape name + confidence bar
- Press **Q** or **Esc** to quit

---

## Tips for Better Accuracy (with few images)

- **Vary your angles** — shoot from above, below, the side
- **Vary backgrounds** — use different surfaces (table, floor, held in hand)
- **Vary lighting** — indoor, near window, different distances
- **Retrain** after adding more images — just re-run `train_model.py`

---

## Frame Glasses (original)

The original `detector.py` is untouched and still works for Frame glasses + Gemini detection:

```bash
python detector.py
```
