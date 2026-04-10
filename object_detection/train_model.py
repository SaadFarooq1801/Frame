"""
train_model.py — Custom Shape Classifier Training Script
---------------------------------------------------------
Trains a MobileNetV2 model on your images in training_data/
Uses heavy augmentation since we only have ~4-5 images per class.
Saves:
  - shape_model.pth      (trained model weights)
  - class_labels.json    (index → class name mapping)

Usage:
    python train_model.py

Put your images in:
    training_data/cylinder/           ← cylinder photos
    training_data/rectangular_prism/  ← rectangular prism photos
"""

import json
import os
import copy
import time
import ssl

# ── SSL fix for macOS Python (python.org installer doesn't link system certs) ──
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# ─── Config ────────────────────────────────────────────────────────────────────

DATA_DIR    = os.path.join(os.path.dirname(__file__), "training_data")
MODEL_OUT   = os.path.join(os.path.dirname(__file__), "shape_model.pth")
LABELS_OUT  = os.path.join(os.path.dirname(__file__), "class_labels.json")

IMG_SIZE    = 224       # MobileNetV2 expects 224×224
BATCH_SIZE  = 4        # Small batch since we have few images
EPOCHS      = 60       # More epochs to compensate for small dataset
LR          = 1e-4     # Low LR for fine-tuning pretrained weights
VAL_SPLIT   = 0.2     # 20% of data for validation (may be 0 if very few images)

# ─── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        print("🍎 Apple Silicon MPS detected — using GPU acceleration!")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("🟢 CUDA detected — using NVIDIA GPU!")
        return torch.device("cuda")
    else:
        print("🔵 Using CPU (no GPU available)")
        return torch.device("cpu")

# ─── Heavy Augmentation (critical for small datasets) ──────────────────────────

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # hide small patches
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Dataset ───────────────────────────────────────────────────────────────────

def load_dataset():
    if not os.path.isdir(DATA_DIR):
        print(f"❌ training_data/ folder not found at: {DATA_DIR}")
        raise SystemExit(1)

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    classes = full_dataset.classes
    n = len(full_dataset)

    print(f"\n📂 Found {n} images across {len(classes)} classes:")
    for i, c in enumerate(classes):
        count = sum(1 for _, label in full_dataset.samples if label == i)
        print(f"   [{i}] {c}: {count} images")

    if n == 0:
        print("❌ No images found! Add images to training_data/cylinder/ and training_data/rectangular_prism/")
        raise SystemExit(1)

    # With very few images, skip validation split to use all data for training
    val_count = max(0, int(n * VAL_SPLIT))
    val_count = min(val_count, n - len(classes))  # keep at least 1 per class in train
    train_count = n - val_count

    if val_count > 0:
        train_ds, val_ds = random_split(full_dataset, [train_count, val_count])
        # Apply val transforms to val split
        val_ds.dataset = copy.deepcopy(full_dataset)
        val_ds.dataset.transform = val_transforms
        print(f"\n✅ Split: {train_count} train / {val_count} validation")
        return train_ds, val_ds, classes
    else:
        print(f"\n✅ Using all {n} images for training (too few for validation split)")
        return full_dataset, None, classes

# ─── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 3 feature layers for fine-tuning
    for layer in list(model.features.children())[-3:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
    )

    return model

# ─── Training Loop ─────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, device, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print(f"\n🚀 Starting training for {epochs} epochs...\n")
    print("─" * 55)

    for epoch in range(epochs):
        t0 = time.time()

        # ── Train phase ──
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss     += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        train_acc  = running_corrects / len(train_loader.dataset)

        # ── Val phase ──
        val_info = ""
        if val_loader:
            model.eval()
            val_corrects = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    val_corrects += (preds == labels).sum().item()
            val_acc = val_corrects / len(val_loader.dataset)
            val_info = f"  |  Val Acc: {val_acc:.0%}"

            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        else:
            if train_acc > best_acc:
                best_acc = train_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        elapsed = time.time() - t0
        print(f"Epoch [{epoch+1:3d}/{epochs}]  "
              f"Loss: {train_loss:.4f}  "
              f"Train Acc: {train_acc:.0%}"
              f"{val_info}  "
              f"({elapsed:.1f}s)")

    print("─" * 55)
    print(f"\n✅ Best accuracy: {best_acc:.0%}")
    model.load_state_dict(best_model_wts)
    return model

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  🔷 Shape Classifier — Training Script")
    print("=" * 55)

    device = get_device()
    train_ds, val_ds, classes = load_dataset()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) if val_ds else None

    model = build_model(num_classes=len(classes)).to(device)
    model = train(model, train_loader, val_loader, device, EPOCHS)

    # ── Save model ──
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": len(classes),
        "img_size": IMG_SIZE,
    }, MODEL_OUT)
    print(f"\n💾 Model saved → {MODEL_OUT}")

    # ── Save labels ──
    label_map = {str(i): name for i, name in enumerate(classes)}
    with open(LABELS_OUT, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"📋 Labels saved → {LABELS_OUT}")

    print("\n🎉 Done! Run mac_detector.py to test with your camera.\n")


if __name__ == "__main__":
    main()
