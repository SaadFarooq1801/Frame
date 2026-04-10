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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

# ─── Config ────────────────────────────────────────────────────────────────────

DATA_DIR    = os.path.join(os.path.dirname(__file__), "training_data")
MODEL_OUT   = os.path.join(os.path.dirname(__file__), "shape_model.pth")
LABELS_OUT  = os.path.join(os.path.dirname(__file__), "class_labels.json")

IMG_SIZE    = 224   # MobileNetV2 expects 224×224
BATCH_SIZE  = 2    # Tiny batches — we only have ~13 images total
EPOCHS      = 100  # More passes over this tiny dataset
LR          = 3e-4 # Higher LR — only the small classifier head is trained

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

    dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    classes = dataset.classes
    n = len(dataset)

    print(f"\n📂 Found {n} images across {len(classes)} classes:")
    class_counts = []
    for i, c in enumerate(classes):
        count = sum(1 for _, label in dataset.samples if label == i)
        class_counts.append(count)
        print(f"   [{i}] {c}: {count} images")

    if n == 0:
        print("❌ No images found! Add images to training_data/ subfolders.")
        raise SystemExit(1)

    # ── Weighted sampler: each class gets equal probability per batch ──
    # This fixes the bias toward the majority class (rectangular_prism).
    sample_weights = [
        1.0 / class_counts[label] for _, label in dataset.samples
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=max(class_counts) * len(classes) * 4,  # virtual epoch size
        replacement=True
    )

    # ── Class weights for loss: inverse-frequency weighting ──
    max_count = max(class_counts)
    class_weights = torch.tensor(
        [max_count / c for c in class_counts], dtype=torch.float
    )
    print(f"\n⚖️  Class weights for loss: { {c: f'{w:.2f}' for c, w in zip(classes, class_weights.tolist())} }")
    print("✅ All images used for training (no val split — dataset too small)")

    return dataset, sampler, class_weights, classes

# ─── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # FREEZE THE ENTIRE BACKBONE.
    # With only ~13 images, updating backbone weights causes severe overfitting.
    # The ImageNet pretrained features are already excellent — we just need to
    # teach the classifier head to tell cylinders from rectangular prisms.
    for param in model.parameters():
        param.requires_grad = False

    # Replace & unfreeze only the classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )
    # classifier params are trainable by default (newly created)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Trainable parameters: {trainable:,} (classifier head only)")
    return model

# ─── Training Loop ─────────────────────────────────────────────────────────────

def train(model, train_loader, device, epochs, class_weights):
    # Weighted loss: heavily penalise wrong predictions on the minority class
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print(f"\n🚀 Training for {epochs} epochs...\n")
    print("─" * 55)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

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
            total            += inputs.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc  = running_corrects / total

        if train_acc > best_acc:
            best_acc = train_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        # Print every 10 epochs to keep output readable
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}]  "
                  f"Loss: {train_loss:.4f}  "
                  f"Train Acc: {train_acc:.0%}")

    print("─" * 55)
    print(f"\n✅ Best train accuracy: {best_acc:.0%}")
    model.load_state_dict(best_model_wts)
    return model

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  🔷 Shape Classifier — Training Script")
    print("=" * 55)

    device = get_device()
    dataset, sampler, class_weights, classes = load_dataset()

    # Use the balanced sampler (no shuffle — sampler handles ordering)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0
    )

    model = build_model(num_classes=len(classes)).to(device)
    model = train(model, train_loader, device, EPOCHS, class_weights)

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
