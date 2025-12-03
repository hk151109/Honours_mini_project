#!/usr/bin/env python3
"""
WildfireNet Single Image Prediction Script (Hardcoded Inputs)

Just edit the variables in the CONFIG block and run:
    python predict_wildfirenet.py
"""

from pathlib import Path
import torch
from PIL import Image

# Import model and transforms from training script
from train_custom_cnn import WildfireNet, get_val_transforms, CONFIG as TRAIN_CONFIG


# ============================================================
# 🔧 HARD CODED INFERENCE CONFIG (EDIT ONLY THIS)
# ============================================================

PREDICT_CONFIG = {
    "checkpoint_path": "./CustomCNNOutput/models/wildfirenet_best_20250101_120000.pth",
    "image_path": "./test_images/sample.jpg",
    "device": "cpu",          # "cpu" or "cuda"
    "threshold": 0.5
}

LABELS = {
    0: "No Wildfire",
    1: "Wildfire"
}

# ============================================================
# CORE FUNCTIONS
# ============================================================

def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = ckpt.get("config", TRAIN_CONFIG)
    dropout = cfg.get("dropout", 0.4)

    model = WildfireNet(num_classes=1, dropout=dropout)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, cfg


def preprocess_image(image_path: Path, img_size: int, device: torch.device):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    transform = get_val_transforms(img_size)
    tensor = transform(img).unsqueeze(0)
    return tensor.to(device)


def predict_single_image():
    cfg = PREDICT_CONFIG

    checkpoint_path = Path(cfg["checkpoint_path"])
    image_path = Path(cfg["image_path"])
    threshold = cfg["threshold"]
    device_str = cfg["device"]

    if device_str == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, switching to CPU")
        device_str = "cpu"

    device = torch.device(device_str)

    print("=" * 60)
    print("WildfireNet Prediction Engine")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image: {image_path}")
    print(f"Threshold: {threshold}")
    print()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model
    model, train_cfg = load_model(checkpoint_path, device)
    img_size = train_cfg.get("input_size", 128)

    print(f"Model Loaded")
    print(f"Input Size: {img_size}x{img_size}")
    print(f"Trainable Params: {model.get_num_params():,}")
    print()

    # Preprocess image
    x = preprocess_image(image_path, img_size, device)

    # Forward pass
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    pred = int(prob >= threshold)
    label = LABELS[pred]

    print("Prediction Result")
    print("=" * 60)
    print(f"Probability of Wildfire: {prob:.4f}")
    print(f"Predicted Class: {pred}")
    print(f"Predicted Label: {label}")
    print("=" * 60)

    return {
        "image": str(image_path),
        "checkpoint": str(checkpoint_path),
        "probability": prob,
        "prediction": pred,
        "label": label,
        "threshold": threshold
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    results = predict_single_image()

    print("\nFinal Output Dictionary:")
    print(results)
