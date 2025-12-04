#!/usr/bin/env python3
from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path
import torch
from PIL import Image
import os

# Import your model code
from train_custom_cnn import WildfireNet, get_val_transforms, CONFIG as TRAIN_CONFIG

# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------

# NEW PUBLIC FOLDER PATH
BASE_DIR = r"C:\Users\heemi\Downloads\enviro-meter"
SENTINEL_DIR = fr"{BASE_DIR}\public\sentinel"

CHECKPOINT_PATH = r"C:\Users\heemi\Downloads\enviro-meter\honours\wildfirenet_final_20251201_194548.pth"

THRESHOLD = 0.5
DEVICE = "cpu"

LABELS = {
    0: "No Wildfire",
    1: "Wildfire"
}

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)


# ----------------------------------------------------------------
# UTILS
# ----------------------------------------------------------------

def get_latest_truecolor_image() -> Path:
    """Finds the highest-numbered true-color-N.png file."""

    folder = Path(SENTINEL_DIR)
    if not folder.exists():
        raise FileNotFoundError(f"Sentinel folder does not exist: {folder}")

    files = list(folder.glob("true-color-*.png"))

    if not files:
        raise FileNotFoundError("No true-color images found in sentinel folder.")

    numbered = []
    for f in files:
        name = f.stem  # true-color-N
        try:
            n = int(name.replace("true-color-", ""))
            numbered.append((n, f))
        except:
            continue

    if not numbered:
        raise FileNotFoundError("No valid true-color-N.png files found.")

    latest_file = max(numbered, key=lambda x: x[0])[1]
    return latest_file


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


# ----------------------------------------------------------------
# FLASK API ROUTE
# ----------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict_latest_image():
    try:
        device = torch.device(DEVICE)

        data = request.get_json()  # get POSTed JSON
        image_url = data.get("image_url")

        if not image_url:
            return jsonify({"error": "No image_url provided"}), 400

        # Convert the URL/path to Path object
        img_path = Path(BASE_DIR, "public", *image_url.lstrip("/").split("/"))

        model, train_cfg = load_model(Path(CHECKPOINT_PATH), device)
        img_size = train_cfg.get("input_size", 128)

        x = preprocess_image(img_path, img_size, device)

        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).item()

        pred = int(prob >= THRESHOLD)
        label = LABELS[pred]

        return jsonify({
            "image_used": str(img_path),
            "checkpoint": CHECKPOINT_PATH,
            "probability": prob,
            "prediction": pred,
            "label": label,
            "threshold": THRESHOLD
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
