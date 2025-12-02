# train_firenet.py
# Trains FireNet-CNN only. First trains the MLP head (frozen conv base),
# then conditionally unfreezes the whole model for fine-tuning.
# Saves outputs to ./FireNetCNNOutput (models, plots, logs).
#
# Usage:
#   python train_firenet.py

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Activation, Flatten, Dense, Dropout, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
)
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report
)
import pandas as pd
import itertools
import argparse

# ---------------------------
# CONFIG - edit as needed
# ---------------------------

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

CONFIG = {
    "data_dir": "./wildfire-prediction-dataset",   # root folder that contains train/valid/test subfolders
    "output_dir": "./FireNetCNNOutput",            # required by user
    "input_size": (150, 150, 3),                   # FireNet paper uses 150x150
    "batch_size": 32,
    "mlp_epochs": 8,           # epochs to train only the MLP head (frozen conv base)
    "fine_tune_epochs": 30,    # epochs to fine-tune entire model (if triggered)
    "learning_rate": 1e-4,
    "fine_tune_lr": 1e-5,
    "patience": 5,
    "min_val_improvement": 0.005,  # if val acc improves by >= this after MLP training, may skip heavy fine-tune
    "use_mixed_precision": False,  # set True if GPU and TF mixed precision desired
    "class_mode": "binary"
}

# ---------------------------
# Create output directories
# ---------------------------
os.makedirs(CONFIG["output_dir"], exist_ok=True)
models_dir = os.path.join(CONFIG["output_dir"], "models")
plots_dir = os.path.join(CONFIG["output_dir"], "plots")
logs_dir = os.path.join(CONFIG["output_dir"], "logs")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Mixed precision if user wants and GPU available
if CONFIG["use_mixed_precision"] and tf.config.list_physical_devices("GPU"):
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision enabled.")

# ---------------------------
# Build FireNet-CNN (paper-based)
# ---------------------------
def build_firenet(input_shape=(150,150,3), dropout_rate=0.5):
    """
    FireNet-CNN as described in the paper:
    - 5 conv blocks with filters [32,64,128,256,512]
    - each block: Conv2D -> ReLU -> MaxPool -> BatchNorm
    - then Flatten -> Dense(512) + Dropout(0.5) -> Dense(256) + Dropout -> output sigmoid
    We structure it so we can freeze the conv 'base' and train the MLP head separately.
    """
    inp = Input(shape=input_shape, name="input_image")

    x = inp
    filters = [32, 64, 128, 256, 512]
    for i, f in enumerate(filters, start=1):
        x = Conv2D(f, (3,3), padding="same", name=f"conv{i}_1")(x)
        x = Activation("relu")(x)
        # optionally add second conv in block if desired (paper implies stronger conv blocks)
        x = Conv2D(f, (3,3), padding="same", name=f"conv{i}_2")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2,2), name=f"pool{i}")(x)
        x = BatchNormalization(name=f"bn{i}")(x)

    x = Flatten(name="flatten")(x)

    # MLP head (this is what we will train first with base frozen)
    x = Dense(512, activation="relu", name="fc1")(x)
    x = Dropout(dropout_rate, name="dropout_fc1")(x)
    x = Dense(256, activation="relu", name="fc2")(x)
    x = Dropout(dropout_rate, name="dropout_fc2")(x)

    out = Dense(1, activation="sigmoid", dtype="float32", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="FireNetCNN")
    return model

# ---------------------------
# Utilities: plotting + metrics
# ---------------------------
def plot_history(history, save_path):
    """Plot accuracy and loss curves from a Keras History object or dict of histories."""
    plt.figure(figsize=(10,6))
    # handle if history is dict from multiple stages
    if isinstance(history, dict):
        # assume history keys: 'accuracy','val_accuracy' etc (lists)
        acc = history.get("accuracy", [])
        val_acc = history.get("val_accuracy", [])
        loss = history.get("loss", [])
        val_loss = history.get("val_loss", [])
    else:
        acc = history.history.get("accuracy", [])
        val_acc = history.history.get("val_accuracy", [])
        loss = history.history.get("loss", [])
        val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(acc) + 1)
    plt.subplot(2,1,1)
    plt.plot(epochs, acc, 'b-', label="train_acc")
    plt.plot(epochs, val_acc, 'r-', label="val_acc")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(2,1,2)
    plt.plot(epochs, loss, 'b-', label="train_loss")
    plt.plot(epochs, val_loss, 'r-', label="val_loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_confusion(cm, classes, save_path, title="Confusion matrix"):
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_roc(y_true, y_score, save_path, label="FireNet"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return roc_auc

# ---------------------------
# Prepare Data Generators
# ---------------------------
def make_generators(config):
    train_dir = os.path.join(config["data_dir"], "train")
    valid_dir = os.path.join(config["data_dir"], "valid")
    test_dir  = os.path.join(config["data_dir"], "test")

    if not os.path.exists(train_dir) or not os.path.exists(valid_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError("train/valid/test directories not found under data_dir: {}".format(config["data_dir"]))

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='reflect'
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    target_size = config["input_size"][:2]
    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=target_size, batch_size=config["batch_size"],
        class_mode=config["class_mode"], shuffle=True
    )
    val_gen = val_test_datagen.flow_from_directory(
        valid_dir, target_size=target_size, batch_size=config["batch_size"],
        class_mode=config["class_mode"], shuffle=False
    )
    test_gen = val_test_datagen.flow_from_directory(
        test_dir, target_size=target_size, batch_size=config["batch_size"],
        class_mode=config["class_mode"], shuffle=False
    )
    return train_gen, val_gen, test_gen

# ---------------------------
# Training routine
# ---------------------------
def train_firenet(config):
    train_gen, val_gen, test_gen = make_generators(config)

    model = build_firenet(input_shape=config["input_size"], dropout_rate=0.5)
    model.summary()

    # Separate 'base' (conv blocks) and 'head' (dense layers) by layer names:
    # We'll freeze layers up to and including 'bn5' for MLP-only training.
    freeze_until_layer = "bn5"

    # Freeze conv base
    freeze = True
    for layer in model.layers:
        layer.trainable = False
        if layer.name == freeze_until_layer:
            break
    # Ensure head layers are trainable
    for layer in model.layers:
        if layer.name.startswith("fc") or layer.name.startswith("dropout") or layer.name == "output" or layer.name == "flatten":
            layer.trainable = True

    # Compile for MLP-only stage
    model.compile(optimizer=Adam(learning_rate=config["learning_rate"]),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    # callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_log_path = os.path.join(logs_dir, f"training_mlp_{timestamp}.csv")
    csv_logger = CSVLogger(csv_log_path)
    checkpoint_mlp = ModelCheckpoint(
        filepath=os.path.join(models_dir, f"firenet_mlp_best_{timestamp}.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=config["patience"], restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)

    print("\n=== Stage 1: Train MLP head (conv base frozen) ===")
    history_mlp = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config["mlp_epochs"],
        callbacks=[csv_logger, checkpoint_mlp, early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate validation metrics after MLP stage
    val_acc_after_mlp = history_mlp.history.get("val_accuracy", [0])[-1]
    print(f"\nValidation accuracy after MLP head training: {val_acc_after_mlp:.4f}")

    # Decide whether to unfreeze and fine-tune
    need_finetune = True
    # If validation accuracy improved a lot during MLP stage compared with baseline 0 (or if user chooses),
    # we can still choose to fine-tune for better generalization. We'll use min_val_improvement threshold:
    if val_acc_after_mlp >= config.get("min_val_improvement", 0.0):
        # this check is simplistic — it's mainly here so user can tweak config
        need_finetune = True

    # If not needed, still optionally save and evaluate
    combined_history = {}
    # combine history into dict (so plots will work)
    for k, v in history_mlp.history.items():
        combined_history[k] = v.copy()

    if need_finetune:
        print("\n=== Stage 2: Unfreeze conv base and fine-tune whole network ===")
        # Unfreeze all layers
        for layer in model.layers:
            layer.trainable = True

        # Recompile with lower LR for fine-tuning
        model.compile(optimizer=Adam(learning_rate=config["fine_tune_lr"]),
                      loss="binary_crossentropy",
                      metrics=["accuracy"])

        csv_log_path2 = os.path.join(logs_dir, f"training_finetune_{timestamp}.csv")
        csv_logger2 = CSVLogger(csv_log_path2)
        checkpoint_ft = ModelCheckpoint(
            filepath=os.path.join(models_dir, f"firenet_finetuned_best_{timestamp}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
        early_stop_ft = EarlyStopping(monitor="val_loss", patience=config["patience"], restore_best_weights=True, verbose=1)
        reduce_lr_ft = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-8, verbose=1)

        history_ft = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config["fine_tune_epochs"],
            callbacks=[csv_logger2, checkpoint_ft, early_stop_ft, reduce_lr_ft],
            verbose=1
        )

        # append second stage history to combined_history
        for k, v in history_ft.history.items():
            if k in combined_history:
                combined_history[k].extend(v)
            else:
                combined_history[k] = v

    # Save final model
    final_model_path = os.path.join(models_dir, f"firenet_final_{timestamp}.keras")
    model.save(final_model_path)
    print(f"\nSaved final model: {final_model_path}")

    # Save combined history to CSV for inspection
    hist_df = pd.DataFrame(combined_history)
    hist_csv_path = os.path.join(logs_dir, f"history_combined_{timestamp}.csv")
    hist_df.to_csv(hist_csv_path, index=False)
    print(f"Saved training history CSV: {hist_csv_path}")

    # Plot training curves
    plot_history(combined_history, os.path.join(plots_dir, f"training_curves_{timestamp}.png"))
    print(f"Saved training plot to {plots_dir}")

    # ---------------------------
    # Evaluate on test set
    # ---------------------------
    print("\n=== Evaluating on test set ===")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"Test loss: {test_loss:.4f}  Test acc: {test_acc:.4f}")

    # Predictions and metrics
    y_true = test_gen.classes
    y_score = model.predict(test_gen, verbose=1).ravel()
    y_pred = (y_score >= 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_indices = test_gen.class_indices
    # create ordered class list (flow_from_directory sorting)
    inv_map = {v:k for k,v in class_indices.items()}
    class_names = [inv_map[i] for i in range(len(inv_map))]

    # save confusion plot
    plot_confusion(cm, class_names, os.path.join(plots_dir, f"confusion_{timestamp}.png"))
    print(f"Saved confusion matrix plot to {plots_dir}")

    # classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_csv = os.path.join(logs_dir, f"classification_report_{timestamp}.csv")
    report_df.to_csv(report_csv)
    print(f"Saved classification report to {report_csv}")
    # print quick summary
    print("\nClassification Report (test):")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ROC
    roc_auc = plot_roc(y_true, y_score, os.path.join(plots_dir, f"roc_{timestamp}.png"))
    print(f"Saved ROC plot to {plots_dir} (AUC={roc_auc:.4f})")

    # Save test metrics summary to CSV
    results_summary = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "roc_auc": roc_auc
    }
    pd.DataFrame([results_summary]).to_csv(os.path.join(logs_dir, f"test_summary_{timestamp}.csv"), index=False)
    print(f"Saved test summary to {logs_dir}")

    return {
        "model_path": final_model_path,
        "history_csv": hist_csv_path,
        "training_plot": os.path.join(plots_dir, f"training_curves_{timestamp}.png"),
        "confusion_plot": os.path.join(plots_dir, f"confusion_{timestamp}.png"),
        "roc_plot": os.path.join(plots_dir, f"roc_{timestamp}.png"),
        "classification_report": report_csv,
        "test_summary_csv": os.path.join(logs_dir, f"test_summary_{timestamp}.csv")
    }


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    # optional: allow overriding data_dir from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=CONFIG["data_dir"], help="Path to dataset root (train/valid/test)")
    args = parser.parse_args()
    CONFIG["data_dir"] = args.data_dir

    print("Starting FireNet training. Outputs will be saved to:", CONFIG["output_dir"])
    out = train_firenet(CONFIG)
    print("\nAll outputs saved. Files:")
    for k, v in out.items():
        print(f" - {k}: {v}")
