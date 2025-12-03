#!/usr/bin/env python3
"""
================================================================================
Wildfire Detection Model Evaluation & Report Generation
================================================================================

This script evaluates all trained wildfire detection models and generates
publication-quality plots suitable for academic reports and presentations.

Generated Outputs:
------------------
For each model:
    - Confusion matrix (counts and normalized)
    - ROC curve with AUC
    - Precision-Recall curve with AP
    - Per-class metrics bar chart
    - Classification report

Comparative Analysis:
    - All models ROC curves overlay
    - All models PR curves overlay
    - Model accuracy comparison
    - Model AUC comparison
    - Comprehensive summary dashboard

Run with: python generate_test_plots.py

Author: Wildfire Detection Project
Date: December 2024
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

# TensorFlow/Keras
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# PyTorch (for WildfireNet)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Metrics
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Robust PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "data_dir": "./wildfire-prediction-dataset",
    "output_dir": "./final_results",
    "batch_size": 32,
    "input_size_default": 224,
    "input_size_pytorch": 128,
    "workers": 0,  # For CPU
    "cache_file": "./final_results/evaluation_cache.json",  # Cache for resume capability
}

# Model paths - Keras models with their expected input sizes
# Format: {"model_name": {"path": "...", "input_size": 224}}
KERAS_MODELS = {
    # Pretrained models (frozen head only) - all use 224x224
    # Note: ResNet18/34 don't exist in keras.applications, only ResNet50/101/152
    "VGG16": {"path": "output/models/VGG16_best.keras", "input_size": 224},
    "VGG19": {"path": "output/models/VGG19_best.keras", "input_size": 224},
    "ResNet50": {"path": "output/models/ResNet50_best.keras", "input_size": 224},
    "ResNet50V2": {"path": "output/models/ResNet50V2_best.keras", "input_size": 224},
    "MobileNetV3Small": {"path": "output/models/MobileNetV3Small_best.keras", "input_size": 224},
    "MobileNetV3Large": {"path": "output/models/MobileNetV3Large_best.keras", "input_size": 224},
    "EfficientNetB0": {"path": "output/models/EfficientNetB0_best.keras", "input_size": 224},
    
    # Fine-tuned models
    "VGG16_finetuned": {"path": "output/models/VGG16_finetuned_final.keras", "input_size": 224},
    "VGG19_finetuned": {"path": "output/models/VGG19_finetuned_final.keras", "input_size": 224},
    "ResNet50V2_finetuned": {"path": "output/models/ResNet50V2_finetuned_final.keras", "input_size": 224},
    
    # FireNet - uses 150x150
    "FireNet": {"path": "FireNetCNNOutput/models/firenet_final_20251201_134936.keras", "input_size": 150},
}

# PyTorch models
PYTORCH_MODELS = {
    "WildfireNet": {"path": "CustomCNNOutput/models/wildfirenet_best_20251201_194548.pth", "input_size": 128},
}


# ============================================================
# WildfireNet Architecture (needed for loading)
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = (avg_out + max_out).view(b, c, 1, 1)
        return x * attention


class WildfireNet(nn.Module):
    def __init__(self, num_classes=1, dropout=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.block1 = ConvBlock(32, 64, stride=2)
        self.attn1 = ChannelAttention(64)
        self.block2 = ConvBlock(64, 128, stride=2)
        self.attn2 = ChannelAttention(128)
        self.block3 = ConvBlock(128, 256, stride=2)
        self.attn3 = ChannelAttention(256)
        self.spatial3 = SpatialAttention()
        self.block4 = ConvBlock(256, 512, stride=2)
        self.attn4 = ChannelAttention(512)
        self.spatial4 = SpatialAttention()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.attn1(x)
        x = self.block2(x)
        x = self.attn2(x)
        x = self.block3(x)
        x = self.attn3(x)
        x = self.spatial3(x)
        x = self.block4(x)
        x = self.attn4(x)
        x = self.spatial4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# ============================================================
# UTILITIES
# ============================================================

def makedirs(p):
    os.makedirs(p, exist_ok=True)


def set_plot_style():
    """Set publication-quality plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'font.family': 'serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


# ============================================================
# CACHE MANAGEMENT - Resume capability
# ============================================================

def load_cache():
    """Load cached evaluation results from JSON file."""
    cache_path = Path(CONFIG["cache_file"])
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            print(f"  ✓ Loaded cache with {len(cache.get('results', {}))} evaluated models")
            return cache
        except Exception as e:
            print(f"  ⚠ Could not load cache: {e}")
    return {"results": {}, "completed_plots": []}


def save_cache(cache):
    """Save evaluation results to JSON cache file."""
    cache_path = Path(CONFIG["cache_file"])
    makedirs(cache_path.parent)
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"  ⚠ Could not save cache: {e}")


def cache_result(cache, model_name, result):
    """Cache a single model's evaluation result."""
    # Convert numpy arrays to lists for JSON serialization
    cache["results"][model_name] = {
        "name": result["name"],
        "y_true": result["y_true"].tolist() if hasattr(result["y_true"], 'tolist') else result["y_true"],
        "y_pred": result["y_pred"].tolist() if hasattr(result["y_pred"], 'tolist') else result["y_pred"],
        "y_score": result["y_score"].tolist() if hasattr(result["y_score"], 'tolist') else result["y_score"],
        "accuracy": float(result["accuracy"]),
        "loss": float(result["loss"])
    }
    save_cache(cache)


def get_cached_result(cache, model_name):
    """Get a cached result, converting lists back to numpy arrays."""
    if model_name in cache.get("results", {}):
        cached = cache["results"][model_name]
        return {
            "name": cached["name"],
            "y_true": np.array(cached["y_true"]),
            "y_pred": np.array(cached["y_pred"]),
            "y_score": np.array(cached["y_score"]),
            "accuracy": cached["accuracy"],
            "loss": cached["loss"]
        }
    return None


def mark_plots_completed(cache, model_name):
    """Mark a model's plots as completed."""
    if model_name not in cache.get("completed_plots", []):
        cache.setdefault("completed_plots", []).append(model_name)
        save_cache(cache)


def is_model_completed(cache, model_name):
    """Check if a model has been fully processed (evaluated + plots generated)."""
    return model_name in cache.get("completed_plots", [])


# ============================================================
# DATA LOADING
# ============================================================

# Cache for test generators to avoid recreating them
_test_gen_cache = {}

def get_keras_test_generator(input_size):
    """Create Keras test data generator for specific input size."""
    if input_size in _test_gen_cache:
        gen = _test_gen_cache[input_size]
        gen.reset()
        return gen
    
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        os.path.join(CONFIG["data_dir"], "test"),
        target_size=(input_size, input_size),
        batch_size=CONFIG["batch_size"],
        class_mode="binary",
        shuffle=False
    )
    _test_gen_cache[input_size] = test_gen
    return test_gen


def get_pytorch_test_loader(input_size):
    """Create PyTorch test data loader."""
    transform = Compose([
        Resize((input_size, input_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_ds = datasets.ImageFolder(
        os.path.join(CONFIG["data_dir"], "test"),
        transform=transform
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["workers"]
    )
    
    return test_loader, test_ds.classes


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_keras_model(model_path, model_name, test_gen):
    """Evaluate a Keras model and return predictions."""
    print(f"  Loading {model_name}...")
    
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"  ✗ Error loading {model_name}: {e}")
        return None
    
    print(f"  Evaluating {model_name}...")
    
    # Get predictions
    test_gen.reset()
    y_pred_prob = model.predict(test_gen, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).ravel()
    y_true = test_gen.classes
    
    # Calculate metrics
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    
    return {
        "name": model_name,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_pred_prob.ravel(),
        "accuracy": test_acc,
        "loss": test_loss
    }


def evaluate_pytorch_model(model_path, model_name, test_loader, device):
    """Evaluate a PyTorch model and return predictions."""
    print(f"  Loading {model_name}...")
    
    try:
        model = WildfireNet(num_classes=1, dropout=0.4)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"  ✗ Error loading {model_name}: {e}")
        return None
    
    print(f"  Evaluating {model_name}...")
    
    all_preds = []
    all_scores = []
    all_true = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            scores = torch.sigmoid(outputs).cpu().numpy().ravel()
            preds = (scores >= 0.5).astype(int)
            
            all_scores.extend(scores.tolist())
            all_preds.extend(preds.tolist())
            all_true.extend(labels.numpy().tolist())
    
    y_true = np.array(all_true)
    y_pred = np.array(all_preds)
    y_score = np.array(all_scores)
    
    accuracy = np.mean(y_pred == y_true)
    
    return {
        "name": model_name,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
        "accuracy": accuracy,
        "loss": 0.0
    }


# ============================================================
# PLOT GENERATORS
# ============================================================

def plot_confusion_matrix_single(y_true, y_pred, model_name, classes, output_dir):
    """
    Generate comprehensive confusion matrix plots for a single model.
    Creates both count-based and normalized confusion matrices.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=axes[0],
                annot_kws={'size': 14, 'weight': 'bold'},
                cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold', pad=10)
    axes[0].tick_params(axis='both', labelsize=11)
    
    # Plot 2: Normalized percentages
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=axes[1],
                annot_kws={'size': 14, 'weight': 'bold'},
                cbar_kws={'label': 'Percentage (%)'})
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1].set_title('Confusion Matrix (Normalized %)', fontsize=14, fontweight='bold', pad=10)
    axes[1].tick_params(axis='both', labelsize=11)
    
    # Calculate accuracy for title
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    fig.suptitle(f'{model_name} - Confusion Matrix Analysis\n(Test Accuracy: {accuracy:.2f}%)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    safe_name = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    plt.savefig(output_dir / f'{safe_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve_single(y_true, y_score, model_name, output_dir):
    """
    Generate ROC curve plot for a single model with detailed annotations.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='#2196F3', lw=3, 
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
            label='Random Classifier (AUC = 0.5)')
    
    # Mark optimal point
    ax.scatter([optimal_fpr], [optimal_tpr], color='red', s=150, zorder=5,
               marker='*', label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    # Add annotation for optimal point
    ax.annotate(f'TPR={optimal_tpr:.3f}\nFPR={optimal_fpr:.3f}',
                xy=(optimal_fpr, optimal_tpr),
                xytext=(optimal_fpr + 0.15, optimal_tpr - 0.1),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red'))
    
    # Fill area under curve
    ax.fill_between(fpr, tpr, alpha=0.2, color='#2196F3')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}\nReceiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add AUC text box
    textstr = f'AUC = {roc_auc:.4f}'
    props = dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#2196F3', alpha=0.9)
    ax.text(0.6, 0.15, textstr, transform=ax.transAxes, fontsize=14,
            fontweight='bold', verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    safe_name = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    plt.savefig(output_dir / f'{safe_name}_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc


def plot_pr_curve_single(y_true, y_score, model_name, output_dir):
    """
    Generate Precision-Recall curve for a single model.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    # Baseline (proportion of positive class)
    baseline = np.sum(y_true) / len(y_true)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot PR curve
    ax.plot(recall, precision, color='#4CAF50', lw=3,
            label=f'Precision-Recall Curve (AP = {ap:.4f})')
    
    # Plot baseline
    ax.axhline(y=baseline, color='gray', lw=2, linestyle='--',
               label=f'Baseline (Positive Rate = {baseline:.3f})')
    
    # Fill area under curve
    ax.fill_between(recall, precision, alpha=0.2, color='#4CAF50')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}\nPrecision-Recall Curve', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add AP text box
    textstr = f'Average Precision = {ap:.4f}'
    props = dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50', alpha=0.9)
    ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            fontweight='bold', verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    safe_name = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    plt.savefig(output_dir / f'{safe_name}_pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return ap


def plot_per_class_metrics_single(y_true, y_pred, model_name, classes, output_dir):
    """
    Generate per-class metrics bar chart for a single model.
    """
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    precisions = [report[c]['precision'] for c in classes]
    recalls = [report[c]['recall'] for c in classes]
    f1_scores = [report[c]['f1-score'] for c in classes]
    supports = [report[c]['support'] for c in classes]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x = np.arange(len(classes))
    width = 0.25
    
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#2196F3', edgecolor='white')
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#4CAF50', edgecolor='white')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#FF9800', edgecolor='white')
    
    # Add value labels on bars
    def add_labels(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_labels(bars1, precisions)
    add_labels(bars2, recalls)
    add_labels(bars3, f1_scores)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}\nPer-Class Performance Metrics', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}\n(n={s})' for c, s in zip(classes, supports)], fontsize=11)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add macro averages as text
    macro_p = report['macro avg']['precision']
    macro_r = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']
    
    textstr = f'Macro Avg:  Precision={macro_p:.3f}  |  Recall={macro_r:.3f}  |  F1={macro_f1:.3f}'
    ax.text(0.5, 1.08, textstr, transform=ax.transAxes, fontsize=11,
            ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#ECEFF1', edgecolor='gray'))
    
    plt.tight_layout()
    
    safe_name = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    plt.savefig(output_dir / f'{safe_name}_per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_summary_single(result, classes, output_dir):
    """
    Generate a comprehensive summary plot for a single model.
    """
    model_name = result["name"]
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    y_score = result["y_score"]
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax1,
                annot_kws={'size': 12, 'weight': 'bold'},
                cbar_kws={'label': '%'})
    ax1.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True', fontsize=11, fontweight='bold')
    ax1.set_title('Confusion Matrix (%)', fontsize=12, fontweight='bold')
    
    # 2. ROC Curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fpr, tpr, color='#2196F3', lw=2.5, label=f'AUC = {roc_auc:.4f}')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax2.fill_between(fpr, tpr, alpha=0.15, color='#2196F3')
    ax2.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax2.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(recall_curve, precision_curve, color='#4CAF50', lw=2.5, label=f'AP = {ap:.4f}')
    ax3.fill_between(recall_curve, precision_curve, alpha=0.15, color='#4CAF50')
    ax3.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax3.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Per-Class Metrics
    ax4 = fig.add_subplot(gs[1, 0:2])
    x = np.arange(len(classes))
    width = 0.25
    
    precisions = [report[c]['precision'] for c in classes]
    recalls = [report[c]['recall'] for c in classes]
    f1_scores = [report[c]['f1-score'] for c in classes]
    
    bars1 = ax4.bar(x - width, precisions, width, label='Precision', color='#2196F3')
    bars2 = ax4.bar(x, recalls, width, label='Recall', color='#4CAF50')
    bars3 = ax4.bar(x + width, f1_scores, width, label='F1-Score', color='#FF9800')
    
    ax4.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax4.set_title('Per-Class Performance Metrics', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(classes, fontsize=10)
    ax4.legend(loc='lower right', fontsize=10)
    ax4.set_ylim([0, 1.1])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Summary Statistics
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    accuracy = result["accuracy"]
    
    summary_text = f"""
    ╔══════════════════════════════════╗
    ║     MODEL PERFORMANCE SUMMARY    ║
    ╠══════════════════════════════════╣
    ║                                  ║
    ║  Accuracy:     {accuracy*100:>6.2f}%           ║
    ║  AUC Score:    {roc_auc:>6.4f}            ║
    ║  Avg Precision:{ap:>6.4f}            ║
    ║                                  ║
    ║  Macro Precision: {report['macro avg']['precision']:>6.4f}        ║
    ║  Macro Recall:    {report['macro avg']['recall']:>6.4f}        ║
    ║  Macro F1-Score:  {report['macro avg']['f1-score']:>6.4f}        ║
    ║                                  ║
    ║  Test Samples: {len(y_true):>6}            ║
    ╚══════════════════════════════════╝
    """
    
    ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes, fontsize=11,
             fontfamily='monospace', verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='#333'))
    
    fig.suptitle(f'{model_name} - Complete Evaluation Summary', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    safe_name = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    plt.savefig(output_dir / f'{safe_name}_complete_summary.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices_grid(results, classes, output_dir):
    """
    Plot confusion matrices for all models in a grid layout.
    Suitable for model comparison in reports.
    """
    n_models = len(results)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten() if n_models > 1 else [axes]
    
    for idx, result in enumerate(results):
        cm = confusion_matrix(result["y_true"], result["y_pred"])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, ax=axes[idx],
                    annot_kws={'size': 11, 'weight': 'bold'},
                    cbar_kws={'label': '%'})
        axes[idx].set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=10, fontweight='bold')
        axes[idx].set_title(f"{result['name']}\nAccuracy: {result['accuracy']*100:.2f}%", 
                           fontsize=11, fontweight='bold', pad=10)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Confusion Matrix Comparison - All Models', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves_comparison(results, output_dir):
    """
    Plot ROC curves for all models overlaid for comparison.
    Publication-quality with detailed legend.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use a colormap for distinguishing models
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Sort by AUC for legend ordering
    results_with_auc = []
    for result in results:
        fpr, tpr, _ = roc_curve(result["y_true"], result["y_score"])
        roc_auc = auc(fpr, tpr)
        results_with_auc.append((result, fpr, tpr, roc_auc))
    
    results_with_auc.sort(key=lambda x: x[3], reverse=True)
    
    for (result, fpr, tpr, roc_auc), color in zip(results_with_auc, colors):
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{result['name']} (AUC = {roc_auc:.4f})")
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curve Comparison - All Models\nWildfire Detection Performance', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95,
              title='Models (sorted by AUC)', title_fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.text(0.55, 0.05, 'Higher AUC = Better Performance', 
            transform=ax.transAxes, fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curves_comparison(results, output_dir):
    """
    Plot Precision-Recall curves for all models overlaid.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Sort by AP for legend ordering
    results_with_ap = []
    for result in results:
        precision, recall, _ = precision_recall_curve(result["y_true"], result["y_score"])
        ap = average_precision_score(result["y_true"], result["y_score"])
        results_with_ap.append((result, precision, recall, ap))
    
    results_with_ap.sort(key=lambda x: x[3], reverse=True)
    
    for (result, precision, recall, ap), color in zip(results_with_ap, colors):
        ax.plot(recall, precision, color=color, lw=2.5,
                label=f"{result['name']} (AP = {ap:.4f})")
    
    # Baseline
    baseline = np.mean([np.mean(r["y_true"]) for r in results])
    ax.axhline(y=baseline, color='gray', lw=2, linestyle='--', 
               label=f'Baseline (Positive Rate ≈ {baseline:.3f})')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=13, fontweight='bold')
    ax.set_title('Precision-Recall Curve Comparison - All Models\nWildfire Detection Performance', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95,
              title='Models (sorted by AP)', title_fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_comparison(results, output_dir):
    """
    Generate horizontal bar chart comparing model accuracies.
    """
    # Sort by accuracy
    results_sorted = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    
    model_names = [r["name"] for r in results_sorted]
    accuracies = [r["accuracy"] * 100 for r in results_sorted]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(results) * 0.6)))
    
    # Color gradient based on accuracy
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(accuracies)))
    
    bars = ax.barh(model_names, accuracies, color=colors, edgecolor='white', height=0.7)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(acc + 0.3, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Model Accuracy Comparison\nWildfire Detection - Test Set Performance', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim([min(accuracies) - 5, 102])
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Highlight best model
    bars[0].set_edgecolor('#2E7D32')
    bars[0].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_accuracy_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_auc_comparison(results, output_dir):
    """
    Generate horizontal bar chart comparing model AUC scores.
    """
    # Calculate AUC for each model
    results_with_auc = []
    for r in results:
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_score"])
        roc_auc = auc(fpr, tpr)
        results_with_auc.append((r["name"], roc_auc, r))
    
    # Sort by AUC
    results_with_auc.sort(key=lambda x: x[1], reverse=True)
    
    model_names = [r[0] for r in results_with_auc]
    aucs = [r[1] for r in results_with_auc]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(results) * 0.6)))
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(aucs)))
    
    bars = ax.barh(model_names, aucs, color=colors, edgecolor='white', height=0.7)
    
    for bar, auc_val in zip(bars, aucs):
        ax.text(auc_val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{auc_val:.4f}', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Area Under ROC Curve (AUC)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Model AUC Comparison\nWildfire Detection - Discrimination Ability', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim([min(aucs) - 0.05, 1.02])
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Highlight best model
    bars[0].set_edgecolor('#2E7D32')
    bars[0].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_auc_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_comprehensive_comparison(results, output_dir):
    """
    Generate a comprehensive comparison chart showing multiple metrics.
    """
    fig, ax = plt.subplots(figsize=(14, max(8, len(results) * 0.7)))
    
    # Calculate all metrics
    data = []
    for r in results:
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_score"])
        roc_auc = auc(fpr, tpr)
        ap = average_precision_score(r["y_true"], r["y_score"])
        f1 = f1_score(r["y_true"], r["y_pred"])
        prec = precision_score(r["y_true"], r["y_pred"])
        rec = recall_score(r["y_true"], r["y_pred"])
        
        data.append({
            'name': r['name'],
            'accuracy': r['accuracy'],
            'auc': roc_auc,
            'ap': ap,
            'f1': f1,
            'precision': prec,
            'recall': rec
        })
    
    # Sort by accuracy
    data.sort(key=lambda x: x['accuracy'], reverse=True)
    
    model_names = [d['name'] for d in data]
    y_pos = np.arange(len(model_names))
    
    # Create grouped bars
    metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall']
    metric_labels = ['Accuracy', 'AUC', 'F1-Score', 'Precision', 'Recall']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    
    bar_width = 0.15
    offsets = np.arange(len(metrics)) - (len(metrics) - 1) / 2
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [d[metric] for d in data]
        bars = ax.barh(y_pos + offsets[i] * bar_width, values, bar_width, 
                       label=label, color=color, alpha=0.85)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names, fontsize=11)
    ax.set_xlabel('Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Comprehensive Model Comparison\nAll Performance Metrics', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.set_xlim([0.5, 1.05])
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary_dashboard(results, classes, output_dir):
    """
    Create a comprehensive summary dashboard with all key visualizations.
    This is the main summary figure for reports.
    """
    results_sorted = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
    
    # 1. Accuracy Ranking (left column, top)
    ax1 = fig.add_subplot(gs[0, 0])
    model_names = [r["name"][:15] + '...' if len(r["name"]) > 15 else r["name"] 
                   for r in results_sorted[:8]]
    accuracies = [r["accuracy"] * 100 for r in results_sorted[:8]]
    colors = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(accuracies)))
    bars = ax1.barh(model_names, accuracies, color=colors, height=0.6)
    ax1.set_xlabel('Accuracy (%)', fontsize=10, fontweight='bold')
    ax1.set_title('Top Models by Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlim([min(accuracies) - 3, 100])
    for bar, acc in zip(bars, accuracies):
        ax1.text(acc + 0.3, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', fontsize=9, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # 2. ROC Curves (top middle, spans 2 columns)
    ax2 = fig.add_subplot(gs[0, 1:3])
    colors_roc = plt.cm.tab10(np.linspace(0, 1, min(len(results_sorted), 6)))
    for result, color in zip(results_sorted[:6], colors_roc):
        fpr, tpr, _ = roc_curve(result["y_true"], result["y_score"])
        roc_auc = auc(fpr, tpr)
        short_name = result['name'][:12] + '..' if len(result['name']) > 12 else result['name']
        ax2.plot(fpr, tpr, color=color, lw=2, label=f"{short_name} ({roc_auc:.3f})")
    ax2.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    ax2.set_xlabel('False Positive Rate', fontsize=10, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=10, fontweight='bold')
    ax2.set_title('ROC Curves (Top 6 Models)', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Best Model Stats (top right)
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis('off')
    best = results_sorted[0]
    fpr, tpr, _ = roc_curve(best["y_true"], best["y_score"])
    best_auc = auc(fpr, tpr)
    best_ap = average_precision_score(best["y_true"], best["y_score"])
    report = classification_report(best["y_true"], best["y_pred"], 
                                   target_names=classes, output_dict=True)
    
    stats_text = f"""
    ╔═══════════════════════╗
    ║    BEST MODEL         ║
    ╠═══════════════════════╣
    ║ {best['name'][:20]:<20} ║
    ╠═══════════════════════╣
    ║ Accuracy:   {best['accuracy']*100:>6.2f}%  ║
    ║ AUC:        {best_auc:>6.4f}   ║
    ║ Avg Prec:   {best_ap:>6.4f}   ║
    ║ F1 (macro): {report['macro avg']['f1-score']:>6.4f}   ║
    ╚═══════════════════════╝
    """
    ax3.text(0.5, 0.5, stats_text, transform=ax3.transAxes, fontsize=10,
             fontfamily='monospace', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50'))
    
    # 4. Best Model Confusion Matrix (middle left)
    ax4 = fig.add_subplot(gs[1, 0:2])
    cm = confusion_matrix(best["y_true"], best["y_pred"])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax4,
                annot_kws={'size': 14, 'weight': 'bold'},
                cbar_kws={'label': '%'})
    ax4.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax4.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax4.set_title(f'Best Model: {best["name"]} - Confusion Matrix', 
                  fontsize=12, fontweight='bold')
    
    # 5. AUC Comparison (middle right, spans 2 columns)
    ax5 = fig.add_subplot(gs[1, 2:4])
    aucs = []
    for r in results_sorted:
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_score"])
        aucs.append(auc(fpr, tpr))
    
    short_names = [r["name"][:12] + '..' if len(r["name"]) > 12 else r["name"] 
                   for r in results_sorted]
    colors_auc = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(aucs)))
    bars = ax5.barh(short_names, aucs, color=colors_auc, height=0.6)
    ax5.set_xlabel('AUC Score', fontsize=11, fontweight='bold')
    ax5.set_title('AUC Comparison - All Models', fontsize=12, fontweight='bold')
    ax5.set_xlim([min(aucs) - 0.03, 1.02])
    for bar, auc_val in zip(bars, aucs):
        ax5.text(auc_val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{auc_val:.3f}', va='center', fontsize=9, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.invert_yaxis()
    
    # 6. Comprehensive Metrics Table (bottom, spans all columns)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Create table data
    table_data = []
    for r in results_sorted:
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_score"])
        roc_auc = auc(fpr, tpr)
        report = classification_report(r["y_true"], r["y_pred"],
                                       target_names=classes, output_dict=True)
        
        table_data.append([
            r["name"][:25],
            f"{r['accuracy']*100:.2f}%",
            f"{roc_auc:.4f}",
            f"{report['macro avg']['precision']:.4f}",
            f"{report['macro avg']['recall']:.4f}",
            f"{report['macro avg']['f1-score']:.4f}"
        ])
    
    table = ax6.table(
        cellText=table_data,
        colLabels=['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1-Score'],
        loc='center',
        cellLoc='center',
        colColours=['#E3F2FD'] * 6
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Highlight best row
    for j in range(6):
        table[(1, j)].set_facecolor('#C8E6C9')
        table[(1, j)].set_text_props(fontweight='bold')
    
    fig.suptitle('Wildfire Detection - Complete Model Evaluation Summary', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_results_csv(results, classes, output_path):
    """Save evaluation results to CSV."""
    rows = []
    for r in results:
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_score"])
        roc_auc = auc(fpr, tpr)
        report = classification_report(r["y_true"], r["y_pred"],
                                       target_names=classes, output_dict=True)
        
        rows.append({
            "Model": r["name"],
            "Test Accuracy (%)": r["accuracy"] * 100,
            "Test Loss": r["loss"],
            "AUC": roc_auc,
            "Precision (Macro)": report['macro avg']['precision'],
            "Recall (Macro)": report['macro avg']['recall'],
            "F1-Score (Macro)": report['macro avg']['f1-score'],
            "Precision (Weighted)": report['weighted avg']['precision'],
            "Recall (Weighted)": report['weighted avg']['recall'],
            "F1-Score (Weighted)": report['weighted avg']['f1-score'],
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("Test Accuracy (%)", ascending=False)
    df.to_csv(output_path, index=False)
    return df


# ============================================================
# HELPER - Generate plots for a single model
# ============================================================

def generate_single_model_plots(result, classes, plots_dir):
    """Generate all plots for a single model and return success status."""
    model_name = result["name"]
    safe_name = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    
    # Create folder for each model
    model_dir = plots_dir / safe_name
    makedirs(model_dir)
    
    print(f"\n  Creating plots for: {model_name}")
    
    try:
        # 1. Confusion Matrix
        print(f"    - Confusion matrix...")
        plot_confusion_matrix_single(
            result["y_true"], result["y_pred"], 
            model_name, classes, model_dir
        )
        
        # 2. ROC Curve
        print(f"    - ROC curve...")
        roc_auc = plot_roc_curve_single(
            result["y_true"], result["y_score"],
            model_name, model_dir
        )
        
        # 3. Precision-Recall Curve
        print(f"    - Precision-Recall curve...")
        ap = plot_pr_curve_single(
            result["y_true"], result["y_score"],
            model_name, model_dir
        )
        
        # 4. Per-Class Metrics
        print(f"    - Per-class metrics...")
        plot_per_class_metrics_single(
            result["y_true"], result["y_pred"],
            model_name, classes, model_dir
        )
        
        # 5. Complete Summary
        print(f"    - Complete summary...")
        plot_model_summary_single(result, classes, model_dir)
        
        # Save classification report as text file
        report = classification_report(result["y_true"], result["y_pred"],
                                       target_names=classes)
        report_path = model_dir / f'{safe_name}_classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(f"Classification Report for {model_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Accuracy: {result['accuracy']*100:.2f}%\n")
            f.write(f"AUC Score: {roc_auc:.4f}\n")
            f.write(f"Average Precision: {ap:.4f}\n\n")
            f.write(report)
        
        print(f"    ✓ All plots saved to: {model_dir}")
        return True
        
    except Exception as e:
        print(f"    ✗ Error generating plots for {model_name}: {e}")
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  Wildfire Detection - Model Evaluation & Plot Generation")
    print("  Publication-Quality Plots for Report Generation")
    print("  (With Incremental Processing & Resume Support)")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup
    set_plot_style()
    output_dir = Path(CONFIG["output_dir"])
    plots_dir = output_dir / "plots"
    comparison_dir = plots_dir / "comparison"
    makedirs(plots_dir)
    makedirs(comparison_dir)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Output Directory: {output_dir}")
    
    # Class names
    classes = ["nowildfire", "wildfire"]
    
    # Load cache for resume support
    print("\n  Loading evaluation cache...")
    cache = load_cache()
    
    # Store all results
    all_results = []
    
    # ========================================
    # PROCESS KERAS MODELS ONE BY ONE
    # ========================================
    print("\n" + "=" * 70)
    print("  Processing Keras Models (One at a Time)")
    print("=" * 70)
    
    for model_name, model_info in KERAS_MODELS.items():
        model_path = model_info["path"]
        input_size = model_info["input_size"]
        
        print(f"\n  [{model_name}] (input_size={input_size}x{input_size})")
        
        # Check if already completed
        if is_model_completed(cache, model_name):
            print(f"    → Already completed, loading from cache...")
            cached_result = get_cached_result(cache, model_name)
            if cached_result:
                all_results.append(cached_result)
                print(f"    ✓ Loaded: Accuracy = {cached_result['accuracy']*100:.2f}%")
            continue
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"    ✗ Model file not found: {model_path}")
            continue
        
        # Get test generator with correct input size for this model
        test_gen = get_keras_test_generator(input_size)
        
        # Evaluate model
        result = evaluate_keras_model(model_path, model_name, test_gen)
        
        if result:
            print(f"    ✓ Evaluated: Accuracy = {result['accuracy']*100:.2f}%")
            
            # Cache the result immediately
            cache_result(cache, model_name, result)
            print(f"    → Result cached")
            
            # Generate plots immediately
            if generate_single_model_plots(result, classes, plots_dir):
                mark_plots_completed(cache, model_name)
                print(f"    → Marked as completed in cache")
            
            all_results.append(result)
        else:
            print(f"    ✗ Evaluation failed")
    
    # ========================================
    # PROCESS PYTORCH MODELS ONE BY ONE
    # ========================================
    print("\n" + "=" * 70)
    print("  Processing PyTorch Models (One at a Time)")
    print("=" * 70)
    
    for model_name, model_info in PYTORCH_MODELS.items():
        model_path = model_info["path"]
        input_size = model_info["input_size"]
        
        print(f"\n  [{model_name}] (input_size={input_size}x{input_size})")
        
        # Check if already completed
        if is_model_completed(cache, model_name):
            print(f"    → Already completed, loading from cache...")
            cached_result = get_cached_result(cache, model_name)
            if cached_result:
                all_results.append(cached_result)
                print(f"    ✓ Loaded: Accuracy = {cached_result['accuracy']*100:.2f}%")
            continue
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"    ✗ Model file not found: {model_path}")
            continue
        
        # Get test loader with correct input size for this model
        test_loader, _ = get_pytorch_test_loader(input_size)
        
        # Evaluate model
        result = evaluate_pytorch_model(model_path, model_name, test_loader, device)
        
        if result:
            print(f"    ✓ Evaluated: Accuracy = {result['accuracy']*100:.2f}%")
            
            # Cache the result immediately
            cache_result(cache, model_name, result)
            print(f"    → Result cached")
            
            # Generate plots immediately
            if generate_single_model_plots(result, classes, plots_dir):
                mark_plots_completed(cache, model_name)
                print(f"    → Marked as completed in cache")
            
            all_results.append(result)
        else:
            print(f"    ✗ Evaluation failed")
    
    if not all_results:
        print("\n✗ No models were successfully evaluated!")
        return
    
    # ========================================
    # GENERATE COMPARISON PLOTS (only if we have multiple models)
    # ========================================
    print("\n" + "=" * 70)
    print("  Generating Comparison Plots")
    print("=" * 70)
    
    if len(all_results) >= 2:
        try:
            # 1. Confusion Matrix Grid
            print("\n  Creating comparison confusion matrices grid...")
            plot_confusion_matrices_grid(all_results, classes, comparison_dir)
            
            # 2. ROC Curves Comparison
            print("  Creating ROC curves comparison...")
            plot_roc_curves_comparison(all_results, comparison_dir)
            
            # 3. Precision-Recall Curves Comparison
            print("  Creating Precision-Recall curves comparison...")
            plot_pr_curves_comparison(all_results, comparison_dir)
            
            # 4. Accuracy Ranking
            print("  Creating accuracy ranking chart...")
            plot_accuracy_comparison(all_results, comparison_dir)
            
            # 5. AUC Ranking
            print("  Creating AUC ranking chart...")
            plot_auc_comparison(all_results, comparison_dir)
            
            # 6. Comprehensive Comparison
            print("  Creating comprehensive comparison chart...")
            plot_comprehensive_comparison(all_results, comparison_dir)
            
            # 7. Summary Dashboard
            print("  Creating summary dashboard...")
            plot_summary_dashboard(all_results, classes, comparison_dir)
            
            print(f"\n  ✓ All comparison plots saved to: {comparison_dir}")
            
        except Exception as e:
            print(f"\n  ✗ Error generating comparison plots: {e}")
    else:
        print("\n  ⚠ Only one model evaluated, skipping comparison plots")
    
    # ========================================
    # SAVE RESULTS CSV
    # ========================================
    print("\n" + "-" * 50)
    print("  Saving Results")
    print("-" * 50)
    
    try:
        results_df = save_results_csv(all_results, classes, 
                                       output_dir / f"evaluation_results_{timestamp}.csv")
        
        # Also save a summary CSV without timestamp for easy reference
        save_results_csv(all_results, classes, output_dir / "evaluation_results_latest.csv")
        print(f"  ✓ Results saved to CSV")
    except Exception as e:
        print(f"  ✗ Error saving CSV: {e}")
    
    # ========================================
    # PRINT SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE!")
    print("=" * 70)
    
    print("\n  Test Results Summary (Ranked by Accuracy):")
    print("  " + "-" * 55)
    
    results_sorted = sorted(all_results, key=lambda x: x["accuracy"], reverse=True)
    
    print(f"  {'Rank':<5} {'Model':<25} {'Accuracy':<12} {'AUC':<10}")
    print("  " + "-" * 55)
    
    for i, r in enumerate(results_sorted, 1):
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_score"])
        roc_auc = auc(fpr, tpr)
        marker = "🏆" if i == 1 else "  "
        print(f"  {marker}{i:<3} {r['name']:<25} {r['accuracy']*100:>6.2f}%     {roc_auc:.4f}")
    
    print("\n  Output Structure:")
    print(f"  {output_dir}/")
    print(f"    ├── evaluation_cache.json (resume support)")
    print(f"    ├── evaluation_results_latest.csv")
    print(f"    ├── evaluation_results_{timestamp}.csv")
    print(f"    └── plots/")
    print(f"        ├── comparison/")
    print(f"        │   ├── comparison_confusion_matrices.png")
    print(f"        │   ├── comparison_roc_curves.png")
    print(f"        │   ├── comparison_pr_curves.png")
    print(f"        │   ├── comparison_accuracy_ranking.png")
    print(f"        │   ├── comparison_auc_ranking.png")
    print(f"        │   ├── comparison_comprehensive.png")
    print(f"        │   └── summary_dashboard.png")
    
    for i, r in enumerate(results_sorted[:3]):
        safe_name = r["name"].replace("/", "_").replace("\\", "_").replace(" ", "_")
        if i < 2:
            print(f"        ├── {safe_name}/")
        else:
            print(f"        ├── {safe_name}/ ...")
    
    print(f"\n  Total Models Evaluated: {len(all_results)}")
    print(f"  Best Model: {results_sorted[0]['name']} ({results_sorted[0]['accuracy']*100:.2f}%)")
    print(f"\n  Cache Status: {len(cache.get('completed_plots', []))} models with saved plots")
    print("  (Run again to resume from where it left off)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
