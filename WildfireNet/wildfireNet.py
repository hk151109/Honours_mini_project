# train_custom_cnn.py
# Custom CNN for Wildfire Detection - Built from scratch
# Optimized for CPU training with 4 vCPU
# Run with: python train_custom_cnn.py

import os
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import (
    Compose, Resize, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip,
    ColorJitter, RandomAffine, ToTensor, Normalize, RandomResizedCrop
)
from torch.optim.lr_scheduler import OneCycleLR

# metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# robust PIL for truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------
# CONFIG - edit here
# -----------------------
CONFIG = {
    "data_dir": "./wildfire-prediction-dataset",
    "output_dir": "./CustomCNNOutput",
    "input_size": 128,          # Smaller size for faster CPU training
    "batch_size": 16,
    "workers": 2,               # For 4 vCPU
    "seed": 42,
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.4,
    "num_classes": 2,
}


# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def makedirs(p):
    os.makedirs(p, exist_ok=True)


def plot_training(history, outpath):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(df['epoch'], df['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(df['epoch'], df['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_confusion(cm, classes, outpath):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, int(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_roc(y_true, y_score, outpath):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2.5, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return roc_auc


# -----------------------
# Data Transforms
# -----------------------
def get_train_transforms(img_size):
    """Strong augmentations for training - important for fire/smoke detection"""
    return Compose([
        RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        RandomRotation(20),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.2),
        ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(img_size):
    """Simple transforms for validation/test"""
    return Compose([
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# -----------------------
# Custom CNN Architecture
# -----------------------
class ConvBlock(nn.Module):
    """Convolutional block with Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection if dimensions change
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
        out = F.relu(out + identity)  # Residual connection
        return out


class SpatialAttention(nn.Module):
    """Spatial attention to focus on fire/smoke regions"""
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
    """Channel attention (SE-like) to weight important features"""
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
    """
    Custom CNN for Wildfire Detection
    
    Architecture:
    - 4 convolutional blocks with residual connections
    - Channel and spatial attention mechanisms
    - Global average pooling
    - Fully connected classifier
    
    Designed to detect fire, smoke patterns in satellite/aerial images
    """
    def __init__(self, num_classes=1, dropout=0.4):
        super().__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # Convolutional blocks with increasing channels
        self.block1 = ConvBlock(32, 64, stride=2)      # 128 -> 64
        self.attn1 = ChannelAttention(64)
        
        self.block2 = ConvBlock(64, 128, stride=2)     # 64 -> 32
        self.attn2 = ChannelAttention(128)
        
        self.block3 = ConvBlock(128, 256, stride=2)    # 32 -> 16
        self.attn3 = ChannelAttention(256)
        self.spatial3 = SpatialAttention()
        
        self.block4 = ConvBlock(256, 512, stride=2)    # 16 -> 8
        self.attn4 = ChannelAttention(512)
        self.spatial4 = SpatialAttention()
        
        # Global pooling and classifier
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
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Block 1
        x = self.block1(x)
        x = self.attn1(x)
        
        # Block 2
        x = self.block2(x)
        x = self.attn2(x)
        
        # Block 3 with spatial attention
        x = self.block3(x)
        x = self.attn3(x)
        x = self.spatial3(x)
        
        # Block 4 with spatial attention
        x = self.block4(x)
        x = self.attn4(x)
        x = self.spatial4(x)
        
        # Classifier
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------
# Training Functions
# -----------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs).detach().cpu().numpy().ravel()
        preds = (probs >= 0.5).astype(int)
        correct += (preds == labels.cpu().numpy().ravel().astype(int)).sum()
        total += images.size(0)
        
        # Progress indicator
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} - Loss: {loss.item():.4f}", end='\r')
    
    print()
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_scores = []
    all_true = []
    num_batches = len(loader)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            scores = torch.sigmoid(outputs).cpu().numpy().ravel()
            preds = (scores >= 0.5).astype(int)
            
            all_scores.extend(scores.tolist())
            all_preds.extend(preds.tolist())
            all_true.extend(labels.cpu().numpy().ravel().astype(int).tolist())
            
            if batch_idx % 10 == 0:
                print(f"  Validating {batch_idx + 1}/{num_batches}", end='\r')
    
    print()
    total = len(all_true)
    acc = np.mean(np.array(all_preds) == np.array(all_true))
    return running_loss / total, acc, np.array(all_true), np.array(all_preds), np.array(all_scores)


# -----------------------
# Main Training Function
# -----------------------
def train_wildfire_net():
    cfg = CONFIG.copy()
    set_seed(cfg["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == 'cpu':
        torch.set_num_threads(4)
        print(f"CPU threads: {torch.get_num_threads()}")
    
    # Output directories
    out_dir = Path(cfg["output_dir"])
    models_dir = out_dir / "models"
    plots_dir = out_dir / "plots"
    logs_dir = out_dir / "logs"
    makedirs(models_dir)
    makedirs(plots_dir)
    makedirs(logs_dir)
    
    # Datasets
    train_folder = Path(cfg["data_dir"]) / "train"
    valid_folder = Path(cfg["data_dir"]) / "valid"
    test_folder = Path(cfg["data_dir"]) / "test"
    
    if not all(p.exists() for p in [train_folder, valid_folder, test_folder]):
        raise FileNotFoundError(f"Expected train/valid/test under {cfg['data_dir']}")
    
    train_ds = datasets.ImageFolder(str(train_folder), transform=get_train_transforms(cfg["input_size"]))
    val_ds = datasets.ImageFolder(str(valid_folder), transform=get_val_transforms(cfg["input_size"]))
    test_ds = datasets.ImageFolder(str(test_folder), transform=get_val_transforms(cfg["input_size"]))
    classes = train_ds.classes
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Valid: {len(val_ds)}")
    print(f"  Test:  {len(test_ds)}")
    print(f"  Classes: {classes}")
    
    pin_mem = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, 
                              num_workers=cfg["workers"], pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["workers"], pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["workers"], pin_memory=pin_mem)
    
    # Model
    model = WildfireNet(num_classes=1, dropout=cfg["dropout"])
    model = model.to(device)
    
    print(f"\n{'='*50}")
    print("WildfireNet - Custom CNN Architecture")
    print(f"{'='*50}")
    print(f"Trainable parameters: {model.get_num_params():,}")
    print(f"Input size: {cfg['input_size']}x{cfg['input_size']}")
    
    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    
    total_steps = len(train_loader) * cfg["epochs"]
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_val_acc = 0.0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    
    print(f"\nStarting training for {cfg['epochs']} epochs...")
    print(f"{'='*50}\n")
    
    for epoch in range(1, cfg["epochs"] + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{cfg['epochs']} (lr={current_lr:.6f})")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = models_dir / f"wildfirenet_best_{timestamp}.pth"
            torch.save({
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
                "config": cfg
            }, ckpt_path)
            print(f"  ✓ New best model saved! (val_acc={val_acc:.4f})")
        
        print()
    
    # Save final model
    final_path = models_dir / f"wildfirenet_final_{timestamp}.pth"
    torch.save({
        "model_state": model.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
        "config": cfg
    }, final_path)
    print(f"Final model saved: {final_path}")
    
    # Save training history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(logs_dir / f"history_{timestamp}.csv", index=False)
    plot_training(history, plots_dir / f"training_curves_{timestamp}.png")
    
    # -----------------------
    # Evaluate on Test Set
    # -----------------------
    print(f"\n{'='*50}")
    print("Evaluating on Test Set")
    print(f"{'='*50}")
    
    # Load best model
    best_ckpt = torch.load(models_dir / f"wildfirenet_best_{timestamp}.pth")
    model.load_state_dict(best_ckpt["model_state"])
    
    test_loss, test_acc, y_true, y_pred, y_score = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc * 100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, classes, plots_dir / f"confusion_{timestamp}.png")
    
    # ROC curve
    roc_auc = plot_roc(y_true, y_score, plots_dir / f"roc_{timestamp}.png")
    print(f"  Test AUC: {roc_auc:.4f}")
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(logs_dir / f"classification_report_{timestamp}.csv")
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    # Save test summary
    summary = {
        "model": "WildfireNet",
        "test_loss": test_loss,
        "test_accuracy": test_acc * 100,
        "test_auc": roc_auc,
        "trainable_params": model.get_num_params(),
        "best_val_acc": best_val_acc * 100,
        "epochs_trained": cfg["epochs"]
    }
    pd.DataFrame([summary]).to_csv(logs_dir / f"test_summary_{timestamp}.csv", index=False)
    
    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"{'='*50}")
    print(f"All outputs saved to: {out_dir}")
    
    return {
        "test_accuracy": test_acc * 100,
        "test_auc": roc_auc,
        "best_val_acc": best_val_acc * 100,
        "model_path": str(final_path)
    }


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    print("="*60)
    print("WildfireNet - Custom CNN for Wildfire Detection")
    print("="*60)
    print("\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print()
    
    results = train_wildfire_net()
    
    print("\n" + "="*60)
    print("Final Results:")
    print("="*60)
    for k, v in results.items():
        print(f"  {k}: {v}")
