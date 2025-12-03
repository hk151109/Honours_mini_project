# Wildfire Detection Using Deep Learning

## Dataset

> ⚠️ **Note**: The dataset included in this repository (`wildfire-prediction-dataset-sample/`) is only a **sample** for demonstration purposes.

To download the **full dataset**, use the following code:

Run the python file - wildfire-prediction-dataset-sample/dataset.py


## Project Overview

This project presents a comprehensive study on wildfire detection from satellite and aerial imagery using deep learning techniques. Wildfires pose a significant threat to ecosystems, human lives, and property worldwide, making early detection crucial for effective disaster management and response. In this work, we systematically explored multiple approaches to image classification for wildfire detection, beginning with transfer learning using seven state-of-the-art pretrained convolutional neural networks (VGG16, VGG19, ResNet50, ResNet50V2, MobileNetV3Small, MobileNetV3Large, and EfficientNetB0) with frozen convolutional bases to leverage features learned from the ImageNet dataset. We then performed fine-tuning on the top three performing models by unfreezing the top 20 layers, allowing the networks to adapt their learned representations specifically for wildfire imagery. Additionally, we implemented FireNet, a specialized CNN architecture proposed in the literature specifically designed for fire detection tasks, and developed WildfireNet, a custom CNN architecture featuring residual connections and attention mechanisms (both channel and spatial attention) to help the model focus on relevant fire and smoke patterns in images. Our experiments were conducted on a binary classification dataset containing wildfire and non-wildfire images, with rigorous evaluation using metrics including accuracy, AUC-ROC, precision, recall, and F1-score. The results demonstrate that fine-tuned VGG19 achieved the highest accuracy of 99.17%, while our custom WildfireNet achieved competitive performance of 98.84% with significantly fewer parameters (~5.1M compared to ~144M for VGG19), highlighting the potential for efficient deployment in resource-constrained environments such as edge devices or real-time monitoring systems.

---

## Training Strategy

### Stage 1: Transfer Learning (Frozen Base)
Trained 7 pretrained models with all convolutional layers frozen, only training the classification head:
- VGG16, VGG19
- ResNet50, ResNet50V2
- MobileNetV3Small, MobileNetV3Large
- EfficientNetB0

### Stage 2: Fine-tuning (Top 3 Models)
Unfroze top 20 layers of the 3 best performers for fine-tuning:
- VGG16 → VGG16_finetuned
- VGG19 → VGG19_finetuned
- ResNet50V2 → ResNet50V2_finetuned

### Stage 3: FireNet Implementation
Implemented the FireNet-CNN as described in the paper: https://arxiv.org/pdf/1905.11922

Details:
- Architecture: FireNet-CNN (paper) — a compact, purpose-built network for fire detection.
- Specs: Custom 5 convolutional blocks with increasing channel widths (e.g. 32→64→128→256→512),
   followed by a Flatten + MLP head (512 → 256 → 1) with dropout.
- Input resolution: 150×150 RGB images (this model was trained and evaluated at 150×150).
- Training procedure implemented in `FireNet/train_firenet.py`:
   - Stage A: Freeze the convolutional base and train the MLP head (fast, low-cost stage).
   - Stage B: unfreeze the entire network and fine-tune at a lower learning rate for additional gains.
- Outputs: Trained models, logs and plots are saved to `FireNet/FireNetOutput/`.

Rationale: FireNet is a lightweight architecture tailored to capture fire/smoke patterns at moderate spatial resolution,
trading off some general-purpose capacity for a focused, efficient detector suitable for this task.

### Stage 4: WildfireNet (Custom Architecture)
Built a custom CNN from scratch with:
- Residual connections
- Channel & Spatial Attention mechanisms
- 128×128 input size
- ~5.1M trainable parameters

---

## All Models - Complete Metrics

| Rank | Model | Parameters | Accuracy | AUC | F1-Score | Precision | Recall |
|------|-------|-----------|----------|-----|----------|-----------|--------|
| 🏆 1 | VGG19_finetuned | ~144M | **99.17%** | 0.9993 | 0.9917 | 0.9916 | 0.9917 |
| 2 | VGG16_finetuned | ~138M | 99.10% | 0.9997 | 0.9909 | 0.9902 | 0.9916 |
| 3 | WildfireNet | ~5.1M | 98.84% | 0.9991 | 0.9883 | 0.9879 | 0.9887 |
| 4 | ResNet50V2_finetuned | ~25.6M | 98.37% | 0.9985 | 0.9835 | 0.9833 | 0.9837 |
| 5 | ResNet50V2 | ~25.6M | 97.24% | 0.9966 | 0.9721 | 0.9720 | 0.9721 |
| 6 | FireNet | ~6.5M | 95.51% | 0.9897 | 0.9547 | 0.9540 | 0.9555 |
| 7 | VGG16 | ~138M | 95.41% | 0.9883 | 0.9535 | 0.9550 | 0.9523 |
| 8 | VGG19 | ~144M | 94.21% | 0.9856 | 0.9412 | 0.9429 | 0.9400 |
| 9 | ResNet50 | ~25.6M | 88.05% | 0.9403 | 0.8789 | 0.8798 | 0.8782 |
| 10 | EfficientNetB0 | ~5.3M | 86.98% | 0.8992 | 0.8664 | 0.8764 | 0.8624 |
| 11 | MobileNetV3Large | ~5.4M | 85.83% | 0.8967 | 0.8559 | 0.8586 | 0.8541 |
| 12 | MobileNetV3Small | ~2.5M | 83.62% | 0.8647 | 0.8334 | 0.8361 | 0.8318 |

---

## Top 4 Performers Comparison

| Model | Parameters | Accuracy | F1-Score | Precision | Recall |
|-------|-----------|----------|----------|-----------|--------|
| **VGG19_finetuned** | ~144M | **99.17%** | 0.9917 | 0.9916 | 0.9917 |
| **VGG16_finetuned** | ~138M | 99.10% | 0.9909 | 0.9902 | 0.9916 |
| **WildfireNet** | ~5.1M | 98.84% | 0.9883 | 0.9879 | 0.9887 |
| **ResNet50V2_finetuned** | ~25.6M | 98.37% | 0.9835 | 0.9833 | 0.9837 |

---

## Key Insights

1. **Fine-tuning significantly improves performance**: 
   - VGG19: 94.21% → 99.17% (+4.96%)
   - VGG16: 95.41% → 99.10% (+3.69%)
   - ResNet50V2: 97.24% → 98.37% (+1.13%)

2. **Custom WildfireNet is highly efficient**: 
   - 98.84% accuracy with only 5.1M parameters
   - Outperforms ResNet50V2_finetuned (25.6M params) by 0.47%
   - **28× fewer parameters** than VGG models

3. **Lightweight models underperform**: MobileNetV3 and EfficientNetB0 had lower accuracy despite being designed for efficiency - likely due to insufficient capacity for this specific task

4. **All top models achieve >0.99 AUC**: Excellent discrimination between wildfire and no-wildfire classes

5. **Trade-off insight**: For deployment scenarios:
   - **Maximum accuracy**: Use VGG19_finetuned (99.17%)
   - **Best efficiency**: Use WildfireNet (98.84% with 28× fewer parameters)

---

## Model Architectures Summary

| Model | Type | Input Size | Key Features |
|-------|------|------------|--------------|
| VGG16/19 | Pretrained (ImageNet) | 224×224 | Very deep, simple 3×3 convolutions |
| ResNet50/V2 | Pretrained (ImageNet) | 224×224 | Residual/skip connections |
| MobileNetV3 | Pretrained (ImageNet) | 224×224 | Depthwise separable convolutions |
| EfficientNetB0 | Pretrained (ImageNet) | 224×224 | NAS-optimized, compound scaling |
| FireNet | Custom (Paper) | 150×150 | 5 conv blocks, fire-specific design |
| WildfireNet | Custom (Ours) | 128×128 | Residual + Channel/Spatial Attention |

---