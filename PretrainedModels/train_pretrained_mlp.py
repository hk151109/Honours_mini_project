"""
Wildfire Detection Using Transfer Learning
Train multiple CNN models and evaluate performance

Usage:
    python train.py
"""

# Standard library imports
import os
import gc
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization,
    Input
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import (
    VGG16,
    VGG19,
    ResNet50,
    ResNet50V2,
    MobileNetV3Small,
    MobileNetV3Large,
    EfficientNetB0
)

# Timm for additional ResNet models
import timm
import torch

# Scikit-learn imports
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class WildfireTrainer:
    """Main trainer class for wildfire detection models."""
    
    def __init__(self, config):
        """Initialize trainer with configuration."""
        self.config = config
        self.histories = {}
        self.trained_models = {}
        self.results = []
        self.test_results = []
        
        # Create output directories
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(config['plots_dir'], exist_ok=True)
        os.makedirs(config['models_dir'], exist_ok=True)
        
        # Enable mixed precision if GPU available
        if config['mixed_precision'] and tf.config.list_physical_devices('GPU'):
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("✓ Mixed precision enabled (float16)")
        
        # Define models
        self.models = {
            "VGG16": VGG16,
            "VGG19": VGG19,
            "ResNet18": "resnet18",
            "ResNet34": "resnet34",
            "ResNet50": ResNet50,
            "ResNet50V2": ResNet50V2,
            "MobileNetV3Small": MobileNetV3Small,
            "MobileNetV3Large": MobileNetV3Large,
            "EfficientNetB0": EfficientNetB0
        }
        
        # Define input shapes
        self.input_shapes = {name: (224, 224, 3) for name in self.models.keys()}
        
        print(f"Models to train: {list(self.models.keys())}")
    
    def setup_data_generators(self):
        """Setup data generators for training, validation, and testing."""
        print("\n" + "="*60)
        print("Setting up data generators...")
        print("="*60)
        
        # Training data with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=25,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
        
        # Validation and test data without augmentation
        valid_test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        
        self.train_datagen = train_datagen
        self.valid_test_datagen = valid_test_datagen
        
        print("✓ Data generators configured successfully")
    
    def build_model(self, BaseModel, input_shape):
        """Build transfer learning model."""
        # Check if it's a timm model (string)
        if isinstance(BaseModel, str):
            # Create timm model
            timm_model = timm.create_model(BaseModel, pretrained=True, num_classes=0)
            timm_model.eval()
            
            # Get feature dimensions
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1])
                features = timm_model(dummy_input)
                feature_dim = features.shape[1]
            
            # Create custom layer that wraps timm model
            class TimmLayer(tf.keras.layers.Layer):
                def __init__(self, timm_model, **kwargs):
                    super().__init__(**kwargs)
                    self.timm_model = timm_model
                    self.timm_model.eval()
                
                def call(self, inputs):
                    def torch_forward(x):
                        x_np = x.numpy()
                        x_np = np.transpose(x_np, (0, 3, 1, 2))
                        x_torch = torch.from_numpy(x_np).float()
                        
                        with torch.no_grad():
                            out = self.timm_model(x_torch)
                        
                        return out.numpy().astype(np.float32)
                    
                    output = tf.py_function(
                        func=torch_forward,
                        inp=[inputs],
                        Tout=tf.float32
                    )
                    output.set_shape([None, feature_dim])
                    return output
                
                def compute_output_shape(self, input_shape):
                    return (input_shape[0], feature_dim)
            
            # Build model with timm layer
            model = Sequential([
                TimmLayer(timm_model, name=f'timm_{BaseModel}'),
                Dropout(0.3),
                Dense(128, activation='gelu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid', dtype='float32')
            ])
            
            model.build(input_shape=(None,) + input_shape)
            
        else:
            # Standard Keras application model
            base = BaseModel(weights='imagenet', include_top=False, input_shape=input_shape)
            base.trainable = False
            
            model = Sequential([
                base,
                GlobalAveragePooling2D(),
                Dropout(0.3),
                Dense(128, activation='gelu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid', dtype='float32')
            ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self):
        """Train all models."""
        print("\n" + "="*60)
        print("Starting model training...")
        print("="*60)
        
        for name, BaseModel in tqdm(self.models.items(), desc="Training models"):
            print(f"\n{'='*60}")
            print(f"Training model: {name}")
            print(f"{'='*60}")
            
            input_shape = self.input_shapes[name]
            
            # Create data generators
            train_gen = self.train_datagen.flow_from_directory(
                self.config['train_dir'],
                target_size=input_shape[:2],
                batch_size=self.config['batch_size'],
                class_mode="binary",
                shuffle=True
            )
            
            valid_gen = self.valid_test_datagen.flow_from_directory(
                self.config['valid_dir'],
                target_size=input_shape[:2],
                batch_size=self.config['batch_size'],
                class_mode="binary",
                shuffle=False
            )
            
            try:
                # Build model
                model = self.build_model(BaseModel, input_shape=input_shape)
                
                # Callbacks
                callbacks = [
                    EarlyStopping(
                        monitor="val_loss",
                        patience=self.config['patience'],
                        restore_best_weights=True,
                        verbose=1
                    ),
                    ModelCheckpoint(
                        filepath=os.path.join(self.config['models_dir'], f'{name}_best.keras'),
                        monitor='val_accuracy',
                        save_best_only=True,
                        verbose=0
                    )
                ]
                
                # Train model
                history = model.fit(
                    train_gen,
                    validation_data=valid_gen,
                    epochs=self.config['epochs'],
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Store results
                self.histories[name] = history
                self.trained_models[name] = model
                
                # Get final metrics
                val_loss = history.history['val_loss'][-1]
                val_acc = history.history['val_accuracy'][-1]
                
                self.results.append({
                    "Model": name,
                    "Val Accuracy (%)": val_acc * 100,
                    "Val Loss": val_loss
                })
                
                print(f"\n{name} - Final validation accuracy: {val_acc * 100:.2f}%")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        print("\n" + "="*60)
        print("All models trained successfully!")
        print("="*60)
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        print("\nGenerating training curves...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot accuracy
        for name, history in self.histories.items():
            axes[0].plot(history.history['accuracy'], label=f'{name} (train)', linestyle='--', alpha=0.7)
            axes[0].plot(history.history['val_accuracy'], label=f'{name} (val)', linewidth=2)
        
        axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        for name, history in self.histories.items():
            axes[1].plot(history.history['loss'], label=f'{name} (train)', linestyle='--', alpha=0.7)
            axes[1].plot(history.history['val_loss'], label=f'{name} (val)', linewidth=2)
        
        axes[1].set_title('Model Loss Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['plots_dir'], 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Training curves saved")
    
    def evaluate_on_test_set(self):
        """Evaluate all models on test set."""
        print("\n" + "="*60)
        print("Evaluating models on test set...")
        print("="*60)
        
        for name, model in tqdm(self.trained_models.items(), desc="Testing models"):
            print(f"\nEvaluating {name}...")
            
            input_shape = self.input_shapes[name]
            
            test_gen = self.valid_test_datagen.flow_from_directory(
                self.config['test_dir'],
                target_size=input_shape[:2],
                batch_size=self.config['batch_size'],
                class_mode="binary",
                shuffle=False
            )
            
            # Evaluate
            test_loss, test_acc = model.evaluate(test_gen, verbose=0)
            
            # Get predictions
            y_true = test_gen.classes
            y_pred_prob = model.predict(test_gen, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype(int).ravel()
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            self.test_results.append({
                "Model": name,
                "Test Accuracy (%)": test_acc * 100,
                "Test Loss": test_loss,
                "Confusion Matrix": cm
            })
            
            print(f"{name} - Test accuracy: {test_acc * 100:.2f}%")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        print("\nGenerating confusion matrices...")
        
        n_models = len(self.test_results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        # Get class names
        test_gen_temp = self.valid_test_datagen.flow_from_directory(
            self.config['test_dir'],
            target_size=(224, 224),
            batch_size=1,
            class_mode="binary",
            shuffle=False
        )
        class_names = list(test_gen_temp.class_indices.keys())
        
        for idx, result in enumerate(self.test_results):
            cm = result["Confusion Matrix"]
            name = result["Model"]
            
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(cmap="Blues", values_format="d", ax=axes[idx], colorbar=False)
            axes[idx].set_title(f"{name}\nAcc: {result['Test Accuracy (%)']:.2f}%", 
                                fontweight='bold', fontsize=11)
            axes[idx].grid(False)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['plots_dir'], 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Confusion matrices saved")
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models."""
        print("\nGenerating ROC curves...")
        
        plt.figure(figsize=(10, 8))
        
        for name, model in self.trained_models.items():
            input_shape = self.input_shapes[name]
            
            test_gen = self.valid_test_datagen.flow_from_directory(
                self.config['test_dir'],
                target_size=input_shape[:2],
                batch_size=self.config['batch_size'],
                class_mode="binary",
                shuffle=False
            )
            
            y_true = test_gen.classes
            y_score = model.predict(test_gen, verbose=0).ravel()
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2.5, label=f"{name} (AUC = {roc_auc:.3f})")
        
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=2, label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=13, fontweight='bold')
        plt.ylabel("True Positive Rate", fontsize=13, fontweight='bold')
        plt.title("ROC Curves - All Models", fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['plots_dir'], 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ ROC curves saved")
    
    def save_results(self):
        """Save results to CSV files."""
        print("\nSaving results...")
        
        # Validation results
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values("Val Accuracy (%)", ascending=False)
        results_df.to_csv(os.path.join(self.config['output_dir'], 'validation_results.csv'), index=False)
        
        # Test results
        test_df = pd.DataFrame([{k: v for k, v in r.items() if k != "Confusion Matrix"} 
                                for r in self.test_results])
        test_df = test_df.sort_values("Test Accuracy (%)", ascending=False)
        test_df.to_csv(os.path.join(self.config['output_dir'], 'test_results.csv'), index=False)
        
        print("\n" + "="*60)
        print("Validation Results:")
        print("="*60)
        print(results_df.to_string(index=False))
        
        print("\n" + "="*60)
        print("Test Results:")
        print("="*60)
        print(test_df.to_string(index=False))
        
        print("\n✓ Results saved to CSV files")
    
    def save_models(self):
        """Save all trained models."""
        print("\nSaving trained models...")
        
        for name, model in tqdm(self.trained_models.items(), desc="Saving models"):
            try:
                keras_path = os.path.join(self.config['models_dir'], f"{name}_final.keras")
                model.save(keras_path)
                print(f"✓ {name} saved: {keras_path}")
            except Exception as e:
                print(f"✗ Error saving {name}: {e}")
        
        print(f"\n✓ All models saved to: {self.config['models_dir']}")
    
    def run(self):
        """Run the complete training pipeline."""
        start_time = datetime.now()
        print("\n" + "="*60)
        print("WILDFIRE DETECTION - TRANSFER LEARNING")
        print("="*60)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Setup data
        self.setup_data_generators()
        
        # Train models
        self.train_models()
        
        # Plot training curves
        self.plot_training_curves()
        
        # Evaluate on test set
        self.evaluate_on_test_set()
        
        # Generate visualizations
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        
        # Save results
        self.save_results()
        
        # Save models
        self.save_models()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration}")
        print(f"Results saved to: {self.config['output_dir']}")
        print("="*60)


# ============================================================
# CONFIGURATION - EDIT THESE SETTINGS
# ============================================================
CONFIG = {
    # Data paths
    'data_dir': './wildfire-prediction-dataset',
    
    # Output paths
    'output_dir': './output',
    
    # Training parameters
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 1e-4,
    'patience': 5,
    
    # Performance
    'mixed_precision': True  # Set to False if no GPU
}


def main():
    """Main entry point."""
    # Build full config
    config = {
        'data_dir': CONFIG['data_dir'],
        'train_dir': os.path.join(CONFIG['data_dir'], 'train'),
        'valid_dir': os.path.join(CONFIG['data_dir'], 'valid'),
        'test_dir': os.path.join(CONFIG['data_dir'], 'test'),
        'output_dir': CONFIG['output_dir'],
        'plots_dir': os.path.join(CONFIG['output_dir'], 'plots'),
        'models_dir': os.path.join(CONFIG['output_dir'], 'models'),
        'batch_size': CONFIG['batch_size'],
        'epochs': CONFIG['epochs'],
        'learning_rate': CONFIG['learning_rate'],
        'patience': CONFIG['patience'],
        'mixed_precision': CONFIG['mixed_precision']
    }
    
    # Verify data directories exist
    for dir_path in [config['train_dir'], config['valid_dir'], config['test_dir']]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    # Create and run trainer
    trainer = WildfireTrainer(config)
    trainer.run()


if __name__ == '__main__':
    main()
