"""
MSTAR-10 SAR Image Classification using CNNs
============================================

This script implements a complete pipeline for SAR target recognition:
1. Custom lightweight CNN (13M params)
2. Data augmentation (standard + physics-based)
3. Pre-trained model comparison (AlexNet, VGG-16, ResNet-50, DenseNet-121)
4. Comprehensive evaluation and visualization

Compatible with:
- Python 3.10.11
- PyTorch 2.10.0+cpu
- Torchvision 0.25.0+cpu
- CPU-only training

Author: Generated for MSTAR SAR Classification Project
Date: 2026
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json

# Core libraries
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Torchvision
import torchvision
from torchvision import models, transforms
from torchvision.models import (
    AlexNet_Weights, VGG16_Weights, ResNet50_Weights, DenseNet121_Weights
)

# Image processing
from PIL import Image

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# Progress bar
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# 2. CONFIGURATION CLASS
# ============================================================================

class Config:
    """
    Configuration class for all hyperparameters and paths.
    Modify these values according to your environment.
    """
    
    # ========== PATHS ==========
    # TODO: Set your MSTAR data path here
    DATA_ROOT = Path('./mstar_data')  # Root directory containing MSTAR dataset
    TRAIN_DIR = DATA_ROOT / 'train'   # Training images (17° depression angle)
    TEST_DIR = DATA_ROOT / 'test'     # Test images (15° depression angle)
    
    # Output paths
    OUTPUT_DIR = Path('./outputs')
    MODELS_DIR = OUTPUT_DIR / 'models'
    PLOTS_DIR = OUTPUT_DIR / 'plots'
    RESULTS_DIR = OUTPUT_DIR / 'results'
    
    # Physics-based augmentation (from GitHub repo)
    # TODO: Clone https://github.com/SenSE-Lab-OSU/mstar_data_aug and set path
    SYNTHETIC_DATA_DIR = Path('./mstar_synthetic')  # Generated synthetic data
    
    # ========== DATASET PARAMETERS ==========
    IMAGE_SIZE = 128  # Resize all images to 128x128
    NUM_CLASSES = 10  # MSTAR-10 has 10 vehicle classes
    
    # Class names (in alphabetical order)
    CLASS_NAMES = [
    'classe1', 'classe2', 'classe3', 'classe4', 'classe5',
    'classe6', 'classe7', 'classe8', 'classe9', 'classe10'
]
    
    # Underrepresented classes (need augmentation)
    UNDERSAMPLE_CLASSES = []
    
    # ========== TRAINING HYPERPARAMETERS ==========
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4  # 0.0001
    WEIGHT_DECAY = 1e-5   # L2 regularization
    
    # Early stopping
    PATIENCE = 15  # Stop if no improvement for 15 epochs
    
    # ========== MODEL PARAMETERS ==========
    DROPOUT_RATE = 0.5
    DENSE_UNITS = 512
    
    # ========== AUGMENTATION PARAMETERS ==========
    USE_AUGMENTATION = True
    AUGMENTATION_FACTOR = 10  # Generate 10 synthetic samples per original
    
    # Standard augmentation parameters
    ROTATION_DEGREES = 10
    TRANSLATION_PERCENT = 0.1
    SPECKLE_NOISE_VAR = 0.05
    
    # ========== HARDWARE & REPRODUCIBILITY ==========
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0  # Set to 0 for CPU-only, increase if using GPU
    RANDOM_SEED = 42
    
    # ========== PRE-TRAINED MODELS TO COMPARE ==========
    PRETRAINED_MODELS = {
        'AlexNet': True,
        'VGG16': True,
        'ResNet50': True,
        'DenseNet121': True,
    }
    
    @classmethod
    def setup_directories(cls):
        """Create all necessary directories."""
        for dir_path in [cls.OUTPUT_DIR, cls.MODELS_DIR, cls.PLOTS_DIR, cls.RESULTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def set_seed(cls, seed: int = None):
        """Set random seed for reproducibility."""
        seed = seed or cls.RANDOM_SEED
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# 3. DATA LOADING & PREPROCESSING MODULE
# ============================================================================

class SARImageTransform:
    """
    Custom transforms for SAR images:
    1. Log scaling: log(1 + intensity)
    2. Normalization: scale to [0, 1]
    """
    
    @staticmethod
    def log_scale(image: np.ndarray) -> np.ndarray:
        """Apply logarithmic scaling to compress dynamic range."""
        return np.log1p(image)  # log(1 + x)
    
    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        img_min, img_max = image.min(), image.max()
        if img_max - img_min < 1e-10:  # Avoid division by zero
            return np.zeros_like(image)
        return (image - img_min) / (img_max - img_min)
    
    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """Complete preprocessing pipeline for SAR images."""
        image = SARImageTransform.log_scale(image)
        image = SARImageTransform.normalize(image)
        return image


class AddSpeckleNoise:
    """
    Add multiplicative speckle noise to SAR images.
    Speckle noise is characteristic of coherent imaging systems like SAR.
    Formula: output = image * (1 + noise)
    where noise ~ N(0, variance)
    """
    
    def __init__(self, variance: float = 0.05):
        self.variance = variance
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply speckle noise to image tensor."""
        noise = torch.randn_like(img) * self.variance
        noisy_img = img * (1 + noise)
        return torch.clamp(noisy_img, 0, 1)  # Ensure values stay in [0, 1]


class MSTARDataset(Dataset):
    """
    Custom Dataset class for MSTAR SAR images.
    
    Handles:
    - Loading images from directory structure
    - Preprocessing (log scaling, normalization)
    - Optional augmentation
    - Converting to PyTorch tensors
    """
    
    def __init__(
        self,
        root_dir: Path,
        transform=None,
        preprocess: bool = True,
        class_to_idx: Dict = None
    ):
        """
        Args:
            root_dir: Path to data directory (should contain subdirs for each class)
            transform: Optional torchvision transforms
            preprocess: Whether to apply SAR-specific preprocessing
            class_to_idx: Optional mapping of class names to indices
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.preprocess = preprocess
        
        # Build dataset
        self.samples = []
        self.class_to_idx = class_to_idx or {}
        
        # If class_to_idx not provided, create it from directories
        if not self.class_to_idx:
            self.class_to_idx = {
                cls_name: idx for idx, cls_name in enumerate(sorted(Config.CLASS_NAMES))
            }
        
        self._load_samples()
    
    def _load_samples(self):
        """Load all image paths and labels from directory structure."""
        for class_name in self.class_to_idx.keys():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Directory {class_dir} does not exist. Skipping.")
                continue
            
            # Find all image files (support multiple formats)
            image_files = list(class_dir.glob('*.png')) + \
                         list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.jpeg'))
            
            for img_path in image_files:
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} samples from {self.root_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Returns:
            image: Preprocessed image tensor [1, H, W]
            label: Class index
        """
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            image = image.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            image = np.array(image, dtype=np.float32)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)
        
        # Apply SAR-specific preprocessing
        if self.preprocess:
            image = SARImageTransform.preprocess(image)
        
        # Convert to tensor [1, H, W]
        image = torch.from_numpy(image).unsqueeze(0).float()
        
        # Apply augmentation transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_augmentation_transforms(train: bool = True):
    """
    Create augmentation transforms for training/testing.
    
    Args:
        train: If True, apply augmentation. If False, only normalize.
    
    Returns:
        torchvision.transforms.Compose object
    """
    if train and Config.USE_AUGMENTATION:
        return transforms.Compose([
            transforms.RandomRotation(Config.ROTATION_DEGREES),
            transforms.RandomAffine(
                degrees=0,
                translate=(Config.TRANSLATION_PERCENT, Config.TRANSLATION_PERCENT)
            ),
            AddSpeckleNoise(variance=Config.SPECKLE_NOISE_VAR),
        ])
    else:
        return None


def generate_synthetic_data():
    """
    Generate synthetic SAR images using physics-based augmentation.
    
    This function interfaces with the code from:
    https://github.com/SenSE-Lab-OSU/mstar_data_aug
    
    NOTE: This is a STUB. You need to:
    1. Clone the GitHub repository
    2. Install MATLAB or use their provided Python conversion
    3. Run their augmentation script on underrepresented classes
    4. Save the generated images to Config.SYNTHETIC_DATA_DIR
    
    The script generates 196 complex SAR images per original sample.
    We recommend generating a subset (e.g., 10 per sample) to avoid
    overwhelming the dataset.
    """
    
    print("\n" + "="*70)
    print("PHYSICS-BASED DATA AUGMENTATION")
    print("="*70)
    
    if not Config.SYNTHETIC_DATA_DIR.exists():
        print(f"\nWARNING: Synthetic data directory not found: {Config.SYNTHETIC_DATA_DIR}")
        print("\nTo generate physics-based synthetic data:")
        print("1. Clone: git clone https://github.com/SenSE-Lab-OSU/mstar_data_aug")
        print("2. Follow their README to generate augmented data")
        print("3. Set Config.SYNTHETIC_DATA_DIR to the output directory")
        print("4. Re-run this script")
        print("\nContinuing without synthetic data augmentation...\n")
        return False
    
    print(f"✓ Synthetic data found at: {Config.SYNTHETIC_DATA_DIR}")
    print(f"✓ Will augment underrepresented classes: {Config.UNDERSAMPLE_CLASSES}")
    
    # TODO: Implement integration with synthetic data
    # For now, assume synthetic images are organized in the same structure
    # as the original dataset (subdirectories per class)
    
    return True


def create_dataloaders(use_synthetic: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and testing DataLoaders.
    
    Args:
        use_synthetic: Whether to include synthetic augmented data
    
    Returns:
        train_loader, test_loader
    """
    print("\n" + "="*70)
    print("CREATING DATALOADERS")
    print("="*70)
    
    # Create transforms
    train_transform = create_augmentation_transforms(train=True)
    test_transform = create_augmentation_transforms(train=False)
    
    # Create datasets
    train_dataset = MSTARDataset(
        root_dir=Config.TRAIN_DIR,
        transform=train_transform,
        preprocess=True
    )
    
    test_dataset = MSTARDataset(
        root_dir=Config.TEST_DIR,
        transform=test_transform,
        preprocess=True
    )
    
    # If using synthetic data, combine with training set
    if use_synthetic and Config.SYNTHETIC_DATA_DIR.exists():
        synthetic_dataset = MSTARDataset(
            root_dir=Config.SYNTHETIC_DATA_DIR,
            transform=train_transform,
            preprocess=True,
            class_to_idx=train_dataset.class_to_idx
        )
        
        # Combine datasets
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([train_dataset, synthetic_dataset])
        print(f"✓ Added {len(synthetic_dataset)} synthetic samples to training set")
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False  # Set to False for CPU
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False
    )
    
    return train_loader, test_loader


# ============================================================================
# 4. MODEL DEFINITION MODULE
# ============================================================================

class LightweightCNN(nn.Module):
    """
    Custom lightweight CNN for SAR image classification.
    
    Architecture (as per the report):
    - Input: (1, 128, 128) grayscale SAR image
    - Conv2D_1: 32 filters, 3x3, ReLU
    - MaxPool2D: 2x2
    - Conv2D_2: 64 filters, 3x3, ReLU
    - MaxPool2D: 2x2
    - Conv2D_3: 128 filters, 3x3, ReLU
    - MaxPool2D: 2x2
    - Flatten
    - Dropout: 0.5
    - Dense: 512 units, ReLU
    - Dropout: 0.5
    - Output: 10 classes, Softmax
    
    Total parameters: ~13M
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(LightweightCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)  # (1, 128, 128) -> (32, 126, 126)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)       # -> (32, 63, 63)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0) # -> (64, 61, 61)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)       # -> (64, 30, 30)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0) # -> (128, 28, 28)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)        # -> (128, 14, 14)
        
        # Calculate flattened size: 128 * 14 * 14 = 25088
        self.flatten_size = 128 * 14 * 14
        
        # Fully connected layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.flatten_size, Config.DENSE_UNITS)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(Config.DENSE_UNITS, num_classes)
    
    def forward(self, x):
        """Forward pass."""
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Flatten
        x = x.view(-1, self.flatten_size)
        
        # Classification head
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)  # Logits (Softmax applied in loss function)
        
        return x
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_pretrained_model(model_name: str, num_classes: int = 10):
    """
    Create and adapt a pre-trained model for MSTAR classification.
    
    Args:
        model_name: One of ['AlexNet', 'VGG16', 'ResNet50', 'DenseNet121']
        num_classes: Number of output classes
    
    Returns:
        Modified model ready for fine-tuning
    """
    print(f"\n{'='*70}")
    print(f"LOADING PRE-TRAINED MODEL: {model_name}")
    print(f"{'='*70}")
    
    if model_name == 'AlexNet':
        model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        # Modify first conv layer to accept 1-channel input
        model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        # Modify classifier
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    
    elif model_name == 'VGG16':
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # Modify first conv layer
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # Modify classifier
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    
    elif model_name == 'ResNet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Modify first conv layer
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify final FC layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    
    elif model_name == 'DenseNet121':
        model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        # Modify first conv layer
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify classifier
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"✓ Model loaded and adapted for {num_classes} classes")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


# ============================================================================
# 5. TRAINING & EVALUATION MODULE
# ============================================================================

class Trainer:
    """
    Trainer class to handle model training and evaluation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_name: str = "CustomCNN"
    ):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_name = model_name
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for images, labels in pbar:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """Validate on test set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Validation', leave=False)
            for images, labels in pbar:
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs: int = None):
        """
        Complete training loop with early stopping.
        
        Args:
            num_epochs: Number of epochs to train (defaults to Config.NUM_EPOCHS)
        """
        num_epochs = num_epochs or Config.NUM_EPOCHS
        
        print(f"\n{'='*70}")
        print(f"TRAINING: {self.model_name}")
        print(f"{'='*70}")
        print(f"Device: {Config.DEVICE}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {Config.BATCH_SIZE}")
        print(f"Learning rate: {Config.LEARNING_RATE}")
        print(f"{'='*70}\n")
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(f"Epoch [{epoch}/{num_epochs}] ({epoch_time:.1f}s) - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Check if best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                self.save_model(f'{self.model_name}_best.pth')
                print(f"  → New best model saved! (Val Acc: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= Config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
                break
        
        # Load best model
        self.load_model(f'{self.model_name}_best.pth')
        print(f"\n{'='*70}")
        print(f"Training complete! Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"{'='*70}\n")
    
    def save_model(self, filename: str):
        """Save model checkpoint."""
        filepath = Config.MODELS_DIR / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }, filepath)
    
    def load_model(self, filename: str):
        """Load model checkpoint."""
        filepath = Config.MODELS_DIR / filename
        if filepath.exists():
            checkpoint = torch.load(filepath, map_location=Config.DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.history = checkpoint.get('history', self.history)
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            self.best_epoch = checkpoint.get('best_epoch', 0)
            print(f"✓ Model loaded from {filepath}")
        else:
            print(f"✗ Model file not found: {filepath}")


class Evaluator:
    """
    Evaluator class for comprehensive model evaluation.
    """
    
    def __init__(self, model: nn.Module, test_loader: DataLoader, model_name: str = "Model"):
        self.model = model.to(Config.DEVICE)
        self.test_loader = test_loader
        self.model_name = model_name
        
        # Results storage
        self.y_true = []
        self.y_pred = []
        self.inference_times = []
    
    def evaluate(self) -> Dict:
        """
        Comprehensive evaluation: accuracy, F1, precision, recall, confusion matrix.
        
        Returns:
            Dictionary of metrics
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING: {self.model_name}")
        print(f"{'='*70}")
        
        self.model.eval()
        self.y_true = []
        self.y_pred = []
        self.inference_times = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Evaluating'):
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                
                # Measure inference time
                start_time = time.time()
                outputs = self.model(images)
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                
                _, predicted = outputs.max(1)
                
                # Store results
                self.y_true.extend(labels.cpu().numpy())
                self.y_pred.extend(predicted.cpu().numpy())
                self.inference_times.append(inference_time / images.size(0))  # Per image
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_true, self.y_pred) * 100
        f1_macro = f1_score(self.y_true, self.y_pred, average='macro')
        precision_macro = precision_score(self.y_true, self.y_pred, average='macro')
        recall_macro = recall_score(self.y_true, self.y_pred, average='macro')
        
        # Model statistics
        num_params = sum(p.numel() for p in self.model.parameters())
        
        # Calculate model size (save temporarily and check size)
        temp_model_path = Config.MODELS_DIR / f'temp_{self.model_name}.pth'
        torch.save(self.model.state_dict(), temp_model_path)
        model_size_mb = temp_model_path.stat().st_size / (1024 * 1024)
        temp_model_path.unlink()  # Delete temporary file
        
        avg_inference_time = np.mean(self.inference_times)
        
        results = {
            'model_name': self.model_name,
            'accuracy': accuracy,
            'f1_score': f1_macro,
            'precision': precision_macro,
            'recall': recall_macro,
            'num_params': num_params,
            'model_size_mb': model_size_mb,
            'avg_inference_time_ms': avg_inference_time
        }
        
        # Print results
        print(f"\nResults for {self.model_name}:")
        print(f"  Accuracy:           {accuracy:.2f}%")
        print(f"  F1-Score (Macro):   {f1_macro:.3f}")
        print(f"  Precision (Macro):  {precision_macro:.3f}")
        print(f"  Recall (Macro):     {recall_macro:.3f}")
        print(f"  Parameters:         {num_params:,}")
        print(f"  Model Size:         {model_size_mb:.2f} MB")
        print(f"  Avg Inference Time: {avg_inference_time:.2f} ms/image")
        print(f"{'='*70}\n")
        
        return results
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Calculate confusion matrix."""
        return confusion_matrix(self.y_true, self.y_pred)
    
    def get_classification_report(self) -> str:
        """Generate detailed classification report."""
        return classification_report(
            self.y_true,
            self.y_pred,
            target_names=Config.CLASS_NAMES,
            digits=3
        )


# ============================================================================
# 6. VISUALIZATION & REPORTING MODULE
# ============================================================================

def plot_learning_curves(history: Dict, model_name: str, save_path: Path = None):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        model_name: Name of the model (for title)
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'{model_name} - Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Learning curves saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    normalize: bool = True,
    save_path: Path = None
):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        model_name: Name of the model (for title)
        normalize: Whether to normalize by row (true labels)
        save_path: Path to save the plot
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = f'{model_name} - Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = f'{model_name} - Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        square=True,
        linewidths=0.5
    )
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.show()


def create_comparison_table(results_list: List[Dict], save_path: Path = None):
    """
    Create and display a comparison table of all models.
    
    Args:
        results_list: List of result dictionaries from Evaluator
        save_path: Path to save the table as CSV
    """
    df = pd.DataFrame(results_list)
    
    # Reorder columns
    column_order = [
        'model_name', 'accuracy', 'num_params', 'model_size_mb', 'avg_inference_time_ms'
    ]
    df = df[column_order]
    
    # Rename columns for display
    df.columns = [
        'Model', 'Accuracy (%)', 'Parameters (M)', 'Size (MB)', 'Inference Time (ms)'
    ]
    
    # Format parameters in millions
    df['Parameters (M)'] = df['Parameters (M)'] / 1e6
    
    # Sort by accuracy (descending)
    df = df.sort_values('Accuracy (%)', ascending=False)
    
    print("\n" + "="*100)
    print("MODEL COMPARISON TABLE")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"✓ Comparison table saved to {save_path}")
    
    return df


def visualize_sample_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    num_samples: int = 12,
    save_path: Path = None
):
    """
    Visualize sample predictions from the model.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Path to save the figure
    """
    model.eval()
    
    # Get a batch of samples
    images, labels = next(iter(test_loader))
    images = images[:num_samples].to(Config.DEVICE)
    labels = labels[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
        probabilities = F.softmax(outputs, dim=1)
    
    # Move to CPU for visualization
    images = images.cpu()
    predicted = predicted.cpu()
    probabilities = probabilities.cpu()
    
    # Plot
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.ravel()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Display image
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        
        # Prepare label text
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted[i]]
        confidence = probabilities[i][predicted[i]] * 100
        
        # Color: green if correct, red if wrong
        color = 'green' if predicted[i] == labels[i] else 'red'
        
        title = f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)"
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sample predictions saved to {save_path}")
    
    plt.show()


def visualize_augmentation_examples(
    dataset: Dataset,
    num_samples: int = 6,
    save_path: Path = None
):
    """
    Visualize original vs augmented images.
    
    Args:
        dataset: Dataset with augmentation transforms
        num_samples: Number of samples to show
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 5))
    
    for i in range(num_samples):
        # Get original image (without augmentation)
        img, label = dataset[i]
        
        # Top row: original
        axes[0, i].imshow(img.squeeze().numpy(), cmap='gray')
        axes[0, i].set_title(f'{Config.CLASS_NAMES[label]}', fontsize=10)
        axes[0, i].axis('off')
        
        # Get augmented version
        img_aug, _ = dataset[i]  # Second call will apply different augmentation
        
        # Bottom row: augmented
        axes[1, i].imshow(img_aug.squeeze().numpy(), cmap='gray')
        axes[1, i].set_title('Augmented', fontsize=10)
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Original', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Augmented', fontsize=12, fontweight='bold')
    
    plt.suptitle('Data Augmentation Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Augmentation examples saved to {save_path}")
    
    plt.show()


# ============================================================================
# 7. MAIN EXECUTION BLOCK
# ============================================================================

def main():
    """
    Main execution function that orchestrates the entire pipeline.
    """
    
    print("\n" + "="*70)
    print("MSTAR-10 SAR IMAGE CLASSIFICATION PROJECT")
    print("="*70)
    print("Objective: Compare lightweight CNN vs pre-trained models")
    print("="*70 + "\n")
    
    # Setup
    Config.setup_directories()
    Config.set_seed()
    
    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Image size: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Learning rate: {Config.LEARNING_RATE}")
    print(f"  Epochs: {Config.NUM_EPOCHS}")
    print(f"  Random seed: {Config.RANDOM_SEED}")
    
    # ========== STEP 1: DATA PREPARATION ==========
    
    # Check for physics-based synthetic data
    use_synthetic = generate_synthetic_data()
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(use_synthetic=use_synthetic)
    
    # ========== STEP 2: TRAIN CUSTOM LIGHTWEIGHT CNN ==========
    
    print("\n" + "="*70)
    print("PHASE 1: CUSTOM LIGHTWEIGHT CNN")
    print("="*70)
    
    # Create model
    custom_model = LightweightCNN(num_classes=Config.NUM_CLASSES, dropout_rate=Config.DROPOUT_RATE)
    print(f"\nCustom CNN Architecture:")
    print(f"  Total parameters: {custom_model.count_parameters():,}")
    
    # Train
    custom_trainer = Trainer(custom_model, train_loader, test_loader, model_name="CustomCNN")
    custom_trainer.train(num_epochs=Config.NUM_EPOCHS)
    
    # Evaluate
    custom_evaluator = Evaluator(custom_model, test_loader, model_name="Custom CNN")
    custom_results = custom_evaluator.evaluate()
    
    # Visualizations
    plot_learning_curves(
        custom_trainer.history,
        "Custom CNN",
        save_path=Config.PLOTS_DIR / "custom_cnn_learning_curves.png"
    )
    
    cm_custom = custom_evaluator.get_confusion_matrix()
    plot_confusion_matrix(
        cm_custom,
        Config.CLASS_NAMES,
        "Custom CNN",
        normalize=True,
        save_path=Config.PLOTS_DIR / "custom_cnn_confusion_matrix.png"
    )
    
    visualize_sample_predictions(
        custom_model,
        test_loader,
        Config.CLASS_NAMES,
        num_samples=12,
        save_path=Config.PLOTS_DIR / "custom_cnn_predictions.png"
    )
    
    # Save classification report
    report = custom_evaluator.get_classification_report()
    with open(Config.RESULTS_DIR / "custom_cnn_classification_report.txt", 'w') as f:
        f.write(report)
    print("✓ Classification report saved\n")
    
    # ========== STEP 3: TRAIN AND EVALUATE PRE-TRAINED MODELS ==========
    
    print("\n" + "="*70)
    print("PHASE 2: PRE-TRAINED MODELS COMPARISON")
    print("="*70)
    
    all_results = [custom_results]  # Start with custom model results
    
    for model_name, enabled in Config.PRETRAINED_MODELS.items():
        if not enabled:
            continue
        
        try:
            # Create pre-trained model
            pretrained_model = create_pretrained_model(model_name, num_classes=Config.NUM_CLASSES)
            
            # Fine-tune (reduced epochs for pre-trained models)
            trainer = Trainer(pretrained_model, train_loader, test_loader, model_name=model_name)
            trainer.train(num_epochs=min(30, Config.NUM_EPOCHS))  # Fine-tuning needs fewer epochs
            
            # Evaluate
            evaluator = Evaluator(pretrained_model, test_loader, model_name=model_name)
            results = evaluator.evaluate()
            all_results.append(results)
            
            # Save confusion matrix
            cm = evaluator.get_confusion_matrix()
            plot_confusion_matrix(
                cm,
                Config.CLASS_NAMES,
                model_name,
                normalize=True,
                save_path=Config.PLOTS_DIR / f"{model_name.lower()}_confusion_matrix.png"
            )
            
        except Exception as e:
            print(f"✗ Error with {model_name}: {e}")
            continue
    
    # ========== STEP 4: COMPARISON TABLE ==========
    
    comparison_df = create_comparison_table(
        all_results,
        save_path=Config.RESULTS_DIR / "model_comparison.csv"
    )
    
    # Save as formatted text
    with open(Config.RESULTS_DIR / "model_comparison.txt", 'w') as f:
        f.write("="*100 + "\n")
        f.write("MODEL COMPARISON TABLE\n")
        f.write("="*100 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n" + "="*100 + "\n")
    
    # ========== FINAL SUMMARY ==========
    
    print("\n" + "="*70)
    print("PROJECT COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {Config.OUTPUT_DIR}")
    print(f"  Models: {Config.MODELS_DIR}")
    print(f"  Plots: {Config.PLOTS_DIR}")
    print(f"  Results: {Config.RESULTS_DIR}")
    print("\nKey Findings:")
    print(f"  Best Accuracy: {comparison_df.iloc[0]['Accuracy (%)']:.2f}% ({comparison_df.iloc[0]['Model']})")
    print(f"  Lightest Model: {comparison_df.iloc[-1]['Model']} "
          f"({comparison_df.iloc[-1]['Parameters (M)']:.1f}M params)")
    print(f"  Fastest Inference: {comparison_df['Inference Time (ms)'].min():.2f} ms/image")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    """
    Entry point of the script.
    
    Before running, ensure:
    1. MSTAR-10 dataset is downloaded and organized in directories
    2. Paths in Config class are set correctly
    3. (Optional) Synthetic data generated using the GitHub repo
    
    Run with: python mstar_sar_classification.py
    """
    
    # Check if data directory exists
    if not Config.DATA_ROOT.exists():
        print("\n" + "="*70)
        print("ERROR: MSTAR data directory not found!")
        print("="*70)
        print(f"\nPlease set up your data at: {Config.DATA_ROOT}")
        print("\nExpected directory structure:")
        print("  mstar_data/")
        print("    train/")
        print("      2S1/")
        print("      BMP2/")
        print("      ... (other classes)")
        print("    test/")
        print("      2S1/")
        print("      BMP2/")
        print("      ... (other classes)")
        print("\nTo obtain MSTAR-10 dataset:")
        print("  1. Download from: https://www.sdms.afrl.af.mil/index.php?collection=mstar")
        print("  2. Organize images into class subdirectories")
        print("  3. Update Config.DATA_ROOT path")
        print("\n" + "="*70 + "\n")
        sys.exit(1)
    
    # Run main pipeline
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
