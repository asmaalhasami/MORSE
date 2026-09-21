"""
Medical Image Classification Benchmark with Experiment Management
Multi-Dataset Classification using timm library + Custom Proposed Model
Author: Research Benchmark Suite
Date: December 2025
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from tqdm import tqdm
import json
import time
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
import glob
from datetime import datetime
warnings.filterwarnings('ignore')

# Import custom proposed model
from proposed import MorphSpectralClassifier

# ==================== GLOBAL CONFIGURATION ====================
CONFIG = {
    # Experiment Settings
    'experiment_name': 'CASE1_with_proposed',
    'output_dir': 'Results',  # Global output folder
    'use_pretrained': False,  # Whether to use pretrained weights
    'seed': 42,

    # Data Split Settings
    'train_split': 0.7,  # 70% training
    'val_split': 0.10,  
    'test_split': 0.20,  

    # Training Hyperparameters
    'epochs': 30,
    'batch_size': 6,
    'learning_rate': 1e-3,
    'weight_decay': 0.01,
    'num_workers': 4,

    # Model Settings
    'input_size': 224,
    'use_mixed_precision': True,  # AMP for faster training

    # Scheduler Settings
    'scheduler_type': 'cosine',  # 'cosine', 'step', 'plateau'
    'warmup_epochs': 5,

    # Early Stopping
    'early_stopping': True,
    'patience': 10,

    # Augmentation
    'use_augmentation': True,
    'augmentation_strength': 'medium',  # 'light', 'medium', 'heavy'
}

# ==================== MODEL CONFIGURATIONS ====================
MODELS_CONFIG = {
    # Proposed Custom Model
    'morph_spectral_net': {
        'name': 'MorphSpectralClassifier',  # Custom model
        'params': '~2.5M',  # Approximate, calculated at runtime
        'type': 'Custom_Hybrid',
        'description': 'Morphological + Spectral CNN with Cross-Attention',
        'is_custom': True  # Flag to identify custom models
    },

    # CNN-Based Models (3)
    'resnet50': {
        'name': 'resnet50.a1_in1k',
        'params': '25.6M',
        'type': 'CNN',
        'description': 'Residual Network 50 layers',
        'is_custom': False
    },
    'convnext_tiny': {
        'name': 'convnext_tiny.fb_in22k_ft_in1k',
        'params': '28.6M',
        'type': 'CNN',
        'description': 'ConvNeXt Tiny - Modern CNN',
        'is_custom': False
    },
    'regnety_040': {
        'name': 'regnety_040.ra3_in1k',
        'params': '20.6M',
        'type': 'CNN',
        'description': 'RegNetY 4.0GF',
        'is_custom': False
    },

    # Transformer-Based Models (4)
    'vit_small': {
        'name': 'vit_small_patch16_224.augreg_in21k_ft_in1k',
        'params': '22.1M',
        'type': 'Transformer',
        'description': 'Vision Transformer Small',
        'is_custom': False
    },
    'deit_small': {
        'name': 'deit_small_patch16_224.fb_in1k',
        'params': '22.1M',
        'type': 'Transformer',
        'description': 'Data-efficient Image Transformer Small',
        'is_custom': False
    },
    'swin_tiny': {
        'name': 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k',
        'params': '28.3M',
        'type': 'Transformer',
        'description': 'Swin Transformer Tiny',
        'is_custom': False
    },
    'maxvit_tiny': {
        'name': 'maxvit_tiny_tf_224.in1k',
        'params': '30.9M',
        'type': 'Transformer',
        'description': 'MaxViT Tiny - Multi-axis attention',
        'is_custom': False
    },
}

# ==================== DATASET CONFIGURATIONS ====================
DATASET_CONFIGS = {
    'skin_disease': {
        'paths': [
            'datasets/skin_disease/Split_smol/train',
            'datasets/skin_disease/Split_smol/val'
        ],
        'classes': ['Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 
                   'Dermatofibroma', 'Melanocytic nevus', 'Melanoma', 
                   'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis', 'Vascular lesion']
    },
    'nail_disease': {
        'paths': [
            'datasets/nail_disease/nail_disease_dataset/train',
            'datasets/nail_disease/nail_disease_dataset/test'
        ],
        'classes': ['healthy', 'onychomycosis', 'psoriasis']
    },
    'eye_disease': {
        'paths': ['datasets/eye_disease/dataset'],
        'classes': ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
    },
    'alzheimers': {
        'paths': ['datasets/alzheimers/combined_images'],
        'classes': ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    }
}

# ==================== EXPERIMENT MANAGER ====================
class ExperimentManager:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)

        # Create auto-increment run folder
        self.run_id = self._get_next_run_id()
        self.run_dir = self.output_dir / f"run_{self.run_id:04d}"
        self.run_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.run_dir / 'models').mkdir(exist_ok=True)
        (self.run_dir / 'logs').mkdir(exist_ok=True)
        (self.run_dir / 'plots').mkdir(exist_ok=True)

        # Results CSV file
        self.results_csv = self.run_dir / 'benchmark_results.csv'
        self.detailed_csv = self.run_dir / 'detailed_metrics.csv'

        # Save config
        with open(self.run_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)

        print(f"\n{'='*80}")
        print(f"Experiment Manager Initialized")
        print(f"{'='*80}")
        print(f"Run ID: {self.run_id}")
        print(f"Output Directory: {self.run_dir}")
        print(f"Pretrained Weights: {config['use_pretrained']}")
        print(f"Seed: {config['seed']}")
        print(f"{'='*80}\n")

    def _get_next_run_id(self):
        existing_runs = list(self.output_dir.glob('run_*'))
        if not existing_runs:
            return 1
        run_numbers = [int(r.name.split('_')[1]) for r in existing_runs]
        return max(run_numbers) + 1

    def save_results(self, results_list):
        """Save results to CSV in formatted order"""
        df = pd.DataFrame(results_list)

        # Define column order
        column_order = [
            'run_id', 'timestamp', 'dataset', 'model', 'model_type', 
            'total_params', 'trainable_params',
            'train_samples', 'val_samples', 'test_samples',
            'best_epoch', 'training_time_minutes',
            'train_acc', 'val_acc', 'test_acc',
            'test_f1_macro', 'test_f1_weighted',
            'test_precision', 'test_recall',
            'pretrained', 'learning_rate', 'batch_size', 'epochs'
        ]

        # Reorder columns
        df = df[[col for col in column_order if col in df.columns]]

        # Sort by dataset and model
        df = df.sort_values(['dataset', 'model'])

        # Save to CSV
        df.to_csv(self.results_csv, index=False)
        print(f"\nResults saved to: {self.results_csv}")

        return df

# ==================== UTILITY FUNCTIONS ====================
def set_seed(seed=42):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# ==================== CUSTOM DATASET CLASS ====================
class MedicalImageDataset(Dataset):
    """Dataset class that loads all images from all paths"""
    def __init__(self, paths, class_names, transform=None, indices=None):
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        self.samples = []

        # Collect all samples from all paths
        for path in paths:
            if not os.path.exists(path):
                print(f"Warning: Path does not exist: {path}")
                continue

            for class_name in class_names:
                class_dir = os.path.join(path, class_name)
                if not os.path.isdir(class_dir):
                    continue

                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))

        # If indices provided, filter samples
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image if loading fails
            return torch.zeros(3, 224, 224), label

# ==================== DATA TRANSFORMS ====================
def get_transforms(config, is_training=True):
    """Get data transforms based on config"""
    input_size = config['input_size']

    if is_training and config['use_augmentation']:
        strength = config['augmentation_strength']

        if strength == 'light':
            return transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif strength == 'medium':
            return transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # heavy
            return transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ==================== DATA LOADING ====================
def create_dataloaders(dataset_name, config):
    """Create train/val/test dataloaders with proper splitting"""
    dataset_config = DATASET_CONFIGS[dataset_name]

    # Load all data
    print(f"Loading {dataset_name} dataset...")
    full_dataset = MedicalImageDataset(
        paths=dataset_config['paths'],
        class_names=dataset_config['classes'],
        transform=None
    )

    total_samples = len(full_dataset)
    print(f"Total samples found: {total_samples}")

    # Create stratified splits
    labels = [sample[1] for sample in full_dataset.samples]
    indices = np.arange(total_samples)

    # First split: train vs (val+test)
    train_indices, temp_indices = train_test_split(
        indices, 
        test_size=(config['val_split'] + config['test_split']),
        stratify=[labels[i] for i in indices],
        random_state=config['seed']
    )

    # Second split: val vs test
    val_ratio = config['val_split'] / (config['val_split'] + config['test_split'])
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_ratio),
        stratify=[labels[i] for i in temp_indices],
        random_state=config['seed']
    )

    print(f"Train samples: {len(train_indices)} ({len(train_indices)/total_samples*100:.1f}%)")
    print(f"Val samples: {len(val_indices)} ({len(val_indices)/total_samples*100:.1f}%)")
    print(f"Test samples: {len(test_indices)} ({len(test_indices)/total_samples*100:.1f}%)\n")

    # Create datasets
    train_dataset = MedicalImageDataset(
        paths=dataset_config['paths'],
        class_names=dataset_config['classes'],
        transform=get_transforms(config, is_training=True),
        indices=train_indices
    )

    val_dataset = MedicalImageDataset(
        paths=dataset_config['paths'],
        class_names=dataset_config['classes'],
        transform=get_transforms(config, is_training=False),
        indices=val_indices
    )

    test_dataset = MedicalImageDataset(
        paths=dataset_config['paths'],
        class_names=dataset_config['classes'],
        transform=get_transforms(config, is_training=False),
        indices=test_indices
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=config['num_workers'], 
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, len(dataset_config['classes'])

# ==================== MODEL CREATION ====================
def create_model(model_key, num_classes, config):
    """Create model from timm or custom implementation"""
    model_config = MODELS_CONFIG[model_key]

    # Check if custom model
    if model_config.get('is_custom', False):
        if model_key == 'morph_spectral_net':
            model = MorphSpectralClassifier(
                in_channels=3,
                num_classes=num_classes,
                morph_radii=(1, 2, 4), 
                base_width=32,
                num_blocks=5,
                dropout=0.3
            )
            print(f"Created custom model: {model_config['name']}")
        else:
            raise ValueError(f"Unknown custom model: {model_key}")
    else:
        # Standard timm model
        model = timm.create_model(
            model_config['name'],
            pretrained=config['use_pretrained'],
            num_classes=num_classes
        )

    return model

# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, loader, criterion, optimizer, device, config):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    scaler = torch.cuda.amp.GradScaler() if (config['use_mixed_precision'] and device.type == 'cuda') else None

    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc='Evaluating', leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = {
        'loss': running_loss / len(loader.dataset),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0)
    }

    return metrics, all_preds, all_labels

# ==================== TRAINING PIPELINE ====================
def train_model(model_key, dataset_name, exp_manager, config, device):
    """Main training function"""
    print(f"\n{'='*80}")
    print(f"Training: {model_key} on {dataset_name}")
    print(f"{'='*80}\n")

    # Create model-specific directory
    model_dir = exp_manager.run_dir / 'models' / f"{dataset_name}_{model_key}"
    model_dir.mkdir(exist_ok=True)

    # Create dataloaders
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(dataset_name, config)

    # Create model
    model = create_model(model_key, num_classes, config)
    model = model.to(device)

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Model: {model_key}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Scheduler
    if config['scheduler_type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    elif config['scheduler_type'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    start_time = time.time()

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print('-' * 50)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, config)

        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if config['scheduler_type'] == 'plateau':
            scheduler.step(val_metrics['accuracy'])
        else:
            scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['lr'].append(current_lr)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val F1 (macro): {val_metrics['f1_macro']:.4f} | LR: {current_lr:.6f}")

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config
            }, model_dir / 'best_model.pth')
            print(f"✓ Best model saved (Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if config['early_stopping'] and patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    training_time = time.time() - start_time

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(model_dir / 'training_history.csv', index=False)

    # Load best model and evaluate on test set
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(model_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    # Save confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    np.save(model_dir / 'confusion_matrix.npy', cm)

    # Prepare results
    results = {
        'run_id': exp_manager.run_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': dataset_name,
        'model': model_key,
        'model_type': MODELS_CONFIG[model_key]['type'],
        'total_params': total_params,
        'trainable_params': trainable_params,
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'best_epoch': best_epoch,
        'training_time_minutes': training_time / 60,
        'train_acc': history['train_acc'][best_epoch-1] if best_epoch <= len(history['train_acc']) else history['train_acc'][-1],
        'val_acc': best_val_acc,
        'test_acc': test_metrics['accuracy'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_f1_weighted': test_metrics['f1_weighted'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'pretrained': config['use_pretrained'],
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'epochs': config['epochs']
    }

    print(f"\n{'='*50}")
    print(f"Final Results:")
    print(f"Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"Test Acc: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"Training Time: {training_time/60:.2f} minutes")
    print(f"{'='*50}\n")

    return results

# ==================== BENCHMARK RUNNER ====================
def run_benchmark(datasets=None, models=None, config=CONFIG):
    """Run full benchmark"""

    # Set seed
    set_seed(config['seed'])

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create experiment manager
    exp_manager = ExperimentManager(config)

    # Set defaults
    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())
    if models is None:
        models = list(MODELS_CONFIG.keys())

    # Print configuration
    print("Models to benchmark:")
    for model_key in models:
        cfg = MODELS_CONFIG[model_key]
        print(f"  - {model_key:20s} | {cfg['type']:15s} | {cfg['params']:8s}")

    print(f"\nDatasets to benchmark: {', '.join(datasets)}\n")

    # Run experiments
    all_results = []

    for dataset_name in datasets:
        for model_key in models:
            try:
                results = train_model(
                    model_key=model_key,
                    dataset_name=dataset_name,
                    exp_manager=exp_manager,
                    config=config,
                    device=device
                )
                all_results.append(results)

                # Save intermediate results
                exp_manager.save_results(all_results)

            except Exception as e:
                print(f"\n❌ Error training {model_key} on {dataset_name}:")
                print(f"   {str(e)}\n")
                import traceback
                traceback.print_exc()
                continue

    # Save final results
    results_df = exp_manager.save_results(all_results)

    # Print summary
    print(f"\n\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}\n")

    for dataset in datasets:
        dataset_results = results_df[results_df['dataset'] == dataset]
        if len(dataset_results) > 0:
            print(f"\n{dataset.upper()}")
            print('-' * 80)
            for _, row in dataset_results.iterrows():
                print(f"{row['model']:20s} | Test Acc: {row['test_acc']:.4f} | "
                      f"F1: {row['test_f1_macro']:.4f} | "
                      f"Time: {row['training_time_minutes']:.1f}m")

    print(f"\n{'='*80}")
    print(f"Benchmark completed!")
    print(f"Results saved to: {exp_manager.run_dir}")
    print(f"{'='*80}\n")

    return results_df, exp_manager

# ==================== MAIN ====================
if __name__ == '__main__':
    # Run benchmark with all datasets and models
    results_df, exp_manager = run_benchmark()
    print(f"\nAll results saved in: {exp_manager.run_dir}")
    print(f"CSV file: {exp_manager.results_csv}")