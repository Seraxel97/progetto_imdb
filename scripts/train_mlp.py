#!/usr/bin/env python3
"""
Enhanced MLP Training Script - Flexible & Robust Training
Trains Multi-Layer Perceptron models with complete flexibility and robust error handling.

🆕 ENHANCED FEATURES:
- ✅ Universal embedding file detection and loading
- ✅ Intelligent training/inference mode detection
- ✅ Robust handling of small datasets and edge cases
- ✅ Automatic fallbacks for missing files or insufficient data
- ✅ Enhanced error recovery and graceful degradation
- ✅ Smart parameter adjustment based on dataset size
- ✅ Comprehensive validation and quality checks

USAGE:
    python scripts/train_mlp.py                                    # Auto-detect and train
    python scripts/train_mlp.py --embeddings-dir data/embeddings   # Specific directory
    python scripts/train_mlp.py --auto-adjust                      # Auto-adjust parameters
    python scripts/train_mlp.py --min-samples 50                   # Minimum samples for training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

# Dynamic project root detection
try:
    CURRENT_FILE = Path(__file__).resolve()
    if CURRENT_FILE.parent.name == 'scripts':
        PROJECT_ROOT = CURRENT_FILE.parent.parent
    else:
        PROJECT_ROOT = CURRENT_FILE.parent
except Exception:
    PROJECT_ROOT = Path.cwd()

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_logging(log_dir):
    """Setup logging configuration with UTF-8 support"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"enhanced_mlp_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure stream handler with UTF-8 encoding
    import sys
    stream_handler = logging.StreamHandler(sys.stdout)
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, Exception):
        pass
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            stream_handler
        ]
    )
    
    return logging.getLogger(__name__)

class EnhancedMLPTrainer:
    """🆕 Enhanced MLP trainer with flexible data handling and robust training"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.device = device
        self.model = None
        self.training_history = {}
        self.model_metadata = {}
        
        self.logger.info(f"🚀 Enhanced MLP Trainer initialized")
        self.logger.info(f"🔧 Device: {self.device}")
    
    def discover_embedding_files(self, embeddings_dir: Path) -> Dict[str, Any]:
        """🆕 Enhanced embedding file discovery with intelligent fallbacks"""
        self.logger.info(f"🔍 Discovering embedding files in: {embeddings_dir}")
        
        discovery_results = {
            'found_files': {},
            'valid_files': {},
            'detected_mode': 'unknown',
            'training_possible': False,
            'recommendations': []
        }
        
        if not embeddings_dir.exists():
            self.logger.error(f"❌ Embeddings directory does not exist: {embeddings_dir}")
            return discovery_results
        
        # List all .npy files
        npy_files = list(embeddings_dir.glob("*.npy"))
        self.logger.info(f"📋 Found {len(npy_files)} .npy files: {[f.name for f in npy_files]}")
        
        # File patterns to look for
        file_patterns = {
            'X_train': 'X_train.npy',
            'y_train': 'y_train.npy',
            'X_val': 'X_val.npy',
            'y_val': 'y_val.npy',
            'X_test': 'X_test.npy',
            'y_test': 'y_test.npy',
            'X_inference': 'X_inference.npy',
            'y_inference': 'y_inference.npy',
            'X_embedded': 'X_embedded.npy',
            'y_labels': 'y_labels.npy'
        }
        
        # Check which files exist
        for file_key, filename in file_patterns.items():
            file_path = embeddings_dir / filename
            if file_path.exists():
                try:
                    # Validate the file by loading it
                    data = np.load(file_path)
                    discovery_results['found_files'][file_key] = {
                        'path': str(file_path),
                        'shape': data.shape,
                        'size_mb': file_path.stat().st_size / (1024 * 1024)
                    }
                    discovery_results['valid_files'][file_key] = data
                    self.logger.info(f"   ✅ {file_key}: {data.shape} ({file_path.stat().st_size / (1024 * 1024):.1f}MB)")
                except Exception as e:
                    self.logger.warning(f"   ⚠️ {file_key}: Invalid file - {str(e)}")
        
        # Detect mode and training possibility
        has_train = 'X_train' in discovery_results['valid_files'] and 'y_train' in discovery_results['valid_files']
        has_val = 'X_val' in discovery_results['valid_files'] and 'y_val' in discovery_results['valid_files']
        has_test = 'X_test' in discovery_results['valid_files'] and 'y_test' in discovery_results['valid_files']
        has_inference = 'X_inference' in discovery_results['valid_files'] and 'y_inference' in discovery_results['valid_files']
        has_embedded = 'X_embedded' in discovery_results['valid_files'] and 'y_labels' in discovery_results['valid_files']
        
        if has_train and has_val:
            discovery_results['detected_mode'] = 'training_standard'
            discovery_results['training_possible'] = True
            discovery_results['recommendations'].append("Standard training mode with train/val splits")
            
        elif has_train:
            discovery_results['detected_mode'] = 'training_single'
            discovery_results['training_possible'] = True
            discovery_results['recommendations'].append("Training with single file - will create validation split")
            
        elif has_embedded:
            discovery_results['detected_mode'] = 'training_embedded'
            discovery_results['training_possible'] = True
            discovery_results['recommendations'].append("Embedded format detected - will create train/val splits")
            
        elif has_test and not has_train:
            discovery_results['detected_mode'] = 'inference_test'
            discovery_results['training_possible'] = False
            discovery_results['recommendations'].append("Only test data available - training not possible")
            
        elif has_inference:
            discovery_results['detected_mode'] = 'inference_only'
            discovery_results['training_possible'] = False
            discovery_results['recommendations'].append("Only inference data available - training not possible")
            
        else:
            discovery_results['detected_mode'] = 'unknown'
            discovery_results['training_possible'] = False
            discovery_results['recommendations'].append("No suitable training data found")
        
        self.logger.info(f"🎯 Detection results:")
        self.logger.info(f"   Mode: {discovery_results['detected_mode']}")
        self.logger.info(f"   Training possible: {discovery_results['training_possible']}")
        
        return discovery_results
    
    def prepare_training_data(self, discovery_results: Dict[str, Any], 
                            min_samples: int = 10) -> Dict[str, Any]:
        """🆕 Prepare training data with intelligent preprocessing and validation"""
        self.logger.info("📊 Preparing training data...")
        
        valid_files = discovery_results['valid_files']
        mode = discovery_results['detected_mode']
        
        data_preparation = {
            'success': False,
            'X_train': None,
            'y_train': None,
            'X_val': None,
            'y_val': None,
            'mode': mode,
            'stats': {},
            'warnings': []
        }
        
        try:
            if mode == 'training_standard':
                # Load existing train/val splits
                X_train = valid_files['X_train']
                y_train = valid_files['y_train'].flatten()
                X_val = valid_files['X_val']
                y_val = valid_files['y_val'].flatten()
                
                self.logger.info("   📂 Using existing train/val splits")
                
            elif mode == 'training_single':
                # Create validation split from training data
                X_train_full = valid_files['X_train']
                y_train_full = valid_files['y_train'].flatten()
                
                # Check if we have labels for stratification
                unique_labels = np.unique(y_train_full)
                valid_labels = unique_labels[unique_labels != -1]  # Remove placeholder labels
                
                if len(valid_labels) > 1 and len(y_train_full[y_train_full != -1]) > min_samples:
                    # Stratified split
                    mask = y_train_full != -1
                    X_labeled = X_train_full[mask]
                    y_labeled = y_train_full[mask]
                    
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
                    )
                    self.logger.info("   🎯 Created stratified train/val split (80/20)")
                else:
                    # Random split or use all for training if too few labeled samples
                    if len(y_train_full) >= min_samples:
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_train_full, y_train_full, test_size=0.2, random_state=42
                        )
                        self.logger.info("   🔀 Created random train/val split (80/20)")
                    else:
                        X_train, y_train = X_train_full, y_train_full
                        X_val, y_val = X_train_full[:min(5, len(X_train_full))], y_train_full[:min(5, len(y_train_full))]
                        data_preparation['warnings'].append(f"Very small dataset ({len(X_train_full)} samples)")
                        self.logger.warning(f"   ⚠️ Very small dataset - using minimal validation set")
                
            elif mode == 'training_embedded':
                # Create splits from embedded data
                X_embedded = valid_files['X_embedded']
                y_labels = valid_files['y_labels'].flatten()
                
                # Filter out placeholder labels for splitting
                valid_mask = y_labels != -1
                
                if valid_mask.sum() >= min_samples:
                    X_valid = X_embedded[valid_mask]
                    y_valid = y_labels[valid_mask]
                    
                    if len(np.unique(y_valid)) > 1:
                        # Stratified split
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_valid, y_valid, test_size=0.2, random_state=42, stratify=y_valid
                        )
                        self.logger.info("   🎯 Created stratified split from embedded data")
                    else:
                        # Random split
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_valid, y_valid, test_size=0.2, random_state=42
                        )
                        self.logger.info("   🔀 Created random split from embedded data")
                else:
                    # Insufficient labeled data for training
                    data_preparation['success'] = False
                    data_preparation['warnings'].append(f"Insufficient labeled data ({valid_mask.sum()} samples)")
                    self.logger.error(f"   ❌ Insufficient labeled data for training")
                    return data_preparation
            
            else:
                # No suitable training mode
                data_preparation['success'] = False
                data_preparation['warnings'].append(f"Training not possible with mode: {mode}")
                self.logger.error(f"   ❌ Training not possible with mode: {mode}")
                return data_preparation
            
            # Validate prepared data
            if X_train is None or len(X_train) == 0:
                data_preparation['warnings'].append("No training data available")
                return data_preparation
            
            # Clean labels (ensure binary classification)
            y_train = self.clean_labels(y_train)
            y_val = self.clean_labels(y_val)
            
            # Final validation
            train_samples = len(X_train)
            val_samples = len(X_val)
            
            if train_samples < min_samples:
                data_preparation['warnings'].append(f"Training set too small ({train_samples} < {min_samples})")
                
            # Calculate statistics
            unique_train_labels = np.unique(y_train)
            unique_val_labels = np.unique(y_val)
            
            stats = {
                'train_samples': train_samples,
                'val_samples': val_samples,
                'total_samples': train_samples + val_samples,
                'feature_dim': X_train.shape[1],
                'train_labels': unique_train_labels.tolist(),
                'val_labels': unique_val_labels.tolist(),
                'train_label_dist': {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
                'val_label_dist': {int(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))}
            }
            
            data_preparation.update({
                'success': True,
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'stats': stats
            })
            
            self.logger.info(f"   📊 Training data prepared successfully:")
            self.logger.info(f"      Train: {train_samples} samples")
            self.logger.info(f"      Val: {val_samples} samples")
            self.logger.info(f"      Features: {X_train.shape[1]}")
            self.logger.info(f"      Train labels: {stats['train_label_dist']}")
            self.logger.info(f"      Val labels: {stats['val_label_dist']}")
            
            return data_preparation
            
        except Exception as e:
            self.logger.error(f"   ❌ Error preparing training data: {str(e)}")
            data_preparation['warnings'].append(f"Data preparation failed: {str(e)}")
            return data_preparation
    
    def clean_labels(self, labels: np.ndarray) -> np.ndarray:
        """Clean and normalize labels for binary classification"""
        # Convert to numpy array if not already
        labels = np.array(labels)
        
        # Handle placeholder labels (-1) by converting to 0
        labels = np.where(labels == -1, 0, labels)
        
        # Ensure binary labels (0 or 1)
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            # Single class - create a small amount of opposite class for training
            if unique_labels[0] == 0:
                labels[-1] = 1  # Change last label to create binary
            else:
                labels[-1] = 0
        elif len(unique_labels) > 2:
            # Multi-class to binary: >0.5 becomes 1, <=0.5 becomes 0
            labels = (labels > 0.5).astype(int)
        
        return labels.astype(int)
    
    def auto_adjust_hyperparameters(self, data_stats: Dict[str, Any]) -> Dict[str, Any]:
        """🆕 Automatically adjust hyperparameters based on dataset characteristics"""
        self.logger.info("🔧 Auto-adjusting hyperparameters based on dataset...")
        
        total_samples = data_stats['total_samples']
        feature_dim = data_stats['feature_dim']
        
        # Base hyperparameters
        params = {
            'input_dim': feature_dim,
            'hidden_dims': [256, 128],
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'early_stopping_patience': 10
        }
        
        # Adjust based on dataset size
        if total_samples < 100:
            # Very small dataset
            params.update({
                'hidden_dims': [64, 32],
                'dropout': 0.1,
                'learning_rate': 0.01,
                'batch_size': min(8, total_samples // 4),
                'epochs': 30,
                'early_stopping_patience': 5
            })
            self.logger.info("   📉 Small dataset adjustments applied")
            
        elif total_samples < 1000:
            # Medium dataset
            params.update({
                'hidden_dims': [128, 64],
                'dropout': 0.2,
                'batch_size': min(16, total_samples // 8),
                'epochs': 40,
                'early_stopping_patience': 7
            })
            self.logger.info("   📊 Medium dataset adjustments applied")
            
        elif total_samples > 10000:
            # Large dataset
            params.update({
                'hidden_dims': [512, 256, 128],
                'dropout': 0.4,
                'batch_size': 64,
                'epochs': 100,
                'early_stopping_patience': 15
            })
            self.logger.info("   📈 Large dataset adjustments applied")
        
        # Adjust based on feature dimension
        if feature_dim < 200:
            # Low dimensional features
            params['hidden_dims'] = [dim // 2 for dim in params['hidden_dims']]
            self.logger.info("   📏 Low dimensional adjustments applied")
            
        elif feature_dim > 1000:
            # High dimensional features
            params['hidden_dims'] = [dim * 2 for dim in params['hidden_dims']]
            params['dropout'] += 0.1
            self.logger.info("   📏 High dimensional adjustments applied")
        
        # Ensure minimum architecture
        params['hidden_dims'] = [max(16, dim) for dim in params['hidden_dims']]
        params['batch_size'] = max(1, min(params['batch_size'], total_samples // 4))
        
        self.logger.info(f"   📋 Final parameters: {params}")
        
        return params

class HateSpeechMLP(nn.Module):
    """Enhanced MLP architecture with flexible configuration"""
    
    def __init__(self, input_dim=384, hidden_dims=[256, 128], dropout=0.3):
        super(HateSpeechMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))
            # Activation
            layers.append(nn.ReLU())
            # Dropout (less dropout in later layers)
            dropout_rate = dropout if i < len(hidden_dims) - 2 else dropout * 0.7
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def enhanced_train_model(embeddings_dir=None, output_dir=None, auto_adjust=True, 
                        min_samples=10, skip_training_if_insufficient=True,
                        custom_params=None, logger=None, **kwargs):
    """🆕 Enhanced model training with comprehensive error handling and fallbacks"""
    if logger is None:
        temp_log_dir = Path("logs") if not output_dir else Path(output_dir) / "logs"
        logger = setup_logging(temp_log_dir)
    
    try:
        logger.info("=" * 60)
        logger.info("🆕 ENHANCED MLP TRAINING STARTED")
        logger.info("=" * 60)
        
        # Initialize trainer
        trainer = EnhancedMLPTrainer(logger)
        
        # Setup paths
        if embeddings_dir is None:
            embeddings_dir = PROJECT_ROOT / "data" / "embeddings"
        else:
            embeddings_dir = Path(embeddings_dir)
            
        if output_dir is None:
            output_dir = PROJECT_ROOT / "results"
        else:
            output_dir = Path(output_dir)
        
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Embeddings directory: {embeddings_dir}")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"🔧 Auto-adjust parameters: {auto_adjust}")
        logger.info(f"📊 Minimum samples: {min_samples}")
        
        # Discover embedding files
        discovery = trainer.discover_embedding_files(embeddings_dir)
        
        if not discovery['training_possible']:
            if skip_training_if_insufficient:
                logger.warning("⚠️ Training not possible - skipping MLP training")
                
                # Create skip status file
                skip_status = {
                    'status': 'skipped',
                    'reason': 'training_not_possible',
                    'timestamp': datetime.now().isoformat(),
                    'discovery_results': discovery,
                    'model_type': 'MLP'
                }
                
                status_file = output_dir / "mlp_training_status.json"
                with open(status_file, 'w') as f:
                    json.dump(skip_status, f, indent=2)
                
                return {
                    'success': True,
                    'skipped': True,
                    'reason': 'training_not_possible',
                    'discovery_results': discovery,
                    'status_file': str(status_file)
                }
            else:
                raise ValueError("Training not possible with available data")
        
        # Prepare training data
        data_prep = trainer.prepare_training_data(discovery, min_samples)
        
        if not data_prep['success']:
            if skip_training_if_insufficient:
                logger.warning("⚠️ Data preparation failed - skipping MLP training")
                
                skip_status = {
                    'status': 'skipped',
                    'reason': 'data_preparation_failed',
                    'timestamp': datetime.now().isoformat(),
                    'warnings': data_prep['warnings'],
                    'model_type': 'MLP'
                }
                
                status_file = output_dir / "mlp_training_status.json"
                with open(status_file, 'w') as f:
                    json.dump(skip_status, f, indent=2)
                
                return {
                    'success': True,
                    'skipped': True,
                    'reason': 'data_preparation_failed',
                    'warnings': data_prep['warnings'],
                    'status_file': str(status_file)
                }
            else:
                raise ValueError(f"Data preparation failed: {data_prep['warnings']}")
        
        # Get training data
        X_train = data_prep['X_train']
        y_train = data_prep['y_train']
        X_val = data_prep['X_val']
        y_val = data_prep['y_val']
        stats = data_prep['stats']
        
        # Auto-adjust hyperparameters or use custom ones
        if auto_adjust and custom_params is None:
            params = trainer.auto_adjust_hyperparameters(stats)
        elif custom_params is not None:
            params = custom_params
            logger.info("🔧 Using custom parameters")
        else:
            # Default parameters
            params = {
                'input_dim': stats['feature_dim'],
                'hidden_dims': [256, 128],
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,
                'early_stopping_patience': 10
            }
            logger.info("🔧 Using default parameters")
        
        # Override with any provided kwargs
        params.update(kwargs)
        
        logger.info(f"📋 Training parameters:")
        for key, value in params.items():
            logger.info(f"   {key}: {value}")
        
        # Create model
        model = HateSpeechMLP(
            input_dim=params['input_dim'],
            hidden_dims=params['hidden_dims'],
            dropout=params['dropout']
        ).to(device)
        
        logger.info("🧠 Model architecture:")
        logger.info(str(model))
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"📊 Total parameters: {total_params:,}")
        
        # Create data loaders
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Setup training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        history = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("🚀 Starting training...")
        
        # Training loop
        for epoch in range(params['epochs']):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            # Update history
            history['train_losses'].append(train_loss)
            history['train_accuracies'].append(train_acc)
            history['val_losses'].append(val_loss)
            history['val_accuracies'].append(val_acc)
            
            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{params['epochs']}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if new_lr != old_lr:
                logger.info(f"   📈 Learning rate updated: {old_lr:.6f} → {new_lr:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                model_path = models_dir / "mlp_model.pth"
                torch.save(model.state_dict(), model_path)
                logger.info(f"   ✅ New best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                
            if patience_counter >= params['early_stopping_patience']:
                logger.info(f"⏹️ Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(torch.load(models_dir / "mlp_model.pth"))
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_predictions = (val_outputs > 0.5).cpu().numpy().flatten()
            val_targets = y_val
        
        final_accuracy = accuracy_score(val_targets, val_predictions)
        
        logger.info(f"📊 Final validation accuracy: {final_accuracy:.4f}")
        
        # Create comprehensive metadata
        metadata = {
            'model_type': 'Enhanced_MLP',
            'training_timestamp': datetime.now().isoformat(),
            'discovery_results': discovery,
            'data_preparation': data_prep,
            'parameters': params,
            'training_history': history,
            'final_performance': {
                'validation_accuracy': final_accuracy,
                'best_validation_loss': best_val_loss,
                'total_epochs': len(history['train_losses'])
            },
            'model_info': {
                'total_parameters': total_params,
                'architecture': params['hidden_dims'],
                'input_dimension': params['input_dim']
            }
        }
        
        # Save metadata
        metadata_file = models_dir / "mlp_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save complete model
        torch.save(model, models_dir / "mlp_model_complete.pth")
        
        # Create success status
        success_status = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'model_type': 'MLP',
            'performance': {
                'accuracy': float(final_accuracy),
                'final_val_accuracy': float(history['val_accuracies'][-1]),
                'best_val_accuracy': float(max(history['val_accuracies'])),
                'total_epochs': len(history['train_losses'])
            },
            'files': {
                'model': 'models/mlp_model.pth',
                'metadata': 'models/mlp_metadata.json',
                'complete_model': 'models/mlp_model_complete.pth'
            }
        }
        
        status_file = output_dir / "mlp_training_status.json"
        with open(status_file, 'w') as f:
            json.dump(success_status, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("✅ ENHANCED MLP TRAINING COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"📊 Final accuracy: {final_accuracy:.4f}")
        logger.info(f"💾 Model saved: {models_dir / 'mlp_model.pth'}")
        logger.info(f"📄 Metadata saved: {metadata_file}")
        
        return {
            'success': True,
            'skipped': False,
            'model_path': str(models_dir / "mlp_model.pth"),
            'metadata_path': str(metadata_file),
            'final_accuracy': final_accuracy,
            'training_history': history,
            'metadata': metadata,
            'status_file': str(status_file)
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced MLP training failed: {str(e)}")
        
        # Save error status
        if output_dir:
            error_status = {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error_message': str(e),
                'model_type': 'MLP'
            }
            try:
                status_file = Path(output_dir) / "mlp_training_status.json"
                with open(status_file, 'w') as f:
                    json.dump(error_status, f, indent=2)
            except:
                pass
        
        return {
            'success': False,
            'skipped': False,
            'error': str(e),
            'embeddings_dir': str(embeddings_dir) if embeddings_dir else None,
            'output_dir': str(output_dir) if output_dir else None
        }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced MLP Training for Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🆕 ENHANCED FEATURES:
- Universal embedding file detection
- Intelligent training/inference mode detection
- Automatic parameter adjustment based on dataset size
- Robust error handling and graceful fallbacks

Examples:
  python scripts/train_mlp.py                                    # Auto-detect and train
  python scripts/train_mlp.py --embeddings-dir data/embeddings   # Specific directory
  python scripts/train_mlp.py --auto-adjust                      # Auto-adjust parameters
  python scripts/train_mlp.py --min-samples 50                   # Minimum samples for training
        """
    )
    
    # Path arguments
    parser.add_argument('--embeddings-dir', type=str, default=None,
                       help='Directory containing embedding files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save model and results')
    
    # Training configuration
    parser.add_argument('--auto-adjust', action='store_true', default=True,
                       help='Automatically adjust hyperparameters based on dataset')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='Minimum samples required for training')
    parser.add_argument('--skip-if-insufficient', action='store_true', default=True,
                       help='Skip training if insufficient data (instead of failing)')
    
    # Model parameters (optional overrides)
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout rate')
    
    # Utility flags
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose logging')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup logging
    if args.output_dir:
        log_dir = Path(args.output_dir) / "logs"
    else:
        log_dir = Path("logs")
    
    logger = setup_logging(log_dir)
    
    # Configure logging level
    if args.quiet:
        logger.setLevel(logging.WARNING)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    
    # Build custom parameters from args
    custom_params = {}
    if args.epochs is not None:
        custom_params['epochs'] = args.epochs
    if args.learning_rate is not None:
        custom_params['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        custom_params['batch_size'] = args.batch_size
    if args.dropout is not None:
        custom_params['dropout'] = args.dropout
    
    # Run enhanced training
    result = enhanced_train_model(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        auto_adjust=args.auto_adjust,
        min_samples=args.min_samples,
        skip_training_if_insufficient=args.skip_if_insufficient,
        custom_params=custom_params if custom_params else None,
        logger=logger
    )
    
    if result['success']:
        if result.get('skipped'):
            logger.warning(f"⚠️ Training skipped: {result['reason']}")
            return 0
        else:
            logger.info("✅ Enhanced MLP training completed successfully!")
            return 0
    else:
        logger.error(f"❌ Enhanced MLP training failed: {result.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
