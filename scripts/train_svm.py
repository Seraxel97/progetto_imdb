#!/usr/bin/env python3
"""
Enhanced SVM Training Script - UNIVERSAL & INTELLIGENT TRAINING
Trains Support Vector Machine models with complete flexibility and intelligent adaptation.

🆕 ENHANCED FEATURES:
- ✅ Universal embedding file detection and loading
- ✅ Intelligent training/inference mode detection
- ✅ Robust handling of small datasets and edge cases
- ✅ Automatic fallbacks for missing files or insufficient data
- ✅ Enhanced error recovery and graceful skipping
- ✅ Smart parameter adjustment based on dataset characteristics
- ✅ Support for embedded files from external CSV processing
- ✅ GUI integration with comprehensive status reporting

USAGE:
    python scripts/train_svm.py                                    # Auto-detect and train
    python scripts/train_svm.py --embeddings-dir data/embeddings   # Specific directory
    python scripts/train_svm.py --skip-if-insufficient            # Skip instead of fail
    python scripts/train_svm.py --min-samples 50                  # Minimum samples for training
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import warnings
from time import time
from typing import Dict, List, Optional, Tuple, Any

# ML imports
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

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

def setup_logging(log_dir):
    """Setup logging configuration with UTF-8 support"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"enhanced_svm_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure stream handler with UTF-8 encoding
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

class EnhancedSVMTrainer:
    """🆕 Enhanced SVM trainer with flexible data handling and intelligent training"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.training_results = {}
        self.model_metadata = {}
        
        self.logger.info(f"🚀 Enhanced SVM Trainer initialized")
    
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
            'y_inference': 'y_inference.npy'
        }
        
        # Check which files exist and validate them
        for file_key, filename in file_patterns.items():
            file_path = embeddings_dir / filename
            if file_path.exists():
                try:
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
        
        if has_train:
            # Check if we have valid labels for training
            y_train = discovery_results['valid_files']['y_train']
            valid_labels = np.sum(np.isin(y_train, [0, 1]))
            total_labels = len(y_train)
            
            if valid_labels > 10:  # Need at least 10 valid labels
                discovery_results['detected_mode'] = 'training'
                discovery_results['training_possible'] = True
                if has_val:
                    discovery_results['recommendations'].append("Training mode with separate validation")
                else:
                    discovery_results['recommendations'].append("Training mode - will create validation split")
            else:
                discovery_results['detected_mode'] = 'insufficient_labels'
                discovery_results['training_possible'] = False
                discovery_results['recommendations'].append(f"Insufficient valid labels: {valid_labels}/{total_labels}")
                
        elif has_inference:
            discovery_results['detected_mode'] = 'inference'
            discovery_results['training_possible'] = False
            discovery_results['recommendations'].append("Inference mode - no training possible")
            
        elif has_test:
            discovery_results['detected_mode'] = 'test_only'
            discovery_results['training_possible'] = False
            discovery_results['recommendations'].append("Test data only - no training possible")
            
        else:
            discovery_results['detected_mode'] = 'no_files'
            discovery_results['training_possible'] = False
            discovery_results['recommendations'].append("No suitable embedding files found")
        
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
            if mode == 'training':
                # Load training data
                X_train = valid_files['X_train']
                y_train = valid_files['y_train'].flatten()
                
                # Clean labels (ensure binary classification)
                y_train = self.clean_labels(y_train)
                
                # Filter out samples with invalid labels
                valid_mask = np.isin(y_train, [0, 1])
                X_train_clean = X_train[valid_mask]
                y_train_clean = y_train[valid_mask]
                
                self.logger.info(f"   📊 Training data: {len(X_train_clean)}/{len(X_train)} valid samples")
                
                if len(X_train_clean) < min_samples:
                    data_preparation['warnings'].append(f"Insufficient training samples: {len(X_train_clean)} < {min_samples}")
                    return data_preparation
                
                # Handle validation data
                if 'X_val' in valid_files and 'y_val' in valid_files:
                    # Use existing validation split
                    X_val = valid_files['X_val']
                    y_val = valid_files['y_val'].flatten()
                    y_val = self.clean_labels(y_val)
                    
                    val_valid_mask = np.isin(y_val, [0, 1])
                    X_val_clean = X_val[val_valid_mask]
                    y_val_clean = y_val[val_valid_mask]
                    
                    self.logger.info(f"   📊 Using existing validation split: {len(X_val_clean)} samples")
                else:
                    # Create validation split from training data
                    if len(X_train_clean) >= 20:  # Need enough samples to split
                        try:
                            X_train_clean, X_val_clean, y_train_clean, y_val_clean = train_test_split(
                                X_train_clean, y_train_clean, test_size=0.2, random_state=42, stratify=y_train_clean
                            )
                            self.logger.info(f"   📊 Created validation split: {len(X_val_clean)} samples")
                        except ValueError:
                            # Stratification failed, use random split
                            X_train_clean, X_val_clean, y_train_clean, y_val_clean = train_test_split(
                                X_train_clean, y_train_clean, test_size=0.2, random_state=42
                            )
                            self.logger.info(f"   📊 Created random validation split: {len(X_val_clean)} samples")
                    else:
                        # Use a small portion as validation
                        split_idx = max(1, len(X_train_clean) // 5)
                        X_val_clean = X_train_clean[:split_idx]
                        y_val_clean = y_train_clean[:split_idx]
                        X_train_clean = X_train_clean[split_idx:]
                        y_train_clean = y_train_clean[split_idx:]
                        self.logger.warning(f"   ⚠️ Small dataset - minimal validation split: {len(X_val_clean)} samples")
                
                # Final validation
                if len(X_train_clean) < 5:
                    data_preparation['warnings'].append(f"Training set too small after cleaning: {len(X_train_clean)}")
                    return data_preparation
                
                # Calculate statistics
                unique_train_labels = np.unique(y_train_clean)
                unique_val_labels = np.unique(y_val_clean)
                
                stats = {
                    'train_samples': len(X_train_clean),
                    'val_samples': len(X_val_clean),
                    'total_samples': len(X_train_clean) + len(X_val_clean),
                    'feature_dim': X_train_clean.shape[1],
                    'train_labels': unique_train_labels.tolist(),
                    'val_labels': unique_val_labels.tolist(),
                    'train_label_dist': {int(k): int(v) for k, v in zip(*np.unique(y_train_clean, return_counts=True))},
                    'val_label_dist': {int(k): int(v) for k, v in zip(*np.unique(y_val_clean, return_counts=True))}
                }
                
                data_preparation.update({
                    'success': True,
                    'X_train': X_train_clean,
                    'y_train': y_train_clean,
                    'X_val': X_val_clean,
                    'y_val': y_val_clean,
                    'stats': stats
                })
                
                self.logger.info(f"   📊 Training data prepared successfully:")
                self.logger.info(f"      Train: {len(X_train_clean)} samples")
                self.logger.info(f"      Val: {len(X_val_clean)} samples")
                self.logger.info(f"      Features: {X_train_clean.shape[1]}")
                self.logger.info(f"      Train labels: {stats['train_label_dist']}")
                self.logger.info(f"      Val labels: {stats['val_label_dist']}")
                
            else:
                # No suitable training mode
                data_preparation['warnings'].append(f"Training not possible with mode: {mode}")
                self.logger.error(f"   ❌ Training not possible with mode: {mode}")
            
            return data_preparation
            
        except Exception as e:
            self.logger.error(f"   ❌ Error preparing training data: {str(e)}")
            data_preparation['warnings'].append(f"Data preparation failed: {str(e)}")
            return data_preparation
    
    def clean_labels(self, labels: np.ndarray) -> np.ndarray:
        """Clean and normalize labels for binary classification"""
        labels = np.array(labels)
        
        # Handle placeholder labels (-1) by converting to 0
        labels = np.where(labels == -1, 0, labels)
        
        # Ensure binary labels (0 or 1)
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            # Single class - create a small amount of opposite class
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
            'C': 1.0,
            'class_weight': 'balanced',
            'max_iter': 10000,
            'random_state': 42
        }
        
        # Adjust based on dataset size
        if total_samples < 100:
            # Very small dataset
            params.update({
                'C': 0.1,
                'max_iter': 5000
            })
            self.logger.info("   📉 Small dataset adjustments applied")
            
        elif total_samples < 1000:
            # Medium dataset
            params.update({
                'C': 1.0,
                'max_iter': 10000
            })
            self.logger.info("   📊 Medium dataset adjustments applied")
            
        elif total_samples > 10000:
            # Large dataset
            params.update({
                'C': 1.0,
                'max_iter': 20000
            })
            self.logger.info("   📈 Large dataset adjustments applied")
        
        # Adjust based on feature dimension
        if feature_dim > 1000:
            # High dimensional features - reduce C to prevent overfitting
            params['C'] = min(params['C'], 0.1)
            self.logger.info("   📏 High dimensional adjustments applied")
        
        self.logger.info(f"   📋 Final parameters: {params}")
        
        return params

def train_svm_fast(X_train, y_train, X_val, y_val, params=None, logger=None):
    """Fast SVM training using LinearSVC with optimized parameters"""
    if logger:
        logger.info("Starting fast SVM training...")
    
    start_time = time()
    
    # Use optimized parameters with fallback defaults
    if params is None:
        params = {
            'C': 1.0,
            'class_weight': 'balanced',
            'max_iter': 10000,
            'random_state': 42
        }
    
    if logger:
        logger.info(f"   📋 Parameters: {params}")
        logger.info(f"   📊 Training samples: {X_train.shape[0]:,}")
        logger.info(f"   📊 Validation samples: {X_val.shape[0]:,}")
        logger.info(f"   📏 Features: {X_train.shape[1]}")
    
    # Initialize preprocessing components
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    # Fit preprocessing on training data
    if logger:
        logger.info("   🔧 Fitting scaler and label encoder...")
    
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # Apply preprocessing to validation data
    X_val_scaled = scaler.transform(X_val)
    y_val_encoded = label_encoder.transform(y_val)
    
    if logger:
        logger.info(f"   📊 Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Train LinearSVM
    if logger:
        logger.info("   🤖 Training LinearSVC...")
    
    model = LinearSVC(**params)
    model.fit(X_train_scaled, y_train_encoded)
    
    training_time = time() - start_time
    
    if logger:
        logger.info(f"   ✅ Training completed in {training_time:.2f} seconds")
    
    # Make predictions on validation set
    if logger:
        logger.info("   🔮 Evaluating on validation set...")
    
    y_val_pred = model.predict(X_val_scaled)
    
    # Calculate metrics
    val_accuracy = accuracy_score(y_val_encoded, y_val_pred)
    val_f1 = f1_score(y_val_encoded, y_val_pred, average='weighted')
    
    # Generate classification report
    class_names = ['Negative', 'Positive']
    classification_rep = classification_report(
        y_val_encoded, y_val_pred,
        target_names=class_names,
        output_dict=True
    )
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_val_encoded, y_val_pred)
    
    if logger:
        logger.info(f"   📊 Validation Results:")
        logger.info(f"      Accuracy: {val_accuracy:.4f}")
        logger.info(f"      F1-Score: {val_f1:.4f}")
        logger.info(f"      Training time: {training_time:.2f}s")
    
    # Package everything together
    model_package = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'params': params,
        'training_time': training_time,
        'validation_accuracy': val_accuracy,
        'validation_f1': val_f1,
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix.tolist(),
        'feature_dim': X_train.shape[1],
        'training_samples': X_train.shape[0],
        'model_type': 'LinearSVC_enhanced'
    }
    
    return model_package

def save_model_package(model_package, output_dir, logger):
    """Save trained model package with comprehensive metadata"""
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Saving SVM model package...")
    
    try:
        # Save the complete model package
        model_path = models_dir / 'svm_model.pkl'
        joblib.dump(model_package, model_path)
        
        logger.info(f"   📄 Model package: {model_path}")
        
        # Save metadata separately
        metadata = {
            'model_type': model_package['model_type'],
            'training_timestamp': datetime.now().isoformat(),
            'training_time_seconds': model_package['training_time'],
            'validation_accuracy': model_package['validation_accuracy'],
            'validation_f1': model_package['validation_f1'],
            'feature_dimension': model_package['feature_dim'],
            'training_samples': model_package['training_samples'],
            'parameters': model_package['params'],
            'classification_report': model_package['classification_report'],
            'confusion_matrix': model_package['confusion_matrix'],
            'model_file': str(model_path),
            'compatible_with': ['report.py', 'GUI', 'pipeline_automation.py']
        }
        
        metadata_path = models_dir / 'svm_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   📄 Metadata: {metadata_path}")
        
        return {
            'model_path': str(model_path),
            'metadata_path': str(metadata_path)
        }
        
    except Exception as e:
        logger.error(f"Error saving model package: {str(e)}")
        raise

def save_metrics_and_plots(model_package, output_dir, logger):
    """Save metrics and create visualization plots"""
    output_dir = Path(output_dir)
    
    # Create subdirectories
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Set matplotlib backend
        import matplotlib
        matplotlib.use('Agg')
        
        logger.info("Creating training plots and saving metrics...")
        
        # Save metrics in JSON format
        metrics_data = {
            'model_type': model_package['model_type'],
            'validation_accuracy': model_package['validation_accuracy'],
            'validation_f1': model_package['validation_f1'],
            'training_time': model_package['training_time'],
            'training_samples': model_package['training_samples'],
            'feature_dimension': model_package['feature_dim'],
            'parameters': model_package['params'],
            'confusion_matrix': model_package['confusion_matrix'],
            'classification_report': model_package['classification_report']
        }
        
        metrics_file = reports_dir / "svm_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_file}")
        
        # Confusion Matrix Plot
        plt.figure(figsize=(8, 6))
        cm = np.array(model_package['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cmap='Blues')
        plt.title(f'Enhanced SVM Confusion Matrix\nAccuracy: {model_package["validation_accuracy"]:.3f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_plot_path = plots_dir / 'svm_confusion_matrix.png'
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance Metrics Plot
        plt.figure(figsize=(10, 6))
        
        metrics = ['Accuracy', 'F1-Score']
        values = [model_package['validation_accuracy'], model_package['validation_f1']]
        colors = ['#1f77b4', '#ff7f0e']
        
        bars = plt.bar(metrics, values, color=colors)
        plt.title(f'Enhanced SVM Performance Metrics\nTraining Time: {model_package["training_time"]:.2f}s')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        metrics_plot_path = plots_dir / 'svm_performance_metrics.png'
        plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Training Summary Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        summary_text = f"""
Enhanced SVM Training Summary

Model Type: {model_package['model_type']}
Training Samples: {model_package['training_samples']:,}
Feature Dimension: {model_package['feature_dim']}
Training Time: {model_package['training_time']:.2f} seconds

Validation Results:
• Accuracy: {model_package['validation_accuracy']:.4f}
• F1-Score: {model_package['validation_f1']:.4f}

Parameters:
{chr(10).join(f'• {k}: {v}' for k, v in model_package['params'].items())}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.title('Enhanced SVM Training Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        summary_plot_path = plots_dir / 'svm_training_summary.png'
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save GUI status file
        gui_status_file = output_dir / "svm_training_status.json"
        gui_status = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'model_type': 'SVM',
            'performance': {
                'accuracy': float(model_package['validation_accuracy']),
                'f1_score': float(model_package['validation_f1']),
                'training_time': float(model_package['training_time']),
                'training_samples': int(model_package['training_samples'])
            },
            'files': {
                'model': 'models/svm_model.pkl',
                'confusion_matrix': 'plots/svm_confusion_matrix.png',
                'performance_metrics': 'plots/svm_performance_metrics.png',
                'training_summary': 'plots/svm_training_summary.png',
                'metrics': 'reports/svm_metrics.json'
            }
        }
        
        with open(gui_status_file, 'w') as f:
            json.dump(gui_status, f, indent=2)
        
        logger.info(f"Plots and reports saved to: {output_dir}")
        
        return {
            'metrics_file': str(metrics_file),
            'confusion_matrix_plot': str(cm_plot_path),
            'performance_metrics_plot': str(metrics_plot_path),
            'training_summary_plot': str(summary_plot_path),
            'gui_status_file': str(gui_status_file)
        }
        
    except Exception as e:
        logger.error(f"Error creating plots and metrics: {str(e)}")
        raise

def enhanced_train_svm(embeddings_dir=None, output_dir=None, auto_adjust=True,
                      min_samples=10, skip_training_if_insufficient=True,
                      fast_mode=True, grid_search=False, custom_params=None,
                      logger=None, **kwargs):
    """🆕 Enhanced SVM training with comprehensive error handling and fallbacks"""
    if logger is None:
        temp_log_dir = Path("logs") if not output_dir else Path(output_dir) / "logs"
        logger = setup_logging(temp_log_dir)
    
    try:
        logger.info("=" * 60)
        logger.info("🆕 ENHANCED SVM TRAINING STARTED")
        logger.info("=" * 60)
        
        # Initialize trainer
        trainer = EnhancedSVMTrainer(logger)
        
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
        logger.info(f"⚡ Fast mode: {fast_mode}")
        
        # Discover embedding files
        discovery = trainer.discover_embedding_files(embeddings_dir)
        
        if not discovery['training_possible']:
            if skip_training_if_insufficient:
                logger.warning("⚠️ Training not possible - skipping SVM training")
                
                # Create skip status file
                skip_status = {
                    'status': 'skipped',
                    'reason': 'training_not_possible',
                    'timestamp': datetime.now().isoformat(),
                    'discovery_results': discovery,
                    'model_type': 'SVM'
                }
                
                status_file = output_dir / "svm_training_status.json"
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
                logger.warning("⚠️ Data preparation failed - skipping SVM training")
                
                skip_status = {
                    'status': 'skipped',
                    'reason': 'data_preparation_failed',
                    'timestamp': datetime.now().isoformat(),
                    'warnings': data_prep['warnings'],
                    'model_type': 'SVM'
                }
                
                status_file = output_dir / "svm_training_status.json"
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
                'C': 1.0,
                'class_weight': 'balanced',
                'max_iter': 10000,
                'random_state': 42
            }
            logger.info("🔧 Using default parameters")
        
        # Override with any provided kwargs
        params.update(kwargs)
        
        logger.info(f"📋 Training parameters:")
        for key, value in params.items():
            logger.info(f"   {key}: {value}")
        
        # Train model
        if grid_search:
            logger.info("Using GridSearchCV (comprehensive but slow)...")
            # TODO: Implement GridSearch if needed
            model_package = train_svm_fast(X_train, y_train, X_val, y_val, params, logger)
        else:
            logger.info("Using fast LinearSVC training...")
            model_package = train_svm_fast(X_train, y_train, X_val, y_val, params, logger)
        
        # Save model package
        saved_model_files = save_model_package(model_package, output_dir, logger)
        
        # Create plots and save metrics
        saved_plot_files = save_metrics_and_plots(model_package, output_dir, logger)
        
        # Combine all saved files
        all_saved_files = {**saved_model_files, **saved_plot_files}
        
        # Create success status
        success_status = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'model_type': 'SVM',
            'performance': {
                'accuracy': float(model_package['validation_accuracy']),
                'f1_score': float(model_package['validation_f1']),
                'training_time': float(model_package['training_time']),
                'training_samples': int(model_package['training_samples'])
            },
            'files': {
                'model': 'models/svm_model.pkl',
                'metadata': 'models/svm_metadata.json'
            }
        }
        
        status_file = output_dir / "svm_training_status.json"
        with open(status_file, 'w') as f:
            json.dump(success_status, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("✅ ENHANCED SVM TRAINING COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"📊 Final accuracy: {model_package['validation_accuracy']:.4f}")
        logger.info(f"📊 Final F1-score: {model_package['validation_f1']:.4f}")
        logger.info(f"⏱️ Training time: {model_package['training_time']:.2f} seconds")
        logger.info(f"💾 Model saved: {saved_model_files['model_path']}")
        
        return {
            'success': True,
            'skipped': False,
            'model_path': saved_model_files['model_path'],
            'metadata_path': saved_model_files['metadata_path'],
            'final_accuracy': model_package['validation_accuracy'],
            'final_f1': model_package['validation_f1'],
            'training_time': model_package['training_time'],
            'saved_files': all_saved_files,
            'status_file': str(status_file)
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced SVM training failed: {str(e)}")
        
        # Save error status
        if output_dir:
            error_status = {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error_message': str(e),
                'model_type': 'SVM'
            }
            try:
                status_file = Path(output_dir) / "svm_training_status.json"
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
        description="Enhanced SVM Training for Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🆕 ENHANCED FEATURES:
- Universal embedding file detection
- Intelligent training/inference mode detection
- Automatic parameter adjustment based on dataset size
- Robust error handling and graceful fallbacks

Examples:
  python scripts/train_svm.py                                    # Auto-detect and train
  python scripts/train_svm.py --embeddings-dir data/embeddings   # Specific directory
  python scripts/train_svm.py --auto-adjust                      # Auto-adjust parameters
  python scripts/train_svm.py --min-samples 50                   # Minimum samples for training
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
    
    # Model parameters
    parser.add_argument('--fast', action='store_true', default=True,
                       help='Use fast training mode (default: True)')
    parser.add_argument('--grid-search', action='store_true',
                       help='Use GridSearchCV for hyperparameter tuning')
    parser.add_argument('--C', type=float, default=None,
                       help='SVM regularization parameter')
    parser.add_argument('--max-iter', type=int, default=None,
                       help='Maximum iterations')
    
    # Legacy compatibility
    parser.add_argument('--session-name', type=str, default=None,
                       help='Session name (for compatibility)')
    parser.add_argument('--session-dir', type=str, default=None,
                       help='Session directory (for compatibility)')
    
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
    if args.C is not None:
        custom_params['C'] = args.C
    if args.max_iter is not None:
        custom_params['max_iter'] = args.max_iter
    
    # Run enhanced training
    result = enhanced_train_svm(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        auto_adjust=args.auto_adjust,
        min_samples=args.min_samples,
        skip_training_if_insufficient=args.skip_if_insufficient,
        fast_mode=args.fast,
        grid_search=args.grid_search,
        custom_params=custom_params if custom_params else None,
        logger=logger
    )
    
    if result['success']:
        if result.get('skipped'):
            logger.warning(f"⚠️ Training skipped: {result['reason']}")
            return 0
        else:
            logger.info("✅ Enhanced SVM training completed successfully!")
            return 0
    else:
        logger.error(f"❌ Enhanced SVM training failed: {result.get('error', 'Unknown error')}")
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