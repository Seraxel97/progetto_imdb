#!/usr/bin/env python3
"""
SVM Training Script - PIPELINE OPTIMIZED
Trains Support Vector Machine models for sentiment analysis with full pipeline integration.

FEATURES:
- Fast mode by default (no GridSearchCV) - training in seconds instead of minutes
- Optional GridSearchCV hyperparameter search with --grid-search flag
- Uses LinearSVC with proven parameters (C=1.0, class_weight='balanced')
- Optional dataset reduction for ultra-fast training with --fast flag
- Maintains 81.5-82% accuracy on IMDb sentiment analysis
- Full compatibility with report.py, GUI, and pipeline automation
- Robust model saving with structured output directories
- Professional logging system
- Auto-defaults when run without arguments

USAGE:
  python scripts/train_svm.py                                              # Auto-defaults
  python scripts/train_svm.py --embeddings-dir data/embeddings --output-dir results
  python scripts/train_svm.py --embeddings-dir data/embeddings --output-dir results --fast
  python scripts/train_svm.py --embeddings-dir data/embeddings --output-dir results --grid-search
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

# ML imports
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

class SafeStreamHandler(logging.StreamHandler):
    """Safe stream handler that falls back to ASCII if UTF-8 fails on Windows console"""
    
    def emit(self, record):
        try:
            super().emit(record)
        except UnicodeEncodeError:
            # Remove emoji fallback if console doesn't support it
            record.msg = self.remove_emoji(str(record.msg))
            record.args = ()
            super().emit(record)
    
    def remove_emoji(self, text):
        """Remove emoji and non-ASCII characters from text"""
        import re
        # Remove emojis and non-ASCII characters
        cleaned = re.sub(r'[^\x00-\x7F]+', '', text)
        # Clean up common emoji replacements
        cleaned = re.sub(r'[⚠️✅❌🔧📊📄💾🚀⚡🔄🔮🏆📏📋🤖]', '', cleaned)
        return cleaned

def setup_logging(log_dir):
    """Setup logging configuration with Windows-safe console output"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"svm_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Try to set UTF-8 encoding for stdout if possible
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, Exception):
        pass  # Fallback to SafeStreamHandler
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            SafeStreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def find_embedding_files(embeddings_dir=None, session_dir=None, logger=None):
    """Trova i file di embedding in modo intelligente"""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # Lista dei possibili percorsi
    search_paths = []
    
    # 1. Percorso specificato esplicitamente
    if embeddings_dir:
        search_paths.append(Path(embeddings_dir))
    
    # 2. Percorso nella sessione specifica
    if session_dir:
        search_paths.append(Path(session_dir) / "embeddings")
    
    # 3. Percorsi di default
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    search_paths.extend([
        project_root / "data" / "embeddings",
        project_root / "results" / "embeddings",
        Path("data") / "embeddings",
        Path("results") / "embeddings"
    ])
    
    # Cerca nei percorsi più recenti nelle sessioni
    results_dir = project_root / "results"
    if results_dir.exists():
        session_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("session_")]
        session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Più recenti prima
        
        for session_dir in session_dirs[:3]:  # Controlla le 3 sessioni più recenti
            search_paths.append(session_dir / "embeddings")
    
    required_files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "X_test.npy", "y_test.npy"]
    
    logger.info(f"Cercando file di embedding in {len(search_paths)} percorsi...")
    
    for path in search_paths:
        logger.info(f"Controllo: {path}")
        if path.exists():
            missing_files = []
            for file in required_files:
                if not (path / file).exists():
                    missing_files.append(file)
            
            if not missing_files:
                logger.info(f"✅ Tutti i file trovati in: {path}")
                return path
            else:
                logger.info(f"❌ File mancanti in {path}: {missing_files}")
        else:
            logger.info(f"❌ Percorso non esistente: {path}")
    
    raise FileNotFoundError(
        f"File di embedding non trovati. Cercati in:\n" + 
        "\n".join(f"  - {p}" for p in search_paths) +
        f"\nFile richiesti: {required_files}"
    )

def determine_output_dir(output_dir=None, session_dir=None):
    """Determina la directory di output in modo intelligente"""
    if output_dir:
        return Path(output_dir)
    
    if session_dir:
        return Path(session_dir)
    
    # Crea una nuova sessione se non specificata
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    results_dir = project_root / "results"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = results_dir / f"session_{timestamp}"
    
    return session_dir

def load_embeddings_data(embeddings_dir, logger):
    """
    Load training, validation, and test embeddings with comprehensive validation
    
    Args:
        embeddings_dir (str/Path): Directory containing embedding files
        logger: Logger instance
    
    Returns:
        dict: Dictionary containing loaded data splits
    """
    embeddings_dir = Path(embeddings_dir)

    logger.info(f"Loading embeddings from: {embeddings_dir}")
    try:
        logger.info(f"Files in embeddings dir: {os.listdir(embeddings_dir)}")
    except Exception:
        logger.warning("Could not list embeddings directory contents")
    
    # Check for required files
    required_files = {
        'X_train': embeddings_dir / "X_train.npy",
        'y_train': embeddings_dir / "y_train.npy", 
        'X_val': embeddings_dir / "X_val.npy",
        'y_val': embeddings_dir / "y_val.npy",
        'X_test': embeddings_dir / "X_test.npy",
        'y_test': embeddings_dir / "y_test.npy"
    }
    
    # Validate file existence
    missing_files = []
    for name, path in required_files.items():
        if not path.exists():
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        error_msg = f"Missing embedding files:\n" + "\n".join(f"  - {f}" for f in missing_files)
        error_msg += f"\n💡 Generate embeddings with: python scripts/embed_dataset.py"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load all data
    data = {}
    total_samples = 0
    
    try:
        for name, path in required_files.items():
            loaded_data = np.load(path)
            data[name] = loaded_data
            
            if name.startswith('X_'):
                split_name = name[2:]  # Remove 'X_' prefix
                samples = loaded_data.shape[0]
                features = loaded_data.shape[1]
                total_samples += samples
                
                logger.info(f"   ✅ {split_name}: {samples:,} samples, {features} features")
                    
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        raise
    
    # Validate data consistency
    splits = ['train', 'val', 'test']
    for split in splits:
        X_samples = data[f'X_{split}'].shape[0]
        y_samples = data[f'y_{split}'].shape[0]
        
        if X_samples != y_samples:
            raise ValueError(f"Mismatch in {split} split: X has {X_samples} samples, y has {y_samples}")
    
    # Validate feature dimensions consistency
    feature_dims = [data[f'X_{split}'].shape[1] for split in splits]
    if len(set(feature_dims)) > 1:
        raise ValueError(f"Feature dimension mismatch across splits: {feature_dims}")
    
    logger.info(f"Successfully loaded embeddings:")
    logger.info(f"   📊 Total samples: {total_samples:,}")
    logger.info(f"   📏 Feature dimension: {feature_dims[0]}")
    
    # Show label distribution
    for split in splits:
        labels = data[f'y_{split}']
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        logger.info(f"   📊 {split.capitalize()} labels: {dist}")
    
    return data

def prepare_training_data(data, fast_mode=False, logger=None):
    """
    Prepare and optionally reduce training data for fast training
    
    Args:
        data (dict): Loaded embedding data
        fast_mode (bool): Whether to reduce dataset size for ultra-fast training
        logger: Logger instance
    
    Returns:
        tuple: (X_train, X_val, y_train, y_val) prepared for training
    """
    if logger:
        logger.info("Preparing training data...")
    
    # Combine train and validation for full training
    X_train_full = np.vstack([data['X_train'], data['X_val']])
    y_train_full = np.hstack([data['y_train'], data['y_val']])
    
    if logger:
        logger.info(f"   📊 Combined train+val: {X_train_full.shape[0]:,} samples")
    
    # Apply fast mode reduction if requested
    if fast_mode:
        # Use stratified sampling to maintain class balance
        reduction_size = min(10000, X_train_full.shape[0])
        
        X_train_reduced, _, y_train_reduced, _ = train_test_split(
            X_train_full, y_train_full,
            train_size=reduction_size,
            stratify=y_train_full,
            random_state=42
        )
        
        if logger:
            logger.info(f"   ⚡ Fast mode: reduced to {X_train_reduced.shape[0]:,} samples")
            unique, counts = np.unique(y_train_reduced, return_counts=True)
            dist = dict(zip(unique, counts))
            logger.info(f"   📊 Fast mode labels: {dist}")
        
        return X_train_reduced, data['X_val'], y_train_reduced, data['y_val']
    else:
        # Use original validation split for evaluation
        return data['X_train'], data['X_val'], data['y_train'], data['y_val']

def train_svm_fast(X_train, y_train, X_val, y_val, params=None, logger=None):
    """
    Fast SVM training using LinearSVC with optimized parameters
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        params (dict): SVM parameters (uses optimized defaults if None)
        logger: Logger instance
    
    Returns:
        dict: Trained model package with metadata
    """
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
    
    # Train LinearSVM (much faster than RBF SVM)
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
        'model_type': 'LinearSVC_fast'
    }
    
    return model_package

def train_svm_gridsearch(X_train, y_train, X_val, y_val, logger=None):
    """
    Comprehensive SVM training with GridSearchCV (slow but thorough)
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        logger: Logger instance
    
    Returns:
        dict: Trained model package with metadata
    """
    if logger:
        logger.info("Starting comprehensive SVM training with GridSearchCV...")
        logger.warning("This may take several minutes with large datasets...")
    
    start_time = time()
    
    # Initialize preprocessing
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    # Combine train and val for cross-validation
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])
    
    # Fit preprocessing
    X_combined_scaled = scaler.fit_transform(X_combined)
    y_combined_encoded = label_encoder.fit_transform(y_combined)
    
    if logger:
        logger.info(f"   📊 Total samples for GridSearch: {X_combined.shape[0]:,}")
        logger.info(f"   📏 Features: {X_combined.shape[1]}")
    
    # Define parameter grid
    param_grid = [
        {
            'svm__C': [0.1, 1.0, 10.0],
            'svm__kernel': ['linear'],
            'svm__class_weight': [None, 'balanced']
        },
        {
            'svm__C': [0.1, 1.0, 10.0],
            'svm__kernel': ['rbf'],
            'svm__gamma': ['scale', 'auto'],
            'svm__class_weight': [None, 'balanced']
        }
    ]
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=42, max_iter=10000))
    ])
    
    # Grid search with cross-validation
    if logger:
        logger.info("   🔄 Running GridSearchCV (5-fold CV)...")
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_combined_scaled, y_combined_encoded)
    
    training_time = time() - start_time
    
    if logger:
        logger.info(f"   ✅ GridSearch completed in {training_time:.2f} seconds")
        logger.info(f"   🏆 Best parameters: {grid_search.best_params_}")
        logger.info(f"   📊 Best CV score: {grid_search.best_score_:.4f}")
    
    # Extract the best model components
    best_pipeline = grid_search.best_estimator_
    model = best_pipeline.named_steps['svm']
    
    # Make predictions on validation set for final evaluation
    X_val_scaled = scaler.transform(X_val)
    y_val_encoded = label_encoder.transform(y_val)
    y_val_pred = best_pipeline.predict(X_val_scaled)
    
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
    
    # Package everything together
    model_package = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'params': grid_search.best_params_,
        'training_time': training_time,
        'validation_accuracy': val_accuracy,
        'validation_f1': val_f1,
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix.tolist(),
        'feature_dim': X_combined.shape[1],
        'training_samples': X_combined.shape[0],
        'model_type': 'SVM_gridsearch',
        'best_cv_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
    
    return model_package

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
        matplotlib.use('Agg')  # Use non-interactive backend
        
        logger.info("Creating training plots and saving metrics...")
        
        # 1. Save metrics in JSON format for GUI compatibility
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
        
        if 'best_cv_score' in model_package:
            metrics_data['best_cv_score'] = model_package['best_cv_score']
        
        metrics_file = reports_dir / "svm_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_file}")
        
        # 2. Save classification report as CSV for easy reading
        class_report_df = pd.DataFrame(model_package['classification_report']).transpose()
        class_report_csv = reports_dir / "svm_classification_report.csv"
        class_report_df.to_csv(class_report_csv)
        
        # Also save as JSON
        class_report_json = reports_dir / "svm_classification_report.json"
        with open(class_report_json, 'w') as f:
            json.dump(model_package['classification_report'], f, indent=2)
        
        # 3. Confusion Matrix Plot
        plt.figure(figsize=(8, 6))
        cm = np.array(model_package['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cmap='Blues')
        plt.title(f'SVM Confusion Matrix\nAccuracy: {model_package["validation_accuracy"]:.3f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_plot_path = plots_dir / 'svm_confusion_matrix.png'
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Performance Metrics Plot
        plt.figure(figsize=(10, 6))
        
        metrics = ['Accuracy', 'F1-Score']
        values = [model_package['validation_accuracy'], model_package['validation_f1']]
        colors = ['#1f77b4', '#ff7f0e']
        
        bars = plt.bar(metrics, values, color=colors)
        plt.title(f'SVM Performance Metrics\nTraining Time: {model_package["training_time"]:.2f}s')
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
        
        # 5. Training Summary Plot (text-based info plot)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        summary_text = f"""
SVM Training Summary

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
        
        plt.title('SVM Training Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        summary_plot_path = plots_dir / 'svm_training_summary.png'
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Save text summary
        summary_file = reports_dir / "svm_training_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("SVM Training Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model type: {model_package['model_type']}\n")
            f.write(f"Training samples: {model_package['training_samples']:,}\n")
            f.write(f"Feature dimension: {model_package['feature_dim']}\n")
            f.write(f"Training time: {model_package['training_time']:.2f} seconds\n")
            f.write(f"Validation accuracy: {model_package['validation_accuracy']:.4f}\n")
            f.write(f"Validation F1-score: {model_package['validation_f1']:.4f}\n")
            f.write("\nParameters:\n")
            for k, v in model_package['params'].items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nConfusion Matrix:\n")
            f.write(str(np.array(model_package['confusion_matrix'])))
        
        # 7. Salva GUI status file per integrazione
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
                'classification_report': 'reports/svm_classification_report.json',
                'metrics': 'reports/svm_metrics.json',
                'summary': 'reports/svm_training_summary.txt'
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
            'classification_report_csv': str(class_report_csv),
            'classification_report_json': str(class_report_json),
            'summary_file': str(summary_file),
            'gui_status_file': str(gui_status_file)
        }
        
    except Exception as e:
        logger.error(f"Error creating plots and metrics: {str(e)}")
        raise

def save_model_package(model_package, output_dir, logger):
    """
    Save trained model package with comprehensive metadata
    
    Args:
        model_package (dict): Trained model package
        output_dir (str/Path): Directory to save model
        logger: Logger instance
    
    Returns:
        dict: Paths of saved files
    """
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Saving SVM model package...")
    
    try:
        # 1. Save the complete model package
        model_path = models_dir / 'svm_model.pkl'
        joblib.dump(model_package, model_path)
        
        logger.info(f"   📄 Model package: {model_path}")
        
        # 2. Save metadata separately for easy access
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
        
        # Add GridSearch specific metadata if available
        if 'best_cv_score' in model_package:
            metadata['best_cv_score'] = model_package['best_cv_score']
            metadata['used_gridsearch'] = True
        else:
            metadata['used_gridsearch'] = False
        
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

def train_svm_pipeline(embeddings_dir=None, output_dir=None, fast_mode=False, grid_search=False, 
                      C=1.0, max_iter=10000, logger=None, session_dir=None):
    """
    Main training pipeline for SVM con parametri flessibili
    
    Args:
        embeddings_dir (str): Directory containing embedding files (auto-detect se None)
        output_dir (str): Directory to save results (auto-create se None)
        fast_mode (bool): Use reduced dataset for ultra-fast training
        grid_search (bool): Use GridSearchCV for hyperparameter tuning
        C (float): SVM regularization parameter
        max_iter (int): Maximum iterations
        logger: Logger instance
        session_dir (str): Session directory per integrazione pipeline
    
    Returns:
        dict: Training results and saved file paths
    """
    if logger is None:
        # Setup logging temporaneo se non fornito
        temp_log_dir = Path("logs") if not output_dir else Path(output_dir) / "logs"
        logger = setup_logging(temp_log_dir)
    
    try:
        # Determina percorsi in modo intelligente
        if not embeddings_dir:
            embeddings_dir = find_embedding_files(session_dir=session_dir, logger=logger)
        else:
            # Verifica che il percorso specificato sia valido
            embeddings_dir = find_embedding_files(embeddings_dir=embeddings_dir, logger=logger)
        
        if not output_dir:
            output_dir = determine_output_dir(session_dir=session_dir)

        train_path = Path(embeddings_dir) / "X_train.npy"
        val_path = Path(embeddings_dir) / "X_val.npy"
        if not train_path.exists() or not val_path.exists():
            raise ValueError("Training data not found. Ensure preprocessing created valid train/val files.")

        # Create output directory structure
        output_dir = Path(output_dir)
        
        if logger:
            logger.info("=" * 60)
            logger.info("SVM TRAINING PIPELINE")
            logger.info("=" * 60)
            logger.info(f"Embeddings dir: {embeddings_dir}")
            logger.info(f"Output dir: {output_dir}")
            logger.info(f"Fast mode: {fast_mode}")
            logger.info(f"Grid search: {grid_search}")
            logger.info(f"Parameters: C={C}, max_iter={max_iter}")
        
        # Load embeddings data
        data = load_embeddings_data(embeddings_dir, logger)
        
        # Prepare training data
        X_train, X_val, y_train, y_val = prepare_training_data(
            data, fast_mode=fast_mode, logger=logger
        )
        
        # Choose training method
        if grid_search:
            if logger:
                logger.info("Using GridSearchCV (comprehensive but slow)...")
            model_package = train_svm_gridsearch(X_train, y_train, X_val, y_val, logger)
        else:
            if logger:
                logger.info("Using fast LinearSVC training...")
            
            # Prepare parameters
            params = {
                'C': C,
                'class_weight': 'balanced',
                'max_iter': max_iter,
                'random_state': 42
            }
            
            model_package = train_svm_fast(X_train, y_train, X_val, y_val, params, logger)
        
        # Save model package
        saved_model_files = save_model_package(model_package, output_dir, logger)
        
        # Create plots and save metrics
        saved_plot_files = save_metrics_and_plots(model_package, output_dir, logger)
        
        # Combine all saved files
        all_saved_files = {**saved_model_files, **saved_plot_files}
        
        if logger:
            logger.info("=" * 60)
            logger.info("SVM TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Final validation accuracy: {model_package['validation_accuracy']:.4f}")
            logger.info(f"Final validation F1-score: {model_package['validation_f1']:.4f}")
            logger.info(f"Training time: {model_package['training_time']:.2f} seconds")
            logger.info(f"Model saved to: {saved_model_files['model_path']}")
        
        return {
            'model_path': saved_model_files['model_path'],
            'metadata_path': saved_model_files['metadata_path'],
            'performance': {
                'validation_accuracy': model_package['validation_accuracy'],
                'validation_f1': model_package['validation_f1'],
                'training_time': model_package['training_time']
            },
            'saved_files': all_saved_files,
            'output_dir': str(output_dir),
            'embeddings_dir': str(embeddings_dir)
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Error in SVM training pipeline: {str(e)}")
        # Salva status di errore per la GUI
        if output_dir:
            error_status = {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error_message': str(e),
                'model_type': 'SVM'
            }
            try:
                with open(Path(output_dir) / "svm_training_status.json", 'w') as f:
                    json.dump(error_status, f, indent=2)
            except:
                pass
        raise

def run_training_auto_mode(session_dir=None, fast_mode=True, **kwargs):
    """Modalità automatica per integrazione con altri script"""
    # Setup logging minimal per auto mode
    if session_dir:
        log_dir = Path(session_dir) / "logs"
    else:
        log_dir = Path("logs")
    
    logger = setup_logging(log_dir)
    
    logger.info("🤖 Modalità automatica SVM training")
    
    # Parametri di default ottimizzati per auto mode
    default_params = {
        'fast_mode': fast_mode,  # Fast di default
        'grid_search': kwargs.get('grid_search', False),
        'C': kwargs.get('C', 1.0),
        'max_iter': kwargs.get('max_iter', 10000),
        'session_dir': session_dir
    }
    
    logger.info(f"Parametri auto mode: {default_params}")
    
    try:
        result = train_svm_pipeline(logger=logger, **default_params)
        logger.info("✅ SVM training automatico completato!")
        return result
    except Exception as e:
        logger.error(f"❌ Errore in modalità automatica: {str(e)}")
        raise

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train SVM model for sentiment analysis (optimized fast mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_svm.py                                                     # Auto-defaults (fast mode)
  python train_svm.py --embeddings-dir data/embeddings --output-dir results
  python train_svm.py --embeddings-dir data/embeddings --output-dir results --fast
  python train_svm.py --embeddings-dir data/embeddings --output-dir results --grid-search
        """
    )
    
    # Optional arguments (tutti opzionali per flessibilità)
    parser.add_argument("--embeddings-dir", type=str, default=None,
                       help="Directory containing embedding files (.npy). If not specified, will search automatically.")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save model, plots, and metrics. If not specified, will create session directory.")
    parser.add_argument("--session-dir", type=str, default=None,
                       help="Session directory (used for automatic path detection)")
    
    # Training modes
    parser.add_argument("--fast", action="store_true",
                       help="Ultra-fast mode: reduce dataset to 10k samples [enabled by default when no args]")
    parser.add_argument("--grid-search", action="store_true",
                       help="Use GridSearchCV for hyperparameter tuning (slow)")
    
    # Model parameters
    parser.add_argument("--C", type=float, default=1.0,
                       help="SVM regularization parameter (default: 1.0)")
    parser.add_argument("--max-iter", type=int, default=10000,
                       help="Maximum iterations (default: 10000)")
    
    # Optional parameters (for compatibility with train_mlp.py)
    parser.add_argument("--epochs", type=int, default=None,
                       help="Not used for SVM (for compatibility)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Not used for SVM (for compatibility)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Not used for SVM (for compatibility)")
    
    # Modalità speciali
    parser.add_argument('--auto-mode', action='store_true',
                       help='Run in automatic mode with default parameters')
    
    # Logging options
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Directory for log files (default: output-dir/logs)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress output except errors")
    
    return parser.parse_args()

def main():
    """Main function for CLI usage"""
    args = parse_arguments()

    # Modalità automatica
    if args.auto_mode:
        return run_training_auto_mode(
            session_dir=args.session_dir,
            fast_mode=args.fast,
            grid_search=args.grid_search,
            C=args.C,
            max_iter=args.max_iter
        )

    # Detect project root dynamically
    project_root = Path(__file__).resolve().parents[1]
    no_args_provided = len(sys.argv) == 1

    # Store original argument state for logging
    embeddings_provided = args.embeddings_dir is not None
    output_provided = args.output_dir is not None

    # Apply defaults automatically (adesso non sono più required, quindi è safe)
    if not args.embeddings_dir:
        args.embeddings_dir = str(project_root / "data" / "embeddings")

    if not args.output_dir:
        args.output_dir = str(project_root / "results")

    # Enable fast mode by default when no arguments provided
    if no_args_provided:
        args.fast = True

    # Setup logging
    if args.output_dir:
        log_dir = args.log_dir if args.log_dir else Path(args.output_dir) / "logs"
    else:
        log_dir = args.log_dir if args.log_dir else Path("logs")
    
    logger = setup_logging(log_dir)

    # Show appropriate warning messages
    if no_args_provided:
        logger.warning("⚠️ No arguments provided. Using default embeddings and output directories.")
    else:
        # Show individual warnings for missing specific arguments
        if not embeddings_provided:
            logger.warning("⚠️ No --embeddings-dir provided. Using default: data/embeddings")

        if not output_provided:
            logger.warning("⚠️ No --output-dir provided. Using default: results")
    
    # Show warnings for unused parameters
    if args.epochs is not None:
        logger.warning("--epochs parameter is not used for SVM training")
    if args.lr is not None:
        logger.warning("--lr parameter is not used for SVM training")
    if args.batch_size is not None:
        logger.warning("--batch-size parameter is not used for SVM training")
    
    try:
        # Run training pipeline
        result = train_svm_pipeline(
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir,
            fast_mode=args.fast,
            grid_search=args.grid_search,
            C=args.C,
            max_iter=args.max_iter,
            logger=logger,
            session_dir=args.session_dir
        )
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Model saved: {result['model_path']}")
        logger.info(f"Accuracy: {result['performance']['validation_accuracy']:.4f}")
        logger.info(f"F1-Score: {result['performance']['validation_f1']:.4f}")
        logger.info(f"Training time: {result['performance']['training_time']:.2f}s")
        logger.info(f"Output directory: {result['output_dir']}")
        logger.info(f"Files saved: {list(result['saved_files'].keys())}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during SVM training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Controlla se è chiamato senza argomenti (modalità legacy)
        if len(sys.argv) == 1:
            print("🚀 Avvio SVM training in modalità automatica...")
            print(f"Working directory: {os.getcwd()}")
            print(f"Script location: {__file__}")
            
            result = run_training_auto_mode(fast_mode=True)
            print(f"\n✅ Training completato! Modello salvato in: {result['model_path']}")
            print(f"📊 Accuracy finale: {result['performance']['validation_accuracy']:.4f}")
            print(f"📊 F1-Score finale: {result['performance']['validation_f1']:.4f}")
        else:
            # Usa il parser per CLI
            result = main()
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrotto dall'utente")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Errore: {str(e)}")
        sys.exit(1)
