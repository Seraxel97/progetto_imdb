#!/usr/bin/env python3
"""
Report Generation Script - FIXED UNICODE + ROBUST PATH HANDLING
Generates comprehensive evaluation reports for trained sentiment analysis models.

🔧 CRITICAL FIXES APPLIED:
- ✅ Fixed UnicodeDecodeError in subprocess execution with proper encoding
- ✅ Added robust file encoding handling (utf-8 with error handling)
- ✅ Enhanced path discovery for session-based model structure
- ✅ Improved error handling and graceful fallbacks
- ✅ Fixed report generation pipeline integration

FEATURES:
- Evaluates both MLP and SVM models automatically
- Multiple evaluation methods using direct embeddings
- Comprehensive reporting with plots, metrics, and insights
- Robust error handling with detailed diagnostics
- Full pipeline integration with structured output
- Professional logging system with UTF-8 support

USAGE:
  python report.py --models-dir results/auto_analysis_*/models --test-data data/processed/test.csv --results-dir results/auto_analysis_*/
  python report.py --auto-default  # Auto-detect latest session
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import torch
import argparse
import logging
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Dynamic project root detection for flexible execution
try:
    CURRENT_FILE = Path(__file__).resolve()
    if CURRENT_FILE.parent.name == 'scripts':
        PROJECT_ROOT = CURRENT_FILE.parent.parent
    else:
        PROJECT_ROOT = CURRENT_FILE.parent
except Exception:
    PROJECT_ROOT = Path.cwd()

def setup_logging(log_dir):
    """Setup logging configuration with robust UTF-8 support"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"report_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 🔧 FIXED: Robust stream handler with UTF-8 support
    try:
        # Try to configure UTF-8 encoding for stdout
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        elif hasattr(sys.stdout, 'encoding'):
            # Create a wrapper if reconfigure is not available
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    except (AttributeError, Exception):
        pass  # Fallback gracefully
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def find_latest_session_directory(project_root=None):
    """🔧 NEW: Find the latest auto_analysis session directory"""
    if project_root is None:
        project_root = PROJECT_ROOT
    
    results_dir = Path(project_root) / "results"
    
    if not results_dir.exists():
        return None
    
    # Find all auto_analysis directories
    session_dirs = []
    for item in results_dir.iterdir():
        if item.is_dir() and item.name.startswith("auto_analysis_"):
            session_dirs.append(item)
    
    if not session_dirs:
        return None
    
    # Sort by modification time (most recent first)
    session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return session_dirs[0]

def find_session_models_and_data(session_dir=None):
    """🔧 NEW: Find models and test data in session directory"""
    if session_dir is None:
        session_dir = find_latest_session_directory()
    
    if session_dir is None:
        return None, None, None
    
    session_dir = Path(session_dir)
    
    # Look for models directory
    models_candidates = [
        session_dir / "models",
        session_dir / "models" / "models",  # Sometimes nested
        session_dir
    ]
    
    models_dir = None
    for candidate in models_candidates:
        if candidate.exists() and any(candidate.glob("*.pth")) or any(candidate.glob("*.pkl")):
            models_dir = candidate
            break
    
    # Look for test data
    test_data_candidates = [
        session_dir / "processed" / "test.csv",
        session_dir / "data" / "processed" / "test.csv",
        session_dir / "embeddings" / "X_test.npy",
        session_dir / "data" / "embeddings" / "X_test.npy"
    ]
    
    test_data = None
    for candidate in test_data_candidates:
        if candidate.exists():
            test_data = candidate
            break
    
    # Results directory (use session_dir as results)
    results_dir = session_dir
    
    return models_dir, test_data, results_dir

class ModelEvaluator:
    """
    🔧 FIXED: Comprehensive model evaluation class with robust error handling
    """
    
    def __init__(self, models_dir, results_dir, logger=None):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Create output directory structure
        self.reports_dir = self.results_dir / "reports"
        self.plots_dir = self.results_dir / "plots"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluation_results = {}
        self.evaluation_errors = []
        self.skipped_models = []
        
        self.logger.info(f"ModelEvaluator initialized:")
        self.logger.info(f"   Models dir: {self.models_dir}")
        self.logger.info(f"   Results dir: {self.results_dir}")
        
    def find_available_models(self):
        """Find available trained models with enhanced search"""
        available_models = {}
        
        self.logger.info(f"Searching for models in: {self.models_dir}")
        
        # Enhanced search patterns
        search_patterns = {
            'mlp': ['mlp_model.pth', 'mlp_model_complete.pth'],
            'svm': ['svm_model.pkl']
        }
        
        for model_type, patterns in search_patterns.items():
            found = False
            for pattern in patterns:
                model_files = list(self.models_dir.glob(f"**/{pattern}"))
                if model_files:
                    available_models[model_type] = model_files[0]
                    self.logger.info(f"✅ Found {model_type.upper()} model: {model_files[0]}")
                    found = True
                    break
            
            if not found:
                self.logger.warning(f"❌ No {model_type.upper()} model found")
        
        return available_models
    
    def load_test_data(self, test_data_path):
        """🔧 FIXED: Load test data with robust encoding"""
        test_path = Path(test_data_path)
        
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        if test_path.suffix == '.csv':
            self.logger.info(f"Loading test data from CSV: {test_path}")
            
            # 🔧 FIXED: Robust CSV loading with encoding handling
            try:
                test_df = pd.read_csv(test_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    test_df = pd.read_csv(test_path, encoding='latin-1')
                    self.logger.warning("Used latin-1 encoding for CSV")
                except UnicodeDecodeError:
                    test_df = pd.read_csv(test_path, encoding='utf-8', errors='replace')
                    self.logger.warning("Used utf-8 with error replacement for CSV")
            
            # Validate required columns
            if 'text' not in test_df.columns or 'label' not in test_df.columns:
                raise ValueError("CSV test data must contain 'text' and 'label' columns")
            
            self.logger.info(f"Loaded test data: {len(test_df)} samples from CSV")
            return test_df, 'csv'
            
        elif test_path.suffix == '.npy':
            self.logger.info(f"Loading test embeddings from numpy: {test_path}")
            
            # Assume this is X_test.npy, look for corresponding y_test.npy
            if test_path.name == 'X_test.npy':
                y_test_path = test_path.parent / 'y_test.npy'
            else:
                # Try to infer y path
                y_test_path = test_path.parent / f"y_{test_path.stem}.npy"
            
            if not y_test_path.exists():
                raise FileNotFoundError(f"Corresponding labels file not found: {y_test_path}")
            
            X_test = np.load(test_path)
            y_test = np.load(y_test_path)
            
            self.logger.info(f"Loaded embeddings: X_test={X_test.shape}, y_test={y_test.shape}")
            
            return {'X_test': X_test, 'y_test': y_test}, 'embeddings'
        else:
            raise ValueError(f"Unsupported test data format: {test_path.suffix}")
    
    def evaluate_model_direct_embeddings(self, model_path, X_test, y_test, model_type):
        """Direct model evaluation using embeddings with robust error handling"""
        try:
            self.logger.info(f"Direct evaluation of {model_type.upper()} model...")
            
            if model_type == 'svm':
                return self._evaluate_svm_direct(model_path, X_test, y_test)
            elif model_type == 'mlp':
                return self._evaluate_mlp_direct(model_path, X_test, y_test)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Error in direct evaluation of {model_type}: {str(e)}")
            raise
    
    def _evaluate_svm_direct(self, model_path, X_test, y_test):
        """🔧 FIXED: Direct SVM evaluation with robust loading"""
        model_path = Path(model_path)

        if not model_path.exists():
            self.logger.error(f"Model not found: {model_path}")
            raise RuntimeError(f"Model not found: {model_path}")

        try:
            self.logger.info("Loading SVM model package...")
            
            # 🔧 FIXED: Robust model loading
            model_package = joblib.load(model_path)
            
            # Extract components with fallbacks
            if isinstance(model_package, dict):
                model = model_package.get('model')
                scaler = model_package.get('scaler')
                label_encoder = model_package.get('label_encoder')
            else:
                # Fallback: assume the loaded object is the model itself
                model = model_package
                scaler = None
                label_encoder = None
            
            if model is None:
                raise ValueError("No model found in the package")
            
            self.logger.info(f"SVM model loaded successfully")
            self.logger.info(f"   Model type: {type(model).__name__}")
            self.logger.info(f"   Has scaler: {scaler is not None}")
            self.logger.info(f"   Has label encoder: {label_encoder is not None}")
            
            # Apply scaling if available
            if scaler is not None:
                self.logger.info("Applying scaling to test features...")
                X_test_scaled = scaler.transform(X_test)
            else:
                self.logger.warning("No scaler found, using raw features")
                X_test_scaled = X_test
            
            # Encode labels if needed
            if label_encoder is not None:
                self.logger.info("Encoding test labels...")
                y_test_encoded = label_encoder.transform(y_test)
            else:
                self.logger.warning("No label encoder found, assuming labels are already encoded")
                y_test_encoded = y_test
            
            # Make predictions
            self.logger.info("Making predictions...")
            predictions = model.predict(X_test_scaled)
            
            # Get prediction probabilities if available
            try:
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_test_scaled)
                    confidences = np.max(probabilities, axis=1)
                else:
                    confidences = np.ones(len(predictions)) * 0.5
            except:
                confidences = np.ones(len(predictions)) * 0.5
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_encoded, predictions)
            f1 = f1_score(y_test_encoded, predictions, average='weighted')
            
            # Detailed classification report
            class_names = ['Negative', 'Positive']
            class_report = classification_report(
                y_test_encoded, predictions,
                target_names=class_names,
                output_dict=True
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test_encoded, predictions)
            
            results = {
                'model_type': 'svm',
                'model_path': str(model_path),
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'classification_report': class_report,
                'confusion_matrix': cm.tolist(),
                'predictions': predictions.tolist(),
                'true_labels': y_test_encoded.tolist(),
                'confidences': confidences.tolist(),
                'num_samples': len(X_test),
                'evaluation_method': 'direct_embeddings',
                'embedding_dimension': X_test.shape[1],
                'used_scaler': scaler is not None,
                'used_label_encoder': label_encoder is not None
            }
            
            self.logger.info(f"SVM evaluation completed:")
            self.logger.info(f"   Accuracy: {accuracy:.4f}")
            self.logger.info(f"   F1-Score: {f1:.4f}")
            self.logger.info(f"   Samples: {len(X_test)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in SVM direct evaluation: {str(e)}")
            raise
    
    def _evaluate_mlp_direct(self, model_path, X_test, y_test):
        """🔧 FIXED: Direct MLP evaluation with robust PyTorch loading"""
        model_path = Path(model_path)

        if not model_path.exists():
            self.logger.error(f"Model not found: {model_path}")
            raise RuntimeError(f"Model not found: {model_path}")

        try:
            self.logger.info("Loading MLP model...")
            
            # Define MLP architecture (should match training)
            class HateSpeechMLP(torch.nn.Module):
                def __init__(self, input_dim=384, hidden_dims=[512, 256, 128, 64], dropout=0.3):
                    super(HateSpeechMLP, self).__init__()
                    
                    layers = []
                    prev_dim = input_dim
                    
                    for i, hidden_dim in enumerate(hidden_dims):
                        layers.append(torch.nn.Linear(prev_dim, hidden_dim))
                        layers.append(torch.nn.BatchNorm1d(hidden_dim))
                        layers.append(torch.nn.ReLU())
                        dropout_rate = dropout if i < len(hidden_dims) - 2 else dropout * 0.7
                        layers.append(torch.nn.Dropout(dropout_rate))
                        prev_dim = hidden_dim
                    
                    layers.append(torch.nn.Linear(prev_dim, 1))
                    layers.append(torch.nn.Sigmoid())
                    
                    self.network = torch.nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.network(x)
            
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 🔧 FIXED: Robust model loading with multiple strategies
            try:
                # Strategy 1: Load state dict
                model = HateSpeechMLP(input_dim=384).to(device)
                checkpoint = torch.load(model_path, map_location=device)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)
                    
            except Exception as load_error:
                self.logger.warning(f"State dict loading failed: {load_error}")
                try:
                    # Strategy 2: Load complete model
                    model = torch.load(model_path, map_location=device)
                except Exception as complete_load_error:
                    self.logger.error(f"Complete model loading also failed: {complete_load_error}")
                    raise
            
            model.eval()
            
            self.logger.info(f"MLP model loaded successfully")
            self.logger.info(f"   Device: {device}")
            self.logger.info(f"   Input dimension: {X_test.shape[1]}")
            
            # Convert to tensors
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            
            self.logger.info(f"Making predictions on {len(X_test)} samples...")
            
            # Make predictions in batches to avoid memory issues
            batch_size = 64
            all_predictions = []
            all_probabilities = []
            
            with torch.no_grad():
                for i in range(0, len(X_test_tensor), batch_size):
                    batch = X_test_tensor[i:i+batch_size]
                    outputs = model(batch)
                    probabilities = outputs.squeeze().cpu().numpy()
                    
                    # Handle single sample case
                    if probabilities.ndim == 0:
                        probabilities = np.array([probabilities])
                    
                    predictions = (probabilities > 0.5).astype(int)
                    
                    all_predictions.extend(predictions)
                    all_probabilities.extend(probabilities)
                    
                    if (i + batch_size) % (batch_size * 10) == 0:
                        self.logger.info(f"   Processed {min(i + batch_size, len(X_test))}/{len(X_test)} samples...")
            
            predictions = np.array(all_predictions)
            probabilities = np.array(all_probabilities)
            
            # Calculate confidence (distance from 0.5 threshold)
            confidences = np.abs(probabilities - 0.5) + 0.5
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')
            
            # Detailed classification report
            class_names = ['Negative', 'Positive']
            class_report = classification_report(
                y_test, predictions,
                target_names=class_names,
                output_dict=True
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test, predictions)
            
            results = {
                'model_type': 'mlp',
                'model_path': str(model_path),
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'classification_report': class_report,
                'confusion_matrix': cm.tolist(),
                'predictions': predictions.tolist(),
                'true_labels': y_test.tolist(),
                'confidences': confidences.tolist(),
                'probabilities': probabilities.tolist(),
                'num_samples': len(X_test),
                'evaluation_method': 'direct_pytorch',
                'embedding_dimension': X_test.shape[1],
                'device': str(device),
                'batch_size': batch_size
            }
            
            self.logger.info(f"MLP evaluation completed:")
            self.logger.info(f"   Accuracy: {accuracy:.4f}")
            self.logger.info(f"   F1-Score: {f1:.4f}")
            self.logger.info(f"   Samples: {len(X_test)}")
            self.logger.info(f"   Average confidence: {np.mean(confidences):.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in MLP direct evaluation: {str(e)}")
            raise
    
    def create_evaluation_plots(self, results, model_type):
        """🔧 FIXED: Create evaluation plots with robust file handling"""
        try:
            # Set matplotlib backend
            import matplotlib
            matplotlib.use('Agg')
            
            self.logger.info(f"Creating plots for {model_type}...")
            
            # 1. Confusion Matrix
            cm = np.array(results['confusion_matrix'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', 
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       cmap='Blues')
            plt.title(f'{model_type.upper()} - Confusion Matrix\nAccuracy: {results["accuracy"]:.3f}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            plot_path = self.plots_dir / f'{model_type}_confusion_matrix.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Performance metrics bar chart
            metrics = ['Accuracy', 'F1-Score']
            values = [results['accuracy'], results['f1_score']]
            
            plt.figure(figsize=(8, 6))
            bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e'])
            plt.title(f'{model_type.upper()} - Performance Metrics')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            metrics_plot_path = self.plots_dir / f'{model_type}_metrics.png'
            plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Plots saved for {model_type}")
            
            return {
                'confusion_matrix': str(plot_path),
                'metrics': str(metrics_plot_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating plots for {model_type}: {str(e)}")
            return {}
    
    def save_detailed_metrics(self, results, model_type):
        """🔧 FIXED: Save detailed metrics with robust encoding"""
        try:
            # Save JSON metrics for GUI
            metrics_data = {
                'model_type': results['model_type'],
                'model_path': results['model_path'],
                'evaluation_method': results['evaluation_method'],
                'accuracy': results['accuracy'],
                'f1_score': results['f1_score'],
                'num_samples': results['num_samples'],
                'confusion_matrix': results['confusion_matrix'],
                'classification_report': results['classification_report'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Add model-specific metadata
            if model_type == 'svm':
                metrics_data.update({
                    'embedding_dimension': results.get('embedding_dimension'),
                    'used_scaler': results.get('used_scaler'),
                    'used_label_encoder': results.get('used_label_encoder')
                })
            elif model_type == 'mlp':
                metrics_data.update({
                    'device': results.get('device'),
                    'batch_size': results.get('batch_size'),
                    'average_confidence': float(np.mean(results.get('confidences', [0.5])))
                })
            
            # 🔧 FIXED: Save JSON with robust encoding
            json_path = self.reports_dir / f'{model_type}_detailed_metrics.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
            # Save classification report as CSV
            class_report_df = pd.DataFrame(results['classification_report']).transpose()
            csv_path = self.reports_dir / f'{model_type}_classification_report.csv'
            class_report_df.to_csv(csv_path, encoding='utf-8')
            
            self.logger.info(f"Detailed metrics saved for {model_type}")
            
            return {
                'json_metrics': str(json_path),
                'csv_report': str(csv_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error saving metrics for {model_type}: {str(e)}")
            return {}
    
    def generate_comprehensive_report(self):
        """🔧 FIXED: Generate comprehensive evaluation report with robust handling"""
        try:
            self.logger.info("Generating comprehensive report...")
            
            # Create summary statistics
            summary = {
                'report_timestamp': datetime.now().isoformat(),
                'models_evaluated': list(self.evaluation_results.keys()),
                'evaluation_errors': self.evaluation_errors,
                'skipped_models': self.skipped_models,
                'total_models_found': len(self.evaluation_results) + len(self.skipped_models),
                'results_directory': str(self.results_dir)
            }
            
            # Compare models if multiple available
            if len(self.evaluation_results) > 1:
                comparison = {}
                for model_type, results in self.evaluation_results.items():
                    comparison[model_type] = {
                        'accuracy': results['accuracy'],
                        'f1_score': results['f1_score'],
                        'num_samples': results['num_samples'],
                        'evaluation_method': results.get('evaluation_method', 'unknown')
                    }
                summary['model_comparison'] = comparison
                
                # Determine best model
                best_model = max(comparison.keys(), key=lambda k: comparison[k]['accuracy'])
                summary['best_model'] = {
                    'model_type': best_model,
                    'accuracy': comparison[best_model]['accuracy'],
                    'f1_score': comparison[best_model]['f1_score']
                }
            
            # Add insights
            summary['insights'] = self._generate_insights()
            
            # 🔧 FIXED: Save comprehensive results with robust encoding
            report_path = self.reports_dir / 'evaluation_report.json'
            full_results = {
                'summary': summary,
                'detailed_results': self.evaluation_results
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)
            
            # Create text report
            text_report_path = self.create_text_report(summary)
            
            # Create insights file
            insights_path = self.create_insights_file(summary)
            
            self.logger.info(f"Comprehensive report generated")
            
            return {
                'summary': summary,
                'report_path': str(report_path),
                'text_report_path': str(text_report_path),
                'insights_path': str(insights_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {str(e)}")
            raise
    
    def _generate_insights(self):
        """Generate insights from evaluation results"""
        insights = []
        
        if not self.evaluation_results:
            insights.append("No models were successfully evaluated")
            return insights
        
        # Performance insights
        best_accuracy = max(r['accuracy'] for r in self.evaluation_results.values())
        worst_accuracy = min(r['accuracy'] for r in self.evaluation_results.values())
        
        if best_accuracy >= 0.85:
            insights.append("Excellent model performance achieved (≥85% accuracy)")
        elif best_accuracy >= 0.80:
            insights.append("Good model performance achieved (≥80% accuracy)")
        elif best_accuracy >= 0.75:
            insights.append("Acceptable model performance achieved (≥75% accuracy)")
        else:
            insights.append("Model performance below expectations (<75% accuracy)")
        
        # Model comparison insights
        if len(self.evaluation_results) > 1:
            accuracy_diff = best_accuracy - worst_accuracy
            if accuracy_diff > 0.05:
                insights.append(f"Significant performance difference between models ({accuracy_diff:.3f})")
            else:
                insights.append("Models show similar performance")
        
        # Data insights
        sample_counts = [r['num_samples'] for r in self.evaluation_results.values()]
        avg_samples = sum(sample_counts) / len(sample_counts)
        insights.append(f"Evaluated on average {avg_samples:.0f} test samples per model")
        
        # Error insights
        if self.evaluation_errors:
            insights.append(f"Encountered {len(self.evaluation_errors)} evaluation errors")
        
        if self.skipped_models:
            insights.append(f"Skipped {len(self.skipped_models)} models due to errors")
        
        return insights
    
    def create_text_report(self, summary):
        """🔧 FIXED: Create human-readable text report with robust encoding"""
        report_lines = [
            "SENTIMENT ANALYSIS MODEL EVALUATION REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Results Directory: {self.results_dir}",
            "",
            "EVALUATION SUMMARY:",
            f"  Models Evaluated: {len(summary['models_evaluated'])}",
            f"  Models Skipped: {len(summary.get('skipped_models', []))}",
            f"  Evaluation Errors: {len(summary.get('evaluation_errors', []))}",
            "",
            "MODELS EVALUATED:"
        ]
        
        for model_type in summary['models_evaluated']:
            results = self.evaluation_results[model_type]
            report_lines.extend([
                f"",
                f"{model_type.upper()} MODEL:",
                f"  Model Path: {results['model_path']}",
                f"  Evaluation Method: {results.get('evaluation_method', 'unknown')}",
                f"  Samples Evaluated: {results['num_samples']:,}",
                f"  Accuracy: {results['accuracy']:.4f}",
                f"  F1-Score: {results['f1_score']:.4f}",
            ])
            
            # Add classification report details
            class_report = results['classification_report']
            report_lines.append(f"  Classification Report:")
            
            # Get class-specific metrics
            for class_name in ['Negative', 'Positive']:
                if class_name in class_report:
                    metrics = class_report[class_name]
                    report_lines.append(
                        f"    {class_name} - Precision: {metrics.get('precision', 0):.3f}, "
                        f"Recall: {metrics.get('recall', 0):.3f}, "
                        f"F1: {metrics.get('f1-score', 0):.3f}, "
                        f"Support: {metrics.get('support', 0)}"
                    )
        
        # Model comparison
        if 'model_comparison' in summary:
            report_lines.extend([
                "",
                "MODEL COMPARISON:",
                "-" * 20
            ])
            
            comparison = summary['model_comparison']
            for model_type, metrics in comparison.items():
                report_lines.append(
                    f"  {model_type.upper()}: Accuracy={metrics['accuracy']:.3f}, "
                    f"F1={metrics['f1_score']:.3f}"
                )
            
            if 'best_model' in summary:
                best = summary['best_model']
                report_lines.extend([
                    "",
                    f"BEST PERFORMING MODEL: {best['model_type'].upper()}",
                    f"  Accuracy: {best['accuracy']:.4f}",
                    f"  F1-Score: {best['f1_score']:.4f}"
                ])
        
        # Insights
        if 'insights' in summary:
            report_lines.extend([
                "",
                "KEY INSIGHTS:",
                "-" * 15
            ])
            for insight in summary['insights']:
                report_lines.append(f"  • {insight}")
        
        # Errors and troubleshooting
        if summary.get('skipped_models') or summary.get('evaluation_errors'):
            report_lines.extend([
                "",
                "ISSUES AND TROUBLESHOOTING:",
                "-" * 30
            ])
            
            if summary.get('skipped_models'):
                report_lines.append("SKIPPED MODELS:")
                for skipped in summary['skipped_models']:
                    report_lines.extend([
                        f"  {skipped['model_type'].upper()}:",
                        f"    Reason: {skipped['reason']}",
                    ])
            
            if summary.get('evaluation_errors'):
                report_lines.append("")
                report_lines.append("EVALUATION ERRORS:")
                for error in summary['evaluation_errors']:
                    report_lines.extend([
                        f"  {error['model_type'].upper()}:",
                        f"    Error: {error['error']}",
                    ])
        
        report_lines.extend([
            "",
            "=" * 50,
            "Report generation completed."
        ])
        
        # 🔧 FIXED: Save text report with robust encoding
        text_report_path = self.reports_dir / 'evaluation_report.txt'
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Text report saved: {text_report_path}")
        return text_report_path
    
    def create_insights_file(self, summary):
        """🔧 FIXED: Create standalone insights file with robust encoding"""
        insights_lines = [
            "SENTIMENT ANALYSIS EVALUATION INSIGHTS",
            "=" * 40,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "KEY FINDINGS:",
        ]
        
        if 'insights' in summary:
            for i, insight in enumerate(summary['insights'], 1):
                insights_lines.append(f"{i}. {insight}")
        
        if 'best_model' in summary:
            best = summary['best_model']
            insights_lines.extend([
                "",
                "RECOMMENDATION:",
                f"Use the {best['model_type'].upper()} model for production deployment.",
                f"It achieved {best['accuracy']:.1%} accuracy on the test set."
            ])
        
        # Add performance benchmarks
        insights_lines.extend([
            "",
            "PERFORMANCE BENCHMARKS:",
            "• >85% accuracy: Excellent performance",
            "• >80% accuracy: Good performance", 
            "• >75% accuracy: Acceptable performance",
            "• <75% accuracy: Needs improvement"
        ])
        
        # 🔧 FIXED: Save insights with robust encoding
        insights_path = self.reports_dir / 'insights.txt'
        with open(insights_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(insights_lines))
        
        self.logger.info(f"Insights saved: {insights_path}")
        return insights_path

def run_subprocess_with_encoding(cmd, cwd=None, timeout=300):
    """🔧 NEW: Run subprocess with proper UTF-8 encoding handling"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',  # 🔧 FIXED: Explicit UTF-8 encoding
            errors='replace',  # 🔧 FIXED: Replace invalid characters instead of failing
            timeout=timeout,
            cwd=cwd
        )
        return result
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, -1, "", f"Command timed out after {timeout}s")
    except Exception as e:
        return subprocess.CompletedProcess(cmd, -1, "", str(e))

def generate_evaluation_report(models_dir, test_data, results_dir, logger=None):
    """
    🔧 FIXED: Main pipeline function for generating evaluation reports
    
    Args:
        models_dir (str): Directory containing trained models
        test_data (str): Path to test data (CSV or numpy)
        results_dir (str): Directory to save results
        logger: Logger instance
    
    Returns:
        dict: Report generation results
    """
    try:
        if logger:
            logger.info("=" * 60)
            logger.info("EVALUATION REPORT GENERATION - FIXED VERSION")
            logger.info("=" * 60)
            logger.info(f"Models dir: {models_dir}")
            logger.info(f"Test data: {test_data}")
            logger.info(f"Results dir: {results_dir}")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(models_dir, results_dir, logger)
        
        # Load test data
        test_data_obj, data_type = evaluator.load_test_data(test_data)
        
        # Find available models
        available_models = evaluator.find_available_models()
        
        if not available_models:
            raise FileNotFoundError("No trained models found in the models directory")
        
        if logger:
            logger.info(f"Found models: {list(available_models.keys())}")
        
        # Evaluate each model
        all_saved_files = {}
        
        for model_type, model_path in available_models.items():
            if logger:
                logger.info(f"Evaluating {model_type.upper()} model...")
            
            try:
                # Evaluate model
                if data_type == 'embeddings':
                    # Direct evaluation with embeddings
                    X_test = test_data_obj['X_test']
                    y_test = test_data_obj['y_test']
                    results = evaluator.evaluate_model_direct_embeddings(
                        model_path, X_test, y_test, model_type
                    )
                else:
                    # For CSV data, need to find embeddings
                    # Try to find corresponding embedding files
                    embeddings_dir = Path(models_dir).parent / "embeddings"
                    if not embeddings_dir.exists():
                        embeddings_dir = Path(models_dir).parents[1] / "embeddings"
                    
                    X_test_path = embeddings_dir / "X_test.npy"
                    y_test_path = embeddings_dir / "y_test.npy"
                    
                    if X_test_path.exists() and y_test_path.exists():
                        X_test = np.load(X_test_path)
                        y_test = np.load(y_test_path)
                        results = evaluator.evaluate_model_direct_embeddings(
                            model_path, X_test, y_test, model_type
                        )
                    else:
                        raise FileNotFoundError("Embedding files not found for CSV test data")
                
                evaluator.evaluation_results[model_type] = results
                
                # Create plots
                plot_files = evaluator.create_evaluation_plots(results, model_type)
                
                # Save detailed metrics
                metric_files = evaluator.save_detailed_metrics(results, model_type)
                
                # Combine saved files
                all_saved_files[model_type] = {**plot_files, **metric_files}
                
                if logger:
                    logger.info(f"{model_type.upper()} evaluation completed successfully")
                
            except Exception as e:
                if logger:
                    logger.error(f"Failed to evaluate {model_type} model: {str(e)}")
                
                evaluator.evaluation_errors.append({
                    'model_type': model_type,
                    'error': str(e)
                })
                
                evaluator.skipped_models.append({
                    'model_type': model_type,
                    'model_path': str(model_path),
                    'reason': f'Evaluation failed: {str(e)}'
                })
        
        # Generate comprehensive report
        if evaluator.evaluation_results:
            report_data = evaluator.generate_comprehensive_report()
            
            if logger:
                logger.info("=" * 60)
                logger.info("REPORT GENERATION COMPLETED!")
                logger.info("=" * 60)
                logger.info(f"Models evaluated: {len(evaluator.evaluation_results)}")
                logger.info(f"Reports saved to: {results_dir}")
                
                if 'best_model' in report_data['summary']:
                    best = report_data['summary']['best_model']
                    logger.info(f"Best model: {best['model_type'].upper()} (accuracy: {best['accuracy']:.3f})")
            
            return {
                'success': True,
                'models_evaluated': len(evaluator.evaluation_results),
                'models_skipped': len(evaluator.skipped_models),
                'evaluation_errors': len(evaluator.evaluation_errors),
                'report_files': {
                    'json_report': report_data['report_path'],
                    'text_report': report_data['text_report_path'],
                    'insights': report_data['insights_path']
                },
                'model_files': all_saved_files,
                'summary': report_data['summary']
            }
        else:
            error_msg = "No models could be evaluated successfully"
            if logger:
                logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'evaluation_errors': evaluator.evaluation_errors,
                'skipped_models': evaluator.skipped_models
            }
        
    except Exception as e:
        if logger:
            logger.error(f"Error in report generation pipeline: {str(e)}")
        raise

def parse_arguments():
    """🔧 FIXED: Parse command line arguments with auto-detection"""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive evaluation reports for trained models - FIXED VERSION",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python report.py --auto-default                                                    # Auto-detect latest session
  python report.py --models-dir results/auto_analysis_*/models --test-data data/processed/test.csv --results-dir results/auto_analysis_*/
  python report.py --models-dir results/models --test-data data/embeddings/X_test.npy --results-dir results
        """
    )
    
    # Optional arguments with auto-detection
    parser.add_argument("--models-dir", type=str, default=None,
                       help="Directory containing trained models (auto-detect if not provided)")
    parser.add_argument("--test-data", type=str, default=None,
                       help="Path to test data (CSV with text/label columns or X_test.npy) (auto-detect if not provided)")
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Directory to save reports and plots (auto-detect if not provided)")
    
    # Optional arguments
    parser.add_argument("--model-type", choices=['mlp', 'svm', 'all'], default='all',
                       help="Specific model type to evaluate (default: all)")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Directory for log files (default: results-dir/logs)")
    parser.add_argument("--auto-default", action="store_true",
                       help="🔧 NEW: Auto-detect paths from latest session")
    
    return parser.parse_args()

def main():
    """🔧 FIXED: Main function for CLI usage with auto-detection"""
    args = parse_arguments()

    # 🔧 NEW: Auto-detection mode
    if args.auto_default or not any([args.models_dir, args.test_data, args.results_dir]):
        print("🔍 Auto-detecting latest session...")
        
        models_dir, test_data, results_dir = find_session_models_and_data()
        
        if models_dir and test_data and results_dir:
            args.models_dir = str(models_dir)
            args.test_data = str(test_data)
            args.results_dir = str(results_dir)
            
            print(f"✅ Auto-detected session:")
            print(f"   Models: {models_dir}")
            print(f"   Test data: {test_data}")
            print(f"   Results: {results_dir}")
        else:
            print("❌ Could not auto-detect session. Please provide paths manually.")
            return 1

    # Verify required arguments
    if not all([args.models_dir, args.test_data, args.results_dir]):
        print("❌ Missing required arguments. Use --auto-default or provide --models-dir, --test-data, --results-dir")
        return 1

    # Setup logging
    if args.results_dir:
        log_dir = args.log_dir if args.log_dir else Path(args.results_dir) / "logs"
    else:
        log_dir = args.log_dir if args.log_dir else Path("logs")
    
    logger = setup_logging(log_dir)

    try:
        # Verify inputs
        models_dir = Path(args.models_dir)
        test_data_path = Path(args.test_data)

        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        if not test_data_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_data_path}")
        
        # Run report generation pipeline
        result = generate_evaluation_report(
            models_dir=args.models_dir,
            test_data=args.test_data,
            results_dir=args.results_dir,
            logger=logger
        )
        
        if result['success']:
            logger.info("=" * 60)
            logger.info("REPORT GENERATION SUCCESSFUL!")
            logger.info("=" * 60)
            logger.info(f"Models evaluated: {result['models_evaluated']}")
            logger.info(f"Reports saved: {list(result['report_files'].keys())}")
            
            if 'best_model' in result.get('summary', {}):
                best = result['summary']['best_model']
                logger.info(f"Best model: {best['model_type'].upper()} ({best['accuracy']:.1%} accuracy)")
            
            return 0
        else:
            logger.error("=" * 60)
            logger.error("REPORT GENERATION FAILED!")
            logger.error("=" * 60)
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
            
            if result.get('evaluation_errors'):
                logger.error("Evaluation errors:")
                for error in result['evaluation_errors']:
                    logger.error(f"  {error['model_type']}: {error['error']}")
            
            return 1
        
    except Exception as e:
        logger.error(f"Error during report generation: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
