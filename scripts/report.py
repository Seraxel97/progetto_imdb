#!/usr/bin/env python3
"""
Report Generation Script - PIPELINE AUTOMATION COMPATIBLE
Generates comprehensive evaluation reports for trained sentiment analysis models.

FEATURES:
- Evaluates both MLP and SVM models automatically
- Multiple evaluation methods using direct embeddings
- Comprehensive reporting with plots, metrics, and insights
- Robust error handling with detailed diagnostics
- Full pipeline integration with structured output
- Professional logging system

USAGE:
  python report.py --models-dir results/models --test-data data/processed/test.csv --results-dir results
  python report.py --models-dir results/models --test-data data/embeddings/X_test.npy --results-dir results
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
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"report_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Use utf-8 for the stream handler to avoid cp1252 errors on Windows
    stream_handler = logging.StreamHandler(sys.stdout)
    try:
        stream_handler.stream.reconfigure(encoding="utf-8")
    except AttributeError:
        pass  # Python < 3.7 doesn't support reconfigure

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            stream_handler
        ]
    )
    
    return logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation class with pipeline integration
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
        """Find available trained models"""
        available_models = {}
        
        self.logger.info(f"Searching for models in: {self.models_dir}")
        
        # Look for MLP model
        mlp_search_paths = [
            self.models_dir / "mlp_model.pth",
            self.models_dir / "*" / "mlp_model.pth",
        ]
        
        for search_path in mlp_search_paths:
            if "*" in str(search_path):
                matches = list(self.models_dir.glob("**/mlp_model.pth"))
                if matches:
                    available_models['mlp'] = matches[0]
                    self.logger.info(f"✅ Found MLP model: {matches[0]}")
                    break
            elif search_path.exists():
                available_models['mlp'] = search_path
                self.logger.info(f"✅ Found MLP model: {search_path}")
                break
        
        # Look for SVM model
        svm_search_paths = [
            self.models_dir / "svm_model.pkl",
            self.models_dir / "*" / "svm_model.pkl",
        ]
        
        for search_path in svm_search_paths:
            if "*" in str(search_path):
                matches = list(self.models_dir.glob("**/svm_model.pkl"))
                if matches:
                    available_models['svm'] = matches[0]
                    self.logger.info(f"✅ Found SVM model: {matches[0]}")
                    break
            elif search_path.exists():
                available_models['svm'] = search_path
                self.logger.info(f"✅ Found SVM model: {search_path}")
                break
        
        if not available_models:
            self.logger.warning(f"No models found in: {self.models_dir}")
            self.logger.info("   Searched for: mlp_model.pth, svm_model.pkl")
        
        return available_models
    
    def load_test_data(self, test_data_path):
        """Load test data from CSV or numpy files"""
        test_path = Path(test_data_path)
        
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        if test_path.suffix == '.csv':
            self.logger.info(f"Loading test data from CSV: {test_path}")
            test_df = pd.read_csv(test_path)
            
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
            
            # Return as dict for compatibility
            return {'X_test': X_test, 'y_test': y_test}, 'embeddings'
        else:
            raise ValueError(f"Unsupported test data format: {test_path.suffix}")
    
    def evaluate_model_direct_embeddings(self, model_path, X_test, y_test, model_type):
        """Direct model evaluation using embeddings"""
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
        """Direct SVM evaluation using embeddings"""
        model_path = Path(model_path)

        if not model_path.exists():
            self.logger.error(f"Model not found: {model_path}")
            raise RuntimeError(f"Model not found: {model_path}")

        try:
            self.logger.info("Loading SVM model package...")
            
            # Load SVM model package
            model_package = joblib.load(model_path)
            
            # Extract components
            model = model_package['model']
            scaler = model_package.get('scaler')
            label_encoder = model_package.get('label_encoder')
            
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
        """Direct MLP evaluation using embeddings and PyTorch model"""
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
            
            # Load model
            if str(model_path).endswith('.pth'):
                # Load state dict
                model = HateSpeechMLP(input_dim=384).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
            else:
                # Try to load complete model
                model = torch.load(model_path, map_location=device)
            
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
        """Create evaluation plots"""
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
        """Save detailed metrics in multiple formats"""
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
            
            # Save JSON
            json_path = self.reports_dir / f'{model_type}_detailed_metrics.json'
            with open(json_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            # Save classification report as CSV
            class_report_df = pd.DataFrame(results['classification_report']).transpose()
            csv_path = self.reports_dir / f'{model_type}_classification_report.csv'
            class_report_df.to_csv(csv_path)
            
            self.logger.info(f"Detailed metrics saved for {model_type}")
            
            return {
                'json_metrics': str(json_path),
                'csv_report': str(csv_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error saving metrics for {model_type}: {str(e)}")
            return {}
    
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
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
            
            # Save comprehensive results
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
        """Create human-readable text report"""
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
        
        # Save text report
        text_report_path = self.reports_dir / 'evaluation_report.txt'
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Text report saved: {text_report_path}")
        return text_report_path
    
    def create_insights_file(self, summary):
        """Create standalone insights file"""
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
        
        # Save insights
        insights_path = self.reports_dir / 'insights.txt'
        with open(insights_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(insights_lines))
        
        self.logger.info(f"Insights saved: {insights_path}")
        return insights_path

def generate_evaluation_report(models_dir, test_data, results_dir, logger=None):
    """
    Main pipeline function for generating evaluation reports
    
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
            logger.info("EVALUATION REPORT GENERATION")
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
                    embeddings_dir = Path(models_dir).parent / "data" / "embeddings"
                    if not embeddings_dir.exists():
                        embeddings_dir = Path(models_dir).parents[1] / "data" / "embeddings"
                    
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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive evaluation reports for trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python report.py --models-dir results/models --test-data data/processed/test.csv --results-dir results
  python report.py --models-dir results/models --test-data data/embeddings/X_test.npy --results-dir results
        """
    )
    
    # Required arguments (now optional with fallback)
    parser.add_argument("--models-dir", type=str, default=None,
                       help="Directory containing trained models")
    parser.add_argument("--test-data", type=str, default=None,
                       help="Path to test data (CSV with text/label columns or X_test.npy)")
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Directory to save reports and plots")
    
    # Optional arguments
    parser.add_argument("--model-type", choices=['mlp', 'svm', 'all'], default='all',
                       help="Specific model type to evaluate (default: all)")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Directory for log files (default: results-dir/logs)")
    parser.add_argument("--auto-default", action="store_true",
                       help="Use default paths if no arguments are provided")
    
    return parser.parse_args()

def main():
    """Main function for CLI usage"""
    args = parse_arguments()

    project_root = PROJECT_ROOT

    # Determine initial results directory for logging setup
    default_results = project_root / "results"
    results_for_logging = args.results_dir if args.results_dir else str(default_results)
    log_dir = args.log_dir if args.log_dir else Path(results_for_logging) / "logs"
    logger = setup_logging(log_dir)

    # Apply fallback defaults when arguments are missing
    if not args.models_dir:
        default_models = project_root / "results" / "models"
        if args.auto_default or default_models.exists():
            logger.warning(f"⚠️ No models-dir provided. Using default: {default_models}")
            args.models_dir = str(default_models)

    if not args.test_data:
        default_test = project_root / "data" / "processed" / "test.csv"
        if args.auto_default or default_test.exists():
            logger.warning(f"⚠️ No test-data provided. Using default: {default_test}")
            args.test_data = str(default_test)

    if not args.results_dir:
        if args.auto_default or default_results.exists():
            logger.warning(f"⚠️ No results-dir provided. Using default: {default_results}")
            args.results_dir = str(default_results)

    # If any critical argument is still missing, exit gracefully
    if not all([args.models_dir, args.test_data, args.results_dir]):
        logger.error("❌ Missing critical arguments and fallback paths are not valid. Provide --models-dir, --test-data, or enable --auto-default.")
        return 1

    try:
        # Verify inputs
        models_dir = Path(args.models_dir)
        test_data_path = Path(args.test_data)

        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        if not test_data_path.exists():
            logger.error("❌ test.csv not found. Provide --test-data or ensure fallback file exists.")
            return 1
        
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
