#!/usr/bin/env python3
"""
Enhanced Report Generation Script - ADAPTIVE & UNIVERSAL REPORTING
Generates comprehensive evaluation reports with intelligent adaptation to available data.

🆕 ENHANCED FEATURES:
- ✅ Adaptive reporting: works with any combination of available models
- ✅ Intelligent data discovery: automatically finds test data and models
- ✅ Graceful degradation: creates meaningful reports even with missing components
- ✅ Universal CSV support: works with inference.csv, test.csv, or any processed data
- ✅ Enhanced visualizations and insights generation
- ✅ Robust error handling with detailed diagnostics
- ✅ GUI integration with comprehensive status reporting

USAGE:
    python scripts/report.py                                       # Auto-detect everything
    python scripts/report.py --auto-default                        # Auto-detect latest session
    python scripts/report.py --models-dir results/models           # Specific models directory
    python scripts/report.py --create-inference-report             # Inference-only report
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
from typing import Dict, List, Optional, Tuple, Any

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
    """Setup logging configuration with robust UTF-8 support"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"enhanced_report_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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

class UniversalReportGenerator:
    """
    Universal report generator that adapts to available data and models.
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.available_models = {}
        self.test_data = None
        self.evaluation_results = {}
        self.report_metadata = {}
        
        self.logger.info(f"🚀 Universal Report Generator initialized")
    
    def discover_available_resources(self, models_dir: Path = None, test_data_path: Path = None,
                                   results_dir: Path = None) -> Dict[str, Any]:
        """🆕 NEW: Discover all available resources for report generation"""
        self.logger.info("🔍 Discovering available resources...")
        
        discovery = {
            'models_found': {},
            'test_data_found': None,
            'embeddings_found': {},
            'results_dirs': [],
            'report_mode': 'unknown',
            'recommendations': []
        }
        
        # Auto-detect paths if not provided
        if models_dir is None or test_data_path is None or results_dir is None:
            auto_paths = self._auto_detect_paths()
            if models_dir is None:
                models_dir = auto_paths.get('models_dir')
            if test_data_path is None:
                test_data_path = auto_paths.get('test_data')
            if results_dir is None:
                results_dir = auto_paths.get('results_dir')
        
        # Discover models
        if models_dir and models_dir.exists():
            discovery['models_found'] = self._discover_models(models_dir)
        
        # Discover test data
        if test_data_path and test_data_path.exists():
            discovery['test_data_found'] = self._analyze_test_data(test_data_path)
        else:
            # Look for test data in common locations
            test_candidates = self._find_test_data_candidates(results_dir)
            if test_candidates:
                discovery['test_data_found'] = self._analyze_test_data(test_candidates[0])
        
        # Discover embeddings (for model evaluation)
        if results_dir:
            discovery['embeddings_found'] = self._discover_embeddings(results_dir)
        
        # Determine report mode
        discovery['report_mode'] = self._determine_report_mode(discovery)
        
        # Generate recommendations
        discovery['recommendations'] = self._generate_recommendations(discovery)
        
        self.logger.info(f"🎯 Discovery results:")
        self.logger.info(f"   Models found: {len(discovery['models_found'])}")
        self.logger.info(f"   Test data: {'✅' if discovery['test_data_found'] else '❌'}")
        self.logger.info(f"   Report mode: {discovery['report_mode']}")
        
        return discovery
    
    def _auto_detect_paths(self) -> Dict[str, Path]:
        """Auto-detect common paths for models and data"""
        auto_paths = {}
        
        # Look for latest session directory
        results_dir = PROJECT_ROOT / "results"
        if results_dir.exists():
            session_dirs = [d for d in results_dir.iterdir() 
                          if d.is_dir() and (d.name.startswith("session_") or d.name.startswith("auto_analysis_"))]
            
            if session_dirs:
                # Sort by modification time (most recent first)
                session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                latest_session = session_dirs[0]
                
                auto_paths['results_dir'] = latest_session
                auto_paths['models_dir'] = latest_session / "models"
                
                # Look for test data in session
                test_candidates = [
                    latest_session / "processed" / "test.csv",
                    latest_session / "processed" / "inference.csv",
                    latest_session / "embeddings" / "X_test.npy"
                ]
                
                for candidate in test_candidates:
                    if candidate.exists():
                        auto_paths['test_data'] = candidate
                        break
        
        # Fallback to standard paths
        if 'models_dir' not in auto_paths:
            auto_paths['models_dir'] = PROJECT_ROOT / "results" / "models"
        if 'test_data' not in auto_paths:
            auto_paths['test_data'] = PROJECT_ROOT / "data" / "processed" / "test.csv"
        if 'results_dir' not in auto_paths:
            auto_paths['results_dir'] = PROJECT_ROOT / "results"
        
        self.logger.info(f"🔧 Auto-detected paths:")
        for key, path in auto_paths.items():
            self.logger.info(f"   {key}: {path}")
        
        return auto_paths
    
    def _discover_models(self, models_dir: Path) -> Dict[str, Any]:
        """Discover available trained models"""
        models_found = {}
        
        model_patterns = {
            'mlp': ['mlp_model.pth', 'mlp_model_complete.pth'],
            'svm': ['svm_model.pkl']
        }
        
        for model_type, patterns in model_patterns.items():
            for pattern in patterns:
                model_files = list(models_dir.glob(f"**/{pattern}"))
                if model_files:
                    model_path = model_files[0]
                    
                    # Try to load model info
                    try:
                        if model_type == 'mlp':
                            model_info = self._analyze_mlp_model(model_path)
                        elif model_type == 'svm':
                            model_info = self._analyze_svm_model(model_path)
                        
                        models_found[model_type] = {
                            'path': model_path,
                            'size_mb': model_path.stat().st_size / (1024 * 1024),
                            'last_modified': datetime.fromtimestamp(model_path.stat().st_mtime),
                            **model_info
                        }
                        
                        self.logger.info(f"   ✅ Found {model_type.upper()} model: {model_path}")
                        
                    except Exception as e:
                        self.logger.warning(f"   ⚠️ {model_type.upper()} model found but not loadable: {e}")
                    
                    break
        
        return models_found
    
    def _analyze_mlp_model(self, model_path: Path) -> Dict[str, Any]:
        """Analyze MLP model file"""
        try:
            # Try to load model metadata if available
            metadata_path = model_path.parent / "mlp_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return {
                    'type': 'MLP',
                    'loadable': True,
                    'metadata': metadata,
                    'architecture': metadata.get('model_info', {}).get('architecture', 'unknown'),
                    'accuracy': metadata.get('final_performance', {}).get('validation_accuracy', 'unknown')
                }
            else:
                # Try to load model directly
                model = torch.load(model_path, map_location='cpu')
                return {
                    'type': 'MLP',
                    'loadable': True,
                    'metadata': None,
                    'architecture': 'unknown',
                    'accuracy': 'unknown'
                }
        except Exception as e:
            return {
                'type': 'MLP',
                'loadable': False,
                'error': str(e)
            }
    
    def _analyze_svm_model(self, model_path: Path) -> Dict[str, Any]:
        """Analyze SVM model file"""
        try:
            # Try to load model metadata if available
            metadata_path = model_path.parent / "svm_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return {
                    'type': 'SVM',
                    'loadable': True,
                    'metadata': metadata,
                    'accuracy': metadata.get('validation_accuracy', 'unknown'),
                    'training_time': metadata.get('training_time_seconds', 'unknown')
                }
            else:
                # Try to load model directly
                model_package = joblib.load(model_path)
                return {
                    'type': 'SVM',
                    'loadable': True,
                    'metadata': None,
                    'accuracy': model_package.get('validation_accuracy', 'unknown'),
                    'training_time': model_package.get('training_time', 'unknown')
                }
        except Exception as e:
            return {
                'type': 'SVM',
                'loadable': False,
                'error': str(e)
            }
    
    def _analyze_test_data(self, test_data_path: Path) -> Dict[str, Any]:
        """Analyze test data file"""
        try:
            if test_data_path.suffix == '.csv':
                df = pd.read_csv(test_data_path)
                df.columns = df.columns.str.strip().str.lower()
                
                has_text = 'text' in df.columns
                has_labels = 'label' in df.columns and not df['label'].isnull().all()
                
                return {
                    'path': test_data_path,
                    'type': 'csv',
                    'samples': len(df),
                    'has_text': has_text,
                    'has_labels': has_labels,
                    'columns': list(df.columns),
                    'valid': has_text
                }
            
            elif test_data_path.suffix == '.npy':
                data = np.load(test_data_path)
                
                # Look for corresponding labels file
                if test_data_path.name.startswith('X_'):
                    labels_file = test_data_path.parent / test_data_path.name.replace('X_', 'y_')
                    has_labels = labels_file.exists()
                else:
                    has_labels = False
                
                return {
                    'path': test_data_path,
                    'type': 'npy',
                    'samples': data.shape[0],
                    'features': data.shape[1] if len(data.shape) > 1 else 1,
                    'has_labels': has_labels,
                    'valid': True
                }
                
        except Exception as e:
            return {
                'path': test_data_path,
                'type': 'unknown',
                'valid': False,
                'error': str(e)
            }
    
    def _find_test_data_candidates(self, results_dir: Path) -> List[Path]:
        """Find test data candidates in results directory"""
        candidates = []
        
        if not results_dir or not results_dir.exists():
            return candidates
        
        # Look for test files in common locations
        search_patterns = [
            "processed/test.csv",
            "processed/inference.csv", 
            "embeddings/X_test.npy",
            "embeddings/X_inference.npy",
            "*/processed/test.csv",
            "*/processed/inference.csv",
            "*/embeddings/X_test.npy"
        ]
        
        for pattern in search_patterns:
            found_files = list(results_dir.glob(pattern))
            candidates.extend(found_files)
        
        # Remove duplicates and sort by modification time
        candidates = list(set(candidates))
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return candidates
    
    def _discover_embeddings(self, results_dir: Path) -> Dict[str, Any]:
        """Discover available embedding files for evaluation"""
        embeddings = {}
        
        if not results_dir or not results_dir.exists():
            return embeddings
        
        # Look for embedding files
        embedding_patterns = ['embeddings/X_*.npy', '*/embeddings/X_*.npy']
        
        for pattern in embedding_patterns:
            embedding_files = list(results_dir.glob(pattern))
            
            for emb_file in embedding_files:
                # Extract split name
                split_name = emb_file.stem.replace('X_', '')
                
                # Look for corresponding labels
                labels_file = emb_file.parent / f"y_{split_name}.npy"
                
                if labels_file.exists():
                    try:
                        X_data = np.load(emb_file)
                        y_data = np.load(labels_file)
                        
                        embeddings[split_name] = {
                            'X_path': emb_file,
                            'y_path': labels_file,
                            'X_shape': X_data.shape,
                            'y_shape': y_data.shape,
                            'valid': X_data.shape[0] == y_data.shape[0]
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"   ⚠️ Invalid embedding files for {split_name}: {e}")
        
        return embeddings
    
    def _determine_report_mode(self, discovery: Dict[str, Any]) -> str:
        """Determine the best report mode based on available resources"""
        has_models = len(discovery['models_found']) > 0
        has_test_data = discovery['test_data_found'] is not None and discovery['test_data_found']['valid']
        has_embeddings = len(discovery['embeddings_found']) > 0
        
        if has_models and has_test_data and has_embeddings:
            return 'full_evaluation'
        elif has_models and has_embeddings:
            return 'model_evaluation'
        elif has_test_data:
            return 'data_analysis'
        elif has_models:
            return 'model_summary'
        else:
            return 'minimal_report'
    
    def _generate_recommendations(self, discovery: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on discovery results"""
        recommendations = []
        
        mode = discovery['report_mode']
        
        if mode == 'full_evaluation':
            recommendations.append("✅ Complete evaluation possible with models and test data")
        elif mode == 'model_evaluation':
            recommendations.append("⚠️ Model evaluation possible but test data limited")
        elif mode == 'data_analysis':
            recommendations.append("📊 Only data analysis possible - no trained models found")
        elif mode == 'model_summary':
            recommendations.append("📋 Only model summary possible - no test data found")
        else:
            recommendations.append("❌ Limited reporting - insufficient data and models")
        
        # Specific recommendations
        if not discovery['models_found']:
            recommendations.append("🔧 Run training scripts to generate models")
        
        if not discovery['test_data_found'] or not discovery['test_data_found']['valid']:
            recommendations.append("📄 Ensure test.csv or inference.csv is available")
        
        return recommendations
    
    def evaluate_models_on_data(self, models_info: Dict[str, Any], 
                               test_data_info: Dict[str, Any],
                               embeddings_info: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate available models on test data"""
        self.logger.info("🧪 Evaluating models on test data...")
        
        evaluation_results = {}
        
        # Determine which data to use for evaluation
        test_split = None
        if 'test' in embeddings_info:
            test_split = 'test'
        elif 'inference' in embeddings_info:
            test_split = 'inference'
        elif embeddings_info:
            # Use any available split
            test_split = list(embeddings_info.keys())[0]
        
        if not test_split:
            self.logger.warning("⚠️ No suitable embedding data found for evaluation")
            return evaluation_results
        
        # Load test embeddings and labels
        try:
            X_test = np.load(embeddings_info[test_split]['X_path'])
            y_test = np.load(embeddings_info[test_split]['y_path'])
            
            self.logger.info(f"   📊 Using {test_split} data: {X_test.shape[0]} samples")
            
            # Filter out placeholder labels if in inference mode
            if test_split == 'inference':
                valid_mask = np.isin(y_test, [0, 1])
                if valid_mask.sum() == 0:
                    self.logger.warning("⚠️ No valid labels in inference data - skipping model evaluation")
                    return evaluation_results
                X_test = X_test[valid_mask]
                y_test = y_test[valid_mask]
                self.logger.info(f"   📊 Filtered to {X_test.shape[0]} samples with valid labels")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading test data: {e}")
            return evaluation_results
        
        # Evaluate each model
        for model_name, model_info in models_info.items():
            if not model_info.get('loadable', False):
                self.logger.warning(f"⚠️ Skipping {model_name} - not loadable")
                continue
            
            try:
                self.logger.info(f"   🔄 Evaluating {model_name.upper()} model...")
                
                if model_name == 'mlp':
                    results = self._evaluate_mlp_model(model_info['path'], X_test, y_test)
                elif model_name == 'svm':
                    results = self._evaluate_svm_model(model_info['path'], X_test, y_test)
                else:
                    continue
                
                evaluation_results[model_name] = results
                
                self.logger.info(f"   ✅ {model_name.upper()}: Accuracy={results['accuracy']:.3f}, F1={results['f1_score']:.3f}")
                
            except Exception as e:
                self.logger.error(f"   ❌ Error evaluating {model_name}: {e}")
                evaluation_results[model_name] = {
                    'error': str(e),
                    'evaluated': False
                }
        
        return evaluation_results
    
    def _evaluate_mlp_model(self, model_path: Path, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate MLP model on test data"""
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
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Load model
            model = HateSpeechMLP(input_dim=X_test.shape[1]).to(device)
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
            
            model.eval()
            
            # Make predictions
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            
            with torch.no_grad():
                outputs = model(X_test_tensor)
                probabilities = outputs.squeeze().cpu().numpy()
                
                if probabilities.ndim == 0:
                    probabilities = np.array([probabilities])
                
                predictions = (probabilities > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')
            
            # Classification report
            class_report = classification_report(
                y_test, predictions,
                target_names=['Negative', 'Positive'],
                output_dict=True
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test, predictions)
            
            return {
                'model_type': 'MLP',
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'true_labels': y_test.tolist(),
                'classification_report': class_report,
                'confusion_matrix': cm.tolist(),
                'num_samples': len(X_test),
                'evaluated': True
            }
            
        except Exception as e:
            raise RuntimeError(f"MLP evaluation failed: {str(e)}")
    
    def _evaluate_svm_model(self, model_path: Path, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate SVM model on test data"""
        try:
            # Load SVM model package
            model_package = joblib.load(model_path)
            
            # Extract components
            if isinstance(model_package, dict):
                model = model_package.get('model')
                scaler = model_package.get('scaler')
                label_encoder = model_package.get('label_encoder')
            else:
                model = model_package
                scaler = None
                label_encoder = None
            
            if model is None:
                raise ValueError("No model found in the package")
            
            # Apply preprocessing
            if scaler is not None:
                X_test_scaled = scaler.transform(X_test)
            else:
                X_test_scaled = X_test
            
            if label_encoder is not None:
                y_test_encoded = label_encoder.transform(y_test)
            else:
                y_test_encoded = y_test
            
            # Make predictions
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
            
            # Classification report
            class_report = classification_report(
                y_test_encoded, predictions,
                target_names=['Negative', 'Positive'],
                output_dict=True
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test_encoded, predictions)
            
            return {
                'model_type': 'SVM',
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'predictions': predictions.tolist(),
                'confidences': confidences.tolist(),
                'true_labels': y_test_encoded.tolist(),
                'classification_report': class_report,
                'confusion_matrix': cm.tolist(),
                'num_samples': len(X_test),
                'evaluated': True
            }
            
        except Exception as e:
            raise RuntimeError(f"SVM evaluation failed: {str(e)}")
    
    def create_comprehensive_report(self, discovery: Dict[str, Any], 
                                  evaluation_results: Dict[str, Any],
                                  output_dir: Path) -> Dict[str, Any]:
        """Create comprehensive report with all available information"""
        self.logger.info("📋 Creating comprehensive report...")
        
        # Create output directories
        reports_dir = output_dir / "reports"
        plots_dir = output_dir / "plots"
        reports_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib backend
        import matplotlib
        matplotlib.use('Agg')
        
        report_data = {
            'report_metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'report_mode': discovery['report_mode'],
                'models_evaluated': len(evaluation_results),
                'total_models_found': len(discovery['models_found']),
                'has_test_data': discovery['test_data_found'] is not None
            },
            'discovery_results': discovery,
            'evaluation_results': evaluation_results,
            'summary': {},
            'insights': [],
            'created_files': []
        }
        
        # Create model comparison plot if multiple models
        if len(evaluation_results) > 1:
            comparison_plot = self._create_model_comparison_plot(evaluation_results, plots_dir)
            if comparison_plot:
                report_data['created_files'].append(comparison_plot)
        
        # Create individual model plots
        for model_name, results in evaluation_results.items():
            if results.get('evaluated', False):
                model_plots = self._create_model_plots(model_name, results, plots_dir)
                report_data['created_files'].extend(model_plots)
        
        # Create data analysis plots
        if discovery['test_data_found']:
            data_plots = self._create_data_analysis_plots(discovery['test_data_found'], plots_dir)
            report_data['created_files'].extend(data_plots)
        
        # Generate summary and insights
        report_data['summary'] = self._generate_summary(discovery, evaluation_results)
        report_data['insights'] = self._generate_insights(discovery, evaluation_results)
        
        # Save JSON report
        json_report_path = reports_dir / "evaluation_report.json"
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        report_data['created_files'].append(str(json_report_path))
        
        # Create text report
        text_report_path = self._create_text_report(report_data, reports_dir)
        report_data['created_files'].append(str(text_report_path))
        
        # Create insights file
        insights_path = self._create_insights_file(report_data, reports_dir)
        report_data['created_files'].append(str(insights_path))
        
        self.logger.info(f"📄 Comprehensive report created with {len(report_data['created_files'])} files")
        
        return report_data
    
    def _create_model_comparison_plot(self, evaluation_results: Dict[str, Any], plots_dir: Path) -> Optional[str]:
        """Create model comparison plot"""
        try:
            evaluated_models = {name: results for name, results in evaluation_results.items() 
                              if results.get('evaluated', False)}
            
            if len(evaluated_models) < 2:
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Accuracy comparison
            models = list(evaluated_models.keys())
            accuracies = [evaluated_models[model]['accuracy'] for model in models]
            f1_scores = [evaluated_models[model]['f1_score'] for model in models]
            
            ax1.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)])
            ax1.set_title('Model Accuracy Comparison')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            
            # Add value labels
            for i, acc in enumerate(accuracies):
                ax1.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
            
            # F1-Score comparison
            ax2.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)])
            ax2.set_title('Model F1-Score Comparison')
            ax2.set_ylabel('F1-Score')
            ax2.set_ylim(0, 1)
            
            # Add value labels
            for i, f1 in enumerate(f1_scores):
                ax2.text(i, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            plot_path = plots_dir / "model_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Error creating model comparison plot: {e}")
            return None
    
    def _create_model_plots(self, model_name: str, results: Dict[str, Any], plots_dir: Path) -> List[str]:
        """Create plots for individual model"""
        created_plots = []
        
        try:
            # Confusion matrix
            if 'confusion_matrix' in results:
                plt.figure(figsize=(8, 6))
                cm = np.array(results['confusion_matrix'])
                
                sns.heatmap(cm, annot=True, fmt='d',
                           xticklabels=['Negative', 'Positive'],
                           yticklabels=['Negative', 'Positive'],
                           cmap='Blues')
                plt.title(f'{model_name.upper()} Confusion Matrix\nAccuracy: {results["accuracy"]:.3f}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                
                plot_path = plots_dir / f"{model_name}_confusion_matrix.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                created_plots.append(str(plot_path))
            
            # Performance metrics
            plt.figure(figsize=(8, 6))
            metrics = ['Accuracy', 'F1-Score']
            values = [results['accuracy'], results['f1_score']]
            
            bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e'])
            plt.title(f'{model_name.upper()} Performance Metrics')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            plot_path = plots_dir / f"{model_name}_metrics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            created_plots.append(str(plot_path))
            
        except Exception as e:
            self.logger.error(f"Error creating plots for {model_name}: {e}")
        
        return created_plots
    
    def _create_data_analysis_plots(self, test_data_info: Dict[str, Any], plots_dir: Path) -> List[str]:
        """Create data analysis plots"""
        created_plots = []
        
        try:
            if test_data_info['type'] == 'csv' and test_data_info['has_text']:
                df = pd.read_csv(test_data_info['path'])
                df.columns = df.columns.str.strip().str.lower()
                
                # Text length distribution
                if 'text' in df.columns:
                    text_lengths = df['text'].astype(str).str.len()
                    
                    plt.figure(figsize=(10, 6))
                    plt.hist(text_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                    plt.title('Text Length Distribution')
                    plt.xlabel('Text Length (characters)')
                    plt.ylabel('Frequency')
                    plt.axvline(text_lengths.mean(), color='red', linestyle='--', 
                               label=f'Mean: {text_lengths.mean():.0f}')
                    plt.legend()
                    plt.tight_layout()
                    
                    plot_path = plots_dir / "text_length_distribution.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    created_plots.append(str(plot_path))
                
                # Label distribution if available
                if test_data_info['has_labels'] and 'label' in df.columns:
                    label_counts = df['label'].value_counts()
                    
                    plt.figure(figsize=(8, 6))
                    bars = plt.bar(label_counts.index.astype(str), label_counts.values, 
                                  color=['#ff7f7f', '#7f7fff'])
                    plt.title('Label Distribution')
                    plt.xlabel('Label')
                    plt.ylabel('Count')
                    
                    # Add value labels
                    for bar, count in zip(bars, label_counts.values):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                str(count), ha='center', va='bottom')
                    
                    plt.tight_layout()
                    
                    plot_path = plots_dir / "label_distribution.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    created_plots.append(str(plot_path))
            
        except Exception as e:
            self.logger.error(f"Error creating data analysis plots: {e}")
        
        return created_plots
    
    def _generate_summary(self, discovery: Dict[str, Any], evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report summary"""
        summary = {
            'report_mode': discovery['report_mode'],
            'models_found': len(discovery['models_found']),
            'models_evaluated': len([r for r in evaluation_results.values() if r.get('evaluated', False)]),
            'test_data_available': discovery['test_data_found'] is not None,
            'best_model': None,
            'performance_summary': {}
        }
        
        # Find best model if any were evaluated
        evaluated_models = {name: results for name, results in evaluation_results.items() 
                          if results.get('evaluated', False)}
        
        if evaluated_models:
            best_model_name = max(evaluated_models.keys(), 
                                key=lambda k: evaluated_models[k]['accuracy'])
            summary['best_model'] = {
                'name': best_model_name,
                'accuracy': evaluated_models[best_model_name]['accuracy'],
                'f1_score': evaluated_models[best_model_name]['f1_score']
            }
            
            # Performance summary
            for model_name, results in evaluated_models.items():
                summary['performance_summary'][model_name] = {
                    'accuracy': results['accuracy'],
                    'f1_score': results['f1_score'],
                    'samples_tested': results['num_samples']
                }
        
        return summary
    
    def _generate_insights(self, discovery: Dict[str, Any], evaluation_results: Dict[str, Any]) -> List[str]:
        """Generate insights from the evaluation"""
        insights = []
        
        # Model availability insights
        if len(discovery['models_found']) == 0:
            insights.append("No trained models found - run training scripts to generate models")
        elif len(discovery['models_found']) == 1:
            model_name = list(discovery['models_found'].keys())[0]
            insights.append(f"Only {model_name.upper()} model available - consider training additional models for comparison")
        else:
            insights.append(f"Multiple models available ({len(discovery['models_found'])}) - comprehensive comparison possible")
        
        # Performance insights
        evaluated_models = {name: results for name, results in evaluation_results.items() 
                          if results.get('evaluated', False)}
        
        if evaluated_models:
            accuracies = [results['accuracy'] for results in evaluated_models.values()]
            best_acc = max(accuracies)
            worst_acc = min(accuracies)
            
            if best_acc >= 0.85:
                insights.append("Excellent model performance achieved (≥85% accuracy)")
            elif best_acc >= 0.80:
                insights.append("Good model performance achieved (≥80% accuracy)")
            elif best_acc >= 0.75:
                insights.append("Acceptable model performance achieved (≥75% accuracy)")
            else:
                insights.append("Model performance below expectations (<75% accuracy)")
            
            if len(evaluated_models) > 1:
                acc_diff = best_acc - worst_acc
                if acc_diff > 0.05:
                    insights.append(f"Significant performance difference between models ({acc_diff:.3f})")
                else:
                    insights.append("Models show similar performance levels")
        
        # Data insights
        if discovery['test_data_found']:
            test_info = discovery['test_data_found']
            insights.append(f"Test data contains {test_info['samples']} samples")
            
            if not test_info['has_labels']:
                insights.append("Test data has no labels - inference mode analysis only")
        else:
            insights.append("No test data available for evaluation")
        
        # Recommendations
        if discovery['report_mode'] == 'minimal_report':
            insights.append("Limited analysis possible - ensure models and test data are available")
        elif discovery['report_mode'] == 'data_analysis':
            insights.append("Only data analysis performed - train models for comprehensive evaluation")
        
        return insights
    
    def _create_text_report(self, report_data: Dict[str, Any], reports_dir: Path) -> str:
        """Create human-readable text report"""
        report_lines = [
            "ENHANCED SENTIMENT ANALYSIS EVALUATION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Report Mode: {report_data['report_metadata']['report_mode'].upper()}",
            "",
            "SUMMARY:",
            f"  Models Found: {report_data['report_metadata']['total_models_found']}",
            f"  Models Evaluated: {report_data['report_metadata']['models_evaluated']}",
            f"  Test Data Available: {'Yes' if report_data['report_metadata']['has_test_data'] else 'No'}",
            ""
        ]
        
        # Model performance section
        if report_data['evaluation_results']:
            report_lines.extend([
                "MODEL PERFORMANCE:",
                "-" * 20
            ])
            
            for model_name, results in report_data['evaluation_results'].items():
                if results.get('evaluated', False):
                    report_lines.extend([
                        f"",
                        f"{model_name.upper()} MODEL:",
                        f"  Accuracy: {results['accuracy']:.4f}",
                        f"  F1-Score: {results['f1_score']:.4f}",
                        f"  Samples Tested: {results['num_samples']:,}"
                    ])
        
        # Best model
        if 'best_model' in report_data['summary'] and report_data['summary']['best_model']:
            best = report_data['summary']['best_model']
            report_lines.extend([
                "",
                "BEST PERFORMING MODEL:",
                f"  Model: {best['name'].upper()}",
                f"  Accuracy: {best['accuracy']:.4f}",
                f"  F1-Score: {best['f1_score']:.4f}"
            ])
        
        # Insights
        if report_data['insights']:
            report_lines.extend([
                "",
                "KEY INSIGHTS:",
                "-" * 15
            ])
            for insight in report_data['insights']:
                report_lines.append(f"  • {insight}")
        
        # Files created
        report_lines.extend([
            "",
            "GENERATED FILES:",
            "-" * 15
        ])
        for file_path in report_data['created_files']:
            report_lines.append(f"  📄 {Path(file_path).name}")
        
        report_lines.extend([
            "",
            "=" * 60,
            "Report generation completed."
        ])
        
        # Save text report
        text_report_path = reports_dir / 'evaluation_report.txt'
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        return str(text_report_path)
    
    def _create_insights_file(self, report_data: Dict[str, Any], reports_dir: Path) -> str:
        """Create standalone insights file"""
        insights_lines = [
            "SENTIMENT ANALYSIS EVALUATION INSIGHTS",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "KEY FINDINGS:"
        ]
        
        if report_data['insights']:
            for i, insight in enumerate(report_data['insights'], 1):
                insights_lines.append(f"{i}. {insight}")
        else:
            insights_lines.append("No specific insights generated.")
        
        if 'best_model' in report_data['summary'] and report_data['summary']['best_model']:
            best = report_data['summary']['best_model']
            insights_lines.extend([
                "",
                "RECOMMENDATION:",
                f"Use the {best['name'].upper()} model for production deployment.",
                f"It achieved {best['accuracy']:.1%} accuracy on the test set."
            ])
        
        insights_lines.extend([
            "",
            "PERFORMANCE BENCHMARKS:",
            "• >85% accuracy: Excellent performance",
            "• >80% accuracy: Good performance", 
            "• >75% accuracy: Acceptable performance",
            "• <75% accuracy: Needs improvement",
            "",
            "=" * 50
        ])
        
        # Save insights file
        insights_path = reports_dir / 'insights.txt'
        with open(insights_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(insights_lines))
        
        return str(insights_path)

def generate_enhanced_report(models_dir: Path = None, test_data: Path = None, 
                           results_dir: Path = None, logger=None) -> Dict[str, Any]:
    """
    🆕 NEW: Generate enhanced evaluation report with intelligent adaptation
    
    Args:
        models_dir: Directory containing trained models (auto-detect if None)
        test_data: Path to test data (auto-detect if None)
        results_dir: Directory to save results (auto-detect if None)
        logger: Logger instance
    
    Returns:
        Report generation results
    """
    try:
        if logger:
            logger.info("=" * 60)
            logger.info("ENHANCED REPORT GENERATION")
            logger.info("=" * 60)
        
        # Initialize generator
        generator = UniversalReportGenerator(logger)
        
        # Discover available resources
        discovery = generator.discover_available_resources(models_dir, test_data, results_dir)
        
        if logger:
            logger.info(f"Discovery completed: {discovery['report_mode']} mode")
            for rec in discovery['recommendations']:
                logger.info(f"  {rec}")
        
        # Set output directory
        if results_dir is None:
            if discovery.get('results_dirs'):
                results_dir = discovery['results_dirs'][0]
            else:
                results_dir = PROJECT_ROOT / "results"
        
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate models if possible
        evaluation_results = {}
        if discovery['models_found'] and discovery['embeddings_found']:
            evaluation_results = generator.evaluate_models_on_data(
                discovery['models_found'],
                discovery['test_data_found'],
                discovery['embeddings_found']
            )
        
        # Create comprehensive report
        report_data = generator.create_comprehensive_report(
            discovery, evaluation_results, results_dir
        )
        
        if logger:
            logger.info("=" * 60)
            logger.info("ENHANCED REPORT GENERATION COMPLETED!")
            logger.info("=" * 60)
            logger.info(f"Report mode: {report_data['report_metadata']['report_mode']}")
            logger.info(f"Models evaluated: {report_data['report_metadata']['models_evaluated']}")
            logger.info(f"Files created: {len(report_data['created_files'])}")
            
            if report_data['summary'].get('best_model'):
                best = report_data['summary']['best_model']
                logger.info(f"Best model: {best['name'].upper()} (accuracy: {best['accuracy']:.3f})")
        
        return {
            'success': True,
            'report_mode': report_data['report_metadata']['report_mode'],
            'models_evaluated': report_data['report_metadata']['models_evaluated'],
            'created_files': report_data['created_files'],
            'summary': report_data['summary'],
            'insights': report_data['insights'],
            'results_dir': str(results_dir)
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Enhanced report generation failed: {str(e)}")
        
        return {
            'success': False,
            'error': str(e),
            'models_dir': str(models_dir) if models_dir else None,
            'test_data': str(test_data) if test_data else None,
            'results_dir': str(results_dir) if results_dir else None
        }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Report Generation for Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🆕 ENHANCED FEATURES:
- Adaptive reporting: works with any combination of available models
- Intelligent data discovery: automatically finds test data and models
- Graceful degradation: creates meaningful reports even with missing components
- Universal CSV support: works with inference.csv, test.csv, or any processed data

Examples:
  python scripts/report.py                                       # Auto-detect everything
  python scripts/report.py --auto-default                        # Auto-detect latest session
  python scripts/report.py --models-dir results/models           # Specific models directory
  python scripts/report.py --create-inference-report             # Inference-only report
        """
    )
    
    # Optional arguments with auto-detection
    parser.add_argument("--models-dir", type=str, default=None,
                       help="Directory containing trained models (auto-detect if not provided)")
    parser.add_argument("--test-data", type=str, default=None,
                       help="Path to test data (auto-detect if not provided)")
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Directory to save reports (auto-detect if not provided)")
    
    # Mode options
    parser.add_argument("--auto-default", action="store_true",
                       help="Auto-detect paths from latest session")
    parser.add_argument("--create-inference-report", action="store_true",
                       help="Create inference-only report (no model evaluation)")
    
    # Output control
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Directory for log files (default: results-dir/logs)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress output except errors")
    
    return parser.parse_args()

def main():
    """Enhanced main function with auto-detection and graceful handling"""
    args = parse_arguments()
    
    # Setup logging
    if args.results_dir:
        log_dir = args.log_dir if args.log_dir else Path(args.results_dir) / "logs"
    else:
        log_dir = args.log_dir if args.log_dir else Path("logs")
    
    logger = setup_logging(log_dir)
    
    # Configure logging level
    if args.quiet:
        logger.setLevel(logging.WARNING)
    
    try:
        # Handle auto-default mode
        if args.auto_default:
            logger.info("🔍 Auto-detecting latest session...")
            result = generate_enhanced_report(logger=logger)
        else:
            # Use specified paths
            models_dir = Path(args.models_dir) if args.models_dir else None
            test_data = Path(args.test_data) if args.test_data else None
            results_dir = Path(args.results_dir) if args.results_dir else None
            
            result = generate_enhanced_report(
                models_dir=models_dir,
                test_data=test_data,
                results_dir=results_dir,
                logger=logger
            )
        
        if result['success']:
            logger.info("✅ Enhanced report generation completed successfully!")
            return 0
        else:
            logger.error(f"❌ Enhanced report generation failed: {result.get('error', 'Unknown error')}")
            return 1
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())