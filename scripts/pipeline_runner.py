#!/usr/bin/env python3
"""
Pipeline Runner - Advanced Sentiment Analysis Orchestration - FIXED VERSION
Complete pipeline orchestration for automated sentiment analysis with GUI integration.

🔧 FIXES APPLIED:
- ✅ Corrected train_mlp.py parameters and paths
- ✅ Corrected train_svm.py parameters and paths  
- ✅ Corrected report.py parameters and paths
- ✅ Added proper directory structure creation
- ✅ Enhanced logging for debugging parameter issues

FEATURES:
- run_full_pipeline(): Complete automated pipeline (CSV → preprocessing → embedding → training → prediction → report)
- GUI integration with timestamped result organization
- Dynamic path detection and robust error handling
- Integration with enhanced_utils_unified.py auto_embed_and_predict()
- Intelligent insights generation and comprehensive reporting
- Compatible with all existing scripts (embed_dataset.py, train_mlp.py, train_svm.py, report.py)
- Command-line interface with multiple execution modes
- Real-time progress tracking and logging

USAGE:
    # Complete automated pipeline
    python scripts/pipeline_runner.py --action full-auto --file dataset.csv
    
    # Traditional pipeline
    python scripts/pipeline_runner.py --action full
    
    # Specific operations
    python scripts/pipeline_runner.py --action check-data
    python scripts/pipeline_runner.py --action embeddings --force-embeddings
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import torch
from pathlib import Path
from datetime import datetime
import logging
import subprocess
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# FIXED: Dynamic project root detection using standardized approach
try:
    CURRENT_FILE = Path(__file__).resolve()
    if CURRENT_FILE.parent.name == 'scripts':
        PROJECT_ROOT = CURRENT_FILE.parent.parent
    else:
        PROJECT_ROOT = CURRENT_FILE.parent
except:
    PROJECT_ROOT = Path.cwd()

# Add scripts to path for imports
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# FIXED: Create logs directory before logging setup
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging with UTF-8 encoding for emoji support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """
    Advanced Pipeline Runner for sentiment analysis with complete automation support.
    Integrates with enhanced_utils_unified.py and provides GUI-compatible orchestration.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize the advanced pipeline runner.
        
        Args:
            project_root: Root directory of the project (auto-detect if None)
        """
        if project_root is None:
            self.project_root = PROJECT_ROOT
        else:
            self.project_root = Path(project_root)
        
        self.setup_paths()
        self.ensure_directories()
        
        logger.info(f"🚀 Advanced Pipeline Runner initialized")
        logger.info(f"📁 Project root: {self.project_root.absolute()}")
        logger.info(f"🔧 Scripts directory: {SCRIPTS_DIR}")
    
    def setup_paths(self):
        """Setup comprehensive path structure."""
        self.paths = {
            # Data paths
            'raw_data': self.project_root / 'data' / 'raw',
            'processed_data': self.project_root / 'data' / 'processed',
            'embeddings_data': self.project_root / 'data' / 'embeddings',
            
            # Results paths
            'results_dir': self.project_root / 'results',
            'models_dir': self.project_root / 'results' / 'models',
            'reports_dir': self.project_root / 'results' / 'reports',
            'plots_dir': self.project_root / 'results' / 'plots',
            
            # Scripts path
            'scripts_dir': SCRIPTS_DIR,
            
            # Specific CSV files
            'train_csv': self.project_root / 'data' / 'processed' / 'train.csv',
            'val_csv': self.project_root / 'data' / 'processed' / 'val.csv',
            'test_csv': self.project_root / 'data' / 'processed' / 'test.csv',
            
            # Embedding files
            'X_train': self.project_root / 'data' / 'embeddings' / 'X_train.npy',
            'y_train': self.project_root / 'data' / 'embeddings' / 'y_train.npy',
            'X_val': self.project_root / 'data' / 'embeddings' / 'X_val.npy',
            'y_val': self.project_root / 'data' / 'embeddings' / 'y_val.npy',
            'X_test': self.project_root / 'data' / 'embeddings' / 'X_test.npy',
            'y_test': self.project_root / 'data' / 'embeddings' / 'y_test.npy',
            
            # Model files
            'mlp_model': self.project_root / 'results' / 'models' / 'mlp_model.pth',
            'svm_model': self.project_root / 'results' / 'models' / 'svm_model.pkl',
            'embedding_metadata': self.project_root / 'data' / 'embeddings' / 'embedding_metadata.json',
            
            # Config and logs
            'config': self.project_root / 'config.yaml',
            'logs_dir': LOGS_DIR
        }
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.paths['raw_data'],
            self.paths['processed_data'],
            self.paths['embeddings_data'],
            self.paths['results_dir'],
            self.paths['models_dir'],
            self.paths['reports_dir'],
            self.paths['plots_dir'],
            self.paths['logs_dir']
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from config.yaml or return defaults."""
        try:
            import yaml
            with open(self.paths['config'], 'r') as f:
                config = yaml.safe_load(f)
            logger.info("✅ Configuration loaded from config.yaml")
            return config
        except Exception as e:
            logger.warning(f"⚠️ Config loading error: {e}. Using defaults.")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration optimized for fast training."""
        return {
            'mlp': {
                'input_dim': 384,
                'hidden_dims': [512, 256, 128, 64],
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 20,
                'early_stopping_patience': 5,
                'fast_mode': True
            },
            'svm': {
                'C': 1.0,
                'class_weight': 'balanced',
                'max_iter': 10000,
                'fast_mode': True
            },
            'embeddings': {
                'model_name': 'all-MiniLM-L6-v2',
                'batch_size': 32,
                'max_length': 512
            },
            'pipeline': {
                'fast_mode_default': True,
                'save_intermediate_results': True,
                'generate_reports': True
            }
        }
    
    def check_data_availability(self) -> Dict[str, bool]:
        """Check comprehensive data availability status."""
        availability = {
            'csv_files': {
                'train': self.paths['train_csv'].exists(),
                'val': self.paths['val_csv'].exists(),
                'test': self.paths['test_csv'].exists()
            },
            'embedding_files': {
                'X_train': self.paths['X_train'].exists(),
                'y_train': self.paths['y_train'].exists(),
                'X_val': self.paths['X_val'].exists(),
                'y_val': self.paths['y_val'].exists(),
                'X_test': self.paths['X_test'].exists(),
                'y_test': self.paths['y_test'].exists()
            },
            'models': {
                'mlp': self.paths['mlp_model'].exists(),
                'svm': self.paths['svm_model'].exists()
            },
            'metadata': {
                'embedding_metadata': self.paths['embedding_metadata'].exists()
            }
        }
        
        # Add summary flags
        availability['all_csv'] = all(availability['csv_files'].values())
        availability['all_embeddings'] = all(availability['embedding_files'].values())
        availability['any_models'] = any(availability['models'].values())
        availability['all_models'] = all(availability['models'].values())
        
        return availability
    
    def run_subprocess_step(self, script_name: str, args: List[str], 
                           description: str, timeout: int = 600) -> Tuple[bool, str, str]:
        """
        Run a subprocess step with comprehensive error handling.
        
        Args:
            script_name: Name of the script to run
            args: Arguments for the script
            description: Human-readable description
            timeout: Timeout in seconds
        
        Returns:
            Tuple of (success, stdout, stderr)
        """
        script_path = self.paths['scripts_dir'] / script_name
        if not script_path.exists():
            return False, "", f"Script not found: {script_path}"
        
        cmd = [sys.executable, str(script_path)] + args
        
        logger.info(f"🔄 {description}")
        logger.info(f"   Command: {' '.join(cmd)}")
        logger.info(f"   Working directory: {self.project_root}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {description} - SUCCESS")
                return True, result.stdout, result.stderr
            else:
                logger.error(f"❌ {description} - FAILED")
                logger.error(f"   Return code: {result.returncode}")
                logger.error(f"   STDOUT: {result.stdout[:500]}...")
                logger.error(f"   STDERR: {result.stderr[:500]}...")
                return False, result.stdout, result.stderr
                
        except subprocess.TimeoutExpired:
            error_msg = f"{description} timed out after {timeout} seconds"
            logger.error(f"⏰ {error_msg}")
            return False, "", error_msg
        except Exception as e:
            error_msg = f"{description} error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, "", error_msg
    
    def run_full_pipeline(self, input_csv_path: str, fast_mode: bool = True) -> Dict[str, Any]:
        """
        Complete automated pipeline using enhanced_utils_unified.py auto_embed_and_predict()
        
        Args:
            input_csv_path: Path to the CSV file to analyze
            fast_mode: Whether to use fast training modes
        
        Returns:
            Comprehensive pipeline results dictionary
        """
        logger.info(f"🚀 Starting complete automated pipeline for: {input_csv_path}")
        
        pipeline_start = datetime.now()
        
        # Initialize results
        results = {
            'pipeline_type': 'full_automated',
            'input_file': input_csv_path,
            'start_time': pipeline_start.isoformat(),
            'fast_mode': fast_mode,
            'project_root': str(self.project_root),
            'success': False,
            'steps': {},
            'final_results': {}
        }
        
        try:
            # Step 1: Import and use enhanced_utils auto_embed_and_predict
            logger.info("📦 Loading enhanced utilities...")
            
            try:
                from scripts.enhanced_utils_unified import auto_embed_and_predict
                results['steps']['enhanced_utils_loaded'] = {'success': True}
            except ImportError as e:
                error_msg = f"Failed to import enhanced_utils_unified: {e}"
                logger.error(f"❌ {error_msg}")
                results['steps']['enhanced_utils_loaded'] = {'success': False, 'error': error_msg}
                results['error'] = error_msg
                return results
            
            # Step 2: Execute complete automated pipeline
            logger.info("🔄 Executing complete automated pipeline...")
            
            auto_results = auto_embed_and_predict(
                file_path=input_csv_path,
                fast_mode=fast_mode,
                save_intermediate=True
            )
            
            results['steps']['auto_pipeline'] = {
                'success': auto_results.get('overall_success', False),
                'details': auto_results
            }
            
            if not auto_results.get('overall_success', False):
                error_msg = auto_results.get('error', 'Auto pipeline failed')
                logger.error(f"❌ Automated pipeline failed: {error_msg}")
                results['error'] = error_msg
                results['session_directory'] = auto_results.get('session_directory')
                return results
            
            # Step 3: Extract and organize results for GUI
            logger.info("📊 Organizing results for GUI integration...")
            
            session_dir = auto_results.get('session_directory')
            results['session_directory'] = session_dir
            
            # Organize final results
            results['final_results'] = {
                'predictions': auto_results.get('gui_predictions', {}),
                'metrics': auto_results.get('gui_metrics', {}),
                'insights': auto_results.get('insights', []),
                'test_data_info': auto_results.get('test_data', {}),
                'session_directory': session_dir,
                'pipeline_steps': auto_results.get('steps', {})
            }
            
            # Step 4: Generate additional analysis if needed
            if session_dir and Path(session_dir).exists():
                logger.info("📈 Generating additional analysis files...")
                
                # Create summary report
                self.create_pipeline_summary_report(results, session_dir)
                
                # Log success metrics
                if results['final_results']['metrics']:
                    for model_name, metrics in results['final_results']['metrics'].items():
                        accuracy = metrics.get('accuracy', 0)
                        f1 = metrics.get('f1_score', 0)
                        logger.info(f"📊 {model_name.upper()}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
            
            # Step 5: Final success assessment
            pipeline_end = datetime.now()
            duration = (pipeline_end - pipeline_start).total_seconds()
            
            results['end_time'] = pipeline_end.isoformat()
            results['duration_seconds'] = duration
            results['success'] = True
            
            logger.info(f"🎉 Complete automated pipeline SUCCESS!")
            logger.info(f"   Duration: {duration:.1f} seconds")
            logger.info(f"   Results saved to: {session_dir}")
            
            # Count insights
            insights_count = len(results['final_results'].get('insights', []))
            logger.info(f"   Generated {insights_count} intelligent insights")
            
            # Count predictions
            total_predictions = 0
            for model_name, preds in results['final_results'].get('predictions', {}).items():
                if isinstance(preds, list):
                    total_predictions += len(preds)
            logger.info(f"   Total predictions: {total_predictions}")
            
            return results
            
        except Exception as e:
            pipeline_end = datetime.now()
            duration = (pipeline_end - pipeline_start).total_seconds()
            
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            results['end_time'] = pipeline_end.isoformat()
            results['duration_seconds'] = duration
            results['success'] = False
            results['error'] = error_msg
            
            return results
    
    def create_pipeline_summary_report(self, results: Dict[str, Any], session_dir: str):
        """Create a comprehensive summary report of the pipeline execution."""
        try:
            session_path = Path(session_dir)
            summary_path = session_path / "pipeline_summary.txt"
            
            summary_lines = [
                "SENTIMENT ANALYSIS PIPELINE SUMMARY",
                "=" * 50,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Input File: {results.get('input_file', 'Unknown')}",
                f"Pipeline Type: {results.get('pipeline_type', 'Unknown')}",
                f"Fast Mode: {results.get('fast_mode', False)}",
                f"Duration: {results.get('duration_seconds', 0):.1f} seconds",
                f"Overall Success: {results.get('success', False)}",
                "",
                "EXECUTION STEPS:",
                "-" * 20
            ]
            
            # Add step details
            for step_name, step_info in results.get('steps', {}).items():
                status = "✅ SUCCESS" if step_info.get('success', False) else "❌ FAILED"
                summary_lines.append(f"{step_name}: {status}")
                
                if not step_info.get('success', False) and 'error' in step_info:
                    summary_lines.append(f"  Error: {step_info['error']}")
            
            # Add final results summary
            final_results = results.get('final_results', {})
            
            summary_lines.extend([
                "",
                "RESULTS SUMMARY:",
                "-" * 20
            ])
            
            # Model predictions
            predictions = final_results.get('predictions', {})
            if predictions:
                summary_lines.append(f"Models with predictions: {len(predictions)}")
                for model_name, preds in predictions.items():
                    if isinstance(preds, list):
                        pred_count = len(preds)
                        positive_count = sum(1 for p in preds if p == 1)
                        summary_lines.append(f"  {model_name.upper()}: {pred_count} predictions, {positive_count} positive ({positive_count/pred_count*100:.1f}%)")
            
            # Model metrics
            metrics = final_results.get('metrics', {})
            if metrics:
                summary_lines.append("")
                summary_lines.append("Model Performance:")
                for model_name, model_metrics in metrics.items():
                    accuracy = model_metrics.get('accuracy', 0)
                    f1 = model_metrics.get('f1_score', 0)
                    summary_lines.append(f"  {model_name.upper()}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
            
            # Insights summary
            insights = final_results.get('insights', [])
            if insights:
                summary_lines.extend([
                    "",
                    f"INTELLIGENT INSIGHTS ({len(insights)} generated):",
                    "-" * 20
                ])
                for i, insight in enumerate(insights[:10], 1):  # Show first 10
                    summary_lines.append(f"{i}. {insight}")
                
                if len(insights) > 10:
                    summary_lines.append(f"... and {len(insights) - 10} more insights")
            
            # Session info
            summary_lines.extend([
                "",
                "SESSION INFORMATION:",
                "-" * 20,
                f"Session Directory: {session_dir}",
                f"Project Root: {results.get('project_root', 'Unknown')}",
                "",
                "=" * 50
            ])
            
            # Write summary
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_lines))
            
            logger.info(f"📄 Pipeline summary saved: {summary_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create pipeline summary: {e}")
    
    def run_traditional_pipeline(self, force_regenerate_embeddings: bool = False) -> Dict[str, Any]:
        """
        Execute traditional pipeline with individual script calls - FIXED VERSION.
        
        Args:
            force_regenerate_embeddings: Force regeneration of embeddings
        
        Returns:
            Pipeline execution results
        """
        logger.info("🔄 Starting traditional pipeline execution...")
        
        pipeline_start = datetime.now()
        results = {
            'pipeline_type': 'traditional',
            'start_time': pipeline_start.isoformat(),
            'steps': {},
            'success': False
        }
        
        try:
            # Step 1: Check and prepare data
            logger.info("Step 1: Data preparation check...")
            if not self.check_and_prepare_data():
                raise Exception("Data preparation failed")
            results['steps']['data_preparation'] = {'success': True}
            
            # Step 2: Generate embeddings - FIXED PARAMETERS
            logger.info("Step 2: Embedding generation...")
            availability = self.check_data_availability()
            
            if force_regenerate_embeddings or not availability['all_embeddings']:
                embed_args = [
                    "--input-dir", str(self.paths['processed_data']),
                    "--output-dir", str(self.paths['embeddings_data'])
                ]
                if force_regenerate_embeddings:
                    embed_args.append("--force-recreate")
                
                success, stdout, stderr = self.run_subprocess_step(
                    'embed_dataset.py',
                    embed_args,
                    'Embedding generation'
                )
                results['steps']['embeddings'] = {
                    'success': success,
                    'stdout': stdout[:500] if stdout else '',
                    'stderr': stderr[:500] if stderr else ''
                }
                
                if not success:
                    raise Exception(f"Embedding generation failed: {stderr}")
            else:
                results['steps']['embeddings'] = {'success': True, 'note': 'Embeddings already available'}
            
            # Step 3: Train MLP - FIXED PARAMETERS
            logger.info("Step 3: MLP training...")
            mlp_args = [
                "--embeddings-dir", str(self.paths['embeddings_data']),
                "--output-dir", str(self.paths['results_dir']),
                "--epochs", "20",
                "--lr", "0.001", 
                "--batch-size", "32"
            ]
            
            success, stdout, stderr = self.run_subprocess_step(
                'train_mlp.py',
                mlp_args,
                'MLP training'
            )
            results['steps']['mlp_training'] = {
                'success': success,
                'stdout': stdout[:500] if stdout else '',
                'stderr': stderr[:500] if stderr else ''
            }
            
            # Step 4: Train SVM - FIXED PARAMETERS  
            logger.info("Step 4: SVM training...")
            svm_args = [
                "--embeddings-dir", str(self.paths['embeddings_data']),
                "--output-dir", str(self.paths['results_dir']),
                "--fast"  # Use fast mode by default
            ]
            
            success, stdout, stderr = self.run_subprocess_step(
                'train_svm.py',
                svm_args,
                'SVM training'
            )
            results['steps']['svm_training'] = {
                'success': success,
                'stdout': stdout[:500] if stdout else '',
                'stderr': stderr[:500] if stderr else ''
            }
            
            # Step 5: Generate reports - FIXED PARAMETERS
            logger.info("Step 5: Report generation...")
            report_args = [
                "--models-dir", str(self.paths['models_dir']),
                "--test-data", str(self.paths['test_csv']),
                "--results-dir", str(self.paths['results_dir'])
            ]
            
            success, stdout, stderr = self.run_subprocess_step(
                'report.py',
                report_args,
                'Report generation'
            )
            results['steps']['report_generation'] = {
                'success': success,
                'stdout': stdout[:500] if stdout else '',
                'stderr': stderr[:500] if stderr else ''
            }
            
            # Final assessment
            critical_steps = ['data_preparation', 'embeddings']
            success_count = sum(1 for step in results['steps'].values() if step['success'])
            total_steps = len(results['steps'])
            
            results['success'] = all(results['steps'][step]['success'] for step in critical_steps)
            results['success_rate'] = success_count / total_steps
            
            pipeline_end = datetime.now()
            results['end_time'] = pipeline_end.isoformat()
            results['duration_seconds'] = (pipeline_end - pipeline_start).total_seconds()
            
            if results['success']:
                logger.info(f"✅ Traditional pipeline completed successfully!")
            else:
                logger.warning(f"⚠️ Traditional pipeline completed with warnings")
            
            logger.info(f"   Success rate: {success_count}/{total_steps} steps")
            logger.info(f"   Duration: {results['duration_seconds']:.1f} seconds")
            
            return results
            
        except Exception as e:
            pipeline_end = datetime.now()
            error_msg = f"Traditional pipeline failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            results['end_time'] = pipeline_end.isoformat()
            results['duration_seconds'] = (pipeline_end - pipeline_start).total_seconds()
            results['success'] = False
            results['error'] = error_msg
            
            return results
    
    def check_and_prepare_data(self) -> bool:
        """
        Check and prepare data for pipeline execution.
        Enhanced version with better error reporting.
        """
        logger.info("🔍 Comprehensive data availability check...")
        
        # Check CSV files
        csv_files = {
            'train': self.paths['train_csv'],
            'val': self.paths['val_csv'],
            'test': self.paths['test_csv']
        }
        
        logger.info("📄 CSV Files Status:")
        missing_csv = []
        for name, path in csv_files.items():
            exists = path.exists()
            status = "✅" if exists else "❌"
            logger.info(f"   {status} {name}.csv: {path}")
            if not exists:
                missing_csv.append(name)
        
        if missing_csv:
            logger.warning(f"❌ Missing CSV files: {missing_csv}")
            logger.info("🔄 Attempting automatic data preparation...")
            
            # Check for raw data
            raw_paths = [
                self.project_root / 'data' / 'raw' / 'imdb_raw.csv',
                self.project_root / 'data' / 'imdb_dataset.csv',
                self.project_root / 'imdb_dataset.csv'
            ]
            
            raw_file_found = None
            for raw_path in raw_paths:
                if raw_path.exists():
                    raw_file_found = raw_path
                    logger.info(f"✅ Found raw data: {raw_path}")
                    break
            
            if raw_file_found:
                # Try preprocessing
                try:
                    preprocess_args = [
                        "--input", str(raw_file_found),
                        "--output-dir", str(self.paths['processed_data'])
                    ]
                    
                    success, stdout, stderr = self.run_subprocess_step(
                        'preprocess.py',
                        preprocess_args,
                        'Data preprocessing',
                        timeout=300
                    )
                    
                    if success:
                        # Re-check CSV files
                        still_missing = [name for name, path in csv_files.items() if not path.exists()]
                        if not still_missing:
                            logger.info("✅ Data preparation completed successfully")
                            return True
                        else:
                            logger.error(f"❌ Still missing after preprocessing: {still_missing}")
                    else:
                        logger.error(f"❌ Preprocessing failed: {stderr}")
                        
                except Exception as e:
                    logger.error(f"❌ Preprocessing error: {e}")
            else:
                logger.error("❌ No raw data file found")
                logger.error("💡 SOLUTION STEPS:")
                logger.error("   1. Place your dataset file in one of these locations:")
                for raw_path in raw_paths:
                    logger.error(f"      - {raw_path}")
                logger.error("   2. Or run preprocessing manually:")
                logger.error("      python scripts/preprocess.py --input your_dataset.csv")
            
            return False
        else:
            logger.info("✅ All required CSV files are available")
            return True
    
    # EXISTING METHODS (maintained for compatibility)
    
    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train, validation, and test datasets."""
        logger.info("📊 Loading datasets...")
        
        try:
            train_df = pd.read_csv(self.paths['train_csv'])
            val_df = pd.read_csv(self.paths['val_csv'])
            test_df = pd.read_csv(self.paths['test_csv'])
            
            logger.info(f"   Train: {len(train_df):,} samples")
            logger.info(f"   Val: {len(val_df):,} samples")
            logger.info(f"   Test: {len(test_df):,} samples")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"❌ Dataset loading error: {e}")
            raise
    
    def analyze_csv_dataset(self, csv_path: str, text_column: str = 'text', 
                          label_column: str = 'label', predictor=None) -> Dict[str, Any]:
        """
        Analyze a CSV dataset with enhanced capabilities.
        
        Args:
            csv_path: Path to CSV file
            text_column: Name of text column
            label_column: Name of label column
            predictor: Optional predictor for sample predictions
        
        Returns:
            Comprehensive analysis results
        """
        logger.info(f"📊 Analyzing CSV dataset: {csv_path}")
        
        try:
            # Load and validate CSV
            if not Path(csv_path).exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            df = pd.read_csv(csv_path)
            
            # Basic file info
            analysis = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'file_info': {
                    'path': csv_path,
                    'filename': Path(csv_path).name,
                    'size_mb': Path(csv_path).stat().st_size / (1024 * 1024),
                    'rows': len(df),
                    'columns': list(df.columns)
                },
                'data_analysis': {
                    'total_samples': len(df),
                    'missing_values': df.isnull().sum().to_dict(),
                    'column_types': df.dtypes.astype(str).to_dict(),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
                }
            }
            
            # Text column analysis
            text_columns = [text_column, 'text', 'review', 'content', 'comment']
            found_text_col = None
            
            for col in text_columns:
                if col in df.columns:
                    found_text_col = col
                    break
            
            if found_text_col:
                text_data = df[found_text_col].fillna('')
                text_lengths = text_data.str.len()
                word_counts = text_data.str.split().str.len()
                
                analysis['text_analysis'] = {
                    'column_used': found_text_col,
                    'avg_char_length': float(text_lengths.mean()),
                    'median_char_length': float(text_lengths.median()),
                    'max_char_length': int(text_lengths.max()),
                    'min_char_length': int(text_lengths.min()),
                    'std_char_length': float(text_lengths.std()),
                    'avg_word_count': float(word_counts.mean()),
                    'median_word_count': float(word_counts.median()),
                    'empty_texts': int((text_lengths == 0).sum()),
                    'very_short_texts': int((text_lengths < 10).sum()),
                    'very_long_texts': int((text_lengths > 1000).sum())
                }
            else:
                analysis['errors'] = [f"No text column found. Expected: {text_columns}"]
            
            # Label column analysis
            label_columns = [label_column, 'label', 'sentiment', 'class', 'target']
            found_label_col = None
            
            for col in label_columns:
                if col in df.columns:
                    found_label_col = col
                    break
            
            if found_label_col:
                label_data = df[found_label_col].dropna()
                label_counts = label_data.value_counts()
                
                analysis['sentiment_analysis'] = {
                    'column_used': found_label_col,
                    'label_distribution': label_counts.to_dict(),
                    'unique_labels': sorted(label_data.unique()),
                    'total_labeled': len(label_data),
                    'missing_labels': len(df) - len(label_data),
                    'most_common_label': label_counts.index[0] if len(label_counts) > 0 else None,
                    'balance_ratio': float(label_counts.min() / label_counts.max()) if len(label_counts) > 1 else 1.0
                }
            else:
                analysis['sentiment_analysis'] = {
                    'has_labels': False,
                    'note': f"No label column found. Expected: {label_columns}"
                }
            
            # Sample data
            analysis['sample_data'] = df.head(5).to_dict('records')
            
            # Generate insights using enhanced_utils if available
            try:
                from scripts.enhanced_utils_unified import generate_insights
                insights = generate_insights(df)
                analysis['insights'] = insights
                logger.info(f"   Generated {len(insights)} insights")
            except ImportError:
                logger.warning("⚠️ Enhanced insights not available (enhanced_utils_unified not found)")
            
            # Predictor-based sample predictions
            if predictor and found_text_col:
                try:
                    sample_texts = df[found_text_col].head(3).tolist()
                    predictions = []
                    
                    for text in sample_texts:
                        if hasattr(predictor, 'predict_single'):
                            pred = predictor.predict_single(str(text))
                            predictions.append(pred)
                        else:
                            # Fallback prediction format
                            predictions.append({'prediction': 0, 'confidence': 0.5})
                    
                    analysis['sample_predictions'] = predictions
                    logger.info(f"   Generated {len(predictions)} sample predictions")
                    
                except Exception as pred_error:
                    analysis['prediction_error'] = str(pred_error)
                    logger.warning(f"⚠️ Sample predictions failed: {pred_error}")
            
            logger.info(f"✅ CSV analysis completed successfully")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ CSV analysis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': csv_path,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        model_info = {
            'mlp': {'exists': False, 'format': None, 'size_mb': 0, 'status': 'missing'},
            'svm': {'exists': False, 'format': None, 'size_mb': 0, 'status': 'missing'},
            'summary': {}
        }
        
        # Check MLP model
        if self.paths['mlp_model'].exists():
            model_info['mlp'] = {
                'exists': True,
                'format': 'pth',
                'size_mb': self.paths['mlp_model'].stat().st_size / (1024 * 1024),
                'status': 'ready',
                'path': str(self.paths['mlp_model']),
                'last_modified': datetime.fromtimestamp(
                    self.paths['mlp_model'].stat().st_mtime
                ).isoformat()
            }
        
        # Check SVM model
        if self.paths['svm_model'].exists():
            model_info['svm'] = {
                'exists': True,
                'format': 'pkl',
                'size_mb': self.paths['svm_model'].stat().st_size / (1024 * 1024),
                'status': 'ready',
                'path': str(self.paths['svm_model']),
                'last_modified': datetime.fromtimestamp(
                    self.paths['svm_model'].stat().st_mtime
                ).isoformat()
            }
        
        # Summary
        model_info['summary'] = {
            'total_models': sum(1 for model in model_info.values() if isinstance(model, dict) and model.get('exists', False)),
            'ready_for_inference': model_info['mlp']['exists'] or model_info['svm']['exists'],
            'both_models_available': model_info['mlp']['exists'] and model_info['svm']['exists']
        }
        
        return model_info
    
    def save_pipeline_report(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save comprehensive pipeline report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'pipeline_report_{timestamp}.json'
        
        report_path = self.paths['results_dir'] / filename
        
        # Add metadata to results
        enhanced_results = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'pipeline_runner_version': '2.0',
                'project_root': str(self.project_root),
                'python_version': sys.version,
                'platform': sys.platform
            },
            'system_info': {
                'data_availability': self.check_data_availability(),
                'model_info': self.get_model_info()
            },
            'pipeline_results': results
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 Pipeline report saved: {report_path}")
        return str(report_path)


# WRAPPER FUNCTIONS FOR GUI INTEGRATION

def run_dataset_analysis(csv_path: str) -> Dict[str, Any]:
    """
    Wrapper function for GUI Streamlit integration.
    Provides comprehensive CSV dataset analysis.
    """
    try:
        if not Path(csv_path).exists():
            return {
                'error': f"CSV file not found: {csv_path}",
                'success': False
            }
        
        # Initialize pipeline runner
        runner = PipelineRunner()
        
        # Initialize internal predictor for sample predictions
        predictor = None
        try:
            from scripts.enhanced_utils_unified import SentimentPredictor
            predictor = SentimentPredictor()
            logger.info("✅ Predictor initialized for analysis")
        except Exception as pred_error:
            logger.warning(f"⚠️ Predictor initialization failed: {pred_error}")
        
        # Execute analysis
        analysis_results = runner.analyze_csv_dataset(
            csv_path=csv_path,
            predictor=predictor
        )
        
        # Add system information
        analysis_results['system_info'] = {
            'model_info': runner.get_model_info(),
            'data_availability': runner.check_data_availability(),
            'project_root': str(runner.project_root)
        }
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"❌ Dataset analysis wrapper error: {e}")
        return {
            'error': f"Analysis error: {str(e)}",
            'success': False,
            'timestamp': datetime.now().isoformat()
        }

def run_complete_csv_analysis(csv_path: str, text_column: str = 'text', 
                            label_column: str = 'label') -> Dict[str, Any]:
    """
    Complete CSV analysis wrapper for command line and GUI usage.
    Uses the enhanced_utils_unified auto_embed_and_predict pipeline.
    """
    try:
        if not Path(csv_path).exists():
            return {
                'success': False,
                'error': f"CSV file not found: {csv_path}"
            }
        
        # Initialize pipeline runner
        runner = PipelineRunner()
        
        # Use the full automated pipeline
        result = runner.run_full_pipeline(csv_path, fast_mode=True)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Complete CSV analysis error: {e}")
        return {
            'success': False,
            'error': f"Complete analysis error: {str(e)}",
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Enhanced main function with complete pipeline automation support."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced Sentiment Analysis Pipeline Runner - FIXED VERSION",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete automated pipeline
  python scripts/pipeline_runner.py --action full-auto --file dataset.csv
  
  # Traditional pipeline
  python scripts/pipeline_runner.py --action full
  
  # Check system status
  python scripts/pipeline_runner.py --action check-data
  
  # Quick CSV analysis
  python scripts/pipeline_runner.py --action analyze --csv-path dataset.csv
        """
    )
    
    parser.add_argument(
        "--action", 
        choices=['full', 'full-auto', 'embeddings', 'train-mlp', 'train-svm', 'analyze', 'check-data'],
        default='full',
        help="Action to execute"
    )
    parser.add_argument("--csv-path", type=str, help="CSV path for analysis")
    parser.add_argument("--file", type=str, help="CSV file for complete automated analysis")
    parser.add_argument("--text-column", type=str, default="text", help="Text column name")
    parser.add_argument("--label-column", type=str, default="label", help="Label column name")
    parser.add_argument("--force-embeddings", action="store_true", help="Force regenerate embeddings")
    parser.add_argument("--fast-mode", action="store_true", default=True, help="Use fast training modes")
    parser.add_argument("--save-report", action="store_true", default=True, help="Save execution report")
    
    args = parser.parse_args()
    
    # Initialize pipeline runner
    runner = PipelineRunner()
    
    # Execute based on action
    if args.action == 'check-data':
        print("🔍 Comprehensive system check...")
        
        # Data availability
        availability = runner.check_data_availability()
        
        print("\n📊 DATA AVAILABILITY STATUS:")
        print(f"✅ CSV Files: {availability['all_csv']}")
        for file, exists in availability['csv_files'].items():
            status = "✅" if exists else "❌"
            print(f"   {status} {file}.csv")
        
        print(f"\n✅ Embeddings: {availability['all_embeddings']}")
        for file, exists in availability['embedding_files'].items():
            status = "✅" if exists else "❌"
            print(f"   {status} {file}")
        
        print(f"\n✅ Models: {availability['any_models']}")
        for model, exists in availability['models'].items():
            status = "✅" if exists else "❌"
            print(f"   {status} {model}")
        
        print(f"\n✅ Metadata: {availability['metadata']['embedding_metadata']}")
        
        # Model information
        model_info = runner.get_model_info()
        print(f"\n🤖 MODEL DETAILS:")
        for model_name, info in model_info.items():
            if model_name != 'summary' and isinstance(info, dict):
                if info['exists']:
                    print(f"   ✅ {model_name.upper()}: {info['size_mb']:.1f}MB ({info['format']})")
                else:
                    print(f"   ❌ {model_name.upper()}: Not available")
        
        # System paths
        print(f"\n📁 SYSTEM PATHS:")
        print(f"   Project Root: {runner.project_root}")
        print(f"   Data Dir: {runner.paths['processed_data']}")
        print(f"   Models Dir: {runner.paths['models_dir']}")
        print(f"   Results Dir: {runner.paths['results_dir']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if not availability['all_csv']:
            print("   🔧 Run: python scripts/preprocess.py")
        if not availability['all_embeddings']:
            print("   🔧 Run: python scripts/pipeline_runner.py --action embeddings")
        if not availability['any_models']:
            print("   🔧 Run: python scripts/pipeline_runner.py --action full")
        if availability['all_csv'] and availability['all_embeddings'] and availability['any_models']:
            print("   🎉 System ready for analysis!")
    
    elif args.action == 'full-auto':
        if not args.file:
            print("❌ Error: --file required for automated pipeline")
            print("💡 Example: python scripts/pipeline_runner.py --action full-auto --file dataset.csv")
            return
        
        print(f"🚀 Starting complete automated pipeline for: {args.file}")
        results = runner.run_full_pipeline(args.file, fast_mode=args.fast_mode)
        
        if results['success']:
            print("✅ AUTOMATED PIPELINE SUCCESS!")
            print(f"   Duration: {results['duration_seconds']:.1f} seconds")
            print(f"   Session: {results['session_directory']}")
            
            # Show key results
            final_results = results.get('final_results', {})
            if final_results.get('metrics'):
                print("   📊 Model Performance:")
                for model_name, metrics in final_results['metrics'].items():
                    acc = metrics.get('accuracy', 0)
                    f1 = metrics.get('f1_score', 0)
                    print(f"      {model_name.upper()}: Accuracy={acc:.3f}, F1={f1:.3f}")
            
            insights_count = len(final_results.get('insights', []))
            print(f"   🧠 Generated {insights_count} intelligent insights")
            
        else:
            print(f"❌ AUTOMATED PIPELINE FAILED: {results.get('error', 'Unknown error')}")
        
        if args.save_report:
            report_path = runner.save_pipeline_report(results)
            print(f"📄 Report saved: {report_path}")
    
    elif args.action == 'full':
        print("🔄 Starting traditional pipeline...")
        results = runner.run_traditional_pipeline(args.force_embeddings)
        
        if results['success']:
            print("✅ TRADITIONAL PIPELINE SUCCESS!")
        else:
            print(f"❌ TRADITIONAL PIPELINE FAILED: {results.get('error', 'Unknown error')}")
        
        print(f"   Duration: {results['duration_seconds']:.1f} seconds")
        print(f"   Success rate: {results.get('success_rate', 0)*100:.1f}%")
        
        if args.save_report:
            report_path = runner.save_pipeline_report(results)
            print(f"📄 Report saved: {report_path}")
    
    elif args.action == 'embeddings':
        print("🔄 Generating embeddings...")
        embed_args = [
            "--input-dir", str(runner.paths['processed_data']),
            "--output-dir", str(runner.paths['embeddings_data'])
        ]
        if args.force_embeddings:
            embed_args.append("--force-recreate")
        
        success, stdout, stderr = runner.run_subprocess_step(
            'embed_dataset.py',
            embed_args,
            'Embedding generation'
        )
        print(f"Embedding generation: {'SUCCESS' if success else 'FAILED'}")
        if not success and stderr:
            print(f"Error: {stderr[:200]}...")
    
    elif args.action == 'train-mlp':
        print("🔄 Training MLP model...")
        mlp_args = [
            "--embeddings-dir", str(runner.paths['embeddings_data']),
            "--output-dir", str(runner.paths['results_dir']),
            "--epochs", "20",
            "--lr", "0.001",
            "--batch-size", "32"
        ]
        
        success, stdout, stderr = runner.run_subprocess_step(
            'train_mlp.py',
            mlp_args,
            'MLP training'
        )
        print(f"MLP training: {'SUCCESS' if success else 'FAILED'}")
        if not success and stderr:
            print(f"Error: {stderr[:200]}...")
    
    elif args.action == 'train-svm':
        print("🔄 Training SVM model...")
        svm_args = [
            "--embeddings-dir", str(runner.paths['embeddings_data']),
            "--output-dir", str(runner.paths['results_dir']),
            "--fast"
        ]
        
        success, stdout, stderr = runner.run_subprocess_step(
            'train_svm.py',
            svm_args,
            'SVM training'
        )
        print(f"SVM training: {'SUCCESS' if success else 'FAILED'}")
        if not success and stderr:
            print(f"Error: {stderr[:200]}...")
    
    elif args.action == 'analyze':
        if not args.csv_path:
            print("❌ Error: --csv-path required for analysis")
            return
        
        print(f"📊 Analyzing dataset: {args.csv_path}")
        results = run_dataset_analysis(args.csv_path)
        
        if results.get('success', False):
            print("✅ Analysis completed!")
            print(f"   Samples: {results['data_analysis']['total_samples']:,}")
            
            if 'text_analysis' in results:
                text_info = results['text_analysis']
                print(f"   Avg text length: {text_info['avg_char_length']:.0f} chars")
            
            if 'sentiment_analysis' in results and results['sentiment_analysis'].get('has_labels', True):
                sent_info = results['sentiment_analysis']
                print(f"   Label balance: {sent_info['balance_ratio']:.2f}")
            
            if 'insights' in results:
                print(f"   Generated {len(results['insights'])} insights")
        else:
            print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
    
    # Handle direct file argument (--file without action)
    elif args.file:
        print(f"🔄 Detected CSV file: {args.file}")
        print("Executing complete automated analysis...")
        
        results = runner.run_full_pipeline(args.file, fast_mode=args.fast_mode)
        
        if results['success']:
            print("✅ Analysis completed!")
            print(f"📁 Results: {results['session_directory']}")
        else:
            print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
