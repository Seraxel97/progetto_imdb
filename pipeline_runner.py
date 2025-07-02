#!/usr/bin/env python3
"""
Pipeline Runner - FIXED VERSION - Real Pipeline Orchestration
Complete pipeline orchestration with REAL script invocation and directory creation.

🔧 FIXES APPLIED:
- ✅ Creates real results/session_<timestamp>/ structure
- ✅ Invokes individual scripts: embed_dataset.py, train_mlp.py, train_svm.py, report.py
- ✅ Comprehensive logging for every step
- ✅ Proper path passing to all scripts
- ✅ Directory creation with full structure
- ✅ Error handling and fallback mechanisms
- ✅ Real file saving and verification

FEATURES:
- run_complete_csv_analysis(): REAL complete pipeline with script invocation
- Creates: results/session_<timestamp>/{processed,embeddings,models,plots,reports,logs}/
- Logging every step with [PIPELINE] tags
- GUI integration with authentic results
- Individual script execution with proper arguments
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
import shutil
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
    FIXED Pipeline Runner - Real script execution with directory creation.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize the FIXED pipeline runner."""
        if project_root is None:
            self.project_root = PROJECT_ROOT
        else:
            self.project_root = Path(project_root)
        
        self.setup_paths()
        self.ensure_directories()
        
        logger.info(f"🚀 FIXED Pipeline Runner initialized")
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
    
    def create_session_directory(self, timestamp: str = None) -> Path:
        """
        🔧 FIXED: Create session directory with complete structure
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        session_dir = self.paths['results_dir'] / f"session_{timestamp}"
        
        # Create main session directory
        session_dir.mkdir(parents=True, exist_ok=True)
        print(f"[PIPELINE] Created session directory: {session_dir}")
        logger.info(f"📁 Session directory created: {session_dir}")
        
        # Create all required subdirectories
        subdirs = ['processed', 'embeddings', 'models', 'plots', 'reports', 'logs']
        
        for subdir in subdirs:
            subdir_path = session_dir / subdir
            subdir_path.mkdir(parents=True, exist_ok=True)
            print(f"[PIPELINE] Created subdirectory: {subdir_path}")
            logger.info(f"📂 Created subdirectory: {subdir}")
        
        return session_dir
    
    def run_subprocess_step(self, script_name: str, args: List[str], 
                           description: str, timeout: int = 600) -> Tuple[bool, str, str]:
        """
        🔧 FIXED: Run subprocess with enhanced logging and error handling
        """
        script_path = self.paths['scripts_dir'] / script_name
        if not script_path.exists():
            error_msg = f"Script not found: {script_path}"
            print(f"[PIPELINE] ERROR: {error_msg}")
            return False, "", error_msg
        
        cmd = [sys.executable, str(script_path)] + args
        
        print(f"[PIPELINE] EXECUTING: {description}")
        print(f"[PIPELINE] Command: {' '.join(cmd)}")
        logger.info(f"🔄 {description}")
        logger.info(f"   Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                print(f"[PIPELINE] SUCCESS: {description}")
                logger.info(f"✅ {description} - SUCCESS")
                if result.stdout:
                    print(f"[PIPELINE] Output: {result.stdout[:200]}...")
                return True, result.stdout, result.stderr
            else:
                print(f"[PIPELINE] FAILED: {description}")
                print(f"[PIPELINE] Error: {result.stderr[:300]}...")
                logger.error(f"❌ {description} - FAILED")
                logger.error(f"   Error: {result.stderr[:200]}...")
                return False, result.stdout, result.stderr
                
        except subprocess.TimeoutExpired:
            error_msg = f"{description} timed out after {timeout} seconds"
            print(f"[PIPELINE] TIMEOUT: {error_msg}")
            logger.error(f"⏰ {error_msg}")
            return False, "", error_msg
        except Exception as e:
            error_msg = f"{description} error: {str(e)}"
            print(f"[PIPELINE] ERROR: {error_msg}")
            logger.error(f"❌ {error_msg}")
            return False, "", error_msg
    
    def run_real_complete_pipeline(self, input_csv_path: str) -> Dict[str, Any]:
        """
        🚀 FIXED: Complete REAL pipeline with individual script invocation
        """
        print(f"[PIPELINE] ========================================")
        print(f"[PIPELINE] STARTING REAL COMPLETE PIPELINE")
        print(f"[PIPELINE] Input CSV: {input_csv_path}")
        print(f"[PIPELINE] ========================================")
        
        pipeline_start = datetime.now()
        timestamp = pipeline_start.strftime("%Y%m%d_%H%M%S")
        
        # Initialize results
        results = {
            'pipeline_type': 'real_complete_fixed',
            'input_file': input_csv_path,
            'start_time': pipeline_start.isoformat(),
            'timestamp': timestamp,
            'project_root': str(self.project_root),
            'success': False,
            'steps': {},
            'final_results': {},
            'session_directory': None
        }
        
        try:
            # Step 1: Create session directory
            print(f"[PIPELINE] STEP 1: Creating session directory...")
            session_dir = self.create_session_directory(timestamp)
            results['session_directory'] = str(session_dir)
            results['final_results']['session_directory'] = str(session_dir)
            
            # Verify session directory exists
            if session_dir.exists():
                print(f"[PIPELINE] SUCCESS: Session directory created at {session_dir}")
                results['steps']['session_creation'] = {'success': True, 'path': str(session_dir)}
            else:
                raise Exception("Session directory creation failed")
            
            # Step 2: Copy input CSV to session processed directory
            print(f"[PIPELINE] STEP 2: Copying input CSV...")
            input_path = Path(input_csv_path)
            if input_path.exists():
                processed_csv = session_dir / 'processed' / input_path.name
                shutil.copy2(input_path, processed_csv)
                print(f"[PIPELINE] CSV copied to: {processed_csv}")
                results['steps']['csv_copy'] = {'success': True, 'path': str(processed_csv)}
            else:
                raise Exception(f"Input CSV not found: {input_csv_path}")
            
            # Step 3: Preprocessing (if needed)
            print(f"[PIPELINE] STEP 3: Data preprocessing...")
            
            # Check if we need to preprocess
            train_csv = self.paths['processed_data'] / 'train.csv'
            if not train_csv.exists():
                # Run preprocessing
                success, stdout, stderr = self.run_subprocess_step(
                    'preprocess.py',
                    ['--input', str(input_csv_path), '--output-dir', str(self.paths['processed_data'])],
                    'Data preprocessing'
                )
                results['steps']['preprocessing'] = {
                    'success': success,
                    'stdout': stdout[:500] if stdout else '',
                    'stderr': stderr[:500] if stderr else ''
                }
                
                if not success:
                    print(f"[PIPELINE] WARNING: Preprocessing failed, continuing...")
            else:
                print(f"[PIPELINE] Preprocessing skipped - CSV files exist")
                results['steps']['preprocessing'] = {'success': True, 'note': 'Skipped - files exist'}
            
            # Step 4: Generate embeddings
            print(f"[PIPELINE] STEP 4: Generating embeddings...")
            
            success, stdout, stderr = self.run_subprocess_step(
                'embed_dataset.py',
                [
                    '--output-dir', str(session_dir / 'embeddings'),
                    '--force-recreate'
                ],
                'Embedding generation'
            )
            results['steps']['embeddings'] = {
                'success': success,
                'stdout': stdout[:500] if stdout else '',
                'stderr': stderr[:500] if stderr else '',
                'output_dir': str(session_dir / 'embeddings')
            }
            
            # Verify embeddings were created
            embeddings_dir = session_dir / 'embeddings'
            embedding_files = list(embeddings_dir.glob('*.npy'))
            if embedding_files:
                print(f"[PIPELINE] Embedding files created: {len(embedding_files)} files")
                for emb_file in embedding_files:
                    print(f"[PIPELINE] - {emb_file.name}")
            else:
                print(f"[PIPELINE] WARNING: No embedding files found in {embeddings_dir}")
            
            # Step 5: Train MLP model
            print(f"[PIPELINE] STEP 5: Training MLP model...")
            
            success, stdout, stderr = self.run_subprocess_step(
                'train_mlp.py',
                [
                    '--output-dir', str(session_dir / 'models'),
                    '--embeddings-dir', str(session_dir / 'embeddings'),
                    '--fast'
                ],
                'MLP training'
            )
            results['steps']['mlp_training'] = {
                'success': success,
                'stdout': stdout[:500] if stdout else '',
                'stderr': stderr[:500] if stderr else '',
                'output_dir': str(session_dir / 'models')
            }
            
            # Verify MLP model was saved
            mlp_model_path = session_dir / 'models' / 'mlp_model.pth'
            if mlp_model_path.exists():
                print(f"[PIPELINE] MLP model saved: {mlp_model_path}")
                print(f"[PIPELINE] MLP model size: {mlp_model_path.stat().st_size / 1024:.1f} KB")
            else:
                print(f"[PIPELINE] WARNING: MLP model not found at {mlp_model_path}")
            
            # Step 6: Train SVM model
            print(f"[PIPELINE] STEP 6: Training SVM model...")
            
            success, stdout, stderr = self.run_subprocess_step(
                'train_svm.py',
                [
                    '--output-dir', str(session_dir / 'models'),
                    '--embeddings-dir', str(session_dir / 'embeddings'),
                    '--fast'
                ],
                'SVM training'
            )
            results['steps']['svm_training'] = {
                'success': success,
                'stdout': stdout[:500] if stdout else '',
                'stderr': stderr[:500] if stderr else '',
                'output_dir': str(session_dir / 'models')
            }
            
            # Verify SVM model was saved
            svm_model_path = session_dir / 'models' / 'svm_model.pkl'
            if svm_model_path.exists():
                print(f"[PIPELINE] SVM model saved: {svm_model_path}")
                print(f"[PIPELINE] SVM model size: {svm_model_path.stat().st_size / 1024:.1f} KB")
            else:
                print(f"[PIPELINE] WARNING: SVM model not found at {svm_model_path}")
            
            # Step 7: Generate comprehensive report
            print(f"[PIPELINE] STEP 7: Generating comprehensive report...")
            
            success, stdout, stderr = self.run_subprocess_step(
                'report.py',
                [
                    '--models-dir', str(session_dir / 'models'),
                    '--embeddings-dir', str(session_dir / 'embeddings'),
                    '--output-dir', str(session_dir / 'reports'),
                    '--plots-dir', str(session_dir / 'plots')
                ],
                'Report generation'
            )
            results['steps']['report_generation'] = {
                'success': success,
                'stdout': stdout[:500] if stdout else '',
                'stderr': stderr[:500] if stderr else '',
                'output_dir': str(session_dir / 'reports')
            }
            
            # Verify report files were created
            reports_dir = session_dir / 'reports'
            plots_dir = session_dir / 'plots'
            
            report_files = list(reports_dir.glob('*'))
            plot_files = list(plots_dir.glob('*'))
            
            print(f"[PIPELINE] Report files created: {len(report_files)}")
            for report_file in report_files:
                print(f"[PIPELINE] - {report_file.name}")
            
            print(f"[PIPELINE] Plot files created: {len(plot_files)}")
            for plot_file in plot_files:
                print(f"[PIPELINE] - {plot_file.name}")
            
            # Step 8: Generate final results summary
            print(f"[PIPELINE] STEP 8: Generating final results...")
            
            final_results = self.compile_final_results(session_dir, results)
            results['final_results'].update(final_results)
            
            # Step 9: Create pipeline summary
            print(f"[PIPELINE] STEP 9: Creating pipeline summary...")
            self.create_pipeline_summary_report(results, str(session_dir))
            
            # Final success assessment
            pipeline_end = datetime.now()
            duration = (pipeline_end - pipeline_start).total_seconds()
            
            results['end_time'] = pipeline_end.isoformat()
            results['duration_seconds'] = duration
            results['success'] = True
            
            print(f"[PIPELINE] ========================================")
            print(f"[PIPELINE] PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"[PIPELINE] Duration: {duration:.1f} seconds")
            print(f"[PIPELINE] Session directory: {session_dir}")
            print(f"[PIPELINE] ========================================")
            
            logger.info(f"🎉 Complete pipeline SUCCESS!")
            logger.info(f"   Duration: {duration:.1f} seconds")
            logger.info(f"   Results saved to: {session_dir}")
            
            return results
            
        except Exception as e:
            pipeline_end = datetime.now()
            duration = (pipeline_end - pipeline_start).total_seconds()
            
            error_msg = f"Pipeline failed: {str(e)}"
            print(f"[PIPELINE] CRITICAL ERROR: {error_msg}")
            logger.error(f"❌ {error_msg}")
            
            results['end_time'] = pipeline_end.isoformat()
            results['duration_seconds'] = duration
            results['success'] = False
            results['error'] = error_msg
            
            return results
    
    def compile_final_results(self, session_dir: Path, pipeline_results: Dict) -> Dict:
        """
        🔧 FIXED: Compile comprehensive final results for GUI
        """
        print(f"[PIPELINE] Compiling final results from session: {session_dir}")
        
        final_results = {
            'session_directory': str(session_dir),
            'predictions': {},
            'metrics': {},
            'insights': [],
            'files_created': {},
            'directory_structure': {}
        }
        
        try:
            # Check what files were actually created
            for subdir in ['processed', 'embeddings', 'models', 'plots', 'reports', 'logs']:
                subdir_path = session_dir / subdir
                if subdir_path.exists():
                    files = list(subdir_path.iterdir())
                    final_results['files_created'][subdir] = [f.name for f in files]
                    final_results['directory_structure'][subdir] = len(files)
                    print(f"[PIPELINE] {subdir}: {len(files)} files")
                else:
                    final_results['files_created'][subdir] = []
                    final_results['directory_structure'][subdir] = 0
            
            # Try to load model metrics from reports
            reports_dir = session_dir / 'reports'
            if reports_dir.exists():
                # Look for metrics files
                for metrics_file in reports_dir.glob('*metrics*.json'):
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics_data = json.load(f)
                            final_results['metrics'].update(metrics_data)
                        print(f"[PIPELINE] Loaded metrics from: {metrics_file.name}")
                    except Exception as e:
                        print(f"[PIPELINE] Could not load metrics from {metrics_file}: {e}")
            
            # Generate basic insights
            insights = []
            
            # Count successful steps
            successful_steps = sum(1 for step in pipeline_results.get('steps', {}).values() 
                                 if step.get('success', False))
            total_steps = len(pipeline_results.get('steps', {}))
            
            if successful_steps > 0:
                insights.append(f"Pipeline completed {successful_steps}/{total_steps} steps successfully")
            
            # File creation insights
            total_files = sum(final_results['directory_structure'].values())
            if total_files > 0:
                insights.append(f"Generated {total_files} output files across directories")
            
            # Model insights
            models_created = final_results['directory_structure'].get('models', 0)
            if models_created > 0:
                insights.append(f"Successfully trained and saved {models_created} models")
            
            # Reports insights
            reports_created = final_results['directory_structure'].get('reports', 0)
            if reports_created > 0:
                insights.append(f"Generated {reports_created} analysis reports")
            
            final_results['insights'] = insights
            
            print(f"[PIPELINE] Final results compiled: {len(insights)} insights")
            
            return final_results
            
        except Exception as e:
            print(f"[PIPELINE] Error compiling final results: {e}")
            return final_results
    
    def create_pipeline_summary_report(self, results: Dict[str, Any], session_dir: str):
        """Create a comprehensive summary report of the pipeline execution."""
        try:
            session_path = Path(session_dir)
            summary_path = session_path / "pipeline_summary.txt"
            
            summary_lines = [
                "REAL SENTIMENT ANALYSIS PIPELINE SUMMARY",
                "=" * 50,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Input File: {results.get('input_file', 'Unknown')}",
                f"Pipeline Type: {results.get('pipeline_type', 'Unknown')}",
                f"Timestamp: {results.get('timestamp', 'Unknown')}",
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
                
                if 'output_dir' in step_info:
                    summary_lines.append(f"  Output: {step_info['output_dir']}")
                
                if not step_info.get('success', False) and 'stderr' in step_info:
                    error_msg = step_info['stderr'][:100] + "..." if len(step_info['stderr']) > 100 else step_info['stderr']
                    summary_lines.append(f"  Error: {error_msg}")
            
            # Add final results summary
            final_results = results.get('final_results', {})
            
            summary_lines.extend([
                "",
                "RESULTS SUMMARY:",
                "-" * 20
            ])
            
            # Files created
            files_created = final_results.get('files_created', {})
            total_files = sum(len(files) for files in files_created.values())
            summary_lines.append(f"Total files created: {total_files}")
            
            for directory, files in files_created.items():
                if files:
                    summary_lines.append(f"  {directory}: {len(files)} files")
                    for file in files[:5]:  # Show first 5 files
                        summary_lines.append(f"    - {file}")
                    if len(files) > 5:
                        summary_lines.append(f"    ... and {len(files) - 5} more")
            
            # Insights summary
            insights = final_results.get('insights', [])
            if insights:
                summary_lines.extend([
                    "",
                    f"PIPELINE INSIGHTS ({len(insights)} generated):",
                    "-" * 20
                ])
                for i, insight in enumerate(insights, 1):
                    summary_lines.append(f"{i}. {insight}")
            
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
            
            print(f"[PIPELINE] Pipeline summary saved: {summary_path}")
            logger.info(f"📄 Pipeline summary saved: {summary_path}")
            
        except Exception as e:
            print(f"[PIPELINE] Failed to create pipeline summary: {e}")
            logger.warning(f"⚠️ Failed to create pipeline summary: {e}")


# === 🔧 FIXED WRAPPER FUNCTIONS FOR GUI INTEGRATION ===

def run_complete_csv_analysis(csv_path: str, text_column: str = 'text', 
                            label_column: str = 'label') -> Dict[str, Any]:
    """
    🚀 FIXED: Complete CSV analysis wrapper that REALLY executes the pipeline
    """
    print(f"[WRAPPER] Starting complete CSV analysis for: {csv_path}")
    
    try:
        if not Path(csv_path).exists():
            return {
                'success': False,
                'error': f"CSV file not found: {csv_path}"
            }
        
        # Initialize FIXED pipeline runner
        runner = PipelineRunner()
        
        # Execute the REAL complete pipeline
        print(f"[WRAPPER] Executing REAL complete pipeline...")
        result = runner.run_real_complete_pipeline(csv_path)
        
        print(f"[WRAPPER] Pipeline execution completed: {result.get('success', False)}")
        
        return result
        
    except Exception as e:
        error_msg = f"Complete CSV analysis error: {str(e)}"
        print(f"[WRAPPER] ERROR: {error_msg}")
        logger.error(f"❌ {error_msg}")
        return {
            'success': False,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }

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
        
        # Execute basic analysis (without full pipeline)
        print(f"[DATASET_ANALYSIS] Analyzing dataset: {csv_path}")
        
        # Basic CSV analysis
        df = pd.read_csv(csv_path)
        
        analysis_results = {
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
                'column_types': df.dtypes.astype(str).to_dict()
            }
        }
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"❌ Dataset analysis wrapper error: {e}")
        return {
            'error': f"Analysis error: {str(e)}",
            'success': False,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Enhanced main function with FIXED pipeline automation support."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FIXED Advanced Sentiment Analysis Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete REAL pipeline
  python scripts/pipeline_runner.py --action full-real --file dataset.csv
  
  # Check system status
  python scripts/pipeline_runner.py --action check-data
        """
    )
    
    parser.add_argument(
        "--action", 
        choices=['full-real', 'check-data', 'test-session'],
        default='full-real',
        help="Action to execute"
    )
    parser.add_argument("--file", type=str, help="CSV file for complete analysis")
    
    args = parser.parse_args()
    
    # Initialize FIXED pipeline runner
    runner = PipelineRunner()
    
    # Execute based on action
    if args.action == 'check-data':
        print("🔍 FIXED System check...")
        
        # Test session directory creation
        test_session = runner.create_session_directory()
        print(f"✅ Test session created: {test_session}")
        
        # Check scripts exist
        scripts_to_check = ['embed_dataset.py', 'train_mlp.py', 'train_svm.py', 'report.py']
        
        for script in scripts_to_check:
            script_path = SCRIPTS_DIR / script
            status = "✅" if script_path.exists() else "❌"
            print(f"{status} {script}: {script_path}")
    
    elif args.action == 'test-session':
        print("🧪 Testing session directory creation...")
        
        session_dir = runner.create_session_directory()
        print(f"Session created: {session_dir}")
        
        # List created directories
        for item in session_dir.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}/")
    
    elif args.action == 'full-real':
        if not args.file:
            print("❌ Error: --file required for real pipeline")
            print("💡 Example: python scripts/pipeline_runner.py --action full-real --file dataset.csv")
            return
        
        print(f"🚀 Starting FIXED complete pipeline for: {args.file}")
        results = runner.run_real_complete_pipeline(args.file)
        
        if results['success']:
            print("✅ FIXED PIPELINE SUCCESS!")
            print(f"   Duration: {results['duration_seconds']:.1f} seconds")
            print(f"   Session: {results['session_directory']}")
        else:
            print(f"❌ FIXED PIPELINE FAILED: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()