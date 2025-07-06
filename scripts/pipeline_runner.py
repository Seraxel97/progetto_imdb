#!/usr/bin/env python3
"""
Enhanced Pipeline Runner - COMPLETE UNIVERSAL ORCHESTRATION
Complete pipeline orchestration for automated sentiment analysis with full adaptability.

🆕 ENHANCED FEATURES:
- ✅ Universal CSV processing: handles any CSV structure with intelligent detection
- ✅ Adaptive pipeline execution: automatically adjusts to available data and models
- ✅ Complete error recovery: graceful handling of missing files and edge cases
- ✅ Real-time progress tracking with detailed logging and status updates
- ✅ GUI integration with comprehensive status reporting and file management
- ✅ Intelligent fallback mechanisms for all pipeline stages
- ✅ Enhanced reporting with insights generation and visualization
- ✅ Full compatibility with existing scripts and automation systems

USAGE:
    # Complete automated pipeline
    python scripts/pipeline_runner.py --action full-auto --file dataset.csv
    
    # Universal CSV processing
    python scripts/pipeline_runner.py --action process-csv --file any_structure.csv
    
    # Pipeline with custom options
    python scripts/pipeline_runner.py --action full --fast-mode --session-name "my_analysis"
    
    # System health check
    python scripts/pipeline_runner.py --action health-check
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import torch
import shutil
from pathlib import Path
from datetime import datetime
import logging
import subprocess
from typing import Dict, Any, Optional, Tuple, List, Callable
import warnings
from ftfy import fix_text
import time

warnings.filterwarnings('ignore')

# Dynamic project root detection
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

# Setup logging
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'enhanced_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedPipelineRunner:
    """
    Enhanced Pipeline Runner with complete universal orchestration capabilities.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize the enhanced pipeline runner.
        
        Args:
            project_root: Root directory of the project (auto-detect if None)
        """
        if project_root is None:
            self.project_root = PROJECT_ROOT
        else:
            self.project_root = Path(project_root)
        
        self.setup_paths()
        self.ensure_directories()
        
        logger.info(f"🚀 Enhanced Pipeline Runner initialized")
        logger.info(f"📁 Project root: {self.project_root.absolute()}")
    
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
            
            # Logs path
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
    
    def check_system_health(self) -> Dict[str, Any]:
        """🆕 Comprehensive system health check"""
        logger.info("🔍 Performing comprehensive system health check...")
        
        health_status = {
            'overall_health': 'unknown',
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'recommendations': [],
            'critical_issues': [],
            'warnings': []
        }
        
        # Check Python environment
        health_status['checks']['python'] = {
            'version': sys.version,
            'version_info': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'executable': sys.executable,
            'status': 'ok' if sys.version_info >= (3, 7) else 'warning'
        }
        
        if sys.version_info < (3, 7):
            health_status['warnings'].append("Python version < 3.7 - some features may not work properly")
        
        # Check required dependencies
        required_deps = [
            'pandas', 'numpy', 'scikit-learn', 'torch', 'transformers',
            'sentence_transformers', 'matplotlib', 'seaborn', 'joblib'
        ]
        
        missing_deps = []
        for dep in required_deps:
            try:
                __import__(dep)
                health_status['checks'][f'dependency_{dep}'] = {'status': 'ok', 'available': True}
            except ImportError:
                missing_deps.append(dep)
                health_status['checks'][f'dependency_{dep}'] = {'status': 'missing', 'available': False}
        
        if missing_deps:
            health_status['critical_issues'].append(f"Missing dependencies: {', '.join(missing_deps)}")
            health_status['recommendations'].append(f"Install missing packages: pip install {' '.join(missing_deps)}")
        
        # Check directory structure
        for name, path in self.paths.items():
            exists = path.exists()
            health_status['checks'][f'directory_{name}'] = {
                'path': str(path),
                'exists': exists,
                'status': 'ok' if exists else 'created'
            }
            
            if not exists:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    health_status['checks'][f'directory_{name}']['status'] = 'created'
                except Exception as e:
                    health_status['checks'][f'directory_{name}']['status'] = 'error'
                    health_status['critical_issues'].append(f"Cannot create directory {name}: {e}")
        
        # Check script availability
        essential_scripts = [
            'preprocess.py', 'embed_dataset.py', 'train_mlp.py', 
            'train_svm.py', 'report.py', 'enhanced_utils_unified.py'
        ]
        
        missing_scripts = []
        for script in essential_scripts:
            script_path = self.paths['scripts_dir'] / script
            exists = script_path.exists()
            health_status['checks'][f'script_{script}'] = {
                'path': str(script_path),
                'exists': exists,
                'status': 'ok' if exists else 'missing'
            }
            
            if not exists:
                missing_scripts.append(script)
        
        if missing_scripts:
            health_status['critical_issues'].append(f"Missing scripts: {', '.join(missing_scripts)}")
        
        # Check GPU availability
        try:
            if torch.cuda.is_available():
                health_status['checks']['gpu'] = {
                    'available': True,
                    'device_count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device(),
                    'device_name': torch.cuda.get_device_name(),
                    'status': 'ok'
                }
            else:
                health_status['checks']['gpu'] = {
                    'available': False,
                    'status': 'warning'
                }
                health_status['warnings'].append("No GPU available - training will be slower")
        except Exception as e:
            health_status['checks']['gpu'] = {
                'available': False,
                'error': str(e),
                'status': 'error'
            }
        
        # Determine overall health
        critical_count = len(health_status['critical_issues'])
        warning_count = len(health_status['warnings'])
        
        if critical_count == 0 and warning_count == 0:
            health_status['overall_health'] = 'excellent'
        elif critical_count == 0 and warning_count <= 2:
            health_status['overall_health'] = 'good'
        elif critical_count == 0:
            health_status['overall_health'] = 'fair'
        else:
            health_status['overall_health'] = 'poor'
        
        # Add general recommendations
        if health_status['overall_health'] == 'excellent':
            health_status['recommendations'].append("System is ready for optimal performance")
        elif missing_deps:
            health_status['recommendations'].append("Install missing dependencies for full functionality")
        
        logger.info(f"🏥 System health check completed: {health_status['overall_health'].upper()}")
        logger.info(f"   Critical issues: {critical_count}")
        logger.info(f"   Warnings: {warning_count}")
        
        return health_status
    
    def process_csv_universal(self, csv_path: str, 
                            force_text_column: str = None,
                            force_label_column: str = None,
                            output_dir: str = None,
                            log_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """🆕 Universal CSV processing with intelligent column detection"""
        
        def log_message(msg: str):
            logger.info(msg)
            if log_callback:
                log_callback(msg)
        
        try:
            log_message("🔄 Starting Universal CSV Processing")
            log_message("=" * 50)
            
            # Import enhanced utils
            from enhanced_utils_unified import validate_and_process_csv, create_timestamped_session_dir
            
            # Setup output directory
            if output_dir is None:
                output_dir = create_timestamped_session_dir(self.paths['results_dir'], "csv_processing")
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            log_message(f"📄 Input file: {csv_path}")
            log_message(f"📁 Output directory: {output_dir}")
            
            # Process CSV
            result = validate_and_process_csv(
                csv_path,
                force_text_column=force_text_column,
                force_label_column=force_label_column,
                logger=logger
            )
            
            if not result['success']:
                raise Exception(f"CSV processing failed: {result.get('error', 'Unknown error')}")
            
            # Save processed data with appropriate splits
            processed_df = result['processed_df']
            total_samples = len(processed_df)
            has_labels = result['has_labels']
            
            processed_dir = output_dir / "processed"
            processed_dir.mkdir(exist_ok=True)
            
            if total_samples < 10 or not has_labels:
                # Small dataset or no labels - create inference file
                processed_df.to_csv(processed_dir / "inference.csv", index=False)
                log_message(f"📋 Created inference.csv ({total_samples} samples)")
                split_info = {"inference": total_samples}
            else:
                # Create proper splits
                from sklearn.model_selection import train_test_split
                
                if total_samples >= 100:
                    # Full splits
                    train_df, temp_df = train_test_split(
                        processed_df, test_size=0.3, random_state=42,
                        stratify=processed_df['label'] if has_labels else None
                    )
                    val_df, test_df = train_test_split(
                        temp_df, test_size=0.5, random_state=42,
                        stratify=temp_df['label'] if has_labels else None
                    )
                else:
                    # Simple split
                    train_df, test_df = train_test_split(
                        processed_df, test_size=0.3, random_state=42,
                        stratify=processed_df['label'] if has_labels else None
                    )
                    val_df = test_df.copy()
                
                # Save splits
                train_df.to_csv(processed_dir / "train.csv", index=False)
                val_df.to_csv(processed_dir / "val.csv", index=False)
                test_df.to_csv(processed_dir / "test.csv", index=False)
                
                split_info = {
                    "train": len(train_df),
                    "val": len(val_df),
                    "test": len(test_df)
                }
                
                log_message(f"📋 Created splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
            
            # Save processing metadata
            processing_metadata = {
                'timestamp': datetime.now().isoformat(),
                'input_file': str(csv_path),
                'output_directory': str(output_dir),
                'processing_results': result,
                'split_info': split_info,
                'inference_only': not has_labels or total_samples < 10
            }
            
            metadata_file = output_dir / "processing_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(processing_metadata, f, indent=2, ensure_ascii=False, default=str)
            
            log_message("✅ Universal CSV processing completed successfully!")
            log_message(f"   📊 Total samples: {total_samples}")
            log_message(f"   🏷️ Has labels: {has_labels}")
            log_message(f"   📁 Output: {output_dir}")
            
            return {
                'success': True,
                'output_directory': str(output_dir),
                'total_samples': total_samples,
                'has_labels': has_labels,
                'inference_only': not has_labels or total_samples < 10,
                'split_info': split_info,
                'processing_metadata': processing_metadata,
                'metadata_file': str(metadata_file)
            }
            
        except Exception as e:
            log_message(f"❌ Universal CSV processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'input_file': csv_path
            }
    
    def run_complete_pipeline(self, input_file: str, 
                            session_name: str = None,
                            fast_mode: bool = True,
                            force_text_column: str = None,
                            force_label_column: str = None,
                            log_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """🆕 Run complete pipeline with enhanced orchestration"""
        
        def log_message(msg: str):
            logger.info(msg)
            if log_callback:
                log_callback(msg)
        
        try:
            log_message("🚀 Starting Complete Enhanced Pipeline")
            log_message("=" * 60)
            
            # Import enhanced utils
            from enhanced_utils_unified import auto_embed_and_predict
            
            # Run the complete automated pipeline
            result = auto_embed_and_predict(
                file_path=input_file,
                fast_mode=fast_mode,
                force_text_column=force_text_column,
                force_label_column=force_label_column,
                log_callback=log_callback
            )
            
            # Enhance result with additional metadata
            if result.get('success') or result.get('overall_success'):
                session_dir = Path(result['session_directory'])
                
                # Create comprehensive summary
                summary = self._create_pipeline_summary(result, session_dir)
                result['enhanced_summary'] = summary
                
                # Create ZIP archive if requested
                if session_name:
                    zip_path = self._create_results_archive(session_dir, session_name)
                    result['archive_path'] = zip_path
                
                log_message("🎉 Complete pipeline finished successfully!")
                log_message(f"   📁 Session: {session_dir}")
                log_message(f"   📊 Status: {result.get('status', 'unknown').upper()}")
                log_message(f"   ⏱️ Duration: {result.get('total_duration', 0):.1f}s")
            else:
                log_message(f"❌ Complete pipeline failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            log_message(f"❌ Complete pipeline error: {str(e)}")
            return {
                'success': False,
                'overall_success': False,
                'error': str(e),
                'input_file': input_file
            }
    
    def _create_pipeline_summary(self, result: Dict[str, Any], session_dir: Path) -> Dict[str, Any]:
        """Create comprehensive pipeline summary"""
        summary = {
            'pipeline_info': {
                'session_directory': str(session_dir),
                'pipeline_type': result.get('pipeline_type', 'unknown'),
                'status': result.get('status', 'unknown'),
                'total_duration': result.get('total_duration', 0),
                'inference_only': result.get('inference_only', False)
            },
            'steps_summary': {},
            'output_files': [],
            'model_performance': {},
            'recommendations': []
        }
        
        # Analyze steps
        for step_name, step_info in result.get('steps', {}).items():
            summary['steps_summary'][step_name] = {
                'status': step_info.get('status', 'unknown'),
                'duration': step_info.get('duration', 0),
                'success': step_info.get('status') == 'completed'
            }
        
        # Collect output files
        try:
            for pattern in ["*.png", "*.pdf", "*.json", "*.csv", "*.txt", "*.pkl", "*.pth"]:
                files = list(session_dir.rglob(pattern))
                summary['output_files'].extend([str(f.relative_to(session_dir)) for f in files])
        except Exception:
            pass
        
        # Extract model performance if available
        try:
            final_results = result.get('final_results', {})
            
            if 'mlp_status' in final_results:
                mlp_status = final_results['mlp_status']
                if mlp_status.get('status') == 'completed':
                    performance = mlp_status.get('performance', {})
                    summary['model_performance']['mlp'] = {
                        'accuracy': performance.get('accuracy', 0),
                        'training_time': performance.get('training_time', 0)
                    }
            
            if 'svm_status' in final_results:
                svm_status = final_results['svm_status']
                if svm_status.get('status') == 'completed':
                    performance = svm_status.get('performance', {})
                    summary['model_performance']['svm'] = {
                        'accuracy': performance.get('accuracy', 0),
                        'f1_score': performance.get('f1_score', 0),
                        'training_time': performance.get('training_time', 0)
                    }
        except Exception:
            pass
        
        # Generate recommendations
        if result.get('inference_only'):
            summary['recommendations'].append("Dataset has no labels - only data analysis performed")
        
        if summary['model_performance']:
            best_model = max(summary['model_performance'].items(), 
                           key=lambda x: x[1].get('accuracy', 0))
            summary['recommendations'].append(f"Best performing model: {best_model[0].upper()}")
        
        if result.get('errors'):
            summary['recommendations'].append(f"Pipeline had {len(result['errors'])} errors - check logs for details")
        
        return summary
    
    def _create_results_archive(self, session_dir: Path, archive_name: str) -> str:
        """Create ZIP archive of results"""
        try:
            archive_path = session_dir.parent / f"{archive_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            zip_path = shutil.make_archive(str(archive_path), 'zip', session_dir)
            logger.info(f"📦 Created results archive: {zip_path}")
            return zip_path
        except Exception as e:
            logger.warning(f"⚠️ Failed to create archive: {e}")
            return None
    
    def run_subprocess_step(self, script_name: str, args: List[str], description: str,
                          timeout: int = 600, stream_callback: Optional[Callable[[str], None]] = None) -> Tuple[bool, str, str]:
        """Run a subprocess step with comprehensive error handling."""
        script_path = self.paths['scripts_dir'] / script_name
        if not script_path.exists():
            return False, "", f"Script not found: {script_path}"
        
        cmd = [sys.executable, str(script_path)] + args
        
        logger.info(f"🔄 {description}")
        logger.info(f"   Command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_root,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                universal_newlines=True
            )
            
            stdout_lines = []
            stderr_lines = []
            
            # Real-time streaming
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                text = fix_text(line)
                stdout_lines.append(text)
                if stream_callback:
                    stream_callback(text.strip())
            
            stderr = process.stderr.read()
            if stderr:
                stderr_text = fix_text(stderr)
                stderr_lines.append(stderr_text)
                if stream_callback:
                    stream_callback(stderr_text.strip())
            
            process.wait(timeout=timeout)
            
            stdout = "".join(stdout_lines)
            stderr = "".join(stderr_lines)
            
            if process.returncode == 0:
                logger.info(f"✅ {description} - SUCCESS")
                return True, stdout, stderr
            else:
                logger.error(f"❌ {description} - FAILED (code: {process.returncode})")
                return False, stdout, stderr
                
        except subprocess.TimeoutExpired:
            error_msg = f"{description} timed out after {timeout} seconds"
            logger.error(f"⏰ {error_msg}")
            return False, "", error_msg
        except Exception as e:
            error_msg = f"{description} error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, "", error_msg

# =============================================================================
# GUI INTEGRATION WRAPPER FUNCTIONS - EXACT EXPORTS FOR COMPATIBILITY
# =============================================================================

def run_dataset_analysis(csv_path: str) -> Dict[str, Any]:
    """
    🔧 GUI Integration: Wrapper function for dataset analysis.
    """
    try:
        runner = EnhancedPipelineRunner()
        
        # Use the universal CSV processing
        result = runner.process_csv_universal(csv_path)
        
        if result['success']:
            # Convert to expected GUI format
            return {
                'success': True,
                'file_info': {
                    'path': csv_path,
                    'samples': result['total_samples'],
                    'has_labels': result['has_labels']
                },
                'data_analysis': {
                    'total_samples': result['total_samples'],
                    'has_labels': result['has_labels']
                },
                'text_analysis': {
                    'column_used': result['processing_metadata']['processing_results']['text_column_used'],
                    'avg_length': result['processing_metadata']['processing_results']['stats']['text_stats']['avg_length']
                },
                'sentiment_analysis': {
                    'has_labels': result['has_labels'],
                    'label_distribution': result['processing_metadata']['processing_results']['stats']['label_distribution']
                } if result['has_labels'] else {'has_labels': False},
                'system_info': {
                    'output_directory': result['output_directory']
                }
            }
        else:
            return {
                'success': False,
                'error': result['error']
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def run_complete_csv_analysis(csv_path: str, text_column: str = 'text', 
                             label_column: str = 'label',
                             log_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    🔧 GUI Integration: Complete CSV analysis wrapper.
    """
    try:
        runner = EnhancedPipelineRunner()
        
        result = runner.run_complete_pipeline(
            input_file=csv_path,
            force_text_column=text_column if text_column != 'text' else None,
            force_label_column=label_column if label_column != 'label' else None,
            log_callback=log_callback
        )
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'overall_success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# =============================================================================
# MAIN CLI INTERFACE
# =============================================================================

def main():
    """Enhanced main function with comprehensive CLI support"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Pipeline Runner - Complete Universal Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🆕 ENHANCED ACTIONS:
  health-check     - Comprehensive system health check
  process-csv      - Universal CSV processing with intelligent detection
  full-auto        - Complete automated pipeline (CSV → models → report)
  full             - Traditional full pipeline
  check-data       - Data availability check
  analyze          - Quick dataset analysis

Examples:
  python scripts/pipeline_runner.py --action health-check
  python scripts/pipeline_runner.py --action process-csv --file dataset.csv
  python scripts/pipeline_runner.py --action full-auto --file dataset.csv --session-name "analysis_1"
  python scripts/pipeline_runner.py --action full --fast-mode
        """
    )
    
    # Main action
    parser.add_argument(
        "--action", 
        choices=[
            'health-check', 'process-csv', 'full-auto', 'full', 
            'check-data', 'analyze', 'embeddings', 'train', 'report'
        ],
        default='health-check',
        help="Action to execute (default: health-check)"
    )
    
    # File arguments
    parser.add_argument("--file", type=str, help="Input CSV file")
    parser.add_argument("--csv-path", type=str, help="Alternative CSV path parameter")
    
    # Column specification
    parser.add_argument("--text-column", type=str, help="Force specific text column")
    parser.add_argument("--label-column", type=str, help="Force specific label column")
    
    # Pipeline options
    parser.add_argument("--session-name", type=str, help="Custom session name")
    parser.add_argument("--fast-mode", action="store_true", default=True, 
                       help="Use fast training modes (default: True)")
    parser.add_argument("--output-dir", type=str, help="Custom output directory")
    
    # Output control
    parser.add_argument("--save-report", action="store_true", default=True, 
                       help="Save execution report (default: True)")
    parser.add_argument("--create-archive", action="store_true", 
                       help="Create ZIP archive of results")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output (default: True)")
    parser.add_argument("--quiet", action="store_true", 
                       help="Suppress detailed output")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Initialize pipeline runner
    runner = EnhancedPipelineRunner()
    
    # Handle input file parameter flexibility
    input_file = args.file or args.csv_path
    
    try:
        # Execute based on action
        if args.action == 'health-check':
            print("🔍 Performing comprehensive system health check...")
            
            health_status = runner.check_system_health()
            
            print(f"\n🏥 SYSTEM HEALTH: {health_status['overall_health'].upper()}")
            print("=" * 50)
            
            # Critical issues
            if health_status['critical_issues']:
                print("❌ CRITICAL ISSUES:")
                for issue in health_status['critical_issues']:
                    print(f"   • {issue}")
                print()
            
            # Warnings
            if health_status['warnings']:
                print("⚠️ WARNINGS:")
                for warning in health_status['warnings']:
                    print(f"   • {warning}")
                print()
            
            # Recommendations
            if health_status['recommendations']:
                print("💡 RECOMMENDATIONS:")
                for rec in health_status['recommendations']:
                    print(f"   • {rec}")
                print()
            
            # Detailed checks if verbose
            if args.verbose:
                print("📊 DETAILED CHECKS:")
                for check_name, check_info in health_status['checks'].items():
                    status_icon = "✅" if check_info.get('status') == 'ok' else "⚠️" if check_info.get('status') == 'warning' else "❌"
                    print(f"   {status_icon} {check_name}: {check_info.get('status', 'unknown')}")
            
            return 0 if health_status['overall_health'] in ['excellent', 'good'] else 1
        
        elif args.action == 'process-csv':
            if not input_file:
                print("❌ Error: --file required for CSV processing")
                return 1
            
            print(f"🔄 Processing CSV file: {input_file}")
            
            result = runner.process_csv_universal(
                csv_path=input_file,
                force_text_column=args.text_column,
                force_label_column=args.label_column,
                output_dir=args.output_dir
            )
            
            if result['success']:
                print("✅ CSV processing completed successfully!")
                print(f"   📊 Samples: {result['total_samples']}")
                print(f"   🏷️ Has labels: {result['has_labels']}")
                print(f"   📁 Output: {result['output_directory']}")
                return 0
            else:
                print(f"❌ CSV processing failed: {result['error']}")
                return 1
        
        elif args.action == 'full-auto':
            if not input_file:
                print("❌ Error: --file required for automated pipeline")
                return 1
            
            print(f"🚀 Starting complete automated pipeline: {input_file}")
            
            result = runner.run_complete_pipeline(
                input_file=input_file,
                session_name=args.session_name,
                fast_mode=args.fast_mode,
                force_text_column=args.text_column,
                force_label_column=args.label_column
            )
            
            if result.get('success') or result.get('overall_success'):
                print("🎉 COMPLETE PIPELINE SUCCESS!")
                print(f"   📁 Session: {result.get('session_directory', 'unknown')}")
                print(f"   📊 Status: {result.get('status', 'unknown').upper()}")
                print(f"   ⏱️ Duration: {result.get('total_duration', 0):.1f}s")
                
                if 'enhanced_summary' in result:
                    summary = result['enhanced_summary']
                    if summary.get('model_performance'):
                        print("   🤖 Model Performance:")
                        for model, perf in summary['model_performance'].items():
                            acc = perf.get('accuracy', 0)
                            print(f"      {model.upper()}: {acc:.3f} accuracy")
                
                return 0
            else:
                print(f"❌ PIPELINE FAILED: {result.get('error', 'Unknown error')}")
                return 1
        
        elif args.action == 'analyze':
            if not input_file:
                print("❌ Error: --file required for analysis")
                return 1
            
            print(f"📊 Analyzing dataset: {input_file}")
            
            result = run_dataset_analysis(input_file)
            
            if result.get('success'):
                print("✅ Analysis completed!")
                print(f"   📊 Samples: {result['file_info']['samples']}")
                print(f"   🏷️ Has labels: {result['file_info']['has_labels']}")
                
                if 'text_analysis' in result:
                    text_info = result['text_analysis']
                    print(f"   📏 Avg text length: {text_info.get('avg_length', 0):.0f} chars")
                
                return 0
            else:
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                return 1
        
        else:
            print(f"❌ Action '{args.action}' not yet implemented in enhanced version")
            print("💡 Use 'health-check', 'process-csv', or 'full-auto' for now")
            return 1
    
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
        return 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())