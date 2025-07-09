#!/usr/bin/env python3
"""
Main Entry Point - Sentiment Analysis System - ENHANCED VERSION WITH PIPELINE CLI
Simple entry point that launches the Streamlit GUI dashboard with enhanced dependency checking
and better error handling for improved user experience.

🆕 NEW FEATURES:
- ✅ Added --run-pipeline option for complete CLI automation
- ✅ Pipeline execution with real-time progress tracking
- ✅ Automatic session directory creation and management
- ✅ Comprehensive error handling and recovery
- ✅ Results summary and file organization
- ✅ Compatible with existing GUI and automation features
- ✅ Standard directory copying for compatibility

🔧 ENHANCEMENTS:
- ✅ Enhanced dependency checking with detailed reporting
- ✅ Better error messages and troubleshooting guidance
- ✅ Robust path detection and validation
- ✅ Improved user feedback and status reporting
- ✅ Graceful handling of missing dependencies
- ✅ Environment validation and setup assistance

USAGE:
    # Launch GUI (default)
    python main.py
    
    # 🆕 NEW: Run complete pipeline from CLI
    python main.py --run-pipeline --input dataset.csv
    python main.py --run-pipeline --input data/raw/imdb_raw.csv --fast-mode
    python main.py --run-pipeline --input my_data.csv --session-name "my_analysis"
    
    # Other CLI options
    python main.py --check
    python main.py --validate
    python main.py --setup
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time
import json
import shutil

def setup_logging():
    """Setup basic logging for the main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def detect_project_root() -> Path:
    """Enhanced project root detection"""
    try:
        # Strategy 1: Use __file__ if available
        if '__file__' in globals():
            current_file = Path(__file__).resolve()
            return current_file.parent
    except:
        pass
    
    # Strategy 2: Look for marker files
    current = Path.cwd()
    marker_files = ['config.yaml', 'requirements.txt', 'gui_data_dashboard.py']
    
    for _ in range(3):  # Search up to 3 levels up
        for marker in marker_files:
            if (current / marker).exists():
                return current
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent
    
    # Fallback to current directory
    return Path.cwd()

def check_gui_file(project_root: Path) -> Tuple[bool, str, Optional[Path]]:
    """Check for GUI dashboard file with enhanced detection"""
    gui_filenames = ['gui_data_dashboard.py', 'gui.py', 'dashboard.py']
    
    for filename in gui_filenames:
        gui_path = project_root / filename
        if gui_path.exists():
            return True, f"Found GUI file: {filename}", gui_path
    
    # Also check in common subdirectories
    common_dirs = ['src', 'app', 'gui']
    for dirname in common_dirs:
        dir_path = project_root / dirname
        if dir_path.exists():
            for filename in gui_filenames:
                gui_path = dir_path / filename
                if gui_path.exists():
                    return True, f"Found GUI file: {dirname}/{filename}", gui_path
    
    return False, "GUI dashboard file not found", None

def enhanced_dependency_check() -> Dict[str, Dict[str, any]]:
    """Enhanced dependency checking with detailed reporting"""
    logger = logging.getLogger(__name__)
    
    # Core dependencies with their import names and descriptions
    dependencies = {
        'streamlit': {
            'import_name': 'streamlit',
            'description': 'Web app framework for the GUI',
            'critical': True,
            'install_cmd': 'pip install streamlit'
        },
        'pandas': {
            'import_name': 'pandas',
            'description': 'Data manipulation and analysis',
            'critical': True,
            'install_cmd': 'pip install pandas'
        },
        'numpy': {
            'import_name': 'numpy', 
            'description': 'Numerical computing',
            'critical': True,
            'install_cmd': 'pip install numpy'
        },
        'plotly': {
            'import_name': 'plotly',
            'description': 'Interactive plotting',
            'critical': True,
            'install_cmd': 'pip install plotly'
        },
        'matplotlib': {
            'import_name': 'matplotlib',
            'description': 'Static plotting',
            'critical': True,
            'install_cmd': 'pip install matplotlib'
        },
        'seaborn': {
            'import_name': 'seaborn',
            'description': 'Statistical plotting',
            'critical': True,
            'install_cmd': 'pip install seaborn'
        },
        'scikit-learn': {
            'import_name': 'sklearn',
            'description': 'Machine learning library',
            'critical': False,
            'install_cmd': 'pip install scikit-learn'
        },
        'torch': {
            'import_name': 'torch',
            'description': 'PyTorch for deep learning',
            'critical': False,
            'install_cmd': 'pip install torch'
        },
        'sentence-transformers': {
            'import_name': 'sentence_transformers',
            'description': 'Sentence embeddings',
            'critical': False,
            'install_cmd': 'pip install sentence-transformers'
        },
        'transformers': {
            'import_name': 'transformers',
            'description': 'HuggingFace transformers',
            'critical': False,
            'install_cmd': 'pip install transformers'
        },
        'nltk': {
            'import_name': 'nltk',
            'description': 'Natural language toolkit',
            'critical': False,
            'install_cmd': 'pip install nltk'
        },
        'joblib': {
            'import_name': 'joblib',
            'description': 'Model serialization',
            'critical': False,
            'install_cmd': 'pip install joblib'
        }
    }
    
    results = {}
    missing_critical = []
    missing_optional = []
    
    logger.info("🔍 Checking dependencies...")
    
    for package_name, info in dependencies.items():
        try:
            __import__(info['import_name'])
            results[package_name] = {
                'available': True,
                'critical': info['critical'],
                'description': info['description'],
                'status': 'OK'
            }
            logger.debug(f"✅ {package_name}: Available")
            
        except ImportError as e:
            results[package_name] = {
                'available': False,
                'critical': info['critical'],
                'description': info['description'],
                'install_cmd': info['install_cmd'],
                'error': str(e),
                'status': 'MISSING'
            }
            
            if info['critical']:
                missing_critical.append(package_name)
                logger.warning(f"❌ {package_name}: Missing (CRITICAL)")
            else:
                missing_optional.append(package_name)
                logger.info(f"⚠️ {package_name}: Missing (optional)")
    
    # Summary
    total_deps = len(dependencies)
    available_deps = sum(1 for r in results.values() if r['available'])
    critical_available = sum(1 for r in results.values() if r['available'] and r['critical'])
    total_critical = sum(1 for info in dependencies.values() if info['critical'])
    
    summary = {
        'total_dependencies': total_deps,
        'available_dependencies': available_deps,
        'missing_critical': missing_critical,
        'missing_optional': missing_optional,
        'critical_available': critical_available,
        'total_critical': total_critical,
        'all_critical_available': len(missing_critical) == 0,
        'completion_percentage': (available_deps / total_deps) * 100
    }
    
    return {
        'dependencies': results,
        'summary': summary
    }

def print_dependency_report(check_result: Dict) -> None:
    """Print a comprehensive dependency report"""
    deps = check_result['dependencies']
    summary = check_result['summary']
    
    print("\n" + "="*60)
    print("🔍 DEPENDENCY CHECK REPORT")
    print("="*60)
    
    print(f"📊 Overall Status: {summary['available_dependencies']}/{summary['total_dependencies']} dependencies available ({summary['completion_percentage']:.1f}%)")
    print(f"🔧 Critical Dependencies: {summary['critical_available']}/{summary['total_critical']} available")
    
    if summary['all_critical_available']:
        print("✅ All critical dependencies are available!")
    else:
        print("❌ Some critical dependencies are missing!")
    
    print("\n📋 DETAILED STATUS:")
    
    # Group by status
    available = [(name, info) for name, info in deps.items() if info['available']]
    missing_critical = [(name, info) for name, info in deps.items() if not info['available'] and info['critical']]
    missing_optional = [(name, info) for name, info in deps.items() if not info['available'] and not info['critical']]
    
    if available:
        print("\n✅ AVAILABLE:")
        for name, info in available:
            status_icon = "🔧" if info['critical'] else "⭐"
            print(f"   {status_icon} {name}: {info['description']}")
    
    if missing_critical:
        print("\n❌ MISSING (CRITICAL):")
        for name, info in missing_critical:
            print(f"   🔧 {name}: {info['description']}")
            print(f"      Install: {info['install_cmd']}")
    
    if missing_optional:
        print("\n⚠️ MISSING (OPTIONAL):")
        for name, info in missing_optional:
            print(f"   ⭐ {name}: {info['description']}")
            print(f"      Install: {info['install_cmd']}")
    
    # Installation suggestions
    if summary['missing_critical'] or summary['missing_optional']:
        print("\n💡 INSTALLATION SUGGESTIONS:")
        
        if summary['missing_critical']:
            critical_packages = [deps[name]['install_cmd'].split()[-1] for name in summary['missing_critical']]
            print(f"   Critical packages: pip install {' '.join(critical_packages)}")
        
        if summary['missing_optional']:
            optional_packages = [deps[name]['install_cmd'].split()[-1] for name in summary['missing_optional']]
            print(f"   Optional packages: pip install {' '.join(optional_packages)}")
        
        print("\n   Or install all at once:")
        print("   pip install -r requirements.txt")
    
    print("\n" + "="*60)

def validate_environment(project_root: Path) -> Tuple[bool, List[str]]:
    """Validate the environment and project setup"""
    logger = logging.getLogger(__name__)
    issues = []
    
    logger.info("🔍 Validating environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 7):
        issues.append(f"Python version {python_version.major}.{python_version.minor} is too old. Required: 3.7+")
    else:
        logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check important directories
    important_dirs = ['data', 'scripts', 'results']
    for dirname in important_dirs:
        dir_path = project_root / dirname
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"📁 Created directory: {dirname}")
            except Exception as e:
                issues.append(f"Cannot create directory {dirname}: {e}")
        else:
            logger.debug(f"✅ Directory exists: {dirname}")
    
    # Check important files
    important_files = ['config.yaml', 'requirements.txt']
    for filename in important_files:
        file_path = project_root / filename
        if not file_path.exists():
            issues.append(f"Important file missing: {filename}")
        else:
            logger.debug(f"✅ File exists: {filename}")
    
    # Check scripts directory
    scripts_dir = project_root / 'scripts'
    if scripts_dir.exists():
        essential_scripts = [
            'preprocess.py', 'embed_dataset.py', 'train_mlp.py', 
            'train_svm.py', 'report.py', 'pipeline_runner.py'
        ]
        missing_scripts = []
        for script in essential_scripts:
            if not (scripts_dir / script).exists():
                missing_scripts.append(script)
        
        if missing_scripts:
            issues.append(f"Missing essential scripts: {', '.join(missing_scripts)}")
        else:
            logger.info(f"✅ All essential scripts found")
    
    # Check write permissions
    try:
        test_file = project_root / 'test_write_permissions.tmp'
        test_file.write_text('test')
        test_file.unlink()
        logger.debug("✅ Write permissions OK")
    except Exception as e:
        issues.append(f"No write permissions in project directory: {e}")
    
    return len(issues) == 0, issues

def launch_streamlit_app(gui_path: Path, project_root: Path) -> int:
    """Launch Streamlit app with enhanced error handling"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"🚀 Starting Sentiment Analysis Dashboard...")
    logger.info(f"📁 GUI Path: {gui_path}")
    logger.info(f"📁 Project Root: {project_root}")
    logger.info("🌐 Opening in browser...")
    logger.info("")
    logger.info("💡 To stop the server, press Ctrl+C")
    logger.info("")
    
    try:
        # Prepare Streamlit command
        cmd = [sys.executable, "-m", "streamlit", "run", str(gui_path)]
        
        # Set up environment
        env = os.environ.copy()
        env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        env["STREAMLIT_SERVER_HEADLESS"] = "true"
        
        # Change to project directory
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        try:
            # Run Streamlit
            result = subprocess.run(
                cmd,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return result.returncode
        finally:
            # Restore original directory
            os.chdir(original_cwd)
        
    except KeyboardInterrupt:
        logger.info("\n✅ Dashboard stopped by user")
        return 0
        
    except FileNotFoundError:
        logger.error("❌ Streamlit not found!")
        logger.error("💡 Install it with: pip install streamlit")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Error launching dashboard: {e}")
        return 1

def setup_environment(project_root: Path) -> bool:
    """Interactive environment setup"""
    logger = logging.getLogger(__name__)
    
    print("\n🛠️ ENVIRONMENT SETUP")
    print("=" * 30)
    
    # Check if requirements.txt exists
    req_file = project_root / 'requirements.txt'
    if not req_file.exists():
        print("❌ requirements.txt not found!")
        print("💡 Create requirements.txt with necessary dependencies")
        return False
    
    print(f"📋 Found requirements.txt: {req_file}")
    
    # Ask user if they want to install dependencies
    try:
        response = input("\n🤔 Install dependencies from requirements.txt? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            print("\n📦 Installing dependencies...")
            try:
                cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                print("✅ Dependencies installed successfully!")
                logger.info("Dependencies installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Installation failed: {e}")
                print("💡 Try installing manually:")
                print(f"   pip install -r {req_file}")
                return False
        else:
            print("⏭️ Skipping dependency installation")
            return True
            
    except KeyboardInterrupt:
        print("\n⏹️ Setup cancelled by user")
        return False

# 🆕 NEW: Complete Pipeline Execution Functions
def create_session_directory(project_root: Path, session_name: Optional[str] = None) -> Path:
    """Create a timestamped session directory for pipeline execution"""
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if session_name:
        session_dir = results_dir / f"session_{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        session_dir = results_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['processed', 'embeddings', 'models', 'plots', 'reports', 'logs']
    for subdir in subdirs:
        (session_dir / subdir).mkdir(exist_ok=True)
    
    return session_dir

def run_script_step(script_name: str, args: List[str], project_root: Path, 
                   session_dir: Path, logger) -> Tuple[bool, str, str, float]:
    """Execute a single pipeline script step with enhanced error handling"""
    script_path = project_root / "scripts" / script_name
    
    if not script_path.exists():
        return False, "", f"Script not found: {script_path}", 0.0
    
    cmd = [sys.executable, str(script_path)] + args
    
    logger.info(f"🔄 Executing: {script_name}")
    logger.info(f"   Command: {' '.join(cmd)}")
    logger.info(f"   Working directory: {project_root}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,  # 30 minutes timeout
            cwd=project_root
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ {script_name} completed successfully in {duration:.2f}s")
            return True, result.stdout, result.stderr, duration
        else:
            logger.error(f"❌ {script_name} failed with return code {result.returncode}")
            logger.error(f"   STDERR: {result.stderr[:500]}...")
            return False, result.stdout, result.stderr, duration
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        error_msg = f"{script_name} timed out after 30 minutes"
        logger.error(f"⏰ {error_msg}")
        return False, "", error_msg, duration
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Error executing {script_name}: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return False, "", error_msg, duration

def run_complete_pipeline(input_file: str, project_root: Path, session_name: Optional[str] = None,
                         fast_mode: bool = False, logger=None) -> Dict:
    """🆕 NEW: Execute the complete sentiment analysis pipeline from CLI"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("🚀 STARTING COMPLETE SENTIMENT ANALYSIS PIPELINE")
    logger.info("=" * 60)
    
    # Validate input file
    input_path = Path(input_file)
    if not input_path.exists():
        error_msg = f"Input file not found: {input_file}"
        logger.error(f"❌ {error_msg}")
        return {
            'success': False,
            'error': error_msg,
            'input_file': input_file
        }
    
    # Create session directory
    try:
        session_dir = create_session_directory(project_root, session_name)
        logger.info(f"📁 Session directory created: {session_dir}")
    except Exception as e:
        error_msg = f"Failed to create session directory: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return {
            'success': False,
            'error': error_msg
        }
    
    # Initialize results tracking
    pipeline_start_time = time.time()
    results = {
        'success': False,
        'input_file': str(input_path),
        'session_directory': str(session_dir),
        'session_name': session_name,
        'fast_mode': fast_mode,
        'start_time': datetime.now().isoformat(),
        'steps': {},
        'total_duration': 0.0,
        'errors': [],
        'final_results': {}
    }
    
    # Setup logging for this session
    log_file = session_dir / "logs" / "pipeline.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Define pipeline steps
    pipeline_steps = [
        {
            'name': 'preprocessing',
            'script': 'preprocess.py',
            'args': [
                '--input', str(input_path),
                '--output-dir', str(session_dir / 'processed')
            ],
            'description': 'Data preprocessing and train/val/test split'
        },
        {
            'name': 'embedding',
            'script': 'embed_dataset.py',
            'args': [
                '--input-dir', str(session_dir / 'processed'),
                '--output-dir', str(session_dir / 'embeddings')
            ],
            'description': 'Generate sentence embeddings using MiniLM'
        },
        {
            'name': 'train_mlp',
            'script': 'train_mlp.py',
            'args': [
                '--embeddings-dir', str(session_dir / 'embeddings'),
                '--output-dir', str(session_dir),
                '--epochs', '50' if fast_mode else '100'
            ] + (['--fast'] if fast_mode else []),
            'description': 'Train Multi-Layer Perceptron model'
        },
        {
            'name': 'train_svm',
            'script': 'train_svm.py',
            'args': [
                '--embeddings-dir', str(session_dir / 'embeddings'),
                '--output-dir', str(session_dir),
                '--session-name', session_name or 'pipeline'
            ] + (['--fast'] if fast_mode else []),
            'description': 'Train Support Vector Machine model'
        },
        {
            'name': 'report',
            'script': 'report.py',
            'args': [
                '--models-dir', str(session_dir / 'models'),
                '--test-data', str(session_dir / 'processed' / 'test.csv'),
                '--results-dir', str(session_dir),
                '--auto-default'
            ],
            'description': 'Generate comprehensive evaluation report'
        }
    ]
    
    # Execute pipeline steps
    total_steps = len(pipeline_steps)
    
    for i, step in enumerate(pipeline_steps, 1):
        step_name = step['name']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STEP {i}/{total_steps}: {step['description']}")
        logger.info(f"{'='*60}")
        
        # Execute step
        success, stdout, stderr, duration = run_script_step(
            step['script'], 
            step['args'], 
            project_root, 
            session_dir, 
            logger
        )
        
        # Record step results
        results['steps'][step_name] = {
            'success': success,
            'duration': duration,
            'description': step['description'],
            'script': step['script'],
            'args': step['args'],
            'stdout_length': len(stdout),
            'stderr_length': len(stderr)
        }
        
        if not success:
            error_msg = f"Step {step_name} failed: {stderr[:200]}..."
            results['errors'].append(error_msg)
            logger.error(f"❌ Pipeline stopped at step: {step_name}")
            
            # Continue with remaining steps anyway (partial results)
            logger.info("⚠️ Continuing with remaining steps to get partial results...")
            continue
        else:
            logger.info(f"✅ Step {step_name} completed successfully")
    
    # 🆕 NUOVO: Copia risultati anche nelle directory standard
    try:
        logger.info("📁 Copying results to standard directories...")
        
        # Directory standard
        standard_processed = project_root / "data" / "processed"
        standard_embeddings = project_root / "data" / "embeddings"
        standard_models = project_root / "results" / "models"
        standard_reports = project_root / "results" / "reports"
        standard_plots = project_root / "results" / "plots"
        
        # Crea directory standard
        for std_dir in [standard_processed, standard_embeddings, standard_models, standard_reports, standard_plots]:
            std_dir.mkdir(parents=True, exist_ok=True)
        
        # Copia processed files
        session_processed = session_dir / "processed"
        if session_processed.exists():
            for file in session_processed.glob("*.csv"):
                shutil.copy2(file, standard_processed / file.name)
                logger.info(f"   📄 Copied {file.name} to standard processed")
        
        # Copia embedding files
        session_embeddings = session_dir / "embeddings"
        if session_embeddings.exists():
            for file in session_embeddings.glob("*.npy"):
                shutil.copy2(file, standard_embeddings / file.name)
                logger.info(f"   🧠 Copied {file.name} to standard embeddings")
        
        # Copia model files
        session_models = session_dir / "models"
        if session_models.exists():
            for file in session_models.glob("*"):
                if file.is_file():
                    shutil.copy2(file, standard_models / file.name)
                    logger.info(f"   🤖 Copied {file.name} to standard models")
        
        # Copia report files
        session_reports = session_dir / "reports"
        if session_reports.exists():
            for file in session_reports.glob("*"):
                if file.is_file():
                    shutil.copy2(file, standard_reports / file.name)
                    logger.info(f"   📊 Copied {file.name} to standard reports")
        
        # Copia plot files
        session_plots = session_dir / "plots"
        if session_plots.exists():
            for file in session_plots.glob("*.png"):
                shutil.copy2(file, standard_plots / file.name)
                logger.info(f"   📈 Copied {file.name} to standard plots")
        
        logger.info("✅ Successfully copied all results to standard directories")
        
        # Aggiorna results con le directory standard
        results["final_results"]["standard_directories"] = {
            "processed": str(standard_processed),
            "embeddings": str(standard_embeddings),
            "models": str(standard_models),
            "reports": str(standard_reports),
            "plots": str(standard_plots)
        }
        
    except Exception as e:
        logger.warning(f"⚠️ Warning: Could not copy to standard directories: {e}")
        # Non fallire se la copia non funziona
    
    # Calculate total duration and final status
    results['total_duration'] = time.time() - pipeline_start_time
    results['end_time'] = datetime.now().isoformat()
    
    # Determine overall success
    successful_steps = sum(1 for step in results['steps'].values() if step['success'])
    total_steps = len(results['steps'])
    
    if successful_steps == total_steps:
        results['success'] = True
        results['status'] = 'complete'
    elif successful_steps > 0:
        results['success'] = True
        results['status'] = 'partial'
    else:
        results['success'] = False
        results['status'] = 'failed'
    
    # Create pipeline summary
    summary_file = session_dir / "pipeline_summary.json"
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        results['summary_file'] = str(summary_file)
    except Exception as e:
        logger.warning(f"Failed to save summary file: {e}")
    
    # Collect final results
    try:
        # Look for evaluation report
        report_json = session_dir / "reports" / "evaluation_report.json"
        if report_json.exists():
            with open(report_json, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            results["final_results"]["evaluation_report"] = report_data
        
        # Look for model status files
        mlp_status = session_dir / "mlp_training_status.json"
        if mlp_status.exists():
            with open(mlp_status, 'r', encoding='utf-8') as f:
                results["final_results"]["mlp_status"] = json.load(f)
        
        svm_status = session_dir / "svm_training_status.json"
        if svm_status.exists():
            with open(svm_status, 'r', encoding='utf-8') as f:
                results["final_results"]["svm_status"] = json.load(f)
        
        # Collect output files
        output_files = []
        for pattern in ["*.png", "*.pdf", "*.json", "*.csv", "*.txt", "*.pkl", "*.pth"]:
            output_files.extend(session_dir.rglob(pattern))
        
        results["final_results"]["output_files"] = [str(f.relative_to(session_dir)) for f in output_files]
        
    except Exception as e:
        logger.warning(f"Error collecting final results: {e}")
    
    # Log final results
    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE EXECUTION COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Status: {results['status'].upper()}")
    logger.info(f"Successful steps: {successful_steps}/{total_steps}")
    logger.info(f"Total duration: {results['total_duration']:.1f} seconds")
    logger.info(f"Session directory: {session_dir}")
    
    if results['errors']:
        logger.warning(f"Errors encountered: {len(results['errors'])}")
        for error in results['errors']:
            logger.warning(f"  • {error}")
    
    # Show step summary
    logger.info("\nStep Summary:")
    for step_name, step_info in results['steps'].items():
        status = "✅" if step_info['success'] else "❌"
        logger.info(f"  {status} {step_name}: {step_info['duration']:.1f}s")
    
    # Remove file handler
    logger.removeHandler(file_handler)
    file_handler.close()
    
    return results

def show_help():
    """Show enhanced help information"""
    print("🤖 Sentiment Analysis System - Enhanced Version with Pipeline CLI")
    print("=" * 70)
    print()
    print("📊 USAGE:")
    print("   python main.py                        # Launch GUI dashboard")
    print("   python main.py --help                 # Show this help")
    print("   python main.py --check                # Check dependencies")
    print("   python main.py --validate             # Validate environment")
    print("   python main.py --setup                # Setup environment")
    print()
    print("🆕 NEW: COMPLETE PIPELINE EXECUTION:")
    print("   python main.py --run-pipeline --input dataset.csv")
    print("   python main.py --run-pipeline --input data/raw/imdb_raw.csv --fast-mode")
    print("   python main.py --run-pipeline --input my_data.csv --session-name 'my_analysis'")
    print()
    print("🎯 FEATURES:")
    print("   • Interactive sentiment analysis GUI")
    print("   • 🆕 Complete CLI pipeline automation")
    print("   • CSV file batch processing") 
    print("   • Complete model training pipeline")
    print("   • Performance evaluation and reporting")
    print("   • Real-time analysis and logging")
    print("   • Enhanced error handling and recovery")
    print("   • 🆕 Standard directory support")
    print()
    print("📁 PROJECT STRUCTURE:")
    print("   main.py                     # This entry point")
    print("   gui_data_dashboard.py       # Streamlit GUI")
    print("   config.yaml                 # Configuration")
    print("   requirements.txt            # Dependencies")
    print("   scripts/                    # Analysis scripts")
    print("   ├── preprocess.py           # Data preprocessing")
    print("   ├── embed_dataset.py        # Embedding generation")
    print("   ├── train_mlp.py            # MLP training")
    print("   ├── train_svm.py            # SVM training")
    print("   └── report.py               # Report generation")
    print("   data/                       # Dataset storage")
    print("   ├── processed/              # 🆕 Processed CSV files")
    print("   └── embeddings/             # 🆕 Generated embeddings")
    print("   results/                    # Output and models")
    print("   ├── models/                 # 🆕 Trained models")
    print("   ├── reports/                # 🆕 Analysis reports")
    print("   └── plots/                  # 🆕 Generated plots")
    print()
    print("🚀 QUICK START:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. GUI mode: python main.py")
    print("   3. 🆕 Pipeline mode: python main.py --run-pipeline --input your_data.csv")
    print("   4. View results in data/ and results/ directories")
    print()
    print("🛠️ PIPELINE DETAILS:")
    print("   The --run-pipeline option executes these steps in sequence:")
    print("   1. 📊 Data preprocessing (preprocess.py)")
    print("   2. 🧠 Embedding generation (embed_dataset.py)")
    print("   3. 🤖 MLP model training (train_mlp.py)")
    print("   4. ⚡ SVM model training (train_svm.py)")
    print("   5. 📋 Report generation (report.py)")
    print("   6. 🆕 Copy to standard directories")
    print()
    print("📊 PIPELINE OPTIONS:")
    print("   --input FILE              Input CSV file (required)")
    print("   --session-name NAME       Custom session name")
    print("   --fast-mode               Use fast training modes")
    print()
    print("🛠️ TROUBLESHOOTING:")
    print("   • Run --check to verify dependencies")
    print("   • Run --validate to check environment")
    print("   • Check results/session_*/logs/ for detailed logs")
    print("   • Ensure all files in requirements.txt are installed")

def main():
    """Enhanced main function with comprehensive error handling and pipeline CLI"""
    
    # Setup logging
    logger = setup_logging()
    
    # Parse arguments with enhanced options
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis System - Enhanced with Pipeline CLI",
        add_help=False  # We'll handle help ourselves
    )
    
    # Action arguments
    parser.add_argument("--help", "-h", action="store_true", help="Show help")
    parser.add_argument("--check", action="store_true", help="Check dependencies")
    parser.add_argument("--validate", action="store_true", help="Validate environment")
    parser.add_argument("--setup", action="store_true", help="Setup environment")
    
    # 🆕 NEW: Pipeline execution arguments
    parser.add_argument("--run-pipeline", action="store_true", help="🆕 Execute complete pipeline from CLI")
    parser.add_argument("--input", type=str, help="Input CSV file for pipeline")
    parser.add_argument("--session-name", type=str, help="Custom session name")
    parser.add_argument("--fast-mode", action="store_true", help="Use fast training modes")
    
    try:
        args = parser.parse_args()
    except SystemExit:
        # If parsing fails, show help and exit
        show_help()
        return 1
    
    # Handle command line arguments
    if args.help:
        show_help()
        return 0
        
    elif args.check:
        print("🔍 Checking dependencies...")
        check_result = enhanced_dependency_check()
        print_dependency_report(check_result)
        
        if check_result['summary']['all_critical_available']:
            return 0
        else:
            print("\n❌ Critical dependencies missing!")
            print("💡 Run: python main.py --setup")
            return 1
            
    elif args.validate:
        project_root = detect_project_root()
        print(f"🔍 Validating environment in: {project_root}")
        
        is_valid, issues = validate_environment(project_root)
        if is_valid:
            print("✅ Environment validation passed!")
            return 0
        else:
            print("❌ Environment validation failed:")
            for issue in issues:
                print(f"   • {issue}")
            return 1
            
    elif args.setup:
        project_root = detect_project_root()
        print(f"🛠️ Setting up environment in: {project_root}")
        
        if setup_environment(project_root):
            print("\n✅ Setup completed!")
            print("🚀 Run: python main.py")
            return 0
        else:
            print("\n❌ Setup failed!")
            return 1
    
    # 🆕 NEW: Handle pipeline execution
    elif args.run_pipeline:
        if not args.input:
            print("❌ Error: --input is required for pipeline execution")
            print("💡 Usage: python main.py --run-pipeline --input your_data.csv")
            print("💡 Run: python main.py --help for more information")
            return 1
        
        # Detect project root
        project_root = detect_project_root()
        logger.info(f"📁 Project root: {project_root}")
        
        # Validate environment first
        is_valid, issues = validate_environment(project_root)
        if not is_valid:
            print("❌ Environment validation failed:")
            for issue in issues:
                print(f"   • {issue}")
            print("\n💡 Run: python main.py --validate for details")
            print("💡 Run: python main.py --setup for guided setup")
            return 1
        
        # Check dependencies
        check_result = enhanced_dependency_check()
        if not check_result['summary']['all_critical_available']:
            print("❌ Critical dependencies missing!")
            print_dependency_report(check_result)
            print("\n💡 Run: python main.py --setup for guided installation")
            return 1
        
        # Execute pipeline
        print(f"🚀 Executing complete sentiment analysis pipeline")
        print(f"📄 Input file: {args.input}")
        if args.session_name:
            print(f"📁 Session name: {args.session_name}")
        if args.fast_mode:
            print(f"⚡ Fast mode: enabled")
        print()
        
        try:
            results = run_complete_pipeline(
                input_file=args.input,
                project_root=project_root,
                session_name=args.session_name,
                fast_mode=args.fast_mode,
                logger=logger
            )
            
            if results['success']:
                print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
                print(f"📁 Session results: {results['session_directory']}")
                print(f"⏱️ Total time: {results['total_duration']:.1f} seconds")
                print(f"📊 Status: {results['status']}")
                
                # Show standard directories
                if 'standard_directories' in results.get('final_results', {}):
                    std_dirs = results['final_results']['standard_directories']
                    print(f"\n📁 Standard directories populated:")
                    print(f"   📄 Processed: {std_dirs['processed']}")
                    print(f"   🧠 Embeddings: {std_dirs['embeddings']}")
                    print(f"   🤖 Models: {std_dirs['models']}")
                    print(f"   📊 Reports: {std_dirs['reports']}")
                    print(f"   📈 Plots: {std_dirs['plots']}")
                
                if 'output_files' in results.get('final_results', {}):
                    print(f"📄 Generated {len(results['final_results']['output_files'])} files")
                
                if results.get('errors'):
                    print(f"⚠️ Warnings: {len(results['errors'])} issues encountered")
                
                return 0
            else:
                print(f"\n❌ PIPELINE FAILED")
                print(f"📁 Partial results in: {results['session_directory']}")
                if results.get('errors'):
                    print("💡 Errors:")
                    for error in results['errors'][:3]:  # Show first 3 errors
                        print(f"   • {error}")
                
                return 1
                
        except KeyboardInterrupt:
            print("\n⏹️ Pipeline cancelled by user")
            return 0
        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            logger.error(f"Pipeline execution failed: {e}")
            return 1
    
    # Default: Launch GUI if no specific action is requested
    else:
        try:
            # Detect project root
            project_root = detect_project_root()
            logger.info(f"📁 Project root: {project_root}")
            
            # Validate environment
            is_valid, issues = validate_environment(project_root)
            if not is_valid:
                print("❌ Environment validation failed:")
                for issue in issues:
                    print(f"   • {issue}")
                print("\n💡 Run: python main.py --validate for details")
                print("💡 Run: python main.py --setup for guided setup")
                return 1
            
            # Check dependencies
            check_result = enhanced_dependency_check()
            if not check_result['summary']['all_critical_available']:
                print("❌ Critical dependencies missing!")
                print_dependency_report(check_result)
                print("\n💡 Run: python main.py --setup for guided installation")
                return 1
            
            # Find GUI file
            gui_found, gui_msg, gui_path = check_gui_file(project_root)
            if not gui_found:
                logger.error(f"❌ {gui_msg}")
                logger.error("💡 Ensure gui_data_dashboard.py exists in the project directory")
                return 1
            
            logger.info(f"✅ {gui_msg}")
            
            # Launch Streamlit app
            return launch_streamlit_app(gui_path, project_root)
            
        except KeyboardInterrupt:
            logger.info("\n⏹️ Startup cancelled by user")
            return 0
            
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            logger.error("💡 Run with --help for usage information")
            return 1

if __name__ == "__main__":
    sys.exit(main())
