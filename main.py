c#!/usr/bin/env python3
"""
Main Entry Point - Sentiment Analysis System - ENHANCED VERSION
Simple entry point that launches the Streamlit GUI dashboard with enhanced dependency checking
and better error handling for improved user experience.

🔧 ENHANCEMENTS:
- ✅ Enhanced dependency checking with detailed reporting
- ✅ Better error messages and troubleshooting guidance
- ✅ Robust path detection and validation
- ✅ Improved user feedback and status reporting
- ✅ Graceful handling of missing dependencies
- ✅ Environment validation and setup assistance
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
            result = subprocess.run(cmd, env=env)
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

def show_help():
    """Show enhanced help information"""
    print("🤖 Sentiment Analysis System - Enhanced Version")
    print("=" * 50)
    print()
    print("📊 USAGE:")
    print("   python main.py              # Launch GUI dashboard")
    print("   python main.py --help       # Show this help")
    print("   python main.py --check      # Check dependencies")
    print("   python main.py --validate   # Validate environment")
    print("   python main.py --setup      # Setup environment")
    print()
    print("🎯 FEATURES:")
    print("   • Interactive sentiment analysis GUI")
    print("   • CSV file batch processing") 
    print("   • Complete model training pipeline")
    print("   • Performance evaluation and reporting")
    print("   • Real-time analysis and logging")
    print("   • Enhanced error handling and recovery")
    print()
    print("📁 PROJECT STRUCTURE:")
    print("   main.py                     # This entry point")
    print("   gui_data_dashboard.py       # Streamlit GUI")
    print("   config.yaml                 # Configuration")
    print("   requirements.txt            # Dependencies")
    print("   scripts/                    # Analysis scripts")
    print("   data/                       # Dataset storage")
    print("   results/                    # Output and models")
    print()
    print("🚀 QUICK START:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Run: python main.py")
    print("   3. Open browser (auto-opens)")
    print("   4. Upload dataset or analyze text")
    print("   5. Train models or use existing ones")
    print()
    print("🛠️ TROUBLESHOOTING:")
    print("   • Run --check to verify dependencies")
    print("   • Run --validate to check environment")
    print("   • Check logs/ directory for detailed logs")
    print("   • Ensure all files in requirements.txt are installed")

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
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
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

def main():
    """Enhanced main function with comprehensive error handling"""
    
    # Setup logging
    logger = setup_logging()
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--help', '-h']:
            show_help()
            return 0
            
        elif arg == '--check':
            print("🔍 Checking dependencies...")
            check_result = enhanced_dependency_check()
            print_dependency_report(check_result)
            
            if check_result['summary']['all_critical_available']:
                return 0
            else:
                print("\n❌ Critical dependencies missing!")
                print("💡 Run: python main.py --setup")
                return 1
                
        elif arg == '--validate':
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
                
        elif arg == '--setup':
            project_root = detect_project_root()
            print(f"🛠️ Setting up environment in: {project_root}")
            
            if setup_environment(project_root):
                print("\n✅ Setup completed!")
                print("🚀 Run: python main.py")
                return 0
            else:
                print("\n❌ Setup failed!")
                return 1
                
        else:
            print(f"❌ Unknown argument: {sys.argv[1]}")
            print("💡 Use --help for usage information")
            return 1
    
    # Main execution flow
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
