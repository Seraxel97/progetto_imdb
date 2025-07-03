#!/usr/bin/env python3
"""
Main Entry Point - Sentiment Analysis System
Simple entry point that launches the Streamlit GUI dashboard
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main entry point - launch Streamlit GUI"""
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Path to GUI dashboard
    gui_path = script_dir / "gui_data_dashboard.py"
    
    if not gui_path.exists():
        print("❌ GUI dashboard not found at:", gui_path)
        print("💡 Make sure gui_data_dashboard.py is in the same directory as main.py")
        return 1
    
    print("🚀 Starting Sentiment Analysis Dashboard...")
    print(f"📁 GUI Path: {gui_path}")
    print("🌐 Opening in browser...")
    print()
    print("💡 To stop the server, press Ctrl+C")
    print()
    
    try:
        # Launch Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(gui_path)]
        
        # Add Streamlit configuration for better UX
        env = os.environ.copy()
        env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        env["STREAMLIT_SERVER_HEADLESS"] = "true"
        
        # Run Streamlit
        subprocess.run(cmd, env=env)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n✅ Dashboard stopped by user")
        return 0
        
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it:")
        print("   pip install streamlit")
        return 1
        
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return 1

def check_dependencies():
    """FIXED: Check if required dependencies are installed with correct module mapping"""
    required_modules = {
        'streamlit': 'streamlit',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'plotly': 'plotly',
        'scikit-learn': 'sklearn',
        'torch': 'torch',
        'sentence-transformers': 'sentence_transformers'
    }
    
    missing_packages = []
    
    for package, module_name in required_modules.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print()
        print("💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def show_help():
    """Show help information"""
    print("🤖 Sentiment Analysis System")
    print("=" * 40)
    print()
    print("📊 USAGE:")
    print("   python main.py              # Launch GUI dashboard")
    print("   python main.py --help       # Show this help")
    print("   python main.py --check      # Check dependencies")
    print()
    print("🎯 FEATURES:")
    print("   • Interactive text analysis")
    print("   • CSV file batch processing") 
    print("   • Model training pipeline")
    print("   • Performance evaluation")
    print("   • Real-time logging")
    print()
    print("📁 PROJECT STRUCTURE:")
    print("   main.py                     # This entry point")
    print("   gui_data_dashboard.py       # Streamlit GUI")
    print("   scripts/config_constants.py # Configuration")
    print("   scripts/pipeline_automation.py # Training pipeline")
    print("   data/                       # Training data")
    print("   results/                    # Models & results")
    print()
    print("🚀 GET STARTED:")
    print("   1. Run: python main.py")
    print("   2. Open browser (auto-opens)")
    print("   3. Upload dataset or analyze text")
    print("   4. Train models or use existing ones")

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            show_help()
            sys.exit(0)
        elif sys.argv[1] == '--check':
            print("🔍 Checking dependencies...")
            if check_dependencies():
                print("✅ All dependencies are installed!")
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            print(f"❌ Unknown argument: {sys.argv[1]}")
            print("💡 Use --help for usage information")
            sys.exit(1)
    
    # Check dependencies before launching
    if not check_dependencies():
        sys.exit(1)
    
    # Launch GUI
    sys.exit(main())
