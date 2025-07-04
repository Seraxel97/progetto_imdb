import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import time


def setup_robust_paths():
    """Setup robust paths for the project"""
    current_dir = Path.cwd()
    project_root = current_dir
    
    # Try to find project root by looking for key directories
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / 'scripts').exists() and (parent / 'data').exists():
            project_root = parent
            break
    
    return project_root


def create_timestamped_session_dir(base_path: Path, prefix: str = "session") -> Path:
    """Create a timestamped session directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_path / f"{prefix}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def run_subprocess_with_timeout(command: List[str], timeout: int = 300, cwd: Optional[Path] = None) -> Dict[str, Any]:
    """Run subprocess with timeout and return detailed results"""
    try:
        start_time = time.time()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            check=False
        )
        end_time = time.time()
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": end_time - start_time,
            "command": " ".join(command)
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "duration": timeout,
            "command": " ".join(command)
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "duration": 0,
            "command": " ".join(command)
        }


def safe_convert_for_json(obj: Any) -> Any:
    """Convert objects to JSON-serializable format"""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: safe_convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_convert_for_json(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)


def auto_embed_and_predict(file_path: str = None, csv_path: str = None, session_dir: Optional[Path] = None, fast_mode: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Automated pipeline for sentiment analysis: preprocessing, embedding, training, and reporting.
    
    Args:
        file_path: Path to input CSV file (preferred parameter name)
        csv_path: Alternative parameter name for input CSV file (for backward compatibility)
        session_dir: Optional session directory. If None, creates a new timestamped directory
        fast_mode: Whether to use fast mode (passed to SVM training)
        **kwargs: Additional keyword arguments (for flexibility)
        
    Returns:
        Dictionary with results and status information
    """
    # Handle parameter flexibility - accept both file_path and csv_path
    input_file = file_path or csv_path
    if not input_file:
        raise ValueError("Either file_path or csv_path must be provided")
    
    # Convert to Path object
    input_file = Path(input_file)
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize results dictionary
    results = {
        "status": "started",
        "steps": {},
        "errors": [],
        "total_duration": 0,
        "session_dir": None,
        "summary_file": None,
        "report_pdf": None,
        "plots": [],
        "predictions": None,
        "session_directory": None,
        "overall_success": False
    }
    
    start_time = time.time()
    
    try:
        # Setup paths
        project_root = setup_robust_paths()
        
        # Validate input file
        if not input_file.exists():
            raise FileNotFoundError(f"Input CSV file not found: {input_file}")
        
        # Create or use session directory
        if session_dir is None:
            results_dir = project_root / "results"
            results_dir.mkdir(exist_ok=True)
            session_dir = create_timestamped_session_dir(results_dir, "auto_analysis")
        
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        results["session_dir"] = str(session_dir)
        results["session_directory"] = str(session_dir)
        
        # Create subdirectories
        subdirs = ["processed", "embeddings", "models", "reports", "plots"]
        for subdir in subdirs:
            (session_dir / subdir).mkdir(exist_ok=True)
        
        # Setup logging for this session (use UTF-8 to avoid emoji issues)
        log_file = session_dir / "pipeline.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        logger.info(f"Starting automated pipeline for {input_file}")
        logger.info(f"Session directory: {session_dir}")
        logger.info(f"Fast mode: {fast_mode}")
        
        # Change to project root for script execution
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        # ----- Step 1: preprocessing -----
        preprocess_cmd = [
            sys.executable, "scripts/preprocess.py",
            "--input", str(input_file),
            "--output-dir", str(session_dir / "processed")
        ]
        step_result = run_subprocess_with_timeout(preprocess_cmd, timeout=300, cwd=project_root)
        results["steps"]["preprocessing"] = {
            "status": "completed" if step_result["success"] else "failed",
            "duration": step_result["duration"],
            "returncode": step_result["returncode"]
        }

        inference_only = False
        if step_result["success"]:
            meta_file = session_dir / "processed" / "preprocessing_metadata.json"
            try:
                if meta_file.exists():
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    inference_only = bool(meta.get("inference_only", False))
                else:
                    inference_only = (session_dir / "processed" / "inference.csv").exists()
            except Exception as e:
                logger.warning(f"Could not determine inference mode: {e}")

        results["inference_only"] = inference_only

        if not step_result["success"]:
            logger.error(f"Preprocessing failed: {step_result['stderr']}")

        # ----- Remaining pipeline steps -----
        pipeline_steps = []

        embed_cmd = [
            sys.executable, "scripts/embed_dataset.py",
            "--input-dir", str(session_dir / "processed"),
            "--output-dir", str(session_dir / "embeddings")
        ]
        if inference_only:
            embed_cmd.append("--inference-only")

        pipeline_steps.append({
            "name": "embedding",
            "command": embed_cmd,
            "timeout": 600
        })

        if not inference_only:
            pipeline_steps.extend([
                {
                    "name": "mlp_training",
                    "command": [
                        sys.executable, "scripts/train_mlp.py",
                        "--embeddings-dir", str(session_dir / "embeddings"),
                        "--output-dir", str(session_dir / "models")
                    ],
                    "timeout": 900
                },
                {
                    "name": "svm_training",
                    "command": [
                        sys.executable, "scripts/train_svm.py",
                        "--embeddings-dir", str(session_dir / "embeddings"),
                        "--output-dir", str(session_dir / "models")
                    ] + (["--fast"] if fast_mode else []),
                    "timeout": 600
                }
            ])
        else:
            # Mark training steps as skipped
            results["steps"]["mlp_training"] = {
                "status": "skipped",
                "duration": 0,
                "returncode": None
            }
            results["steps"]["svm_training"] = {
                "status": "skipped",
                "duration": 0,
                "returncode": None
            }
            logger.info("Training skipped due to missing labels")

        pipeline_steps.append({
            "name": "report",
            "command": [
                sys.executable, "scripts/report.py",
                "--input-dir", str(session_dir),
                "--output-dir", str(session_dir / "reports")
            ],
            "timeout": 300
        })
        
        # Execute pipeline steps
        for step in pipeline_steps:
            step_name = step["name"]
            logger.info(f"Starting step: {step_name}")
            
            step_result = run_subprocess_with_timeout(
                step["command"],
                timeout=step["timeout"],
                cwd=project_root
            )
            
            results["steps"][step_name] = {
                "status": "completed" if step_result["success"] else "failed",
                "duration": step_result["duration"],
                "returncode": step_result["returncode"]
            }
            
            if step_result["success"]:
                logger.info(f"Step {step_name} completed successfully in {step_result['duration']:.2f}s")
            else:
                error_msg = f"Step {step_name} failed: {step_result['stderr']}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                
                # Continue with next step even if current fails
                continue
        
        # Restore original working directory
        os.chdir(original_cwd)
        
        # Collect output files
        try:
            # Look for predictions CSV
            predictions_files = list(session_dir.glob("**/predictions*.csv"))
            if predictions_files:
                results["predictions"] = str(predictions_files[0])
            
            # Look for report PDF
            report_files = list(session_dir.glob("**/report*.pdf"))
            if report_files:
                results["report_pdf"] = str(report_files[0])
            
            # Look for plots
            plot_files = list(session_dir.glob("plots/*.png")) + list(session_dir.glob("plots/*.jpg"))
            results["plots"] = [str(f) for f in plot_files]
            
        except Exception as e:
            logger.warning(f"Error collecting output files: {e}")
        
        # Create pipeline summary
        summary_file = session_dir / "pipeline_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=== AUTOMATED SENTIMENT ANALYSIS PIPELINE SUMMARY ===\n\n")
            f.write(f"Input File: {input_file}\n")
            f.write(f"Fast Mode: {fast_mode}\n")
            f.write(f"Session Directory: {session_dir}\n")
            f.write(f"Start Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration: {time.time() - start_time:.2f} seconds\n\n")
            
            f.write("=== PIPELINE STEPS ===\n")
            for step_name, step_info in results["steps"].items():
                f.write(f"{step_name}: {step_info['status']} ({step_info['duration']:.2f}s)\n")

            if inference_only:
                f.write("\nTraining skipped due to missing labels\n")

            if results["errors"]:
                f.write("\n=== ERRORS ===\n")
                for error in results["errors"]:
                    f.write(f"- {error}\n")
            
            f.write("\n=== OUTPUT FILES ===\n")
            if results["predictions"]:
                f.write(f"Predictions: {results['predictions']}\n")
            if results["report_pdf"]:
                f.write(f"Report PDF: {results['report_pdf']}\n")
            if results["plots"]:
                f.write(f"Plots: {len(results['plots'])} files\n")
        
        results["summary_file"] = str(summary_file)
        
        # Determine overall status
        failed_steps = [name for name, info in results["steps"].items() if info["status"] == "failed"]
        if not failed_steps:
            results["status"] = "success"
        elif len(failed_steps) < len(results["steps"]):
            results["status"] = "partial_success"
        else:
            results["status"] = "failed"

        results["total_duration"] = time.time() - start_time
        results["overall_success"] = results["status"] == "success"
        
        logger.info(f"Pipeline completed with status: {results['status']}")
        logger.info(f"Total duration: {results['total_duration']:.2f} seconds")
        
    except Exception as e:
        results["status"] = "error"
        results["errors"].append(str(e))
        logger.error(f"Pipeline failed with error: {e}")
        
        # Restore working directory even on error
        try:
            os.chdir(original_cwd)
        except:
            pass
    
    # Convert all paths to strings for JSON serialization
    return safe_convert_for_json(results)
