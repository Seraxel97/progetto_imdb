#!/usr/bin/env python3
"""
Enhanced Utils Unified - UNIVERSAL DATA MANAGEMENT & PIPELINE ORCHESTRATION
Comprehensive utilities for sentiment analysis pipeline with intelligent fallbacks and data management.

🆕 ENHANCED FEATURES:
- ✅ Universal CSV processing with intelligent column detection
- ✅ Robust fallback management between inference, train, val files
- ✅ Smart data validation and normalization
- ✅ Comprehensive pipeline orchestration with error recovery
- ✅ GUI integration with real-time status reporting
- ✅ Automatic mode detection and graceful degradation
- ✅ Enhanced logging and progress tracking

USAGE:
    from enhanced_utils_unified import auto_embed_and_predict, validate_and_process_csv
    
    # Complete automated pipeline
    result = auto_embed_and_predict("dataset.csv")
    
    # CSV validation and processing
    processed_data = validate_and_process_csv("raw_data.csv")
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
import time
import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

def setup_robust_paths():
    """Setup robust paths for the project with enhanced detection"""
    current_dir = Path.cwd()
    project_root = current_dir
    
    # Strategy 1: Look for scripts directory in current or parent directories
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / 'scripts').exists() and (parent / 'data').exists():
            project_root = parent
            break
    
    # Strategy 2: Use __file__ if we're being imported
    try:
        if '__file__' in globals():
            file_path = Path(__file__).resolve()
            if file_path.parent.name == 'scripts':
                project_root = file_path.parent.parent
            else:
                project_root = file_path.parent
    except:
        pass
    
    return project_root

def create_timestamped_session_dir(base_path: Path, prefix: str = "session") -> Path:
    """Create a timestamped session directory with proper structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_path / f"{prefix}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Create standard subdirectories
    subdirs = ['processed', 'embeddings', 'models', 'plots', 'reports', 'logs']
    for subdir in subdirs:
        (session_dir / subdir).mkdir(exist_ok=True)
    
    return session_dir

def load_csv_robust(path, logger=None):
    """🆕 Enhanced CSV loader with comprehensive fallback strategies"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    logger.info(f"🔍 Loading CSV with robust detection: {path}")
    
    # Multiple loading strategies with different encodings and separators
    loading_strategies = [
        {'encoding': 'utf-8', 'sep': ','},
        {'encoding': 'latin-1', 'sep': ','},
        {'encoding': 'utf-8', 'sep': ';'},
        {'encoding': 'latin-1', 'sep': ';'},
        {'encoding': 'utf-8', 'sep': '\t'},
        {'encoding': 'utf-8', 'sep': None, 'engine': 'python'},  # Auto-detect
        {'encoding': 'cp1252', 'sep': ','},  # Windows encoding
        {'encoding': 'iso-8859-1', 'sep': ','}  # Alternative encoding
    ]
    
    df = None
    used_strategy = None
    
    for i, strategy in enumerate(loading_strategies):
        try:
            if strategy.get('sep') is None:
                # Use pandas' automatic separator detection
                df = pd.read_csv(path, encoding=strategy['encoding'], 
                               engine=strategy.get('engine', 'c'))
            else:
                df = pd.read_csv(path, encoding=strategy['encoding'], 
                               sep=strategy['sep'], engine=strategy.get('engine', 'c'))
            
            # Validate that we got meaningful data
            if len(df.columns) > 0 and len(df) > 0:
                used_strategy = strategy
                logger.info(f"✅ Successfully loaded with strategy {i+1}: {strategy}")
                break
                
        except Exception as e:
            logger.debug(f"Strategy {i+1} failed: {str(e)}")
            continue
    
    if df is None:
        raise ValueError(f"Failed to load CSV with any strategy: {path}")
    
    logger.info(f"📊 Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def smart_column_detection(df, logger=None):
    """🆕 Enhanced smart column detection with confidence scoring and fallbacks"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("🧠 Starting intelligent column detection...")
    
    # Clean column names for analysis
    df_clean = df.copy()
    original_columns = list(df.columns)
    df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(' ', '_')
    clean_to_original = dict(zip(df_clean.columns, original_columns))
    
    detection_results = {
        'text_column': None,
        'text_confidence': 0,
        'label_column': None,
        'label_confidence': 0,
        'text_candidates': [],
        'label_candidates': [],
        'analysis': {}
    }
    
    # Text column candidates with priority scores
    text_candidates = {
        # High priority - exact matches
        'text': 100, 'review': 95, 'comment': 90, 'content': 85,
        'description': 80, 'message': 75, 'body': 70, 'post': 65,
        
        # Medium priority - partial matches
        'review_text': 60, 'comment_text': 55, 'user_review': 50,
        'product_review': 45, 'feedback': 40, 'opinion': 35,
        
        # Lower priority - broader terms
        'title': 30, 'summary': 25, 'abstract': 20, 'notes': 15
    }
    
    # Label column candidates with priority scores
    label_candidates = {
        # High priority - exact matches
        'label': 100, 'sentiment': 95, 'class': 90, 'target': 85,
        'rating': 80, 'score': 75, 'category': 70,
        
        # Medium priority - variations
        'sentiment_label': 65, 'class_label': 60, 'star_rating': 55,
        'review_score': 50, 'polarity': 45,
        
        # Lower priority
        'outcome': 40, 'result': 35, 'type': 30, 'group': 25
    }
    
    # Analyze text columns
    text_scores = []
    for clean_col in df_clean.columns:
        original_col = clean_to_original[clean_col]
        score = 0
        
        # Exact name matching
        for candidate, candidate_score in text_candidates.items():
            if clean_col == candidate:
                score = candidate_score
                break
            elif candidate in clean_col:
                score = candidate_score * 0.7  # Partial match penalty
        
        # Content-based scoring if no name match
        if score == 0:
            try:
                sample_data = df_clean[clean_col].dropna().astype(str).head(100)
                if len(sample_data) > 0:
                    # Check text characteristics
                    avg_length = sample_data.str.len().mean()
                    word_count = sample_data.str.split().str.len().mean()
                    
                    # Text quality indicators
                    if avg_length > 50 and word_count > 5:
                        score = 25  # Reasonable text content
                    elif avg_length > 20 and word_count > 3:
                        score = 15  # Minimal text content
                    elif avg_length > 10:
                        score = 5   # Very minimal text
                    
                    # Bonus for text-like patterns
                    text_sample = ' '.join(sample_data.head(10))
                    common_words = ['the', 'and', 'is', 'was', 'good', 'bad', 'love', 'hate', 'great', 'nice']
                    word_matches = sum(1 for word in common_words if word in text_sample.lower())
                    if word_matches > 2:
                        score += 10  # Contains common English words
                    
                    logger.debug(f"   📊 {original_col}: avg_len={avg_length:.1f}, avg_words={word_count:.1f}, score={score}")
            except Exception as e:
                logger.warning(f"   ⚠️ Error analyzing {original_col}: {e}")
                score = 0
        
        if score > 0:
            text_scores.append((original_col, clean_col, score))
    
    # Sort text candidates by score
    text_scores.sort(key=lambda x: x[2], reverse=True)
    detection_results['text_candidates'] = [(col, score) for col, _, score in text_scores[:5]]
    
    # Analyze label columns
    label_scores = []
    for clean_col in df_clean.columns:
        original_col = clean_to_original[clean_col]
        score = 0
        
        # Exact name matching
        for candidate, candidate_score in label_candidates.items():
            if clean_col == candidate:
                score = candidate_score
                break
            elif candidate in clean_col:
                score = candidate_score * 0.7
        
        # Content-based scoring for labels
        if score == 0:
            try:
                sample_data = df_clean[clean_col].dropna()
                if len(sample_data) > 0:
                    unique_values = sample_data.unique()
                    n_unique = len(unique_values)
                    
                    # Check for typical label patterns
                    str_values = [str(v).lower() for v in unique_values]
                    
                    # Binary classification indicators
                    if set(str_values).intersection({'0', '1', 'true', 'false', 'yes', 'no'}):
                        score = 20
                    elif set(str_values).intersection({'positive', 'negative', 'pos', 'neg'}):
                        score = 30
                    elif set(str_values).intersection({'good', 'bad', 'neutral'}):
                        score = 25
                    elif 2 <= n_unique <= 10:  # Reasonable number of classes
                        score = 15
                    elif n_unique == 1:
                        score = 1  # Single value, probably not useful
                    
                    logger.debug(f"   🏷️ {original_col}: unique={n_unique}, values={str_values[:5]}, score={score}")
            except Exception as e:
                logger.warning(f"   ⚠️ Error analyzing {original_col}: {e}")
                score = 0
        
        if score > 0:
            label_scores.append((original_col, clean_col, score))
    
    # Sort label candidates by score
    label_scores.sort(key=lambda x: x[2], reverse=True)
    detection_results['label_candidates'] = [(col, score) for col, _, score in label_scores[:5]]
    
    # Select best candidates
    if text_scores:
        best_text = text_scores[0]
        detection_results['text_column'] = best_text[0]
        detection_results['text_confidence'] = best_text[2]
    
    if label_scores:
        best_label = label_scores[0]
        detection_results['label_column'] = best_label[0]
        detection_results['label_confidence'] = best_label[2]
    
    # Fallback strategies for text column
    if detection_results['text_confidence'] < 10:
        logger.warning("⚠️ No clear text column found, using fallback strategies...")
        
        # Strategy 1: Use first string column with reasonable content
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    sample = df[col].dropna().astype(str).head(50)
                    if len(sample) > 0:
                        avg_len = sample.str.len().mean()
                        if avg_len > 10:
                            detection_results['text_column'] = col
                            detection_results['text_confidence'] = 5
                            logger.info(f"🔄 Fallback: Using {col} (avg_len={avg_len:.1f})")
                            break
            except Exception:
                continue
        
        # Strategy 2: Use column with most total content
        if detection_results['text_confidence'] < 5:
            max_content_col = None
            max_content_score = 0
            
            for col in df.columns:
                try:
                    content_score = df[col].astype(str).str.len().sum()
                    if content_score > max_content_score:
                        max_content_score = content_score
                        max_content_col = col
                except Exception:
                    continue
            
            if max_content_col:
                detection_results['text_column'] = max_content_col
                detection_results['text_confidence'] = 1
                logger.info(f"🔄 Last resort: Using {max_content_col} (total_content={max_content_score})")
    
    # Log final results
    logger.info("🎯 Column detection results:")
    logger.info(f"   📄 Text column: {detection_results['text_column']} (confidence: {detection_results['text_confidence']})")
    logger.info(f"   🏷️ Label column: {detection_results['label_column']} (confidence: {detection_results['label_confidence']})")
    
    return detection_results

def validate_and_process_csv(csv_path: str, force_text_column: str = None, 
                           force_label_column: str = None, logger=None) -> Dict[str, Any]:
    """🆕 Comprehensive CSV validation and processing with intelligent fallbacks"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        logger.info("=" * 50)
        logger.info("🆕 ENHANCED CSV VALIDATION & PROCESSING")
        logger.info("=" * 50)
        logger.info(f"📄 Input file: {csv_path}")
        
        # Load CSV
        df = load_csv_robust(csv_path, logger)
        original_shape = df.shape
        
        # Column detection
        if force_text_column or force_label_column:
            logger.info("🔧 Using forced column assignments:")
            detection = {
                'text_column': force_text_column,
                'text_confidence': 100 if force_text_column else 0,
                'label_column': force_label_column,
                'label_confidence': 100 if force_label_column else 0
            }
            if force_text_column:
                logger.info(f"   📄 Forced text column: {force_text_column}")
            if force_label_column:
                logger.info(f"   🏷️ Forced label column: {force_label_column}")
        else:
            detection = smart_column_detection(df, logger)
        
        # Validate detected columns
        if not detection['text_column'] or detection['text_column'] not in df.columns:
            available_cols = list(df.columns)
            raise ValueError(f"No valid text column found. Available columns: {available_cols}")
        
        # Create processed dataset
        processed_data = {
            'success': True,
            'original_file': str(csv_path),
            'original_shape': original_shape,
            'text_column_used': detection['text_column'],
            'label_column_used': detection['label_column'],
            'has_labels': False,
            'processed_df': None,
            'stats': {},
            'warnings': []
        }
        
        # Extract text data
        text_col = detection['text_column']
        text_data = df[text_col].fillna('').astype(str)
        
        # Clean text data
        cleaned_texts = []
        for text in text_data:
            cleaned = clean_text_enhanced(text)
            cleaned_texts.append(cleaned)
        
        # Filter out empty texts
        valid_indices = [i for i, text in enumerate(cleaned_texts) if len(text.strip()) > 3]
        valid_texts = [cleaned_texts[i] for i in valid_indices]
        
        logger.info(f"📊 Text processing: {len(valid_texts)}/{len(cleaned_texts)} valid texts")
        
        if len(valid_texts) == 0:
            raise ValueError("No valid text data found after cleaning")
        
        # Handle labels
        labels = []
        if detection['label_column'] and detection['label_column'] in df.columns:
            label_col = detection['label_column']
            all_labels = df[label_col].tolist()
            
            # Filter labels to match valid texts
            labels = [all_labels[i] for i in valid_indices]
            
            # Normalize labels
            normalized_labels = []
            for label in labels:
                normalized = normalize_label(label)
                normalized_labels.append(normalized)
            
            # Check if we have valid labels
            valid_label_count = sum(1 for l in normalized_labels if l in [0, 1])
            
            if valid_label_count > 0:
                processed_data['has_labels'] = True
                labels = normalized_labels
                logger.info(f"🏷️ Found {valid_label_count} valid labels out of {len(labels)}")
                
                # Label distribution
                label_dist = Counter(l for l in labels if l in [0, 1])
                logger.info(f"📊 Label distribution: {dict(label_dist)}")
            else:
                logger.warning("⚠️ No valid labels found, treating as inference data")
                labels = [-1] * len(valid_texts)
        else:
            logger.info("🔍 No label column found, creating placeholder labels")
            labels = [-1] * len(valid_texts)
        
        # Create final DataFrame
        final_df = pd.DataFrame({
            'text': valid_texts,
            'label': labels
        })
        
        processed_data['processed_df'] = final_df
        processed_data['final_shape'] = final_df.shape
        
        # Calculate statistics
        text_lengths = final_df['text'].str.len()
        word_counts = final_df['text'].str.split().str.len()
        
        processed_data['stats'] = {
            'total_samples': len(final_df),
            'has_labels': processed_data['has_labels'],
            'valid_label_count': sum(1 for l in labels if l in [0, 1]),
            'text_stats': {
                'avg_length': float(text_lengths.mean()),
                'median_length': float(text_lengths.median()),
                'avg_words': float(word_counts.mean()),
                'median_words': float(word_counts.median()),
                'min_length': int(text_lengths.min()),
                'max_length': int(text_lengths.max())
            },
            'label_distribution': dict(Counter(labels)) if labels else {}
        }
        
        # Quality warnings
        if len(valid_texts) < original_shape[0] * 0.8:
            processed_data['warnings'].append("More than 20% of texts were filtered out")
        
        if processed_data['has_labels'] and processed_data['stats']['valid_label_count'] < len(labels) * 0.5:
            processed_data['warnings'].append("Less than 50% of labels are valid for binary classification")
        
        logger.info("✅ CSV processing completed successfully!")
        logger.info(f"   📊 Final dataset: {final_df.shape[0]} samples")
        logger.info(f"   🏷️ Has labels: {processed_data['has_labels']}")
        logger.info(f"   📏 Avg text length: {processed_data['stats']['text_stats']['avg_length']:.1f}")
        
        return processed_data
        
    except Exception as e:
        logger.error(f"❌ CSV processing failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'original_file': str(csv_path)
        }

def clean_text_enhanced(text: str) -> str:
    """🆕 Enhanced text cleaning optimized for sentiment analysis"""
    if not isinstance(text, str):
        text = str(text)
    
    # Handle null/empty indicators
    if text.lower().strip() in ['null', 'none', 'nan', 'n/a', '', ' ']:
        return ""
    
    # Fix encoding issues
    try:
        text = text.encode('utf-8').decode('utf-8')
    except:
        pass
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    
    # Clean HTML/XML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean URLs but preserve context
    text = re.sub(r'http[s]?://\S+', '[URL]', text)
    text = re.sub(r'www\.\S+', '[URL]', text)
    
    # Clean email addresses
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
    
    # Normalize punctuation (preserve sentiment indicators)
    text = re.sub(r'[.]{2,}', '...', text)
    text = re.sub(r'[!]{2,}', '!!', text)
    text = re.sub(r'[?]{2,}', '??', text)
    
    # Remove excessive special characters but preserve sentiment
    text = re.sub(r'[^\w\s.,!?:;\'"()\-\[\]]', ' ', text)
    
    # Fix common text artifacts
    text = re.sub(r'\b[A-Z]{3,}\b', lambda m: m.group().lower(), text)  # ALL CAPS
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)  # Repeated characters
    
    # Final cleanup
    text = ' '.join(text.split())
    text = text.strip()
    
    return text

def normalize_label(label) -> int:
    """🆕 Enhanced label normalization for binary classification"""
    if pd.isna(label):
        return -1
    
    label_str = str(label).lower().strip()
    
    # Binary text labels
    if label_str in ['positive', 'pos', '1', 'true', 'good', 'like', 'love', 'great']:
        return 1
    elif label_str in ['negative', 'neg', '0', 'false', 'bad', 'dislike', 'hate', 'terrible']:
        return 0
    elif label_str in ['neutral', 'mixed', '0.5']:
        return 0  # Treat neutral as negative for binary classification
    
    # Numeric labels
    try:
        num_val = float(label)
        if num_val > 0.5:
            return 1
        elif num_val <= 0.5:
            return 0
        else:
            return -1
    except:
        return -1

def run_subprocess_with_timeout(command: List[str], timeout: int = 300, 
                               cwd: Optional[Path] = None,
                               stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """🆕 Enhanced subprocess execution with real-time streaming and timeout"""
    try:
        start_time = time.time()
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            bufsize=1,
            universal_newlines=True
        )
        
        stdout_lines = []
        stderr_lines = []
        
        # Real-time output streaming
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                stdout_lines.append(output)
                if stream_callback:
                    stream_callback(output.strip())
        
        # Get any remaining stderr
        stderr_output = process.stderr.read()
        if stderr_output:
            stderr_lines.append(stderr_output)
            if stream_callback:
                stream_callback(stderr_output.strip())
        
        # Wait for process to complete
        return_code = process.wait(timeout=timeout)
        end_time = time.time()
        
        return {
            "success": return_code == 0,
            "returncode": return_code,
            "stdout": "".join(stdout_lines),
            "stderr": "".join(stderr_lines),
            "duration": end_time - start_time,
            "command": " ".join(command)
        }
        
    except subprocess.TimeoutExpired:
        process.kill()
        return {
            "success": False,
            "returncode": -1,
            "stdout": "".join(stdout_lines),
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
    """Convert objects to JSON-serializable format with enhanced handling"""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: safe_convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_convert_for_json(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)

def auto_embed_and_predict(file_path: str = None, csv_path: str = None, 
                         session_dir: Optional[Path] = None, fast_mode: bool = True,
                         force_text_column: str = None, force_label_column: str = None,
                         log_callback: Optional[Callable[[str], None]] = None, 
                         **kwargs) -> Dict[str, Any]:
    """
    🆕 ENHANCED: Complete automated pipeline for sentiment analysis with intelligent processing.
    
    Args:
        file_path: Path to input CSV file (preferred parameter name)
        csv_path: Alternative parameter name for input CSV file (backward compatibility)
        session_dir: Optional session directory. If None, creates timestamped directory
        fast_mode: Whether to use fast mode (passed to training scripts)
        force_text_column: Force specific text column name
        force_label_column: Force specific label column name
        log_callback: Optional callback that receives real-time output lines
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary with comprehensive results and status information
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Handle parameter flexibility
    input_file = file_path or csv_path
    if not input_file:
        raise ValueError("Either file_path or csv_path must be provided")
    
    input_file = Path(input_file)
    
    # Initialize results dictionary
    results = {
        "status": "started",
        "pipeline_type": "enhanced_automated",
        "steps": {},
        "errors": [],
        "warnings": [],
        "total_duration": 0,
        "session_dir": None,
        "session_directory": None,
        "overall_success": False,
        "has_labels": False,
        "inference_only": False,
        "final_results": {}
    }
    
    start_time = time.time()
    
    def log_message(message: str):
        """Internal logging with optional callback"""
        logger.info(message)
        if log_callback:
            log_callback(message)
    
    try:
        log_message("🚀 Starting Enhanced Automated Sentiment Analysis Pipeline")
        log_message("=" * 60)
        
        # Setup paths
        project_root = setup_robust_paths()
        
        # Validate input file
        if not input_file.exists():
            raise FileNotFoundError(f"Input CSV file not found: {input_file}")
        
        log_message(f"📄 Input file: {input_file}")
        log_message(f"📁 Project root: {project_root}")
        
        # Create or use session directory
        if session_dir is None:
            results_dir = project_root / "results"
            results_dir.mkdir(exist_ok=True)
            session_dir = create_timestamped_session_dir(results_dir, "enhanced_analysis")
        
        session_dir = Path(session_dir)
        results["session_dir"] = str(session_dir)
        results["session_directory"] = str(session_dir)
        
        log_message(f"📁 Session directory: {session_dir}")
        
        # Setup session logging
        log_file = session_dir / "logs" / "pipeline.log"
        log_file.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Change to project root for script execution
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        # Step 1: Enhanced CSV Processing
        log_message("🔄 Step 1: Enhanced CSV Processing & Validation")
        
        processing_result = validate_and_process_csv(
            str(input_file), 
            force_text_column=force_text_column,
            force_label_column=force_label_column,
            logger=logger
        )
        
        if not processing_result['success']:
            raise Exception(f"CSV processing failed: {processing_result.get('error', 'Unknown error')}")
        
        results['has_labels'] = processing_result['has_labels']
        results['inference_only'] = not processing_result['has_labels']
        
        # Save processed data
        processed_df = processing_result['processed_df']
        
        # Determine data splits based on size and labels
        total_samples = len(processed_df)
        has_labels = processing_result['has_labels']
        
        log_message(f"📊 Dataset info: {total_samples} samples, labels={has_labels}")
        
        if total_samples < 10:
            # Very small dataset - create single inference file
            processed_df.to_csv(session_dir / "processed" / "inference.csv", index=False)
            results['inference_only'] = True
            log_message("📋 Created inference.csv (dataset too small for splits)")
            
        elif not has_labels:
            # No labels - create inference file
            processed_df.to_csv(session_dir / "processed" / "inference.csv", index=False)
            results['inference_only'] = True
            log_message("📋 Created inference.csv (no labels found)")
            
        else:
            # Create train/val/test splits
            from sklearn.model_selection import train_test_split
            
            if total_samples >= 100:
                # Large enough for full splits
                train_df, temp_df = train_test_split(processed_df, test_size=0.3, random_state=42, 
                                                   stratify=processed_df['label'] if has_labels else None)
                val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, 
                                                 stratify=temp_df['label'] if has_labels else None)
            else:
                # Small dataset - simple split
                train_df, test_df = train_test_split(processed_df, test_size=0.3, random_state=42, 
                                                   stratify=processed_df['label'] if has_labels else None)
                val_df = test_df.copy()  # Use test as validation for small datasets
            
            # Save splits
            train_df.to_csv(session_dir / "processed" / "train.csv", index=False)
            val_df.to_csv(session_dir / "processed" / "val.csv", index=False)
            test_df.to_csv(session_dir / "processed" / "test.csv", index=False)
            
            log_message(f"📋 Created splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        results["steps"]["csv_processing"] = {
            "status": "completed",
            "duration": time.time() - start_time,
            "samples_processed": total_samples,
            "has_labels": has_labels,
            "inference_only": results['inference_only']
        }
        
        # Step 2: Embedding Generation
        log_message("🔄 Step 2: Enhanced Embedding Generation")
        embed_start = time.time()
        
        embed_cmd = [
            sys.executable, "scripts/embed_dataset.py",
            "--input-dir", str(session_dir / "processed"),
            "--output-dir", str(session_dir / "embeddings"),
            "--force-recreate"
        ]
        
        if results['inference_only']:
            embed_cmd.append("--force-inference")
        
        embed_result = run_subprocess_with_timeout(
            embed_cmd, timeout=600, cwd=project_root, stream_callback=log_callback
        )
        
        results["steps"]["embedding"] = {
            "status": "completed" if embed_result["success"] else "failed",
            "duration": embed_result["duration"],
            "returncode": embed_result["returncode"]
        }
        
        if not embed_result["success"]:
            results["errors"].append(f"Embedding generation failed: {embed_result['stderr']}")
            log_message(f"❌ Embedding generation failed: {embed_result['stderr']}")
        else:
            log_message("✅ Embedding generation completed")
        
        # Step 3: Model Training (only if we have labels)
        if not results['inference_only'] and embed_result["success"]:
            # MLP Training
            log_message("🔄 Step 3a: Enhanced MLP Training")
            
            mlp_cmd = [
                sys.executable, "scripts/train_mlp.py",
                "--embeddings-dir", str(session_dir / "embeddings"),
                "--output-dir", str(session_dir),
                "--skip-if-insufficient"
            ]
            
            if fast_mode:
                mlp_cmd.extend(["--epochs", "20"])
            
            mlp_result = run_subprocess_with_timeout(
                mlp_cmd, timeout=900, cwd=project_root, stream_callback=log_callback
            )
            
            results["steps"]["mlp_training"] = {
                "status": "completed" if mlp_result["success"] else "failed",
                "duration": mlp_result["duration"],
                "returncode": mlp_result["returncode"]
            }
            
            if mlp_result["success"]:
                log_message("✅ MLP training completed")
            else:
                results["errors"].append(f"MLP training failed: {mlp_result['stderr']}")
                log_message(f"⚠️ MLP training failed: {mlp_result['stderr']}")
            
            # SVM Training
            log_message("🔄 Step 3b: Enhanced SVM Training")
            
            svm_cmd = [
                sys.executable, "scripts/train_svm.py",
                "--embeddings-dir", str(session_dir / "embeddings"),
                "--output-dir", str(session_dir),
                "--skip-if-insufficient"
            ]
            
            if fast_mode:
                svm_cmd.append("--fast")
            
            svm_result = run_subprocess_with_timeout(
                svm_cmd, timeout=600, cwd=project_root, stream_callback=log_callback
            )
            
            results["steps"]["svm_training"] = {
                "status": "completed" if svm_result["success"] else "failed",
                "duration": svm_result["duration"],
                "returncode": svm_result["returncode"]
            }
            
            if svm_result["success"]:
                log_message("✅ SVM training completed")
            else:
                results["errors"].append(f"SVM training failed: {svm_result['stderr']}")
                log_message(f"⚠️ SVM training failed: {svm_result['stderr']}")
        else:
            # Mark training steps as skipped
            results["steps"]["mlp_training"] = {"status": "skipped", "reason": "inference_only"}
            results["steps"]["svm_training"] = {"status": "skipped", "reason": "inference_only"}
            log_message("🔍 Training skipped (inference mode or embedding failed)")
        
        # Step 4: Report Generation
        log_message("🔄 Step 4: Enhanced Report Generation")
        
        report_cmd = [
            sys.executable, "scripts/report.py",
            "--models-dir", str(session_dir / "models"),
            "--results-dir", str(session_dir),
            "--auto-default"
        ]
        
        report_result = run_subprocess_with_timeout(
            report_cmd, timeout=300, cwd=project_root, stream_callback=log_callback
        )
        
        results["steps"]["report"] = {
            "status": "completed" if report_result["success"] else "failed",
            "duration": report_result["duration"],
            "returncode": report_result["returncode"]
        }
        
        if report_result["success"]:
            log_message("✅ Report generation completed")
        else:
            results["errors"].append(f"Report generation failed: {report_result['stderr']}")
            log_message(f"⚠️ Report generation failed: {report_result['stderr']}")
        
        # Restore original working directory
        os.chdir(original_cwd)
        
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
            log_message(f"⚠️ Error collecting final results: {e}")
        
        # Calculate total duration and determine overall status
        results["total_duration"] = time.time() - start_time
        
        # Determine overall success
        completed_steps = sum(1 for step_info in results["steps"].values() 
                            if step_info.get("status") == "completed")
        total_steps = len([s for s in results["steps"].values() if s.get("status") != "skipped"])
        
        if completed_steps == total_steps and total_steps > 0:
            results["status"] = "success"
            results["overall_success"] = True
        elif completed_steps > 0:
            results["status"] = "partial_success"
            results["overall_success"] = True
        else:
            results["status"] = "failed"
            results["overall_success"] = False
        
        # Create pipeline summary
        summary_file = session_dir / "enhanced_pipeline_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(safe_convert_for_json(results), f, indent=2, ensure_ascii=False)
        
        results["summary_file"] = str(summary_file)
        
        log_message("=" * 60)
        log_message(f"🎉 Enhanced Pipeline completed with status: {results['status'].upper()}")
        log_message(f"⏱️ Total duration: {results['total_duration']:.2f} seconds")
        log_message(f"📁 Session directory: {session_dir}")
        log_message(f"📊 Inference mode: {results['inference_only']}")
        log_message(f"✅ Completed steps: {completed_steps}/{total_steps}")
        
        if results["errors"]:
            log_message(f"⚠️ Errors encountered: {len(results['errors'])}")
        
        return safe_convert_for_json(results)
        
    except Exception as e:
        results["status"] = "error"
        results["errors"].append(str(e))
        results["total_duration"] = time.time() - start_time
        logger.error(f"❌ Enhanced pipeline failed with error: {e}")
        
        # Restore working directory even on error
        try:
            os.chdir(original_cwd)
        except:
            pass
        
        return safe_convert_for_json(results)

# Legacy compatibility functions
def run_dataset_analysis(csv_path: str) -> Dict[str, Any]:
    """Legacy wrapper for dataset analysis"""
    try:
        result = validate_and_process_csv(csv_path)
        if result['success']:
            return {
                'success': True,
                'file_info': {
                    'path': result['original_file'],
                    'samples': result['stats']['total_samples'],
                    'has_labels': result['has_labels']
                },
                'data_analysis': result['stats'],
                'text_analysis': result['stats']['text_stats'],
                'sentiment_analysis': {
                    'has_labels': result['has_labels'],
                    'label_distribution': result['stats']['label_distribution']
                } if result['has_labels'] else {'has_labels': False}
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
    """Legacy wrapper for complete CSV analysis"""
    return auto_embed_and_predict(
        file_path=csv_path,
        force_text_column=text_column if text_column != 'text' else None,
        force_label_column=label_column if label_column != 'label' else None,
        log_callback=log_callback,
        fast_mode=True
    )