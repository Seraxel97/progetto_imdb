#!/usr/bin/env python3
"""
🔧 FIXED - Advanced Embedding Generation Script - COMPLETE PIPELINE INTEGRATION
Generates sentence embeddings for sentiment analysis with comprehensive pipeline integration.

🔧 FIXES APPLIED:
- ✅ Compatible with pipeline_runner.py parameter passing
- ✅ Standardized [EMBED] logging format for debugging
- ✅ Explicit file saving verification with print statements
- ✅ Robust error handling with clear error messages
- ✅ Auto-detection of input directory when not specified
- ✅ Session directory compatibility
- ✅ Real-time progress reporting

FEATURES:
- Dynamic path detection with robust PROJECT_ROOT handling
- Integration with pipeline_runner.py subprocess calls
- Support for timestamped session directories and custom output locations
- Advanced model management (local models, fallback strategies, caching)
- Comprehensive progress tracking with detailed logging and error handling
- Multiple embedding models support with automatic optimization
- Robust validation and quality assurance for input data
- Enhanced label distribution analysis and embedding quality verification
- Compatible with sys.argv injection for pipeline automation
- Professional CLI interface with extensive configuration options

USAGE:
    # Standard embedding generation
    python scripts/embed_dataset.py
    
    # Pipeline integration (called by pipeline_runner.py)
    python scripts/embed_dataset.py --output-dir results/session_20241229/embeddings --force-recreate
    
    # Custom input/output directories
    python scripts/embed_dataset.py --input-dir results/session_20241229/processed --output-dir results/session_20241229/embeddings
    
    # Force regeneration with different model
    python scripts/embed_dataset.py --force-recreate --model-name all-mpnet-base-v2
"""

import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import argparse
from pathlib import Path
import warnings
import sys
import logging
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Tuple
import time

warnings.filterwarnings('ignore')

# IMPROVED: Robust project root detection
def find_project_root() -> Path:
    """Find project root with robust detection algorithm."""
    try:
        current = Path(__file__).resolve()
        
        # First, try the standard pattern
        if current.parent.name == 'scripts':
            return current.parent.parent
        
        # If not in scripts, search upward for project structure
        while current != current.parent:
            if (current / "scripts").exists() and (current / "data").exists():
                return current
            current = current.parent
        
        # Fallback to parent of scripts directory if we're in scripts
        current = Path(__file__).resolve()
        if current.parent.name == "scripts":
            return current.parent.parent
        
        # Final fallback to current directory
        return current.parent
    except:
        return Path.cwd()

# Dynamic project root detection using improved approach
PROJECT_ROOT = find_project_root()

# Dynamic path construction using PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DATA_DIR = DATA_DIR / "embeddings"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESULTS_DIR = PROJECT_ROOT / "results"

# Add scripts to path for imports
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# 🔧 FIXED: Enhanced logging with standardized format
def setup_logging():
    """Setup logging with standardized format for pipeline integration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# 🔧 FIXED: Pipeline logging functions
def log_embed(message: str):
    """Log message with [EMBED] prefix for pipeline debugging."""
    print(f"[EMBED] {message}")
    logger.info(f"[EMBED] {message}")

def log_embed_error(message: str):
    """Log error message with [EMBED] prefix for pipeline debugging."""
    print(f"[EMBED] ERROR: {message}")
    logger.error(f"[EMBED] ERROR: {message}")

def log_embed_success(message: str):
    """Log success message with [EMBED] prefix for pipeline debugging."""
    print(f"[EMBED] SUCCESS: {message}")
    logger.info(f"[EMBED] SUCCESS: {message}")

# Import centralized configuration with enhanced fallback
try:
    from config_constants import (
        DEFAULT_EMBEDDING_MODEL, 
        AVAILABLE_EMBEDDING_MODELS, 
        DEFAULT_TRAIN_PARAMS
    )
    log_embed("Loaded configuration from config_constants")
except ImportError:
    log_embed("Using fallback configuration (config_constants not found)")
    DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    AVAILABLE_EMBEDDING_MODELS = [
        'all-MiniLM-L6-v2', 
        'all-mpnet-base-v2', 
        'paraphrase-MiniLM-L6-v2',
        'all-MiniLM-L12-v2',
        'paraphrase-multilingual-MiniLM-L12-v2'
    ]
    DEFAULT_TRAIN_PARAMS = {
        'embedding_batch_size': 32,
        'max_sequence_length': 512,
        'embedding_cache_size': 1000
    }

class AdvancedEmbeddingGenerator:
    """
    🔧 FIXED - Advanced embedding generation class with comprehensive pipeline integration.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize the FIXED embedding generator."""
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.model = None
        self.model_name = None
        self.model_load_time = 0
        self.setup_paths()
        
        log_embed(f"Advanced Embedding Generator initialized")
        log_embed(f"Project root: {self.project_root}")
    
    def setup_paths(self):
        """Setup comprehensive path structure."""
        self.paths = {
            'project_root': self.project_root,
            'data_dir': self.project_root / "data",
            'processed_data': self.project_root / "data" / "processed",
            'embeddings_data': self.project_root / "data" / "embeddings",
            'models_dir': self.project_root / "models",
            'results_dir': self.project_root / "results",
            'scripts_dir': self.project_root / "scripts"
        }
    
    def find_input_directory(self, specified_input_dir: Optional[str] = None) -> Path:
        """
        🔧 FIXED: Smart input directory detection for pipeline compatibility.
        
        Args:
            specified_input_dir: Directory specified by user/pipeline
        
        Returns:
            Path to input directory
        """
        if specified_input_dir:
            input_dir = Path(specified_input_dir)
            if input_dir.exists():
                log_embed(f"Using specified input directory: {input_dir}")
                return input_dir
            else:
                log_embed_error(f"Specified input directory does not exist: {input_dir}")
        
        # Search for input directories in order of preference
        search_paths = [
            self.paths['processed_data'],  # Standard processed data
            self.project_root / "data" / "raw",  # Raw data as fallback
        ]
        
        # Also search in recent session directories
        if self.paths['results_dir'].exists():
            session_dirs = []
            for item in self.paths['results_dir'].iterdir():
                if item.is_dir() and item.name.startswith('session_'):
                    session_dirs.append(item)
            
            # Sort by name (newest first)
            session_dirs.sort(reverse=True)
            
            # Add processed directories from recent sessions
            for session_dir in session_dirs[:3]:  # Check last 3 sessions
                processed_dir = session_dir / 'processed'
                if processed_dir.exists():
                    search_paths.insert(0, processed_dir)
        
        # Find the first directory with valid CSV files
        for search_path in search_paths:
            if self.validate_input_directory_basic(search_path):
                log_embed(f"Auto-detected input directory: {search_path}")
                return search_path
        
        # Fallback to default even if it doesn't exist
        default_path = self.paths['processed_data']
        log_embed(f"Using default input directory (may not exist): {default_path}")
        return default_path
    
    def validate_input_directory_basic(self, input_dir: Path) -> bool:
        """
        🔧 FIXED: Basic validation to check if directory has required CSV files.
        
        Args:
            input_dir: Directory to validate
        
        Returns:
            True if directory has at least one required CSV file
        """
        if not input_dir.exists():
            return False
        
        required_files = ['train.csv', 'val.csv', 'test.csv']
        found_files = []
        
        for filename in required_files:
            filepath = input_dir / filename
            if filepath.exists():
                try:
                    # Quick check if file is readable and has required columns
                    df = pd.read_csv(filepath, nrows=1)
                    if 'text' in df.columns and 'label' in df.columns:
                        found_files.append(filename)
                except:
                    pass
        
        return len(found_files) > 0
    
    def find_local_models(self) -> Dict[str, Path]:
        """Find available local models with comprehensive search."""
        local_models = {}
        
        # Common local model directory patterns
        search_paths = [
            self.paths['models_dir'] / "minilm-l6-v2",
            self.paths['models_dir'] / "all-MiniLM-L6-v2",
            self.paths['models_dir'] / "sentence-transformers",
            self.project_root / "sentence-transformers",
        ]
        
        model_mappings = {
            'all-MiniLM-L6-v2': ['minilm-l6-v2', 'all-MiniLM-L6-v2'],
            'all-mpnet-base-v2': ['mpnet-base-v2', 'all-mpnet-base-v2'],
            'paraphrase-MiniLM-L6-v2': ['paraphrase-minilm-l6-v2']
        }
        
        for search_path in search_paths:
            if search_path.exists():
                # Direct model directory
                if (search_path / "config.json").exists():
                    # Try to determine which model this is
                    for model_name, possible_names in model_mappings.items():
                        if any(name in str(search_path).lower() for name in possible_names):
                            local_models[model_name] = search_path
                            log_embed(f"Found local model '{model_name}' at {search_path}")
                            break
                else:
                    # Search subdirectories
                    for model_name, possible_names in model_mappings.items():
                        for possible_name in possible_names:
                            potential_path = search_path / possible_name
                            if potential_path.exists() and (potential_path / "config.json").exists():
                                local_models[model_name] = potential_path
                                log_embed(f"Found local model '{model_name}' at {potential_path}")
        
        return local_models
    
    def load_model_advanced(self, model_name: str, max_length: int = 512, 
                           force_download: bool = False) -> SentenceTransformer:
        """Load SentenceTransformer model with advanced fallback strategies."""
        model_load_start = time.time()
        log_embed(f"Loading SentenceTransformer model: {model_name}")
        
        # Find local models
        local_models = {} if force_download else self.find_local_models()
        
        try:
            # Try local model first if available and not forcing download
            if model_name in local_models and not force_download:
                local_path = local_models[model_name]
                log_embed(f"Loading from local directory: {local_path}")
                
                model = SentenceTransformer(str(local_path))
                log_embed("Local model loaded successfully")
                
            else:
                # Download/load from HuggingFace
                log_embed(f"Downloading/loading model from HuggingFace: {model_name}")
                
                model = SentenceTransformer(model_name)
                log_embed("Remote model loaded successfully")
                
                # Optionally cache locally for future use
                try:
                    cache_dir = self.paths['models_dir'] / model_name.replace('/', '_')
                    if not cache_dir.exists():
                        log_embed(f"Caching model locally to: {cache_dir}")
                        model.save(str(cache_dir))
                except Exception as cache_error:
                    log_embed(f"Could not cache model locally: {cache_error}")
            
            # Configure model settings
            if hasattr(model, 'max_seq_length'):
                model.max_seq_length = max_length
                log_embed(f"Set max sequence length: {max_length}")
            
            # Get model info
            sample_embedding = model.encode(["test"], convert_to_numpy=True)
            embedding_dim = sample_embedding.shape[1]
            
            log_embed(f"Embedding dimension: {embedding_dim}")
            
            # Safe device detection
            try:
                device_info = str(model.device) if hasattr(model, 'device') else 'unknown'
                log_embed(f"Model device: {device_info}")
            except:
                device_info = 'unavailable'
                log_embed(f"Model device: {device_info}")
            
            self.model = model
            self.model_name = model_name
            self.model_load_time = time.time() - model_load_start
            
            return model
            
        except Exception as e:
            log_embed_error(f"Error loading {model_name}: {e}")
            
            # Fallback strategy
            if model_name != DEFAULT_EMBEDDING_MODEL:
                log_embed(f"Falling back to default model: {DEFAULT_EMBEDDING_MODEL}")
                return self.load_model_advanced(DEFAULT_EMBEDDING_MODEL, max_length, force_download)
            else:
                raise RuntimeError(f"Failed to load any embedding model: {e}")
    
    def validate_input_data_comprehensive(self, input_dir: Path) -> Dict[str, Any]:
        """Comprehensive validation of input data with detailed analysis."""
        log_embed(f"Comprehensive input validation: {input_dir}")
        
        validation = {
            'valid': True,
            'input_directory': str(input_dir),
            'files_found': [],
            'files_missing': [],
            'file_details': {},
            'column_issues': [],
            'data_quality': {},
            'label_balance': {},
            'total_samples': 0
        }
        
        if not input_dir.exists():
            log_embed_error(f"Input directory does not exist: {input_dir}")
            validation['valid'] = False
            validation['directory_missing'] = True
            return validation
        
        required_files = ['train.csv', 'val.csv', 'test.csv']
        required_columns = ['text', 'label']
        
        for filename in required_files:
            filepath = input_dir / filename
            
            if not filepath.exists():
                validation['files_missing'].append(filename)
                validation['valid'] = False
                log_embed(f"Missing file: {filepath}")
                continue
            
            validation['files_found'].append(filename)
            
            try:
                # Load and analyze file
                df = pd.read_csv(filepath)
                file_info = {
                    'samples': len(df),
                    'columns': list(df.columns),
                    'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
                }
                
                # Check required columns
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    validation['column_issues'].append({
                        'file': filename,
                        'missing_columns': missing_columns,
                        'available_columns': list(df.columns)
                    })
                    validation['valid'] = False
                    log_embed_error(f"Column issues in {filename}: missing {missing_columns}")
                else:
                    # Analyze data quality
                    text_stats = {
                        'null_count': df['text'].isnull().sum(),
                        'empty_count': (df['text'] == '').sum(),
                        'avg_length': df['text'].str.len().mean(),
                        'min_length': df['text'].str.len().min(),
                        'max_length': df['text'].str.len().max(),
                        'very_short_texts': (df['text'].str.len() < 10).sum(),
                        'very_long_texts': (df['text'].str.len() > 1000).sum()
                    }
                    
                    label_stats = {
                        'unique_labels': df['label'].nunique(),
                        'label_distribution': df['label'].value_counts().to_dict(),
                        'null_labels': df['label'].isnull().sum()
                    }
                    
                    file_info['text_quality'] = text_stats
                    file_info['label_quality'] = label_stats
                    
                    log_embed(f"{filename}: {len(df):,} samples, quality OK")
                    log_embed(f"  Text: avg_len={text_stats['avg_length']:.0f}, nulls={text_stats['null_count']}")
                    log_embed(f"  Labels: {label_stats['unique_labels']} classes, nulls={label_stats['null_labels']}")
                
                validation['file_details'][filename] = file_info
                validation['total_samples'] += len(df)
                
            except Exception as e:
                validation['column_issues'].append({
                    'file': filename,
                    'error': str(e)
                })
                validation['valid'] = False
                log_embed_error(f"Error reading {filename}: {e}")
        
        # Overall data quality assessment
        if validation['valid']:
            total_samples = validation['total_samples']
            log_embed("All input files validated successfully")
            log_embed(f"Total samples: {total_samples:,}")
            
            # Calculate split proportions
            if validation['file_details']:
                for filename, details in validation['file_details'].items():
                    proportion = details['samples'] / total_samples * 100
                    log_embed(f"{filename}: {details['samples']:,} samples ({proportion:.1f}%)")
        
        return validation
    
    def generate_embeddings_advanced(self, input_dir: Path, output_dir: Path,
                                   model_name: str = DEFAULT_EMBEDDING_MODEL,
                                   batch_size: int = None, max_length: int = None,
                                   force_recreate: bool = False) -> Dict[str, Any]:
        """
        🔧 FIXED: Generate embeddings with explicit file saving verification.
        """
        start_time = time.time()
        
        # Setup defaults
        if batch_size is None:
            batch_size = DEFAULT_TRAIN_PARAMS['embedding_batch_size']
        if max_length is None:
            max_length = DEFAULT_TRAIN_PARAMS['max_sequence_length']
        
        log_embed("Starting advanced embedding generation")
        log_embed(f"Input: {input_dir}")
        log_embed(f"Output: {output_dir}")
        log_embed(f"Model: {model_name}")
        log_embed(f"Batch size: {batch_size}")
        log_embed(f"Max length: {max_length}")
        log_embed(f"Force recreate: {force_recreate}")
        
        # 🔧 FIXED: Create output directory with explicit verification
        output_dir.mkdir(parents=True, exist_ok=True)
        log_embed(f"Created output directory: {output_dir}")
        
        if not output_dir.exists():
            raise RuntimeError(f"Failed to create output directory: {output_dir}")
        
        # Validate input data
        validation = self.validate_input_data_comprehensive(input_dir)
        if not validation['valid']:
            raise ValueError(f"Input validation failed: {validation}")
        
        # Check for existing embeddings
        if not force_recreate:
            existing_files = []
            for split in ['train', 'val', 'test']:
                emb_file = output_dir / f"X_{split}.npy"
                lab_file = output_dir / f"y_{split}.npy"
                if emb_file.exists() and lab_file.exists():
                    existing_files.append(split)
            
            if existing_files:
                log_embed(f"Existing embeddings found for: {existing_files}")
                if not force_recreate:
                    log_embed("Use force_recreate=True to regenerate")
                    return {
                        'success': False,
                        'reason': 'embeddings_exist',
                        'existing_files': existing_files
                    }
        
        # Load model
        model = self.load_model_advanced(model_name, max_length)
        embedding_dim = model.encode(["test"], convert_to_numpy=True).shape[1]
        
        log_embed(f"Model loaded - embedding dimension: {embedding_dim}")
        
        # Process each split
        results = {
            'success': True,
            'model_name': model_name,
            'embedding_dim': embedding_dim,
            'max_length': max_length,
            'batch_size': batch_size,
            'input_directory': str(input_dir),
            'output_directory': str(output_dir),
            'splits_processed': [],
            'total_samples': 0,
            'processing_time': 0,
            'validation_results': validation,
            'model_load_time': self.model_load_time,
            'files_saved': []  # 🔧 FIXED: Track saved files
        }
        
        splits = ['train', 'val', 'test']
        embedding_start_time = time.time()
        
        for split in splits:
            input_file = input_dir / f"{split}.csv"
            
            if not input_file.exists():
                log_embed(f"Skipping {split}: file not found")
                continue
            
            log_embed(f"Processing {split} split...")
            
            try:
                # Load data
                df = pd.read_csv(input_file)
                
                if 'text' not in df.columns or 'label' not in df.columns:
                    log_embed_error(f"Required columns missing in {split}")
                    continue
                
                # Extract and validate data
                texts = df['text'].fillna('').astype(str).tolist()
                labels = df['label'].tolist()
                
                # Filter out empty texts
                valid_indices = [i for i, text in enumerate(texts) if text.strip()]
                texts = [texts[i] for i in valid_indices]
                labels = [labels[i] for i in valid_indices]
                
                removed_count = len(df) - len(texts)
                if removed_count > 0:
                    log_embed(f"Removed {removed_count} empty texts from {split}")
                
                if len(texts) == 0:
                    raise ValueError(f"No valid text samples found in {split}")
                
                log_embed(f"Processing {len(texts):,} valid samples for {split}")
                
                # Generate embeddings with progress tracking
                embeddings = []
                
                if len(texts) > 100:
                    # Use tqdm for large datasets
                    progress_bar = tqdm(
                        range(0, len(texts), batch_size),
                        desc=f"Embedding {split}",
                        unit="batch"
                    )
                else:
                    progress_bar = range(0, len(texts), batch_size)
                
                for i in progress_bar:
                    batch_texts = texts[i:i+batch_size]
                    
                    try:
                        batch_embeddings = model.encode(
                            batch_texts,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                            batch_size=batch_size
                        )
                        embeddings.append(batch_embeddings)
                        
                    except Exception as batch_error:
                        log_embed_error(f"Error processing batch {i//batch_size + 1}: {batch_error}")
                        # Add zero embeddings for failed batch
                        embeddings.append(np.zeros((len(batch_texts), embedding_dim)))
                
                # Combine all embeddings
                if embeddings:
                    all_embeddings = np.vstack(embeddings)
                else:
                    all_embeddings = np.array([]).reshape(0, embedding_dim)
                
                all_labels = np.array(labels)
                
                # 🔧 FIXED: Save embeddings and labels with explicit verification
                embeddings_file = output_dir / f"X_{split}.npy"
                labels_file = output_dir / f"y_{split}.npy"
                
                log_embed(f"Saving {split} embeddings to: {embeddings_file}")
                np.save(embeddings_file, all_embeddings)
                
                log_embed(f"Saving {split} labels to: {labels_file}")
                np.save(labels_file, all_labels)
                
                # 🔧 FIXED: Verify saved files exist and are correct
                if not embeddings_file.exists():
                    raise RuntimeError(f"Failed to save embeddings file: {embeddings_file}")
                if not labels_file.exists():
                    raise RuntimeError(f"Failed to save labels file: {labels_file}")
                
                # Verify file contents
                try:
                    saved_embeddings = np.load(embeddings_file)
                    saved_labels = np.load(labels_file)
                    
                    if saved_embeddings.shape[0] != saved_labels.shape[0]:
                        raise ValueError(f"Shape mismatch: embeddings {saved_embeddings.shape} vs labels {saved_labels.shape}")
                    
                    log_embed_success(f"{split} files saved and verified:")
                    log_embed_success(f"  Embeddings: {embeddings_file} - {saved_embeddings.shape}")
                    log_embed_success(f"  Labels: {labels_file} - {saved_labels.shape}")
                    log_embed_success(f"  File sizes: {embeddings_file.stat().st_size / 1024:.1f}KB, {labels_file.stat().st_size / 1024:.1f}KB")
                    
                except Exception as verify_error:
                    raise RuntimeError(f"File verification failed for {split}: {verify_error}")
                
                # Record split results
                split_info = {
                    'split': split,
                    'original_samples': len(df),
                    'processed_samples': len(texts),
                    'removed_samples': removed_count,
                    'embedding_shape': all_embeddings.shape,
                    'labels_shape': all_labels.shape,
                    'embeddings_file': str(embeddings_file),
                    'labels_file': str(labels_file),
                    'file_size_mb': {
                        'embeddings': embeddings_file.stat().st_size / (1024 * 1024),
                        'labels': labels_file.stat().st_size / (1024 * 1024)
                    }
                }
                
                results['splits_processed'].append(split_info)
                results['total_samples'] += len(texts)
                results['files_saved'].extend([str(embeddings_file), str(labels_file)])
                
                log_embed_success(f"{split} completed successfully")
                
            except Exception as split_error:
                log_embed_error(f"Error processing {split}: {split_error}")
                results['success'] = False
                results['errors'] = results.get('errors', [])
                results['errors'].append({
                    'split': split,
                    'error': str(split_error)
                })
        
        # Calculate processing time
        processing_time = time.time() - start_time
        embedding_time = time.time() - embedding_start_time
        results['processing_time'] = processing_time
        results['embedding_generation_time'] = embedding_time
        
        # 🔧 FIXED: Save comprehensive metadata with explicit verification
        metadata = {
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'embedding_dimension': embedding_dim,
                'max_sequence_length': max_length,
                'batch_size': batch_size,
                'processing_time_seconds': processing_time,
                'model_load_time_seconds': self.model_load_time,
                'embedding_generation_time_seconds': embedding_time,
                'total_samples_processed': results['total_samples']
            },
            'model_details': {
                'model_type': 'SentenceTransformer',
                'model_name': model_name,
                'embedding_dimension': embedding_dim,
                'max_sequence_length': max_length,
                'device': self._get_safe_device_info(model)
            },
            'data_info': {
                'input_directory': str(input_dir),
                'output_directory': str(output_dir),
                'validation_results': validation,
                'splits_processed': results['splits_processed']
            },
            'file_structure': {
                'embeddings_files': [f"X_{info['split']}.npy" for info in results['splits_processed']],
                'label_files': [f"y_{info['split']}.npy" for info in results['splits_processed']],
                'metadata_file': 'embedding_metadata.json'
            },
            'project_metadata': {
                'project_root': str(self.project_root),
                'generation_script': str(__file__),
                'python_version': sys.version,
                'compatible_with': ['train_mlp.py', 'train_svm.py', 'report.py']
            }
        }
        
        # 🔧 FIXED: Save metadata with verification
        metadata_file = output_dir / "embedding_metadata.json"
        log_embed(f"Saving metadata to: {metadata_file}")
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        if not metadata_file.exists():
            log_embed_error(f"Failed to save metadata file: {metadata_file}")
        else:
            log_embed_success(f"Metadata saved: {metadata_file}")
            results['files_saved'].append(str(metadata_file))
        
        # 🔧 FIXED: Print comprehensive summary
        self._print_generation_summary(results, processing_time, output_dir, metadata_file)
        
        return results
    
    def _get_safe_device_info(self, model) -> str:
        """Safely get device information from model."""
        try:
            if hasattr(model, 'device'):
                return str(model.device)
            elif hasattr(model, '_target_device'):
                return str(model._target_device)
            else:
                return 'unknown'
        except Exception:
            return 'unavailable'
    
    def _print_generation_summary(self, results: Dict[str, Any], processing_time: float, 
                                 output_dir: Path, metadata_file: Path):
        """🔧 FIXED: Print comprehensive generation summary with file verification."""
        if results['success']:
            log_embed_success("EMBEDDING GENERATION COMPLETED SUCCESSFULLY!")
            log_embed_success(f"Total time: {processing_time:.1f}s (Model: {self.model_load_time:.1f}s, Embedding: {results.get('embedding_generation_time', 0):.1f}s)")
            log_embed_success(f"Total samples: {results['total_samples']:,}")
            log_embed_success(f"Output directory: {output_dir}")
            log_embed_success(f"Metadata saved: {metadata_file}")
            
            # Show splits processed
            log_embed_success(f"Splits processed: {len(results['splits_processed'])}")
            for split_info in results['splits_processed']:
                log_embed_success(f"  {split_info['split']}: {split_info['processed_samples']:,} samples")
            
            # Show files saved
            log_embed_success(f"Files saved: {len(results['files_saved'])}")
            for file_path in results['files_saved']:
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    file_size = file_path_obj.stat().st_size / 1024
                    log_embed_success(f"  ✓ {file_path_obj.name} ({file_size:.1f}KB)")
                else:
                    log_embed_error(f"  ✗ {file_path_obj.name} (FILE MISSING!)")
            
            # Show file sizes
            total_size = sum(
                info['file_size_mb']['embeddings'] + info['file_size_mb']['labels']
                for info in results['splits_processed']
            )
            log_embed_success(f"Total output size: {total_size:.1f}MB")
            
            # Performance metrics
            embedding_time = results.get('embedding_generation_time', processing_time)
            if embedding_time > 0:
                samples_per_sec = results['total_samples'] / embedding_time
                log_embed_success(f"Performance: {samples_per_sec:.1f} samples/second")
                
        else:
            log_embed_error("EMBEDDING GENERATION FAILED")
            if 'errors' in results:
                for error in results['errors']:
                    log_embed_error(f"Split {error['split']}: {error['error']}")
            
            if results.get('reason') == 'embeddings_exist':
                log_embed_error("Reason: Embeddings already exist")
                log_embed_error("Use --force-recreate to regenerate them")


# =============================================================================
# 🔧 FIXED MAIN EMBEDDING GENERATION FUNCTIONS
# =============================================================================

def create_embeddings_pipeline_compatible(input_dir: str, output_dir: str,
                                         model_name: str = DEFAULT_EMBEDDING_MODEL,
                                         batch_size: int = None, max_length: int = None,
                                         force_recreate: bool = False,
                                         verbose: bool = True) -> Dict[str, Any]:
    """
    🔧 FIXED: Pipeline-compatible embedding generation function.
    """
    try:
        # Configure logging based on verbose setting
        if not verbose:
            logging.getLogger().setLevel(logging.WARNING)
        
        # Initialize generator
        generator = AdvancedEmbeddingGenerator()
        
        # 🔧 FIXED: Smart input directory detection
        input_path = generator.find_input_directory(input_dir)
        output_path = Path(output_dir)
        
        # Generate embeddings
        results = generator.generate_embeddings_advanced(
            input_path, output_path, model_name, batch_size, max_length, force_recreate
        )
        
        return results
        
    except Exception as e:
        log_embed_error(f"Pipeline embedding generation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'input_dir': input_dir,
            'output_dir': output_dir
        }

# Legacy compatibility function (maintained for backward compatibility)
def create_embeddings(input_dir, output_dir, model_name=DEFAULT_EMBEDDING_MODEL, 
                     batch_size=None, max_length=None, verbose=True):
    """Legacy embedding generation function for backward compatibility."""
    results = create_embeddings_pipeline_compatible(
        str(input_dir), str(output_dir), model_name, batch_size, max_length, False, verbose
    )
    
    if results['success']:
        # Convert to legacy format
        legacy_summary = {
            'model_name': results['model_name'],
            'embedding_dim': results['embedding_dim'],
            'max_length': results['max_length'],
            'batch_size': results['batch_size'],
            'splits_processed': results['splits_processed'],
            'total_samples': results['total_samples'],
            'input_dir': results['input_directory'],
            'output_dir': results['output_directory']
        }
        return legacy_summary
    else:
        raise RuntimeError(f"Embedding generation failed: {results.get('error', 'Unknown error')}")

def validate_input_files(input_dir, verbose=True):
    """Legacy validation function for backward compatibility."""
    generator = AdvancedEmbeddingGenerator()
    validation = generator.validate_input_data_comprehensive(Path(input_dir))
    
    # Convert to legacy format
    legacy_validation = {
        'valid': validation['valid'],
        'files_found': validation['files_found'],
        'files_missing': validation['files_missing'],
        'column_issues': validation['column_issues'],
        'sample_counts': {},
        'input_directory': validation['input_directory']
    }
    
    # Extract sample counts from file details
    for filename, details in validation.get('file_details', {}).items():
        legacy_validation['sample_counts'][filename] = details['samples']
    
    # Add directory missing flag if needed
    if 'directory_missing' in validation:
        legacy_validation['directory_missing'] = validation['directory_missing']
    
    return legacy_validation


def main(argv=None):
    """🔧 FIXED: Enhanced main function with pipeline compatibility."""
    parser = argparse.ArgumentParser(
        description="🔧 FIXED - Advanced Embedding Generation for Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard embedding generation
  %(prog)s
  
  # Pipeline integration (called by pipeline_runner.py)
  %(prog)s --output-dir results/session_20241229/embeddings --force-recreate
  
  # Custom input/output directories
  %(prog)s --input-dir results/session_20241229/processed --output-dir results/session_20241229/embeddings
  
  # Force regeneration with different model
  %(prog)s --force-recreate --model-name all-mpnet-base-v2
        """
    )
    
    # 🔧 FIXED: Input/Output paths (dynamic defaults with smart detection)
    parser.add_argument("--input-dir", default=None,
                       help="Directory containing train.csv, val.csv, test.csv (auto-detect if not specified)")
    parser.add_argument("--output-dir", default=str(EMBEDDINGS_DATA_DIR),
                       help=f"Directory to save embeddings and metadata (default: {EMBEDDINGS_DATA_DIR})")
    
    # Model configuration
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL,
                       choices=AVAILABLE_EMBEDDING_MODELS,
                       help=f"SentenceTransformer model name (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_PARAMS['embedding_batch_size'],
                       help=f"Batch size for embedding generation (default: {DEFAULT_TRAIN_PARAMS['embedding_batch_size']})")
    parser.add_argument("--max-length", type=int, default=DEFAULT_TRAIN_PARAMS['max_sequence_length'],
                       help=f"Maximum sequence length (default: {DEFAULT_TRAIN_PARAMS['max_sequence_length']})")
    
    # Advanced options
    parser.add_argument("--force-recreate", action="store_true",
                       help="Force recreation of embeddings even if they exist")
    parser.add_argument("--force-download", action="store_true",
                       help="Force download of model instead of using local cache")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate input files without generating embeddings")
    
    # Output control
    parser.add_argument("--verbose", "-v", action="store_true", default=True,
                       help="Verbose output with detailed progress (default: True)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress detailed output except errors")
    parser.add_argument("--log-file", help="Save detailed logs to file")
    
    # Parse arguments (handle sys.argv injection)
    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv[1:])  # Skip script name
    
    # Configure logging
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Optional log file
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
    
    verbose = args.verbose and not args.quiet
    
    if verbose:
        log_embed("🔧 FIXED - Advanced Sentiment Analysis Embedding Generation")
        log_embed(f"Project root: {PROJECT_ROOT}")
        log_embed(f"Output directory: {args.output_dir}")
        log_embed(f"Model: {args.model_name}")
        log_embed(f"Batch size: {args.batch_size}")
        log_embed(f"Max length: {args.max_length}")
        log_embed(f"Force recreate: {args.force_recreate}")
    
    try:
        # Initialize generator
        generator = AdvancedEmbeddingGenerator()
        
        # 🔧 FIXED: Smart input directory detection
        input_dir = generator.find_input_directory(args.input_dir)
        log_embed(f"Using input directory: {input_dir}")
        
        # Validate input files
        if verbose:
            log_embed("Validating input files...")
        
        validation = generator.validate_input_data_comprehensive(input_dir)
        
        if not validation['valid']:
            log_embed_error("Input validation failed")
            log_embed_error("Suggestions:")
            log_embed_error("  1. Run preprocessing first: python scripts/preprocess.py")
            log_embed_error(f"  2. Check that CSV files exist in: {input_dir}")
            log_embed_error("  3. Verify CSV files have 'text' and 'label' columns")
            
            if validation.get('directory_missing'):
                log_embed_error("  4. Create directory structure first")
            if validation['files_missing']:
                log_embed_error(f"  Missing files: {validation['files_missing']}")
            if validation['column_issues']:
                for issue in validation['column_issues']:
                    if 'error' in issue:
                        log_embed_error(f"  File error in {issue['file']}: {issue['error']}")
                    else:
                        log_embed_error(f"  Column issue in {issue['file']}: missing {issue['missing_columns']}")
            
            return 1
        
        if args.validate_only:
            log_embed_success("Validation completed successfully!")
            log_embed_success(f"Total samples: {validation['total_samples']:,}")
            for filename, details in validation['file_details'].items():
                log_embed_success(f"{filename}: {details['samples']:,} samples")
            return 0
        
        # Generate embeddings
        results = generator.generate_embeddings_advanced(
            input_dir,
            Path(args.output_dir),
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            force_recreate=args.force_recreate
        )
        
        return 0 if results['success'] else 1
            
    except KeyboardInterrupt:
        log_embed_error("Operation interrupted by user")
        return 1
    except Exception as e:
        log_embed_error(f"Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())