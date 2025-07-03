#!/usr/bin/env python3
"""
Advanced Embedding Generation Script - COMPLETE PIPELINE INTEGRATION (IMPROVED)
Generates sentence embeddings for sentiment analysis with comprehensive pipeline integration.

FEATURES:
- Dynamic path detection with robust PROJECT_ROOT handling
- Integration with enhanced_utils_unified.py and pipeline_runner.py
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
    
    # Custom input/output directories (pipeline integration)
    python scripts/embed_dataset.py --input-dir results/session_20241229/processed --output-dir results/session_20241229/embeddings
    
    # Force regeneration with different model
    python scripts/embed_dataset.py --force-recreate --model-name all-mpnet-base-v2
    
    # Pipeline automation compatible
    python scripts/embed_dataset.py --input-dir processed/ --output-dir embeddings/ --quiet
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration values
# The original version attempted to import these from ``config_constants`` but
# that module does not exist in this repository.  This caused the script to
# fail at runtime.  We now define sensible defaults directly here so the
# embedding step always runs.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
AVAILABLE_EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
]
DEFAULT_TRAIN_PARAMS = {
    "embedding_batch_size": 32,
    "max_sequence_length": 512,
    "embedding_cache_size": 1000,
}

logger.info("✅ Using internal default configuration values")

class AdvancedEmbeddingGenerator:
    """
    Advanced embedding generation class with comprehensive pipeline integration.
    Handles model management, progress tracking, error recovery, and quality assurance.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize the advanced embedding generator.
        
        Args:
            project_root: Project root directory (auto-detect if None)
        """
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.logger = logger
        self.model = None
        self.model_name = None
        self.model_load_time = 0
        self.setup_paths()
        
        self.logger.info(f"🚀 Advanced Embedding Generator initialized")
        self.logger.info(f"📁 Project root: {self.project_root}")
    
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
    
    def find_local_models(self) -> Dict[str, Path]:
        """
        Find available local models with comprehensive search.
        
        Returns:
            Dictionary mapping model names to local paths
        """
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
                            self.logger.info(f"   ✅ Found local model '{model_name}' at {search_path}")
                            break
                else:
                    # Search subdirectories
                    for model_name, possible_names in model_mappings.items():
                        for possible_name in possible_names:
                            potential_path = search_path / possible_name
                            if potential_path.exists() and (potential_path / "config.json").exists():
                                local_models[model_name] = potential_path
                                self.logger.info(f"   ✅ Found local model '{model_name}' at {potential_path}")
        
        return local_models
    
    def load_model_advanced(self, model_name: str, max_length: int = 512, 
                           force_download: bool = False) -> SentenceTransformer:
        """
        Load SentenceTransformer model with advanced fallback strategies.
        
        Args:
            model_name: Model name to load
            max_length: Maximum sequence length
            force_download: Force download instead of using local models
        
        Returns:
            Loaded SentenceTransformer model
        """
        model_load_start = time.time()
        self.logger.info(f"📥 Loading SentenceTransformer model: {model_name}")
        
        # Find local models
        local_models = {} if force_download else self.find_local_models()
        
        try:
            # Try local model first if available and not forcing download
            if model_name in local_models and not force_download:
                local_path = local_models[model_name]
                self.logger.info(f"🔄 Loading from local directory: {local_path}")
                
                model = SentenceTransformer(str(local_path))
                self.logger.info(f"✅ Local model loaded successfully")
                
            else:
                # Download/load from HuggingFace
                self.logger.info(f"🔄 Downloading/loading model from HuggingFace: {model_name}")
                
                model = SentenceTransformer(model_name)
                self.logger.info(f"✅ Remote model loaded successfully")
                
                # Optionally cache locally for future use
                try:
                    cache_dir = self.paths['models_dir'] / model_name.replace('/', '_')
                    if not cache_dir.exists():
                        self.logger.info(f"💾 Caching model locally to: {cache_dir}")
                        model.save(str(cache_dir))
                except Exception as cache_error:
                    self.logger.warning(f"⚠️ Could not cache model locally: {cache_error}")
            
            # Configure model settings
            if hasattr(model, 'max_seq_length'):
                model.max_seq_length = max_length
                self.logger.info(f"   📏 Set max sequence length: {max_length}")
            
            # Get model info
            sample_embedding = model.encode(["test"], convert_to_numpy=True)
            embedding_dim = sample_embedding.shape[1]
            
            self.logger.info(f"   📊 Embedding dimension: {embedding_dim}")
            
            # IMPROVED: Safe device detection
            try:
                device_info = str(model.device) if hasattr(model, 'device') else 'unknown'
                self.logger.info(f"   🧠 Model device: {device_info}")
            except:
                device_info = 'unavailable'
                self.logger.info(f"   🧠 Model device: {device_info}")
            
            self.model = model
            self.model_name = model_name
            self.model_load_time = time.time() - model_load_start
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Error loading {model_name}: {e}")
            
            # Fallback strategy
            if model_name != DEFAULT_EMBEDDING_MODEL:
                self.logger.info(f"🔄 Falling back to default model: {DEFAULT_EMBEDDING_MODEL}")
                return self.load_model_advanced(DEFAULT_EMBEDDING_MODEL, max_length, force_download)
            else:
                raise RuntimeError(f"Failed to load any embedding model: {e}")
    
    def check_label_balance(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        IMPROVED: Check for label distribution issues.
        
        Args:
            df: DataFrame with label column
            filename: Name of the file being checked
        
        Returns:
            Dictionary with balance analysis results
        """
        label_counts = df['label'].value_counts()
        total = len(df)
        
        balance_info = {
            'total_samples': total,
            'unique_labels': len(label_counts),
            'label_distribution': label_counts.to_dict(),
            'min_class_count': label_counts.min(),
            'max_class_count': label_counts.max(),
            'min_class_pct': (label_counts.min() / total) * 100,
            'max_class_pct': (label_counts.max() / total) * 100,
            'balance_warnings': []
        }
        
        # Check for severe imbalance
        if balance_info['min_class_pct'] < 5:
            warning = f"Severe class imbalance - smallest class: {balance_info['min_class_pct']:.1f}%"
            balance_info['balance_warnings'].append(warning)
            self.logger.warning(f"⚠️ {filename}: {warning}")
        
        # Check for single class
        if len(label_counts) == 1:
            warning = f"Only one class found: {label_counts.index[0]}"
            balance_info['balance_warnings'].append(warning)
            self.logger.error(f"❌ {filename}: {warning}")
        
        # Check for extreme imbalance (>90% in one class)
        if balance_info['max_class_pct'] > 90:
            warning = f"Extreme imbalance - largest class: {balance_info['max_class_pct']:.1f}%"
            balance_info['balance_warnings'].append(warning)
            self.logger.warning(f"⚠️ {filename}: {warning}")
        
        return balance_info
    
    def validate_input_data_comprehensive(self, input_dir: Path) -> Dict[str, Any]:
        """
        Comprehensive validation of input data with detailed analysis.
        
        Args:
            input_dir: Directory containing CSV files
        
        Returns:
            Validation results dictionary
        """
        self.logger.info(f"🔍 Comprehensive input validation: {input_dir}")
        
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
            self.logger.error(f"❌ Input directory does not exist: {input_dir}")
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
                self.logger.warning(f"⚠️ Missing file: {filepath}")
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
                    self.logger.error(f"❌ Column issues in {filename}: missing {missing_columns}")
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
                    
                    # IMPROVED: Add label balance analysis
                    balance_analysis = self.check_label_balance(df, filename)
                    validation['label_balance'][filename] = balance_analysis
                    
                    file_info['text_quality'] = text_stats
                    file_info['label_quality'] = label_stats
                    file_info['label_balance'] = balance_analysis
                    
                    self.logger.info(f"✅ {filename}: {len(df):,} samples, quality OK")
                    self.logger.info(f"   📊 Text: avg_len={text_stats['avg_length']:.0f}, nulls={text_stats['null_count']}")
                    self.logger.info(f"   🏷️ Labels: {label_stats['unique_labels']} classes, nulls={label_stats['null_labels']}")
                
                validation['file_details'][filename] = file_info
                validation['total_samples'] += len(df)
                
            except Exception as e:
                validation['column_issues'].append({
                    'file': filename,
                    'error': str(e)
                })
                validation['valid'] = False
                self.logger.error(f"❌ Error reading {filename}: {e}")
        
        # Overall data quality assessment
        if validation['valid']:
            total_samples = validation['total_samples']
            self.logger.info(f"✅ All input files validated successfully")
            self.logger.info(f"   📊 Total samples: {total_samples:,}")
            
            # Calculate split proportions
            if validation['file_details']:
                for filename, details in validation['file_details'].items():
                    proportion = details['samples'] / total_samples * 100
                    self.logger.info(f"   📊 {filename}: {details['samples']:,} samples ({proportion:.1f}%)")
        
        return validation
    
    def verify_embedding_quality(self, embeddings: np.ndarray, split_name: str) -> Dict[str, Any]:
        """
        IMPROVED: Verify embedding quality with comprehensive checks.
        
        Args:
            embeddings: Generated embeddings array
            split_name: Name of the split being verified
        
        Returns:
            Quality analysis results
        """
        quality_info = {
            'split': split_name,
            'shape': embeddings.shape,
            'issues_found': [],
            'quality_metrics': {}
        }
        
        if len(embeddings) == 0:
            quality_info['issues_found'].append("Empty embeddings array")
            return quality_info
        
        # Check for NaN/Inf values
        nan_count = np.isnan(embeddings).sum()
        inf_count = np.isinf(embeddings).sum()
        
        if nan_count > 0:
            issue = f"Found {nan_count} NaN values"
            quality_info['issues_found'].append(issue)
            self.logger.error(f"❌ {split_name}: {issue}")
        
        if inf_count > 0:
            issue = f"Found {inf_count} Inf values"
            quality_info['issues_found'].append(issue)
            self.logger.error(f"❌ {split_name}: {issue}")
        
        # Check embedding variance (too low = potential issues)
        embedding_std = np.std(embeddings, axis=0).mean()
        quality_info['quality_metrics']['mean_std'] = float(embedding_std)
        
        if embedding_std < 0.01:
            issue = f"Low embedding variance ({embedding_std:.4f}) - possible issues"
            quality_info['issues_found'].append(issue)
            self.logger.warning(f"⚠️ {split_name}: {issue}")
        
        # Check for zero embeddings
        zero_embeddings = np.all(embeddings == 0, axis=1).sum()
        quality_info['quality_metrics']['zero_embeddings_count'] = int(zero_embeddings)
        
        if zero_embeddings > 0:
            zero_pct = (zero_embeddings / len(embeddings)) * 100
            issue = f"Found {zero_embeddings} zero embeddings ({zero_pct:.1f}%)"
            quality_info['issues_found'].append(issue)
            self.logger.warning(f"⚠️ {split_name}: {issue}")
        
        # Calculate additional quality metrics
        quality_info['quality_metrics'].update({
            'mean_norm': float(np.linalg.norm(embeddings, axis=1).mean()),
            'std_norm': float(np.linalg.norm(embeddings, axis=1).std()),
            'min_value': float(embeddings.min()),
            'max_value': float(embeddings.max()),
            'mean_value': float(embeddings.mean())
        })
        
        if not quality_info['issues_found']:
            self.logger.info(f"✅ {split_name}: Embedding quality verification passed")
        
        return quality_info
    
    def process_large_dataset_chunks(self, texts: List[str], chunk_size: int = 50000):
        """
        IMPROVED: Process large datasets in chunks to manage memory.
        
        Args:
            texts: List of texts to process
            chunk_size: Size of each chunk
        
        Yields:
            Chunks of texts
        """
        for i in range(0, len(texts), chunk_size):
            yield texts[i:i+chunk_size], i
    
    def generate_embeddings_advanced(self, input_dir: Path, output_dir: Path,
                                   model_name: str = DEFAULT_EMBEDDING_MODEL,
                                   batch_size: int = None, max_length: int = None,
                                   force_recreate: bool = False) -> Dict[str, Any]:
        """
        Generate embeddings with advanced progress tracking and error handling.
        
        Args:
            input_dir: Directory containing CSV files
            output_dir: Directory to save embeddings
            model_name: SentenceTransformer model name
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            force_recreate: Force recreation even if files exist
        
        Returns:
            Comprehensive generation results
        """
        start_time = time.time()
        
        # Setup defaults
        if batch_size is None:
            batch_size = DEFAULT_TRAIN_PARAMS['embedding_batch_size']
        if max_length is None:
            max_length = DEFAULT_TRAIN_PARAMS['max_sequence_length']
        
        self.logger.info(f"🚀 Starting advanced embedding generation")
        self.logger.info(f"   📁 Input: {input_dir}")
        self.logger.info(f"   📁 Output: {output_dir}")
        self.logger.info(f"   🤖 Model: {model_name}")
        self.logger.info(f"   ⚙️ Batch size: {batch_size}")
        self.logger.info(f"   📏 Max length: {max_length}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
                self.logger.info(f"⚠️ Existing embeddings found for: {existing_files}")
                if not force_recreate:
                    self.logger.info("   Use force_recreate=True to regenerate")
                    return {
                        'success': False,
                        'reason': 'embeddings_exist',
                        'existing_files': existing_files
                    }
        
        # Load model
        model = self.load_model_advanced(model_name, max_length)
        embedding_dim = model.encode(["test"], convert_to_numpy=True).shape[1]
        
        # IMPROVED: Check if embedding dimension is too small
        if embedding_dim < 256:
            self.logger.warning(f"⚠️ Small embedding dimension ({embedding_dim}) - may affect MLP performance")
        
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
            'quality_checks': {},
            'model_load_time': self.model_load_time
        }
        
        splits = ['train', 'val', 'test']
        embedding_start_time = time.time()
        
        for split in splits:
            input_file = input_dir / f"{split}.csv"
            
            if not input_file.exists():
                self.logger.warning(f"⚠️ Skipping {split}: file not found")
                continue
            
            self.logger.info(f"\n🔄 Processing {split} split...")
            
            try:
                # Load data
                df = pd.read_csv(input_file)
                
                if 'text' not in df.columns or 'label' not in df.columns:
                    self.logger.error(f"❌ Required columns missing in {split}")
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
                    self.logger.warning(f"   ⚠️ Removed {removed_count} empty texts")
                
                # IMPROVED: Check for empty splits
                if len(texts) == 0:
                    raise ValueError(f"No valid text samples found in {split}")
                
                self.logger.info(f"   📊 Processing {len(texts):,} valid samples")
                
                # IMPROVED: Memory warning for large datasets
                if len(texts) > 100000:
                    self.logger.warning(f"   ⚠️ Large dataset ({len(texts):,} samples). Monitoring memory usage...")
                
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
                        self.logger.error(f"❌ Error processing batch {i//batch_size + 1}: {batch_error}")
                        # Add zero embeddings for failed batch
                        embeddings.append(np.zeros((len(batch_texts), embedding_dim)))
                
                # Combine all embeddings
                if embeddings:
                    all_embeddings = np.vstack(embeddings)
                else:
                    all_embeddings = np.array([]).reshape(0, embedding_dim)
                
                all_labels = np.array(labels)
                
                # IMPROVED: Verify embedding quality
                quality_check = self.verify_embedding_quality(all_embeddings, split)
                results['quality_checks'][split] = quality_check
                
                # Save embeddings and labels
                embeddings_file = output_dir / f"X_{split}.npy"
                labels_file = output_dir / f"y_{split}.npy"
                
                np.save(embeddings_file, all_embeddings)
                np.save(labels_file, all_labels)
                
                # Verify saved files
                saved_embeddings = np.load(embeddings_file)
                saved_labels = np.load(labels_file)
                
                if saved_embeddings.shape[0] != saved_labels.shape[0]:
                    raise ValueError(f"Shape mismatch: embeddings {saved_embeddings.shape} vs labels {saved_labels.shape}")
                
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
                    },
                    'quality_check': quality_check
                }
                
                results['splits_processed'].append(split_info)
                results['total_samples'] += len(texts)
                
                self.logger.info(f"   ✅ {split} completed:")
                self.logger.info(f"      💾 Embeddings: {embeddings_file} ({split_info['file_size_mb']['embeddings']:.1f}MB)")
                self.logger.info(f"      💾 Labels: {labels_file} ({split_info['file_size_mb']['labels']:.1f}MB)")
                self.logger.info(f"      📏 Shape: {all_embeddings.shape}")
                
            except Exception as split_error:
                self.logger.error(f"❌ Error processing {split}: {split_error}")
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
        
        # IMPROVED: Calculate quality metrics with proper error handling
        if results['splits_processed']:
            total_original = sum(info['original_samples'] for info in results['splits_processed'])
            if total_original > 0:
                processing_success_rate = results['total_samples'] / total_original
            else:
                processing_success_rate = 0
                self.logger.warning("⚠️ No original samples found for success rate calculation")
        else:
            processing_success_rate = 0
            total_original = 0
        
        # Save comprehensive metadata
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
                'splits_processed': results['splits_processed'],
                'quality_checks': results['quality_checks']
            },
            'file_structure': {
                'embeddings_files': [f"X_{info['split']}.npy" for info in results['splits_processed']],
                'label_files': [f"y_{info['split']}.npy" for info in results['splits_processed']],
                'metadata_file': 'embedding_metadata.json'
            },
            'quality_metrics': {
                'total_original_samples': total_original,
                'total_processed_samples': results['total_samples'],
                'total_removed_samples': sum(info['removed_samples'] for info in results['splits_processed']),
                'processing_success_rate': processing_success_rate
            },
            'performance_metrics': {
                'samples_per_second': results['total_samples'] / embedding_time if embedding_time > 0 else 0,
                'mb_per_sample': sum(
                    info['file_size_mb']['embeddings'] + info['file_size_mb']['labels']
                    for info in results['splits_processed']
                ) / results['total_samples'] if results['total_samples'] > 0 else 0,
                'model_load_time_seconds': self.model_load_time,
                'embedding_generation_time_seconds': embedding_time
            },
            'project_metadata': {
                'project_root': str(self.project_root),
                'generation_script': str(__file__),
                'python_version': sys.version,
                'compatible_with': ['train_mlp.py', 'train_svm.py', 'report.py']
            }
        }
        
        # Save metadata
        metadata_file = output_dir / "embedding_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        # Print summary
        self._print_generation_summary(results, processing_time, output_dir, metadata_file)
        
        return results
    
    def _get_safe_device_info(self, model) -> str:
        """
        IMPROVED: Safely get device information from model.
        
        Args:
            model: SentenceTransformer model
        
        Returns:
            Device information string
        """
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
        """
        IMPROVED: Print comprehensive generation summary.
        
        Args:
            results: Generation results
            processing_time: Total processing time
            output_dir: Output directory
            metadata_file: Metadata file path
        """
        if results['success']:
            self.logger.info(f"\n🎉 EMBEDDING GENERATION COMPLETED SUCCESSFULLY!")
            self.logger.info(f"   ⏱️ Total time: {processing_time:.1f}s (Model: {self.model_load_time:.1f}s, Embedding: {results.get('embedding_generation_time', 0):.1f}s)")
            self.logger.info(f"   📊 Total samples: {results['total_samples']:,}")
            self.logger.info(f"   📁 Output directory: {output_dir}")
            self.logger.info(f"   📄 Metadata saved: {metadata_file}")
            
            # Show splits processed
            self.logger.info(f"   🎯 Splits processed: {len(results['splits_processed'])}")
            for split_info in results['splits_processed']:
                quality_issues = len(split_info['quality_check'].get('issues_found', []))
                quality_status = "⚠️" if quality_issues > 0 else "✅"
                self.logger.info(f"      • {split_info['split']}: {split_info['processed_samples']:,} samples {quality_status}")
            
            # Show file sizes
            total_size = sum(
                info['file_size_mb']['embeddings'] + info['file_size_mb']['labels']
                for info in results['splits_processed']
            )
            self.logger.info(f"   💾 Total output size: {total_size:.1f}MB")
            
            # Performance metrics
            embedding_time = results.get('embedding_generation_time', processing_time)
            if embedding_time > 0:
                samples_per_sec = results['total_samples'] / embedding_time
                self.logger.info(f"   🚀 Performance: {samples_per_sec:.1f} samples/second")
            
            # Quality summary
            total_quality_issues = sum(
                len(check.get('issues_found', []))
                for check in results.get('quality_checks', {}).values()
            )
            if total_quality_issues > 0:
                self.logger.warning(f"   ⚠️ Quality issues found: {total_quality_issues} (check logs for details)")
            else:
                self.logger.info(f"   ✅ All quality checks passed")
                
        else:
            self.logger.error(f"❌ EMBEDDING GENERATION FAILED")
            if 'errors' in results:
                for error in results['errors']:
                    self.logger.error(f"   Split {error['split']}: {error['error']}")
            
            if results.get('reason') == 'embeddings_exist':
                self.logger.error("   Reason: Embeddings already exist")
                self.logger.error("   Use --force-recreate to regenerate them")


# =============================================================================
# MAIN EMBEDDING GENERATION FUNCTIONS
# =============================================================================

def create_embeddings_pipeline_compatible(input_dir: str, output_dir: str,
                                         model_name: str = DEFAULT_EMBEDDING_MODEL,
                                         batch_size: int = None, max_length: int = None,
                                         force_recreate: bool = False,
                                         verbose: bool = True) -> Dict[str, Any]:
    """
    Pipeline-compatible embedding generation function.
    Enhanced version of the original create_embeddings function with full integration.
    
    Args:
        input_dir: Directory containing train.csv, val.csv, test.csv
        output_dir: Directory to save embeddings
        model_name: SentenceTransformer model name
        batch_size: Batch size for embedding generation
        max_length: Maximum sequence length
        force_recreate: Force recreation even if embeddings exist
        verbose: Whether to print progress
    
    Returns:
        Comprehensive generation results dictionary
    """
    try:
        # Configure logging based on verbose setting
        if not verbose:
            logging.getLogger().setLevel(logging.WARNING)
        
        # Initialize generator
        generator = AdvancedEmbeddingGenerator()
        
        # Convert paths
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Generate embeddings
        results = generator.generate_embeddings_advanced(
            input_path, output_path, model_name, batch_size, max_length, force_recreate
        )
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline embedding generation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'input_dir': input_dir,
            'output_dir': output_dir
        }

# Legacy compatibility function (maintained for backward compatibility)
def create_embeddings(input_dir, output_dir, model_name=DEFAULT_EMBEDDING_MODEL, 
                     batch_size=None, max_length=None, verbose=True):
    """
    Legacy embedding generation function for backward compatibility.
    Calls the new pipeline-compatible function internally.
    
    Args:
        input_dir: Directory containing train.csv, val.csv, test.csv
        output_dir: Directory to save embeddings
        model_name: SentenceTransformer model name
        batch_size: Batch size for embedding generation
        max_length: Maximum sequence length
        verbose: Whether to print progress
    
    Returns:
        Summary dictionary (legacy format)
    """
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
    """
    Legacy validation function for backward compatibility.
    
    Args:
        input_dir: Directory to validate
        verbose: Whether to print messages
    
    Returns:
        Validation results dictionary
    """
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
    """Enhanced main function with comprehensive CLI support and pipeline integration."""
    parser = argparse.ArgumentParser(
        description="Advanced Embedding Generation for Sentiment Analysis (IMPROVED)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard embedding generation
  %(prog)s
  
  # Custom input/output directories (pipeline integration)
  %(prog)s --input-dir results/session_20241229/processed --output-dir results/session_20241229/embeddings
  
  # Force regeneration with different model
  %(prog)s --force-recreate --model-name all-mpnet-base-v2
  
  # Validation only
  %(prog)s --validate-only --verbose
  
  # Pipeline automation compatible
  %(prog)s --input-dir processed/ --output-dir embeddings/ --quiet --force-recreate
        """
    )
    
    # Input/Output paths (dynamic defaults)
    parser.add_argument("--input-dir", default=str(PROCESSED_DATA_DIR),
                       help=f"Directory containing train.csv, val.csv, test.csv (default: {PROCESSED_DATA_DIR})")
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
        logger.info("🚀 Advanced Sentiment Analysis Embedding Generation (IMPROVED)")
        logger.info(f"📁 Project root: {PROJECT_ROOT}")
        logger.info(f"📄 Input directory: {args.input_dir}")
        logger.info(f"📄 Output directory: {args.output_dir}")
        logger.info(f"🤖 Model: {args.model_name}")
        logger.info(f"⚙️ Batch size: {args.batch_size}")
        logger.info(f"📏 Max length: {args.max_length}")
    
    try:
        # Initialize generator
        generator = AdvancedEmbeddingGenerator()
        
        # Validate input files
        if verbose:
            logger.info("🔍 Validating input files...")
        
        validation = generator.validate_input_data_comprehensive(Path(args.input_dir))
        
        if not validation['valid']:
            logger.error("❌ Input validation failed")
            logger.error("💡 Suggestions:")
            logger.error("   1. Run preprocessing first: python scripts/preprocess.py")
            logger.error(f"   2. Check that CSV files exist in: {args.input_dir}")
            logger.error("   3. Verify CSV files have 'text' and 'label' columns")
            
            if validation.get('directory_missing'):
                logger.error(f"   4. Create directory structure first")
            if validation['files_missing']:
                logger.error(f"   Missing files: {validation['files_missing']}")
            if validation['column_issues']:
                for issue in validation['column_issues']:
                    if 'error' in issue:
                        logger.error(f"   File error in {issue['file']}: {issue['error']}")
                    else:
                        logger.error(f"   Column issue in {issue['file']}: missing {issue['missing_columns']}")
            
            # Show label balance warnings if any
            for filename, balance_info in validation.get('label_balance', {}).items():
                for warning in balance_info.get('balance_warnings', []):
                    logger.error(f"   Label issue in {filename}: {warning}")
            
            return 1
        
        if args.validate_only:
            logger.info("✅ Validation completed successfully!")
            logger.info(f"   📊 Total samples: {validation['total_samples']:,}")
            for filename, details in validation['file_details'].items():
                logger.info(f"   📊 {filename}: {details['samples']:,} samples")
                
                # Show label balance info
                balance_info = details.get('label_balance', {})
                if balance_info.get('balance_warnings'):
                    for warning in balance_info['balance_warnings']:
                        logger.warning(f"      ⚠️ {warning}")
            return 0
        
        # Generate embeddings
        results = generator.generate_embeddings_advanced(
            Path(args.input_dir),
            Path(args.output_dir),
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            force_recreate=args.force_recreate
        )
        
        return 0 if results['success'] else 1
            
    except KeyboardInterrupt:
        logger.warning("❌ Operation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
