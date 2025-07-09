#!/usr/bin/env python3
"""
Enhanced Embedding Generation Script - UNIVERSAL CSV + DYNAMIC FILE DETECTION
Generates sentence embeddings for sentiment analysis with complete flexibility and dynamic file handling.

🆕 ENHANCED FEATURES:
- ✅ Dynamic file detection: automatically finds available CSV files
- ✅ Universal CSV support: works with any preprocessed CSV structure
- ✅ Intelligent mode detection: training vs inference based on available files
- ✅ Robust fallback mechanisms for missing files
- ✅ Enhanced error recovery and graceful degradation
- ✅ Smart parameter adjustment based on dataset characteristics
- ✅ Support for single file processing (inference.csv, train.csv, etc.)
- ✅ Always produces embeddable output compatible with downstream models

USAGE:
    python scripts/embed_dataset.py                                    # Auto-detect files and mode
    python scripts/embed_dataset.py --input-dir data/processed         # Specific directory
    python scripts/embed_dataset.py --force-inference                  # Force inference mode
    python scripts/embed_dataset.py --input-file external.csv          # Single external file
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
import re

warnings.filterwarnings('ignore')

def load_csv_robust(path):
    """Load CSV with fallback encodings and normalized headers."""
    try:
        df = pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", engine="python")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8", errors="replace", engine="python")

    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns=lambda c: (
        c.replace("review", "text").replace("content", "text")
         .replace("sentiment", "label").replace("class", "label")
    ))

    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df["label"] = df["label"].replace({"positive": 1, "negative": 0, "pos": 1, "neg": 0})

    return df

# Dynamic project root detection
def find_project_root() -> Path:
    """Find project root with robust detection algorithm."""
    try:
        current = Path(__file__).resolve()
        
        if current.parent.name == 'scripts':
            return current.parent.parent
        
        while current != current.parent:
            if (current / "scripts").exists() and (current / "data").exists():
                return current
            current = current.parent
        
        current = Path(__file__).resolve()
        if current.parent.name == "scripts":
            return current.parent.parent
        
        return current.parent
    except:
        return Path.cwd()

PROJECT_ROOT = find_project_root()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
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

class UniversalEmbeddingGenerator:
    """
    Universal embedding generation class with dynamic file detection and flexible processing.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.logger = logger
        self.model = None
        self.model_name = None
        self.model_load_time = 0
        
        self.logger.info(f"🚀 Universal Embedding Generator initialized")
        self.logger.info(f"📁 Project root: {self.project_root}")
    
    def discover_available_files(self, input_dir: Path) -> Dict[str, Any]:
        """🆕 NEW: Dynamically discover available CSV files and determine processing mode"""
        self.logger.info(f"🔍 Discovering available files in: {input_dir}")
        
        discovery_result = {
            'available_files': {},
            'detected_mode': 'unknown',
            'processing_strategy': 'unknown',
            'recommended_splits': [],
            'total_samples': 0,
            'issues': []
        }
        
        if not input_dir.exists():
            discovery_result['issues'].append(f"Input directory does not exist: {input_dir}")
            return discovery_result
        
        # Look for CSV files
        csv_files = list(input_dir.glob("*.csv"))
        self.logger.info(f"📋 Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
        
        if not csv_files:
            discovery_result['issues'].append("No CSV files found in input directory")
            return discovery_result
        
        # Analyze each CSV file
        for csv_file in csv_files:
            try:
                df = load_csv_robust(csv_file)
                
                # Basic validation
                if len(df) == 0:
                    self.logger.warning(f"⚠️ Empty file: {csv_file.name}")
                    continue
                
                if 'text' not in df.columns:
                    self.logger.warning(f"⚠️ No 'text' column in: {csv_file.name}")
                    continue
                
                # Analyze file content
                has_labels = 'label' in df.columns and not df['label'].isnull().all()
                valid_labels = 0
                if has_labels:
                    valid_labels = len(df[df['label'].isin([0, 1, '0', '1', 'positive', 'negative', 'pos', 'neg'])])
                
                file_info = {
                    'path': csv_file,
                    'samples': len(df),
                    'has_text': True,
                    'has_labels': has_labels,
                    'valid_labels': valid_labels,
                    'label_ratio': valid_labels / len(df) if len(df) > 0 else 0,
                    'avg_text_length': df['text'].astype(str).str.len().mean(),
                    'columns': list(df.columns)
                }
                
                discovery_result['available_files'][csv_file.stem] = file_info
                discovery_result['total_samples'] += len(df)
                
                self.logger.info(f"   ✅ {csv_file.name}: {len(df)} samples, labels={has_labels} ({valid_labels} valid)")
                
            except Exception as e:
                self.logger.error(f"   ❌ Error analyzing {csv_file.name}: {str(e)}")
                discovery_result['issues'].append(f"Error reading {csv_file.name}: {str(e)}")
        
        # Determine processing mode and strategy
        self._determine_processing_strategy(discovery_result)
        
        return discovery_result
    
    def _determine_processing_strategy(self, discovery_result: Dict[str, Any]):
        """Determine the best processing strategy based on available files"""
        files = discovery_result['available_files']
        
        # Check for standard training splits
        has_train = 'train' in files and files['train']['has_labels']
        has_val = 'val' in files and files['val']['has_labels'] 
        has_test = 'test' in files
        
        # Check for inference file
        has_inference = 'inference' in files
        
        # Check for single files with labels
        labeled_files = [name for name, info in files.items() if info['has_labels'] and info['valid_labels'] > 5]
        unlabeled_files = [name for name, info in files.items() if not info['has_labels']]
        
        if has_train and has_val:
            discovery_result['detected_mode'] = 'training_standard'
            discovery_result['processing_strategy'] = 'process_all_splits'
            discovery_result['recommended_splits'] = ['train', 'val'] + (['test'] if has_test else [])
            
        elif has_train and not has_val:
            discovery_result['detected_mode'] = 'training_single'
            discovery_result['processing_strategy'] = 'create_val_split'
            discovery_result['recommended_splits'] = ['train'] + (['test'] if has_test else [])
            
        elif has_inference:
            discovery_result['detected_mode'] = 'inference'
            discovery_result['processing_strategy'] = 'inference_only'
            discovery_result['recommended_splits'] = ['inference']
            
        elif len(labeled_files) == 1 and not has_train:
            # Single file with labels - treat as training data
            discovery_result['detected_mode'] = 'single_file_training'
            discovery_result['processing_strategy'] = 'split_single_file'
            discovery_result['recommended_splits'] = [labeled_files[0]]
            
        elif len(unlabeled_files) >= 1:
            # Only unlabeled files - inference mode
            discovery_result['detected_mode'] = 'inference_unlabeled'
            discovery_result['processing_strategy'] = 'inference_unlabeled'
            discovery_result['recommended_splits'] = unlabeled_files[:1]  # Use first unlabeled file
            
        elif len(labeled_files) > 1:
            # Multiple labeled files - use the largest one
            largest_file = max(labeled_files, key=lambda x: files[x]['samples'])
            discovery_result['detected_mode'] = 'multi_file_training'
            discovery_result['processing_strategy'] = 'use_largest_file'
            discovery_result['recommended_splits'] = [largest_file]
            
        else:
            discovery_result['detected_mode'] = 'unknown'
            discovery_result['processing_strategy'] = 'error'
            discovery_result['issues'].append("No suitable files found for processing")
        
        self.logger.info(f"🎯 Processing strategy determined:")
        self.logger.info(f"   Mode: {discovery_result['detected_mode']}")
        self.logger.info(f"   Strategy: {discovery_result['processing_strategy']}")
        self.logger.info(f"   Files to process: {discovery_result['recommended_splits']}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for processing."""
        if not isinstance(text, str):
            text = str(text)
        
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
        text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        text = re.sub(r'[.]{2,}', '...', text)
        text = re.sub(r'[!]{2,}', '!!', text)
        text = re.sub(r'[?]{2,}', '??', text)
        text = re.sub(r'[^\w\s.,!?:;\'"()\-\[\]]', ' ', text)
        text = ' '.join(text.split())
        
        return text.strip()
    
    def load_model_advanced(self, model_name: str, max_length: int = 512) -> SentenceTransformer:
        """Load SentenceTransformer model with error handling."""
        model_load_start = time.time()
        self.logger.info(f"📥 Loading SentenceTransformer model: {model_name}")
        
        try:
            model = SentenceTransformer(model_name)
            
            if hasattr(model, 'max_seq_length'):
                model.max_seq_length = max_length
                self.logger.info(f"   📏 Set max sequence length: {max_length}")
            
            sample_embedding = model.encode(["test"], convert_to_numpy=True)
            embedding_dim = sample_embedding.shape[1]
            
            self.logger.info(f"   📊 Embedding dimension: {embedding_dim}")
            
            try:
                device_info = str(model.device) if hasattr(model, 'device') else 'unknown'
                self.logger.info(f"   🧠 Model device: {device_info}")
            except:
                device_info = 'unavailable'
            
            self.model = model
            self.model_name = model_name
            self.model_load_time = time.time() - model_load_start
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Error loading {model_name}: {e}")
            if model_name != DEFAULT_EMBEDDING_MODEL:
                self.logger.info(f"🔄 Falling back to default model: {DEFAULT_EMBEDDING_MODEL}")
                return self.load_model_advanced(DEFAULT_EMBEDDING_MODEL, max_length)
            else:
                raise RuntimeError(f"Failed to load any embedding model: {e}")
    
    def process_file_for_embeddings(self, file_path: Path, split_name: str, model: SentenceTransformer, 
                                  batch_size: int, embedding_dim: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Process a single file and generate embeddings."""
        processing_info = {
            'split': split_name,
            'input_file': str(file_path),
            'original_samples': 0,
            'processed_samples': 0,
            'removed_samples': 0,
            'has_labels': False,
            'embedding_shape': None,
            'labels_shape': None,
            'error': None
        }
        
        try:
            df = load_csv_robust(file_path)
            processing_info['original_samples'] = len(df)
            
            if len(df) == 0:
                self.logger.warning(f"⚠️ Empty file: {split_name}")
                return (np.array([]).reshape(0, embedding_dim), 
                       np.array([]), 
                       processing_info)
            
            if 'text' not in df.columns:
                raise ValueError(f"Missing 'text' column in {split_name}")
            
            # Extract and clean texts
            texts = df['text'].fillna('').astype(str).tolist()
            cleaned_texts = [self.clean_text(text) for text in texts]
            
            # Filter out empty texts
            valid_indices = [i for i, text in enumerate(cleaned_texts) if len(text) > 5]
            valid_texts = [cleaned_texts[i] for i in valid_indices]
            
            processing_info['processed_samples'] = len(valid_texts)
            processing_info['removed_samples'] = processing_info['original_samples'] - processing_info['processed_samples']
            
            if len(valid_texts) == 0:
                self.logger.warning(f"⚠️ No valid texts found in {split_name}")
                return (np.array([]).reshape(0, embedding_dim), 
                       np.array([]), 
                       processing_info)
            
            # Handle labels
            if 'label' in df.columns and not df['label'].isnull().all():
                labels = df['label'].tolist()
                labels = [labels[i] for i in valid_indices]
                
                # Normalize labels
                normalized_labels = []
                for label in labels:
                    if pd.isna(label):
                        normalized_labels.append(-1)
                    elif str(label).lower() in ['positive', 'pos', '1', 1]:
                        normalized_labels.append(1)
                    elif str(label).lower() in ['negative', 'neg', '0', 0]:
                        normalized_labels.append(0)
                    else:
                        normalized_labels.append(-1)  # Unknown labels
                
                labels = normalized_labels
                processing_info['has_labels'] = True
                
                # Check if we have valid labels
                valid_label_count = sum(1 for l in labels if l in [0, 1])
                if valid_label_count > 0:
                    self.logger.info(f"   🏷️ {split_name}: {valid_label_count} valid labels out of {len(labels)}")
                else:
                    self.logger.info(f"   🔍 {split_name}: No valid labels, treating as inference")
                    processing_info['has_labels'] = False
            else:
                labels = [-1] * len(valid_texts)  # Placeholder labels for inference
                processing_info['has_labels'] = False
                self.logger.info(f"   🔍 {split_name}: No labels found, using placeholder labels")
            
            self.logger.info(f"   📊 Processing {len(valid_texts):,} valid samples in {split_name}")
            
            # Generate embeddings with progress tracking
            embeddings = []
            
            if len(valid_texts) > 100:
                progress_bar = tqdm(
                    range(0, len(valid_texts), batch_size),
                    desc=f"Embedding {split_name}",
                    unit="batch"
                )
            else:
                progress_bar = range(0, len(valid_texts), batch_size)
            
            for i in progress_bar:
                batch_texts = valid_texts[i:i+batch_size]
                
                try:
                    batch_embeddings = model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=batch_size
                    )
                    embeddings.append(batch_embeddings)
                    
                except Exception as batch_error:
                    self.logger.error(f"❌ Error processing batch {i//batch_size + 1} in {split_name}: {batch_error}")
                    embeddings.append(np.zeros((len(batch_texts), embedding_dim)))
            
            # Combine all embeddings
            if embeddings:
                all_embeddings = np.vstack(embeddings)
            else:
                all_embeddings = np.array([]).reshape(0, embedding_dim)
            
            all_labels = np.array(labels)
            
            processing_info['embedding_shape'] = all_embeddings.shape
            processing_info['labels_shape'] = all_labels.shape
            
            return all_embeddings, all_labels, processing_info
            
        except Exception as e:
            processing_info['error'] = str(e)
            self.logger.error(f"❌ Error processing {split_name}: {e}")
            return (np.array([]).reshape(0, embedding_dim), 
                   np.array([]), 
                   processing_info)
    
    def generate_embeddings_universal(self, input_dir: Path, output_dir: Path,
                                    model_name: str = DEFAULT_EMBEDDING_MODEL,
                                    batch_size: int = None, max_length: int = None,
                                    force_recreate: bool = False,
                                    force_inference: bool = False) -> Dict[str, Any]:
        """🆕 NEW: Universal embedding generation with dynamic file detection."""
        start_time = time.time()
        
        # Setup defaults
        if batch_size is None:
            batch_size = DEFAULT_TRAIN_PARAMS['embedding_batch_size']
        if max_length is None:
            max_length = DEFAULT_TRAIN_PARAMS['max_sequence_length']
        
        self.logger.info(f"🚀 Starting universal embedding generation")
        self.logger.info(f"   📁 Input: {input_dir}")
        self.logger.info(f"   📁 Output: {output_dir}")
        self.logger.info(f"   🤖 Model: {model_name}")
        self.logger.info(f"   ⚙️ Batch size: {batch_size}")
        self.logger.info(f"   📏 Max length: {max_length}")
        self.logger.info(f"   🔍 Force inference: {force_inference}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Discover available files
        discovery = self.discover_available_files(input_dir)
        
        if discovery['issues']:
            return {
                'success': False,
                'error': 'File discovery failed',
                'issues': discovery['issues'],
                'discovery_results': discovery
            }
        
        # Override detection if force_inference is specified
        if force_inference:
            discovery['detected_mode'] = 'inference'
            discovery['processing_strategy'] = 'inference_forced'
            self.logger.info("🔍 Forced inference mode activated")
        
        # Check for existing embeddings (if not forcing recreation)
        if not force_recreate:
            existing_files = []
            for split in discovery['recommended_splits']:
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
                        'existing_files': existing_files,
                        'discovery_results': discovery
                    }
        
        # Load model
        model = self.load_model_advanced(model_name, max_length)
        embedding_dim = model.encode(["test"], convert_to_numpy=True).shape[1]
        
        # Process files
        results = {
            'success': True,
            'mode': discovery['detected_mode'],
            'processing_strategy': discovery['processing_strategy'],
            'model_name': model_name,
            'embedding_dim': embedding_dim,
            'max_length': max_length,
            'batch_size': batch_size,
            'input_directory': str(input_dir),
            'output_directory': str(output_dir),
            'splits_processed': [],
            'total_samples': 0,
            'processing_time': 0,
            'discovery_results': discovery,
            'model_load_time': self.model_load_time
        }
        
        embedding_start_time = time.time()
        
        for split_name in discovery['recommended_splits']:
            if split_name not in discovery['available_files']:
                self.logger.warning(f"⚠️ Recommended split '{split_name}' not found in available files")
                continue
                
            file_info = discovery['available_files'][split_name]
            file_path = file_info['path']
            
            self.logger.info(f"\n🔄 Processing {split_name} split...")
            
            # Process file and generate embeddings
            embeddings, labels, processing_info = self.process_file_for_embeddings(
                file_path, split_name, model, batch_size, embedding_dim
            )
            
            if processing_info.get('error'):
                self.logger.error(f"❌ Failed to process {split_name}: {processing_info['error']}")
                results['success'] = False
                continue
            
            if len(embeddings) == 0:
                self.logger.warning(f"⚠️ No embeddings generated for {split_name}")
                continue
            
            # Save embeddings and labels
            embeddings_file = output_dir / f"X_{split_name}.npy"
            labels_file = output_dir / f"y_{split_name}.npy"
            
            np.save(embeddings_file, embeddings)
            np.save(labels_file, labels)
            
            # Verify saved files
            saved_embeddings = np.load(embeddings_file)
            saved_labels = np.load(labels_file)
            
            if saved_embeddings.shape[0] != saved_labels.shape[0]:
                raise ValueError(f"Shape mismatch: embeddings {saved_embeddings.shape} vs labels {saved_labels.shape}")
            
            # Record split results
            split_info = {
                'split': split_name,
                'original_samples': processing_info['original_samples'],
                'processed_samples': processing_info['processed_samples'],
                'removed_samples': processing_info['removed_samples'],
                'has_labels': processing_info['has_labels'],
                'embedding_shape': embeddings.shape,
                'labels_shape': labels.shape,
                'embeddings_file': str(embeddings_file),
                'labels_file': str(labels_file),
                'file_size_mb': {
                    'embeddings': embeddings_file.stat().st_size / (1024 * 1024),
                    'labels': labels_file.stat().st_size / (1024 * 1024)
                }
            }
            
            results['splits_processed'].append(split_info)
            results['total_samples'] += processing_info['processed_samples']
            
            self.logger.info(f"   ✅ {split_name} completed:")
            self.logger.info(f"      💾 Embeddings: {embeddings_file} ({split_info['file_size_mb']['embeddings']:.1f}MB)")
            self.logger.info(f"      💾 Labels: {labels_file} ({split_info['file_size_mb']['labels']:.1f}MB)")
            self.logger.info(f"      📏 Shape: {embeddings.shape}")
            self.logger.info(f"      🏷️ Has labels: {processing_info['has_labels']}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        embedding_time = time.time() - embedding_start_time
        results['processing_time'] = processing_time
        results['embedding_generation_time'] = embedding_time
        
        # Save comprehensive metadata
        metadata = {
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'mode': discovery['detected_mode'],
                'processing_strategy': discovery['processing_strategy'],
                'model_name': model_name,
                'embedding_dimension': embedding_dim,
                'max_sequence_length': max_length,
                'batch_size': batch_size,
                'processing_time_seconds': processing_time,
                'model_load_time_seconds': self.model_load_time,
                'embedding_generation_time_seconds': embedding_time,
                'total_samples_processed': results['total_samples']
            },
            'discovery_results': discovery,
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
        """Print comprehensive generation summary."""
        mode_emoji = "🔍" if 'inference' in results['mode'] else "🎯"
        
        if results['success']:
            self.logger.info(f"\n🎉 UNIVERSAL EMBEDDING GENERATION COMPLETED!")
            self.logger.info(f"   {mode_emoji} Mode: {results['mode'].upper()}")
            self.logger.info(f"   🔧 Strategy: {results['processing_strategy']}")
            self.logger.info(f"   ⏱️ Total time: {processing_time:.1f}s")
            self.logger.info(f"   📊 Total samples: {results['total_samples']:,}")
            self.logger.info(f"   📁 Output directory: {output_dir}")
            self.logger.info(f"   📄 Metadata saved: {metadata_file}")
            
            # Show splits processed
            self.logger.info(f"   🎯 Splits processed: {len(results['splits_processed'])}")
            for split_info in results['splits_processed']:
                label_status = "🏷️" if split_info['has_labels'] else "🔍"
                self.logger.info(f"      • {split_info['split']}: {split_info['processed_samples']:,} samples {label_status}")
            
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
                
        else:
            self.logger.error(f"❌ UNIVERSAL EMBEDDING GENERATION FAILED")
            self.logger.error(f"   {mode_emoji} Mode: {results.get('mode', 'unknown').upper()}")

# =============================================================================
# MAIN INTERFACE FUNCTIONS
# =============================================================================

def create_embeddings_universal(input_dir: str, output_dir: str,
                               model_name: str = DEFAULT_EMBEDDING_MODEL,
                               batch_size: int = None, max_length: int = None,
                               force_recreate: bool = False,
                               force_inference: bool = False,
                               verbose: bool = True) -> Dict[str, Any]:
    """
    Universal embedding generation function with dynamic file detection.
    
    Args:
        input_dir: Directory containing CSV files
        output_dir: Directory to save embeddings
        model_name: SentenceTransformer model name
        batch_size: Batch size for embedding generation
        max_length: Maximum sequence length
        force_recreate: Force recreation even if embeddings exist
        force_inference: Force inference mode
        verbose: Whether to print progress
    
    Returns:
        Comprehensive generation results dictionary
    """
    try:
        # Configure logging based on verbose setting
        if not verbose:
            logging.getLogger().setLevel(logging.WARNING)
        
        # Initialize generator
        generator = UniversalEmbeddingGenerator()
        
        # Convert paths
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Generate embeddings
        results = generator.generate_embeddings_universal(
            input_path, output_path, model_name, batch_size, max_length, 
            force_recreate, force_inference
        )
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Universal embedding generation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'input_dir': input_dir,
            'output_dir': output_dir
        }

# Legacy compatibility function
def create_embeddings(input_dir, output_dir, model_name=DEFAULT_EMBEDDING_MODEL, 
                     batch_size=None, max_length=None, verbose=True):
    """Legacy embedding generation function for backward compatibility."""
    results = create_embeddings_universal(
        str(input_dir), str(output_dir), model_name, batch_size, max_length, False, False, verbose
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
            'output_dir': results['output_directory'],
            'mode': results['mode']
        }
        return legacy_summary
    else:
        raise RuntimeError(f"Embedding generation failed: {results.get('error', 'Unknown error')}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Universal Embedding Generation for Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🆕 ENHANCED FEATURES:
- Dynamic file detection and processing mode determination
- Universal CSV support with intelligent fallbacks
- Automatic handling of training vs inference modes
- Robust error recovery and graceful degradation

Examples:
  python scripts/embed_dataset.py                                   # Auto-detect files and mode
  python scripts/embed_dataset.py --input-dir data/processed        # Specific directory
  python scripts/embed_dataset.py --force-inference                 # Force inference mode
  python scripts/embed_dataset.py --model-name all-mpnet-base-v2    # Custom model
        """
    )
    
    # Input/Output paths
    parser.add_argument("--input-dir", default=None,
                       help=f"Directory containing CSV files (default: auto-detect)")
    parser.add_argument("--output-dir", default=None,
                       help=f"Directory to save embeddings (default: auto-detect)")
    
    # Model configuration
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL,
                       choices=AVAILABLE_EMBEDDING_MODELS,
                       help=f"SentenceTransformer model name (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_PARAMS['embedding_batch_size'],
                       help=f"Batch size for embedding generation (default: {DEFAULT_TRAIN_PARAMS['embedding_batch_size']})")
    parser.add_argument("--max-length", type=int, default=DEFAULT_TRAIN_PARAMS['max_sequence_length'],
                       help=f"Maximum sequence length (default: {DEFAULT_TRAIN_PARAMS['max_sequence_length']})")
    
    # Mode configuration
    parser.add_argument("--force-inference", action="store_true",
                       help="Force inference mode (ignore training files)")
    
    # Advanced options
    parser.add_argument("--force-recreate", action="store_true",
                       help="Force recreation of embeddings even if they exist")
    
    # Output control
    parser.add_argument("--verbose", "-v", action="store_true", default=True,
                       help="Verbose output with detailed progress (default: True)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress detailed output except errors")
    
    return parser.parse_args()

def main(argv=None):
    """Enhanced main function with universal CSV support."""
    parser = argparse.ArgumentParser(
        description="Universal Embedding Generation for Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Parse arguments
    if argv is None:
        args = parse_arguments()
    else:
        args = parser.parse_args(argv[1:])
    
    # Configure logging
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    verbose = args.verbose and not args.quiet
    
    # Setup default paths
    if args.input_dir is None:
        args.input_dir = str(PROJECT_ROOT / "data" / "processed")
    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "data" / "embeddings")
    
    if verbose:
        logger.info("🚀 Universal Sentiment Analysis Embedding Generation")
        logger.info(f"📁 Project root: {PROJECT_ROOT}")
        logger.info(f"📄 Input directory: {args.input_dir}")
        logger.info(f"📄 Output directory: {args.output_dir}")
        logger.info(f"🤖 Model: {args.model_name}")
        logger.info(f"⚙️ Batch size: {args.batch_size}")
        logger.info(f"📏 Max length: {args.max_length}")
        logger.info(f"🔍 Force inference: {args.force_inference}")
    
    try:
        # Run universal embedding generation
        results = create_embeddings_universal(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            force_recreate=args.force_recreate,
            force_inference=args.force_inference,
            verbose=verbose
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