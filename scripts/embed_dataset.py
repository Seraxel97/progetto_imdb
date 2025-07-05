#!/usr/bin/env python3
"""
Advanced Embedding Generation Script - COMPLETE PIPELINE + EXTERNAL CSV INTEGRATION
Generates sentence embeddings for sentiment analysis with comprehensive pipeline integration.
Now supports both TRAINING and INFERENCE modes with robust error handling + EXTERNAL CSV PROCESSING.

🆕 NEW FEATURES:
- ✅ Support for external CSV files via --input-file parameter
- ✅ Automatic output directory creation: results/embedded/FILENAME_TIMESTAMP/
- ✅ Graceful handling of CSV files with/without label columns
- ✅ Compatible output format for train_mlp.py and train_svm.py integration
- ✅ Comprehensive logging and metadata generation
- ✅ Full backward compatibility with existing pipeline behavior

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
- **NEW: Full support for both TRAINING and INFERENCE modes**
- **NEW: Automatic mode detection and graceful handling of missing labels**
- **NEW: Support for inference.csv files**
- **NEW: Support for external CSV files with automatic processing**
- **NEW: Robust handling of empty/missing files**

USAGE:
    # Standard embedding generation (training mode)
    python scripts/embed_dataset.py
    
    # Custom input/output directories (pipeline integration)
    python scripts/embed_dataset.py --input-dir results/session_20241229/processed --output-dir results/session_20241229/embeddings
    
    # NEW: External CSV file processing
    python scripts/embed_dataset.py --input-file data/external/reviews.csv
    python scripts/embed_dataset.py --input-file uploaded_file.csv --model-name all-mpnet-base-v2
    
    # Inference mode with inference.csv
    python scripts/embed_dataset.py --inference-only --input-dir data/inference
    
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
    Now supports both TRAINING and INFERENCE modes with robust error handling + EXTERNAL CSV PROCESSING.
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
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Keep letters, numbers, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def validate_csv_file(self, csv_path: Path) -> Dict[str, Any]:
        """
        🆕 NEW: Validate and analyze an external CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Dictionary with validation results and file info
        """
        validation_info = {
            'valid': False,
            'file_path': str(csv_path),
            'file_size_mb': 0,
            'total_rows': 0,
            'columns': [],
            'text_column': None,
            'label_column': None,
            'has_labels': False,
            'valid_text_rows': 0,
            'empty_text_rows': 0,
            'label_distribution': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check file existence
            if not csv_path.exists():
                validation_info['errors'].append(f"File not found: {csv_path}")
                return validation_info
            
            # Get file size
            validation_info['file_size_mb'] = csv_path.stat().st_size / (1024 * 1024)
            
            # Load and analyze CSV
            try:
                df = load_csv_robust(csv_path)
                validation_info['total_rows'] = len(df)
                validation_info['columns'] = list(df.columns)
                
                if len(df) == 0:
                    validation_info['errors'].append("CSV file is empty")
                    return validation_info
                
                self.logger.info(f"📊 Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
                self.logger.info(f"📋 Columns: {list(df.columns)}")
                
            except Exception as e:
                validation_info['errors'].append(f"Error reading CSV: {str(e)}")
                return validation_info
            
            # Find text column
            text_columns = ['text', 'review', 'content', 'comment', 'description', 'message', 'body', 'post']
            found_text_col = None
            
            for col in text_columns:
                if col in df.columns:
                    found_text_col = col
                    break
            
            if not found_text_col:
                # Try to find any string column that might contain text
                for col in df.columns:
                    if df[col].dtype == 'object':
                        sample_text = str(df[col].iloc[0]) if len(df) > 0 else ""
                        if len(sample_text.split()) > 3:  # Likely to be text if more than 3 words
                            found_text_col = col
                            validation_info['warnings'].append(f"Using column '{col}' as text column (auto-detected)")
                            break
            
            if not found_text_col:
                validation_info['errors'].append(f"No text column found. Expected: {text_columns}")
                return validation_info
            
            validation_info['text_column'] = found_text_col
            
            # Analyze text data
            text_data = df[found_text_col].fillna('').astype(str)
            text_data_cleaned = text_data.apply(self.clean_text)
            
            # Count valid vs empty texts
            valid_mask = text_data_cleaned.str.len() > 5  # Minimum 5 characters
            validation_info['valid_text_rows'] = valid_mask.sum()
            validation_info['empty_text_rows'] = len(df) - valid_mask.sum()
            
            if validation_info['valid_text_rows'] == 0:
                validation_info['errors'].append("No valid text data found (all texts are empty or too short)")
                return validation_info
            
            # Find label column
            label_columns = ['label', 'sentiment', 'class', 'target', 'rating', 'score']
            found_label_col = None
            
            for col in label_columns:
                if col in df.columns:
                    found_label_col = col
                    break
            
            if found_label_col:
                validation_info['label_column'] = found_label_col
                validation_info['has_labels'] = True
                
                # Analyze label distribution
                label_data = df[found_label_col].dropna()
                if len(label_data) > 0:
                    label_counts = label_data.value_counts()
                    validation_info['label_distribution'] = label_counts.to_dict()
                    
                    self.logger.info(f"🏷️ Found labels in column '{found_label_col}'")
                    self.logger.info(f"📊 Label distribution: {dict(label_counts)}")
                    
                    # Check for valid binary/multi-class labels
                    unique_labels = label_data.unique()
                    if len(unique_labels) < 2:
                        validation_info['warnings'].append("Only one unique label found")
                    elif len(unique_labels) > 10:
                        validation_info['warnings'].append(f"Many unique labels found ({len(unique_labels)}), this might be a regression task")
                else:
                    validation_info['warnings'].append(f"Label column '{found_label_col}' contains no valid data")
                    validation_info['has_labels'] = False
            else:
                validation_info['warnings'].append(f"No label column found. Expected: {label_columns}")
                validation_info['has_labels'] = False
            
            # Final validation
            if validation_info['valid_text_rows'] > 0:
                validation_info['valid'] = True
                self.logger.info(f"✅ CSV validation passed")
                self.logger.info(f"   📊 Valid text rows: {validation_info['valid_text_rows']}")
                self.logger.info(f"   🏷️ Has labels: {validation_info['has_labels']}")
                
                if validation_info['empty_text_rows'] > 0:
                    self.logger.warning(f"   ⚠️ Empty/invalid text rows: {validation_info['empty_text_rows']}")
            
            return validation_info
            
        except Exception as e:
            validation_info['errors'].append(f"Unexpected error during validation: {str(e)}")
            return validation_info
    
    def create_external_output_directory(self, csv_path: Path) -> Path:
        """
        🆕 NEW: Create output directory for external CSV processing.
        
        Args:
            csv_path: Path to the input CSV file
            
        Returns:
            Path to the created output directory
        """
        # Extract filename without extension
        filename_base = csv_path.stem
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory name
        dir_name = f"{filename_base}_{timestamp}"
        
        # Create full path
        output_dir = self.paths['results_dir'] / "embedded" / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['embeddings', 'logs', 'metadata']:
            (output_dir / subdir).mkdir(exist_ok=True)
        
        self.logger.info(f"📁 Created external processing directory: {output_dir}")
        
        return output_dir
    
    def process_external_csv_file(self, csv_path: Path, model_name: str = DEFAULT_EMBEDDING_MODEL,
                                 batch_size: int = None, max_length: int = None,
                                 force_recreate: bool = False) -> Dict[str, Any]:
        """
        🆕 NEW: Process an external CSV file and generate embeddings.
        
        Args:
            csv_path: Path to the CSV file
            model_name: SentenceTransformer model name
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            force_recreate: Force recreation even if files exist
            
        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        
        # Setup defaults
        if batch_size is None:
            batch_size = DEFAULT_TRAIN_PARAMS['embedding_batch_size']
        if max_length is None:
            max_length = DEFAULT_TRAIN_PARAMS['max_sequence_length']
        
        self.logger.info(f"🆕 Processing external CSV file: {csv_path}")
        self.logger.info(f"   🤖 Model: {model_name}")
        self.logger.info(f"   ⚙️ Batch size: {batch_size}")
        self.logger.info(f"   📏 Max length: {max_length}")
        
        # Validate CSV file
        validation = self.validate_csv_file(csv_path)
        if not validation['valid']:
            return {
                'success': False,
                'error': 'CSV validation failed',
                'validation_results': validation,
                'errors': validation['errors']
            }
        
        # Create output directory
        output_dir = self.create_external_output_directory(csv_path)
        
        # Setup logging for this processing session
        log_file = output_dir / "logs" / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Load CSV data
        try:
            df = load_csv_robust(csv_path)
            self.logger.info(f"📊 Loaded CSV data: {len(df)} rows")
            
            # Extract and clean text data
            text_column = validation['text_column']
            texts = df[text_column].fillna('').astype(str).tolist()
            
            # Clean texts
            cleaned_texts = [self.clean_text(text) for text in texts]
            
            # Filter out empty texts
            valid_indices = [i for i, text in enumerate(cleaned_texts) if len(text) > 5]
            valid_texts = [cleaned_texts[i] for i in valid_indices]
            
            original_count = len(texts)
            valid_count = len(valid_texts)
            removed_count = original_count - valid_count
            
            self.logger.info(f"📊 Text processing results:")
            self.logger.info(f"   Original texts: {original_count}")
            self.logger.info(f"   Valid texts: {valid_count}")
            self.logger.info(f"   Removed texts: {removed_count}")
            
            if valid_count == 0:
                return {
                    'success': False,
                    'error': 'No valid texts found after cleaning',
                    'original_count': original_count,
                    'removed_count': removed_count
                }
            
            # Handle labels
            labels = []
            if validation['has_labels']:
                label_column = validation['label_column']
                all_labels = df[label_column].tolist()
                
                # Filter labels to match valid texts
                labels = [all_labels[i] for i in valid_indices]
                
                # Normalize labels if needed
                unique_labels = list(set(label for label in labels if pd.notna(label)))
                
                if set(unique_labels) == {'positive', 'negative'}:
                    labels = [1 if label == 'positive' else 0 if label == 'negative' else -1 for label in labels]
                    self.logger.info("🔄 Mapped text labels: negative→0, positive→1")
                elif set(unique_labels) == {'pos', 'neg'}:
                    labels = [1 if label == 'pos' else 0 if label == 'neg' else -1 for label in labels]
                    self.logger.info("🔄 Mapped short labels: neg→0, pos→1")
                elif len(unique_labels) == 2 and all(isinstance(l, (int, float)) for l in unique_labels if pd.notna(l)):
                    # Ensure binary labels are 0 and 1
                    sorted_labels = sorted(unique_labels)
                    label_mapping = {sorted_labels[0]: 0, sorted_labels[1]: 1}
                    labels = [label_mapping.get(label, -1) for label in labels]
                    self.logger.info(f"🔄 Mapped numeric labels: {label_mapping}")
                
            else:
                # Create placeholder labels for texts without labels
                labels = [-1] * valid_count  # Use -1 as placeholder for no labels
                self.logger.info("🔍 No labels found, using placeholder labels (-1)")
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error processing CSV data: {str(e)}',
                'csv_path': str(csv_path)
            }
        
        # Load embedding model
        try:
            model = self.load_model_advanced(model_name, max_length)
            embedding_dim = model.encode(["test"], convert_to_numpy=True).shape[1]
            
            self.logger.info(f"✅ Model loaded: {model_name}")
            self.logger.info(f"📏 Embedding dimension: {embedding_dim}")
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error loading model: {str(e)}',
                'model_name': model_name
            }
        
        # Generate embeddings
        try:
            self.logger.info(f"🔄 Generating embeddings for {valid_count} texts...")
            
            embeddings = []
            
            # Process in batches with progress tracking
            if valid_count > 100:
                progress_bar = tqdm(
                    range(0, valid_count, batch_size),
                    desc="Generating embeddings",
                    unit="batch"
                )
            else:
                progress_bar = range(0, valid_count, batch_size)
            
            embedding_start_time = time.time()
            
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
                    self.logger.error(f"❌ Error processing batch {i//batch_size + 1}: {batch_error}")
                    # Add zero embeddings for failed batch
                    embeddings.append(np.zeros((len(batch_texts), embedding_dim)))
            
            # Combine all embeddings
            if embeddings:
                all_embeddings = np.vstack(embeddings)
            else:
                return {
                    'success': False,
                    'error': 'No embeddings generated',
                    'valid_count': valid_count
                }
            
            all_labels = np.array(labels)
            
            embedding_time = time.time() - embedding_start_time
            
            self.logger.info(f"✅ Embeddings generated successfully")
            self.logger.info(f"   📊 Shape: {all_embeddings.shape}")
            self.logger.info(f"   ⏱️ Generation time: {embedding_time:.2f} seconds")
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error generating embeddings: {str(e)}',
                'valid_count': valid_count
            }
        
        # Save embeddings and labels
        try:
            embeddings_file = output_dir / "embeddings" / "X_embedded.npy"
            labels_file = output_dir / "embeddings" / "y_labels.npy"
            
            np.save(embeddings_file, all_embeddings)
            np.save(labels_file, all_labels)
            
            self.logger.info(f"💾 Saved embeddings: {embeddings_file}")
            self.logger.info(f"💾 Saved labels: {labels_file}")
            
            # Verify saved files
            saved_embeddings = np.load(embeddings_file)
            saved_labels = np.load(labels_file)
            
            if saved_embeddings.shape[0] != saved_labels.shape[0]:
                raise ValueError(f"Shape mismatch: embeddings {saved_embeddings.shape} vs labels {saved_labels.shape}")
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error saving embeddings: {str(e)}',
                'output_dir': str(output_dir)
            }
        
        # Create comprehensive metadata
        total_time = time.time() - start_time
        
        metadata = {
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'processing_type': 'external_csv',
                'input_file': str(csv_path),
                'output_directory': str(output_dir),
                'model_name': model_name,
                'embedding_dimension': embedding_dim,
                'max_sequence_length': max_length,
                'batch_size': batch_size,
                'total_processing_time_seconds': total_time,
                'embedding_generation_time_seconds': embedding_time,
                'model_load_time_seconds': self.model_load_time
            },
            'input_file_info': {
                'file_path': str(csv_path),
                'file_size_mb': validation['file_size_mb'],
                'original_rows': validation['total_rows'],
                'text_column_used': validation['text_column'],
                'label_column_used': validation['label_column'],
                'has_labels': validation['has_labels']
            },
            'data_processing': {
                'original_text_count': original_count,
                'valid_text_count': valid_count,
                'removed_text_count': removed_count,
                'processing_success_rate': valid_count / original_count if original_count > 0 else 0,
                'label_distribution': validation['label_distribution']
            },
            'model_details': {
                'model_type': 'SentenceTransformer',
                'model_name': model_name,
                'embedding_dimension': embedding_dim,
                'max_sequence_length': max_length,
                'device': self._get_safe_device_info(model)
            },
            'output_files': {
                'embeddings_file': str(embeddings_file),
                'labels_file': str(labels_file),
                'metadata_file': str(output_dir / "metadata" / "processing_metadata.json"),
                'log_file': str(log_file)
            },
            'quality_metrics': {
                'embeddings_shape': all_embeddings.shape,
                'labels_shape': all_labels.shape,
                'mean_embedding_norm': float(np.linalg.norm(all_embeddings, axis=1).mean()),
                'embedding_std': float(np.std(all_embeddings.mean(axis=0))),
                'nan_count': int(np.isnan(all_embeddings).sum()),
                'inf_count': int(np.isinf(all_embeddings).sum())
            },
            'performance_metrics': {
                'texts_per_second': valid_count / embedding_time if embedding_time > 0 else 0,
                'mb_per_sample': (embeddings_file.stat().st_size + labels_file.stat().st_size) / (1024 * 1024) / valid_count if valid_count > 0 else 0,
                'total_output_size_mb': (embeddings_file.stat().st_size + labels_file.stat().st_size) / (1024 * 1024)
            },
            'compatibility_info': {
                'compatible_with': ['train_mlp.py', 'train_svm.py', 'report.py'],
                'usage_instructions': {
                    'mlp_training': f'python scripts/train_mlp.py --embeddings-dir {output_dir}/embeddings --output-dir {output_dir}',
                    'svm_training': f'python scripts/train_svm.py --embeddings-dir {output_dir}/embeddings --output-dir {output_dir}',
                    'generate_report': f'python scripts/report.py --models-dir {output_dir}/models --test-data {output_dir}/embeddings/X_embedded.npy --results-dir {output_dir}'
                }
            }
        }
        
        # Save metadata
        metadata_file = output_dir / "metadata" / "processing_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        # Create processing summary
        summary_file = output_dir / "processing_summary.txt"
        summary_lines = [
            "EXTERNAL CSV PROCESSING SUMMARY",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Input File: {csv_path}",
            f"Output Directory: {output_dir}",
            "",
            "PROCESSING RESULTS:",
            f"  Original texts: {original_count}",
            f"  Valid texts processed: {valid_count}",
            f"  Removed texts: {removed_count}",
            f"  Success rate: {valid_count/original_count*100:.1f}%",
            "",
            "EMBEDDINGS:",
            f"  Model: {model_name}",
            f"  Dimension: {embedding_dim}",
            f"  Shape: {all_embeddings.shape}",
            f"  Generation time: {embedding_time:.2f}s",
            "",
            "LABELS:",
            f"  Has labels: {validation['has_labels']}",
            f"  Label column: {validation['label_column'] or 'None'}",
            f"  Label distribution: {validation['label_distribution']}",
            "",
            "OUTPUT FILES:",
            f"  Embeddings: {embeddings_file}",
            f"  Labels: {labels_file}",
            f"  Metadata: {metadata_file}",
            "",
            "NEXT STEPS:",
            "  1. Train MLP model:",
            f"     python scripts/train_mlp.py --embeddings-dir {output_dir}/embeddings --output-dir {output_dir}",
            "  2. Train SVM model:",
            f"     python scripts/train_svm.py --embeddings-dir {output_dir}/embeddings --output-dir {output_dir}",
            "  3. Generate report:",
            f"     python scripts/report.py --models-dir {output_dir}/models --test-data {output_dir}/embeddings/X_embedded.npy --results-dir {output_dir}",
            "",
            "=" * 50
        ]
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        # Log final results
        self.logger.info(f"🎉 EXTERNAL CSV PROCESSING COMPLETED SUCCESSFULLY!")
        self.logger.info(f"   ⏱️ Total time: {total_time:.1f}s")
        self.logger.info(f"   📊 Processed: {valid_count}/{original_count} texts")
        self.logger.info(f"   📁 Output directory: {output_dir}")
        self.logger.info(f"   📄 Summary: {summary_file}")
        
        # Return comprehensive results
        return {
            'success': True,
            'processing_type': 'external_csv',
            'input_file': str(csv_path),
            'output_directory': str(output_dir),
            'embeddings_file': str(embeddings_file),
            'labels_file': str(labels_file),
            'metadata_file': str(metadata_file),
            'summary_file': str(summary_file),
            'processing_results': {
                'original_count': original_count,
                'valid_count': valid_count,
                'removed_count': removed_count,
                'success_rate': valid_count / original_count if original_count > 0 else 0,
                'has_labels': validation['has_labels'],
                'embedding_shape': all_embeddings.shape,
                'labels_shape': all_labels.shape
            },
            'model_info': {
                'model_name': model_name,
                'embedding_dimension': embedding_dim,
                'batch_size': batch_size,
                'max_length': max_length
            },
            'timing': {
                'total_time': total_time,
                'embedding_time': embedding_time,
                'model_load_time': self.model_load_time
            },
            'next_steps': {
                'mlp_training': f'python scripts/train_mlp.py --embeddings-dir {output_dir}/embeddings --output-dir {output_dir}',
                'svm_training': f'python scripts/train_svm.py --embeddings-dir {output_dir}/embeddings --output-dir {output_dir}',
                'generate_report': f'python scripts/report.py --models-dir {output_dir}/models --test-data {output_dir}/embeddings/X_embedded.npy --results-dir {output_dir}'
            }
        }
    
    def detect_operation_mode(self, input_dir: Path, inference_only: bool = False) -> Dict[str, Any]:
        """
        Detect whether we're in training or inference mode based on available files.
        
        Args:
            input_dir: Input directory to analyze
            inference_only: Force inference mode
            
        Returns:
            Dictionary with mode detection results
        """
        mode_info = {
            'mode': 'unknown',
            'files_available': [],
            'files_missing': [],
            'inference_file': None,
            'training_files': [],
            'reason': '',
            'valid_files': []
        }
        
        # Check for inference.csv
        inference_file = input_dir / "inference.csv"
        if inference_file.exists():
            mode_info['inference_file'] = str(inference_file)
            mode_info['files_available'].append('inference.csv')
            
            # Check if inference.csv is valid
            try:
                df = load_csv_robust(inference_file)
                if len(df) > 0 and 'text' in df.columns:
                    mode_info['valid_files'].append('inference.csv')
                    if inference_only or len(mode_info['files_available']) == 1:
                        mode_info['mode'] = 'inference'
                        mode_info['reason'] = 'inference.csv found and valid'
                        self.logger.info(f"🔍 Detected INFERENCE mode: {mode_info['reason']}")
                        return mode_info
            except Exception as e:
                self.logger.warning(f"⚠️ inference.csv exists but invalid: {e}")
        
        # Check for training files
        training_files = ['train.csv', 'val.csv', 'test.csv']
        valid_training_files = []
        
        for filename in training_files:
            filepath = input_dir / filename
            if filepath.exists():
                mode_info['files_available'].append(filename)
                try:
                    df = load_csv_robust(filepath)
                    if len(df) > 0 and 'text' in df.columns:
                        valid_training_files.append(filename)
                        mode_info['valid_files'].append(filename)
                        mode_info['training_files'].append(str(filepath))
                except Exception as e:
                    self.logger.warning(f"⚠️ {filename} exists but invalid: {e}")
            else:
                mode_info['files_missing'].append(filename)
        
        # Determine mode based on valid files
        if inference_only:
            mode_info['mode'] = 'inference'
            mode_info['reason'] = 'inference-only flag specified'
        elif len(valid_training_files) >= 2:  # Need at least 2 valid files for training
            mode_info['mode'] = 'training'
            mode_info['reason'] = f'Valid training files found: {valid_training_files}'
        elif len(valid_training_files) == 1 and 'test.csv' in valid_training_files:
            mode_info['mode'] = 'inference'
            mode_info['reason'] = 'Only test.csv is valid, treating as inference'
        elif mode_info['inference_file'] and 'inference.csv' in mode_info['valid_files']:
            mode_info['mode'] = 'inference'
            mode_info['reason'] = 'inference.csv is the only valid file'
        else:
            mode_info['mode'] = 'training'  # Default to training, but will likely fail validation
            mode_info['reason'] = f'Insufficient valid files for inference, defaulting to training mode'
        
        self.logger.info(f"🔍 Detected {mode_info['mode'].upper()} mode: {mode_info['reason']}")
        return mode_info
    
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
        Check for label distribution issues with inference mode support.
        
        Args:
            df: DataFrame with optional label column
            filename: Name of the file being checked
        
        Returns:
            Dictionary with balance analysis results
        """
        balance_info = {
            'total_samples': len(df),
            'has_labels': 'label' in df.columns,
            'unique_labels': 0,
            'label_distribution': {},
            'min_class_count': 0,
            'max_class_count': 0,
            'min_class_pct': 0,
            'max_class_pct': 0,
            'balance_warnings': []
        }
        
        if not balance_info['has_labels']:
            balance_info['balance_warnings'].append("No label column found (inference mode)")
            self.logger.info(f"🔍 {filename}: No labels found, inference mode")
            return balance_info
        
        # Check if label column has valid data
        if df['label'].isnull().all():
            balance_info['balance_warnings'].append("All labels are null (inference mode)")
            self.logger.info(f"🔍 {filename}: All labels are null, inference mode")
            return balance_info
        
        # Analyze label distribution
        label_counts = df['label'].value_counts()
        total = len(df)
        
        balance_info.update({
            'unique_labels': len(label_counts),
            'label_distribution': label_counts.to_dict(),
            'min_class_count': label_counts.min(),
            'max_class_count': label_counts.max(),
            'min_class_pct': (label_counts.min() / total) * 100,
            'max_class_pct': (label_counts.max() / total) * 100
        })
        
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
    
    def validate_input_data_comprehensive(self, input_dir: Path, mode_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive validation of input data with training/inference mode support.
        
        Args:
            input_dir: Directory containing CSV files
            mode_info: Mode detection results
        
        Returns:
            Validation results dictionary
        """
        self.logger.info(f"🔍 Comprehensive input validation: {input_dir} (Mode: {mode_info['mode']})")
        
        validation = {
            'valid': True,
            'mode': mode_info['mode'],
            'input_directory': str(input_dir),
            'files_found': [],
            'files_missing': [],
            'file_details': {},
            'column_issues': [],
            'data_quality': {},
            'label_balance': {},
            'total_samples': 0,
            'inference_mode': mode_info['mode'] == 'inference'
        }
        
        if not input_dir.exists():
            self.logger.error(f"❌ Input directory does not exist: {input_dir}")
            validation['valid'] = False
            validation['directory_missing'] = True
            return validation
        
        # Determine files to check based on mode
        if mode_info['mode'] == 'inference':
            if mode_info['inference_file']:
                files_to_check = ['inference.csv']
            else:
                # Check only valid files from mode detection
                files_to_check = [f for f in ['train.csv', 'val.csv', 'test.csv'] 
                                 if f in mode_info['valid_files']]
        else:
            files_to_check = ['train.csv', 'val.csv', 'test.csv']
        
        required_columns = ['text']  # 'label' is optional in inference mode
        
        for filename in files_to_check:
            filepath = input_dir / filename
            
            if not filepath.exists():
                validation['files_missing'].append(filename)
                if mode_info['mode'] == 'training':
                    validation['valid'] = False
                    self.logger.warning(f"⚠️ Missing file: {filepath}")
                else:
                    self.logger.info(f"🔍 Skipping missing file in inference mode: {filepath}")
                continue
            
            validation['files_found'].append(filename)
            
            try:
                # Load and analyze file
                df = load_csv_robust(filepath)
                
                # Skip empty files in inference mode
                if len(df) == 0:
                    if mode_info['mode'] == 'inference':
                        self.logger.warning(f"⚠️ Empty file skipped in inference mode: {filename}")
                        continue
                    else:
                        validation['valid'] = False
                        self.logger.error(f"❌ Empty file in training mode: {filename}")
                        continue
                
                file_info = {
                    'samples': len(df),
                    'columns': list(df.columns),
                    'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                    'has_labels': 'label' in df.columns
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
                    continue
                
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
                
                # Handle labels (optional in inference mode)
                if 'label' in df.columns:
                    label_stats = {
                        'unique_labels': df['label'].nunique(),
                        'label_distribution': df['label'].value_counts().to_dict(),
                        'null_labels': df['label'].isnull().sum()
                    }
                else:
                    label_stats = {
                        'unique_labels': 0,
                        'label_distribution': {},
                        'null_labels': 0,
                        'inference_mode': True
                    }
                
                # Label balance analysis
                balance_analysis = self.check_label_balance(df, filename)
                validation['label_balance'][filename] = balance_analysis
                
                file_info['text_quality'] = text_stats
                file_info['label_quality'] = label_stats
                file_info['label_balance'] = balance_analysis
                
                self.logger.info(f"✅ {filename}: {len(df):,} samples, quality OK")
                self.logger.info(f"   📊 Text: avg_len={text_stats['avg_length']:.0f}, nulls={text_stats['null_count']}")
                
                if file_info['has_labels']:
                    self.logger.info(f"   🏷️ Labels: {label_stats['unique_labels']} classes, nulls={label_stats['null_labels']}")
                else:
                    self.logger.info(f"   🔍 No labels (inference mode)")
                
                validation['file_details'][filename] = file_info
                validation['total_samples'] += len(df)
                
            except Exception as e:
                validation['column_issues'].append({
                    'file': filename,
                    'error': str(e)
                })
                validation['valid'] = False
                self.logger.error(f"❌ Error reading {filename}: {e}")
        
        # Final validation check
        if validation['inference_mode']:
            # In inference mode, we just need at least one valid file with text
            if validation['total_samples'] == 0:
                validation['valid'] = False
                self.logger.error(f"❌ No valid samples found for inference")
            else:
                self.logger.info(f"✅ Inference mode validation passed")
        else:
            # In training mode, we need proper training files
            if not validation['files_found'] or validation['total_samples'] == 0:
                validation['valid'] = False
                self.logger.error(f"❌ Training mode validation failed")
        
        # Overall data quality assessment
        if validation['valid']:
            total_samples = validation['total_samples']
            self.logger.info(f"✅ All input files validated successfully")
            self.logger.info(f"   📊 Total samples: {total_samples:,}")
            self.logger.info(f"   🎯 Mode: {validation['mode']}")
            
            # Calculate split proportions
            if validation['file_details']:
                for filename, details in validation['file_details'].items():
                    proportion = details['samples'] / total_samples * 100
                    self.logger.info(f"   📊 {filename}: {details['samples']:,} samples ({proportion:.1f}%)")
        
        return validation
    
    def verify_embedding_quality(self, embeddings: np.ndarray, split_name: str) -> Dict[str, Any]:
        """
        Verify embedding quality with comprehensive checks.
        
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
    
    def load_and_embed_split(self, input_dir: Path, split_name: str, model: SentenceTransformer, 
                           batch_size: int, embedding_dim: int, mode_info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load and embed a single split with robust error handling for training/inference modes.
        
        Args:
            input_dir: Input directory
            split_name: Name of split (train/val/test/inference)
            model: Loaded SentenceTransformer model
            batch_size: Batch size for processing
            embedding_dim: Embedding dimension
            mode_info: Mode detection results
            
        Returns:
            Tuple of (embeddings, labels, processing_info)
        """
        # Handle inference.csv special case
        if split_name == 'inference' and mode_info.get('inference_file'):
            input_file = Path(mode_info['inference_file'])
        else:
            input_file = input_dir / f"{split_name}.csv"
        
        processing_info = {
            'split': split_name,
            'input_file': str(input_file),
            'original_samples': 0,
            'processed_samples': 0,
            'removed_samples': 0,
            'has_labels': False,
            'embedding_shape': None,
            'labels_shape': None,
            'error': None
        }
        
        try:
            if not input_file.exists():
                raise FileNotFoundError(f"File not found: {input_file}")
            
            # Load data
            df = load_csv_robust(input_file)
            processing_info['original_samples'] = len(df)
            
            if len(df) == 0:
                self.logger.warning(f"⚠️ Empty file: {split_name}")
                return (np.array([]).reshape(0, embedding_dim), 
                       np.array([]), 
                       processing_info)
            
            if 'text' not in df.columns:
                raise ValueError(f"Missing 'text' column in {split_name}")
            
            # Extract texts
            texts = df['text'].fillna('').astype(str).tolist()
            
            # Handle labels
            if 'label' in df.columns and not df['label'].isnull().all():
                labels = df['label'].tolist()
                processing_info['has_labels'] = True
            else:
                # Create placeholder labels for inference mode
                labels = [-1] * len(texts)  # Use -1 as placeholder
                processing_info['has_labels'] = False
                self.logger.info(f"🔍 No labels found in {split_name}, using placeholder labels")
            
            # Filter out empty texts
            valid_indices = [i for i, text in enumerate(texts) if text.strip()]
            texts = [texts[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            
            processing_info['processed_samples'] = len(texts)
            processing_info['removed_samples'] = processing_info['original_samples'] - processing_info['processed_samples']
            
            if processing_info['removed_samples'] > 0:
                self.logger.warning(f"   ⚠️ Removed {processing_info['removed_samples']} empty texts from {split_name}")
            
            if len(texts) == 0:
                self.logger.warning(f"⚠️ No valid texts found in {split_name}")
                return (np.array([]).reshape(0, embedding_dim), 
                       np.array([]), 
                       processing_info)
            
            self.logger.info(f"   📊 Processing {len(texts):,} valid samples in {split_name}")
            
            # Generate embeddings with progress tracking
            embeddings = []
            
            if len(texts) > 100:
                # Use tqdm for large datasets
                progress_bar = tqdm(
                    range(0, len(texts), batch_size),
                    desc=f"Embedding {split_name}",
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
                    self.logger.error(f"❌ Error processing batch {i//batch_size + 1} in {split_name}: {batch_error}")
                    # Add zero embeddings for failed batch
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
    
    def generate_embeddings_advanced(self, input_dir: Path, output_dir: Path,
                                   model_name: str = DEFAULT_EMBEDDING_MODEL,
                                   batch_size: int = None, max_length: int = None,
                                   force_recreate: bool = False,
                                   inference_only: bool = False) -> Dict[str, Any]:
        """
        Generate embeddings with advanced progress tracking and training/inference mode support.
        
        Args:
            input_dir: Directory containing CSV files
            output_dir: Directory to save embeddings
            model_name: SentenceTransformer model name
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            force_recreate: Force recreation even if files exist
            inference_only: Force inference mode
        
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
        self.logger.info(f"   🔍 Inference only: {inference_only}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect operation mode
        mode_info = self.detect_operation_mode(input_dir, inference_only)
        
        # Validate input data
        validation = self.validate_input_data_comprehensive(input_dir, mode_info)
        if not validation['valid']:
            return {
                'success': False,
                'error': 'Input validation failed',
                'validation_results': validation,
                'mode_info': mode_info
            }
        
        # Check for existing embeddings
        if not force_recreate:
            existing_files = []
            
            # Check based on mode
            if mode_info['mode'] == 'inference':
                if mode_info.get('inference_file'):
                    # Check for inference embeddings
                    files_to_check = ['inference']
                else:
                    # Check for test embeddings (inference mode using test.csv)
                    files_to_check = ['test']
            else:
                # Training mode
                files_to_check = ['train', 'val', 'test']
            
            for split in files_to_check:
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
                        'mode_info': mode_info
                    }
        
        # Load model
        model = self.load_model_advanced(model_name, max_length)
        embedding_dim = model.encode(["test"], convert_to_numpy=True).shape[1]
        
        # Check if embedding dimension is too small
        if embedding_dim < 256:
            self.logger.warning(f"⚠️ Small embedding dimension ({embedding_dim}) - may affect MLP performance")
        
        # Process splits based on mode
        results = {
            'success': True,
            'mode': mode_info['mode'],
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
            'model_load_time': self.model_load_time,
            'mode_info': mode_info
        }
        
        # Determine splits to process
        if mode_info['mode'] == 'inference':
            if mode_info.get('inference_file'):
                splits_to_process = ['inference']
            else:
                # Use available valid files from training set
                splits_to_process = [f.replace('.csv', '') for f in mode_info['valid_files']]
        else:
            splits_to_process = ['train', 'val', 'test']
        
        embedding_start_time = time.time()
        
        for split in splits_to_process:
            self.logger.info(f"\n🔄 Processing {split} split...")
            
            # Load and embed split
            embeddings, labels, processing_info = self.load_and_embed_split(
                input_dir, split, model, batch_size, embedding_dim, mode_info
            )
            
            if processing_info.get('error'):
                self.logger.error(f"❌ Failed to process {split}: {processing_info['error']}")
                results['success'] = False
                results['errors'] = results.get('errors', [])
                results['errors'].append(processing_info)
                continue
            
            if len(embeddings) == 0:
                self.logger.warning(f"⚠️ No embeddings generated for {split}")
                continue
            
            # Verify embedding quality
            quality_check = self.verify_embedding_quality(embeddings, split)
            results['quality_checks'][split] = quality_check
            
            # Save embeddings and labels
            embeddings_file = output_dir / f"X_{split}.npy"
            labels_file = output_dir / f"y_{split}.npy"
            
            np.save(embeddings_file, embeddings)
            np.save(labels_file, labels)
            
            # Verify saved files
            saved_embeddings = np.load(embeddings_file)
            saved_labels = np.load(labels_file)
            
            if saved_embeddings.shape[0] != saved_labels.shape[0]:
                raise ValueError(f"Shape mismatch: embeddings {saved_embeddings.shape} vs labels {saved_labels.shape}")
            
            # Record split results
            split_info = {
                'split': split,
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
                },
                'quality_check': quality_check
            }
            
            results['splits_processed'].append(split_info)
            results['total_samples'] += processing_info['processed_samples']
            
            self.logger.info(f"   ✅ {split} completed:")
            self.logger.info(f"      💾 Embeddings: {embeddings_file} ({split_info['file_size_mb']['embeddings']:.1f}MB)")
            self.logger.info(f"      💾 Labels: {labels_file} ({split_info['file_size_mb']['labels']:.1f}MB)")
            self.logger.info(f"      📏 Shape: {embeddings.shape}")
            self.logger.info(f"      🏷️ Has labels: {processing_info['has_labels']}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        embedding_time = time.time() - embedding_start_time
        results['processing_time'] = processing_time
        results['embedding_generation_time'] = embedding_time
        
        # Calculate quality metrics
        if results['splits_processed']:
            total_original = sum(info['original_samples'] for info in results['splits_processed'])
            processing_success_rate = results['total_samples'] / total_original if total_original > 0 else 0
        else:
            processing_success_rate = 0
            total_original = 0
        
        # Save comprehensive metadata
        metadata = {
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'mode': mode_info['mode'],
                'model_name': model_name,
                'embedding_dimension': embedding_dim,
                'max_sequence_length': max_length,
                'batch_size': batch_size,
                'processing_time_seconds': processing_time,
                'model_load_time_seconds': self.model_load_time,
                'embedding_generation_time_seconds': embedding_time,
                'total_samples_processed': results['total_samples']
            },
            'mode_info': mode_info,
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
                'processing_success_rate': processing_success_rate,
                'inference_mode': mode_info['mode'] == 'inference'
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
        Safely get device information from model.
        
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
        Print comprehensive generation summary with mode information.
        
        Args:
            results: Generation results
            processing_time: Total processing time
            output_dir: Output directory
            metadata_file: Metadata file path
        """
        mode_emoji = "🔍" if results['mode'] == 'inference' else "🎯"
        
        if results['success']:
            self.logger.info(f"\n🎉 EMBEDDING GENERATION COMPLETED SUCCESSFULLY!")
            self.logger.info(f"   {mode_emoji} Mode: {results['mode'].upper()}")
            self.logger.info(f"   ⏱️ Total time: {processing_time:.1f}s (Model: {self.model_load_time:.1f}s, Embedding: {results.get('embedding_generation_time', 0):.1f}s)")
            self.logger.info(f"   📊 Total samples: {results['total_samples']:,}")
            self.logger.info(f"   📁 Output directory: {output_dir}")
            self.logger.info(f"   📄 Metadata saved: {metadata_file}")
            
            # Show splits processed
            self.logger.info(f"   🎯 Splits processed: {len(results['splits_processed'])}")
            for split_info in results['splits_processed']:
                quality_issues = len(split_info['quality_check'].get('issues_found', []))
                quality_status = "⚠️" if quality_issues > 0 else "✅"
                label_status = "🏷️" if split_info['has_labels'] else "🔍"
                self.logger.info(f"      • {split_info['split']}: {split_info['processed_samples']:,} samples {quality_status} {label_status}")
            
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
            self.logger.error(f"   {mode_emoji} Mode: {results.get('mode', 'unknown').upper()}")
            
            if 'errors' in results:
                for error in results['errors']:
                    self.logger.error(f"   Split {error['split']}: {error.get('error', 'Unknown error')}")
            
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
                                         inference_only: bool = False,
                                         verbose: bool = True) -> Dict[str, Any]:
    """
    Pipeline-compatible embedding generation function.
    Enhanced version with full training/inference mode support.
    
    Args:
        input_dir: Directory containing train.csv, val.csv, test.csv, or inference.csv
        output_dir: Directory to save embeddings
        model_name: SentenceTransformer model name
        batch_size: Batch size for embedding generation
        max_length: Maximum sequence length
        force_recreate: Force recreation even if embeddings exist
        inference_only: Force inference mode
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
            input_path, output_path, model_name, batch_size, max_length, force_recreate, inference_only
        )
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline embedding generation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'input_dir': input_dir,
            'output_dir': output_dir,
            'inference_only': inference_only
        }

def process_external_csv_file(csv_path: str, model_name: str = DEFAULT_EMBEDDING_MODEL,
                             batch_size: int = None, max_length: int = None,
                             force_recreate: bool = False) -> Dict[str, Any]:
    """
    🆕 NEW: Process a single external CSV file and generate embeddings.
    
    Args:
        csv_path: Path to the CSV file
        model_name: SentenceTransformer model name
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        force_recreate: Force recreation even if files exist
    
    Returns:
        Processing results dictionary
    """
    try:
        # Initialize generator
        generator = AdvancedEmbeddingGenerator()
        
        # Convert path
        csv_file_path = Path(csv_path)
        
        # Process the CSV file
        results = generator.process_external_csv_file(
            csv_file_path, model_name, batch_size, max_length, force_recreate
        )
        
        return results
        
    except Exception as e:
        logger.error(f"❌ External CSV processing failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'csv_path': csv_path
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
    mode_info = generator.detect_operation_mode(Path(input_dir))
    validation = generator.validate_input_data_comprehensive(Path(input_dir), mode_info)
    
    # Convert to legacy format
    legacy_validation = {
        'valid': validation['valid'],
        'mode': validation['mode'],
        'files_found': validation['files_found'],
        'files_missing': validation['files_missing'],
        'column_issues': validation['column_issues'],
        'sample_counts': {},
        'input_directory': validation['input_directory'],
        'inference_mode': validation['inference_mode']
    }
    
    # Extract sample counts from file details
    for filename, details in validation.get('file_details', {}).items():
        legacy_validation['sample_counts'][filename] = details['samples']
    
    # Add directory missing flag if needed
    if 'directory_missing' in validation:
        legacy_validation['directory_missing'] = validation['directory_missing']
    
    return legacy_validation


def main(argv=None):
    """Enhanced main function with comprehensive CLI support and external CSV integration."""
    parser = argparse.ArgumentParser(
        description="Advanced Embedding Generation for Sentiment Analysis (TRAINING/INFERENCE/EXTERNAL CSV SUPPORT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard embedding generation (auto-detect mode)
  %(prog)s
  
  # Custom input/output directories (pipeline integration)
  %(prog)s --input-dir results/session_20241229/processed --output-dir results/session_20241229/embeddings
  
  # NEW: External CSV file processing
  %(prog)s --input-file data/external/reviews.csv
  %(prog)s --input-file uploaded_file.csv --model-name all-mpnet-base-v2
  
  # Force inference mode
  %(prog)s --inference-only --input-dir data/inference
  
  # Force regeneration with different model
  %(prog)s --force-recreate --model-name all-mpnet-base-v2
  
  # Validation only
  %(prog)s --validate-only --verbose
  
  # Pipeline automation compatible
  %(prog)s --input-dir processed/ --output-dir embeddings/ --quiet --force-recreate
        """
    )
    
    # 🆕 NEW: External CSV file processing
    parser.add_argument("--input-file", type=str, default=None,
                       help="🆕 NEW: Path to external CSV file for processing (alternative to --input-dir)")
    
    # Input/Output paths (dynamic defaults)
    parser.add_argument("--input-dir", default=str(PROCESSED_DATA_DIR),
                       help=f"Directory containing CSV files (default: {PROCESSED_DATA_DIR})")
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
    
    # Mode configuration
    parser.add_argument("--inference-only", action="store_true",
                       help="Force inference mode (automatically detected if inference.csv exists)")
    
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
    
    # 🆕 NEW: Handle external CSV file processing
    if args.input_file:
        if verbose:
            logger.info("🆕 Processing external CSV file with REAL pipeline")
            logger.info(f"📄 Input file: {args.input_file}")
            logger.info(f"🤖 Model: {args.model_name}")
            logger.info(f"⚙️ Batch size: {args.batch_size}")
            logger.info(f"📏 Max length: {args.max_length}")
        
        try:
            # Process the external CSV file
            results = process_external_csv_file(
                csv_path=args.input_file,
                model_name=args.model_name,
                batch_size=args.batch_size,
                max_length=args.max_length,
                force_recreate=args.force_recreate
            )
            
            if results['success']:
                logger.info(f"🎉 EXTERNAL CSV PROCESSING COMPLETED SUCCESSFULLY!")
                logger.info(f"   📄 Input file: {results['input_file']}")
                logger.info(f"   📁 Output directory: {results['output_directory']}")
                logger.info(f"   📊 Processed: {results['processing_results']['valid_count']}/{results['processing_results']['original_count']} texts")
                logger.info(f"   ⏱️ Total time: {results['timing']['total_time']:.1f}s")
                logger.info(f"   🚀 Next steps:")
                logger.info(f"      • Train MLP: {results['next_steps']['mlp_training']}")
                logger.info(f"      • Train SVM: {results['next_steps']['svm_training']}")
                logger.info(f"      • Generate report: {results['next_steps']['generate_report']}")
                
                return 0
            else:
                logger.error("❌ External CSV processing failed")
                if 'error' in results:
                    logger.error(f"Error: {results['error']}")
                
                logger.info("💡 Troubleshooting tips:")
                logger.info("   1. Ensure CSV has a text column (text, review, content, comment)")
                logger.info("   2. Check file format and encoding")
                logger.info("   3. Verify file is not empty or corrupted")
                
                return 1
                
        except Exception as e:
            logger.error(f"❌ External CSV processing error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    # Standard embedding generation (existing functionality)
    if verbose:
        logger.info("🚀 Advanced Sentiment Analysis Embedding Generation (TRAINING/INFERENCE SUPPORT)")
        logger.info(f"📁 Project root: {PROJECT_ROOT}")
        logger.info(f"📄 Input directory: {args.input_dir}")
        logger.info(f"📄 Output directory: {args.output_dir}")
        logger.info(f"🤖 Model: {args.model_name}")
        logger.info(f"⚙️ Batch size: {args.batch_size}")
        logger.info(f"📏 Max length: {args.max_length}")
        logger.info(f"🔍 Inference only: {args.inference_only}")
    
    try:
        # Initialize generator
        generator = AdvancedEmbeddingGenerator()
        
        # Detect mode first
        mode_info = generator.detect_operation_mode(Path(args.input_dir), args.inference_only)
        
        # Validate input files
        if verbose:
            logger.info(f"🔍 Validating input files... (Mode: {mode_info['mode']})")
        
        validation = generator.validate_input_data_comprehensive(Path(args.input_dir), mode_info)
        
        if not validation['valid']:
            logger.error(f"❌ Input validation failed for {mode_info['mode']} mode")
            logger.error("💡 Suggestions:")
            
            if mode_info['mode'] == 'inference':
                logger.error("   1. For inference mode, ensure you have:")
                logger.error("      • inference.csv with 'text' column, OR")
                logger.error("      • test.csv with 'text' column")
                logger.error("   2. Labels are optional in inference mode")
            else:
                logger.error("   1. For training mode, run preprocessing first: python scripts/preprocess.py")
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
                    if 'inference mode' not in warning:  # Don't show inference mode warnings as errors
                        logger.error(f"   Label issue in {filename}: {warning}")
            
            return 1
        
        if args.validate_only:
            logger.info(f"✅ Validation completed successfully! (Mode: {mode_info['mode']})")
            logger.info(f"   📊 Total samples: {validation['total_samples']:,}")
            for filename, details in validation['file_details'].items():
                label_info = "🏷️" if details['has_labels'] else "🔍"
                logger.info(f"   📊 {filename}: {details['samples']:,} samples {label_info}")
                
                # Show label balance info
                balance_info = details.get('label_balance', {})
                if balance_info.get('balance_warnings'):
                    for warning in balance_info['balance_warnings']:
                        if 'inference mode' not in warning:
                            logger.warning(f"      ⚠️ {warning}")
            return 0
        
        # Generate embeddings
        results = generator.generate_embeddings_advanced(
            Path(args.input_dir),
            Path(args.output_dir),
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            force_recreate=args.force_recreate,
            inference_only=args.inference_only
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
