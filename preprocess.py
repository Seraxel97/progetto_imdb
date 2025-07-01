#!/usr/bin/env python3
"""
Advanced Text Preprocessing Script - INTEGRATED WITH ENHANCED ARCHITECTURE
Advanced text preprocessing with seamless integration into the sentiment analysis pipeline.

FEATURES:
- Dynamic path detection and standardized PROJECT_ROOT handling
- Integration with enhanced_utils_unified.py for comprehensive pipeline support
- Intelligent column auto-detection (text/review/content + label/sentiment/target)
- Timestamp-based result organization compatible with pipeline_runner.py
- Multiple preprocessing modes (clean-only, full preprocessing, IMDb dataset preparation)
- Support for sys.argv injection from pipeline automation
- Comprehensive error handling and logging
- Compatible with GUI integration and command-line usage

USAGE:
    # IMDb dataset preprocessing for pipeline
    python scripts/preprocess.py --imdb data/raw/imdb_dataset.csv --output-dir data/processed
    
    # General CSV preprocessing
    python scripts/preprocess.py --file dataset.csv --text-col "review" --label-col "sentiment"
    
    # Pipeline automation compatible
    python scripts/preprocess.py --input dataset.csv --output-dir results/session_20241229/processed
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
import logging

warnings.filterwarnings('ignore')

# FIXED: Dynamic project root detection using standardized approach
try:
    CURRENT_FILE = Path(__file__).resolve()
    if CURRENT_FILE.parent.name == 'scripts':
        PROJECT_ROOT = CURRENT_FILE.parent.parent
    else:
        PROJECT_ROOT = CURRENT_FILE.parent
except:
    PROJECT_ROOT = Path.cwd()

# FIXED: Dynamic path construction
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
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

class AdvancedTextPreprocessor:
    """
    Advanced text preprocessing class with full pipeline integration.
    Enhanced version with dynamic paths, intelligent column detection, and robust error handling.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize the advanced text preprocessor.
        
        Args:
            project_root: Project root directory (auto-detect if None)
        """
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.logger = logger
        self.setup_paths()
        
        self.logger.info(f"📁 Advanced Text Preprocessor initialized")
        self.logger.info(f"   Project root: {self.project_root}")
    
    def setup_paths(self):
        """Setup comprehensive path structure."""
        self.paths = {
            'project_root': self.project_root,
            'data_dir': self.project_root / "data",
            'raw_data': self.project_root / "data" / "raw",
            'processed_data': self.project_root / "data" / "processed",
            'results_dir': self.project_root / "results",
            'scripts_dir': self.project_root / "scripts"
        }
        
        # Ensure directories exist
        for path_name, path in self.paths.items():
            if path_name != 'project_root':
                path.mkdir(parents=True, exist_ok=True)
    
    def intelligent_column_detection(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """
        Intelligently detect text and label columns with comprehensive fallback logic.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Tuple of (text_column, label_column) or (None, None) if not found
        """
        # Text column candidates (order matters - most specific first)
        text_candidates = [
            'text', 'review', 'content', 'comment', 'message', 'description',
            'body', 'tweet', 'post', 'article', 'summary', 'feedback'
        ]
        
        # Label column candidates
        label_candidates = [
            'label', 'sentiment', 'class', 'target', 'category', 'rating',
            'score', 'polarity', 'emotion', 'classification'
        ]
        
        # Find text column
        text_col = None
        for candidate in text_candidates:
            # Exact match first
            if candidate in df.columns:
                text_col = candidate
                break
            # Partial match (case insensitive)
            for col in df.columns:
                if candidate.lower() in col.lower():
                    text_col = col
                    break
            if text_col:
                break
        
        # If no standard names found, look for string columns with long text
        if text_col is None:
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if this column contains long text (average > 50 chars)
                    sample_texts = df[col].dropna().astype(str).head(100)
                    if len(sample_texts) > 0:
                        avg_length = sample_texts.str.len().mean()
                        if avg_length > 50:
                            text_col = col
                            self.logger.info(f"   📝 Auto-detected text column '{col}' (avg length: {avg_length:.0f})")
                            break
        
        # Find label column
        label_col = None
        for candidate in label_candidates:
            # Exact match first
            if candidate in df.columns:
                label_col = candidate
                break
            # Partial match (case insensitive)
            for col in df.columns:
                if candidate.lower() in col.lower():
                    label_col = col
                    break
            if label_col:
                break
        
        # If no standard names found, look for categorical columns with few unique values
        if label_col is None:
            for col in df.columns:
                if col != text_col and df[col].dtype in ['object', 'int64', 'float64']:
                    unique_vals = df[col].nunique()
                    # Reasonable number of classes for sentiment (2-10)
                    if 2 <= unique_vals <= 10:
                        # Check if values look like labels
                        sample_vals = set(df[col].dropna().astype(str).str.lower().head(20))
                        sentiment_indicators = {
                            'positive', 'negative', 'neutral', 'pos', 'neg', 'neu',
                            '1', '0', '2', 'good', 'bad', 'ok'
                        }
                        if sample_vals & sentiment_indicators:
                            label_col = col
                            self.logger.info(f"   🏷️ Auto-detected label column '{col}' ({unique_vals} classes)")
                            break
        
        self.logger.info(f"🔍 Column detection results:")
        self.logger.info(f"   Text column: {text_col}")
        self.logger.info(f"   Label column: {label_col}")
        
        return text_col, label_col
    
    def load_data_enhanced(self, file_path: str, text_col: Optional[str] = None, 
                          label_col: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Enhanced data loading with automatic format detection and column identification.
        
        Args:
            file_path: Path to the data file
            text_col: Text column name (auto-detect if None)
            label_col: Label column name (auto-detect if None)
        
        Returns:
            Loaded DataFrame or None if loading fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"❌ File not found: {file_path}")
            return None
        
        self.logger.info(f"📄 Loading data from: {file_path}")
        
        try:
            # Determine file format and load
            if file_path.suffix.lower() == '.csv':
                # Try different encodings and separators
                for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
                    for sep in [',', ';', '\t']:
                        try:
                            data = pd.read_csv(file_path, encoding=encoding, sep=sep)
                            if len(data.columns) > 1:  # Valid CSV should have multiple columns
                                self.logger.info(f"   ✅ CSV loaded with encoding='{encoding}', sep='{sep}'")
                                break
                        except:
                            continue
                    else:
                        continue
                    break
                else:
                    raise ValueError("Could not load CSV with any encoding/separator combination")
                    
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path)
                self.logger.info("   ✅ Excel file loaded")
                
            elif file_path.suffix.lower() == '.json':
                data = pd.read_json(file_path)
                self.logger.info("   ✅ JSON file loaded")
                
            elif file_path.suffix.lower() == '.txt':
                # For text files, create DataFrame with one column
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                data = pd.DataFrame({'text': lines})
                self.logger.info("   ✅ Text file loaded as single column")
                
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            self.logger.info(f"   📊 Data shape: {data.shape}")
            self.logger.info(f"   📋 Columns: {list(data.columns)}")
            
            # Auto-detect columns if not specified
            if text_col is None or label_col is None:
                detected_text, detected_label = self.intelligent_column_detection(data)
                
                if text_col is None:
                    text_col = detected_text
                if label_col is None:
                    label_col = detected_label
            
            # Validate detected columns
            if text_col and text_col not in data.columns:
                self.logger.error(f"❌ Specified text column '{text_col}' not found")
                self.logger.info(f"   Available columns: {list(data.columns)}")
                return None
            
            if label_col and label_col not in data.columns:
                self.logger.warning(f"⚠️ Specified label column '{label_col}' not found, continuing without labels")
                label_col = None
            
            # Add metadata to DataFrame
            data.attrs['text_column'] = text_col
            data.attrs['label_column'] = label_col
            data.attrs['source_file'] = str(file_path)
            
            self.logger.info(f"✅ Data loaded successfully: {len(data):,} records")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading data: {e}")
            return None
    
    def clean_text_advanced(self, text: str, mode: str = 'balanced') -> str:
        """
        Advanced text cleaning with multiple modes.
        
        Args:
            text: Text to clean
            mode: Cleaning mode ('light', 'balanced', 'aggressive')
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        import re
        
        result = text
        
        # Basic normalization (all modes)
        result = result.strip()
        result = re.sub(r'\s+', ' ', result)  # Normalize whitespace
        
        if mode in ['balanced', 'aggressive']:
            # Remove URLs
            result = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', result)
            
            # Remove email addresses
            result = re.sub(r'\S+@\S+', '', result)
            
            # Remove excessive punctuation
            result = re.sub(r'[!]{2,}', '!', result)
            result = re.sub(r'[?]{2,}', '?', result)
            result = re.sub(r'[.]{3,}', '...', result)
            
            # Convert to lowercase
            result = result.lower()
        
        if mode == 'aggressive':
            # Remove all punctuation except sentence endings
            result = re.sub(r'[^\w\s.!?]', ' ', result)
            
            # Remove numbers
            result = re.sub(r'\d+', '', result)
            
            # Remove extra short words
            words = result.split()
            words = [w for w in words if len(w) > 2]
            result = ' '.join(words)
        
        # Final cleanup
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def preprocess_comprehensive(self, data: pd.DataFrame, text_col: str,
                               label_col: Optional[str] = None,
                               cleaning_mode: str = 'balanced',
                               min_length: int = 10,
                               max_length: int = 5000) -> pd.DataFrame:
        """
        Comprehensive text preprocessing with quality filtering.
        
        Args:
            data: Input DataFrame
            text_col: Text column name
            label_col: Label column name (optional)
            cleaning_mode: Text cleaning mode ('light', 'balanced', 'aggressive')
            min_length: Minimum text length (characters)
            max_length: Maximum text length (characters)
        
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info(f"🔄 Starting comprehensive preprocessing...")
        self.logger.info(f"   Text column: {text_col}")
        self.logger.info(f"   Label column: {label_col}")
        self.logger.info(f"   Cleaning mode: {cleaning_mode}")
        
        result_data = data.copy()
        
        # Statistics tracking
        original_count = len(result_data)
        processed_texts = []
        quality_stats = {
            'too_short': 0,
            'too_long': 0,
            'empty_after_cleaning': 0,
            'processed_successfully': 0
        }
        
        # Process each text
        for idx, text in enumerate(result_data[text_col]):
            if pd.isna(text):
                processed_texts.append("")
                continue
            
            original_text = str(text)
            
            # Clean text
            cleaned_text = self.clean_text_advanced(original_text, cleaning_mode)
            
            # Quality filtering
            if len(cleaned_text.strip()) == 0:
                quality_stats['empty_after_cleaning'] += 1
                processed_texts.append("")
            elif len(cleaned_text) < min_length:
                quality_stats['too_short'] += 1
                processed_texts.append("")
            elif len(cleaned_text) > max_length:
                quality_stats['too_long'] += 1
                # Truncate instead of removing
                processed_texts.append(cleaned_text[:max_length])
                quality_stats['processed_successfully'] += 1
            else:
                processed_texts.append(cleaned_text)
                quality_stats['processed_successfully'] += 1
            
            # Progress logging
            if (idx + 1) % 1000 == 0:
                self.logger.info(f"   Processed {idx + 1:,}/{len(result_data):,} texts...")
        
        # Add processed column
        result_data['text_processed'] = processed_texts
        
        # Add quality metrics
        result_data['original_length'] = result_data[text_col].str.len()
        result_data['processed_length'] = result_data['text_processed'].str.len()
        result_data['length_reduction'] = ((result_data['original_length'] - result_data['processed_length']) / 
                                         result_data['original_length']).fillna(0)
        
        # Log quality statistics
        self.logger.info(f"📊 Preprocessing quality statistics:")
        self.logger.info(f"   Original texts: {original_count:,}")
        self.logger.info(f"   Successfully processed: {quality_stats['processed_successfully']:,}")
        self.logger.info(f"   Too short (removed): {quality_stats['too_short']:,}")
        self.logger.info(f"   Too long (truncated): {quality_stats['too_long']:,}")
        self.logger.info(f"   Empty after cleaning: {quality_stats['empty_after_cleaning']:,}")
        
        success_rate = quality_stats['processed_successfully'] / original_count * 100
        self.logger.info(f"   Success rate: {success_rate:.1f}%")
        
        # Store quality stats in DataFrame attributes
        result_data.attrs['quality_stats'] = quality_stats
        result_data.attrs['preprocessing_mode'] = cleaning_mode
        result_data.attrs['success_rate'] = success_rate
        
        return result_data
    
    def create_train_val_test_splits(self, data: pd.DataFrame, text_col: str = 'text_processed',
                                   label_col: Optional[str] = None,
                                   train_size: float = 0.7, val_size: float = 0.15,
                                   test_size: float = 0.15, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/validation/test splits with quality filtering.
        
        Args:
            data: Preprocessed DataFrame
            text_col: Text column to use for splitting
            label_col: Label column for stratification (optional)
            train_size: Training set proportion
            val_size: Validation set proportion
            test_size: Test set proportion
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        self.logger.info(f"📊 Creating dataset splits...")
        
        # Validate split sizes
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError("Split sizes must sum to 1.0")
        
        # Filter out empty processed texts
        valid_data = data[data[text_col].str.strip() != ''].copy()
        removed_count = len(data) - len(valid_data)
        
        if removed_count > 0:
            self.logger.info(f"   Removed {removed_count:,} empty processed texts")
        
        self.logger.info(f"   Valid samples for splitting: {len(valid_data):,}")
        
        if len(valid_data) == 0:
            raise ValueError("No valid samples remaining after filtering")
        
        # Prepare final dataset with standardized column names
        final_data = pd.DataFrame()
        final_data['text'] = valid_data[text_col]
        
        if label_col and label_col in valid_data.columns:
            # Normalize labels to 0/1 if they're string labels
            labels = valid_data[label_col]
            unique_labels = labels.dropna().unique()
            
            if set(str(label).lower() for label in unique_labels) <= {'positive', 'negative', 'pos', 'neg'}:
                # Map string labels to integers
                label_mapping = {}
                for label in unique_labels:
                    if str(label).lower() in ['negative', 'neg']:
                        label_mapping[label] = 0
                    elif str(label).lower() in ['positive', 'pos']:
                        label_mapping[label] = 1
                
                final_data['label'] = labels.map(label_mapping).fillna(labels)
                self.logger.info(f"   Mapped string labels to integers: {label_mapping}")
            else:
                final_data['label'] = labels
            
            # Use stratified splits
            stratify = final_data['label'] if final_data['label'].nunique() > 1 else None
        else:
            stratify = None
            self.logger.info("   No labels available - using random splits")
        
        # Create splits
        if stratify is not None:
            # Two-step stratified split
            train_df, temp_df = train_test_split(
                final_data, 
                train_size=train_size,
                random_state=random_state,
                stratify=stratify
            )
            
            # Calculate relative sizes for val/test split
            val_test_ratio = val_size / (val_size + test_size)
            
            val_df, test_df = train_test_split(
                temp_df,
                train_size=val_test_ratio,
                random_state=random_state,
                stratify=temp_df['label'] if temp_df['label'].nunique() > 1 else None
            )
        else:
            # Simple random splits
            train_df, temp_df = train_test_split(
                final_data, 
                train_size=train_size,
                random_state=random_state
            )
            
            val_test_ratio = val_size / (val_size + test_size)
            val_df, test_df = train_test_split(
                temp_df,
                train_size=val_test_ratio,
                random_state=random_state
            )
        
        # Log split statistics
        self.logger.info(f"📊 Dataset splits created:")
        self.logger.info(f"   Train: {len(train_df):,} samples ({len(train_df)/len(final_data)*100:.1f}%)")
        self.logger.info(f"   Val: {len(val_df):,} samples ({len(val_df)/len(final_data)*100:.1f}%)")
        self.logger.info(f"   Test: {len(test_df):,} samples ({len(test_df)/len(final_data)*100:.1f}%)")
        
        # Log label distribution if available
        if 'label' in final_data.columns:
            for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
                label_dist = split_df['label'].value_counts().to_dict()
                self.logger.info(f"   {split_name} label distribution: {label_dist}")
        
        return train_df, val_df, test_df
    
    def save_processed_dataset(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                             test_df: pd.DataFrame, output_dir: str,
                             original_file: str, metadata: Optional[Dict] = None) -> Dict[str, str]:
        """
        Save processed dataset splits with comprehensive metadata.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            output_dir: Output directory
            original_file: Original source file path
            metadata: Additional metadata to save
        
        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"💾 Saving processed dataset to: {output_dir}")
        
        # Save CSV files
        saved_files = {}
        
        train_path = output_dir / "train.csv"
        val_path = output_dir / "val.csv"
        test_path = output_dir / "test.csv"
        
        train_df.to_csv(train_path, index=False, encoding='utf-8')
        val_df.to_csv(val_path, index=False, encoding='utf-8')
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        
        saved_files['train'] = str(train_path)
        saved_files['val'] = str(val_path)
        saved_files['test'] = str(test_path)
        
        self.logger.info(f"   ✅ Train: {train_path} ({len(train_df):,} samples)")
        self.logger.info(f"   ✅ Val: {val_path} ({len(val_df):,} samples)")
        self.logger.info(f"   ✅ Test: {test_path} ({len(test_df):,} samples)")
        
        # Create comprehensive metadata
        processing_metadata = {
            'preprocessing_info': {
                'timestamp': datetime.now().isoformat(),
                'source_file': str(original_file),
                'output_directory': str(output_dir),
                'project_root': str(self.project_root),
                'total_original_samples': len(train_df) + len(val_df) + len(test_df),
                'splits': {
                    'train': len(train_df),
                    'val': len(val_df),
                    'test': len(test_df)
                }
            },
            'data_quality': {
                'text_column_used': train_df.attrs.get('text_column', 'text'),
                'label_column_used': train_df.attrs.get('label_column', None),
                'has_labels': 'label' in train_df.columns,
                'preprocessing_mode': train_df.attrs.get('preprocessing_mode', 'unknown'),
                'success_rate': train_df.attrs.get('success_rate', 0)
            },
            'files_generated': saved_files
        }
        
        # Add custom metadata if provided
        if metadata:
            processing_metadata.update(metadata)
        
        # Add label statistics if available
        if 'label' in train_df.columns:
            processing_metadata['label_statistics'] = {}
            for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
                label_dist = split_df['label'].value_counts().to_dict()
                processing_metadata['label_statistics'][split_name] = {
                    str(k): int(v) for k, v in label_dist.items()
                }
        
        # Save metadata
        metadata_path = output_dir / "preprocessing_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(processing_metadata, f, indent=2, ensure_ascii=False, default=str)
        
        saved_files['metadata'] = str(metadata_path)
        self.logger.info(f"   ✅ Metadata: {metadata_path}")
        
        # Create summary report
        summary_lines = [
            "PREPROCESSING SUMMARY",
            "=" * 40,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Source: {original_file}",
            f"Output: {output_dir}",
            "",
            "DATASET SPLITS:",
            f"  Train: {len(train_df):,} samples",
            f"  Val: {len(val_df):,} samples", 
            f"  Test: {len(test_df):,} samples",
            f"  Total: {len(train_df) + len(val_df) + len(test_df):,} samples",
            ""
        ]
        
        if 'label' in train_df.columns:
            summary_lines.extend([
                "LABEL DISTRIBUTION:",
                f"  Train: {dict(train_df['label'].value_counts())}",
                f"  Val: {dict(val_df['label'].value_counts())}",
                f"  Test: {dict(test_df['label'].value_counts())}",
                ""
            ])
        
        summary_lines.extend([
            "FILES GENERATED:",
            f"  • {train_path.name}",
            f"  • {val_path.name}",
            f"  • {test_path.name}",
            f"  • {metadata_path.name}",
            "",
            "=" * 40
        ])
        
        summary_path = output_dir / "preprocessing_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        saved_files['summary'] = str(summary_path)
        self.logger.info(f"   ✅ Summary: {summary_path}")
        
        return saved_files


# =============================================================================
# PIPELINE INTEGRATION FUNCTIONS
# =============================================================================

def preprocess_dataset_for_pipeline(input_path: str, output_dir: str, 
                                   text_col: Optional[str] = None,
                                   label_col: Optional[str] = None,
                                   cleaning_mode: str = 'balanced') -> Dict[str, Any]:
    """
    Main preprocessing function for pipeline integration.
    Enhanced version with comprehensive error handling and metadata generation.
    
    Args:
        input_path: Path to input CSV/Excel/JSON file
        output_dir: Output directory for processed files
        text_col: Text column name (auto-detect if None)
        label_col: Label column name (auto-detect if None)
        cleaning_mode: Text cleaning mode ('light', 'balanced', 'aggressive')
    
    Returns:
        Dictionary with processing results and metadata
    """
    try:
        # Initialize preprocessor
        preprocessor = AdvancedTextPreprocessor()
        
        # Load data with intelligent column detection
        data = preprocessor.load_data_enhanced(input_path, text_col, label_col)
        
        if data is None:
            return {
                'success': False,
                'error': 'Failed to load input data',
                'input_path': input_path
            }
        
        # Get detected column names
        detected_text_col = data.attrs.get('text_column')
        detected_label_col = data.attrs.get('label_column')
        
        if not detected_text_col:
            return {
                'success': False,
                'error': 'No text column found or detected',
                'available_columns': list(data.columns)
            }
        
        # Comprehensive preprocessing
        processed_data = preprocessor.preprocess_comprehensive(
            data, 
            detected_text_col,
            detected_label_col,
            cleaning_mode=cleaning_mode
        )
        
        # Create dataset splits
        train_df, val_df, test_df = preprocessor.create_train_val_test_splits(
            processed_data,
            text_col='text_processed',
            label_col=detected_label_col
        )
        
        # Save processed dataset
        saved_files = preprocessor.save_processed_dataset(
            train_df, val_df, test_df, output_dir, input_path
        )
        
        # Compile results
        results = {
            'success': True,
            'input_path': input_path,
            'output_directory': output_dir,
            'columns_detected': {
                'text_column': detected_text_col,
                'label_column': detected_label_col
            },
            'dataset_info': {
                'original_samples': len(data),
                'processed_samples': len(train_df) + len(val_df) + len(test_df),
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'has_labels': detected_label_col is not None
            },
            'quality_metrics': processed_data.attrs.get('quality_stats', {}),
            'files_generated': saved_files,
            'preprocessing_mode': cleaning_mode
        }
        
        # Add label distribution if available
        if detected_label_col and 'label' in train_df.columns:
            results['label_distribution'] = {
                'train': dict(train_df['label'].value_counts()),
                'val': dict(val_df['label'].value_counts()),
                'test': dict(test_df['label'].value_counts())
            }
        
        logger.info(f"✅ Preprocessing completed successfully")
        logger.info(f"   Input: {input_path}")
        logger.info(f"   Output: {output_dir}")
        logger.info(f"   Samples: {results['dataset_info']['original_samples']:,} → {results['dataset_info']['processed_samples']:,}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'input_path': input_path,
            'output_directory': output_dir
        }

# Legacy compatibility function (maintained for backward compatibility)
def preprocess_dataset(input_path: str, output_dir: str) -> None:
    """
    Legacy preprocessing function for backward compatibility with existing code.
    
    Args:
        input_path: Path to input file
        output_dir: Output directory
    """
    result = preprocess_dataset_for_pipeline(input_path, output_dir)
    
    if not result['success']:
        raise RuntimeError(f"Preprocessing failed: {result.get('error', 'Unknown error')}")
    
    logger.info("✅ Legacy preprocessing compatibility completed")


def main(argv=None):
    """Enhanced main function with comprehensive CLI support and pipeline integration."""
    parser = argparse.ArgumentParser(
        description="Advanced Text Preprocessing Tool - Pipeline Integrated",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # IMDb dataset preprocessing (pipeline integration)
  %(prog)s --imdb data/raw/imdb_dataset.csv --output-dir data/processed
  
  # General CSV preprocessing with auto-detection
  %(prog)s --file dataset.csv --output results/processed/
  
  # Specify columns explicitly
  %(prog)s --file reviews.csv --text-col "review" --label-col "sentiment"
  
  # Pipeline automation compatible
  %(prog)s --input dataset.csv --output-dir results/session_20241229/processed
  
  # Different cleaning modes
  %(prog)s --file data.csv --cleaning-mode aggressive --output clean_data/
        """
    )
    
    # Primary input options
    parser.add_argument('--file', '-f', help='Input file to preprocess')
    parser.add_argument('--input', help='Input file (alternative to --file, for pipeline compatibility)')
    parser.add_argument('--output', '-o', help='Output file or directory')
    parser.add_argument('--output-dir', help='Output directory (alternative to --output)')
    
    # IMDb specific mode for backward compatibility
    parser.add_argument('--imdb', help='IMDb dataset preprocessing mode (provide raw file path)')
    
    # Column specification
    parser.add_argument('--text-col', default=None, help='Text column name (auto-detect if not specified)')
    parser.add_argument('--label-col', default=None, help='Label column name (auto-detect if not specified)')
    
    # Preprocessing options
    parser.add_argument('--cleaning-mode', choices=['light', 'balanced', 'aggressive'], 
                       default='balanced', help='Text cleaning intensity (default: balanced)')
    parser.add_argument('--min-length', type=int, default=10, help='Minimum text length (default: 10)')
    parser.add_argument('--max-length', type=int, default=5000, help='Maximum text length (default: 5000)')
    
    # Split configuration
    parser.add_argument('--train-size', type=float, default=0.7, help='Training set proportion (default: 0.7)')
    parser.add_argument('--val-size', type=float, default=0.15, help='Validation set proportion (default: 0.15)')
    parser.add_argument('--test-size', type=float, default=0.15, help='Test set proportion (default: 0.15)')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed (default: 42)')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress non-essential output')
    
    # Parse arguments (handle sys.argv injection)
    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv[1:])  # Skip script name
    
    # Configure logging
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Determine input and output paths
        input_path = args.file or args.input or args.imdb
        output_path = args.output_dir or args.output
        
        if not input_path:
            logger.error("❌ No input file specified. Use --file, --input, or --imdb")
            return 1
        
        # Handle IMDb mode for backward compatibility
        if args.imdb:
            output_path = output_path or "data/processed"
            logger.info(f"🔄 IMDb preprocessing mode")
            logger.info(f"   Input: {args.imdb}")
            logger.info(f"   Output: {output_path}")
            
            result = preprocess_dataset_for_pipeline(
                args.imdb, 
                output_path,
                text_col=args.text_col,
                label_col=args.label_col,
                cleaning_mode=args.cleaning_mode
            )
        else:
            # General preprocessing mode
            if not output_path:
                # Auto-generate output directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"processed_{timestamp}"
            
            logger.info(f"🔄 General preprocessing mode")
            logger.info(f"   Input: {input_path}")
            logger.info(f"   Output: {output_path}")
            logger.info(f"   Cleaning mode: {args.cleaning_mode}")
            
            result = preprocess_dataset_for_pipeline(
                input_path,
                output_path,
                text_col=args.text_col,
                label_col=args.label_col,
                cleaning_mode=args.cleaning_mode
            )
        
        # Handle results
        if result['success']:
            logger.info("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
            logger.info(f"   📁 Output directory: {result['output_directory']}")
            logger.info(f"   📊 Processed samples: {result['dataset_info']['processed_samples']:,}")
            
            # Show dataset splits
            dataset_info = result['dataset_info']
            logger.info(f"   📊 Dataset splits:")
            logger.info(f"      Train: {dataset_info['train_samples']:,}")
            logger.info(f"      Val: {dataset_info['val_samples']:,}")  
            logger.info(f"      Test: {dataset_info['test_samples']:,}")
            
            # Show quality metrics
            if 'quality_metrics' in result:
                quality = result['quality_metrics']
                success_rate = (quality.get('processed_successfully', 0) / 
                              result['dataset_info']['original_samples'] * 100)
                logger.info(f"   📈 Processing success rate: {success_rate:.1f}%")
            
            # Show label information
            if result['dataset_info']['has_labels']:
                logger.info(f"   🏷️ Labels detected and processed")
                if 'label_distribution' in result:
                    train_dist = result['label_distribution']['train']
                    logger.info(f"      Train distribution: {train_dist}")
            else:
                logger.info(f"   🏷️ No labels detected - unsupervised dataset")
            
            # Show generated files
            if not args.quiet:
                logger.info(f"   📄 Generated files:")
                for file_type, file_path in result['files_generated'].items():
                    logger.info(f"      • {file_type}: {Path(file_path).name}")
            
            return 0
        else:
            logger.error(f"❌ PREPROCESSING FAILED: {result.get('error', 'Unknown error')}")
            
            if 'available_columns' in result:
                logger.error(f"   Available columns: {result['available_columns']}")
            
            return 1
            
    except KeyboardInterrupt:
        logger.warning("❌ Operation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())