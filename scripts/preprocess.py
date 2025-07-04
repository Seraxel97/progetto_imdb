#!/usr/bin/env python3
"""
Data Preprocessing Script - IMDb Sentiment Analysis
Preprocesses raw CSV data for sentiment analysis with train/val/test splits.

FEATURES:
- Automatic CSV format detection and validation
- Text cleaning and normalization
- Intelligent train/validation/test splitting (70/15/15)
- Class balancing and distribution analysis
- Compatible with embed_dataset.py and training scripts
- Robust error handling and logging
- Multiple input format support

USAGE:
    python scripts/preprocess.py
    python scripts/preprocess.py --input data/raw/imdb_dataset.csv
    python scripts/preprocess.py --input dataset.csv --output-dir data/processed
"""

import pandas as pd
import numpy as np
import re
import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from collections import Counter
import json

# Dynamic project root detection
try:
    CURRENT_FILE = Path(__file__).resolve()
    if CURRENT_FILE.parent.name == 'scripts':
        PROJECT_ROOT = CURRENT_FILE.parent.parent
    else:
        PROJECT_ROOT = CURRENT_FILE.parent
except:
    PROJECT_ROOT = Path.cwd()

# Setup paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def find_dataset_file():
    """Find dataset file automatically"""
    search_paths = [
        RAW_DATA_DIR / "imdb_raw.csv",
        RAW_DATA_DIR / "imdb_dataset.csv", 
        RAW_DATA_DIR / "dataset.csv",
        PROJECT_ROOT / "imdb_dataset.csv",
        PROJECT_ROOT / "dataset.csv",
        Path("imdb_dataset.csv"),
        Path("dataset.csv")
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None

def detect_csv_format(df, logger):
    """
    Detect and validate CSV format
    
    Args:
        df: DataFrame to analyze
        logger: Logger instance
    
    Returns:
        dict: Format information
    """
    logger.info("Detecting CSV format...")
    
    # Possible text column names
    text_columns = ['text', 'review', 'content', 'comment', 'description', 'message']
    label_columns = ['label', 'sentiment', 'class', 'target', 'rating']
    
    format_info = {
        'text_column': None,
        'label_column': None,
        'valid': False,
        'samples': len(df),
        'columns': list(df.columns)
    }
    
    # Find text column
    for col in text_columns:
        if col in df.columns:
            format_info['text_column'] = col
            break
    
    # Find label column
    for col in label_columns:
        if col in df.columns:
            format_info['label_column'] = col
            break
    
    if format_info['text_column']:
        format_info['valid'] = True
        logger.info(f"✅ Text column: {format_info['text_column']}")
        
        if format_info['label_column']:
            logger.info(f"✅ Label column: {format_info['label_column']}")
            
            # Analyze label distribution
            unique_labels = df[format_info['label_column']].unique()
            logger.info(f"📊 Unique labels: {unique_labels}")
            
            label_counts = df[format_info['label_column']].value_counts()
            logger.info(f"📊 Label distribution: {dict(label_counts)}")
        else:
            logger.warning("⚠️ No label column found - inference mode")
    else:
        logger.error(f"❌ No text column found. Available columns: {df.columns.tolist()}")
    
    return format_info

def clean_text(text):
    """
    Clean and normalize text
    
    Args:
        text: Raw text string
    
    Returns:
        str: Cleaned text
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

def preprocess_dataset(input_file, output_dir, logger):
    """
    Main preprocessing function
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save processed files
        logger: Logger instance
    
    Returns:
        dict: Processing results
    """
    logger.info(f"Starting preprocessing: {input_file}")
    
    # Load data
    try:
        df = pd.read_csv(input_file)
        logger.info(f"📊 Loaded dataset: {df.shape}")
    except Exception as e:
        logger.error(f"❌ Error loading CSV: {e}")
        raise
    
    # Detect format
    format_info = detect_csv_format(df, logger)
    if not format_info['valid']:
        raise ValueError("Invalid CSV format")
    
    # Standardize column names
    df_processed = df.copy()
    
    # Rename text column to 'text'
    if format_info['text_column'] != 'text':
        df_processed = df_processed.rename(columns={format_info['text_column']: 'text'})
        logger.info(f"🔄 Renamed '{format_info['text_column']}' to 'text'")
    
    # Rename label column to 'label' if exists
    if format_info['label_column'] and format_info['label_column'] != 'label':
        df_processed = df_processed.rename(columns={format_info['label_column']: 'label'})
        logger.info(f"🔄 Renamed '{format_info['label_column']}' to 'label'")
    
    # Clean text data
    logger.info("🧹 Cleaning text data...")
    original_count = len(df_processed)
    
    # Remove null/empty texts
    df_processed = df_processed.dropna(subset=['text'])
    df_processed['text'] = df_processed['text'].astype(str)
    df_processed['text'] = df_processed['text'].apply(clean_text)
    
    # Remove empty texts after cleaning
    df_processed = df_processed[df_processed['text'].str.len() > 5]
    
    cleaned_count = len(df_processed)
    removed_count = original_count - cleaned_count
    
    logger.info(f"📊 Removed {removed_count} invalid/empty texts")
    logger.info(f"📊 Final dataset size: {cleaned_count} samples")
    
    # Normalize labels if present
    dataset_size = len(df_processed)
    inference_only = False

    if 'label' in df_processed.columns:
        logger.info("🏷️ Processing labels...")
        
        # Handle different label formats
        unique_labels = df_processed['label'].unique()
        
        if set(unique_labels) == {'positive', 'negative'}:
            df_processed['label'] = df_processed['label'].map({'negative': 0, 'positive': 1})
            logger.info("🔄 Mapped text labels: negative→0, positive→1")
        elif set(unique_labels) == {'pos', 'neg'}:
            df_processed['label'] = df_processed['label'].map({'neg': 0, 'pos': 1})
            logger.info("🔄 Mapped short labels: neg→0, pos→1")
        elif len(unique_labels) == 2 and all(isinstance(l, (int, float)) for l in unique_labels):
            # Ensure binary labels are 0 and 1
            label_mapping = {sorted(unique_labels)[0]: 0, sorted(unique_labels)[1]: 1}
            df_processed['label'] = df_processed['label'].map(label_mapping)
            logger.info(f"🔄 Mapped numeric labels: {label_mapping}")
        
        # Final label distribution
        final_label_dist = df_processed['label'].value_counts().to_dict()
        logger.info(f"📊 Final label distribution: {final_label_dist}")
        
        # Check class balance
        label_counts = list(final_label_dist.values())
        balance_ratio = min(label_counts) / max(label_counts) if label_counts else 0
        
        if balance_ratio < 0.3:
            logger.warning(f"⚠️ Imbalanced dataset: {balance_ratio:.3f} ratio")
        else:
            logger.info(f"✅ Balanced dataset: {balance_ratio:.3f} ratio")

    # Determine if we have enough data for training
    if ('label' not in df_processed.columns or
            df_processed['label'].nunique() < 2 or
            dataset_size < 10):
        logger.warning("🚫 Skipping training — No label column or not enough samples")
        inference_only = True

    # Create train/validation/test splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if inference_only:
        inference_path = output_dir / "inference.csv"
        df_processed.to_csv(inference_path, index=False)
        logger.info(f"💾 Saved inference CSV for inference-only mode: {inference_path}")
        split_info = {
            'inference': len(df_processed),
            'strategy': 'inference_only'
        }
        train_df = val_df = test_df = df_processed.copy()
    elif dataset_size < 3:
        logger.warning("⚠️ Dataset too small for splitting. Replicating data in all sets.")
        train_df = df_processed.copy()
        val_df = df_processed.copy()
        test_df = df_processed.copy()
        split_info = {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df),
            'strategy': 'no_split'
        }
    elif 'label' in df_processed.columns:
        logger.info("📂 Creating stratified train/val/test splits (70/15/15)...")
        
        # First split: 70% train, 30% temp
        train_df, temp_df = train_test_split(
            df_processed,
            test_size=0.3,
            random_state=42,
            stratify=df_processed['label']
        )
        
        # Second split: 15% val, 15% test from the 30% temp
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df['label']
        )
        
        split_info = {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df),
            'strategy': 'stratified'
        }
    else:
        logger.info("📂 Creating random train/val/test splits (70/15/15) - no labels...")
        
        # Random splits without stratification
        train_df, temp_df = train_test_split(
            df_processed,
            test_size=0.3,
            random_state=42
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42
        )
        
        split_info = {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df),
            'strategy': 'random'
        }
    
    # Save splits or inference file
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    if not inference_only:
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"💾 Saved splits:")
        logger.info(f"   📄 Train: {train_path} ({len(train_df)} samples)")
        logger.info(f"   📄 Val: {val_path} ({len(val_df)} samples)")
        logger.info(f"   📄 Test: {test_path} ({len(test_df)} samples)")
    
    # Create metadata
    metadata = {
        'preprocessing_info': {
            'timestamp': datetime.now().isoformat(),
            'input_file': str(input_file),
            'output_directory': str(output_dir),
            'original_samples': original_count,
            'final_samples': cleaned_count,
            'removed_samples': removed_count,
            'text_column_used': format_info['text_column'],
            'label_column_used': format_info['label_column']
        },
        'dataset_splits': split_info,
        'file_paths': {},
        'inference_only': inference_only
    }

    if inference_only:
        metadata['file_paths']['inference'] = str(inference_path)
    else:
        metadata['file_paths'].update({
            'train': str(train_path),
            'val': str(val_path),
            'test': str(test_path)
        })
    
    if 'label' in df_processed.columns:
        metadata['label_info'] = {
            'final_distribution': final_label_dist,
            'balance_ratio': balance_ratio,
            'normalization_applied': True
        }
    
    # Save metadata
    metadata_path = output_dir / "preprocessing_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 Metadata saved: {metadata_path}")
    
    # Text statistics
    logger.info("📊 Text Statistics:")
    text_lengths = df_processed['text'].str.len()
    logger.info(f"   📏 Average length: {text_lengths.mean():.1f} characters")
    logger.info(f"   📏 Median length: {text_lengths.median():.1f} characters")
    logger.info(f"   📏 Min length: {text_lengths.min()} characters")
    logger.info(f"   📏 Max length: {text_lengths.max()} characters")
    
    # Word statistics
    word_counts = df_processed['text'].str.split().str.len()
    logger.info(f"   📝 Average words: {word_counts.mean():.1f}")
    logger.info(f"   📝 Median words: {word_counts.median():.1f}")
    
    output_files = {
        'metadata': str(metadata_path)
    }
    if inference_only:
        output_files['inference'] = str(inference_path)
    else:
        output_files.update({
            'train': str(train_path),
            'val': str(val_path),
            'test': str(test_path)
        })

    return {
        'success': True,
        'original_samples': original_count,
        'final_samples': cleaned_count,
        'removed_samples': removed_count,
        'split_info': split_info,
        'inference_only': inference_only,
        'output_files': output_files,
        'text_stats': {
            'avg_length': float(text_lengths.mean()),
            'median_length': float(text_lengths.median()),
            'avg_words': float(word_counts.mean())
        }
    }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Preprocess CSV data for sentiment analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/preprocess.py                                    # Auto-find dataset
  python scripts/preprocess.py --input data/raw/imdb_dataset.csv
  python scripts/preprocess.py --input dataset.csv --output-dir data/processed
        """
    )
    
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='Input CSV file path (auto-detect if not specified)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory for processed files (default: data/processed)')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Directory for log files (default: logs)')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup paths
    if args.input:
        input_file = Path(args.input)
    else:
        input_file = find_dataset_file()
        if not input_file:
            print("❌ No dataset file found!")
            print("💡 Place your CSV file in one of these locations:")
            print("   - data/raw/imdb_raw.csv")
            print("   - data/raw/imdb_dataset.csv")
            print("   - data/raw/dataset.csv")
            print("   - Or specify with --input path/to/your/file.csv")
            return 1
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return 1
    
    output_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DATA_DIR
    log_dir = Path(args.log_dir) if args.log_dir else PROJECT_ROOT / "logs"
    
    # Setup logging
    logger = setup_logging(log_dir)
    
    logger.info("=" * 60)
    logger.info("IMDb SENTIMENT DATA PREPROCESSING")
    logger.info("=" * 60)
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Project root: {PROJECT_ROOT}")
    
    try:
        # Run preprocessing
        result = preprocess_dataset(input_file, output_dir, logger)
        
        if result['success']:
            logger.info("=" * 60)
            logger.info("PREPROCESSING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"📊 Processed {result['final_samples']:,} samples")
            logger.info(f"📂 Files saved to: {output_dir}")
            logger.info("🚀 Ready for embedding generation!")
            logger.info("   Next step: python scripts/embed_dataset.py")
            
            return 0
        else:
            logger.error("❌ Preprocessing failed")
            return 1
        
    except Exception as e:
        logger.error(f"❌ Error during preprocessing: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
