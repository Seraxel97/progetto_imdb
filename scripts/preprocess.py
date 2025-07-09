#!/usr/bin/env python3
"""
Enhanced Data Preprocessing Script - Universal CSV Handler
Preprocesses ANY CSV data for sentiment analysis with complete flexibility.

🆕 ENHANCED FEATURES:
- ✅ Universal CSV detection and processing
- ✅ Smart text column auto-detection with confidence scoring
- ✅ Flexible label handling (with/without labels)
- ✅ Robust content cleaning and normalization
- ✅ Always produces embeddable output for MiniLM-L6-v2
- ✅ Handles edge cases: single column, weird structures, etc.
- ✅ Comprehensive fallback mechanisms
- ✅ Enhanced logging and validation

USAGE:
    python scripts/preprocess.py --input any_file.csv
    python scripts/preprocess.py --input complex_structure.csv --force-text-column "description"
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
import chardet
from typing import Dict, List, Tuple, Optional, Any

def load_csv_universal(path, logger=None):
    """🆕 Enhanced universal CSV loader with advanced detection"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    logger.info(f"🔍 Loading CSV with universal detection: {path}")
    
    # 1. Detect file encoding
    with open(path, 'rb') as f:
        raw_data = f.read()
        encoding_result = chardet.detect(raw_data)
        detected_encoding = encoding_result['encoding']
        confidence = encoding_result['confidence']
    
    logger.info(f"📊 Detected encoding: {detected_encoding} (confidence: {confidence:.2f})")
    
    # 2. Try multiple loading strategies
    loading_strategies = [
        {'encoding': detected_encoding, 'sep': ','},
        {'encoding': 'utf-8', 'sep': ','},
        {'encoding': 'latin-1', 'sep': ','},
        {'encoding': detected_encoding, 'sep': ';'},
        {'encoding': 'utf-8', 'sep': ';'},
        {'encoding': detected_encoding, 'sep': '\t'},
        {'encoding': 'utf-8', 'sep': '\t', 'engine': 'python'},
        {'encoding': 'utf-8', 'sep': None, 'engine': 'python'},  # Auto-detect separator
    ]
    
    df = None
    used_strategy = None
    
    for i, strategy in enumerate(loading_strategies):
        try:
            logger.info(f"📋 Trying loading strategy {i+1}: {strategy}")
            
            # Handle auto-detect separator
            if strategy.get('sep') is None:
                # Use pandas' automatic separator detection
                df_test = pd.read_csv(path, encoding=strategy['encoding'], 
                                    engine=strategy.get('engine', 'c'), nrows=5)
            else:
                df_test = pd.read_csv(path, encoding=strategy['encoding'], 
                                    sep=strategy['sep'], engine=strategy.get('engine', 'c'))
            
            # Validate that we got meaningful data
            if len(df_test.columns) > 0 and len(df_test) > 0:
                # Load the full file with the successful strategy
                if strategy.get('sep') is None:
                    df = pd.read_csv(path, encoding=strategy['encoding'], 
                                   engine=strategy.get('engine', 'c'))
                else:
                    df = pd.read_csv(path, encoding=strategy['encoding'], 
                                   sep=strategy['sep'], engine=strategy.get('engine', 'c'))
                
                used_strategy = strategy
                logger.info(f"✅ Successfully loaded with strategy {i+1}")
                break
                
        except Exception as e:
            logger.warning(f"⚠️ Strategy {i+1} failed: {str(e)}")
            continue
    
    if df is None:
        raise ValueError("Failed to load CSV with any strategy")
    
    logger.info(f"📊 Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"📋 Columns: {list(df.columns)}")
    logger.info(f"🔧 Used strategy: {used_strategy}")
    
    return df

def smart_column_detection(df, logger=None):
    """🆕 Enhanced smart column detection with confidence scoring"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("🧠 Starting smart column detection...")
    
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
    
    detection_results = {
        'text_column': None,
        'text_confidence': 0,
        'label_column': None,
        'label_confidence': 0,
        'text_candidates': [],
        'label_candidates': [],
        'analysis': {}
    }
    
    # Clean column names for analysis
    df_clean = df.copy()
    original_columns = list(df.columns)
    df_clean.columns = df_clean.columns.str.strip().str.lower()
    clean_to_original = dict(zip(df_clean.columns, original_columns))
    
    logger.info(f"📋 Analyzing {len(df_clean.columns)} columns...")
    
    # 1. Direct name matching for text columns
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
        
        # Content-based scoring
        if score == 0:  # No name match, analyze content
            try:
                sample_data = df_clean[clean_col].dropna().astype(str).head(100)
                if len(sample_data) > 0:
                    # Check if contains substantial text
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
                    if re.search(r'\b(the|and|is|was|good|bad|love|hate)\b', text_sample.lower()):
                        score += 10  # Contains common English words
                    
                    logger.info(f"   📊 {original_col}: avg_len={avg_length:.1f}, avg_words={word_count:.1f}, score={score}")
            except Exception as e:
                logger.warning(f"   ⚠️ Error analyzing {original_col}: {e}")
                score = 0
        
        if score > 0:
            text_scores.append((original_col, clean_col, score))
    
    # Sort by score and log results
    text_scores.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"🏆 Text column candidates: {[(col, score) for col, _, score in text_scores[:5]]}")
    
    # 2. Direct name matching for label columns
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
                    
                    logger.info(f"   🏷️ {original_col}: unique={n_unique}, values={str_values[:5]}, score={score}")
            except Exception as e:
                logger.warning(f"   ⚠️ Error analyzing {original_col}: {e}")
                score = 0
        
        if score > 0:
            label_scores.append((original_col, clean_col, score))
    
    # Sort label scores
    label_scores.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"🏷️ Label column candidates: {[(col, score) for col, _, score in label_scores[:5]]}")
    
    # 3. Select best candidates
    if text_scores:
        best_text = text_scores[0]
        detection_results['text_column'] = best_text[0]
        detection_results['text_confidence'] = best_text[2]
        detection_results['text_candidates'] = [(col, score) for col, _, score in text_scores]
    
    if label_scores:
        best_label = label_scores[0]
        detection_results['label_column'] = best_label[0]
        detection_results['label_confidence'] = best_label[2]
        detection_results['label_candidates'] = [(col, score) for col, _, score in label_scores]
    
    # 4. Fallback strategies if no clear text column found
    if detection_results['text_confidence'] < 10:
        logger.warning("⚠️ No clear text column found, using fallback strategies...")
        
        # Strategy 1: Use first string column with reasonable content
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    sample = df[col].dropna().astype(str).head(50)
                    if len(sample) > 0:
                        avg_len = sample.str.len().mean()
                        if avg_len > 10:  # Minimal text requirement
                            detection_results['text_column'] = col
                            detection_results['text_confidence'] = 5
                            logger.info(f"🔄 Fallback: Using {col} (avg_len={avg_len:.1f})")
                            break
            except Exception:
                continue
        
        # Strategy 2: Use the column with most content if still nothing
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
    
    # 5. Log final results
    logger.info("🎯 Final detection results:")
    logger.info(f"   📄 Text column: {detection_results['text_column']} (confidence: {detection_results['text_confidence']})")
    logger.info(f"   🏷️ Label column: {detection_results['label_column']} (confidence: {detection_results['label_confidence']})")
    
    return detection_results

def advanced_text_cleaning(text):
    """🆕 Enhanced text cleaning optimized for MiniLM-L6-v2"""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove null/empty indicators
    if text.lower() in ['null', 'none', 'nan', 'n/a', '', ' ']:
        return ""
    
    # 1. Fix encoding issues
    try:
        # Common encoding fixes
        text = text.encode('utf-8').decode('utf-8')
    except:
        pass
    
    # 2. Normalize whitespace and structure
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces -> single space
    text = re.sub(r'\n+', ' ', text)  # Multiple newlines -> space
    text = re.sub(r'\t+', ' ', text)  # Tabs -> space
    
    # 3. Clean HTML/XML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # 4. Clean URLs (but preserve context)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
    
    # 5. Clean email addresses
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
    
    # 6. Normalize punctuation (preserve sentiment indicators)
    text = re.sub(r'[.]{2,}', '...', text)  # Multiple dots
    text = re.sub(r'[!]{2,}', '!!', text)   # Multiple exclamations
    text = re.sub(r'[?]{2,}', '??', text)   # Multiple questions
    
    # 7. Remove excessive special characters but preserve sentiment
    text = re.sub(r'[^\w\s.,!?:;\'"()\-\[\]]', ' ', text)
    
    # 8. Fix common text artifacts
    text = re.sub(r'\b[A-Z]{3,}\b', lambda m: m.group().lower(), text)  # ALL CAPS words
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)  # Repeated characters (hellllllo -> hello)
    
    # 9. Final cleanup
    text = ' '.join(text.split())  # Remove extra whitespace
    text = text.strip()
    
    return text

def validate_and_enhance_dataset(df, text_col, label_col, logger=None):
    """🆕 Comprehensive dataset validation and enhancement"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("🔍 Validating and enhancing dataset...")
    
    original_count = len(df)
    df_enhanced = df.copy()
    
    # 1. Validate text column
    if text_col not in df_enhanced.columns:
        raise ValueError(f"Text column '{text_col}' not found in dataset")
    
    logger.info(f"📄 Processing text column: {text_col}")
    
    # 2. Clean and validate text data
    logger.info("🧹 Cleaning text data...")
    df_enhanced['text_original'] = df_enhanced[text_col].copy()  # Backup
    df_enhanced['text'] = df_enhanced[text_col].fillna('').astype(str).apply(advanced_text_cleaning)
    
    # Remove empty texts after cleaning
    empty_mask = (df_enhanced['text'].str.len() < 3)
    empty_count = empty_mask.sum()
    
    if empty_count > 0:
        logger.warning(f"⚠️ Removing {empty_count} empty/invalid texts")
        df_enhanced = df_enhanced[~empty_mask].reset_index(drop=True)
    
    # 3. Handle labels
    has_labels = False
    if label_col and label_col in df_enhanced.columns:
        logger.info(f"🏷️ Processing label column: {label_col}")
        
        # Clean and analyze labels
        df_enhanced['label_original'] = df_enhanced[label_col].copy()  # Backup
        labels_clean = df_enhanced[label_col].fillna(-1)
        
        # Convert various label formats to standard binary
        def normalize_label(label):
            if pd.isna(label):
                return -1
            
            label_str = str(label).lower().strip()
            
            # Binary text labels
            if label_str in ['positive', 'pos', '1', 'true', 'good', 'like']:
                return 1
            elif label_str in ['negative', 'neg', '0', 'false', 'bad', 'dislike']:
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
        
        df_enhanced['label'] = labels_clean.apply(normalize_label)
        
        # Check if we have valid labels
        valid_labels = df_enhanced['label'].isin([0, 1])
        valid_label_count = valid_labels.sum()
        
        if valid_label_count > 0:
            has_labels = True
            logger.info(f"✅ Found {valid_label_count} samples with valid labels")
            
            # Show label distribution
            label_dist = df_enhanced[valid_labels]['label'].value_counts().to_dict()
            logger.info(f"📊 Label distribution: {label_dist}")
        else:
            logger.warning("⚠️ No valid labels found, treating as inference mode")
            df_enhanced['label'] = -1  # Placeholder for inference
    else:
        logger.info("🔍 No label column found, creating placeholder labels for inference")
        df_enhanced['label'] = -1  # Placeholder for inference
    
    # 4. Final dataset statistics
    final_count = len(df_enhanced)
    removed_count = original_count - final_count
    
    logger.info("📊 Dataset validation results:")
    logger.info(f"   Original samples: {original_count:,}")
    logger.info(f"   Valid samples: {final_count:,}")
    logger.info(f"   Removed samples: {removed_count:,}")
    logger.info(f"   Has labels: {has_labels}")
    
    # Text statistics
    text_lengths = df_enhanced['text'].str.len()
    word_counts = df_enhanced['text'].str.split().str.len()
    
    logger.info(f"   📏 Text length: avg={text_lengths.mean():.1f}, median={text_lengths.median():.1f}")
    logger.info(f"   📝 Word count: avg={word_counts.mean():.1f}, median={word_counts.median():.1f}")
    
    # 5. Quality checks
    quality_issues = []
    
    # Check for very short texts
    very_short = (text_lengths < 5).sum()
    if very_short > 0:
        quality_issues.append(f"{very_short} very short texts (<5 chars)")
    
    # Check for very long texts
    very_long = (text_lengths > 1000).sum()
    if very_long > 0:
        quality_issues.append(f"{very_long} very long texts (>1000 chars)")
    
    # Check for duplicates
    duplicates = df_enhanced['text'].duplicated().sum()
    if duplicates > 0:
        quality_issues.append(f"{duplicates} duplicate texts")
    
    if quality_issues:
        logger.warning(f"⚠️ Quality issues detected: {'; '.join(quality_issues)}")
    else:
        logger.info("✅ No major quality issues detected")
    
    # 6. Ensure minimum dataset size
    min_samples = 3  # Minimum for meaningful splits
    if final_count < min_samples:
        raise ValueError(f"Dataset too small ({final_count} samples). Minimum required: {min_samples}")
    
    # 7. Create standardized output with essential columns only
    output_df = pd.DataFrame({
        'text': df_enhanced['text'],
        'label': df_enhanced['label']
    })
    
    return output_df, has_labels, {
        'original_count': original_count,
        'final_count': final_count,
        'removed_count': removed_count,
        'has_labels': has_labels,
        'quality_issues': quality_issues,
        'text_stats': {
            'avg_length': float(text_lengths.mean()),
            'median_length': float(text_lengths.median()),
            'avg_words': float(word_counts.mean()),
            'median_words': float(word_counts.median())
        }
    }

def create_robust_splits(df, has_labels, min_split_size=1, logger=None):
    """🆕 Create robust train/val/test splits with intelligent fallbacks"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("📂 Creating robust dataset splits...")
    
    total_samples = len(df)
    logger.info(f"📊 Total samples to split: {total_samples:,}")
    
    # Define split strategies based on dataset size and label availability
    if total_samples < 3:
        # Too small for splits, create single file
        logger.warning("⚠️ Dataset too small for splits, creating single test file")
        return {
            'test': df,
            'strategy': 'single_file',
            'split_info': {'test': total_samples}
        }
    
    elif total_samples < 10:
        # Small dataset: minimal splits
        if has_labels:
            # Try simple split
            try:
                train_df, test_df = train_test_split(
                    df, test_size=0.3, random_state=42, stratify=df['label']
                )
                logger.info("📂 Small dataset: train/test split (70/30)")
                return {
                    'train': train_df,
                    'test': test_df,
                    'strategy': 'train_test_small',
                    'split_info': {'train': len(train_df), 'test': len(test_df)}
                }
            except ValueError:
                # Stratification failed, use random split
                train_df, test_df = train_test_split(
                    df, test_size=0.3, random_state=42
                )
                logger.warning("⚠️ Stratification failed, using random split")
                return {
                    'train': train_df,
                    'test': test_df,
                    'strategy': 'train_test_random',
                    'split_info': {'train': len(train_df), 'test': len(test_df)}
                }
        else:
            # No labels, create inference file
            logger.info("📂 Small dataset without labels: inference mode")
            return {
                'inference': df,
                'strategy': 'inference_small',
                'split_info': {'inference': total_samples}
            }
    
    elif total_samples < 50:
        # Medium dataset: train/val/test
        if has_labels:
            try:
                # 60/20/20 split for medium datasets
                train_df, temp_df = train_test_split(
                    df, test_size=0.4, random_state=42, stratify=df['label']
                )
                val_df, test_df = train_test_split(
                    temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
                )
                
                logger.info("📂 Medium dataset: train/val/test split (60/20/20)")
                return {
                    'train': train_df,
                    'val': val_df,
                    'test': test_df,
                    'strategy': 'train_val_test_medium',
                    'split_info': {
                        'train': len(train_df),
                        'val': len(val_df),
                        'test': len(test_df)
                    }
                }
            except ValueError:
                # Fallback to random split
                train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
                val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
                
                logger.warning("⚠️ Stratification failed, using random split")
                return {
                    'train': train_df,
                    'val': val_df,
                    'test': test_df,
                    'strategy': 'train_val_test_random',
                    'split_info': {
                        'train': len(train_df),
                        'val': len(val_df),
                        'test': len(test_df)
                    }
                }
        else:
            # No labels: inference mode
            logger.info("📂 Medium dataset without labels: inference mode")
            return {
                'inference': df,
                'strategy': 'inference_medium',
                'split_info': {'inference': total_samples}
            }
    
    else:
        # Large dataset: standard 70/15/15 split
        if has_labels:
            try:
                # Standard split
                train_df, temp_df = train_test_split(
                    df, test_size=0.3, random_state=42, stratify=df['label']
                )
                val_df, test_df = train_test_split(
                    temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
                )
                
                logger.info("📂 Large dataset: train/val/test split (70/15/15)")
                return {
                    'train': train_df,
                    'val': val_df,
                    'test': test_df,
                    'strategy': 'train_val_test_standard',
                    'split_info': {
                        'train': len(train_df),
                        'val': len(val_df),
                        'test': len(test_df)
                    }
                }
            except ValueError:
                # Fallback to random split
                train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
                val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
                
                logger.warning("⚠️ Stratification failed, using random split")
                return {
                    'train': train_df,
                    'val': val_df,
                    'test': test_df,
                    'strategy': 'train_val_test_random',
                    'split_info': {
                        'train': len(train_df),
                        'val': len(val_df),
                        'test': len(test_df)
                    }
                }
        else:
            # Large dataset without labels: still inference mode
            logger.info("📂 Large dataset without labels: inference mode")
            return {
                'inference': df,
                'strategy': 'inference_large',
                'split_info': {'inference': total_samples}
            }

def enhanced_preprocess_pipeline(input_file, output_dir, force_text_column=None, logger=None):
    """🆕 Main enhanced preprocessing pipeline"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        logger.info("=" * 60)
        logger.info("🆕 ENHANCED UNIVERSAL CSV PREPROCESSING")
        logger.info("=" * 60)
        logger.info(f"📄 Input file: {input_file}")
        logger.info(f"📁 Output directory: {output_dir}")
        
        # 1. Load CSV with universal detection
        df = load_csv_universal(input_file, logger)
        
        # 2. Smart column detection
        if force_text_column:
            logger.info(f"🔧 Forcing text column: {force_text_column}")
            if force_text_column not in df.columns:
                raise ValueError(f"Forced text column '{force_text_column}' not found")
            detection = {
                'text_column': force_text_column,
                'text_confidence': 100,
                'label_column': None,
                'label_confidence': 0
            }
        else:
            detection = smart_column_detection(df, logger)
        
        if not detection['text_column']:
            raise ValueError("No suitable text column found in the dataset")
        
        # 3. Validate and enhance dataset
        processed_df, has_labels, stats = validate_and_enhance_dataset(
            df, detection['text_column'], detection['label_column'], logger
        )
        
        # 4. Create robust splits
        splits = create_robust_splits(processed_df, has_labels, logger=logger)
        
        # 5. Save files
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        for split_name, split_df in splits.items():
            if split_name not in ['strategy', 'split_info']:
                file_path = output_dir / f"{split_name}.csv"
                split_df.to_csv(file_path, index=False)
                saved_files[split_name] = str(file_path)
                logger.info(f"💾 Saved {split_name}: {file_path} ({len(split_df)} samples)")
        
        # 6. Create comprehensive metadata
        metadata = {
            'preprocessing_info': {
                'timestamp': datetime.now().isoformat(),
                'input_file': str(input_file),
                'output_directory': str(output_dir),
                'processing_version': 'enhanced_universal_v1.0'
            },
            'detection_results': detection,
            'dataset_stats': stats,
            'split_strategy': splits['strategy'],
            'split_info': splits['split_info'],
            'saved_files': saved_files,
            'inference_only': not has_labels,
            'embeddable': True,  # Always true with enhanced processing
            'recommendations': []
        }
        
        # Add recommendations based on analysis
        if stats['final_count'] < 100:
            metadata['recommendations'].append("Small dataset - consider using --fast mode for training")
        
        if not has_labels:
            metadata['recommendations'].append("No labels detected - pipeline will run in inference mode")
        
        if stats['quality_issues']:
            metadata['recommendations'].append(f"Quality issues detected: {'; '.join(stats['quality_issues'])}")
        
        # Save metadata
        metadata_path = output_dir / "preprocessing_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Metadata saved: {metadata_path}")
        
        logger.info("=" * 60)
        logger.info("✅ ENHANCED PREPROCESSING COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"📊 Processed: {stats['final_count']}/{stats['original_count']} samples")
        logger.info(f"🔧 Strategy: {splits['strategy']}")
        logger.info(f"🏷️ Has labels: {has_labels}")
        logger.info(f"📁 Files saved: {list(saved_files.keys())}")
        logger.info("🚀 Ready for embedding generation!")
        
        return {
            'success': True,
            'metadata': metadata,
            'saved_files': saved_files,
            'stats': stats,
            'has_labels': has_labels
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced preprocessing failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'input_file': str(input_file)
        }

# Dynamic project root detection
try:
    CURRENT_FILE = Path(__file__).resolve()
    if CURRENT_FILE.parent.name == 'scripts':
        PROJECT_ROOT = CURRENT_FILE.parent.parent
    else:
        PROJECT_ROOT = CURRENT_FILE.parent
except:
    PROJECT_ROOT = Path.cwd()

def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"enhanced_preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Universal CSV Preprocessing for Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🆕 ENHANCED FEATURES:
- Universal CSV detection and processing
- Smart text column auto-detection
- Flexible label handling (with/without labels)
- Always produces embeddable output for MiniLM-L6-v2

Examples:
  python scripts/preprocess.py --input any_file.csv
  python scripts/preprocess.py --input complex_data.csv --output-dir custom_output
  python scripts/preprocess.py --input weird_structure.csv --force-text-column "description"
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input CSV file path (any structure)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory for processed files (default: data/processed)')
    parser.add_argument('--force-text-column', type=str, default=None,
                       help='Force specific column as text column (bypasses auto-detection)')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Directory for log files (default: logs)')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup paths
    input_file = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "processed"
    log_dir = Path(args.log_dir) if args.log_dir else PROJECT_ROOT / "logs"
    
    # Setup logging
    logger = setup_logging(log_dir)
    
    if not input_file.exists():
        logger.error(f"❌ Input file not found: {input_file}")
        return 1
    
    try:
        # Run enhanced preprocessing pipeline
        result = enhanced_preprocess_pipeline(
            input_file, output_dir, args.force_text_column, logger
        )
        
        if result['success']:
            logger.info("🎉 SUCCESS! Enhanced preprocessing completed.")
            logger.info("   Next step: python scripts/embed_dataset.py")
            return 0
        else:
            logger.error(f"❌ FAILED: {result['error']}")
            return 1
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())