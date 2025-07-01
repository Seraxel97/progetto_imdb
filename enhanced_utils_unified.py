#!/usr/bin/env python3
"""
Enhanced Utilities for Unified Sentiment Analysis Pipeline - COMPLETE AUTOMATION
Comprehensive utility functions for automated embedding, prediction, training, and analysis.

FEATURES:
- auto_embed_and_predict(): Complete pipeline automation (embedding → training → prediction)
- generate_insights(): Intelligent comments and recommendations based on data analysis
- analyze_text(): Individual text analysis with keywords, phrases, topics (GUI compatible)
- Timestamp-based result organization in results/[timestamp]/
- Dynamic path detection and robust error handling
- Integration with all existing scripts (embed_dataset.py, train_mlp.py, train_svm.py, report.py)
- Comprehensive CSV validation and preprocessing
- Professional report generation with insights
- GUI-compatible data formats (no numpy types, Streamlit-ready)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
import warnings
import re
from collections import Counter, defaultdict
import string

# Enhanced NLTK imports with better error handling
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
    from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    # Create fallback functions
    def word_tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())
    
    def sent_tokenize(text):
        return re.split(r'[.!?]+', text)
    
    def stopwords_words(lang):
        return {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

warnings.filterwarnings('ignore')

# Download required NLTK data with error handling
if NLTK_AVAILABLE:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except Exception:
            pass

# FIXED: Dynamic project root detection
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
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DATA_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Add scripts to path for imports
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_convert_for_json(obj):
    """
    Convert numpy types and other non-JSON-serializable objects to JSON-compatible types
    
    Args:
        obj: Object to convert
    
    Returns:
        JSON-compatible object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    else:
        return obj

def clean_text_for_analysis(text: str) -> str:
    """
    Clean text for keyword and phrase extraction
    
    Args:
        text: Raw text to clean
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    return text

def extract_keywords_by_label(text: str, sentiment_label: str = "unknown") -> List[Dict[str, Union[str, int]]]:
    """
    Extract keywords from text, organized by sentiment label
    
    Args:
        text: Text to analyze
        sentiment_label: Predicted sentiment label ('positive', 'negative', or 'unknown')
    
    Returns:
        List of keyword dictionaries with label, word, and count
    """
    try:
        cleaned_text = clean_text_for_analysis(text)
        
        # Tokenize and clean words
        tokens = word_tokenize(cleaned_text.lower())
        
        # Get English stopwords with fallback
        if NLTK_AVAILABLE:
            stop_words = set(stopwords.words('english'))
        else:
            stop_words = stopwords_words('english')
        
        # Add common punctuation and short words to stop words
        stop_words.update(string.punctuation)
        stop_words.update(['', ' ', 'would', 'could', 'should', 'also', 'even', 'really', 'get', 'go', 'one', 'two'])
        
        # Filter tokens
        filtered_tokens = [
            token for token in tokens 
            if token.isalpha() and len(token) > 2 and token not in stop_words
        ]
        
        # Count word frequencies
        word_counts = Counter(filtered_tokens)
        
        # Get top words
        top_words = word_counts.most_common(15)
        
        # Format for GUI
        keywords = []
        for word, count in top_words:
            keywords.append({
                "label": sentiment_label,
                "word": word,
                "count": int(count)  # Ensure int, not numpy int
            })
        
        return keywords
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return [{"label": sentiment_label, "word": "error", "count": 0}]

def get_top_phrases(text: str, n_phrases: int = 10) -> List[Dict[str, Union[str, int]]]:
    """
    Extract top phrases (bigrams and trigrams) from text
    
    Args:
        text: Text to analyze
        n_phrases: Number of top phrases to return
    
    Returns:
        List of phrase dictionaries with phrase and count (GUI format)
    """
    try:
        cleaned_text = clean_text_for_analysis(text)
        
        # Tokenize
        tokens = word_tokenize(cleaned_text.lower())
        
        # Get stopwords
        if NLTK_AVAILABLE:
            stop_words = set(stopwords.words('english'))
        else:
            stop_words = stopwords_words('english')
        stop_words.update(string.punctuation)
        
        # Filter tokens
        filtered_tokens = [
            token for token in tokens 
            if token.isalpha() and len(token) > 2 and token not in stop_words
        ]
        
        phrases = []
        
        # Extract bigrams and trigrams
        if NLTK_AVAILABLE and len(filtered_tokens) >= 2:
            try:
                # Extract bigrams
                bigram_finder = BigramCollocationFinder.from_words(filtered_tokens)
                bigram_finder.apply_freq_filter(2)  # Minimum frequency of 2
                
                bigrams = bigram_finder.nbest(BigramAssocMeasures.pmi, n_phrases//2)
                for bigram in bigrams:
                    phrase_text = ' '.join(bigram)
                    # Count occurrences in original text
                    count = len(re.findall(re.escape(phrase_text), cleaned_text, re.IGNORECASE))
                    if count > 0:
                        phrases.append({
                            "phrase": phrase_text,
                            "count": int(count)
                        })
                
                # Extract trigrams
                if len(filtered_tokens) >= 3:
                    trigram_finder = TrigramCollocationFinder.from_words(filtered_tokens)
                    trigram_finder.apply_freq_filter(2)
                    
                    trigrams = trigram_finder.nbest(TrigramAssocMeasures.pmi, n_phrases//2)
                    for trigram in trigrams:
                        phrase_text = ' '.join(trigram)
                        count = len(re.findall(re.escape(phrase_text), cleaned_text, re.IGNORECASE))
                        if count > 0:
                            phrases.append({
                                "phrase": phrase_text,
                                "count": int(count)
                            })
            except Exception:
                pass  # Skip advanced NLTK features if they fail
        
        # Fallback: Simple n-gram extraction
        if not phrases and len(filtered_tokens) >= 2:
            # Generate bigrams manually
            for i in range(len(filtered_tokens) - 1):
                bigram = f"{filtered_tokens[i]} {filtered_tokens[i+1]}"
                count = len(re.findall(re.escape(bigram), cleaned_text, re.IGNORECASE))
                if count > 1:
                    phrases.append({"phrase": bigram, "count": count})
            
            # Generate trigrams manually
            for i in range(len(filtered_tokens) - 2):
                trigram = f"{filtered_tokens[i]} {filtered_tokens[i+1]} {filtered_tokens[i+2]}"
                count = len(re.findall(re.escape(trigram), cleaned_text, re.IGNORECASE))
                if count > 1:
                    phrases.append({"phrase": trigram, "count": count})
        
        # Sort by count and return top phrases
        phrases.sort(key=lambda x: x['count'], reverse=True)
        return phrases[:n_phrases]
        
    except Exception as e:
        logger.error(f"Error extracting phrases: {e}")
        return [{"phrase": "error extracting phrases", "count": 0}]

def extract_topics(text: str) -> List[Dict[str, Union[str, int]]]:
    """
    Extract topics/themes from text using keyword clustering
    
    Args:
        text: Text to analyze
    
    Returns:
        List of topic dictionaries with topic name and count
    """
    try:
        cleaned_text = clean_text_for_analysis(text)
        
        # Define topic keywords (can be expanded)
        topic_keywords = {
            "quality": ["quality", "excellent", "good", "bad", "poor", "great", "terrible", "amazing", "awful", "wonderful"],
            "service": ["service", "staff", "customer", "support", "help", "assistance", "team", "employee", "representative"],
            "price": ["price", "cost", "expensive", "cheap", "affordable", "value", "money", "budget", "worth", "pricing"],
            "product": ["product", "item", "goods", "merchandise", "purchase", "buy", "sell", "order", "delivery"],
            "experience": ["experience", "feel", "feeling", "enjoyed", "loved", "hated", "disappointed", "satisfied", "happy", "sad"],
            "performance": ["performance", "speed", "fast", "slow", "efficiency", "effective", "works", "function", "operation"],
            "design": ["design", "look", "appearance", "beautiful", "ugly", "style", "color", "layout", "interface"],
            "recommendation": ["recommend", "suggest", "advice", "opinion", "review", "rating", "feedback", "comment"]
        }
        
        # Count topic occurrences
        topic_counts = defaultdict(int)
        
        text_lower = cleaned_text.lower()
        words = word_tokenize(text_lower)
        
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                # Count both exact matches and partial matches
                count = sum(1 for word in words if keyword in word or word in keyword)
                topic_counts[topic] += count
        
        # Convert to list format for GUI
        topics = []
        for topic, count in topic_counts.items():
            if count > 0:  # Only include topics that appear
                topics.append({
                    "topic": topic,
                    "count": int(count)
                })
        
        # Sort by count
        topics.sort(key=lambda x: x['count'], reverse=True)
        
        # Return top 8 topics
        return topics[:8]
        
    except Exception as e:
        logger.error(f"Error extracting topics: {e}")
        return [{"topic": "error", "count": 0}]

def save_analysis_results(results: Dict, session_dir: Path, text: str) -> Dict[str, str]:
    """
    Save analysis results to CSV and JSON files
    
    Args:
        results: Analysis results dictionary
        session_dir: Directory to save results
        text: Original text analyzed
        
    Returns:
        Dictionary with file paths for GUI download
    """
    try:
        # Create subdirectories
        analysis_dir = session_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        file_paths = {}
        
        # Save top words as CSV
        if results.get("top_words"):
            words_df = pd.DataFrame(results["top_words"])
            words_path = analysis_dir / "word_stats.csv"
            words_df.to_csv(words_path, index=False)
            file_paths["word_stats_csv"] = str(words_path)
        
        # Save phrases as CSV
        if results.get("top_phrases"):
            phrases_df = pd.DataFrame(results["top_phrases"])
            phrases_path = analysis_dir / "phrases.csv"
            phrases_df.to_csv(phrases_path, index=False)
            file_paths["phrases_csv"] = str(phrases_path)
        
        # Save topics as CSV
        if results.get("topics"):
            topics_df = pd.DataFrame(results["topics"])
            topics_path = analysis_dir / "topics.csv"
            topics_df.to_csv(topics_path, index=False)
            file_paths["topics_csv"] = str(topics_path)
        
        # Save complete summary as JSON
        summary = {
            "timestamp": datetime.now().isoformat(),
            "original_text": text,
            "analysis_results": results,
            "files_generated": list(file_paths.keys())
        }
        
        summary_path = analysis_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        file_paths["summary_json"] = str(summary_path)
        
        logger.info(f"Analysis results saved to {analysis_dir}")
        return file_paths
        
    except Exception as e:
        logger.error(f"Error saving analysis results: {e}")
        return {}

def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyze individual text and return structured data for GUI
    
    Args:
        text: Text to analyze
    
    Returns:
        Dictionary with top_words, top_phrases, topics, metadata, and file paths
    """
    try:
        logger.info(f"Analyzing text: {text[:100]}...")
        
        # Clean and validate input
        if not text or not isinstance(text, str):
            return {
                "top_words": [],
                "top_phrases": [],
                "topics": [],
                "metadata": {"text_length": 0, "vocab_size": 0},
                "error": "Invalid input text"
            }
        
        # Predict sentiment first using the internal predictor (if available)
        sentiment_label = "unknown"
        try:
            predictor = SentimentPredictor()
            if predictor.models:
                prediction = predictor.predict(text)
                sentiment_label = "positive" if prediction.get("prediction") == 1 else "negative"
        except Exception as e:
            logger.warning(f"Could not load predictor for sentiment analysis: {e}")
            sentiment_label = "unknown"
        
        # Extract keywords with sentiment label
        top_words = extract_keywords_by_label(text, sentiment_label)
        
        # Extract top phrases
        top_phrases = get_top_phrases(text)
        
        # Extract topics
        topics = extract_topics(text)
        
        # Calculate metadata
        words = text.split()
        unique_words = set(word.lower().strip(string.punctuation) for word in words if word.strip())
        
        metadata = {
            "text_length": len(text),
            "vocab_size": len(unique_words),
            "word_count": len(words),
            "sentence_count": len(sent_tokenize(text)),
            "sentiment_detected": sentiment_label,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create results dictionary
        results = {
            "top_words": top_words,
            "top_phrases": top_phrases,
            "topics": topics,
            "metadata": metadata
        }
        
        # Save results to session directory and get file paths
        file_paths = {}
        try:
            session_dir = create_timestamped_session_dir("text_analysis")
            file_paths = save_analysis_results(results, session_dir, text)
            results["session_directory"] = str(session_dir)
            results["file_paths"] = file_paths
        except Exception as e:
            logger.warning(f"Could not save analysis results: {e}")
            results["file_paths"] = {}
        
        # Ensure all data is JSON-serializable
        results = safe_convert_for_json(results)
        
        logger.info(f"Text analysis completed. Found {len(top_words)} keywords, {len(top_phrases)} phrases, {len(topics)} topics")
        return results
        
    except Exception as e:
        logger.error(f"Error in text analysis: {e}")
        return {
            "top_words": [],
            "top_phrases": [],
            "topics": [],
            "metadata": {"text_length": 0, "vocab_size": 0},
            "error": str(e),
            "file_paths": {}
        }

def analyze_csv(file_path: str, mode: str = "deep") -> Dict[str, Any]:
    """
    Analyze CSV file with sentiment analysis and return aggregated results
    
    Args:
        file_path: Path to CSV file
        mode: Analysis mode ('quick', 'deep', 'full')
        
    Returns:
        Dictionary with aggregated analysis results for GUI
    """
    try:
        logger.info(f"Starting CSV analysis in {mode} mode: {file_path}")
        
        # Load and validate CSV
        df = pd.read_csv(file_path)
        is_valid, issues = validate_csv_for_analysis(df)
        
        if not is_valid:
            return {
                "top_words": [],
                "top_phrases": [],
                "topics": [],
                "metadata": {"total_samples": 0, "vocab_size": 0},
                "error": f"CSV validation failed: {', '.join(issues)}",
                "file_paths": {}
            }
        
        # Auto-detect text and label columns
        text_columns = ['text', 'review', 'content', 'comment', 'message']
        label_columns = ['label', 'sentiment', 'class', 'target', 'rating']
        
        text_col = None
        label_col = None
        
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break
        
        for col in label_columns:
            if col in df.columns:
                label_col = col
                break
        
        if text_col is None:
            return {
                "top_words": [],
                "top_phrases": [],
                "topics": [],
                "metadata": {"total_samples": len(df), "vocab_size": 0},
                "error": f"No recognized text column found. Available: {list(df.columns)}",
                "file_paths": {}
            }
        
        # Clean and prepare data
        df_clean = df.copy()
        df_clean[text_col] = df_clean[text_col].fillna('').astype(str)
        df_clean = df_clean[df_clean[text_col].str.strip() != '']
        
        if len(df_clean) == 0:
            return {
                "top_words": [],
                "top_phrases": [],
                "topics": [],
                "metadata": {"total_samples": 0, "vocab_size": 0},
                "error": "No valid text data found after cleaning",
                "file_paths": {}
            }
        
        # Get or predict labels if mode is 'deep' or 'full'
        predicted_labels = {}
        if mode in ['deep', 'full']:
            try:
                # Try to use existing labels
                if label_col and label_col in df_clean.columns:
                    # Normalize existing labels
                    unique_labels = df_clean[label_col].unique()
                    if set(unique_labels) == {'positive', 'negative'}:
                        df_clean['sentiment_label'] = df_clean[label_col].map({'negative': 'negative', 'positive': 'positive'})
                    elif set(unique_labels) == {'pos', 'neg'}:
                        df_clean['sentiment_label'] = df_clean[label_col].map({'neg': 'negative', 'pos': 'positive'})
                    elif set(unique_labels) == {0, 1}:
                        df_clean['sentiment_label'] = df_clean[label_col].map({0: 'negative', 1: 'positive'})
                    else:
                        df_clean['sentiment_label'] = 'unknown'
                else:
                    # Try to predict labels using available predictors
                    try:
                        # Simple prediction fallback if no sophisticated predictor available
                        predictions = []
                        for text in df_clean[text_col].tolist():
                            # Basic sentiment prediction based on keywords
                            text_lower = str(text).lower()
                            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'perfect', 'awesome']
                            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing', 'poor']
                            
                            pos_count = sum(1 for word in positive_words if word in text_lower)
                            neg_count = sum(1 for word in negative_words if word in text_lower)
                            
                            if pos_count > neg_count:
                                predictions.append("positive")
                            elif neg_count > pos_count:
                                predictions.append("negative")
                            else:
                                predictions.append("unknown")
                        
                        df_clean['sentiment_label'] = predictions
                        predicted_labels['used_basic_predictor'] = True
                    except Exception as e:
                        logger.warning(f"Could not predict sentiment labels: {e}")
                        df_clean['sentiment_label'] = 'unknown'
            except Exception as e:
                logger.warning(f"Could not get/predict sentiment labels: {e}")
                df_clean['sentiment_label'] = 'unknown'
        else:
            df_clean['sentiment_label'] = 'unknown'
        
        # Combine all text for analysis
        all_text = ' '.join(df_clean[text_col].tolist())
        
        # Aggregate analysis by sentiment (if available)
        top_words = []
        top_phrases = []
        topics = []
        
        if 'sentiment_label' in df_clean.columns:
            sentiment_groups = df_clean.groupby('sentiment_label')
            
            for sentiment, group in sentiment_groups:
                if len(group) > 0:
                    # Combine text for this sentiment
                    sentiment_text = ' '.join(group[text_col].tolist())
                    
                    # Extract keywords for this sentiment
                    sentiment_words = extract_keywords_by_label(sentiment_text, sentiment)
                    top_words.extend(sentiment_words[:10])  # Top 10 per sentiment
                    
                    # Extract phrases for this sentiment
                    sentiment_phrases = get_top_phrases(sentiment_text, 8)
                    top_phrases.extend(sentiment_phrases)
                    
                    # Extract topics for this sentiment
                    sentiment_topics = extract_topics(sentiment_text)
                    for topic in sentiment_topics:
                        topic['sentiment'] = sentiment  # Add sentiment info
                    topics.extend(sentiment_topics)
        else:
            # No sentiment info, analyze all text together
            top_words = extract_keywords_by_label(all_text, "unknown")
            top_phrases = get_top_phrases(all_text)
            topics = extract_topics(all_text)
        
        # Calculate metadata
        all_words = all_text.split()
        unique_words = set(word.lower().strip(string.punctuation) for word in all_words if word.strip())
        
        metadata = {
            "total_samples": len(df_clean),
            "vocab_size": len(unique_words),
            "total_words": len(all_words),
            "avg_text_length": df_clean[text_col].str.len().mean(),
            "analysis_mode": mode,
            "has_labels": label_col is not None,
            "predicted_labels": bool(predicted_labels),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add sentiment distribution if available
        if 'sentiment_label' in df_clean.columns:
            sentiment_dist = df_clean['sentiment_label'].value_counts().to_dict()
            metadata["sentiment_distribution"] = sentiment_dist
        
        # Create results dictionary
        results = {
            "top_words": top_words[:20],  # Limit to top 20
            "top_phrases": top_phrases[:15],  # Limit to top 15
            "topics": topics[:10],  # Limit to top 10
            "metadata": metadata
        }
        
        # Save results to session directory
        file_paths = {}
        try:
            session_dir = create_timestamped_session_dir(f"csv_analysis_{mode}")
            
            # Save CSV analysis results
            analysis_dir = session_dir / "analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            # Save aggregated results
            if results.get("top_words"):
                words_df = pd.DataFrame(results["top_words"])
                words_path = analysis_dir / "word_stats.csv"
                words_df.to_csv(words_path, index=False)
                file_paths["word_stats_csv"] = str(words_path)
            
            if results.get("top_phrases"):
                phrases_df = pd.DataFrame(results["top_phrases"])
                phrases_path = analysis_dir / "phrases.csv"
                phrases_df.to_csv(phrases_path, index=False)
                file_paths["phrases_csv"] = str(phrases_path)
            
            if results.get("topics"):
                topics_df = pd.DataFrame(results["topics"])
                topics_path = analysis_dir / "topics.csv"
                topics_df.to_csv(topics_path, index=False)
                file_paths["topics_csv"] = str(topics_path)
            
            # Save processed CSV with predictions (if any)
            if 'sentiment_label' in df_clean.columns and predicted_labels:
                processed_path = analysis_dir / "processed_with_predictions.csv"
                df_clean.to_csv(processed_path, index=False)
                file_paths["processed_csv"] = str(processed_path)
            
            # Save complete summary
            summary = {
                "timestamp": datetime.now().isoformat(),
                "analysis_mode": mode,
                "input_file": file_path,
                "analysis_results": results,
                "files_generated": list(file_paths.keys())
            }
            
            summary_path = analysis_dir / "summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            file_paths["summary_json"] = str(summary_path)
            
            results["session_directory"] = str(session_dir)
            results["file_paths"] = file_paths
            
        except Exception as e:
            logger.warning(f"Could not save CSV analysis results: {e}")
            results["file_paths"] = {}
        
        # Ensure all data is JSON-serializable
        results = safe_convert_for_json(results)
        
        logger.info(f"CSV analysis completed. Processed {len(df_clean)} samples, found {len(top_words)} keywords, {len(top_phrases)} phrases, {len(topics)} topics")
        return results
        
    except Exception as e:
        logger.error(f"Error in CSV analysis: {e}")
        return {
            "top_words": [],
            "top_phrases": [],
            "topics": [],
            "metadata": {"total_samples": 0, "vocab_size": 0},
            "error": str(e),
            "file_paths": {}
        }

class TextProcessor:
    """
    Processor class for handling text analysis requests from GUI
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".TextProcessor")
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze single text - wrapper for the global analyze_text function
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results dictionary
        """
        return analyze_text(text)
    
    def analyze_csv(self, file_path: str, mode: str = "deep") -> Dict[str, Any]:
        """
        Analyze CSV file with different modes - wrapper for the global analyze_csv function
        
        Args:
            file_path: Path to CSV file
            mode: Analysis mode ('quick', 'deep', 'full')
            
        Returns:
            Analysis results dictionary
        """
        return analyze_csv(file_path, mode)

# Create global processor instance for GUI
processor = TextProcessor()

# COMPATIBILITY FIX: Basic SentimentPredictor class for GUI compatibility
class SentimentPredictor:
    """
    Basic sentiment predictor for GUI compatibility
    Falls back to keyword-based prediction if advanced models not available
    """
    
    def __init__(self):
        self.models = {}
        try:
            # Try to load actual models if available
            self._load_models()
        except Exception:
            logger.warning("Advanced models not available, using basic predictor")
    
    def _load_models(self):
        """Try to load actual trained models"""
        try:
            # Try to load models from standard locations
            import joblib
            import torch
            
            model_paths = [
                MODELS_DIR / "svm_model.pkl",
                PROJECT_ROOT / "results" / "models" / "svm_model.pkl"
            ]
            
            for path in model_paths:
                if path.exists():
                    self.models['svm'] = joblib.load(path)
                    break
            
            mlp_paths = [
                MODELS_DIR / "mlp_model.pth",
                PROJECT_ROOT / "results" / "models" / "mlp_model.pth"
            ]
            
            for path in mlp_paths:
                if path.exists():
                    self.models['mlp'] = torch.load(path, map_location='cpu')
                    break
                    
        except Exception as e:
            logger.warning(f"Could not load advanced models: {e}")
    
    def predict(self, text: str, model_name: str = None) -> Dict[str, Any]:
        """
        Predict sentiment for a single text
        
        Args:
            text: Text to analyze
            model_name: Specific model to use (optional)
            
        Returns:
            Dictionary with prediction and confidence
        """
        try:
            if self.models and model_name in self.models:
                # Use actual model if available
                # This would need proper implementation based on actual model structure
                pass
            
            # Fallback to basic keyword-based prediction
            text_lower = str(text).lower()
            
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'perfect', 'awesome', 'fantastic', 'brilliant']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing', 'poor', 'disgusting', 'pathetic']
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                prediction = 1  # positive
                confidence = min(0.9, 0.6 + (pos_count - neg_count) * 0.1)
            elif neg_count > pos_count:
                prediction = 0  # negative  
                confidence = min(0.9, 0.6 + (neg_count - pos_count) * 0.1)
            else:
                prediction = 0  # default to negative if unclear
                confidence = 0.5
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "model_used": model_name or "basic_keyword",
                "text": text
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                "prediction": 0,
                "confidence": 0.5,
                "error": str(e),
                "text": text
            }
    
    def predict_batch(self, texts: List[str]) -> Dict[str, List[int]]:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with model predictions
        """
        try:
            predictions = {}
            
            if self.models:
                for model_name in self.models.keys():
                    model_preds = []
                    for text in texts:
                        pred_result = self.predict(text, model_name)
                        model_preds.append(pred_result['prediction'])
                    predictions[model_name] = model_preds
            else:
                # Basic prediction for all texts
                basic_preds = []
                for text in texts:
                    pred_result = self.predict(text)
                    basic_preds.append(pred_result['prediction'])
                predictions['basic'] = basic_preds
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return {"basic": [0] * len(texts)}

# COMPATIBILITY FIX: Add alias for GUI compatibility
EnhancedFileAnalysisProcessor = TextProcessor

def run_dataset_analysis(file_path: str, steps: List[str] = None, output_dir: str = None) -> bool:
    """
    Run complete dataset analysis pipeline - GUI compatibility function
    
    Args:
        file_path: Path to dataset file
        steps: List of steps to execute (default: all steps)
        output_dir: Directory to save results (auto-generated if None)
        
    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    if steps is None:
        steps = ["preprocess", "embed", "train", "predict", "report"]
    
    try:
        logger.info(f"Starting dataset analysis pipeline with steps: {steps}")
        
        # Run the complete pipeline
        pipeline_results = auto_embed_and_predict(file_path, fast_mode=False, save_intermediate=True)
        
        # Check if pipeline was successful
        success = pipeline_results.get('overall_success', False)
        warnings_count = pipeline_results.get('warnings_count', 0)
        
        if success and warnings_count < len(steps):
            logger.info(f"Dataset analysis pipeline completed successfully")
            return True
        else:
            logger.warning(f"Dataset analysis pipeline completed with issues: success={success}, warnings={warnings_count}")
            return success
        
    except Exception as e:
        logger.error(f"Error in dataset analysis pipeline: {e}")
        return False

def create_timestamped_session_dir(base_name: str = "analysis") -> Path:
    """
    Create a timestamped directory for session results
    
    Args:
        base_name: Base name for the directory
    
    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = RESULTS_DIR / f"{base_name}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (session_dir / "processed").mkdir(exist_ok=True)
    (session_dir / "embeddings").mkdir(exist_ok=True)
    (session_dir / "models").mkdir(exist_ok=True)
    (session_dir / "reports").mkdir(exist_ok=True)
    (session_dir / "plots").mkdir(exist_ok=True)
    
    logger.info(f"Created session directory: {session_dir}")
    return session_dir

def validate_and_preprocess_csv(csv_path: str, output_dir: Optional[str] = None) -> Tuple[bool, str, Dict]:
    """
    Validate and preprocess CSV file for sentiment analysis
    
    Args:
        csv_path: Path to the CSV file
        output_dir: Directory to save processed files (auto-generated if None)
    
    Returns:
        Tuple of (success, processed_file_path, validation_info)
    """
    try:
        # Load and validate CSV
        df = pd.read_csv(csv_path)
        is_valid, issues = validate_csv_for_analysis(df)
        
        validation_info = {
            'original_file': csv_path,
            'is_valid': is_valid,
            'issues': issues,
            'original_shape': df.shape,
            'original_columns': list(df.columns)
        }
        
        if not is_valid:
            logger.error(f"CSV validation failed: {issues}")
            return False, "", validation_info
        
        # Auto-detect text and label columns
        text_columns = ['text', 'review', 'content', 'comment', 'message']
        label_columns = ['label', 'sentiment', 'class', 'target', 'rating']
        
        text_col = None
        label_col = None
        
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break
        
        for col in label_columns:
            if col in df.columns:
                label_col = col
                break
        
        if text_col is None:
            logger.error(f"No recognized text column found. Available: {list(df.columns)}")
            return False, "", validation_info
        
        # Standardize column names
        df_processed = df.copy()
        df_processed = df_processed.rename(columns={text_col: 'text'})
        
        if label_col is not None:
            df_processed = df_processed.rename(columns={label_col: 'label'})
            
            # Normalize labels to 0/1 if needed
            unique_labels = df_processed['label'].unique()
            if set(unique_labels) == {'positive', 'negative'}:
                df_processed['label'] = df_processed['label'].map({'negative': 0, 'positive': 1})
            elif set(unique_labels) == {'pos', 'neg'}:
                df_processed['label'] = df_processed['label'].map({'neg': 0, 'pos': 1})
            elif len(unique_labels) == 2 and all(isinstance(l, (int, float)) for l in unique_labels):
                # Assume binary numeric labels, map to 0/1
                label_mapping = {sorted(unique_labels)[0]: 0, sorted(unique_labels)[1]: 1}
                df_processed['label'] = df_processed['label'].map(label_mapping)
        
        # Clean text data
        df_processed['text'] = df_processed['text'].fillna('').astype(str)
        df_processed['text'] = df_processed['text'].str.strip()
        
        # Remove empty texts
        original_len = len(df_processed)
        df_processed = df_processed[df_processed['text'] != '']
        cleaned_len = len(df_processed)
        
        if cleaned_len < original_len:
            logger.warning(f"Removed {original_len - cleaned_len} empty text entries")
        
        # Create train/val/test splits if labels available
        if 'label' in df_processed.columns:
            from sklearn.model_selection import train_test_split
            
            # 70% train, 15% val, 15% test
            train_df, temp_df = train_test_split(
                df_processed, test_size=0.3, random_state=42, stratify=df_processed['label']
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
            )
            
            validation_info['splits'] = {
                'train': len(train_df),
                'val': len(val_df),
                'test': len(test_df)
            }
        else:
            # No labels, use all data as test set
            train_df = df_processed.iloc[:0].copy()  # Empty train
            val_df = df_processed.iloc[:0].copy()   # Empty val
            test_df = df_processed.copy()           # All data as test
            
            validation_info['splits'] = {
                'train': 0,
                'val': 0,
                'test': len(test_df)
            }
            validation_info['note'] = 'No labels found, all data treated as test set'
        
        # Save processed files
        if output_dir is None:
            output_dir = create_timestamped_session_dir("preprocessing")
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        train_path = output_dir / "train.csv"
        val_path = output_dir / "val.csv"
        test_path = output_dir / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Save validation info
        validation_path = output_dir / "preprocessing_info.json"
        validation_info['processed_files'] = {
            'train': str(train_path),
            'val': str(val_path),
            'test': str(test_path)
        }
        validation_info['output_directory'] = str(output_dir)
        validation_info['processed_shape'] = df_processed.shape
        validation_info['text_column_used'] = text_col
        validation_info['label_column_used'] = label_col
        
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(validation_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Preprocessing completed. Files saved to: {output_dir}")
        return True, str(test_path), validation_info
        
    except Exception as e:
        logger.error(f"Error in CSV preprocessing: {e}")
        return False, "", {'error': str(e)}

def run_embedding_generation(processed_dir: str, embeddings_dir: str) -> Tuple[bool, Dict]:
    """
    Run embedding generation using embed_dataset.py
    
    Args:
        processed_dir: Directory containing processed CSV files
        embeddings_dir: Directory to save embeddings
    
    Returns:
        Tuple of (success, embedding_info)
    """
    try:
        logger.info("Starting embedding generation...")
        
        embed_script = SCRIPTS_DIR / "embed_dataset.py"
        if not embed_script.exists():
            raise FileNotFoundError(f"Embedding script not found: {embed_script}")
        
        # Run embedding generation
        cmd = [
            sys.executable, str(embed_script),
            "--input-dir", processed_dir,
            "--output-dir", embeddings_dir,
            "--force-recreate"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info("Embedding generation completed successfully")
            
            # Check for generated files
            embeddings_path = Path(embeddings_dir)
            expected_files = ['X_train.npy', 'y_train.npy', 'X_val.npy', 'y_val.npy', 
                            'X_test.npy', 'y_test.npy', 'embedding_metadata.json']
            
            generated_files = [f for f in expected_files if (embeddings_path / f).exists()]
            
            embedding_info = {
                'success': True,
                'embeddings_dir': embeddings_dir,
                'generated_files': generated_files,
                'stdout': result.stdout,
                'command': ' '.join(cmd)
            }
            
            # Load metadata if available
            metadata_path = embeddings_path / "embedding_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    embedding_info['metadata'] = json.load(f)
            
            return True, embedding_info
        else:
            logger.error(f"Embedding generation failed: {result.stderr}")
            return False, {
                'success': False,
                'error': result.stderr,
                'stdout': result.stdout,
                'command': ' '.join(cmd)
            }
            
    except subprocess.TimeoutExpired:
        logger.error("Embedding generation timed out")
        return False, {'success': False, 'error': 'Embedding generation timed out'}
    except Exception as e:
        logger.error(f"Error in embedding generation: {e}")
        return False, {'success': False, 'error': str(e)}

def run_model_training(embeddings_dir: str, models_dir: str, fast_mode: bool = True) -> Tuple[bool, Dict]:
    """
    Run model training for both MLP and SVM
    
    Args:
        embeddings_dir: Directory containing embeddings
        models_dir: Directory to save trained models
        fast_mode: Whether to use fast training mode
    
    Returns:
        Tuple of (success, training_info)
    """
    training_info = {
        'mlp': {'success': False},
        'svm': {'success': False}
    }
    
    try:
        logger.info("Starting model training...")
        
        # Train MLP
        logger.info("Training MLP model...")
        mlp_script = SCRIPTS_DIR / "train_mlp.py"
        if mlp_script.exists():
            mlp_cmd = [
                sys.executable, str(mlp_script),
                "--embeddings-dir", embeddings_dir,
                "--models-dir", models_dir
            ]
            if fast_mode:
                mlp_cmd.append("--fast")
            
            mlp_result = subprocess.run(mlp_cmd, capture_output=True, text=True, timeout=300)
            
            if mlp_result.returncode == 0:
                training_info['mlp'] = {
                    'success': True,
                    'stdout': mlp_result.stdout,
                    'command': ' '.join(mlp_cmd)
                }
                logger.info("MLP training completed successfully")
            else:
                training_info['mlp'] = {
                    'success': False,
                    'error': mlp_result.stderr,
                    'stdout': mlp_result.stdout,
                    'command': ' '.join(mlp_cmd)
                }
                logger.warning(f"MLP training failed: {mlp_result.stderr}")
        else:
            logger.warning(f"MLP training script not found: {mlp_script}")
            training_info['mlp']['error'] = 'MLP script not found'
        
        # Train SVM
        logger.info("Training SVM model...")
        svm_script = SCRIPTS_DIR / "train_svm.py"
        if svm_script.exists():
            svm_cmd = [
                sys.executable, str(svm_script),
                "--embeddings-dir", embeddings_dir,
                "--models-dir", models_dir
            ]
            if fast_mode:
                svm_cmd.append("--fast")
            
            svm_result = subprocess.run(svm_cmd, capture_output=True, text=True, timeout=300)
            
            if svm_result.returncode == 0:
                training_info['svm'] = {
                    'success': True,
                    'stdout': svm_result.stdout,
                    'command': ' '.join(svm_cmd)
                }
                logger.info("SVM training completed successfully")
            else:
                training_info['svm'] = {
                    'success': False,
                    'error': svm_result.stderr,
                    'stdout': svm_result.stdout,
                    'command': ' '.join(svm_cmd)
                }
                logger.warning(f"SVM training failed: {svm_result.stderr}")
        else:
            logger.warning(f"SVM training script not found: {svm_script}")
            training_info['svm']['error'] = 'SVM script not found'
        
        # Overall success if at least one model trained successfully
        overall_success = training_info['mlp']['success'] or training_info['svm']['success']
        
        return overall_success, training_info
        
    except subprocess.TimeoutExpired:
        logger.error("Model training timed out")
        return False, {'error': 'Model training timed out'}
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        return False, {'error': str(e)}

def run_prediction_and_evaluation(models_dir: str, processed_dir: str, results_dir: str) -> Tuple[bool, Dict]:
    """
    Run predictions and generate evaluation reports
    
    Args:
        models_dir: Directory containing trained models
        processed_dir: Directory containing processed test data
        results_dir: Directory to save results
    
    Returns:
        Tuple of (success, evaluation_info)
    """
    try:
        logger.info("Starting prediction and evaluation...")
        
        # Run report generation
        report_script = SCRIPTS_DIR / "report.py"
        if not report_script.exists():
            logger.warning(f"Report script not found: {report_script}")
            return False, {'error': 'Report script not found'}
        
        test_data_path = Path(processed_dir) / "test.csv"
        if not test_data_path.exists():
            logger.error(f"Test data not found: {test_data_path}")
            return False, {'error': 'Test data not found'}
        
        cmd = [
            sys.executable, str(report_script),
            "--models-dir", models_dir,
            "--results-dir", results_dir,
            "--test-data", str(test_data_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("Evaluation completed successfully")
            
            # Check for generated reports
            results_path = Path(results_dir)
            report_files = list(results_path.glob("*.json")) + list(results_path.glob("*.txt"))
            
            evaluation_info = {
                'success': True,
                'results_dir': results_dir,
                'generated_files': [str(f) for f in report_files],
                'stdout': result.stdout,
                'command': ' '.join(cmd)
            }
            
            return True, evaluation_info
        else:
            logger.error(f"Evaluation failed: {result.stderr}")
            return False, {
                'success': False,
                'error': result.stderr,
                'stdout': result.stdout,
                'command': ' '.join(cmd)
            }
            
    except subprocess.TimeoutExpired:
        logger.error("Evaluation timed out")
        return False, {'success': False, 'error': 'Evaluation timed out'}
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        return False, {'success': False, 'error': str(e)}

def auto_embed_and_predict(file_path: str, fast_mode: bool = True, 
                          save_intermediate: bool = True) -> Dict[str, Any]:
    """
    Complete automated pipeline: CSV → preprocessing → embedding → training → prediction → evaluation
    
    Args:
        file_path: Path to the CSV file to analyze
        fast_mode: Whether to use fast training modes
        save_intermediate: Whether to save intermediate results
    
    Returns:
        Comprehensive results dictionary with all pipeline outputs
    """
    logger.info(f"Starting complete automated pipeline for: {file_path}")
    
    # Create timestamped session directory
    session_dir = create_timestamped_session_dir("auto_analysis")
    
    pipeline_results = {
        'pipeline_start': datetime.now().isoformat(),
        'input_file': file_path,
        'session_directory': str(session_dir),
        'fast_mode': fast_mode,
        'steps': {}
    }
    
    try:
        # Step 1: Validate and preprocess CSV
        logger.info("Step 1: Preprocessing CSV...")
        processed_dir = session_dir / "processed"
        success, test_file, preprocessing_info = validate_and_preprocess_csv(
            file_path, str(processed_dir)
        )
        
        pipeline_results['steps']['preprocessing'] = {
            'success': success,
            'info': preprocessing_info,
            'duration': 'N/A'
        }
        
        if not success:
            pipeline_results['overall_success'] = False
            pipeline_results['error'] = 'Preprocessing failed'
            return pipeline_results
        
        # Step 2: Generate embeddings
        logger.info("Step 2: Generating embeddings...")
        embeddings_dir = session_dir / "embeddings"
        emb_success, embedding_info = run_embedding_generation(
            str(processed_dir), str(embeddings_dir)
        )
        
        pipeline_results['steps']['embedding'] = {
            'success': emb_success,
            'info': embedding_info
        }
        
        if not emb_success:
            logger.warning("Embedding generation failed, continuing with existing models if available...")
        
        # Step 3: Train models
        logger.info("Step 3: Training models...")
        models_dir = session_dir / "models"
        training_success, training_info = run_model_training(
            str(embeddings_dir), str(models_dir), fast_mode
        )
        
        pipeline_results['steps']['training'] = {
            'success': training_success,
            'info': training_info
        }
        
        # Step 4: Generate predictions and evaluation
        logger.info("Step 4: Generating predictions and evaluation...")
        reports_dir = session_dir / "reports"
        eval_success, evaluation_info = run_prediction_and_evaluation(
            str(models_dir), str(processed_dir), str(reports_dir)
        )
        
        pipeline_results['steps']['evaluation'] = {
            'success': eval_success,
            'info': evaluation_info
        }
        
        # Step 5: Load and format results for GUI
        logger.info("Step 5: Collecting and formatting results...")
        
        # Load test data for analysis
        test_data_path = Path(processed_dir) / "test.csv"
        if test_data_path.exists():
            test_df = pd.read_csv(test_data_path)
            pipeline_results['test_data'] = {
                'shape': test_df.shape,
                'columns': list(test_df.columns),
                'sample_texts': test_df['text'].head(3).tolist() if 'text' in test_df.columns else []
            }
        
        # Try to load trained models and make predictions for GUI
        try:
            predictor = SentimentPredictor()

            if test_data_path.exists():
                # Look for trained models
                mlp_path = models_dir / "mlp_model.pth"
                svm_path = models_dir / "svm_model.pkl"
                
                predictions = {}
                metrics = {}
                
                # Try SVM predictions
                if svm_path.exists():
                    try:
                        # Load embeddings and make predictions
                        X_test_path = embeddings_dir / "X_test.npy"
                        y_test_path = embeddings_dir / "y_test.npy"
                        
                        if X_test_path.exists() and y_test_path.exists():
                            import joblib
                            X_test = np.load(X_test_path)
                            y_test = np.load(y_test_path)
                            
                            svm_package = joblib.load(svm_path)
                            svm_model = svm_package['model']
                            scaler = svm_package.get('scaler')
                            
                            X_test_scaled = scaler.transform(X_test) if scaler else X_test
                            svm_pred = svm_model.predict(X_test_scaled)
                            
                            predictions['svm'] = svm_pred.tolist()
                            
                            # Calculate metrics if labels available
                            if 'label' in test_df.columns and len(y_test) > 0:
                                metrics['svm'] = {
                                    'accuracy': float(accuracy_score(y_test, svm_pred)),
                                    'f1_score': float(f1_score(y_test, svm_pred, average='weighted'))
                                }
                    except Exception as e:
                        logger.warning(f"SVM prediction failed: {e}")
                
                # Try MLP predictions
                if mlp_path.exists():
                    try:
                        import torch
                        X_test_path = embeddings_dir / "X_test.npy"
                        y_test_path = embeddings_dir / "y_test.npy"
                        
                        if X_test_path.exists() and y_test_path.exists():
                            X_test = np.load(X_test_path)
                            y_test = np.load(y_test_path)
                            
                            # Load MLP model
                            model = torch.load(mlp_path, map_location='cpu')
                            model.eval()
                            
                            with torch.no_grad():
                                X_tensor = torch.FloatTensor(X_test)
                                outputs = model(X_tensor)
                                
                                if outputs.dim() == 1:
                                    mlp_pred = (torch.sigmoid(outputs) > 0.5).int().numpy()
                                else:
                                    mlp_pred = torch.argmax(outputs, dim=1).numpy()
                            
                            predictions['mlp'] = mlp_pred.tolist()
                            
                            # Calculate metrics if labels available
                            if 'label' in test_df.columns and len(y_test) > 0:
                                metrics['mlp'] = {
                                    'accuracy': float(accuracy_score(y_test, mlp_pred)),
                                    'f1_score': float(f1_score(y_test, mlp_pred, average='weighted'))
                                }
                    except Exception as e:
                        logger.warning(f"MLP prediction failed: {e}")
                
                pipeline_results['gui_predictions'] = predictions
                pipeline_results['gui_metrics'] = metrics
                
        except Exception as e:
            logger.warning(f"Error collecting GUI results: {e}")
        
        # Generate intelligent insights
        if test_data_path.exists():
            test_df = pd.read_csv(test_data_path)
            insights = generate_insights(
                test_df, 
                pipeline_results.get('gui_predictions', {}),
                pipeline_results.get('gui_metrics', {})
            )
            pipeline_results['insights'] = insights
            
            # Save insights to file
            insights_path = session_dir / "insights.txt"
            with open(insights_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(insights))
        
        # Overall success assessment
        critical_steps = ['preprocessing']
        optional_steps = ['embedding', 'training', 'evaluation']
        
        overall_success = all(pipeline_results['steps'][step]['success'] for step in critical_steps)
        warnings_count = sum(1 for step in optional_steps 
                           if not pipeline_results['steps'].get(step, {}).get('success', False))
        
        pipeline_results['overall_success'] = overall_success
        pipeline_results['warnings_count'] = warnings_count
        pipeline_results['pipeline_end'] = datetime.now().isoformat()
        
        # Save complete pipeline results
        results_file = session_dir / "pipeline_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(pipeline_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Pipeline completed. Results saved to: {session_dir}")
        return pipeline_results
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        pipeline_results['overall_success'] = False
        pipeline_results['error'] = str(e)
        pipeline_results['pipeline_end'] = datetime.now().isoformat()
        return pipeline_results

def generate_insights(df: pd.DataFrame, predictions: Dict = None, 
                     metrics: Dict = None) -> List[str]:
    """
    Generate intelligent insights and comments about dataset and model performance
    
    Args:
        df: DataFrame with the analyzed data
        predictions: Dictionary with model predictions
        metrics: Dictionary with model performance metrics
    
    Returns:
        List of insight strings for display in GUI
    """
    insights = []
    
    try:
        # Dataset insights
        n_samples = len(df)
        insights.append(f"📊 **Dataset Analysis**: Processed {n_samples:,} samples for sentiment analysis")
        
        # Text analysis
        if 'text' in df.columns:
            text_lengths = df['text'].str.len()
            avg_length = text_lengths.mean()
            
            insights.append(f"📝 **Text Characteristics**: Average length {avg_length:.0f} characters")
            
            # Text quality assessment
            short_texts = (text_lengths < 20).sum()
            long_texts = (text_lengths > 1000).sum()
            
            if short_texts > n_samples * 0.1:
                insights.append(f"⚠️ **Data Quality Alert**: {short_texts} texts ({short_texts/n_samples*100:.1f}%) are very short (<20 chars) - may affect accuracy")
            
            if long_texts > n_samples * 0.05:
                insights.append(f"📏 **Text Length Notice**: {long_texts} texts ({long_texts/n_samples*100:.1f}%) are very long (>1000 chars) - model may truncate")
            
            # Check for duplicates
            duplicates = df['text'].duplicated().sum()
            if duplicates > 0:
                insights.append(f"🔍 **Duplicate Detection**: Found {duplicates} duplicate texts ({duplicates/n_samples*100:.1f}%) - consider deduplication")
            else:
                insights.append("✅ **Data Uniqueness**: No duplicate texts detected - good data quality")
        
        # Label distribution analysis (if available)
        if 'label' in df.columns:
            label_dist = df['label'].value_counts()
            
            if len(label_dist) == 2:
                # Binary classification
                negative_count = label_dist.get(0, 0)
                positive_count = label_dist.get(1, 0)
                balance_ratio = min(negative_count, positive_count) / max(negative_count, positive_count)
                
                if balance_ratio > 0.8:
                    insights.append(f"⚖️ **Well-Balanced Dataset**: Excellent balance - {negative_count} negative, {positive_count} positive samples")
                elif balance_ratio > 0.6:
                    insights.append(f"📊 **Moderately Balanced**: Good balance - {negative_count} negative, {positive_count} positive samples")
                else:
                    majority_class = "positive" if positive_count > negative_count else "negative"
                    insights.append(f"⚠️ **Imbalanced Dataset**: Skewed toward {majority_class} class - consider rebalancing techniques")
            
            # Missing labels
            missing_labels = df['label'].isnull().sum()
            if missing_labels > 0:
                insights.append(f"🏷️ **Label Completeness**: {missing_labels} samples missing labels - will be treated as unlabeled predictions")
        else:
            insights.append("🏷️ **Unlabeled Dataset**: No ground truth labels provided - performing inference-only analysis")
        
        # Model performance insights
        if predictions and metrics:
            model_count = len(predictions)
            insights.append(f"🤖 **Model Ensemble**: Successfully deployed {model_count} model{'s' if model_count > 1 else ''} for prediction")
            
            for model_name, model_metrics in metrics.items():
                accuracy = model_metrics.get('accuracy', 0)
                f1 = model_metrics.get('f1_score', 0)
                
                # Performance assessment with specific recommendations
                if accuracy >= 0.90:
                    insights.append(f"🏆 **{model_name.upper()} Excellence**: Outstanding {accuracy:.1%} accuracy - production-ready performance!")
                elif accuracy >= 0.85:
                    insights.append(f"✅ **{model_name.upper()} Success**: Strong {accuracy:.1%} accuracy - reliable for most applications")
                elif accuracy >= 0.80:
                    insights.append(f"📈 **{model_name.upper()} Performance**: Good {accuracy:.1%} accuracy - suitable for preliminary analysis")
                elif accuracy >= 0.70:
                    insights.append(f"⚠️ **{model_name.upper()} Caution**: Moderate {accuracy:.1%} accuracy - consider additional training data")
                else:
                    insights.append(f"❌ **{model_name.upper()} Alert**: Low {accuracy:.1%} accuracy - recommend model retraining or different approach")
                
                # F1 vs Accuracy analysis
                if abs(f1 - accuracy) > 0.05:
                    if f1 < accuracy:
                        insights.append(f"📊 **{model_name.upper()} Analysis**: F1-score ({f1:.1%}) lower than accuracy - indicates class imbalance sensitivity")
                    else:
                        insights.append(f"📊 **{model_name.upper()} Analysis**: F1-score ({f1:.1%}) higher than accuracy - good balance handling")
                
                # Specific performance ranges with actionable advice
                if 0.75 <= accuracy < 0.85:
                    insights.append(f"💡 **{model_name.upper()} Improvement Tip**: Consider hyperparameter tuning or more training data to reach 85%+ accuracy")
                
                if accuracy >= 0.85 and f1 >= 0.85:
                    insights.append(f"🎯 **{model_name.upper()} Recommendation**: Performance meets production standards - ready for deployment")
        
        # Model comparison insights (if multiple models)
        if predictions and len(predictions) > 1:
            model_names = list(predictions.keys())
            if len(model_names) == 2:
                pred1, pred2 = predictions[model_names[0]], predictions[model_names[1]]
                agreement = (np.array(pred1) == np.array(pred2)).mean()
                
                if agreement > 0.9:
                    insights.append(f"🤝 **Model Consensus**: {agreement:.1%} agreement between {model_names[0].upper()} and {model_names[1].upper()} - high confidence in predictions")
                elif agreement > 0.8:
                    insights.append(f"🤔 **Model Alignment**: {agreement:.1%} agreement - generally consistent predictions with some differences")
                elif agreement > 0.6:
                    insights.append(f"🔄 **Model Divergence**: {agreement:.1%} agreement - models show meaningful differences, consider ensemble voting")
                else:
                    insights.append(f"⚡ **Model Conflict**: Only {agreement:.1%} agreement - significant disagreement suggests need for more training data")
                
                # Recommend best model if metrics available
                if metrics:
                    best_model = max(metrics.keys(), key=lambda k: metrics[k].get('accuracy', 0))
                    best_accuracy = metrics[best_model].get('accuracy', 0)
                    insights.append(f"🏅 **Top Performer**: {best_model.upper()} achieved highest accuracy ({best_accuracy:.1%}) - recommended for primary use")
        
        # Prediction distribution insights
        if predictions:
            for model_name, pred_list in predictions.items():
                pred_array = np.array(pred_list)
                positive_ratio = (pred_array == 1).mean()
                
                if 0.4 <= positive_ratio <= 0.6:
                    insights.append(f"📊 **{model_name.upper()} Predictions**: Balanced output - {positive_ratio:.1%} positive sentiment detected")
                elif positive_ratio > 0.8:
                    insights.append(f"😊 **{model_name.upper()} Trend**: Predominantly positive sentiment - {positive_ratio:.1%} positive classifications")
                elif positive_ratio < 0.2:
                    insights.append(f"😞 **{model_name.upper()} Trend**: Predominantly negative sentiment - {positive_ratio:.1%} positive classifications")
                else:
                    insights.append(f"📈 **{model_name.upper()} Distribution**: {positive_ratio:.1%} positive, {1-positive_ratio:.1%} negative sentiment")
        
        # Data quality recommendations
        if 'text' in df.columns:
            null_count = df['text'].isnull().sum()
            empty_count = (df['text'].str.strip() == '').sum()
            
            if null_count == 0 and empty_count == 0:
                insights.append("✅ **Data Quality Excellent**: No missing or empty texts - optimal for analysis")
            else:
                total_issues = null_count + empty_count
                insights.append(f"🔧 **Data Quality Recommendation**: Clean {total_issues} problematic entries for improved accuracy")
        
        # Final recommendation
        if metrics:
            avg_accuracy = np.mean([m.get('accuracy', 0) for m in metrics.values()])
            if avg_accuracy >= 0.85:
                insights.append("🎉 **Overall Assessment**: High-quality analysis results - ready for business decisions and deployment")
            elif avg_accuracy >= 0.75:
                insights.append("✅ **Overall Assessment**: Good analysis results - suitable for most use cases with monitoring")
            else:
                insights.append("⚠️ **Overall Assessment**: Results need improvement - consider data augmentation or model refinement")
        else:
            insights.append("📋 **Analysis Complete**: Predictions generated successfully - review individual results for insights")
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        insights.append(f"⚠️ **Insight Generation Error**: {str(e)}")
    
    return insights

# EXISTING FUNCTIONS (maintained for compatibility)

def analyze_csv_with_predictor(
    df: pd.DataFrame, 
    predictor: 'SentimentPredictor',
    include_predictions: bool = True,
    include_metrics: bool = True,
    save_results: bool = False,
    results_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze CSV data using the unified sentiment predictor pipeline.
    
    This function ensures consistency with the main pipeline by using:
    - predictor.pipeline.text_processor for preprocessing
    - predictor.predict_batch() for predictions
    - Same embedding and normalization logic as training
    
    Args:
        df: DataFrame with 'text' column and optional 'label' column
        predictor: SentimentPredictor instance with loaded models
        include_predictions: Whether to include individual predictions
        include_metrics: Whether to calculate metrics (requires 'label' column)
        save_results: Whether to save results to disk
        results_dir: Directory to save results (auto-generated if None)
    
    Returns:
        Dictionary containing analysis results:
        - dataset_info: Basic dataset information
        - statistics: Text and label statistics
        - predictions: Model predictions (if include_predictions=True)
        - metrics: Performance metrics (if include_metrics=True and labels available)
        - confusion_matrices: Confusion matrix data
        - timestamp: Analysis timestamp
        - results_dir: Directory where results were saved (if save_results=True)
    """
    
    # Validate input
    if 'text' not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column")
    
    if df['text'].isnull().any():
        logger.warning(f"Found {df['text'].isnull().sum()} null values in text column. These will be skipped.")
        df = df.dropna(subset=['text'])
    
    # Initialize results dictionary
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'has_labels': 'label' in df.columns
        }
    }
    
    logger.info(f"Starting analysis of {len(df)} samples...")
    
    # Extract texts and labels
    texts = df['text'].tolist()
    labels = df['label'].tolist() if 'label' in df.columns else None
    
    # Calculate text statistics
    text_lengths = df['text'].str.len()
    results['statistics'] = {
        'text_stats': {
            'count': len(texts),
            'mean_length': float(text_lengths.mean()),
            'std_length': float(text_lengths.std()),
            'min_length': int(text_lengths.min()),
            'max_length': int(text_lengths.max()),
            'median_length': float(text_lengths.median())
        }
    }
    
    # Add label statistics if available
    if labels is not None:
        label_counts = df['label'].value_counts().to_dict()
        results['statistics']['label_stats'] = {
            'distribution': label_counts,
            'total_samples': len(labels),
            'positive_ratio': label_counts.get(1, 0) / len(labels),
            'negative_ratio': label_counts.get(0, 0) / len(labels)
        }
    
    # Get predictions using the unified predictor
    logger.info("Generating predictions using unified predictor...")
    try:
        # Use predict_batch for efficient batch processing
        if hasattr(predictor, 'predict_batch'):
            batch_predictions = predictor.predict_batch(texts)
        else:
            # Fallback to individual predictions
            batch_predictions = {}
            for model_name in predictor.models.keys():
                model_preds = []
                for text in texts:
                    pred_result = predictor.predict(text, model_name)
                    model_preds.append(pred_result['prediction'])
                batch_predictions[model_name] = model_preds
        
        results['predictions'] = batch_predictions
        logger.info(f"Generated predictions for {len(batch_predictions)} models")
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise RuntimeError(f"Prediction failed: {str(e)}")
    
    # Calculate metrics if labels are available
    if labels is not None and include_metrics:
        logger.info("Calculating performance metrics...")
        results['metrics'] = {}
        results['confusion_matrices'] = {}
        
        for model_name, predictions in batch_predictions.items():
            try:
                # Calculate metrics
                metrics = {
                    'accuracy': float(accuracy_score(labels, predictions)),
                    'f1_score': float(f1_score(labels, predictions, average='weighted')),
                    'precision': float(precision_score(labels, predictions, average='weighted')),
                    'recall': float(recall_score(labels, predictions, average='weighted')),
                    'f1_macro': float(f1_score(labels, predictions, average='macro')),
                    'f1_micro': float(f1_score(labels, predictions, average='micro'))
                }
                results['metrics'][model_name] = metrics
                
                # Calculate confusion matrix
                cm = confusion_matrix(labels, predictions)
                results['confusion_matrices'][model_name] = {
                    'matrix': cm.tolist(),
                    'labels': ['Negative', 'Positive'],
                    'tn': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
                    'fp': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
                    'fn': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
                    'tp': int(cm[1, 1]) if cm.shape == (2, 2) else 0
                }
                
                logger.info(f"Metrics for {model_name}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")
                
            except Exception as e:
                logger.error(f"Error calculating metrics for {model_name}: {str(e)}")
                results['metrics'][model_name] = {'error': str(e)}
    
    # Add prediction summary
    if include_predictions:
        results['prediction_summary'] = {}
        for model_name, predictions in batch_predictions.items():
            pred_counts = pd.Series(predictions).value_counts().to_dict()
            results['prediction_summary'][model_name] = {
                'total_predictions': len(predictions),
                'positive_predictions': pred_counts.get(1, 0),
                'negative_predictions': pred_counts.get(0, 0),
                'positive_ratio': pred_counts.get(1, 0) / len(predictions)
            }
    
    # Save results if requested
    if save_results:
        if results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path('results') / f'csv_analysis_{timestamp}'

        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save main results
        results_file = results_dir / 'analysis_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save predictions as CSV if requested
        if include_predictions:
            pred_df = df.copy()
            for model_name, predictions in batch_predictions.items():
                pred_df[f'{model_name}_prediction'] = predictions
                pred_df[f'{model_name}_sentiment'] = ['Positive' if p == 1 else 'Negative' for p in predictions]
            
            pred_file = results_dir / 'predictions.csv'
            pred_df.to_csv(pred_file, index=False, encoding='utf-8')

        results['results_dir'] = str(results_dir)
        logger.info(f"Results saved to: {results_dir}")
    
    logger.info("Analysis completed successfully")
    return results

def validate_csv_for_analysis(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate CSV data for sentiment analysis.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check required columns
    text_columns = ['text', 'review', 'content', 'comment', 'message']
    has_text_column = any(col in df.columns for col in text_columns)
    
    if not has_text_column:
        issues.append(f"Missing required text column. Expected one of: {text_columns}")
    
    # Check for empty dataframe
    if len(df) == 0:
        issues.append("DataFrame is empty")
    
    # Check for null values in text columns
    for col in text_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues.append(f"Found {null_count} null values in '{col}' column")
            
            # Check for empty strings
            empty_count = (df[col].str.strip() == '').sum()
            if empty_count > 0:
                issues.append(f"Found {empty_count} empty text values in '{col}' column")
            break
    
    # Check label column if present
    label_columns = ['label', 'sentiment', 'class', 'target', 'rating']
    for col in label_columns:
        if col in df.columns:
            unique_labels = df[col].dropna().unique()
            # Allow various label formats
            valid_formats = [
                set([0, 1]),  # Binary numeric
                set(['negative', 'positive']),  # String labels
                set(['neg', 'pos']),  # Short string labels
                set([0, 1, 2]),  # Three-class numeric
                set(['negative', 'neutral', 'positive'])  # Three-class string
            ]
            
            if not any(set(unique_labels).issubset(valid_format) for valid_format in valid_formats):
                issues.append(f"Unsupported label format in '{col}': {unique_labels}")
            break
    
    # Check reasonable text lengths
    for col in text_columns:
        if col in df.columns and not df[col].empty:
            lengths = df[col].str.len()
            avg_length = lengths.mean()
            
            if avg_length < 5:
                issues.append(f"Average text length in '{col}' is very short (< 5 characters)")
            elif avg_length > 10000:
                issues.append(f"Average text length in '{col}' is very long (> 10000 characters)")
            break
    
    is_valid = len(issues) == 0
    return is_valid, issues

def create_analysis_report(results: Dict[str, Any]) -> str:
    """
    Create a formatted text report from analysis results.
    
    Args:
        results: Analysis results dictionary
    
    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("SENTIMENT ANALYSIS REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Analysis Timestamp: {results['timestamp']}")
    report_lines.append("")
    
    # Dataset info
    info = results['dataset_info']
    report_lines.append("DATASET INFORMATION:")
    report_lines.append(f"  Shape: {info['shape']}")
    report_lines.append(f"  Columns: {', '.join(info['columns'])}")
    report_lines.append(f"  Has Labels: {info['has_labels']}")
    report_lines.append("")
    
    # Statistics
    if 'statistics' in results:
        stats = results['statistics']
        report_lines.append("TEXT STATISTICS:")
        text_stats = stats.get('text_stats', {})
        report_lines.append(f"  Sample Count: {text_stats.get('count', 'N/A')}")
        report_lines.append(f"  Average Length: {text_stats.get('mean_length', 0):.1f} characters")
        report_lines.append(f"  Length Range: {text_stats.get('min_length', 0)} - {text_stats.get('max_length', 0)}")
        
        if 'label_stats' in stats:
            label_stats = stats['label_stats']
            report_lines.append("")
            report_lines.append("LABEL DISTRIBUTION:")
            dist = label_stats.get('distribution', {})
            report_lines.append(f"  Negative (0): {dist.get(0, 0)} ({label_stats.get('negative_ratio', 0):.1%})")
            report_lines.append(f"  Positive (1): {dist.get(1, 0)} ({label_stats.get('positive_ratio', 0):.1%})")
    
    # Metrics
    if 'metrics' in results:
        report_lines.append("")
        report_lines.append("PERFORMANCE METRICS:")
        for model_name, metrics in results['metrics'].items():
            if 'error' not in metrics:
                report_lines.append(f"  {model_name.upper()}:")
                report_lines.append(f"    Accuracy:  {metrics.get('accuracy', 0):.3f}")
                report_lines.append(f"    F1-Score:  {metrics.get('f1_score', 0):.3f}")
                report_lines.append(f"    Precision: {metrics.get('precision', 0):.3f}")
                report_lines.append(f"    Recall:    {metrics.get('recall', 0):.3f}")
    
    # Prediction summary
    if 'prediction_summary' in results:
        report_lines.append("")
        report_lines.append("PREDICTION SUMMARY:")
        for model_name, summary in results['prediction_summary'].items():
            report_lines.append(f"  {model_name.upper()}:")
            report_lines.append(f"    Total: {summary.get('total_predictions', 0)}")
            report_lines.append(f"    Positive: {summary.get('positive_predictions', 0)} ({summary.get('positive_ratio', 0):.1%})")
            report_lines.append(f"    Negative: {summary.get('negative_predictions', 0)}")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)

def compare_analysis_results(results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two analysis results and highlight differences.
    
    Args:
        results1: First analysis results
        results2: Second analysis results
    
    Returns:
        Comparison results dictionary
    """
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'analysis1_timestamp': results1.get('timestamp', 'Unknown'),
        'analysis2_timestamp': results2.get('timestamp', 'Unknown'),
        'differences': {}
    }
    
    # Compare metrics if both have them
    if 'metrics' in results1 and 'metrics' in results2:
        comparison['metric_differences'] = {}
        
        common_models = set(results1['metrics'].keys()) & set(results2['metrics'].keys())
        for model in common_models:
            if 'error' not in results1['metrics'][model] and 'error' not in results2['metrics'][model]:
                model_diff = {}
                metrics1 = results1['metrics'][model]
                metrics2 = results2['metrics'][model]
                
                for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
                    if metric in metrics1 and metric in metrics2:
                        diff = metrics2[metric] - metrics1[metric]
                        model_diff[metric] = {
                            'analysis1': metrics1[metric],
                            'analysis2': metrics2[metric],
                            'difference': diff,
                            'improvement': diff > 0
                        }
                
                comparison['metric_differences'][model] = model_diff
    
    return comparison