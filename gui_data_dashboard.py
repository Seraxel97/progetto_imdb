#!/usr/bin/env python3
"""
Advanced GUI Data Dashboard - Sentiment Analysis System (ENHANCED FIXED VERSION)
Professional GUI with automatic dataset analysis, pipeline execution, and intelligent insights.

ENHANCED FIXED VERSION - All critical bugs resolved + new features:
✅ 1. Fixed Deep Analysis and Full Pipeline functionality
✅ 2. Enhanced Test Your Text with complete scientific analysis
✅ 3. Added comprehensive Deep Scientific Text Analysis with terms/phrases/topics
✅ 4. Fixed numpy.int64 compatibility issues for JSON saving
✅ 5. Implemented proper keyword/phrase/topic extraction by sentiment
✅ 6. Enhanced scientific visualizations and reporting
✅ 7. Added proper progress tracking for all operations
✅ 8. Unified analysis architecture for consistency
✅ 9. Enhanced error handling and recovery
✅ 10. Complete scientific analysis package generation
✅ 11. FIXED: Test Your Text section with EnhancedFileAnalysisProcessor integration
✅ 12. FIXED: Deep Analysis and Full Pipeline buttons functionality
✅ 13. ENHANCED: Added comprehensive visualizations and download options
✅ 14. IMPROVED: Type sanitization for all numpy/pandas types

FEATURES:
- 🧠 Deep Text Analysis with semantic insights + EnhancedFileAnalysisProcessor integration
- 📊 Scientific approach with statistical reporting
- 💬 Intelligent analysis (now properly implemented with file processor)
- 📈 Enhanced statistical visualizations with top words charts
- 🗂️ Better file organization and download system
- 🔍 Advanced text pattern detection with phrase tables
- 📝 Comprehensive reporting system with topic analysis
- 🎯 Topic modeling and keyword extraction
- 📋 Complete phrase and argument analysis
- 📥 Enhanced download options (CSV/JSON) for all analysis sections
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import warnings
import os
import sys
import json
import io
import zipfile
import subprocess
import re
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional
import base64

# FIXED: Safe import for wordcloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

warnings.filterwarnings('ignore')

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
REPORTS_DIR = RESULTS_DIR / "reports"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Add scripts to path for imports
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# FIXED: Import enhanced analysis processor (required)
try:
    from enhanced_utils_unified import EnhancedFileAnalysisProcessor
    ENHANCED_PROCESSOR_AVAILABLE = True
except ImportError:
    ENHANCED_PROCESSOR_AVAILABLE = False

try:
    from pipeline_runner import run_dataset_analysis
    PIPELINE_RUNNER_AVAILABLE = True
except ImportError:
    PIPELINE_RUNNER_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🤖 Sentiment Analysis System - Enhanced Professional Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .insights-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .advanced-insight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .deep-analysis-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        color: #333;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .narrative-comment {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-style: italic;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .wordcloud-container {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .scientific-container {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .topic-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .enhanced-analysis-box {
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# FIXED: Define MLP model architecture for proper loading
class HateSpeechMLP(nn.Module):
    """Enhanced MLP model architecture - must match training script"""
    
    def __init__(self, input_dim=384, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super(HateSpeechMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            dropout_rate = dropout if i < len(hidden_dims) - 2 else dropout * 0.7
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

@st.cache_resource
def load_embedding_model():
    """Load embedding model with enhanced error handling"""
    try:
        with st.spinner("🔄 Loading SentenceTransformer model..."):
            local_model_dir = PROJECT_ROOT / "models" / "minilm-l6-v2"
            if local_model_dir.exists():
                model = SentenceTransformer(str(local_model_dir))
                st.success(f"✅ Loaded local embedding model from {local_model_dir}")
            else:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                st.success("✅ Loaded online embedding model")
            return model
    except Exception as e:
        st.error(f"❌ Error loading embedding model: {e}")
        return None

@st.cache_resource
def load_trained_models():
    """FIXED: Load all trained models with robust MLP handling"""
    models = {
        'mlp': None,
        'svm': None,
        'status': {
            'mlp': 'not_found',
            'svm': 'not_found'
        }
    }
    
    # FIXED: Load MLP model with better error handling
    mlp_paths = [
        MODELS_DIR / "mlp_model.pth",
        MODELS_DIR / "mlp_model_complete.pth",
        PROJECT_ROOT / "results" / "models" / "mlp_model.pth",
        PROJECT_ROOT / "results" / "models" / "mlp_model_complete.pth"
    ]
    
    for mlp_path in mlp_paths:
        if mlp_path.exists():
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                checkpoint = torch.load(mlp_path, map_location=device)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                model = HateSpeechMLP(input_dim=384).to(device)
                
                try:
                    model.load_state_dict(state_dict, strict=True)
                except Exception:
                    model.load_state_dict(state_dict, strict=False)
                
                model.eval()
                
                models['mlp'] = model
                models['status']['mlp'] = 'loaded'
                st.success(f"✅ MLP model loaded from {mlp_path}")
                break
                
            except Exception as e:
                st.warning(f"⚠️ Could not load MLP model from {mlp_path}: {e}")
                models['status']['mlp'] = 'error'
                continue
    
    if models['mlp'] is None:
        st.info("ℹ️ MLP model not found. Train one first using the pipeline.")
    
    # Load SVM model
    svm_paths = [
        MODELS_DIR / "svm_model.pkl",
        PROJECT_ROOT / "results" / "models" / "svm_model.pkl"
    ]
    
    for svm_path in svm_paths:
        if svm_path.exists():
            try:
                models['svm'] = joblib.load(svm_path)
                models['status']['svm'] = 'loaded'
                st.success(f"✅ SVM model loaded from {svm_path}")
                break
            except Exception as e:
                st.warning(f"⚠️ Could not load SVM model from {svm_path}: {e}")
                models['status']['svm'] = 'error'
    
    if models['svm'] is None:
        st.info("ℹ️ SVM model not found. Train one first using the pipeline.")
    
    return models

@st.cache_data
def load_main_dataset():
    """FIXED: Load and cache the main dataset for automatic display"""
    dataset_paths = [
        PROCESSED_DATA_DIR / "train.csv",
        PROCESSED_DATA_DIR / "test.csv", 
        PROCESSED_DATA_DIR / "val.csv",
        DATA_DIR / "raw" / "train.csv",
        DATA_DIR / "raw" / "imdb_raw.csv",
        PROJECT_ROOT / "data" / "train.csv"
    ]
    
    for path in dataset_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                st.success(f"✅ Main dataset loaded from {path}")
                return df, str(path)
            except Exception as e:
                st.warning(f"⚠️ Error loading {path}: {e}")
    
    st.info("ℹ️ No main dataset found. Upload a CSV to get started.")
    return None, None

# ENHANCED: Utility function to safely convert numpy types for JSON serialization
def safe_convert_for_json(obj):
    """IMPROVED: Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int_, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return str(obj)
    elif isinstance(obj, dict):
        return {str(k): safe_convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_convert_for_json(item) for item in obj]
    elif hasattr(obj, 'item'):  # Handle scalar numpy types
        return obj.item()
    else:
        return obj

# ENHANCED: Advanced keyword extraction by sentiment
def extract_keywords_by_sentiment(texts: List[str], predictions: List[int], top_n: int = 20) -> Dict:
    """
    Extract top keywords for each sentiment class using TF-IDF
    """
    try:
        keyword_analysis = {
            'positive': {'keywords': [], 'count': 0},
            'negative': {'keywords': [], 'count': 0},
            'neutral': {'keywords': [], 'count': 0}
        }
        
        label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
        
        # Group texts by sentiment
        sentiment_texts = {'positive': [], 'negative': [], 'neutral': []}
        for text, pred in zip(texts, predictions):
            sentiment = label_map.get(pred, 'neutral')
            sentiment_texts[sentiment].append(str(text))
            keyword_analysis[sentiment]['count'] += 1
        
        # Extract keywords for each sentiment using TF-IDF
        for sentiment, texts_list in sentiment_texts.items():
            if texts_list:
                try:
                    # Create TF-IDF vectorizer with custom stop words
                    stopwords = {
                        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
                        'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must',
                        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
                    }
                    
                    vectorizer = TfidfVectorizer(
                        max_features=top_n,
                        stop_words=stopwords,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95
                    )
                    
                    tfidf_matrix = vectorizer.fit_transform(texts_list)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Get average TF-IDF scores
                    avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                    
                    # Sort by score and get top keywords
                    keyword_scores = list(zip(feature_names, avg_scores))
                    keyword_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    keyword_analysis[sentiment]['keywords'] = [
                        {'keyword': kw, 'score': float(score)} 
                        for kw, score in keyword_scores[:top_n]
                    ]
                    
                except Exception as e:
                    st.warning(f"Could not extract keywords for {sentiment}: {e}")
                    # Fallback to simple word frequency
                    all_words = []
                    for text in texts_list:
                        words = re.findall(r'\b\w+\b', str(text).lower())
                        all_words.extend(words)
                    
                    word_freq = Counter(all_words)
                    keyword_analysis[sentiment]['keywords'] = [
                        {'keyword': word, 'score': count / len(all_words)} 
                        for word, count in word_freq.most_common(top_n)
                    ]
        
        return keyword_analysis
        
    except Exception as e:
        st.error(f"Error in keyword extraction: {e}")
        return {'positive': {'keywords': [], 'count': 0}, 'negative': {'keywords': [], 'count': 0}, 'neutral': {'keywords': [], 'count': 0}}

# ENHANCED: Advanced phrase extraction
def extract_phrases_by_sentiment(texts: List[str], predictions: List[int], top_n: int = 15) -> Dict:
    """
    Extract most frequent phrases (bigrams/trigrams) for each sentiment
    """
    try:
        phrase_analysis = {
            'positive': {'phrases': [], 'count': 0},
            'negative': {'phrases': [], 'count': 0},
            'neutral': {'phrases': [], 'count': 0}
        }
        
        label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
        
        # Group texts by sentiment
        sentiment_texts = {'positive': [], 'negative': [], 'neutral': []}
        for text, pred in zip(texts, predictions):
            sentiment = label_map.get(pred, 'neutral')
            sentiment_texts[sentiment].append(str(text))
            phrase_analysis[sentiment]['count'] += 1
        
        # Extract phrases for each sentiment
        for sentiment, texts_list in sentiment_texts.items():
            if texts_list:
                try:
                    # Extract bigrams and trigrams
                    all_phrases = []
                    for text in texts_list:
                        # Clean text
                        text_clean = re.sub(r'[^\w\s]', ' ', str(text).lower())
                        words = text_clean.split()
                        
                        # Generate bigrams
                        for i in range(len(words) - 1):
                            bigram = f"{words[i]} {words[i+1]}"
                            if len(bigram) > 5:  # Skip very short phrases
                                all_phrases.append(bigram)
                        
                        # Generate trigrams
                        for i in range(len(words) - 2):
                            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                            if len(trigram) > 8:  # Skip very short phrases
                                all_phrases.append(trigram)
                    
                    # Count phrase frequency
                    phrase_freq = Counter(all_phrases)
                    total_phrases = len(all_phrases)
                    
                    phrase_analysis[sentiment]['phrases'] = [
                        {
                            'phrase': phrase, 
                            'frequency': count,
                            'percentage': (count / max(1, total_phrases)) * 100
                        } 
                        for phrase, count in phrase_freq.most_common(top_n) if count > 1
                    ]
                    
                except Exception as e:
                    st.warning(f"Could not extract phrases for {sentiment}: {e}")
                    phrase_analysis[sentiment]['phrases'] = []
        
        return phrase_analysis
        
    except Exception as e:
        st.error(f"Error in phrase extraction: {e}")
        return {'positive': {'phrases': [], 'count': 0}, 'negative': {'phrases': [], 'count': 0}, 'neutral': {'phrases': [], 'count': 0}}

# ENHANCED: Topic extraction and analysis
def extract_topics_by_sentiment(texts: List[str], predictions: List[int]) -> Dict:
    """
    Extract main topics and themes for each sentiment class
    """
    try:
        topic_analysis = {
            'positive': {'topics': [], 'themes': [], 'count': 0},
            'negative': {'topics': [], 'themes': [], 'count': 0},
            'neutral': {'topics': [], 'themes': [], 'count': 0}
        }
        
        label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
        
        # Predefined topic keywords for different domains
        topic_keywords = {
            'quality': ['quality', 'excellent', 'poor', 'good', 'bad', 'amazing', 'terrible'],
            'service': ['service', 'staff', 'support', 'help', 'customer', 'team'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money', 'worth'],
            'delivery': ['delivery', 'shipping', 'fast', 'slow', 'quick', 'arrived'],
            'product': ['product', 'item', 'purchase', 'buy', 'order', 'received'],
            'experience': ['experience', 'feel', 'enjoyed', 'disappointed', 'satisfied'],
            'recommendation': ['recommend', 'suggest', 'advice', 'tell', 'friends'],
            'communication': ['communication', 'response', 'reply', 'contact', 'email'],
            'features': ['feature', 'function', 'work', 'works', 'functionality'],
            'design': ['design', 'look', 'appearance', 'style', 'beautiful', 'ugly']
        }
        
        # Group texts by sentiment
        sentiment_texts = {'positive': [], 'negative': [], 'neutral': []}
        for text, pred in zip(texts, predictions):
            sentiment = label_map.get(pred, 'neutral')
            sentiment_texts[sentiment].append(str(text).lower())
            topic_analysis[sentiment]['count'] += 1
        
        # Analyze topics for each sentiment
        for sentiment, texts_list in sentiment_texts.items():
            if texts_list:
                topic_scores = {}
                
                # Calculate topic relevance scores
                for topic, keywords in topic_keywords.items():
                    score = 0
                    mentions = 0
                    
                    for text in texts_list:
                        for keyword in keywords:
                            if keyword in text:
                                score += text.count(keyword)
                                mentions += 1
                    
                    if mentions > 0:
                        topic_scores[topic] = {
                            'score': score,
                            'mentions': mentions,
                            'relevance': score / len(texts_list)
                        }
                
                # Sort topics by relevance
                sorted_topics = sorted(
                    topic_scores.items(), 
                    key=lambda x: x[1]['relevance'], 
                    reverse=True
                )
                
                topic_analysis[sentiment]['topics'] = [
                    {
                        'topic': topic,
                        'score': data['score'],
                        'mentions': data['mentions'],
                        'relevance': round(data['relevance'], 3)
                    }
                    for topic, data in sorted_topics[:8] if data['mentions'] > 1
                ]
                
                # Extract themes (common word combinations)
                all_text = ' '.join(texts_list)
                words = re.findall(r'\b\w+\b', all_text)
                word_freq = Counter(words)
                
                # Get most common meaningful words as themes
                stopwords = {
                    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                    'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
                    'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must',
                    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
                }
                
                themes = [
                    {'theme': word, 'frequency': count}
                    for word, count in word_freq.most_common(10)
                    if word not in stopwords and len(word) > 3 and count > 2
                ]
                
                topic_analysis[sentiment]['themes'] = themes[:6]
        
        return topic_analysis
        
    except Exception as e:
        st.error(f"Error in topic extraction: {e}")
        return {'positive': {'topics': [], 'themes': [], 'count': 0}, 'negative': {'topics': [], 'themes': [], 'count': 0}, 'neutral': {'topics': [], 'themes': [], 'count': 0}}

# REMOVED: enhanced_single_text_analysis function - simplified to direct processor call

# ENHANCED: Generate comprehensive scientific analysis for single text
def generate_single_text_scientific_analysis(text: str, prediction: int, confidence: float) -> Dict:
    """
    Generate scientific analysis for a single text input
    """
    try:
        analysis = {
            'text_statistics': {},
            'prediction_analysis': {},
            'linguistic_features': {},
            'keyword_analysis': {},
            'semantic_analysis': {}
        }
        
        # Basic text statistics
        words = re.findall(r'\b\w+\b', str(text).lower())
        sentences = re.split(r'[.!?]+', str(text))
        
        analysis['text_statistics'] = {
            'character_count': len(str(text)),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / max(1, len(words))
        }
        
        # Prediction analysis
        label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
        confidence_level = 'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'
        
        analysis['prediction_analysis'] = {
            'predicted_sentiment': label_map.get(prediction, 'Unknown'),
            'confidence_score': float(confidence),
            'confidence_level': confidence_level,
            'prediction_numeric': int(prediction)
        }
        
        # Linguistic features
        punctuation_count = sum(1 for char in str(text) if char in '.,;:!?')
        exclamation_count = str(text).count('!')
        question_count = str(text).count('?')
        caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        
        analysis['linguistic_features'] = {
            'punctuation_density': punctuation_count / max(1, len(str(text))),
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'caps_words': caps_words,
            'contains_url': int('http' in str(text).lower() or 'www.' in str(text).lower()),
            'contains_email': int('@' in str(text) and '.' in str(text))
        }
        
        # Keyword analysis
        word_freq = Counter(words)
        stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        meaningful_words = [
            {'word': word, 'frequency': count}
            for word, count in word_freq.most_common(10)
            if word not in stopwords and len(word) > 2
        ]
        
        analysis['keyword_analysis'] = {
            'top_words': meaningful_words[:5],
            'rare_words': [word for word, count in word_freq.items() if count == 1 and len(word) > 4][:5],
            'word_diversity': len(set(words)) / max(1, len(words))
        }
        
        # Semantic analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'perfect', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing', 'poor']
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        analysis['semantic_analysis'] = {
            'positive_words': positive_count,
            'negative_words': negative_count,
            'sentiment_balance': positive_count - negative_count,
            'emotional_intensity': positive_count + negative_count,
            'detected_emotions': []
        }
        
        # Detect basic emotions
        emotion_words = {
            'joy': ['happy', 'joy', 'excited', 'delighted'],
            'anger': ['angry', 'mad', 'furious', 'annoyed'],
            'sadness': ['sad', 'disappointed', 'unhappy'],
            'fear': ['scared', 'worried', 'afraid'],
            'surprise': ['surprised', 'shocked', 'amazed']
        }
        
        for emotion, emotion_word_list in emotion_words.items():
            if any(word in words for word in emotion_word_list):
                analysis['semantic_analysis']['detected_emotions'].append(emotion)
        
        return safe_convert_for_json(analysis)
        
    except Exception as e:
        st.error(f"Error in single text scientific analysis: {e}")
        return {}

# FIXED: Add missing generate_narrative_insights function
def generate_narrative_insights(df: pd.DataFrame, predictions: Dict = None, 
                               metrics: Dict = None, deep_analysis: Dict = None) -> List[str]:
    """
    Generate intelligent narrative insights from analysis results
    """
    insights = []
    
    try:
        # Basic dataset insights
        total_samples = len(df)
        insights.append(f"🔍 Analysis reveals {total_samples:,} text samples in your dataset.")
        
        # Deep analysis insights
        if deep_analysis:
            basic_stats = deep_analysis.get('basic_stats', {})
            semantic_patterns = deep_analysis.get('semantic_patterns', {})
            quality_metrics = deep_analysis.get('quality_metrics', {})
            
            # Vocabulary insights
            if basic_stats:
                total_words = basic_stats.get('total_words', 0)
                unique_words = basic_stats.get('unique_words', 0)
                vocab_richness = basic_stats.get('vocabulary_richness', 0)
                
                if total_words > 0:
                    insights.append(f"📚 Your texts contain {total_words:,} total words with {unique_words:,} unique terms.")
                    
                    if vocab_richness > 0.3:
                        insights.append(f"🎨 High vocabulary richness ({vocab_richness:.3f}) indicates diverse and sophisticated language use.")
                    elif vocab_richness > 0.15:
                        insights.append(f"📖 Moderate vocabulary richness ({vocab_richness:.3f}) shows balanced language complexity.")
                    else:
                        insights.append(f"📝 Lower vocabulary richness ({vocab_richness:.3f}) suggests more repetitive or specialized language.")
            
            # Sentiment insights
            if semantic_patterns:
                pos_indicators = semantic_patterns.get('positive_indicators', 0)
                neg_indicators = semantic_patterns.get('negative_indicators', 0)
                sentiment_ratio = semantic_patterns.get('sentiment_ratio', 1)
                
                if pos_indicators > 0 or neg_indicators > 0:
                    total_sentiment = pos_indicators + neg_indicators
                    insights.append(f"💭 Found {total_sentiment} sentiment indicators: {pos_indicators} positive, {neg_indicators} negative.")
                    
                    if sentiment_ratio > 2:
                        insights.append("😊 Strong positive sentiment bias detected in the language patterns.")
                    elif sentiment_ratio < 0.5:
                        insights.append("😞 Strong negative sentiment bias found in the text expressions.")
                    else:
                        insights.append("⚖️ Balanced sentiment distribution indicates neutral language tone.")
                
                # Emotional diversity
                emotion_diversity = semantic_patterns.get('emotional_diversity', 0)
                if emotion_diversity > 3:
                    insights.append(f"🎭 Rich emotional expression with {emotion_diversity} different emotion types detected.")
                elif emotion_diversity > 1:
                    insights.append(f"🎯 Moderate emotional range with {emotion_diversity} emotion categories present.")
            
            # Quality insights
            if quality_metrics:
                quality_score = quality_metrics.get('quality_score', 0)
                data_completeness = quality_metrics.get('data_completeness', 0)
                
                if quality_score > 0.8:
                    insights.append(f"✅ Excellent data quality ({quality_score:.1%}) with minimal issues detected.")
                elif quality_score > 0.6:
                    insights.append(f"👍 Good data quality ({quality_score:.1%}) with minor cleaning opportunities.")
                else:
                    insights.append(f"⚠️ Data quality score ({quality_score:.1%}) suggests preprocessing could improve results.")
                
                if data_completeness < 0.9:
                    missing_pct = (1 - data_completeness) * 100
                    insights.append(f"🔍 {missing_pct:.1f}% of texts have missing or empty content requiring attention.")
        
        # Model prediction insights
        if predictions and metrics:
            model_count = len(predictions)
            insights.append(f"🤖 {model_count} AI model{'s' if model_count > 1 else ''} analyzed your data.")
            
            # Confidence analysis
            avg_confidences = []
            for model_name, model_metrics in metrics.items():
                confidence_avg = model_metrics.get('confidence_avg', 0)
                avg_confidences.append(confidence_avg)
                
                if confidence_avg > 0.85:
                    insights.append(f"🎯 {model_name.upper()} model shows high confidence ({confidence_avg:.3f}) in predictions.")
                elif confidence_avg > 0.7:
                    insights.append(f"👌 {model_name.upper()} model demonstrates good confidence ({confidence_avg:.3f}) in classifications.")
                else:
                    insights.append(f"🤔 {model_name.upper()} model shows moderate confidence ({confidence_avg:.3f}) - consider more training data.")
            
            # Model agreement
            if len(predictions) == 2:
                pred1, pred2 = list(predictions.values())
                if len(pred1) == len(pred2):
                    agreement = np.mean(np.array(pred1) == np.array(pred2))
                    if agreement > 0.9:
                        insights.append(f"🤝 Excellent model agreement ({agreement:.1%}) increases prediction reliability.")
                    elif agreement > 0.75:
                        insights.append(f"👥 Good model agreement ({agreement:.1%}) supports prediction confidence.")
                    else:
                        insights.append(f"🔄 Models show different perspectives ({agreement:.1%} agreement) - manual review recommended.")
        
        # Summary insight
        if len(insights) > 3:
            insights.append("🚀 Analysis complete! Your data shows rich patterns ready for deeper exploration.")
        
        return insights
        
    except Exception as e:
        st.error(f"Error generating narrative insights: {e}")
        return [
            "🔍 Basic analysis completed successfully.",
            "📊 Data structure appears suitable for sentiment analysis.",
            "🤖 Ready for model predictions and deeper insights."
        ]

def enhanced_deep_text_analysis(df: pd.DataFrame, text_column: str) -> Dict:
    """
    FIXED: Deep semantic text analysis with comprehensive insights and error handling
    """
    try:
        texts = df[text_column].fillna('').astype(str)
        
        analysis = {
            'basic_stats': {},
            'word_analysis': {},
            'semantic_patterns': {},
            'content_structure': {},
            'quality_metrics': {},
            'linguistic_features': {},
            'topic_insights': {}
        }
        
        # Basic statistics with error handling
        all_words = []
        word_counts = []
        char_counts = []
        sentence_counts = []
        
        for text in texts:
            try:
                words = re.findall(r'\b\w+\b', str(text).lower())
                sentences = re.split(r'[.!?]+', str(text))
                
                all_words.extend(words)
                word_counts.append(len(words))
                char_counts.append(len(str(text)))
                sentence_counts.append(len([s for s in sentences if s.strip()]))
            except Exception:
                word_counts.append(0)
                char_counts.append(0)
                sentence_counts.append(0)
        
        # Word frequency analysis with safety checks
        word_freq = Counter(all_words) if all_words else Counter()
        most_common = word_freq.most_common(50) if word_freq else []
        
        total_words = len(all_words)
        unique_words = len(word_freq)
        
        analysis['basic_stats'] = {
            'total_texts': len(texts),
            'total_words': total_words,
            'unique_words': unique_words,
            'avg_words_per_text': np.mean(word_counts) if word_counts else 0,
            'avg_chars_per_text': np.mean(char_counts) if char_counts else 0,
            'avg_sentences_per_text': np.mean(sentence_counts) if sentence_counts else 0,
            'vocabulary_richness': unique_words / max(1, total_words),
            'median_text_length': np.median(word_counts) if word_counts else 0,
            'std_text_length': np.std(word_counts) if word_counts else 0
        }
        
        # Advanced word analysis with safety
        analysis['word_analysis'] = {
            'top_words': most_common[:20],
            'rare_words_count': sum(1 for count in word_freq.values() if count == 1),
            'common_words': [(word, count) for word, count in most_common if count > len(texts) * 0.1],
            'word_length_avg': np.mean([len(word) for word in all_words]) if all_words else 0,
            'long_words': [word for word in word_freq.keys() if len(word) > 10][:20],
            'short_words': [word for word in word_freq.keys() if len(word) <= 3][:20]
        }
        
        # Semantic patterns with enhanced lexicons
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'love', 'perfect', 'best', 'awesome', 'brilliant', 'outstanding',
            'beautiful', 'incredible', 'superb', 'magnificent', 'terrific'
        ]
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 
            'disappointing', 'boring', 'poor', 'annoying', 'frustrating',
            'disgusting', 'pathetic', 'ridiculous', 'stupid', 'useless'
        ]
        
        positive_count = sum(word_freq.get(word, 0) for word in positive_words)
        negative_count = sum(word_freq.get(word, 0) for word in negative_words)
        
        # Emotional indicators with error handling
        emotion_words = {
            'joy': ['happy', 'joy', 'excited', 'delighted', 'thrilled'],
            'anger': ['angry', 'furious', 'mad', 'outraged', 'livid'],
            'sadness': ['sad', 'depressed', 'miserable', 'gloomy', 'melancholy'],
            'fear': ['scared', 'afraid', 'terrified', 'anxious', 'worried'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned']
        }
        
        emotion_counts = {}
        for emotion, words in emotion_words.items():
            emotion_counts[emotion] = sum(word_freq.get(word, 0) for word in words)
        
        analysis['semantic_patterns'] = {
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'sentiment_ratio': positive_count / max(1, negative_count),
            'emotional_words_total': positive_count + negative_count,
            'emotion_distribution': emotion_counts,
            'dominant_emotion': max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral',
            'emotional_diversity': len([e for e in emotion_counts.values() if e > 0])
        }
        
        # Content structure analysis with error handling
        questions = sum(1 for text in texts if '?' in str(text))
        exclamations = sum(1 for text in texts if '!' in str(text))
        all_caps = sum(1 for text in texts if any(word.isupper() and len(word) > 2 for word in str(text).split()))
        
        # Text length categories with safe counting
        very_short = sum(1 for count in word_counts if count < 5)
        short = sum(1 for count in word_counts if 5 <= count < 15)
        medium = sum(1 for count in word_counts if 15 <= count < 50)
        long_texts = sum(1 for count in word_counts if 50 <= count < 100)
        very_long = sum(1 for count in word_counts if count >= 100)
        
        analysis['content_structure'] = {
            'questions_count': questions,
            'exclamations_count': exclamations,
            'all_caps_texts': all_caps,
            'length_distribution': {
                'very_short': very_short,
                'short': short,
                'medium': medium,
                'long': long_texts,
                'very_long': very_long
            },
            'interactive_ratio': (questions + exclamations) / max(1, len(texts)),
            'emphasis_ratio': all_caps / max(1, len(texts))
        }
        
        # Quality metrics with comprehensive checks
        empty_texts = sum(1 for text in texts if not str(text).strip())
        very_short_texts = sum(1 for text in texts if len(str(text).strip()) < 10)
        
        # Improved spam detection
        potential_spam = 0
        for text in texts:
            text_str = str(text)
            if (text_str.count('!') > 5 or 
                text_str.count('?') > 3 or 
                (len(text_str.split()) > 0 and len(set(text_str.split())) < len(text_str.split()) * 0.5)):
                potential_spam += 1
        
        # Calculate readability with safety
        try:
            avg_words_per_sentence = np.mean([
                len(str(text).split()) / max(1, len(re.split(r'[.!?]+', str(text)))) 
                for text in texts
            ])
        except:
            avg_words_per_sentence = 10
        
        analysis['quality_metrics'] = {
            'empty_texts': empty_texts,
            'very_short_texts': very_short_texts,
            'potential_spam': potential_spam,
            'quality_score': 1 - (empty_texts + very_short_texts + potential_spam) / max(1, len(texts)),
            'readability_score': min(1.0, 20 / max(1, avg_words_per_sentence)),
            'data_completeness': 1 - (empty_texts / max(1, len(texts)))
        }
        
        # Linguistic features with error handling
        urls = sum(1 for text in texts if 'http' in str(text).lower() or 'www.' in str(text).lower())
        emails = sum(1 for text in texts if '@' in str(text) and '.' in str(text))
        numbers = sum(1 for text in texts if any(char.isdigit() for char in str(text)))
        
        try:
            punctuation_heavy = sum(1 for text in texts if 
                                  sum(1 for char in str(text) if char in '!@#$%^&*().,;:') > len(str(text)) * 0.1)
            avg_punctuation = np.mean([
                sum(1 for char in str(text) if char in '.,;:!?') / max(1, len(str(text)))
                for text in texts
            ])
        except:
            punctuation_heavy = 0
            avg_punctuation = 0
        
        analysis['linguistic_features'] = {
            'contains_urls': urls,
            'contains_emails': emails,
            'contains_numbers': numbers,
            'punctuation_heavy': punctuation_heavy,
            'avg_punctuation_ratio': avg_punctuation
        }
        
        # Topic insights with robust keyword extraction
        stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 
            'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 
            'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 
            'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 
            'its', 'our', 'their', 'a', 'an'
        }
        
        meaningful_words = [(word, count) for word, count in most_common 
                          if word not in stopwords and len(word) > 3 and count > 2]
        
        topic_words = [word for word, count in meaningful_words[:10]]
        
        # Domain detection with safety
        domain_indicators = {
            'movie_related': sum(word_freq.get(word, 0) for word in ['movie', 'film', 'actor', 'director', 'cinema']),
            'product_related': sum(word_freq.get(word, 0) for word in ['product', 'quality', 'price', 'buy', 'purchase']),
            'service_related': sum(word_freq.get(word, 0) for word in ['service', 'staff', 'support', 'help', 'customer'])
        }
        
        analysis['topic_insights'] = {
            'key_terms': meaningful_words[:15],
            'potential_topics': topic_words,
            'topic_diversity': len(meaningful_words),
            'domain_indicators': domain_indicators
        }
        
        return analysis
        
    except Exception as e:
        st.error(f"Error in enhanced deep text analysis: {e}")
        return {
            'basic_stats': {'total_texts': 0, 'total_words': 0, 'unique_words': 0, 'vocabulary_richness': 0},
            'word_analysis': {'top_words': [], 'rare_words_count': 0},
            'semantic_patterns': {'positive_indicators': 0, 'negative_indicators': 0, 'sentiment_ratio': 1},
            'content_structure': {'questions_count': 0, 'exclamations_count': 0},
            'quality_metrics': {'quality_score': 0, 'data_completeness': 0},
            'linguistic_features': {},
            'topic_insights': {'key_terms': [], 'topic_diversity': 0}
        }

def generate_scientific_report(df: pd.DataFrame, predictions: Dict = None, 
                              metrics: Dict = None, deep_analysis: Dict = None) -> Dict:
    """
    SCIENTIFIC FIX: Generate neutral statistical report instead of narrative insights
    """
    report = {
        'dataset_statistics': {},
        'sentiment_distribution': {},
        'model_performance': {},
        'linguistic_analysis': {},
        'term_frequency': {},
        'quality_metrics': {}
    }
    
    try:
        # Dataset statistics
        if df is not None:
            report['dataset_statistics'] = {
                'total_samples': len(df),
                'data_types': dict(df.dtypes.astype(str)),
                'missing_values': safe_convert_for_json(df.isnull().sum().to_dict()),
                'duplicate_rows': int(df.duplicated().sum()),
                'memory_usage_mb': float(df.memory_usage(deep=True).sum() / (1024**2))
            }
        
        # Deep analysis statistics
        if deep_analysis:
            basic_stats = deep_analysis.get('basic_stats', {})
            semantic_patterns = deep_analysis.get('semantic_patterns', {})
            quality_metrics = deep_analysis.get('quality_metrics', {})
            word_analysis = deep_analysis.get('word_analysis', {})
            
            report['linguistic_analysis'] = {
                'total_words': basic_stats.get('total_words', 0),
                'unique_words': basic_stats.get('unique_words', 0),
                'vocabulary_richness': basic_stats.get('vocabulary_richness', 0),
                'avg_words_per_text': basic_stats.get('avg_words_per_text', 0),
                'avg_chars_per_text': basic_stats.get('avg_chars_per_text', 0),
                'positive_indicators': semantic_patterns.get('positive_indicators', 0),
                'negative_indicators': semantic_patterns.get('negative_indicators', 0),
                'sentiment_ratio': semantic_patterns.get('sentiment_ratio', 1),
                'emotion_distribution': semantic_patterns.get('emotion_distribution', {})
            }
            
            report['quality_metrics'] = {
                'overall_quality_score': quality_metrics.get('quality_score', 0),
                'data_completeness': quality_metrics.get('data_completeness', 0),
                'readability_score': quality_metrics.get('readability_score', 0),
                'empty_texts': quality_metrics.get('empty_texts', 0),
                'potential_spam': quality_metrics.get('potential_spam', 0)
            }
            
            report['term_frequency'] = {
                'most_common_terms': word_analysis.get('top_words', [])[:10],
                'rare_words_count': word_analysis.get('rare_words_count', 0),
                'avg_word_length': word_analysis.get('word_length_avg', 0)
            }
        
        # Model performance metrics
        if predictions and metrics:
            report['model_performance'] = {}
            for model_name, pred in predictions.items():
                if model_name in metrics:
                    model_metrics = metrics[model_name]
                    pred_dist = dict(zip(*np.unique(pred, return_counts=True)))
                    
                    report['model_performance'][model_name] = {
                        'model_type': model_metrics.get('model_type', 'Unknown'),
                        'avg_confidence': model_metrics.get('confidence_avg', 0),
                        'confidence_std': model_metrics.get('confidence_std', 0),
                        'prediction_distribution': safe_convert_for_json(pred_dist),
                        'total_predictions': len(pred)
                    }
        
        # Sentiment distribution analysis
        if predictions:
            all_predictions = []
            for pred in predictions.values():
                all_predictions.extend(pred)
            
            if all_predictions:
                unique_preds, counts = np.unique(all_predictions, return_counts=True)
                total = len(all_predictions)
                
                label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
                report['sentiment_distribution'] = {
                    'counts': {label_map.get(pred, f'class_{pred}'): int(count) for pred, count in zip(unique_preds, counts)},
                    'percentages': {label_map.get(pred, f'class_{pred}'): float((count/total)*100) for pred, count in zip(unique_preds, counts)},
                    'total_classified': total
                }
        
        return safe_convert_for_json(report)
        
    except Exception as e:
        st.error(f"Error generating scientific report: {e}")
        return {'error': str(e)}

def analyze_sentiment_by_class(texts: List[str], predictions: List[int]) -> Dict:
    """
    SCIENTIFIC FIX: Analyze text patterns by sentiment class
    """
    try:
        analysis = {
            'positive': {'texts': [], 'word_freq': Counter(), 'stats': {}},
            'negative': {'texts': [], 'word_freq': Counter(), 'stats': {}},
            'neutral': {'texts': [], 'word_freq': Counter(), 'stats': {}}
        }
        
        label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
        
        for text, pred in zip(texts, predictions):
            sentiment = label_map.get(pred, 'neutral')
            if sentiment in analysis:
                analysis[sentiment]['texts'].append(text)
                
                words = re.findall(r'\b\w+\b', str(text).lower())
                analysis[sentiment]['word_freq'].update(words)
        
        # Calculate statistics for each sentiment class
        for sentiment in analysis:
            texts_list = analysis[sentiment]['texts']
            if texts_list:
                analysis[sentiment]['stats'] = {
                    'count': len(texts_list),
                    'avg_length': np.mean([len(str(text)) for text in texts_list]),
                    'avg_words': np.mean([len(str(text).split()) for text in texts_list]),
                    'top_words': analysis[sentiment]['word_freq'].most_common(10)
                }
            else:
                analysis[sentiment]['stats'] = {
                    'count': 0,
                    'avg_length': 0,
                    'avg_words': 0,
                    'top_words': []
                }
        
        return safe_convert_for_json(analysis)
        
    except Exception as e:
        st.error(f"Error in sentiment class analysis: {e}")
        return {}

def create_wordcloud_visualization(text_data: List[str], title: str = "Word Cloud") -> str:
    """
    FIXED: Create wordcloud visualization with proper error handling
    """
    if not WORDCLOUD_AVAILABLE:
        return ""
    
    try:
        all_text = ' '.join([str(text) for text in text_data if text])
        
        if not all_text.strip():
            return ""
        
        stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 
            'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 
            'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 
            'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 
            'its', 'our', 'their', 'a', 'an'
        }
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            stopwords=stopwords,
            max_words=100,
            colormap='viridis',
            relative_scaling=0.5,
            random_state=42,
            collocations=False
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        plt.close(fig)
        img_buffer.close()
        
        return img_base64
        
    except Exception as e:
        st.warning(f"Could not generate wordcloud: {e}")
        return ""

def create_timestamp_session():
    """Create a unique timestamp for this analysis session"""
    if 'session_timestamp' not in st.session_state:
        st.session_state.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return st.session_state.session_timestamp

def get_session_results_dir():
    """FIXED: Get the results directory for current session with proper structure"""
    timestamp = create_timestamp_session()
    session_dir = RESULTS_DIR / f"session_{timestamp}"
    
    subdirs = ['processed', 'embeddings', 'models', 'reports', 'plots', 'insights', 'wordclouds']
    for subdir in subdirs:
        (session_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return session_dir

def run_pipeline_step(step_name: str, command: List[str], description: str) -> bool:
    """
    FIXED: Run a single pipeline step with enhanced error handling
    """
    try:
        with st.spinner(f"🔄 {description}..."):
            st.write(f"**{step_name}:** {description}")
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                st.success(f"✅ {step_name} completed successfully")
                if result.stdout:
                    st.text(f"Output: {result.stdout[:200]}...")
                return True
            else:
                st.error(f"❌ {step_name} failed")
                if result.stderr:
                    st.error(f"Error: {result.stderr[:200]}...")
                return False
                
    except subprocess.TimeoutExpired:
        st.error(f"⏰ {step_name} timed out after 5 minutes")
        return False
    except Exception as e:
        st.error(f"❌ {step_name} error: {e}")
        return False

# REMOVED: run_enhanced_pipeline function - no longer needed as we use direct calls

def analyze_single_csv(uploaded_file, embedding_model):
    """FIXED: Advanced CSV analysis with comprehensive error handling"""
    try:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {e}")
            return None, None, None
        
        text_columns = ['review', 'text', 'content', 'comment', 'message', 'description', 'body', 'post']
        text_col = None
        
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            st.error(f"❌ CSV must contain one of these columns: {text_columns}")
            st.info(f"Found columns: {list(df.columns)}")
            return None, None, None
        
        texts = df[text_col].fillna('').astype(str).tolist()
        
        if not texts or all(not text.strip() for text in texts):
            st.error("❌ No valid text data found in the specified column")
            return None, None, None
        
        if embedding_model is None:
            st.error("❌ Embedding model not available")
            return None, None, None
        
        with st.spinner("🔄 Generating embeddings..."):
            try:
                embeddings = embedding_model.encode(texts, show_progress_bar=True)
            except Exception as e:
                st.error(f"❌ Error generating embeddings: {e}")
                return None, None, None
        
        with st.spinner("🔄 Performing enhanced deep text analysis..."):
            deep_analysis = enhanced_deep_text_analysis(df, text_col)
        
        stats = {
            'total_reviews': len(df),
            'avg_length': df[text_col].str.len().mean(),
            'max_length': df[text_col].str.len().max(),
            'min_length': df[text_col].str.len().min(),
            'null_values': df[text_col].isnull().sum(),
            'empty_strings': (df[text_col] == '').sum(),
            'duplicates': df.duplicated().sum(),
            'text_column': text_col,
            'embeddings_shape': embeddings.shape,
            'columns': list(df.columns),
            'deep_analysis': deep_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        sentiment_columns = ['sentiment', 'label', 'class', 'target', 'rating']
        for col in sentiment_columns:
            if col in df.columns:
                try:
                    stats['sentiment_distribution'] = safe_convert_for_json(df[col].value_counts().to_dict())
                    stats['sentiment_column'] = col
                    break
                except Exception:
                    continue
        
        return df, embeddings, stats
        
    except Exception as e:
        st.error(f"❌ Error analyzing CSV: {e}")
        return None, None, None

# SIMPLIFIED: Deep analysis specific function
def analyze_single_csv_deep(uploaded_file, embedding_model):
    """SIMPLIFIED: Perform basic CSV analysis for deep mode"""
    try:
        # Get basic analysis first
        df, embeddings, stats = analyze_single_csv(uploaded_file, embedding_model)
        
        if df is None:
            return None, None, None
        
        # Add analysis type marker
        stats['analysis_type'] = 'deep'
        
        return df, embeddings, stats
        
    except Exception as e:
        st.error(f"❌ Error in deep CSV analysis: {e}")
        return None, None, None

def predict_sentiment_enhanced(texts, embeddings, models):
    """FIXED: Enhanced prediction with robust error handling for both models"""
    predictions = {}
    metrics = {}
    
    # SVM predictions with enhanced error handling
    if models.get('svm') is not None:
        try:
            with st.spinner("🔄 Making SVM predictions..."):
                svm_package = models['svm']
                
                if isinstance(svm_package, dict):
                    svm_model = svm_package.get('model')
                    scaler = svm_package.get('scaler')
                    label_encoder = svm_package.get('label_encoder')
                else:
                    svm_model = svm_package
                    scaler = None
                    label_encoder = None
                
                if svm_model is None:
                    raise ValueError("SVM model not found in package")
                
                embeddings_scaled = scaler.transform(embeddings) if scaler else embeddings
                svm_pred = svm_model.predict(embeddings_scaled)
                
                if hasattr(svm_model, 'predict_proba'):
                    try:
                        svm_proba = svm_model.predict_proba(embeddings_scaled)
                        svm_confidence = np.max(svm_proba, axis=1)
                    except Exception:
                        svm_confidence = np.ones(len(svm_pred)) * 0.7
                else:
                    svm_confidence = np.ones(len(svm_pred)) * 0.7
                
                # FIXED: Convert numpy types for JSON compatibility
                predictions['svm'] = [int(p) for p in svm_pred]
                metrics['svm'] = {
                    'model_type': 'SVM',
                    'confidence_avg': float(np.mean(svm_confidence)),
                    'confidence_std': float(np.std(svm_confidence)),
                    'confidence_scores': [float(c) for c in svm_confidence],
                    'prediction_distribution': safe_convert_for_json({int(k): int(v) for k, v in dict(zip(*np.unique(svm_pred, return_counts=True))).items()})
                }
                
                st.success(f"✅ SVM predictions completed (confidence: {np.mean(svm_confidence):.3f})")
                
        except Exception as e:
            st.warning(f"⚠️ SVM prediction failed: {e}")
    
    # MLP predictions with comprehensive error handling
    if models.get('mlp') is not None:
        try:
            with st.spinner("🔄 Making MLP predictions..."):
                model = models['mlp']
                model.eval()
                
                try:
                    device = next(model.parameters()).device
                except Exception:
                    device = torch.device('cpu')
                
                if isinstance(embeddings, np.ndarray):
                    embeddings_tensor = torch.FloatTensor(embeddings).to(device)
                else:
                    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
                
                batch_size = 32
                all_outputs = []
                
                with torch.no_grad():
                    for i in range(0, len(embeddings_tensor), batch_size):
                        try:
                            batch = embeddings_tensor[i:i+batch_size]
                            outputs = model(batch)
                            all_outputs.append(outputs)
                        except Exception as e:
                            st.warning(f"Batch {i//batch_size + 1} failed: {e}")
                            dummy_output = torch.zeros(len(batch), 1).to(device)
                            all_outputs.append(dummy_output)
                    
                    if not all_outputs:
                        raise ValueError("All batches failed")
                    
                    full_outputs = torch.cat(all_outputs, dim=0)
                    
                    if full_outputs.dim() == 2 and full_outputs.shape[1] == 1:
                        probabilities = full_outputs.squeeze().cpu().numpy()
                        mlp_pred = (probabilities > 0.5).astype(int)
                        mlp_confidence = np.abs(probabilities - 0.5) + 0.5
                    elif full_outputs.dim() == 1:
                        probabilities = full_outputs.cpu().numpy()
                        mlp_pred = (probabilities > 0.5).astype(int)
                        mlp_confidence = np.abs(probabilities - 0.5) + 0.5
                    else:
                        probabilities = torch.softmax(full_outputs, dim=1).cpu().numpy()
                        mlp_pred = torch.argmax(full_outputs, dim=1).cpu().numpy()
                        mlp_confidence = np.max(probabilities, axis=1)
                
                # FIXED: Convert numpy types for JSON compatibility
                predictions['mlp'] = [int(p) for p in mlp_pred]
                metrics['mlp'] = {
                    'model_type': 'MLP',
                    'confidence_avg': float(np.mean(mlp_confidence)),
                    'confidence_std': float(np.std(mlp_confidence)),
                    'confidence_scores': [float(c) for c in mlp_confidence],
                    'prediction_distribution': safe_convert_for_json({int(k): int(v) for k, v in dict(zip(*np.unique(mlp_pred, return_counts=True))).items()}),
                    'batch_processed': True
                }
                
                st.success(f"✅ MLP predictions completed (confidence: {np.mean(mlp_confidence):.3f})")
                
        except Exception as e:
            st.error(f"❌ MLP prediction failed: {e}")
            st.error(f"Debug info: Model type: {type(models.get('mlp', 'None'))}")
            st.error(f"Embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'Unknown'}")
    
    return predictions, metrics

def create_scientific_visualizations(df, embeddings, stats, predictions, metrics, sentiment_analysis=None):
    """SCIENTIFIC FIX: Create advanced scientific visualizations with class-based analysis"""
    
    try:
        # Dataset Overview
        st.subheader("📊 Scientific Dataset Analysis")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Samples", f"{stats.get('total_reviews', 0):,}")
        with col2:
            st.metric("Avg Text Length", f"{stats.get('avg_length', 0):.0f} chars")
        with col3:
            st.metric("Missing Values", stats.get('null_values', 0))
        with col4:
            st.metric("Duplicates", stats.get('duplicates', 0))
        with col5:
            total_words = stats.get('deep_analysis', {}).get('basic_stats', {}).get('total_words', 0)
            st.metric("Total Words", f"{total_words:,}")
        
        # Sentiment Distribution Analysis
        if predictions:
            st.subheader("🎯 Sentiment Classification Results")
            
            all_predictions = []
            for pred in predictions.values():
                all_predictions.extend(pred)
            
            if all_predictions:
                unique_preds, counts = np.unique(all_predictions, return_counts=True)
                total = len(all_predictions)
                
                label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
                sentiment_df = pd.DataFrame({
                    'Sentiment': [label_map.get(pred, f'Class_{pred}') for pred in unique_preds],
                    'Count': counts,
                    'Percentage': [(count/total)*100 for count in counts]
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_sentiment = px.bar(
                        sentiment_df,
                        x='Sentiment',
                        y='Count',
                        title="Sentiment Classification Distribution",
                        text='Percentage',
                        color='Sentiment',
                        color_discrete_map={'Positive': '#2E8B57', 'Negative': '#DC143C', 'Neutral': '#708090'}
                    )
                    fig_sentiment.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                
                with col2:
                    st.markdown("**Statistical Summary:**")
                    for _, row in sentiment_df.iterrows():
                        st.write(f"• **{row['Sentiment']}**: {row['Count']:,} samples ({row['Percentage']:.1f}%)")
        
        # ENHANCED: Advanced keyword analysis by sentiment
        if predictions and stats.get('advanced_analysis'):
            st.subheader("🔑 Advanced Keyword Analysis by Sentiment")
            
            advanced = stats['advanced_analysis']
            keyword_analysis = advanced.get('keywords_by_sentiment', {})
            
            tab_pos, tab_neg, tab_neu = st.tabs(["😊 Positive Keywords", "😞 Negative Keywords", "😐 Neutral Keywords"])
            
            with tab_pos:
                if keyword_analysis.get('positive', {}).get('keywords'):
                    keywords = keyword_analysis['positive']['keywords'][:15]
                    if keywords:
                        kw_df = pd.DataFrame(keywords)
                        
                        fig_kw = px.bar(
                            kw_df,
                            x='keyword',
                            y='score',
                            title="Top Keywords in Positive Texts",
                            color='score',
                            color_continuous_scale='Greens'
                        )
                        fig_kw.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_kw, use_container_width=True)
                        
                        st.write(f"**Sample Count:** {keyword_analysis['positive']['count']:,}")
            
            with tab_neg:
                if keyword_analysis.get('negative', {}).get('keywords'):
                    keywords = keyword_analysis['negative']['keywords'][:15]
                    if keywords:
                        kw_df = pd.DataFrame(keywords)
                        
                        fig_kw = px.bar(
                            kw_df,
                            x='keyword',
                            y='score',
                            title="Top Keywords in Negative Texts",
                            color='score',
                            color_continuous_scale='Reds'
                        )
                        fig_kw.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_kw, use_container_width=True)
                        
                        st.write(f"**Sample Count:** {keyword_analysis['negative']['count']:,}")
            
            with tab_neu:
                if keyword_analysis.get('neutral', {}).get('keywords'):
                    keywords = keyword_analysis['neutral']['keywords'][:15]
                    if keywords:
                        kw_df = pd.DataFrame(keywords)
                        
                        fig_kw = px.bar(
                            kw_df,
                            x='keyword',
                            y='score',
                            title="Top Keywords in Neutral Texts",
                            color='score',
                            color_continuous_scale='Blues'
                        )
                        fig_kw.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_kw, use_container_width=True)
                        
                        st.write(f"**Sample Count:** {keyword_analysis['neutral']['count']:,}")
        
        # ENHANCED: Advanced phrase analysis
        if predictions and stats.get('advanced_analysis'):
            st.subheader("📝 Most Frequent Phrases by Sentiment")
            
            phrase_analysis = stats['advanced_analysis'].get('phrases_by_sentiment', {})
            
            for sentiment in ['positive', 'negative', 'neutral']:
                if phrase_analysis.get(sentiment, {}).get('phrases'):
                    phrases = phrase_analysis[sentiment]['phrases'][:10]
                    
                    if phrases:
                                                    with st.expander(f"🔍 {sentiment.title()} Phrases ({len(phrases)} found)"):
                            phrase_df = pd.DataFrame(phrases)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_phrase = px.bar(
                                    phrase_df,
                                    x='frequency',
                                    y='phrase',
                                    orientation='h',
                                    title=f"Most Frequent {sentiment.title()} Phrases",
                                    color='percentage',
                                    color_continuous_scale='viridis'
                                )
                                st.plotly_chart(fig_phrase, use_container_width=True)
                            
                            with col2:
                                st.markdown("**Top Phrases:**")
                                for phrase_info in phrases[:5]:
                                    st.write(f"• **\"{phrase_info['phrase']}\"**: {phrase_info['frequency']} times ({phrase_info['percentage']:.1f}%)")
        
        # ENHANCED: Topic analysis
        if predictions and stats.get('advanced_analysis'):
            st.subheader("🎯 Topic Analysis by Sentiment")
            
            topic_analysis = stats['advanced_analysis'].get('topics_by_sentiment', {})
            
            for sentiment in ['positive', 'negative', 'neutral']:
                if topic_analysis.get(sentiment, {}).get('topics'):
                    topics = topic_analysis[sentiment]['topics']
                    themes = topic_analysis[sentiment].get('themes', [])
                    
                    if topics or themes:
                        with st.expander(f"📊 {sentiment.title()} Topics & Themes"):
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if topics:
                                    st.markdown("**Main Topics:**")
                                    topic_df = pd.DataFrame(topics)
                                    
                                    fig_topic = px.bar(
                                        topic_df,
                                        x='topic',
                                        y='relevance',
                                        title=f"{sentiment.title()} Topics by Relevance",
                                        color='mentions',
                                        color_continuous_scale='plasma'
                                    )
                                    fig_topic.update_layout(xaxis_tickangle=-45)
                                    st.plotly_chart(fig_topic, use_container_width=True)
                            
                            with col2:
                                if themes:
                                    st.markdown("**Common Themes:**")
                                    for theme in themes[:6]:
                                        st.write(f"• **{theme['theme']}**: {theme['frequency']} mentions")
                                
                                if topics:
                                    st.markdown("**Topic Details:**")
                                    for topic in topics[:5]:
                                        st.write(f"• **{topic['topic'].title()}**: {topic['mentions']} mentions (relevance: {topic['relevance']:.3f})")
        
        # Confidence Score Analysis
        if metrics:
            st.subheader("📈 Model Confidence Analysis")
            
            for model_name, model_metrics in metrics.items():
                if 'confidence_scores' in model_metrics:
                    confidence_scores = model_metrics['confidence_scores']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_conf = px.histogram(
                            x=confidence_scores,
                            nbins=20,
                            title=f'{model_name.upper()} Confidence Score Distribution',
                            labels={'x': 'Confidence Score', 'y': 'Count'}
                        )
                        fig_conf.add_vline(x=np.mean(confidence_scores), line_dash="dash", 
                                         annotation_text=f"Mean: {np.mean(confidence_scores):.3f}")
                        st.plotly_chart(fig_conf, use_container_width=True)
                    
                    with col2:
                        st.markdown(f"**{model_name.upper()} Confidence Statistics:**")
                        st.write(f"• Mean: {np.mean(confidence_scores):.3f}")
                        st.write(f"• Median: {np.median(confidence_scores):.3f}")
                        st.write(f"• Std Dev: {np.std(confidence_scores):.3f}")
                        st.write(f"• Min: {np.min(confidence_scores):.3f}")
                        st.write(f"• Max: {np.max(confidence_scores):.3f}")
                        
                        high_conf = len([c for c in confidence_scores if c > 0.8])
                        medium_conf = len([c for c in confidence_scores if 0.6 <= c <= 0.8])
                        low_conf = len([c for c in confidence_scores if c < 0.6])
                        
                        st.write(f"• High Confidence (>0.8): {high_conf} ({high_conf/len(confidence_scores)*100:.1f}%)")
                        st.write(f"• Medium Confidence (0.6-0.8): {medium_conf} ({medium_conf/len(confidence_scores)*100:.1f}%)")
                        st.write(f"• Low Confidence (<0.6): {low_conf} ({low_conf/len(confidence_scores)*100:.1f}%)")
        
        # Text Length Distribution Analysis
        text_col = stats.get('text_column')
        if text_col and text_col in df.columns:
            st.subheader("📏 Text Length Distribution Analysis")
            
            try:
                lengths = df[text_col].str.len().fillna(0)
                word_counts = df[text_col].str.split().str.len().fillna(0)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_char = px.histogram(
                        x=lengths,
                        nbins=50,
                        title="Character Length Distribution",
                        labels={'x': 'Characters', 'y': 'Frequency'}
                    )
                    fig_char.add_vline(x=lengths.mean(), line_dash="dash", 
                                     annotation_text=f"Mean: {lengths.mean():.0f}")
                    fig_char.add_vline(x=lengths.median(), line_dash="dot", 
                                     annotation_text=f"Median: {lengths.median():.0f}")
                    st.plotly_chart(fig_char, use_container_width=True)
                
                with col2:
                    fig_words = px.histogram(
                        x=word_counts,
                        nbins=30,
                        title="Word Count Distribution",
                        labels={'x': 'Words', 'y': 'Frequency'}
                    )
                    fig_words.add_vline(x=word_counts.mean(), line_dash="dash",
                                      annotation_text=f"Mean: {word_counts.mean():.1f}")
                    fig_words.add_vline(x=word_counts.median(), line_dash="dot",
                                      annotation_text=f"Median: {word_counts.median():.0f}")
                    st.plotly_chart(fig_words, use_container_width=True)
                
                st.markdown("**Text Length Statistics:**")
                length_stats = pd.DataFrame({
                    'Metric': ['Character Length', 'Word Count'],
                    'Mean': [lengths.mean(), word_counts.mean()],
                    'Median': [lengths.median(), word_counts.median()],
                    'Std Dev': [lengths.std(), word_counts.std()],
                    'Min': [lengths.min(), word_counts.min()],
                    'Max': [lengths.max(), word_counts.max()]
                })
                st.dataframe(length_stats, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not create length analysis: {e}")
        
        # Word Cloud by Sentiment Class
        if WORDCLOUD_AVAILABLE and sentiment_analysis and text_col:
            st.subheader("☁️ Word Clouds by Sentiment Class")
            
            col1, col2, col3 = st.columns(3)
            
            sentiments = ['positive', 'negative', 'neutral']
            colors = ['Greens', 'Reds', 'Blues']
            
            for i, (sentiment, colormap) in enumerate(zip(sentiments, colors)):
                if sentiment in sentiment_analysis and sentiment_analysis[sentiment]['texts']:
                    try:
                        texts = sentiment_analysis[sentiment]['texts']
                        wordcloud_img = create_wordcloud_visualization(
                            texts, 
                            f"{sentiment.title()} Word Cloud"
                        )
                        
                        if wordcloud_img:
                            if i == 0:
                                col1.markdown(f"**{sentiment.title()}**")
                                col1.markdown(
                                    f'<img src="data:image/png;base64,{wordcloud_img}" '
                                    f'style="width: 100%; border-radius: 10px;">',
                                    unsafe_allow_html=True
                                )
                            elif i == 1:
                                col2.markdown(f"**{sentiment.title()}**")
                                col2.markdown(
                                    f'<img src="data:image/png;base64,{wordcloud_img}" '
                                    f'style="width: 100%; border-radius: 10px;">',
                                    unsafe_allow_html=True
                                )
                            else:
                                col3.markdown(f"**{sentiment.title()}**")
                                col3.markdown(
                                    f'<img src="data:image/png;base64,{wordcloud_img}" '
                                    f'style="width: 100%; border-radius: 10px;">',
                                    unsafe_allow_html=True
                                )
                    except Exception as e:
                        st.warning(f"Could not generate {sentiment} wordcloud: {e}")
    
    except Exception as e:
        st.error(f"Error creating scientific visualizations: {e}")

def save_scientific_results(scientific_report, sentiment_analysis, session_dir, filename):
    """
    SCIENTIFIC FIX: Save scientific analysis results to structured files
    """
    try:
        session_path = Path(session_dir)
        saved_files = {}
        
        # Save summary report (TXT)
        summary_content = []
        summary_content.append("SCIENTIFIC SENTIMENT ANALYSIS REPORT")
        summary_content.append("=" * 50)
        summary_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_content.append(f"Source File: {filename}")
        summary_content.append("")
        
        if 'dataset_statistics' in scientific_report:
            stats = scientific_report['dataset_statistics']
            summary_content.append("DATASET STATISTICS:")
            summary_content.append("-" * 20)
            summary_content.append(f"Total Samples: {stats.get('total_samples', 0):,}")
            summary_content.append(f"Memory Usage: {stats.get('memory_usage_mb', 0):.2f} MB")
            summary_content.append(f"Missing Values: {sum(stats.get('missing_values', {}).values())}")
            summary_content.append(f"Duplicate Rows: {stats.get('duplicate_rows', 0)}")
            summary_content.append("")
        
        if 'sentiment_distribution' in scientific_report:
            sent_dist = scientific_report['sentiment_distribution']
            summary_content.append("SENTIMENT DISTRIBUTION:")
            summary_content.append("-" * 22)
            if 'percentages' in sent_dist:
                for sentiment, percentage in sent_dist['percentages'].items():
                    count = sent_dist['counts'].get(sentiment, 0)
                    summary_content.append(f"{sentiment.title()}: {count:,} samples ({percentage:.1f}%)")
            summary_content.append("")
        
        if 'linguistic_analysis' in scientific_report:
            ling = scientific_report['linguistic_analysis']
            summary_content.append("LINGUISTIC ANALYSIS:")
            summary_content.append("-" * 20)
            summary_content.append(f"Total Words: {ling.get('total_words', 0):,}")
            summary_content.append(f"Unique Words: {ling.get('unique_words', 0):,}")
            summary_content.append(f"Vocabulary Richness: {ling.get('vocabulary_richness', 0):.3f}")
            summary_content.append(f"Average Words per Text: {ling.get('avg_words_per_text', 0):.1f}")
            summary_content.append(f"Positive Indicators: {ling.get('positive_indicators', 0)}")
            summary_content.append(f"Negative Indicators: {ling.get('negative_indicators', 0)}")
            summary_content.append(f"Sentiment Ratio: {ling.get('sentiment_ratio', 1):.2f}")
            summary_content.append("")
        
        if 'model_performance' in scientific_report:
            summary_content.append("MODEL PERFORMANCE:")
            summary_content.append("-" * 18)
            for model_name, performance in scientific_report['model_performance'].items():
                summary_content.append(f"{model_name.upper()} ({performance.get('model_type', 'Unknown')}):")
                summary_content.append(f"  Average Confidence: {performance.get('avg_confidence', 0):.3f}")
                summary_content.append(f"  Confidence Std Dev: {performance.get('confidence_std', 0):.3f}")
                summary_content.append(f"  Total Predictions: {performance.get('total_predictions', 0):,}")
                
                pred_dist = performance.get('prediction_distribution', {})
                for class_id, count in pred_dist.items():
                    label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
                    label = label_map.get(class_id, f'Class_{class_id}')
                    summary_content.append(f"    {label}: {count}")
                summary_content.append("")
        
        if 'quality_metrics' in scientific_report:
            quality = scientific_report['quality_metrics']
            summary_content.append("DATA QUALITY METRICS:")
            summary_content.append("-" * 21)
            summary_content.append(f"Overall Quality Score: {quality.get('overall_quality_score', 0):.1%}")
            summary_content.append(f"Data Completeness: {quality.get('data_completeness', 0):.1%}")
            summary_content.append(f"Readability Score: {quality.get('readability_score', 0):.2f}")
            summary_content.append(f"Empty Texts: {quality.get('empty_texts', 0)}")
            summary_content.append(f"Potential Spam: {quality.get('potential_spam', 0)}")
            summary_content.append("")
        
        summary_path = session_path / "report_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_content))
        saved_files['summary_report'] = str(summary_path)
        
        # Save term distribution (CSV)
        if sentiment_analysis:
            term_data = []
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment in sentiment_analysis:
                    stats = sentiment_analysis[sentiment]['stats']
                    top_words = stats.get('top_words', [])
                    
                    for word, freq in top_words:
                        term_data.append({
                            'sentiment': sentiment,
                            'word': word,
                            'frequency': freq,
                            'sentiment_count': stats.get('count', 0),
                            'avg_length': stats.get('avg_length', 0),
                            'avg_words': stats.get('avg_words', 0)
                        })
            
            if term_data:
                term_df = pd.DataFrame(term_data)
                term_path = session_path / "term_distribution.csv"
                term_df.to_csv(term_path, index=False)
                saved_files['term_distribution'] = str(term_path)
        
        # Save graph data (JSON)
        graph_data = {
            'scientific_report': scientific_report,
            'sentiment_analysis': sentiment_analysis,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'source_file': filename,
                'analysis_type': 'scientific_sentiment_analysis'
            }
        }
        
        graph_path = session_path / "graph_data.json"
        with open(graph_path, 'w', encoding='utf-8') as f:
            json.dump(safe_convert_for_json(graph_data), f, indent=2, ensure_ascii=False, default=str)
        saved_files['graph_data'] = str(graph_path)
        
        return saved_files
        
    except Exception as e:
        st.error(f"Error saving scientific results: {e}")
        return {}

def create_enhanced_results_download_package(df, predictions, metrics, stats, scientific_report, sentiment_analysis) -> bytes:
    """
    SCIENTIFIC FIX: Create a comprehensive ZIP package with scientific analysis results
    """
    zip_buffer = io.BytesIO()
    
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            # Enhanced predictions CSV with error handling
            if predictions:
                try:
                    results_df = df.copy()
                    
                    for model_name, pred in predictions.items():
                        try:
                            label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
                            pred_labels = [label_map.get(p, f'class_{p}') for p in pred]
                            results_df[f'{model_name}_prediction'] = pred_labels
                            results_df[f'{model_name}_prediction_numeric'] = pred
                            
                            if model_name in metrics and 'confidence_scores' in metrics[model_name]:
                                confidence_scores = metrics[model_name]['confidence_scores']
                                results_df[f'{model_name}_confidence'] = confidence_scores
                                
                                confidence_levels = ['High' if c > 0.8 else 'Medium' if c > 0.6 else 'Low' for c in confidence_scores]
                                results_df[f'{model_name}_confidence_level'] = confidence_levels
                        except Exception as e:
                            st.warning(f"Could not process {model_name} predictions: {e}")
                    
                    if len(predictions) == 2:
                        try:
                            model_names = list(predictions.keys())
                            pred1, pred2 = list(predictions.values())
                            agreement = ['Agree' if p1 == p2 else 'Disagree' for p1, p2 in zip(pred1, pred2)]
                            results_df['models_agreement'] = agreement
                        except Exception:
                            pass
                    
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    zip_file.writestr("01_comprehensive_predictions.csv", csv_buffer.getvalue())
                except Exception as e:
                    st.warning(f"Could not create predictions CSV: {e}")
            
            # Scientific analysis report
            try:
                scientific_analysis_report = {
                    'analysis_metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'version': 'scientific_v2.0',
                        'analysis_type': 'comprehensive_scientific_sentiment_analysis'
                    },
                    'dataset_summary': {
                        'total_samples': stats.get('total_reviews', 0),
                        'text_column': stats.get('text_column', 'Unknown'),
                        'data_quality_score': stats.get('deep_analysis', {}).get('quality_metrics', {}).get('quality_score', 0),
                        'vocabulary_richness': stats.get('deep_analysis', {}).get('basic_stats', {}).get('vocabulary_richness', 0)
                    },
                    'scientific_analysis': scientific_report or {},
                    'sentiment_class_analysis': sentiment_analysis or {},
                    'model_performance': metrics,
                    'advanced_features': stats.get('advanced_analysis', {})
                }
                
                zip_file.writestr("02_scientific_analysis_report.json", 
                                 json.dumps(safe_convert_for_json(scientific_analysis_report), indent=2, ensure_ascii=False, default=str))
            except Exception as e:
                st.warning(f"Could not create scientific analysis report: {e}")
            
            # Term distribution CSV
            try:
                if sentiment_analysis:
                    term_data = []
                    for sentiment in ['positive', 'negative', 'neutral']:
                        if sentiment in sentiment_analysis and sentiment_analysis[sentiment].get('stats', {}).get('top_words'):
                            stats_data = sentiment_analysis[sentiment]['stats']
                            top_words = stats_data.get('top_words', [])
                            
                            for word, freq in top_words:
                                term_data.append({
                                    'sentiment': sentiment,
                                    'word': word,
                                    'frequency': freq,
                                    'sentiment_sample_count': stats_data.get('count', 0),
                                    'avg_text_length': stats_data.get('avg_length', 0),
                                    'avg_word_count': stats_data.get('avg_words', 0)
                                })
                    
                    if term_data:
                        term_df = pd.DataFrame(term_data)
                        term_csv = term_df.to_csv(index=False)
                        zip_file.writestr("03_term_distribution_by_sentiment.csv", term_csv)
                        
                # ENHANCED: Save advanced keyword/phrase analysis
                if stats.get('advanced_analysis'):
                    advanced = stats['advanced_analysis']
                    
                    # Keyword analysis CSV
                    if 'keywords_by_sentiment' in advanced:
                        keyword_data = []
                        for sentiment, kw_info in advanced['keywords_by_sentiment'].items():
                            for kw in kw_info.get('keywords', []):
                                keyword_data.append({
                                    'sentiment': sentiment,
                                    'keyword': kw['keyword'],
                                    'score': kw['score'],
                                    'sample_count': kw_info['count']
                                })
                        
                        if keyword_data:
                            kw_df = pd.DataFrame(keyword_data)
                            zip_file.writestr("04_keywords_by_sentiment.csv", kw_df.to_csv(index=False))
                    
                    # Phrase analysis CSV
                    if 'phrases_by_sentiment' in advanced:
                        phrase_data = []
                        for sentiment, ph_info in advanced['phrases_by_sentiment'].items():
                            for phrase in ph_info.get('phrases', []):
                                phrase_data.append({
                                    'sentiment': sentiment,
                                    'phrase': phrase['phrase'],
                                    'frequency': phrase['frequency'],
                                    'percentage': phrase['percentage'],
                                    'sample_count': ph_info['count']
                                })
                        
                        if phrase_data:
                            ph_df = pd.DataFrame(phrase_data)
                            zip_file.writestr("05_phrases_by_sentiment.csv", ph_df.to_csv(index=False))
                    
                    # Topic analysis CSV
                    if 'topics_by_sentiment' in advanced:
                        topic_data = []
                        for sentiment, tp_info in advanced['topics_by_sentiment'].items():
                            for topic in tp_info.get('topics', []):
                                topic_data.append({
                                    'sentiment': sentiment,
                                    'topic': topic['topic'],
                                    'score': topic['score'],
                                    'mentions': topic['mentions'],
                                    'relevance': topic['relevance'],
                                    'sample_count': tp_info['count']
                                })
                        
                        if topic_data:
                            tp_df = pd.DataFrame(topic_data)
                            zip_file.writestr("06_topics_by_sentiment.csv", tp_df.to_csv(index=False))
                            
            except Exception as e:
                st.warning(f"Could not create term distribution: {e}")
            
            # Statistical summary report
            try:
                summary_content = f"""
ENHANCED SCIENTIFIC SENTIMENT ANALYSIS REPORT
{'='*55}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Version: Enhanced Scientific v2.0
Dataset: {stats.get('total_reviews', 0):,} texts analyzed

OBJECTIVE STATISTICAL ANALYSIS:
{'='*35}

Dataset Statistics:
- Total Samples: {stats.get('total_reviews', 0):,}
- Text Column: {stats.get('text_column', 'Unknown')}
- Data Quality Score: {stats.get('deep_analysis', {}).get('quality_metrics', {}).get('quality_score', 0):.1%}

Linguistic Analysis:
- Total Words: {stats.get('deep_analysis', {}).get('basic_stats', {}).get('total_words', 0):,}
- Unique Words: {stats.get('deep_analysis', {}).get('basic_stats', {}).get('unique_words', 0):,}
- Vocabulary Richness: {stats.get('deep_analysis', {}).get('basic_stats', {}).get('vocabulary_richness', 0):.3f}
- Average Words per Text: {stats.get('deep_analysis', {}).get('basic_stats', {}).get('avg_words_per_text', 0):.1f}

Sentiment Indicators:
- Positive Indicators: {stats.get('deep_analysis', {}).get('semantic_patterns', {}).get('positive_indicators', 0)}
- Negative Indicators: {stats.get('deep_analysis', {}).get('semantic_patterns', {}).get('negative_indicators', 0)}
- Sentiment Ratio: {stats.get('deep_analysis', {}).get('semantic_patterns', {}).get('sentiment_ratio', 1):.2f}

Model Performance Summary:
{'-'*25}
"""
                
                if predictions and metrics:
                    for model_name, pred in predictions.items():
                        if model_name in metrics:
                            model_metrics = metrics[model_name]
                            pred_dist = dict(zip(*np.unique(pred, return_counts=True)))
                            
                            summary_content += f"""
{model_name.upper()} ({model_metrics.get('model_type', 'Unknown')}):
  - Total Predictions: {len(pred):,}
  - Average Confidence: {model_metrics.get('confidence_avg', 0):.3f}
  - Prediction Distribution:
"""
                            label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
                            for class_id, count in pred_dist.items():
                                label = label_map.get(class_id, f'Class_{class_id}')
                                percentage = (count / len(pred)) * 100
                                summary_content += f"    {label}: {count:,} ({percentage:.1f}%)\n"
                
                # ENHANCED: Add advanced analysis summary
                if stats.get('advanced_analysis'):
                    summary_content += f"""

ADVANCED ANALYSIS SUMMARY:
{'='*27}
"""
                    advanced = stats['advanced_analysis']
                    
                    if 'keywords_by_sentiment' in advanced:
                        summary_content += "\nKeyword Analysis:\n"
                        for sentiment, kw_info in advanced['keywords_by_sentiment'].items():
                            kw_count = len(kw_info.get('keywords', []))
                            summary_content += f"  {sentiment.title()}: {kw_count} keywords identified\n"
                    
                    if 'phrases_by_sentiment' in advanced:
                        summary_content += "\nPhrase Analysis:\n"
                        for sentiment, ph_info in advanced['phrases_by_sentiment'].items():
                            ph_count = len(ph_info.get('phrases', []))
                            summary_content += f"  {sentiment.title()}: {ph_count} frequent phrases found\n"
                    
                    if 'topics_by_sentiment' in advanced:
                        summary_content += "\nTopic Analysis:\n"
                        for sentiment, tp_info in advanced['topics_by_sentiment'].items():
                            tp_count = len(tp_info.get('topics', []))
                            summary_content += f"  {sentiment.title()}: {tp_count} topics identified\n"
                
                summary_content += f"""

METHODOLOGY:
{'='*12}
This analysis uses objective statistical methods for sentiment classification.
All metrics are computed using quantitative measures without subjective interpretation.
Results are reproducible and based on established computational linguistics techniques.

Enhanced features include:
- Advanced keyword extraction using TF-IDF
- N-gram phrase analysis (bigrams/trigrams)
- Topic modeling with domain-specific indicators
- Statistical significance testing

Generated by Enhanced Scientific Sentiment Analysis System v2.0
Objective • Reproducible • Research-Grade • Enhanced
                """
                
                zip_file.writestr("07_enhanced_scientific_summary.txt", summary_content)
            except Exception as e:
                st.warning(f"Could not create summary report: {e}")
            
            # Enhanced README
            try:
                readme_content = f"""
ENHANCED SCIENTIFIC SENTIMENT ANALYSIS PACKAGE
{'='*48}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Type: Enhanced Scientific Statistical Analysis
Dataset: {stats.get('total_reviews', 0):,} texts analyzed

PACKAGE CONTENTS:
{'='*17}

01_comprehensive_predictions.csv
   Complete dataset with model predictions and confidence scores
   Includes: original data, predictions, confidence levels, model agreement
   
02_scientific_analysis_report.json
   Comprehensive technical analysis in structured JSON format
   Includes: statistical metrics, model performance, linguistic analysis
   
03_term_distribution_by_sentiment.csv
   Word frequency analysis organized by sentiment class
   Includes: top words per sentiment with frequency counts
   
04_keywords_by_sentiment.csv (ENHANCED)
   Advanced keyword analysis using TF-IDF scoring
   Includes: keyword relevance scores by sentiment class
   
05_phrases_by_sentiment.csv (ENHANCED)
   Most frequent phrases (bigrams/trigrams) by sentiment
   Includes: phrase frequency and percentage analysis
   
06_topics_by_sentiment.csv (ENHANCED)
   Topic modeling results with domain-specific indicators
   Includes: topic relevance scores and mention counts
   
07_enhanced_scientific_summary.txt
   Comprehensive human-readable analysis summary
   Includes: objective metrics, methodology, enhanced findings
   
README.txt
   This enhanced documentation file

ENHANCED SCIENTIFIC APPROACH:
{'='*31}

This package contains objective statistical analysis results with advanced
NLP features including keyword extraction, phrase analysis, and topic modeling.
All metrics are based on quantitative measurements and reproducible methods.

Enhanced Features:
- TF-IDF keyword extraction
- N-gram phrase analysis
- Domain-specific topic modeling
- Statistical significance testing
- Advanced sentiment classification

Key Features:
- Objective statistical reporting
- Reproducible analysis methods
- Quantitative sentiment classification
- Research-grade documentation
- Enhanced linguistic analysis

USAGE INSTRUCTIONS:
{'='*18}

For Researchers:
- Use JSON files for programmatic analysis
- CSV files are ready for statistical software
- Enhanced features provide deeper insights
- Methodology is documented for replication

For Business Users:
- Read the enhanced summary report (07_) for key findings
- Use prediction CSV (01_) for further analysis
- Keyword/phrase analysis (04_/05_) shows specific patterns
- Topic analysis (06_) reveals thematic insights

For Developers:
- JSON format enables easy integration
- All data structures are documented
- API-ready format for automated processing
- Enhanced features accessible via CSV exports

For Advanced Analysis:
- Combine keyword and topic analysis for deeper insights
- Use phrase analysis to identify common expressions
- Statistical metrics enable rigorous evaluation
- Confidence scores support decision-making

Generated by Enhanced Scientific Sentiment Analysis System v2.0
Research-Grade • Objective • Reproducible • Enhanced
                """
                
                zip_file.writestr("README.txt", readme_content)
            except Exception as e:
                st.warning(f"Could not create README: {e}")
    
    except Exception as e:
        st.error(f"Error creating enhanced scientific ZIP package: {e}")
        return io.BytesIO().getvalue()
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def main():
    """ENHANCED FIXED: Main Streamlit application with comprehensive error handling"""
    
    try:
        # Enhanced Header
        st.markdown('<div class="main-header">🤖 Enhanced Scientific Sentiment Analysis System - Professional Dashboard v2.0 (FULLY FIXED)</div>', 
                    unsafe_allow_html=True)
        
        # FIXED: Enhanced Sidebar with proper error handling
        with st.sidebar:
            st.header("🔧 System Information")
            st.info(f"📁 Project Root: {PROJECT_ROOT}")
            st.info(f"🗃️ Data Directory: {DATA_DIR}")
            st.info(f"📊 Results Directory: {RESULTS_DIR}")
            
            # Session info
            timestamp = create_timestamp_session()
            st.info(f"🕐 Session: {timestamp}")
            
            # Enhanced Quick Actions
            st.header("📋 Quick Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Refresh", help="Refresh models and cache"):
                    st.cache_resource.clear()
                    st.rerun()
            
            with col2:
                if st.button("🗂️ Clear All", help="Clear all cache and session data"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    # Clear session state
                    for key in list(st.session_state.keys()):
                        if key not in ['session_timestamp']:
                            del st.session_state[key]
                    st.success("✅ All cache cleared!")
                    st.rerun()
            
            # FIXED: System status with enhanced error handling
            st.header("🚦 System Status")
            
            # Analysis status
            if 'current_analysis' in st.session_state:
                try:
                    analysis = st.session_state['current_analysis']
                    st.success("✅ Analysis Available")
                    st.caption(f"📄 File: {analysis.get('filename', 'Unknown')}")
                    st.caption(f"📊 Samples: {len(analysis.get('df', []))}")
                    if 'predictions' in analysis:
                        st.caption(f"🤖 Models: {len(analysis['predictions'])}")
                except Exception:
                    st.warning("⚠️ Analysis data corrupted")
            else:
                st.info("ℹ️ No Analysis")
            
            # Pipeline status
            if 'pipeline_results' in st.session_state:
                try:
                    pipeline = st.session_state['pipeline_results']
                    status = pipeline['status']
                    if status == 'completed':
                        st.success("✅ Pipeline Complete")
                        st.caption(f"📈 Success: {pipeline['success_count']}/{pipeline['total_steps']}")
                    elif status == 'partial':
                        st.warning("⚠️ Pipeline Partial")
                        st.caption(f"📈 Success: {pipeline['success_count']}/{pipeline['total_steps']}")
                except Exception:
                    st.warning("⚠️ Pipeline data corrupted")
            else:
                st.info("ℹ️ No Pipeline Run")
            
            # FIXED: Model status with error handling
            st.header("🧠 Model Status")
            try:
                models = load_trained_models()
                for model_name, status in models['status'].items():
                    if status == 'loaded':
                        st.success(f"✅ {model_name.upper()}")
                    elif status == 'error':
                        st.error(f"❌ {model_name.upper()}")
                    else:
                        st.info(f"⚠️ {model_name.upper()}")
            except Exception:
                st.warning("⚠️ Model status unknown")
        
        # FIXED: Load system resources with enhanced error handling
        with st.spinner("🔄 Loading enhanced system resources..."):
            embedding_model = load_embedding_model()
            models = load_trained_models()
            main_df, main_dataset_path = load_main_dataset()
        
        # FIXED: Create tabs with proper structure
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dataset Overview", 
            "🧠 Models & Predictions", 
            "📈 Graphics & Statistics",
            "🔍 Deep Text Analysis",
            "📂 CSV Analysis", 
            "📥 Download Results"
        ])
        
        # Tab 1: FIXED Dataset Overview
        with tab1:
            st.header("📊 Comprehensive Dataset Overview")
            
            if main_df is not None:
                st.success(f"✅ Main dataset loaded from: {main_dataset_path}")
                
                # FIXED: Full dataset display with search and filter capabilities
                st.subheader("👀 Complete Data Preview")
                
                # Add search functionality
                search_term = st.text_input("🔍 Search in data:", placeholder="Enter search term...")
                
                display_df = main_df
                if search_term:
                    # FIXED: Search across all text columns with error handling
                    text_columns = ['review', 'text', 'content', 'comment', 'message']
                    mask = False
                    try:
                        for col in text_columns:
                            if col in main_df.columns:
                                mask |= main_df[col].astype(str).str.contains(search_term, case=False, na=False)
                        if mask.any():
                            display_df = main_df[mask]
                            st.info(f"🔍 Found {len(display_df)} matching records")
                        else:
                            st.warning("⚠️ No matches found")
                    except Exception as e:
                        st.warning(f"Search error: {e}")
                
                # FIXED: Show complete dataset with pagination
                st.dataframe(
                    display_df, 
                    use_container_width=True,
                    height=400
                )
                
                # FIXED: Enhanced dataset statistics with error handling
                try:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Samples", f"{len(main_df):,}")
                    with col2:
                        st.metric("Columns", len(main_df.columns))
                    with col3:
                        st.metric("Missing Values", int(main_df.isnull().sum().sum()))
                    with col4:
                        st.metric("Duplicates", int(main_df.duplicated().sum()))
                    with col5:
                        memory_usage = main_df.memory_usage(deep=True).sum() / (1024**2)
                        st.metric("Memory Usage", f"{memory_usage:.1f} MB")
                except Exception as e:
                    st.warning(f"Could not display statistics: {e}")
                
                # FIXED: Column information with comprehensive error handling
                st.subheader("🏗️ Enhanced Dataset Structure")
                try:
                    col_info = []
                    for col in main_df.columns:
                        try:
                            sample_values = main_df[col].dropna().head(3).tolist()
                            sample_str = ", ".join([str(val)[:30] for val in sample_values])
                            
                            col_info.append({
                                'Column': col,
                                'Type': str(main_df[col].dtype),
                                'Non-Null': f"{main_df[col].notna().sum():,}",
                                'Null': f"{main_df[col].isna().sum():,}",
                                'Unique': f"{main_df[col].nunique():,}",
                                'Sample Values': sample_str[:50] + "..." if len(sample_str) > 50 else sample_str
                            })
                        except Exception:
                            col_info.append({
                                'Column': col,
                                'Type': 'Error',
                                'Non-Null': 'N/A',
                                'Null': 'N/A',
                                'Unique': 'N/A',
                                'Sample Values': 'Error reading samples'
                            })
                    
                    col_df = pd.DataFrame(col_info)
                    st.dataframe(col_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not display column information: {e}")
                
                # FIXED: Auto-generate advanced insights with error handling
                st.subheader("🧠 Intelligent Dataset Insights")
                
                # Check for text column to do advanced analysis
                text_columns = ['review', 'text', 'content', 'comment', 'message', 'description']
                text_col = None
                for col in text_columns:
                    if col in main_df.columns:
                        text_col = col
                        break
                
                if text_col:
                    try:
                        with st.spinner("🔄 Generating comprehensive insights..."):
                            # Run enhanced deep analysis
                            deep_analysis = enhanced_deep_text_analysis(main_df, text_col)
                            main_insights = generate_narrative_insights(main_df, deep_analysis=deep_analysis)
                            
                            # Display insights in enhanced format
                            for i, insight in enumerate(main_insights):
                                if i % 2 == 0:
                                    st.markdown(f'<div class="insights-box">{insight}</div>', 
                                               unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="narrative-comment">{insight}</div>', 
                                               unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Could not generate insights: {e}")
                        st.info("Basic dataset loaded successfully, but advanced insights failed.")
                else:
                    st.info("No text column found for advanced analysis.")
            else:
                st.info("ℹ️ No main dataset found. Please upload a CSV file in the 'CSV Analysis' tab.")
                
                st.markdown("""
                ### 📋 Expected Dataset Format:
                
                Your CSV should contain:
                - **Text column**: 'review', 'text', 'content', 'comment', 'message', or 'description'  
                - **Label column** (optional): 'sentiment', 'label', 'class', or 'target'
                
                Example:
                ```
                review,sentiment
                "This movie is absolutely fantastic!",positive
                "I didn't like it at all",negative
                "It was okay, nothing special",neutral
                ```
                """)
        
        # Tab 2: ENHANCED Models & Predictions with comprehensive error handling
        with tab2:
            st.header("🧠 Enhanced Models & Predictions")
            
            # Display model status
            st.subheader("🔧 Model Status Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🤖 MLP Neural Network")
                try:
                    if models['status']['mlp'] == 'loaded':
                        st.markdown('<div class="status-success">✅ MLP Model: Loaded & Ready</div>', 
                                   unsafe_allow_html=True)
                        # Try to get model info
                        try:
                            mlp_model = models['mlp']
                            total_params = sum(p.numel() for p in mlp_model.parameters())
                            st.info(f"📊 Parameters: {total_params:,}")
                            st.info(f"🔧 Device: {next(mlp_model.parameters()).device}")
                        except Exception:
                            pass
                    elif models['status']['mlp'] == 'error':
                        st.markdown('<div class="status-error">❌ MLP Model: Error Loading</div>', 
                                   unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-warning">⚠️ MLP Model: Not Found</div>', 
                                   unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error checking MLP status: {e}")
            
            with col2:
                st.markdown("### ⚡ SVM Classifier")
                try:
                    if models['status']['svm'] == 'loaded':
                        st.markdown('<div class="status-success">✅ SVM Model: Loaded & Ready</div>', 
                                   unsafe_allow_html=True)
                        # Try to get SVM info
                        try:
                            svm_package = models['svm']
                            if isinstance(svm_package, dict):
                                svm_model = svm_package.get('model')
                                if svm_model:
                                    st.info(f"🔧 Kernel: {getattr(svm_model, 'kernel', 'Unknown')}")
                                    support_vectors = getattr(svm_model, 'n_support_', 'Unknown')
                                    st.info(f"📊 Support Vectors: {support_vectors}")
                        except Exception:
                            pass
                    elif models['status']['svm'] == 'error':
                        st.markdown('<div class="status-error">❌ SVM Model: Error Loading</div>', 
                                   unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-warning">⚠️ SVM Model: Not Found</div>', 
                                   unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error checking SVM status: {e}")
            
            # Enhanced Model training section
            st.subheader("🏋️ Enhanced Model Training")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Train Both Models (Enhanced)", type="primary"):
                    if main_df is not None:
                        st.info("🔄 Starting enhanced model training pipeline...")
                        
                        try:
                            # Save main dataset temporarily for training
                            temp_csv_path = get_session_results_dir() / "temp_dataset.csv"
                            main_df.to_csv(temp_csv_path, index=False)
                            
                            # Run complete pipeline
                            success = run_complete_pipeline(str(temp_csv_path))
                            
                            if success:
                                st.success("🎉 Enhanced model training completed successfully!")
                                if st.button("🔄 Load New Models"):
                                    st.cache_resource.clear()
                                    st.rerun()
                            else:
                                st.error("❌ Model training failed. Check logs for details.")
                        except Exception as e:
                            st.error(f"Training error: {e}")
                    else:
                        st.warning("⚠️ Please load a dataset first!")
            
            with col2:
                if st.button("⚡ Quick Training (Fast Mode)", type="secondary"):
                    st.info("🚀 Fast training mode reduces epochs for quick testing")
            
            # ENHANCED: Prediction section with scientific analysis
            if any(model is not None for model in models.values()):
                st.subheader("🔮 Enhanced Prediction Interface")
                
                # Enhanced text input with examples
                st.markdown("### 💬 Test Your Text with Complete Scientific Analysis")
                
                # Provide example texts
                example_texts = [
                    "This movie is absolutely fantastic! I loved every minute of it.",
                    "Terrible service, worst experience ever. Would not recommend.",
                    "The product is okay, nothing special but does the job.",
                    "Amazing quality and fast delivery. Highly recommended!",
                    "Disappointed with the purchase. Poor quality for the price."
                ]
                
                selected_example = st.selectbox(
                    "🎯 Choose an example or write your own:",
                    [""] + example_texts,
                    index=0
                )
                
                test_text = st.text_area(
                    "Enter text to analyze:",
                    value=selected_example,
                    placeholder="Type your text here... (e.g., 'This product exceeded my expectations!')",
                    height=100
                )
                
                if st.button("🔬 Scientific Text Analysis") and test_text.strip():
                    if not ENHANCED_PROCESSOR_AVAILABLE:
                        st.error("❌ EnhancedFileAnalysisProcessor not available!")
                        return
                    
                    if embedding_model:
                        try:
                            with st.spinner("🔄 Analyzing text..."):
                                # Generate embedding
                                text_embedding = embedding_model.encode([test_text])
                                
                                # Make predictions with enhanced handling
                                predictions, metrics = predict_sentiment_enhanced(
                                    [test_text], text_embedding, models
                                )
                                
                                if predictions:
                                    # Get first prediction and confidence
                                    first_model = list(predictions.keys())[0]
                                    prediction = predictions[first_model][0]
                                    confidence = metrics[first_model]['confidence_scores'][0]
                                    
                                    # Display prediction results
                                    st.subheader("🎯 Prediction Results")
                                    
                                    # Create comparison table
                                    results_data = []
                                    for model_name, pred in predictions.items():
                                        label_map = {0: '😞 Negative', 1: '😊 Positive', 2: '😐 Neutral'}
                                        result = label_map.get(pred[0], f'Class {pred[0]}')
                                        
                                        confidence_score = metrics[model_name]['confidence_avg']
                                        model_type = metrics[model_name]['model_type']
                                        
                                        results_data.append({
                                            'Model': model_name.upper(),
                                            'Type': model_type,
                                            'Prediction': result,
                                            'Confidence': f"{confidence_score:.3f}",
                                            'Level': 'High' if confidence_score > 0.8 else 'Medium' if confidence_score > 0.6 else 'Low'
                                        })
                                    
                                    results_df = pd.DataFrame(results_data)
                                    st.dataframe(results_df, use_container_width=True)
                                    
                                    # 🔧 FIXED: Call processor.analyze_text() directly
                                    st.subheader("📊 Scientific Text Analysis")
                                    
                                    try:
                                        processor = EnhancedFileAnalysisProcessor()
                                        analysis = processor.analyze_text(test_text)
                                        
                                        if analysis:
                                            # 📊 Top Words (bar chart)
                                            if 'top_words' in analysis:
                                                st.markdown("### 📊 Top Words")
                                                top_words = analysis['top_words'][:10]
                                                if top_words:
                                                    words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                                                    fig_words = px.bar(
                                                        words_df,
                                                        x='Word',
                                                        y='Frequency',
                                                        title="Top 10 Words"
                                                    )
                                                    st.plotly_chart(fig_words, use_container_width=True)
                                            
                                            # 📝 Frequent Phrases (dataframe)
                                            if 'phrases' in analysis:
                                                st.markdown("### 📝 Frequent Phrases")
                                                phrases = analysis['phrases'][:10]
                                                if phrases:
                                                    phrases_df = pd.DataFrame(phrases, columns=['Phrase', 'Frequency'])
                                                    st.dataframe(phrases_df, use_container_width=True)
                                            
                                            # 🧠 Topics (table or bar chart)
                                            if 'topics' in analysis:
                                                st.markdown("### 🧠 Topics")
                                                topics = analysis['topics'][:10]
                                                if topics:
                                                    topics_df = pd.DataFrame(topics, columns=['Topic', 'Score'])
                                                    st.dataframe(topics_df, use_container_width=True)
                                            
                                            # 📥 Download buttons
                                            st.markdown("### 📥 Download Results")
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                # Download CSV
                                                csv_data = pd.DataFrame({
                                                    'text': [test_text],
                                                    'prediction': [label_map.get(prediction, 'Unknown')],
                                                    'confidence': [confidence]
                                                })
                                                
                                                st.download_button(
                                                    label="📄 Download CSV",
                                                    data=csv_data.to_csv(index=False),
                                                    file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                    mime="text/csv"
                                                )
                                            
                                            with col2:
                                                # Download JSON
                                                json_data = {
                                                    'text': test_text,
                                                    'prediction': label_map.get(prediction, 'Unknown'),
                                                    'confidence': float(confidence),
                                                    'analysis': safe_convert_for_json(analysis)
                                                }
                                                
                                                st.download_button(
                                                    label="📊 Download JSON",
                                                    data=json.dumps(json_data, indent=2),
                                                    file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                                    mime="application/json"
                                                )
                                        else:
                                            st.warning("⚠️ Analysis returned no results")
                                            
                                    except Exception as e:
                                        st.error(f"❌ Analysis error: {e}")
                                    
                                    # Model agreement analysis
                                    if len(predictions) > 1:
                                        preds_list = [pred[0] for pred in predictions.values()]
                                        if len(set(preds_list)) == 1:
                                            st.success("🤝 **Model Agreement**: All models agree!")
                                        else:
                                            st.warning("⚡ **Model Disagreement**: Different predictions")
                                
                                else:
                                    st.warning("⚠️ No predictions generated. Check model status.")
                        except Exception as e:
                            st.error(f"❌ Analysis error: {e}")
                    else:
                        st.error("❌ Embedding model not available!")
            else:
                st.info("ℹ️ No trained models available. Train models first or upload a CSV for analysis.")
        
        # Tab 3: ENHANCED Graphics & Statistics with scientific visualizations
        with tab3:
            st.header("📈 Enhanced Scientific Graphics & Statistical Analysis")
            
            if main_df is not None:
                try:
                    # SCIENTIFIC FIX: Use scientific visualizations for main dataset
                    st.subheader("📊 Scientific Dataset Visualizations")
                    
                    # Find text column
                    text_columns = ['review', 'text', 'content', 'comment', 'message', 'description']
                    text_col = None
                    for col in text_columns:
                        if col in main_df.columns:
                            text_col = col
                            break
                    
                    if text_col:
                        # Perform deep analysis for scientific visualization
                        with st.spinner("🔄 Performing enhanced scientific analysis..."):
                            deep_analysis = enhanced_deep_text_analysis(main_df, text_col)
                            
                            # Create mock stats structure for compatibility
                            stats = {
                                'total_reviews': len(main_df),
                                'text_column': text_col,
                                'deep_analysis': deep_analysis
                            }
                            
                            # Generate mock embeddings for visualization (not used in scientific viz)
                            mock_embeddings = np.zeros((len(main_df), 384))
                            
                            # Check if we have predictions from current analysis
                            predictions = {}
                            metrics = {}
                            sentiment_analysis = None
                            
                            if 'current_analysis' in st.session_state:
                                analysis = st.session_state['current_analysis']
                                predictions = analysis.get('predictions', {})
                                metrics = analysis.get('metrics', {})
                                sentiment_analysis = analysis.get('sentiment_analysis', None)
                            
                            # SCIENTIFIC FIX: Use scientific visualization function
                            create_scientific_visualizations(
                                main_df, mock_embeddings, stats, predictions, metrics, sentiment_analysis
                            )
                    
                    # Label distribution if available
                    label_columns = ['sentiment', 'label', 'class', 'target']
                    for col in label_columns:
                        if col in main_df.columns:
                            st.subheader(f"🏷️ {col.title()} Distribution Analysis")
                            
                            label_dist = main_df[col].value_counts()
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                fig_pie = px.pie(
                                    values=label_dist.values,
                                    names=label_dist.index,
                                    title=f"{col.title()} Distribution (Pie Chart)"
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                            with col2:
                                fig_bar = px.bar(
                                    x=label_dist.index,
                                    y=label_dist.values,
                                    title=f"{col.title()} Distribution (Bar Chart)",
                                    labels={'x': col.title(), 'y': 'Count'},
                                    color=label_dist.values,
                                    color_continuous_scale='plasma'
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)
                            
                            with col3:
                                # Enhanced donut chart
                                fig_donut = go.Figure(data=[go.Pie(
                                    labels=label_dist.index,
                                    values=label_dist.values,
                                    hole=.4
                                )])
                                fig_donut.update_layout(title_text=f"{col.title()} Distribution (Donut)")
                                st.plotly_chart(fig_donut, use_container_width=True)
                            
                            # Statistics for the label distribution
                            st.markdown(f"**📊 {col.title()} Statistics:**")
                            total = label_dist.sum()
                            for idx, (label, count) in enumerate(label_dist.items()):
                                percentage = (count / total) * 100
                                st.write(f"• **{label}**: {count:,} samples ({percentage:.1f}%)")
                            
                            break
                    
                    # Correlation heatmap for numeric columns
                    numeric_cols = main_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        st.subheader("🔗 Correlation Analysis")
                        
                        corr_matrix = main_df[numeric_cols].corr()
                        
                        fig_corr = px.imshow(
                            corr_matrix,
                            title="Correlation Matrix - Numeric Features",
                            color_continuous_scale="RdBu",
                            aspect="auto",
                            text_auto=True,
                            zmin=-1,
                            zmax=1
                        )
                        fig_corr.update_layout(height=500)
                        st.plotly_chart(fig_corr, use_container_width=True)
                except Exception as e:
                    st.error(f"Error in graphics and statistics: {e}")
            else:
                st.info("ℹ️ No dataset loaded. Upload a CSV to see enhanced scientific visualizations.")
                
                # Enhanced guidance for graphics tab
                st.markdown("""
                ### 📈 Available Enhanced Scientific Visualizations:
                
                Once you upload a dataset, you'll see:
                
                **📊 Enhanced Scientific Analysis**
                - Sentiment distribution with statistical breakdown
                - Advanced keyword analysis by sentiment class
                - Phrase frequency analysis (bigrams/trigrams)
                - Topic modeling with domain-specific indicators
                - Confidence score distributions
                - Text length statistical analysis
                
                **🔬 Advanced Analytics**  
                - TF-IDF keyword extraction by sentiment
                - N-gram phrase analysis
                - Domain-specific topic modeling
                - Separate word clouds for each sentiment class
                - Model performance metrics visualization
                - Statistical correlation analysis
                
                **📋 Enhanced Scientific Reports**
                - Neutral statistical summaries
                - Advanced term frequency distributions
                - Keyword relevance scoring
                - Phrase occurrence analysis
                - Topic relevance measurements
                - Data quality assessments
                - Exportable enhanced analysis results
                
                ### 🎯 Enhanced Scientific Features:
                - Advanced TF-IDF keyword extraction
                - Statistical phrase analysis
                - Domain-specific topic modeling
                - Objective metric reporting
                - Reproducible analysis methods
                - Export-ready visualizations
                """)
        
        # Tab 4: ENHANCED Deep Scientific Text Analysis with complete features
        with tab4:
            st.header("🔍 Enhanced Deep Scientific Text Analysis")
            st.markdown("*Advanced scientific semantic pattern recognition, keyword extraction, and topic modeling*")
            
            # Check if we have analysis data
            analysis_available = False
            analysis_data = None
            
            try:
                if 'current_analysis' in st.session_state:
                    analysis_data = st.session_state['current_analysis']
                    analysis_available = True
                elif main_df is not None:
                    # Use main dataset
                    text_columns = ['review', 'text', 'content', 'comment', 'message', 'description']
                    text_col = None
                    for col in text_columns:
                        if col in main_df.columns:
                            text_col = col
                            break
                    
                    if text_col:
                        analysis_available = True
                
                if analysis_available:
                    if analysis_data:
                        # Use existing analysis
                        df = analysis_data['df']
                        text_col = analysis_data['stats']['text_column']
                        deep_analysis = analysis_data['stats'].get('deep_analysis', {})
                        scientific_report = analysis_data.get('scientific_report', {})
                        sentiment_analysis = analysis_data.get('sentiment_analysis', {})
                        advanced_analysis = analysis_data['stats'].get('advanced_analysis', {})
                    else:
                        # Perform analysis on main dataset
                        df = main_df
                        with st.spinner("🔄 Performing comprehensive enhanced scientific analysis..."):
                            deep_analysis = enhanced_deep_text_analysis(df, text_col)
                            scientific_report = generate_scientific_report(df, {}, {}, deep_analysis)
                            sentiment_analysis = {}
                            advanced_analysis = {}
                    
                    # ENHANCED: Display comprehensive scientific metrics
                    if scientific_report:
                        # Scientific Intelligence Overview
                        st.subheader("🧠 Enhanced Scientific Analysis Overview")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        if 'linguistic_analysis' in scientific_report:
                            ling = scientific_report['linguistic_analysis']
                            
                            with col1:
                                vocab_richness = ling.get('vocabulary_richness', 0)
                                st.metric("🎨 Vocabulary Richness", f"{vocab_richness:.3f}")
                            
                            with col2:
                                sentiment_ratio = ling.get('sentiment_ratio', 1)
                                st.metric("💭 Sentiment Ratio", f"{sentiment_ratio:.2f}")
                            
                            with col3:
                                quality_score = scientific_report.get('quality_metrics', {}).get('overall_quality_score', 0)
                                st.metric("✅ Quality Score", f"{quality_score:.1%}")
                            
                            with col4:
                                emotion_diversity = len([e for e in ling.get('emotion_distribution', {}).values() if e > 0])
                                st.metric("🎭 Emotion Types", emotion_diversity)
                            
                            with col5:
                                total_words = ling.get('total_words', 0)
                                st.metric("📚 Total Words", f"{total_words:,}")
                        
                        # ENHANCED: Display scientific report with advanced features
                        st.subheader("📊 Enhanced Scientific Analysis Report")
                        
                        # Linguistic Analysis
                        if 'linguistic_analysis' in scientific_report:
                            ling = scientific_report['linguistic_analysis']
                            
                            with st.expander("📝 Comprehensive Linguistic Analysis", expanded=True):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Basic Statistics:**")
                                    st.write(f"• Total Words: {ling.get('total_words', 0):,}")
                                    st.write(f"• Unique Words: {ling.get('unique_words', 0):,}")
                                    st.write(f"• Vocabulary Richness: {ling.get('vocabulary_richness', 0):.3f}")
                                    st.write(f"• Average Words per Text: {ling.get('avg_words_per_text', 0):.1f}")
                                    st.write(f"• Average Characters per Text: {ling.get('avg_chars_per_text', 0):.1f}")
                                
                                with col2:
                                    st.markdown("**Sentiment Indicators:**")
                                    st.write(f"• Positive Indicators: {ling.get('positive_indicators', 0)}")
                                    st.write(f"• Negative Indicators: {ling.get('negative_indicators', 0)}")
                                    st.write(f"• Sentiment Ratio: {ling.get('sentiment_ratio', 1):.2f}")
                                    
                                    # Emotion distribution
                                    emotion_dist = ling.get('emotion_distribution', {})
                                    if emotion_dist:
                                        st.markdown("**Emotion Distribution:**")
                                        for emotion, count in emotion_dist.items():
                                            if count > 0:
                                                st.write(f"• {emotion.title()}: {count}")
                        
                        # Quality Metrics
                        if 'quality_metrics' in scientific_report:
                            quality = scientific_report['quality_metrics']
                            
                            with st.expander("✅ Data Quality Assessment"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Quality Metrics:**")
                                    st.write(f"• Overall Quality Score: {quality.get('overall_quality_score', 0):.1%}")
                                    st.write(f"• Data Completeness: {quality.get('data_completeness', 0):.1%}")
                                    st.write(f"• Readability Score: {quality.get('readability_score', 0):.2f}")
                                
                                with col2:
                                    st.markdown("**Data Issues:**")
                                    st.write(f"• Empty Texts: {quality.get('empty_texts', 0)}")
                                    st.write(f"• Potential Spam: {quality.get('potential_spam', 0)}")
                        
                        # Term Frequency
                        if 'term_frequency' in scientific_report:
                            terms = scientific_report['term_frequency']
                            
                            with st.expander("📊 Term Frequency Analysis"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Statistics:**")
                                    st.write(f"• Rare Words Count: {terms.get('rare_words_count', 0)}")
                                    st.write(f"• Average Word Length: {terms.get('avg_word_length', 0):.1f} characters")
                                
                                with col2:
                                    most_common = terms.get('most_common_terms', [])
                                    if most_common:
                                        st.markdown("**Most Common Terms:**")
                                        for word, freq in most_common[:8]:
                                            st.write(f"• **{word}**: {freq} occurrences")
                        
                        # ENHANCED: Wordcloud in Deep Analysis
                        if WORDCLOUD_AVAILABLE and text_col in df.columns:
                            st.subheader("☁️ Enhanced Word Cloud Analysis")
                            try:
                                text_data = df[text_col].fillna('').astype(str).tolist()
                                wordcloud_img = create_wordcloud_visualization(
                                    text_data, 
                                    "Enhanced Scientific Text Analysis Word Cloud"
                                )
                                
                                if wordcloud_img:
                                    st.markdown(
                                        f'<div class="wordcloud-container">'
                                        f'<img src="data:image/png;base64,{wordcloud_img}" '
                                        f'style="width: 100%; max-width: 800px; border-radius: 10px;">'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.info("Could not generate word cloud for this dataset")
                            except Exception as e:
                                st.warning(f"Word cloud generation failed: {e}")
                        elif not WORDCLOUD_AVAILABLE:
                            st.info("💡 Install wordcloud for visualization: `pip install wordcloud`")
                    else:
                        st.warning("⚠️ Could not perform enhanced scientific analysis. Please check the text data format.")
                else:
                    st.info("ℹ️ No text data available for enhanced scientific analysis.")
                    
                    st.markdown("""
                    ### 🔬 About Enhanced Scientific Text Analysis:
                    
                    This enhanced scientific analysis provides:
                    
                    **📊 Advanced Statistical Metrics**
                    - TF-IDF keyword extraction by sentiment class
                    - N-gram phrase frequency analysis
                    - Domain-specific topic modeling
                    - Vocabulary richness assessment
                    - Sentiment ratio calculations
                    - Text quality scoring
                    - Linguistic feature measurements
                    
                    **🔍 Enhanced Objective Analysis**
                    - Neutral statistical reporting
                    - Reproducible measurements
                    - Quantitative assessments
                    - Evidence-based insights
                    - Advanced NLP feature extraction
                    
                    **📈 Enhanced Scientific Visualizations**
                    - Keyword relevance scoring charts
                    - Phrase frequency distributions
                    - Topic relevance measurements
                    - Statistical distributions
                    - Correlation studies
                    - Objective comparisons
                    
                    **📋 Research-Grade Output**
                    - Exportable enhanced data files
                    - Advanced statistical summaries
                    - Methodology documentation
                    - Replicable results
                    - Keyword/phrase/topic matrices
                    
                    ### 📊 To Enable Enhanced Scientific Analysis:
                    1. Upload a CSV file with text data
                    2. Run "Deep Analysis" in the CSV Analysis tab
                    3. Return here to see comprehensive enhanced scientific insights
                    4. Download complete analysis packages with advanced features
                    """)
            except Exception as e:
                st.error(f"Error in Enhanced Scientific Text Analysis: {e}")
        
        # Tab 5: ENHANCED CSV Analysis with Deep and Full Pipeline functionality
        with tab5:
            st.header("📂 Enhanced CSV File Analysis")
            
            # Enhanced file upload section
            st.subheader("📁 Upload & Analyze CSV File")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Choose a CSV file for comprehensive enhanced analysis",
                    type=['csv'],
                    help="CSV must contain a text column ('review', 'text', 'content', 'comment', 'message', or 'description')"
                )
            
            with col2:
                if uploaded_file:
                    # File info
                    try:
                        file_size = len(uploaded_file.getvalue()) / (1024**2)  # MB
                        st.info(f"📄 **File:** {uploaded_file.name}")
                        st.info(f"📊 **Size:** {file_size:.2f} MB")
                    except Exception:
                        st.info(f"📄 **File:** {uploaded_file.name}")
            
            if uploaded_file is not None:
                st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
                
                # Enhanced analysis options
                st.subheader("🔬 Enhanced Analysis Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    quick_analysis = st.button(
                        "⚡ Quick Analysis", 
                        type="primary",
                        help="Fast analysis with predictions and insights"
                    )
                with col2:
                    # 🔧 FIXED: Deep Analysis button with proper functionality
                    deep_analysis_btn = st.button(
                        "🔍 Deep Analysis (ENHANCED)", 
                        type="secondary",
                        help="🔧 FIXED: Comprehensive analysis with EnhancedFileAnalysisProcessor integration"
                    )
                with col3:
                    # 🔧 FIXED: Full Pipeline button with proper functionality
                    full_pipeline = st.button(
                        "🚀 Full Pipeline (ENHANCED)", 
                        help="🔧 FIXED: Complete pipeline with run_dataset_analysis integration"
                    )
                
                # FIXED: Analysis execution with proper Deep Analysis and Full Pipeline
                if quick_analysis:
                    analysis_type = "quick"
                    
                    try:
                        with st.spinner(f"🔄 Performing {analysis_type} analysis..."):
                            df, embeddings, stats = analyze_single_csv(uploaded_file, embedding_model)
                        
                        if df is not None and embeddings is not None:
                            st.success(f"✅ {analysis_type.title()} analysis completed!")
                            
                            # Get session directory for saving
                            session_dir = get_session_results_dir()
                            
                            # Store comprehensive analysis in session state
                            st.session_state['current_analysis'] = {
                                'df': df,
                                'embeddings': embeddings,
                                'stats': stats,
                                'filename': uploaded_file.name,
                                'timestamp': datetime.now(),
                                'session_dir': str(session_dir),
                                'analysis_type': analysis_type
                            }
                            
                            # Enhanced results display
                            st.subheader("📊 Quick Analysis Results Overview")
                            
                            # Key metrics
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                st.metric("📄 Total Samples", f"{len(df):,}")
                            with col2:
                                text_col = stats['text_column']
                                avg_len = stats['avg_length']
                                st.metric("📝 Avg Length", f"{avg_len:.0f} chars")
                            with col3:
                                total_words = stats.get('deep_analysis', {}).get('basic_stats', {}).get('total_words', 0)
                                st.metric("📚 Total Words", f"{total_words:,}")
                            with col4:
                                quality_score = stats.get('deep_analysis', {}).get('quality_metrics', {}).get('quality_score', 0)
                                st.metric("✅ Quality Score", f"{quality_score:.1%}")
                            with col5:
                                vocab_richness = stats.get('deep_analysis', {}).get('basic_stats', {}).get('vocabulary_richness', 0)
                                st.metric("🎨 Vocab Richness", f"{vocab_richness:.3f}")
                            
                            # Data preview
                            st.subheader("👀 Data Preview")
                            preview_rows = st.slider("Rows to display:", 5, 50, 20)
                            st.dataframe(df.head(preview_rows), use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Quick analysis error: {e}")
                
                elif deep_analysis_btn:
                    # 🔧 FIXED: Deep Analysis with direct processor call
                    if not ENHANCED_PROCESSOR_AVAILABLE:
                        st.error("❌ EnhancedFileAnalysisProcessor not available!")
                        return
                    
                    try:
                        # Save uploaded file temporarily
                        session_dir = get_session_results_dir()
                        temp_csv_path = session_dir / f"uploaded_{uploaded_file.name}"
                        
                        with open(temp_csv_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        st.info(f"💾 File saved to: {temp_csv_path}")
                        
                        with st.spinner("🔄 Running Deep Analysis..."):
                            # 🔧 FIXED: Direct call to processor.analyze_csv
                            processor = EnhancedFileAnalysisProcessor()
                            analysis_results = processor.analyze_csv(str(temp_csv_path), mode="deep")
                        
                        if analysis_results:
                            st.success("✅ Deep Analysis completed!")
                            
                            # Store results in session
                            st.session_state['current_analysis'] = {
                                'df': pd.read_csv(temp_csv_path),
                                'filename': uploaded_file.name,
                                'timestamp': datetime.now(),
                                'analysis_type': 'deep',
                                'analysis_results': analysis_results
                            }
                            
                            # Display results
                            st.subheader("📊 Deep Analysis Results")
                            
                            # 📊 Top Words per classe (if available)
                            if 'top_words_by_class' in analysis_results:
                                st.markdown("### 📊 Top Words by Class")
                                top_words = analysis_results['top_words_by_class']
                                
                                for sentiment_class, words in top_words.items():
                                    if words:
                                        st.markdown(f"**{sentiment_class.title()}:**")
                                        words_df = pd.DataFrame(words[:10], columns=['Word', 'Score'])
                                        
                                        fig = px.bar(
                                            words_df,
                                            x='Word',
                                            y='Score',
                                            title=f"Top Words - {sentiment_class.title()}"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                            
                            # 📝 Frasi più ricorrenti (if available)
                            if 'frequent_phrases' in analysis_results:
                                st.markdown("### 📝 Most Frequent Phrases")
                                phrases = analysis_results['frequent_phrases'][:20]
                                if phrases:
                                    phrases_df = pd.DataFrame(phrases, columns=['Phrase', 'Frequency'])
                                    st.dataframe(phrases_df, use_container_width=True)
                            
                            # 🧠 Argomenti (if available)
                            if 'topics' in analysis_results:
                                st.markdown("### 🧠 Topics")
                                topics = analysis_results['topics'][:15]
                                if topics:
                                    topics_df = pd.DataFrame(topics, columns=['Topic', 'Relevance'])
                                    
                                    fig_topics = px.bar(
                                        topics_df,
                                        x='Topic',
                                        y='Relevance',
                                        title="Topics by Relevance"
                                    )
                                    st.plotly_chart(fig_topics, use_container_width=True)
                            
                            # 📤 Download buttons
                            st.markdown("### 📥 Download Results")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Download CSV
                                results_csv = pd.DataFrame(safe_convert_for_json(analysis_results))
                                st.download_button(
                                    label="📄 Download CSV",
                                    data=results_csv.to_csv(index=False),
                                    file_name=f"deep_analysis_{uploaded_file.name}",
                                    mime="text/csv"
                                )
                            
                            with col2:
                                # Download JSON
                                st.download_button(
                                    label="📊 Download JSON",
                                    data=json.dumps(safe_convert_for_json(analysis_results), indent=2),
                                    file_name=f"deep_analysis_{uploaded_file.name.replace('.csv', '.json')}",
                                    mime="application/json"
                                )
                        else:
                            st.error("❌ Deep analysis failed or returned no results")
                            
                    except Exception as e:
                        st.error(f"❌ Deep analysis error: {e}")
                
                elif full_pipeline:
                    # 🔧 FIXED: Full Pipeline with direct function call
                    if not PIPELINE_RUNNER_AVAILABLE:
                        st.error("❌ run_dataset_analysis not available!")
                        return
                    
                    try:
                        # Save uploaded file temporarily
                        session_dir = get_session_results_dir()
                        temp_csv_path = session_dir / f"uploaded_{uploaded_file.name}"
                        
                        with open(temp_csv_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        st.info(f"💾 File saved to: {temp_csv_path}")
                        
                        with st.spinner("🔄 Running Full Pipeline..."):
                            # 🔧 FIXED: Direct call to run_dataset_analysis
                            success = run_dataset_analysis(
                                str(temp_csv_path),
                                steps=["preprocess", "embed", "train", "predict", "report"],
                                output_dir=str(session_dir)
                            )
                        
                        if success:
                            st.success("✅ Full Pipeline completed!")
                            
                            # Store results in session
                            st.session_state['pipeline_results'] = {
                                'session_dir': str(session_dir),
                                'timestamp': datetime.now(),
                                'status': 'completed',
                                'pipeline_function_used': 'run_dataset_analysis'
                            }
                            
                            # Display results
                            st.subheader("📊 Pipeline Results")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Status", "✅ Complete")
                            with col2:
                                st.metric("Models", "Trained")
                            with col3:
                                st.metric("Reports", "Generated")
                            with col4:
                                st.metric("Session", session_dir.name)
                            
                            st.markdown(f"""
                            **📊 Pipeline Results:**
                            - Data preprocessing: ✅ Complete
                            - Embedding generation: ✅ Complete
                            - Model training: ✅ Complete
                            - Predictions: ✅ Complete
                            - Reports: ✅ Generated
                            - Output directory: `{session_dir}`
                            
                            **⚡ Next Steps:**
                            1. Refresh to use newly trained models
                            2. Check 'Models & Predictions' tab
                            3. Download results from 'Download Results' tab
                            """)
                        else:
                            st.error("❌ Full pipeline failed")
                            
                    except Exception as e:
                        st.error(f"❌ Pipeline error: {e}")
            else:
                # Enhanced guidance when no file uploaded
                st.markdown("""
                ### 📂 Enhanced CSV Analysis Features:
                
                Upload a CSV file to access these powerful enhanced analysis capabilities:
                
                **⚡ Quick Analysis**
                - Fast text processing and embedding generation
                - Immediate sentiment predictions (if models available)
                - Basic statistical analysis and insights
                - Quick visualization generation
                
                **🔍 Deep Analysis (ENHANCED)**  
                - 🔧 FIXED: EnhancedFileAnalysisProcessor integration
                - Comprehensive semantic pattern recognition
                - Advanced TF-IDF keyword extraction by sentiment
                - N-gram phrase analysis (bigrams/trigrams)
                - Domain-specific topic modeling
                - Enhanced linguistic feature extraction
                - Detailed quality assessment and recommendations
                - Advanced visualization suite with wordclouds
                
                **🚀 Full Pipeline (ENHANCED)**
                - 🔧 FIXED: run_dataset_analysis function integration
                - Complete end-to-end processing workflow
                - Automated model training (MLP + SVM)
                - Enhanced keyword/phrase/topic extraction
                - Advanced evaluation and reporting
                - Production-ready model generation with enhanced features
                
                ### 📋 Supported File Formats:
                
                **Required Columns:**
                - Text data: `review`, `text`, `content`, `comment`, `message`, `description`
                - Labels (optional): `sentiment`, `label`, `class`, `target`
                
                **Example CSV Structure:**
                ```
                text,sentiment
                "Great product, highly recommend!",positive
                "Poor quality, waste of money",negative
                "Average product, does the job",neutral
                ```
                
                ### 🚀 Enhanced Features:
                - **Advanced Keyword Extraction**: TF-IDF scoring by sentiment class
                - **Phrase Analysis**: Bigram and trigram frequency analysis
                - **Topic Modeling**: Domain-specific topic identification
                - **Enhanced Visualizations**: Interactive charts and word clouds
                - **Scientific Reporting**: Objective statistical analysis
                - **Export-Ready Results**: Comprehensive downloadable packages
                - **EnhancedFileAnalysisProcessor Integration**: Advanced file processing
                - **Pipeline Integration**: run_dataset_analysis function support
                """)
        
        # Tab 6: ENHANCED Download Results with scientific files
        with tab6:
            st.header("📥 Enhanced Scientific Results Download Center")
            st.markdown("*Comprehensive enhanced scientific analysis packages with advanced NLP features*")
            
            # Check for analysis results
            if 'current_analysis' in st.session_state:
                try:
                    analysis = st.session_state['current_analysis']
                    
                    # Enhanced results summary
                    st.success(f"✅ Enhanced scientific analysis results available for: **{analysis['filename']}**")
                    
                    # Results overview
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.info(f"🕐 **Generated:** {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.info(f"🔬 **Analysis Type:** {analysis.get('analysis_type', 'Standard').title()}")
                        st.info(f"📁 **Session Directory:** {Path(analysis.get('session_dir', '')).name}")
                        
                        # Enhanced features indicator
                        if analysis.get('analysis_type') == 'deep':
                            st.info("🚀 **Enhanced Features:** Keyword/Phrase/Topic Analysis + EnhancedFileAnalysisProcessor")
                    
                    with col2:
                        # Quick stats
                        df_size = len(analysis.get('df', []))
                        models_used = len(analysis.get('predictions', {}))
                        saved_files = len(analysis.get('saved_files', {}))
                        
                        st.metric("📊 Samples", f"{df_size:,}")
                        st.metric("🤖 Models", models_used)
                        st.metric("📄 Files", saved_files)
                        
                        # Enhanced features indicator
                        if analysis.get('stats', {}).get('advanced_analysis'):
                            st.metric("🚀 Enhanced", "✅")
                    
                    # ENHANCED: Download options with advanced features
                    st.subheader("📦 Enhanced Scientific Download Options")
                    
                    tab_individual, tab_complete, tab_scientific, tab_advanced = st.tabs([
                        "📄 Individual Files", 
                        "📦 Complete Package", 
                        "🔬 Scientific Reports",
                        "🚀 Advanced Features"
                    ])
                    
                    with tab_individual:
                        st.markdown("### 📄 Individual File Downloads")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Predictions CSV
                            if 'predictions' in analysis:
                                try:
                                    df_results = analysis['df'].copy()
                                    
                                    for model_name, pred in analysis['predictions'].items():
                                        label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
                                        pred_labels = [label_map.get(p, f'class_{p}') for p in pred]
                                        df_results[f'{model_name}_prediction'] = pred_labels
                                        df_results[f'{model_name}_prediction_numeric'] = pred
                                        
                                        # Add confidence if available
                                        if model_name in analysis.get('metrics', {}) and 'confidence_scores' in analysis['metrics'][model_name]:
                                            df_results[f'{model_name}_confidence'] = analysis['metrics'][model_name]['confidence_scores']
                                            
                                            # Add confidence levels
                                            confidence_scores = analysis['metrics'][model_name]['confidence_scores']
                                            confidence_levels = ['High' if c > 0.8 else 'Medium' if c > 0.6 else 'Low' for c in confidence_scores]
                                            df_results[f'{model_name}_confidence_level'] = confidence_levels
                                    
                                    csv_data = df_results.to_csv(index=False)
                                    
                                    st.download_button(
                                        label="📄 Download Enhanced Predictions CSV",
                                        data=csv_data,
                                        file_name=f"enhanced_predictions_{analysis['filename']}",
                                        mime="text/csv",
                                        help="Complete dataset with model predictions and confidence scores"
                                    )
                                except Exception as e:
                                    st.error(f"Error preparing predictions CSV: {e}")
                        
                        with col2:
                            # ENHANCED: Scientific summary report
                            if 'scientific_report' in analysis:
                                try:
                                    scientific_report = analysis['scientific_report']
                                    
                                    # Create enhanced summary text
                                    summary_lines = []
                                    summary_lines.append("ENHANCED SCIENTIFIC ANALYSIS SUMMARY")
                                    summary_lines.append("=" * 45)
                                    summary_lines.append(f"Generated: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                                    summary_lines.append(f"File: {analysis['filename']}")
                                    summary_lines.append(f"Analysis Type: {analysis.get('analysis_type', 'Standard').title()}")
                                    summary_lines.append("")
                                    
                                    # Add enhanced processor integration info
                                    if analysis.get('stats', {}).get('advanced_analysis', {}).get('enhanced_processor_results'):
                                        summary_lines.append("ENHANCED FILE PROCESSOR INTEGRATION:")
                                        summary_lines.append("-" * 35)
                                        summary_lines.append("✅ EnhancedFileAnalysisProcessor successfully integrated")
                                        summary_lines.append("")
                                    
                                    if 'sentiment_distribution' in scientific_report:
                                        sent_dist = scientific_report['sentiment_distribution']
                                        summary_lines.append("SENTIMENT DISTRIBUTION:")
                                        summary_lines.append("-" * 22)
                                        if 'percentages' in sent_dist:
                                            for sentiment, percentage in sent_dist['percentages'].items():
                                                count = sent_dist['counts'].get(sentiment, 0)
                                                summary_lines.append(f"{sentiment.title()}: {count:,} ({percentage:.1f}%)")
                                        summary_lines.append("")
                                    
                                    if 'linguistic_analysis' in scientific_report:
                                        ling = scientific_report['linguistic_analysis']
                                        summary_lines.append("ENHANCED LINGUISTIC ANALYSIS:")
                                        summary_lines.append("-" * 29)
                                        summary_lines.append(f"Total Words: {ling.get('total_words', 0):,}")
                                        summary_lines.append(f"Unique Words: {ling.get('unique_words', 0):,}")
                                        summary_lines.append(f"Vocabulary Richness: {ling.get('vocabulary_richness', 0):.3f}")
                                        summary_lines.append("")
                                    
                                    # Add advanced features summary
                                    if analysis.get('stats', {}).get('advanced_analysis'):
                                        summary_lines.append("ADVANCED FEATURES:")
                                        summary_lines.append("-" * 18)
                                        summary_lines.append("✅ Keyword Extraction (TF-IDF)")
                                        summary_lines.append("✅ Phrase Analysis (N-grams)")
                                        summary_lines.append("✅ Topic Modeling")
                                        if ENHANCED_PROCESSOR_AVAILABLE:
                                            summary_lines.append("✅ EnhancedFileAnalysisProcessor Integration")
                                        summary_lines.append("")
                                    
                                    summary_text = '\n'.join(summary_lines)
                                    
                                    st.download_button(
                                        label="📊 Download Enhanced Scientific Report",
                                        data=summary_text,
                                        file_name=f"enhanced_scientific_report_{analysis['filename'].replace('.csv', '.txt')}",
                                        mime="text/plain",
                                        help="Enhanced scientific statistical analysis summary with processor integration"
                                    )
                                except Exception as e:
                                    st.error(f"Error preparing enhanced scientific report: {e}")
                    
                    with tab_complete:
                        st.markdown("### 📦 Complete Enhanced Analysis Package")
                        
                        # Package generation
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if st.button("🎁 Generate Enhanced Scientific Results Package", type="primary"):
                                try:
                                    with st.spinner("🔄 Creating comprehensive enhanced scientific package..."):
                                        # ENHANCED: Create enhanced scientific package
                                        zip_data = create_enhanced_results_download_package(
                                            analysis['df'],
                                            analysis.get('predictions', {}),
                                            analysis.get('metrics', {}),
                                            analysis['stats'],
                                            analysis.get('scientific_report', {}),
                                            analysis.get('sentiment_analysis', {})
                                        )
                                        
                                        # Success message
                                        st.success("✅ Enhanced scientific package generated successfully!")
                                        
                                        # Download button
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        filename = f"enhanced_scientific_analysis_{analysis['filename'].replace('.csv', '')}_{timestamp}.zip"
                                        
                                        st.download_button(
                                            label="📥 Download Enhanced Scientific Package (ZIP)",
                                            data=zip_data,
                                            file_name=filename,
                                            mime="application/zip",
                                            help="Complete enhanced scientific analysis package with advanced NLP features and processor integration"
                                        )
                                        
                                except Exception as e:
                                    st.error(f"❌ Error creating enhanced scientific package: {e}")
                        
                        with col2:
                            # Enhanced package preview
                            st.markdown("""
                            **📋 Enhanced Package Contents:**
                            - 📄 Predictions CSV with classifications
                            - 📊 Statistical analysis JSON
                            - 📈 Term frequency distributions
                            - 🔑 Keyword analysis by sentiment
                            - 📝 Phrase frequency analysis
                            - 🎯 Topic modeling results
                            - 🚀 EnhancedFileAnalysisProcessor results
                            - 📋 Enhanced methodology README
                            """)
                    
                    with tab_scientific:
                        st.markdown("### 🔬 Scientific Analysis Files")
                        
                        # ENHANCED: Show scientific files if available
                        if 'saved_files' in analysis and analysis['saved_files']:
                            st.success("✅ Enhanced scientific analysis files are available!")
                            
                            saved_files = analysis['saved_files']
                            
                            for file_type, file_path in saved_files.items():
                                if Path(file_path).exists():
                                    file_size = Path(file_path).stat().st_size / 1024  # KB
                                    file_name = Path(file_path).name
                                    
                                    col1, col2, col3 = st.columns([2, 1, 1])
                                    
                                    with col1:
                                        st.write(f"📄 **{file_type.replace('_', ' ').title()}**")
                                        st.caption(f"File: {file_name}")
                                    
                                    with col2:
                                        st.write(f"{file_size:.1f} KB")
                                    
                                    with col3:
                                        # Individual file download
                                        try:
                                            with open(file_path, 'r', encoding='utf-8') as f:
                                                file_content = f.read()
                                            
                                            st.download_button(
                                                label="⬇️ Download",
                                                data=file_content,
                                                file_name=file_name,
                                                mime="text/plain" if file_name.endswith('.txt') else "application/json" if file_name.endswith('.json') else "text/csv",
                                                key=f"download_{file_type}"
                                            )
                                        except Exception:
                                            st.write("❌ Error")
                        else:
                            st.info("ℹ️ No scientific files available for this analysis.")
                    
                    with tab_advanced:
                        st.markdown("### 🚀 Advanced NLP Features")
                        
                        # ENHANCED: Show advanced features if available
                        if analysis.get('stats', {}).get('advanced_analysis'):
                            advanced = analysis['stats']['advanced_analysis']
                            
                            st.success("✅ Advanced NLP features are available for download!")
                            
                            # Enhanced processor results
                            if advanced.get('enhanced_processor_results'):
                                st.markdown("#### 🚀 EnhancedFileAnalysisProcessor Results")
                                st.success("✅ EnhancedFileAnalysisProcessor integration completed successfully!")
                                
                                if st.button("📥 Download Enhanced Processor Results JSON"):
                                    try:
                                        processor_data = json.dumps(
                                            safe_convert_for_json(advanced['enhanced_processor_results']), 
                                            indent=2, 
                                            ensure_ascii=False
                                        )
                                        
                                        st.download_button(
                                            label="🚀 Download Processor Results",
                                            data=processor_data,
                                            file_name=f"enhanced_processor_results_{analysis['filename'].replace('.csv', '.json')}",
                                            mime="application/json",
                                            key="download_processor_results"
                                        )
                                        st.success("✅ Enhanced processor results ready for download!")
                                    except Exception as e:
                                        st.error(f"Error preparing processor download: {e}")
                            
                            # Keyword analysis
                            if 'keywords_by_sentiment' in advanced:
                                st.markdown("#### 🔑 Keyword Analysis by Sentiment")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                for i, sentiment in enumerate(['positive', 'negative', 'neutral']):
                                    if sentiment in advanced['keywords_by_sentiment']:
                                        kw_data = advanced['keywords_by_sentiment'][sentiment]
                                        kw_count = len(kw_data.get('keywords', []))
                                        
                                        if i == 0:
                                            col1.metric(f"😊 {sentiment.title()}", f"{kw_count} keywords")
                                        elif i == 1:
                                            col2.metric(f"😞 {sentiment.title()}", f"{kw_count} keywords")
                                        else:
                                            col3.metric(f"😐 {sentiment.title()}", f"{kw_count} keywords")
                                
                                # Download keyword analysis
                                if st.button("📥 Download Keyword Analysis CSV"):
                                    try:
                                        keyword_data = []
                                        for sentiment, kw_info in advanced['keywords_by_sentiment'].items():
                                            for kw in kw_info.get('keywords', []):
                                                keyword_data.append({
                                                    'sentiment': sentiment,
                                                    'keyword': kw['keyword'],
                                                    'score': kw['score'],
                                                    'sample_count': kw_info['count']
                                                })
                                        
                                        if keyword_data:
                                            kw_df = pd.DataFrame(keyword_data)
                                            csv_data = kw_df.to_csv(index=False)
                                            
                                            st.download_button(
                                                label="🔑 Download Keywords CSV",
                                                data=csv_data,
                                                file_name=f"keywords_analysis_{analysis['filename']}",
                                                mime="text/csv",
                                                key="download_keywords"
                                            )
                                            st.success("✅ Keyword analysis ready for download!")
                                    except Exception as e:
                                        st.error(f"Error preparing keyword download: {e}")
                            
                            # Phrase analysis
                            if 'phrases_by_sentiment' in advanced:
                                st.markdown("#### 📝 Phrase Analysis")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                for i, sentiment in enumerate(['positive', 'negative', 'neutral']):
                                    if sentiment in advanced['phrases_by_sentiment']:
                                        ph_data = advanced['phrases_by_sentiment'][sentiment]
                                        ph_count = len(ph_data.get('phrases', []))
                                        
                                        if i == 0:
                                            col1.metric(f"😊 {sentiment.title()}", f"{ph_count} phrases")
                                        elif i == 1:
                                            col2.metric(f"😞 {sentiment.title()}", f"{ph_count} phrases")
                                        else:
                                            col3.metric(f"😐 {sentiment.title()}", f"{ph_count} phrases")
                                
                                # Download phrase analysis
                                if st.button("📥 Download Phrase Analysis CSV"):
                                    try:
                                        phrase_data = []
                                        for sentiment, ph_info in advanced['phrases_by_sentiment'].items():
                                            for phrase in ph_info.get('phrases', []):
                                                phrase_data.append({
                                                    'sentiment': sentiment,
                                                    'phrase': phrase['phrase'],
                                                    'frequency': phrase['frequency'],
                                                    'percentage': phrase['percentage'],
                                                    'sample_count': ph_info['count']
                                                })
                                        
                                        if phrase_data:
                                            ph_df = pd.DataFrame(phrase_data)
                                            csv_data = ph_df.to_csv(index=False)
                                            
                                            st.download_button(
                                                label="📝 Download Phrases CSV",
                                                data=csv_data,
                                                file_name=f"phrases_analysis_{analysis['filename']}",
                                                mime="text/csv",
                                                key="download_phrases"
                                            )
                                            st.success("✅ Phrase analysis ready for download!")
                                    except Exception as e:
                                        st.error(f"Error preparing phrase download: {e}")
                            
                            # Topic analysis
                            if 'topics_by_sentiment' in advanced:
                                st.markdown("#### 🎯 Topic Analysis")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                for i, sentiment in enumerate(['positive', 'negative', 'neutral']):
                                    if sentiment in advanced['topics_by_sentiment']:
                                        tp_data = advanced['topics_by_sentiment'][sentiment]
                                        tp_count = len(tp_data.get('topics', []))
                                        
                                        if i == 0:
                                            col1.metric(f"😊 {sentiment.title()}", f"{tp_count} topics")
                                        elif i == 1:
                                            col2.metric(f"😞 {sentiment.title()}", f"{tp_count} topics")
                                        else:
                                            col3.metric(f"😐 {sentiment.title()}", f"{tp_count} topics")
                                
                                # Download topic analysis
                                if st.button("📥 Download Topic Analysis CSV"):
                                    try:
                                        topic_data = []
                                        for sentiment, tp_info in advanced['topics_by_sentiment'].items():
                                            for topic in tp_info.get('topics', []):
                                                topic_data.append({
                                                    'sentiment': sentiment,
                                                    'topic': topic['topic'],
                                                    'score': topic['score'],
                                                    'mentions': topic['mentions'],
                                                    'relevance': topic['relevance'],
                                                    'sample_count': tp_info['count']
                                                })
                                        
                                        if topic_data:
                                            tp_df = pd.DataFrame(topic_data)
                                            csv_data = tp_df.to_csv(index=False)
                                            
                                            st.download_button(
                                                label="🎯 Download Topics CSV",
                                                data=csv_data,
                                                file_name=f"topics_analysis_{analysis['filename']}",
                                                mime="text/csv",
                                                key="download_topics"
                                            )
                                            st.success("✅ Topic analysis ready for download!")
                                    except Exception as e:
                                        st.error(f"Error preparing topic download: {e}")
                        else:
                            st.info("ℹ️ No advanced NLP features available. Run 'Deep Analysis' to generate advanced features.")
                            
                            st.markdown("""
                            **🚀 Advanced Features Include:**
                            - **Keyword Extraction**: TF-IDF scoring by sentiment class
                            - **Phrase Analysis**: Bigram and trigram frequency analysis
                            - **Topic Modeling**: Domain-specific topic identification
                            - **EnhancedFileAnalysisProcessor**: Advanced file processing integration
                            
                            To generate these features:
                            1. Go to 'CSV Analysis' tab
                            2. Upload your CSV file
                            3. Click 'Deep Analysis (ENHANCED)'
                            4. Return here for advanced downloads
                            """)
                    
                    # Enhanced package contents preview
                    st.subheader("📋 Enhanced Scientific Analysis Features")
                    st.markdown("""
                    **🔬 Enhanced Scientific Approach:**
                    - Objective statistical measurements with advanced NLP
                    - TF-IDF keyword extraction by sentiment class
                    - N-gram phrase frequency analysis
                    - Domain-specific topic modeling
                    - EnhancedFileAnalysisProcessor integration
                    - Reproducible analysis methods
                    - Quantitative metrics and distributions
                    
                    **📊 Advanced Statistical Reports:**
                    - Enhanced sentiment distribution percentages
                    - Keyword relevance scoring by sentiment
                    - Phrase occurrence statistics
                    - Topic relevance measurements
                    - Vocabulary richness calculations
                    - Text quality scoring metrics
                    - Model performance statistics
                    - Enhanced processor integration results
                    
                    **📈 Research-Grade Output:**
                    - Enhanced CSV files for further analysis
                    - JSON data for programmatic access
                    - Statistical summaries in TXT format
                    - Keyword/phrase/topic matrices
                    - Comprehensive methodology documentation
                    - EnhancedFileAnalysisProcessor results
                    
                    **🎯 Enhanced Use Cases:**
                    - Academic research and publications
                    - Business intelligence and reporting
                    - Advanced sentiment analysis projects
                    - Keyword and topic trend analysis
                    - Model validation and comparison
                    - Advanced statistical analysis and visualization
                    - Processor-enhanced text analysis workflows
                    """)
                except Exception as e:
                    st.error(f"Error in enhanced download results section: {e}")
            else:
                # Enhanced guidance when no results available
                st.info("ℹ️ No enhanced scientific analysis results available for download.")
                
                st.markdown("""
                ### 📥 How to Generate Enhanced Scientific Results:
                
                **🔄 Enhanced Scientific Analysis Path:**
                1. 📂 Go to **'CSV Analysis'** tab
                2. 📁 Upload your CSV file
                3. 🔍 Click **'Deep Analysis (ENHANCED)'** for advanced features with EnhancedFileAnalysisProcessor
                4. 🔄 Return here to download enhanced scientific results
                
                **🚀 Complete Enhanced Scientific Pipeline:**
                1. 📂 Upload CSV in **'CSV Analysis'** tab
                2. 🚀 Run **'Full Pipeline (ENHANCED)'** for complete analysis with run_dataset_analysis
                3. 📊 Get comprehensive analysis with advanced NLP features
                4. 📥 Download all enhanced scientific results and reports
                
                ### 🔬 What You'll Get (Enhanced Scientific):
                
                **📊 Enhanced Statistical Data Files:**
                - Predictions CSV with confidence scores
                - Keyword analysis by sentiment (TF-IDF scoring)
                - Phrase frequency distributions (N-grams)
                - Topic modeling results with relevance scores
                - Statistical analysis summaries
                - EnhancedFileAnalysisProcessor results
                
                **📈 Advanced Scientific Reports:**
                - Objective statistical measurements
                - Advanced NLP feature extraction
                - Keyword/phrase/topic analysis
                - Reproducible analysis methodology
                - Quantitative performance metrics
                - Research-grade documentation
                - Enhanced processor integration
                
                **🎯 Enhanced Research Features:**
                - TF-IDF keyword extraction by sentiment
                - Bigram and trigram phrase analysis
                - Domain-specific topic modeling
                - Statistical significance testing
                - Advanced sentiment classification
                - Neutral statistical reporting
                - Evidence-based conclusions
                - Exportable data for further analysis
                - Enhanced methodology transparency
                - EnhancedFileAnalysisProcessor integration
                - run_dataset_analysis pipeline support
                
                ### 💡 Enhanced Scientific Approach:
                - **Advanced NLP Analysis**: Keyword, phrase, and topic extraction
                - **Processor Integration**: EnhancedFileAnalysisProcessor support
                - **Pipeline Integration**: run_dataset_analysis function support
                - **Objective Analysis**: Statistical measurements without subjective interpretation
                - **Reproducible Methods**: Documented methodology for result replication
                - **Quantitative Focus**: Numerical metrics and statistical distributions
                - **Research Grade**: Suitable for academic and professional research
                - **Enhanced Features**: TF-IDF, N-grams, topic modeling, processor integration
                
                Start your enhanced scientific analysis in the **CSV Analysis** tab! 🔬🚀
                """)
        
        # Enhanced Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            🤖 <strong>Scientific Sentiment Analysis System v2.0 (SIMPLIFIED & FIXED)</strong> | 
            🔬 Direct AI-Powered Text Analysis | 
            📊 Research-Grade Statistical Platform<br>
            <small>Built with Streamlit • Powered by PyTorch & scikit-learn • Direct processor integration</small><br>
            <small>✨ <em>Simplified Interface • Direct Function Calls • Production Ready</em> ✨</small><br>
            <small>🔧 <em>FIXED: Direct EnhancedFileAnalysisProcessor calls • Direct run_dataset_analysis support</em> 🔧</small>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Critical application error: {e}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()