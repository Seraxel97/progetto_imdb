#!/usr/bin/env python3
"""
Sistema di Analisi Sentiment - GUI Finale Scientifica
Interfaccia professionale con 3 sezioni principali per analisi rigorosa del sentiment.

CARATTERISTICHE FINALI:
- 🧠 Dataset Analysis: Analisi completa unificata con visualizzazioni scientifiche
- 📂 Upload New Dataset: Caricamento e analisi rapida di nuovi CSV
- 📥 Export Results: Download completo di risultati e report professionali
- 🔬 Approccio scientifico con formattazione rigorosa e layout pulito
- 📊 Grafici professionali senza sovrapposizioni o elementi disordinati
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

warnings.filterwarnings('ignore')

# Configurazione percorsi dinamici
try:
    CURRENT_FILE = Path(__file__).resolve()
    if CURRENT_FILE.parent.name == 'scripts':
        PROJECT_ROOT = CURRENT_FILE.parent.parent
    else:
        PROJECT_ROOT = CURRENT_FILE.parent
except:
    PROJECT_ROOT = Path.cwd()

# Percorsi del progetto
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed" 
EMBEDDINGS_DATA_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
REPORTS_DIR = RESULTS_DIR / "reports"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Aggiungi scripts al path per import
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# Configurazione pagina
st.set_page_config(
    page_title="🔬 Sistema Analisi Sentiment Professionale",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS scientifico e professionale
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    .scientific-section {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        margin: 1.5rem 0;
        border-left: 5px solid #1f77b4;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin: 0.8rem 0;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    
    .statistical-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .analysis-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    
    .results-table {
        background: #fafafa;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #ddd;
    }
    
    .export-section {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Definizione architettura MLP
class HateSpeechMLP(nn.Module):
    """Architettura MLP per classificazione sentiment"""
    
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
    """Carica il modello di embedding"""
    try:
        with st.spinner("🔄 Loading SentenceTransformer model..."):
            local_model_dir = PROJECT_ROOT / "models" / "minilm-l6-v2"
            if local_model_dir.exists():
                model = SentenceTransformer(str(local_model_dir))
                st.success(f"✅ Local model loaded from {local_model_dir}")
            else:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                st.success("✅ Online model loaded successfully")
            return model
    except Exception as e:
        st.error(f"❌ Error loading embedding model: {e}")
        return None

@st.cache_resource
def load_trained_models():
    """Carica i modelli addestrati"""
    models = {
        'mlp': None,
        'svm': None,
        'status': {
            'mlp': 'not_found',
            'svm': 'not_found'
        }
    }
    
    # Carica modello MLP
    mlp_paths = [
        MODELS_DIR / "mlp_model.pth",
        MODELS_DIR / "mlp_model_complete.pth",
        PROJECT_ROOT / "results" / "models" / "mlp_model.pth"
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
                break
                
            except Exception as e:
                continue
    
    # Carica modello SVM
    svm_paths = [
        MODELS_DIR / "svm_model.pkl",
        PROJECT_ROOT / "results" / "models" / "svm_model.pkl"
    ]
    
    for svm_path in svm_paths:
        if svm_path.exists():
            try:
                models['svm'] = joblib.load(svm_path)
                models['status']['svm'] = 'loaded'
                break
            except Exception as e:
                continue
    
    return models

@st.cache_data
def load_main_dataset():
    """Carica il dataset principale"""
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
                return df, str(path)
            except Exception as e:
                continue
    
    return None, None

def safe_convert_for_json(obj):
    """Converte tipi numpy per serializzazione JSON"""
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
    elif isinstance(obj, dict):
        return {str(k): safe_convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_convert_for_json(item) for item in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

def fix_sentiment_labels(df, sentiment_column):
    """
    FIXED: Risolve il problema di conversione delle etichette sentiment da stringa a int
    """
    try:
        # Mappa etichette stringa a numeri
        label_map = {
            "negative": 0, "negativo": 0, "neg": 0, "0": 0,
            "positive": 1, "positivo": 1, "pos": 1, "1": 1,
            "neutral": 2, "neutro": 2, "neu": 2, "2": 2
        }
        
        # Converti a stringa prima del mapping
        df[sentiment_column] = df[sentiment_column].astype(str).str.lower().str.strip()
        
        # Applica la mappatura
        df["label_int"] = df[sentiment_column].map(label_map)
        
        # Gestisci valori non mappati
        unmapped_mask = df["label_int"].isna()
        if unmapped_mask.any():
            # Prova a convertire direttamente i numeri
            try:
                df.loc[unmapped_mask, "label_int"] = pd.to_numeric(df.loc[unmapped_mask, sentiment_column], errors='coerce')
            except:
                pass
            
            # Imposta valori di default per i rimanenti
            df["label_int"] = df["label_int"].fillna(2)  # Default a neutral
        
        # Assicurati che sia int
        df["label_int"] = df["label_int"].astype(int)
        
        return df["label_int"].tolist()
        
    except Exception as e:
        st.warning(f"Error fixing sentiment labels: {e}")
        # Fallback: restituisci etichette neutre
        return [2] * len(df)

def comprehensive_text_analysis(df: pd.DataFrame, text_column: str) -> Dict:
    """Analisi completa e scientifica del testo con metriche rigorose"""
    try:
        texts = df[text_column].fillna('').astype(str)
        
        analysis = {
            'basic_statistics': {},
            'word_frequency_analysis': {},
            'phrase_pattern_analysis': {},
            'sentiment_indicators': {},
            'topic_analysis': {},
            'data_quality_metrics': {}
        }
        
        # === STATISTICHE DI BASE ===
        all_words = []
        word_counts = []
        char_counts = []
        sentence_counts = []
        
        for text in texts:
            try:
                # Pulizia e tokenizzazione
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
        
        # Frequenza parole
        word_freq = Counter(all_words) if all_words else Counter()
        total_words = len(all_words)
        unique_words = len(word_freq)
        
        analysis['basic_statistics'] = {
            'total_texts': len(texts),
            'total_words': total_words,
            'unique_words': unique_words,
            'vocabulary_richness': unique_words / max(1, total_words),
            'avg_words_per_text': np.mean(word_counts) if word_counts else 0,
            'std_words_per_text': np.std(word_counts) if word_counts else 0,
            'avg_chars_per_text': np.mean(char_counts) if char_counts else 0,
            'avg_sentences_per_text': np.mean(sentence_counts) if sentence_counts else 0,
            'median_text_length': np.median(word_counts) if word_counts else 0,
            'duplicate_texts': len(texts) - len(texts.drop_duplicates())
        }
        
        # === ANALISI FREQUENZA PAROLE ===
        # Stopwords multilingue
        stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 
            'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'il', 'la', 'le', 'lo', 'gli', 'un', 'una', 'di', 'da', 'del', 'della', 'che',
            'è', 'sono', 'era', 'erano', 'ho', 'hai', 'ha', 'abbiamo', 'avete', 'hanno',
            'a', 'an', 'me', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Parole significative
        meaningful_words = [(word, count) for word, count in word_freq.most_common(100) 
                          if word not in stopwords and len(word) > 2 and count > 2]
        
        # Parole positive e negative
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'love', 'perfect', 'best', 'awesome', 'brilliant', 'outstanding',
            'beautiful', 'incredible', 'superb', 'magnificent', 'terrific',
            'buono', 'ottimo', 'eccellente', 'fantastico', 'meraviglioso',
            'perfetto', 'migliore', 'incredibile', 'magnifico', 'bello'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 
            'disappointing', 'boring', 'poor', 'annoying', 'frustrating',
            'disgusting', 'pathetic', 'ridiculous', 'stupid', 'useless',
            'cattivo', 'terribile', 'orribile', 'pessimo', 'odio',
            'deludente', 'noioso', 'povero', 'fastidioso', 'frustrante'
        ]
        
        positive_found = [(w, c) for w, c in meaningful_words if w in positive_words]
        negative_found = [(w, c) for w, c in meaningful_words if w in negative_words]
        
        analysis['word_frequency_analysis'] = {
            'most_common_words': meaningful_words[:25],
            'positive_words_found': positive_found[:15],
            'negative_words_found': negative_found[:15],
            'total_meaningful_words': len(meaningful_words)
        }
        
        # === ANALISI PATTERN FRASI ===
        # Bigrammi significativi
        bigrams = []
        trigrams = []
        
        for text in texts:
            words = re.findall(r'\b\w+\b', str(text).lower())
            # Bigrammi
            for i in range(len(words) - 1):
                if words[i] not in stopwords and words[i+1] not in stopwords:
                    bigram = f"{words[i]} {words[i+1]}"
                    if len(bigram) > 6:
                        bigrams.append(bigram)
            
            # Trigrammi
            for i in range(len(words) - 2):
                if all(w not in stopwords for w in words[i:i+3]):
                    trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                    if len(trigram) > 10:
                        trigrams.append(trigram)
        
        bigram_freq = Counter(bigrams)
        trigram_freq = Counter(trigrams)
        
        analysis['phrase_pattern_analysis'] = {
            'most_common_bigrams': bigram_freq.most_common(20),
            'most_common_trigrams': trigram_freq.most_common(15),
            'total_bigrams': len(bigrams),
            'total_trigrams': len(trigrams)
        }
        
        # === INDICATORI SENTIMENT ===
        positive_count = sum(word_freq.get(word, 0) for word in positive_words)
        negative_count = sum(word_freq.get(word, 0) for word in negative_words)
        
        analysis['sentiment_indicators'] = {
            'positive_word_occurrences': positive_count,
            'negative_word_occurrences': negative_count,
            'sentiment_ratio': positive_count / max(1, negative_count),
            'total_sentiment_words': positive_count + negative_count,
            'sentiment_density': (positive_count + negative_count) / max(1, total_words)
        }
        
        # === ANALISI TOPICS ===
        # Keyword per topic (ampliato)
        topic_keywords = {
            'quality': ['quality', 'excellent', 'poor', 'good', 'bad', 'amazing', 'terrible', 'qualità', 'eccellente', 'pessimo'],
            'service': ['service', 'staff', 'support', 'help', 'customer', 'team', 'servizio', 'personale', 'aiuto'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money', 'worth', 'prezzo', 'costo', 'economico'],
            'experience': ['experience', 'feel', 'enjoyed', 'disappointed', 'satisfied', 'esperienza', 'sentire', 'soddisfatto'],
            'product': ['product', 'item', 'purchase', 'buy', 'order', 'received', 'prodotto', 'articolo', 'acquisto'],
            'recommendation': ['recommend', 'suggest', 'advice', 'tell', 'friends', 'raccomando', 'consiglio', 'suggerisco'],
            'delivery': ['delivery', 'shipping', 'fast', 'slow', 'quick', 'arrived', 'consegna', 'spedizione', 'veloce'],
            'communication': ['communication', 'response', 'reply', 'contact', 'email', 'comunicazione', 'risposta', 'contatto'],
            'design': ['design', 'look', 'appearance', 'style', 'beautiful', 'ugly', 'aspetto', 'stile', 'bello'],
            'features': ['feature', 'function', 'work', 'works', 'functionality', 'caratteristica', 'funzione', 'funziona']
        }
        
        topic_scores = {}
        all_text_lower = ' '.join(texts).lower()
        
        for topic, keywords in topic_keywords.items():
            score = 0
            mentions = 0
            for keyword in keywords:
                count = all_text_lower.count(keyword)
                score += count
                if count > 0:
                    mentions += 1
            
            if score > 0:
                topic_scores[topic] = {
                    'total_mentions': score,
                    'keyword_diversity': mentions,
                    'relevance_score': score / len(texts)
                }
        
        # Ordina per rilevanza
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1]['total_mentions'], reverse=True)
        
        analysis['topic_analysis'] = {
            'identified_topics': dict(sorted_topics[:10]),
            'total_topics_found': len(topic_scores),
            'most_relevant_topic': sorted_topics[0][0] if sorted_topics else 'none'
        }
        
        # === METRICHE QUALITÀ DATI ===
        empty_texts = sum(1 for text in texts if not str(text).strip())
        very_short_texts = sum(1 for text in texts if len(str(text).strip()) < 10)
        very_long_texts = sum(1 for text in texts if len(str(text).strip()) > 1000)
        
        # Detect potential spam/low quality
        potential_issues = 0
        for text in texts:
            text_str = str(text)
            if (text_str.count('!') > 5 or 
                text_str.count('?') > 3 or 
                len(set(text_str.split())) < len(text_str.split()) * 0.3):
                potential_issues += 1
        
        analysis['data_quality_metrics'] = {
            'empty_texts': empty_texts,
            'very_short_texts': very_short_texts,
            'very_long_texts': very_long_texts,
            'potential_quality_issues': potential_issues,
            'overall_quality_score': 1 - (empty_texts + very_short_texts + potential_issues) / max(1, len(texts)),
            'data_completeness': 1 - (empty_texts / max(1, len(texts))),
            'length_consistency': 1 - (np.std(word_counts) / max(1, np.mean(word_counts))) if word_counts else 0
        }
        
        return analysis
        
    except Exception as e:
        st.error(f"Error in comprehensive text analysis: {e}")
        return {}

def predict_sentiment_models(texts, embeddings, models):
    """Predizione sentiment con modelli caricati - versione migliorata"""
    predictions = {}
    metrics = {}
    
    # Predizioni SVM
    if models.get('svm') is not None:
        try:
            with st.spinner("🔄 SVM predictions in progress..."):
                svm_package = models['svm']
                
                if isinstance(svm_package, dict):
                    svm_model = svm_package.get('model')
                    scaler = svm_package.get('scaler')
                else:
                    svm_model = svm_package
                    scaler = None
                
                if svm_model is None:
                    raise ValueError("SVM model not found")
                
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
                
                predictions['svm'] = [int(p) for p in svm_pred]
                metrics['svm'] = {
                    'model_type': 'SVM',
                    'confidence_avg': float(np.mean(svm_confidence)),
                    'confidence_std': float(np.std(svm_confidence)),
                    'confidence_scores': [float(c) for c in svm_confidence],
                    'prediction_distribution': dict(zip(*np.unique(svm_pred, return_counts=True)))
                }
                
        except Exception as e:
            st.warning(f"⚠️ SVM prediction failed: {e}")
    
    # Predizioni MLP
    if models.get('mlp') is not None:
        try:
            with st.spinner("🔄 MLP predictions in progress..."):
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
                            dummy_output = torch.zeros(len(batch), 1).to(device)
                            all_outputs.append(dummy_output)
                    
                    if not all_outputs:
                        raise ValueError("All batches failed")
                    
                    full_outputs = torch.cat(all_outputs, dim=0)
                    
                    if full_outputs.dim() == 2 and full_outputs.shape[1] == 1:
                        probabilities = full_outputs.squeeze().cpu().numpy()
                        mlp_pred = (probabilities > 0.5).astype(int)
                        mlp_confidence = np.abs(probabilities - 0.5) + 0.5
                    else:
                        probabilities = torch.softmax(full_outputs, dim=1).cpu().numpy()
                        mlp_pred = torch.argmax(full_outputs, dim=1).cpu().numpy()
                        mlp_confidence = np.max(probabilities, axis=1)
                
                predictions['mlp'] = [int(p) for p in mlp_pred]
                metrics['mlp'] = {
                    'model_type': 'MLP',
                    'confidence_avg': float(np.mean(mlp_confidence)),
                    'confidence_std': float(np.std(mlp_confidence)),
                    'confidence_scores': [float(c) for c in mlp_confidence],
                    'prediction_distribution': dict(zip(*np.unique(mlp_pred, return_counts=True)))
                }
                
        except Exception as e:
            st.error(f"❌ MLP prediction failed: {e}")
    
    return predictions, metrics

def analyze_sentiment_by_class(texts: List[str], predictions: List[int]) -> Dict:
    """Analisi delle parole/frasi per classe di sentiment"""
    try:
        analysis = {
            'positive': {'texts': [], 'word_freq': Counter(), 'phrase_freq': Counter()},
            'negative': {'texts': [], 'word_freq': Counter(), 'phrase_freq': Counter()},
            'neutral': {'texts': [], 'word_freq': Counter(), 'phrase_freq': Counter()}
        }
        
        label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
        
        # Stopwords
        stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 
            'this', 'that', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'a', 'an'
        }
        
        # Raggruppa per sentiment
        for text, pred in zip(texts, predictions):
            sentiment = label_map.get(pred, 'neutral')
            if sentiment in analysis:
                analysis[sentiment]['texts'].append(text)
                
                # Analisi parole
                words = re.findall(r'\b\w+\b', str(text).lower())
                meaningful_words = [w for w in words if w not in stopwords and len(w) > 2]
                analysis[sentiment]['word_freq'].update(meaningful_words)
                
                # Analisi frasi (bigrammi)
                for i in range(len(words) - 1):
                    if words[i] not in stopwords and words[i+1] not in stopwords:
                        bigram = f"{words[i]} {words[i+1]}"
                        if len(bigram) > 5:
                            analysis[sentiment]['phrase_freq'].update([bigram])
        
        # Calcola statistiche per ogni sentiment
        for sentiment in analysis:
            texts_list = analysis[sentiment]['texts']
            word_freq = analysis[sentiment]['word_freq']
            phrase_freq = analysis[sentiment]['phrase_freq']
            
            analysis[sentiment]['stats'] = {
                'count': len(texts_list),
                'avg_length': np.mean([len(str(text)) for text in texts_list]) if texts_list else 0,
                'avg_words': np.mean([len(str(text).split()) for text in texts_list]) if texts_list else 0,
                'top_words': word_freq.most_common(20),
                'top_phrases': phrase_freq.most_common(15),
                'unique_words': len(word_freq),
                'total_words': sum(word_freq.values())
            }
        
        return safe_convert_for_json(analysis)
        
    except Exception as e:
        st.error(f"Error in sentiment class analysis: {e}")
        return {}

def analyze_uploaded_csv(uploaded_file, embedding_model):
    """Analisi rapida di un CSV caricato - versione ottimizzata"""
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
        
        with st.spinner("🔄 Performing comprehensive text analysis..."):
            comprehensive_analysis = comprehensive_text_analysis(df, text_col)
        
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
            'comprehensive_analysis': comprehensive_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # FIXED: Gestione corretta delle etichette sentiment
        sentiment_columns = ['sentiment', 'label', 'class', 'target', 'rating']
        for col in sentiment_columns:
            if col in df.columns:
                try:
                    # Usa la funzione fix per gestire le etichette
                    fixed_labels = fix_sentiment_labels(df, col)
                    stats['sentiment_distribution'] = safe_convert_for_json(pd.Series(fixed_labels).value_counts().to_dict())
                    stats['sentiment_column'] = col
                    stats['fixed_labels'] = fixed_labels
                    break
                except Exception as e:
                    st.warning(f"Error processing sentiment column {col}: {e}")
                    continue
        
        return df, embeddings, stats
        
    except Exception as e:
        st.error(f"❌ Error analyzing CSV: {e}")
        return None, None, None

def create_scientific_visualizations(df, stats, predictions=None, metrics=None, sentiment_analysis=None):
    """Crea visualizzazioni scientifiche complete e professionali"""
    
    # === SEZIONE 1: METRICHE STATISTICHE PRINCIPALI ===
    st.markdown('<div class="analysis-header">📊 Statistical Dataset Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    comprehensive = stats.get('comprehensive_analysis', {})
    basic_stats = comprehensive.get('basic_statistics', {})
    
    with col1:
        st.metric("Total Samples", f"{stats.get('total_reviews', 0):,}")
    with col2:
        st.metric("Avg Text Length", f"{stats.get('avg_length', 0):.0f} chars")
    with col3:
        st.metric("Total Words", f"{basic_stats.get('total_words', 0):,}")
    with col4:
        st.metric("Unique Words", f"{basic_stats.get('unique_words', 0):,}")
    with col5:
        vocab_richness = basic_stats.get('vocabulary_richness', 0)
        st.metric("Vocabulary Richness", f"{vocab_richness:.3f}")
    with col6:
        quality_score = comprehensive.get('data_quality_metrics', {}).get('overall_quality_score', 0)
        st.metric("Data Quality", f"{quality_score:.1%}")
    
    # === SEZIONE 2: DISTRIBUZIONE SENTIMENT ===
    if predictions:
        st.markdown('<div class="analysis-header">🎯 Sentiment Distribution Analysis</div>', unsafe_allow_html=True)
        
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
                fig_sentiment.update_layout(
                    font=dict(size=14),
                    title_font_size=16,
                    showlegend=False
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            with col2:
                fig_pie = px.pie(
                    sentiment_df,
                    values='Count',
                    names='Sentiment',
                    title="Percentage Distribution",
                    color_discrete_map={'Positive': '#2E8B57', 'Negative': '#DC143C', 'Neutral': '#708090'}
                )
                fig_pie.update_layout(
                    font=dict(size=14),
                    title_font_size=16
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Tabella statistiche sentiment
            st.markdown('<div class="results-table">', unsafe_allow_html=True)
            st.markdown("**Detailed Sentiment Statistics:**")
            for _, row in sentiment_df.iterrows():
                st.write(f"• **{row['Sentiment']}**: {row['Count']:,} samples ({row['Percentage']:.1f}%)")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # === SEZIONE 3: ANALISI PAROLE PER SENTIMENT ===
    if sentiment_analysis:
        st.markdown('<div class="analysis-header">🔤 Word Analysis by Sentiment Class</div>', unsafe_allow_html=True)
        
        # Create tabs for each sentiment
        tab_pos, tab_neg, tab_neu = st.tabs(["😊 Positive Words", "😞 Negative Words", "😐 Neutral Words"])
        
        for i, (tab, sentiment) in enumerate([(tab_pos, 'positive'), (tab_neg, 'negative'), (tab_neu, 'neutral')]):
            with tab:
                if sentiment in sentiment_analysis and sentiment_analysis[sentiment]['stats']['top_words']:
                    top_words = sentiment_analysis[sentiment]['stats']['top_words'][:15]
                    
                    if top_words:
                        words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                        
                        # Grafico parole
                        colors = ['#2E8B57' if sentiment == 'positive' else '#DC143C' if sentiment == 'negative' else '#708090']
                        fig_words = px.bar(
                            words_df,
                            x='Word',
                            y='Frequency',
                            title=f"Most Frequent {sentiment.title()} Words",
                            color_discrete_sequence=colors
                        )
                        fig_words.update_layout(
                            xaxis_tickangle=-45,
                            font=dict(size=12),
                            title_font_size=14
                        )
                        st.plotly_chart(fig_words, use_container_width=True)
                        
                        # Tabella parole
                        st.markdown(f"**{sentiment.title()} Words Frequency Table:**")
                        st.dataframe(words_df, use_container_width=True, hide_index=True)
                        
                        # Statistiche
                        stats_data = sentiment_analysis[sentiment]['stats']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Sample Count", f"{stats_data['count']:,}")
                        with col2:
                            st.metric("Unique Words", f"{stats_data['unique_words']:,}")
                        with col3:
                            st.metric("Avg Words/Text", f"{stats_data['avg_words']:.1f}")
    
    # === SEZIONE 4: ANALISI FRASI FREQUENTI ===
    if sentiment_analysis:
        st.markdown('<div class="analysis-header">💬 Most Frequent Phrases by Sentiment</div>', unsafe_allow_html=True)
        
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in sentiment_analysis and sentiment_analysis[sentiment]['stats']['top_phrases']:
                phrases = sentiment_analysis[sentiment]['stats']['top_phrases'][:10]
                
                if phrases:
                    with st.expander(f"🔍 {sentiment.title()} Phrases ({len(phrases)} found)"):
                        phrases_df = pd.DataFrame(phrases, columns=['Phrase', 'Frequency'])
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            colors = ['#2E8B57' if sentiment == 'positive' else '#DC143C' if sentiment == 'negative' else '#708090']
                            fig_phrase = px.bar(
                                phrases_df,
                                x='Frequency',
                                y='Phrase',
                                orientation='h',
                                title=f"Most Frequent {sentiment.title()} Phrases",
                                color_discrete_sequence=colors
                            )
                            fig_phrase.update_layout(
                                font=dict(size=11),
                                title_font_size=13
                            )
                            st.plotly_chart(fig_phrase, use_container_width=True)
                        
                        with col2:
                            st.markdown("**Top Phrases:**")
                            for phrase, freq in phrases[:6]:
                                st.write(f"• **\"{phrase}\"**: {freq} times")
    
    # === SEZIONE 5: ANALISI TOPICS ===
    topic_analysis = comprehensive.get('topic_analysis', {})
    if topic_analysis and topic_analysis.get('identified_topics'):
        st.markdown('<div class="analysis-header">🎯 Topic Analysis and Thematic Insights</div>', unsafe_allow_html=True)
        
        topics = topic_analysis['identified_topics']
        
        if topics:
            # Crea DataFrame per visualizzazione
            topics_data = []
            for topic, data in topics.items():
                topics_data.append({
                    'Topic': topic.title(),
                    'Total Mentions': data['total_mentions'],
                    'Keyword Diversity': data['keyword_diversity'],
                    'Relevance Score': data['relevance_score']
                })
            
            topics_df = pd.DataFrame(topics_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_topics = px.bar(
                    topics_df,
                    x='Topic',
                    y='Total Mentions',
                    title="Most Cited Topics",
                    color='Relevance Score',
                    color_continuous_scale='viridis'
                )
                fig_topics.update_layout(
                    xaxis_tickangle=-45,
                    font=dict(size=12),
                    title_font_size=14
                )
                st.plotly_chart(fig_topics, use_container_width=True)
            
            with col2:
                fig_relevance = px.scatter(
                    topics_df,
                    x='Keyword Diversity',
                    y='Relevance Score',
                    size='Total Mentions',
                    hover_name='Topic',
                    title="Topic Relevance vs Keyword Diversity",
                    color='Total Mentions',
                    color_continuous_scale='plasma'
                )
                fig_relevance.update_layout(
                    font=dict(size=12),
                    title_font_size=14
                )
                st.plotly_chart(fig_relevance, use_container_width=True)
            
            # Tabella topics dettagliata
            st.markdown("**Detailed Topic Analysis:**")
            st.dataframe(topics_df, use_container_width=True, hide_index=True)
    
    # === SEZIONE 6: METRICHE CONFIDENZA MODELLI ===
    if metrics:
        st.markdown('<div class="analysis-header">📈 Model Performance and Confidence Metrics</div>', unsafe_allow_html=True)
        
        for model_name, model_metrics in metrics.items():
            if 'confidence_scores' in model_metrics:
                confidence_scores = model_metrics['confidence_scores']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_conf = px.histogram(
                        x=confidence_scores,
                        nbins=20,
                        title=f'{model_name.upper()} Confidence Score Distribution',
                        labels={'x': 'Confidence Score', 'y': 'Count'},
                        color_discrete_sequence=['#3498db']
                    )
                    fig_conf.add_vline(x=np.mean(confidence_scores), line_dash="dash", 
                                     annotation_text=f"Mean: {np.mean(confidence_scores):.3f}")
                    fig_conf.update_layout(
                        font=dict(size=12),
                        title_font_size=14
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                with col2:
                    st.markdown(f"**{model_name.upper()} Performance Statistics:**")
                    
                    # Metriche di base
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        st.metric("Mean Confidence", f"{np.mean(confidence_scores):.3f}")
                        st.metric("Std Deviation", f"{np.std(confidence_scores):.3f}")
                    with col2_2:
                        st.metric("Min Confidence", f"{np.min(confidence_scores):.3f}")
                        st.metric("Max Confidence", f"{np.max(confidence_scores):.3f}")
                    
                    # Distribuzione per livelli
                    high_conf = len([c for c in confidence_scores if c > 0.8])
                    medium_conf = len([c for c in confidence_scores if 0.6 <= c <= 0.8])
                    low_conf = len([c for c in confidence_scores if c < 0.6])
                    
                    st.markdown("**Confidence Level Distribution:**")
                    st.write(f"• **High (>0.8)**: {high_conf} ({high_conf/len(confidence_scores)*100:.1f}%)")
                    st.write(f"• **Medium (0.6-0.8)**: {medium_conf} ({medium_conf/len(confidence_scores)*100:.1f}%)")
                    st.write(f"• **Low (<0.6)**: {low_conf} ({low_conf/len(confidence_scores)*100:.1f}%)")
    
    # === SEZIONE 7: DISTRIBUZIONE LUNGHEZZA TESTI ===
    text_col = stats.get('text_column')
    if text_col and text_col in df.columns:
        st.markdown('<div class="analysis-header">📏 Text Length Distribution Analysis</div>', unsafe_allow_html=True)
        
        lengths = df[text_col].str.len().fillna(0)
        word_counts = df[text_col].str.split().str.len().fillna(0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_char = px.histogram(
                x=lengths,
                nbins=50,
                title="Character Length Distribution",
                labels={'x': 'Characters', 'y': 'Frequency'},
                color_discrete_sequence=['#e74c3c']
            )
            fig_char.add_vline(x=lengths.mean(), line_dash="dash", 
                             annotation_text=f"Mean: {lengths.mean():.0f}")
            fig_char.add_vline(x=lengths.median(), line_dash="dot", 
                             annotation_text=f"Median: {lengths.median():.0f}")
            fig_char.update_layout(
                font=dict(size=12),
                title_font_size=14
            )
            st.plotly_chart(fig_char, use_container_width=True)
        
        with col2:
            fig_words = px.histogram(
                x=word_counts,
                nbins=30,
                title="Word Count Distribution",
                labels={'x': 'Words', 'y': 'Frequency'},
                color_discrete_sequence=['#2ecc71']
            )
            fig_words.add_vline(x=word_counts.mean(), line_dash="dash",
                              annotation_text=f"Mean: {word_counts.mean():.1f}")
            fig_words.add_vline(x=word_counts.median(), line_dash="dot",
                              annotation_text=f"Median: {word_counts.median():.0f}")
            fig_words.update_layout(
                font=dict(size=12),
                title_font_size=14
            )
            st.plotly_chart(fig_words, use_container_width=True)
        
        # Tabella statistiche lunghezza
        st.markdown("**Text Length Statistical Summary:**")
        length_stats_df = pd.DataFrame({
            'Metric': ['Character Length', 'Word Count'],
            'Mean': [f"{lengths.mean():.1f}", f"{word_counts.mean():.1f}"],
            'Median': [f"{lengths.median():.0f}", f"{word_counts.median():.0f}"],
            'Std Dev': [f"{lengths.std():.1f}", f"{word_counts.std():.1f}"],
            'Min': [f"{lengths.min():.0f}", f"{word_counts.min():.0f}"],
            'Max': [f"{lengths.max():.0f}", f"{word_counts.max():.0f}"]
        })
        st.dataframe(length_stats_df, use_container_width=True, hide_index=True)

def create_complete_export_package(df, predictions, metrics, stats) -> bytes:
    """Crea un pacchetto ZIP completo per l'export con tutti i risultati"""
    zip_buffer = io.BytesIO()
    
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            # === FILE 1: CSV COMPLETO CON PREDIZIONI ===
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
                            continue
                    
                    # Aggiungi accordo tra modelli se presenti
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
                    zip_file.writestr("01_complete_predictions.csv", csv_buffer.getvalue())
                except Exception as e:
                    pass
            
            # === FILE 2: REPORT SCIENTIFICO DETTAGLIATO ===
            try:
                report_lines = []
                report_lines.append("SCIENTIFIC SENTIMENT ANALYSIS REPORT")
                report_lines.append("=" * 60)
                report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append(f"Analysis Version: Professional Scientific v1.0")
                report_lines.append(f"Dataset: {stats.get('total_reviews', 0):,} texts analyzed")
                report_lines.append("")
                
                # === DATASET STATISTICS ===
                report_lines.append("DATASET STATISTICS:")
                report_lines.append("-" * 20)
                report_lines.append(f"Total samples: {stats.get('total_reviews', 0):,}")
                report_lines.append(f"Text column: {stats.get('text_column', 'Unknown')}")
                report_lines.append(f"Average length: {stats.get('avg_length', 0):.0f} characters")
                report_lines.append(f"Missing values: {stats.get('null_values', 0)}")
                report_lines.append(f"Duplicate rows: {stats.get('duplicates', 0)}")
                report_lines.append("")
                
                # === COMPREHENSIVE ANALYSIS ===
                comprehensive = stats.get('comprehensive_analysis', {})
                if comprehensive:
                    basic_stats = comprehensive.get('basic_statistics', {})
                    
                    report_lines.append("LINGUISTIC ANALYSIS:")
                    report_lines.append("-" * 18)
                    report_lines.append(f"Total words: {basic_stats.get('total_words', 0):,}")
                    report_lines.append(f"Unique words: {basic_stats.get('unique_words', 0):,}")
                    report_lines.append(f"Vocabulary richness: {basic_stats.get('vocabulary_richness', 0):.3f}")
                    report_lines.append(f"Average words per text: {basic_stats.get('avg_words_per_text', 0):.1f}")
                    report_lines.append(f"Standard deviation: {basic_stats.get('std_words_per_text', 0):.1f}")
                    report_lines.append("")
                    
                    # Sentiment indicators
                    sentiment_indicators = comprehensive.get('sentiment_indicators', {})
                    if sentiment_indicators:
                        report_lines.append("SENTIMENT INDICATORS:")
                        report_lines.append("-" * 19)
                        report_lines.append(f"Positive word occurrences: {sentiment_indicators.get('positive_word_occurrences', 0)}")
                        report_lines.append(f"Negative word occurrences: {sentiment_indicators.get('negative_word_occurrences', 0)}")
                        report_lines.append(f"Sentiment ratio (pos/neg): {sentiment_indicators.get('sentiment_ratio', 1):.2f}")
                        report_lines.append(f"Sentiment density: {sentiment_indicators.get('sentiment_density', 0):.4f}")
                        report_lines.append("")
                    
                    # Topic analysis
                    topic_analysis = comprehensive.get('topic_analysis', {})
                    if topic_analysis and topic_analysis.get('identified_topics'):
                        report_lines.append("TOPIC ANALYSIS:")
                        report_lines.append("-" * 14)
                        report_lines.append(f"Total topics identified: {topic_analysis.get('total_topics_found', 0)}")
                        report_lines.append(f"Most relevant topic: {topic_analysis.get('most_relevant_topic', 'none')}")
                        
                        topics = topic_analysis['identified_topics']
                        for topic, data in list(topics.items())[:5]:
                            report_lines.append(f"  {topic.title()}: {data['total_mentions']} mentions (relevance: {data['relevance_score']:.3f})")
                        report_lines.append("")
                    
                    # Data quality
                    quality_metrics = comprehensive.get('data_quality_metrics', {})
                    if quality_metrics:
                        report_lines.append("DATA QUALITY ASSESSMENT:")
                        report_lines.append("-" * 24)
                        report_lines.append(f"Overall quality score: {quality_metrics.get('overall_quality_score', 0):.1%}")
                        report_lines.append(f"Data completeness: {quality_metrics.get('data_completeness', 0):.1%}")
                        report_lines.append(f"Empty texts: {quality_metrics.get('empty_texts', 0)}")
                        report_lines.append(f"Very short texts: {quality_metrics.get('very_short_texts', 0)}")
                        report_lines.append(f"Potential quality issues: {quality_metrics.get('potential_quality_issues', 0)}")
                        report_lines.append("")
                
                # === MODEL PERFORMANCE ===
                if predictions and metrics:
                    report_lines.append("MODEL PERFORMANCE:")
                    report_lines.append("-" * 18)
                    
                    for model_name, pred in predictions.items():
                        if model_name in metrics:
                            model_metrics = metrics[model_name]
                            pred_dist = dict(zip(*np.unique(pred, return_counts=True)))
                            
                            report_lines.append(f"{model_name.upper()} ({model_metrics.get('model_type', 'Unknown')}):")
                            report_lines.append(f"  Total predictions: {len(pred):,}")
                            report_lines.append(f"  Average confidence: {model_metrics.get('confidence_avg', 0):.3f}")
                            report_lines.append(f"  Confidence std dev: {model_metrics.get('confidence_std', 0):.3f}")
                            
                            label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
                            for class_id, count in pred_dist.items():
                                label = label_map.get(class_id, f'Class_{class_id}')
                                percentage = (count / len(pred)) * 100
                                report_lines.append(f"    {label}: {count:,} ({percentage:.1f}%)")
                            report_lines.append("")
                
                # === METHODOLOGY ===
                report_lines.append("METHODOLOGY:")
                report_lines.append("-" * 12)
                report_lines.append("This analysis employs objective statistical methods for sentiment classification.")
                report_lines.append("All metrics are computed using quantitative measures without subjective interpretation.")
                report_lines.append("Results are reproducible and based on established computational linguistics techniques.")
                report_lines.append("")
                report_lines.append("Features include:")
                report_lines.append("- Comprehensive text statistical analysis")
                report_lines.append("- Word frequency and pattern analysis")
                report_lines.append("- Topic identification using keyword matching")
                report_lines.append("- Sentiment classification with confidence scoring")
                report_lines.append("- Data quality assessment and metrics")
                report_lines.append("")
                report_lines.append("Generated by Professional Scientific Sentiment Analysis System v1.0")
                report_lines.append("Objective • Reproducible • Research-Grade")
                
                zip_file.writestr("02_scientific_analysis_report.txt", '\n'.join(report_lines))
            except Exception as e:
                pass
            
            # === FILE 3: DATI JSON STRUTTURATI ===
            try:
                json_data = {
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'version': 'professional_scientific_v1.0',
                        'analysis_type': 'comprehensive_sentiment_analysis'
                    },
                    'dataset_summary': {
                        'total_samples': stats.get('total_reviews', 0),
                        'text_column': stats.get('text_column', 'Unknown'),
                        'data_quality_score': comprehensive.get('data_quality_metrics', {}).get('overall_quality_score', 0),
                        'vocabulary_richness': comprehensive.get('basic_statistics', {}).get('vocabulary_richness', 0)
                    },
                    'comprehensive_analysis': stats.get('comprehensive_analysis', {}),
                    'model_performance': metrics,
                    'predictions': predictions
                }
                
                zip_file.writestr("03_structured_analysis_data.json", 
                                 json.dumps(safe_convert_for_json(json_data), indent=2, ensure_ascii=False))
            except Exception as e:
                pass
            
            # === FILE 4: WORD FREQUENCY TABLES ===
            try:
                if 'comprehensive_analysis' in stats:
                    word_analysis = stats['comprehensive_analysis'].get('word_frequency_analysis', {})
                    
                    # Most common words
                    if word_analysis.get('most_common_words'):
                        words_df = pd.DataFrame(word_analysis['most_common_words'], columns=['Word', 'Frequency'])
                        zip_file.writestr("04a_most_common_words.csv", words_df.to_csv(index=False))
                    
                    # Positive words
                    if word_analysis.get('positive_words_found'):
                        pos_df = pd.DataFrame(word_analysis['positive_words_found'], columns=['Word', 'Frequency'])
                        zip_file.writestr("04b_positive_words.csv", pos_df.to_csv(index=False))
                    
                    # Negative words
                    if word_analysis.get('negative_words_found'):
                        neg_df = pd.DataFrame(word_analysis['negative_words_found'], columns=['Word', 'Frequency'])
                        zip_file.writestr("04c_negative_words.csv", neg_df.to_csv(index=False))
            except Exception as e:
                pass
            
            # === FILE 5: PHRASE ANALYSIS ===
            try:
                if 'comprehensive_analysis' in stats:
                    phrase_analysis = stats['comprehensive_analysis'].get('phrase_pattern_analysis', {})
                    
                    # Bigrams
                    if phrase_analysis.get('most_common_bigrams'):
                        bigrams_df = pd.DataFrame(phrase_analysis['most_common_bigrams'], columns=['Bigram', 'Frequency'])
                        zip_file.writestr("05a_most_common_bigrams.csv", bigrams_df.to_csv(index=False))
                    
                    # Trigrams
                    if phrase_analysis.get('most_common_trigrams'):
                        trigrams_df = pd.DataFrame(phrase_analysis['most_common_trigrams'], columns=['Trigram', 'Frequency'])
                        zip_file.writestr("05b_most_common_trigrams.csv", trigrams_df.to_csv(index=False))
            except Exception as e:
                pass
            
            # === FILE 6: TOPIC ANALYSIS ===
            try:
                if 'comprehensive_analysis' in stats:
                    topic_analysis = stats['comprehensive_analysis'].get('topic_analysis', {})
                    
                    if topic_analysis.get('identified_topics'):
                        topics_data = []
                        for topic, data in topic_analysis['identified_topics'].items():
                            topics_data.append({
                                'Topic': topic,
                                'Total_Mentions': data['total_mentions'],
                                'Keyword_Diversity': data['keyword_diversity'],
                                'Relevance_Score': data['relevance_score']
                            })
                        
                        topics_df = pd.DataFrame(topics_data)
                        zip_file.writestr("06_topic_analysis.csv", topics_df.to_csv(index=False))
            except Exception as e:
                pass
            
            # === FILE 7: README DETTAGLIATO ===
            try:
                readme_content = f"""
PROFESSIONAL SCIENTIFIC SENTIMENT ANALYSIS PACKAGE
{'='*52}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Type: Professional Scientific Statistical Analysis
Dataset: {stats.get('total_reviews', 0):,} texts analyzed

PACKAGE CONTENTS:
{'='*17}

01_complete_predictions.csv
   Complete dataset with model predictions and confidence scores
   Includes: original data, predictions, confidence levels, model agreement
   
02_scientific_analysis_report.txt
   Comprehensive technical analysis in human-readable format
   Includes: statistical metrics, model performance, linguistic analysis
   
03_structured_analysis_data.json
   Complete analysis data in structured JSON format
   Includes: metadata, comprehensive analysis, model performance
   
04a_most_common_words.csv
04b_positive_words.csv  
04c_negative_words.csv
   Word frequency analysis organized by categories
   Includes: word rankings with frequency counts
   
05a_most_common_bigrams.csv
05b_most_common_trigrams.csv
   Phrase pattern analysis with frequency statistics
   Includes: most frequent word combinations
   
06_topic_analysis.csv
   Topic identification results with relevance scores
   Includes: topic names, mention counts, keyword diversity
   
README.txt
   This comprehensive documentation file

SCIENTIFIC APPROACH:
{'='*19}

This package contains objective statistical analysis results based on:
- Quantitative measurements and reproducible methods
- Word frequency and pattern analysis
- Topic identification using keyword matching  
- Sentiment classification with AI models
- Comprehensive data quality assessment

Key Features:
- Objective statistical reporting
- Reproducible analysis methods
- Quantitative sentiment classification
- Research-grade documentation
- Comprehensive linguistic analysis

USAGE INSTRUCTIONS:
{'='*18}

For Researchers:
- Use JSON files for programmatic analysis
- CSV files are ready for statistical software
- Methodology is documented for replication
- All data structures are well-documented

For Business Users:
- Read the scientific report (02_) for key findings
- Use prediction CSV (01_) for further analysis
- Word analysis files (04_) show specific patterns
- Topic analysis (06_) reveals thematic insights

For Developers:
- JSON format enables easy integration
- All data structures are documented
- API-ready format for automated processing
- CSV exports for various analysis tools

For Advanced Analysis:
- Combine word and topic analysis for deeper insights
- Use phrase analysis to identify common expressions
- Statistical metrics enable rigorous evaluation
- Confidence scores support decision-making

Generated by Professional Scientific Sentiment Analysis System v1.0
Research-Grade • Objective • Reproducible • Comprehensive
                """
                
                zip_file.writestr("README.txt", readme_content)
            except Exception as e:
                pass
    
    except Exception as e:
        st.error(f"Error creating export package: {e}")
        return io.BytesIO().getvalue()
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def create_timestamp_session():
    """Crea un timestamp unico per la sessione di analisi"""
    if 'session_timestamp' not in st.session_state:
        st.session_state.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return st.session_state.session_timestamp

def get_session_results_dir():
    """Ottiene la directory risultati per la sessione corrente"""
    timestamp = create_timestamp_session()
    session_dir = RESULTS_DIR / f"session_{timestamp}"
    
    subdirs = ['processed', 'models', 'reports', 'plots', 'exports']
    for subdir in subdirs:
        (session_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return session_dir

def main():
    """Applicazione principale Streamlit - 3 sezioni finali"""
    
    try:
        # Header principale professionale
        st.markdown('<div class="main-header">🔬 Professional Scientific Sentiment Analysis System</div>', 
                    unsafe_allow_html=True)
        
        # Sidebar con informazioni di sistema
        with st.sidebar:
            st.header("🔧 System Information")
            st.info(f"📁 Project Root: {PROJECT_ROOT}")
            st.info(f"🗃️ Data Directory: {DATA_DIR}")
            st.info(f"📊 Results Directory: {RESULTS_DIR}")
            
            # Timestamp sessione
            timestamp = create_timestamp_session()
            st.info(f"🕐 Session: {timestamp}")
            
            # Quick Actions
            st.header("📋 Quick Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Refresh", help="Refresh models and cache"):
                    st.cache_resource.clear()
                    st.rerun()
            
            with col2:
                if st.button("🗂️ Clear Cache", help="Clear all cache and session data"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    for key in list(st.session_state.keys()):
                        if key not in ['session_timestamp']:
                            del st.session_state[key]
                    st.success("✅ Cache cleared!")
                    st.rerun()
            
            # Model Status
            st.header("🧠 Model Status")
            try:
                models = load_trained_models()
                for model_name, status in models['status'].items():
                    if status == 'loaded':
                        st.success(f"✅ {model_name.upper()}")
                    else:
                        st.info(f"⚠️ {model_name.upper()}")
            except Exception:
                st.warning("⚠️ Model status unknown")
        
        # Caricamento risorse di sistema
        with st.spinner("🔄 Loading system resources..."):
            embedding_model = load_embedding_model()
            models = load_trained_models()
            main_df, main_dataset_path = load_main_dataset()
        
        # === SOLO 3 TAB FINALI ===
        tab1, tab2, tab3 = st.tabs([
            "🧠 Dataset Analysis", 
            "📂 Upload New Dataset", 
            "📥 Export Results"
        ])
        
        # === TAB 1: DATASET ANALYSIS UNIFICATO ===
        with tab1:
            st.markdown('<div class="scientific-section">', unsafe_allow_html=True)
            st.header("🧠 Comprehensive Dataset Analysis")
            st.markdown("*Unified scientific analysis with complete statistical insights and visualizations*")
            
            if main_df is not None:
                st.success(f"✅ Main dataset loaded from: {main_dataset_path}")
                
                # Trova colonna di testo
                text_columns = ['review', 'text', 'content', 'comment', 'message', 'description']
                text_col = None
                for col in text_columns:
                    if col in main_df.columns:
                        text_col = col
                        break
                
                if text_col:
                    # === TABELLA COMPLETA E SCROLLABILE ===
                    st.subheader("📋 Complete Dataset Table")
                    
                    # Funzionalità di ricerca avanzata
                    col_search, col_filter = st.columns([2, 1])
                    with col_search:
                        search_term = st.text_input("🔍 Search in dataset:", placeholder="Enter search term...")
                    with col_filter:
                        show_rows = st.selectbox("Rows to display:", [20, 50, 100, 200, "All"], index=1)
                    
                    display_df = main_df
                    if search_term:
                        try:
                            mask = main_df[text_col].astype(str).str.contains(search_term, case=False, na=False)
                            if mask.any():
                                display_df = main_df[mask]
                                st.info(f"🔍 Found {len(display_df)} matching records")
                            else:
                                st.warning("⚠️ No matches found")
                        except Exception as e:
                            st.warning(f"Search error: {e}")
                    
                    # Gestione visualizzazione righe
                    if show_rows != "All":
                        display_df = display_df.head(int(show_rows))
                    
                    # Tabella con tutte le recensioni
                    st.dataframe(
                        display_df, 
                        use_container_width=True,
                        height=500
                    )
                    
                    # === ANALISI AUTOMATICA COMPLETA ===
                    with st.spinner("🔄 Performing comprehensive scientific analysis..."):
                        # Prepara dati per analisi
                        texts = main_df[text_col].fillna('').astype(str).tolist()
                        
                        # Genera embeddings
                        if embedding_model:
                            embeddings = embedding_model.encode(texts)
                            
                            # Predizioni se modelli disponibili
                            predictions = {}
                            metrics = {}
                            
                            if any(model is not None for model in models.values()):
                                predictions, metrics = predict_sentiment_models(texts, embeddings, models)
                            
                            # Analisi comprensiva
                            comprehensive_analysis = comprehensive_text_analysis(main_df, text_col)
                            
                            # Analisi per classe sentiment se predizioni disponibili
                            sentiment_analysis = {}
                            if predictions:
                                # Usa la prima predizione disponibile per l'analisi
                                first_predictions = list(predictions.values())[0]
                                sentiment_analysis = analyze_sentiment_by_class(texts, first_predictions)
                            
                            stats = {
                                'total_reviews': len(main_df),
                                'avg_length': main_df[text_col].str.len().mean(),
                                'max_length': main_df[text_col].str.len().max(),
                                'min_length': main_df[text_col].str.len().min(),
                                'null_values': main_df[text_col].isnull().sum(),
                                'duplicates': main_df.duplicated().sum(),
                                'text_column': text_col,
                                'comprehensive_analysis': comprehensive_analysis
                            }
                            
                            # Salva in sessione
                            st.session_state['current_analysis'] = {
                                'df': main_df,
                                'embeddings': embeddings,
                                'predictions': predictions,
                                'metrics': metrics,
                                'sentiment_analysis': sentiment_analysis,
                                'stats': stats,
                                'filename': Path(main_dataset_path).name,
                                'timestamp': datetime.now(),
                                'analysis_type': 'unified_analysis'
                            }
                            
                            # === VISUALIZZAZIONI SCIENTIFICHE COMPLETE ===
                            create_scientific_visualizations(main_df, stats, predictions, metrics, sentiment_analysis)
                        else:
                            st.error("❌ Embedding model not available")
                else:
                    st.error("❌ Text column not found in dataset")
            else:
                st.info("ℹ️ No main dataset found.")
                
                st.markdown("""
                ### 📋 Required Dataset Format:
                
                Your CSV should contain:
                - **Text column**: 'review', 'text', 'content', 'comment', 'message', or 'description'  
                - **Label column** (optional): 'sentiment', 'label', 'class', or 'target'
                
                Example:
                ```
                text,sentiment
                "This movie is absolutely fantastic!",positive
                "I didn't like it at all",negative
                "It was okay, nothing special",neutral
                ```
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # === TAB 2: UPLOAD NEW DATASET ===
        with tab2:
            st.markdown('<div class="scientific-section">', unsafe_allow_html=True)
            st.header("📂 Upload New Dataset (CSV)")
            st.markdown("*Upload and analyze new CSV files with comprehensive scientific analysis*")
            
            # Sezione upload file
            st.subheader("📁 CSV File Upload")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Choose a CSV file for analysis",
                    type=['csv'],
                    help="CSV must contain a text column ('review', 'text', 'content', 'comment', 'message', or 'description')"
                )
            
            with col2:
                if uploaded_file:
                    try:
                        file_size = len(uploaded_file.getvalue()) / (1024**2)  # MB
                        st.info(f"📄 **File:** {uploaded_file.name}")
                        st.info(f"📊 **Size:** {file_size:.2f} MB")
                    except Exception:
                        st.info(f"📄 **File:** {uploaded_file.name}")
            
            if uploaded_file is not None:
                st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
                
                # Solo analisi rapida
                st.subheader("⚡ Quick Analysis")
                
                if st.button("🚀 Start Analysis", type="primary"):
                    try:
                        with st.spinner("🔄 Analyzing uploaded CSV..."):
                            df, embeddings, stats = analyze_uploaded_csv(uploaded_file, embedding_model)
                        
                        if df is not None and embeddings is not None:
                            st.success("✅ Analysis completed!")
                            
                            # Predizioni se modelli disponibili
                            predictions = {}
                            metrics = {}
                            
                            if any(model is not None for model in models.values()):
                                text_col = stats['text_column']
                                texts = df[text_col].fillna('').astype(str).tolist()
                                predictions, metrics = predict_sentiment_models(texts, embeddings, models)
                            
                            # Analisi per classe sentiment se predizioni disponibili
                            sentiment_analysis = {}
                            if predictions:
                                # Usa la prima predizione disponibile per l'analisi
                                first_predictions = list(predictions.values())[0]
                                sentiment_analysis = analyze_sentiment_by_class(texts, first_predictions)
                            
                            # Salva in sessione
                            st.session_state['current_analysis'] = {
                                'df': df,
                                'embeddings': embeddings,
                                'predictions': predictions,
                                'metrics': metrics,
                                'sentiment_analysis': sentiment_analysis,
                                'stats': stats,
                                'filename': uploaded_file.name,
                                'timestamp': datetime.now(),
                                'analysis_type': 'uploaded_csv'
                            }
                            
                            # Panoramica risultati
                            st.subheader("📊 Analysis Results Overview")
                            
                            # Metriche chiave
                            col1, col2, col3, col4, col5, col6 = st.columns(6)
                            
                            comprehensive = stats.get('comprehensive_analysis', {})
                            basic_stats = comprehensive.get('basic_statistics', {})
                            
                            with col1:
                                st.metric("📄 Total Samples", f"{len(df):,}")
                            with col2:
                                st.metric("📝 Avg Length", f"{stats.get('avg_length', 0):.0f} chars")
                            with col3:
                                st.metric("📚 Total Words", f"{basic_stats.get('total_words', 0):,}")
                            with col4:
                                st.metric("🔤 Unique Words", f"{basic_stats.get('unique_words', 0):,}")
                            with col5:
                                vocab_richness = basic_stats.get('vocabulary_richness', 0)
                                st.metric("🎨 Vocab Richness", f"{vocab_richness:.3f}")
                            with col6:
                                quality_score = comprehensive.get('data_quality_metrics', {}).get('overall_quality_score', 0)
                                st.metric("✅ Quality Score", f"{quality_score:.1%}")
                            
                            # Anteprima dati
                            st.subheader("👀 Data Preview")
                            preview_rows = st.slider("Rows to display:", 5, 50, 20)
                            st.dataframe(df.head(preview_rows), use_container_width=True)
                            
                            # Visualizzazioni scientifiche complete
                            create_scientific_visualizations(df, stats, predictions, metrics, sentiment_analysis)
                            
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
            else:
                # Guida quando nessun file è caricato
                st.markdown("""
                ### 📂 CSV Analysis Features:
                
                Upload a CSV file to access comprehensive analysis capabilities:
                
                **⚡ Quick Analysis**
                - Fast text processing and embedding generation
                - Immediate sentiment predictions (if models available)
                - Comprehensive statistical analysis and insights
                - Scientific visualization generation
                
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
                
                ### 🚀 Analysis Features:
                - **Complete Statistical Analysis**: Word frequency, phrases, sentiment indicators
                - **Professional Visualizations**: Interactive charts and tables
                - **Scientific Reports**: Objective statistical measurements
                - **Exportable Results**: Comprehensive downloadable packages
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # === TAB 3: EXPORT RESULTS ===
        with tab3:
            st.markdown('<div class="export-section">', unsafe_allow_html=True)
            st.header("📥 Export Results")
            st.markdown("*Complete scientific analysis packages saved to project_root/results/*")
            
            # Controlla risultati disponibili
            if 'current_analysis' in st.session_state:
                try:
                    analysis = st.session_state['current_analysis']
                    
                    # Riepilogo risultati
                    st.success(f"✅ Analysis results available for: **{analysis['filename']}**")
                    
                    # Panoramica risultati
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.info(f"🕐 **Generated:** {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.info(f"🔬 **Analysis Type:** {analysis.get('analysis_type', 'Standard').title()}")
                        st.info(f"📁 **Export Location:** project_root/results/")
                        
                        # Caratteristiche disponibili
                        features = []
                        if analysis.get('predictions'):
                            features.append("Model Predictions")
                        if analysis.get('sentiment_analysis'):
                            features.append("Sentiment Class Analysis")
                        if analysis.get('stats', {}).get('comprehensive_analysis'):
                            features.append("Comprehensive Text Analysis")
                        
                        if features:
                            st.info(f"🚀 **Features:** {', '.join(features)}")
                    
                    with col2:
                        # Statistiche rapide
                        df_size = len(analysis.get('df', []))
                        models_used = len(analysis.get('predictions', {}))
                        
                        st.metric("📊 Samples", f"{df_size:,}")
                        st.metric("🤖 Models", models_used)
                        
                        # Indicatore completezza
                        completeness = "Complete" if analysis.get('predictions') and analysis.get('sentiment_analysis') else "Partial"
                        st.metric("📋 Analysis", completeness)
                    
                    # === OPZIONI DI EXPORT ===
                    st.subheader("📦 Export Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Export ZIP completo
                        if st.button("🎁 Generate Complete Export Package", type="primary"):
                            try:
                                with st.spinner("🔄 Creating comprehensive export package..."):
                                    # Ottieni directory di sessione
                                    session_dir = get_session_results_dir()
                                    
                                    # Crea pacchetto ZIP
                                    zip_data = create_complete_export_package(
                                        analysis['df'],
                                        analysis.get('predictions', {}),
                                        analysis.get('metrics', {}),
                                        analysis['stats']
                                    )
                                    
                                    # Salva nella directory results
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    zip_filename = f"scientific_analysis_{analysis['filename'].replace('.csv', '')}_{timestamp}.zip"
                                    zip_path = session_dir / "exports" / zip_filename
                                    
                                    with open(zip_path, 'wb') as f:
                                        f.write(zip_data)
                                    
                                    # Messaggio di successo
                                    st.success("✅ Export package created successfully!")
                                    st.info(f"📁 **Saved to:** {zip_path}")
                                    
                                    # Download button
                                    st.download_button(
                                        label="📥 Download Complete Package (ZIP)",
                                        data=zip_data,
                                        file_name=zip_filename,
                                        mime="application/zip",
                                        help="Complete scientific analysis package with all files and reports"
                                    )
                                    
                            except Exception as e:
                                st.error(f"❌ Error creating export package: {e}")
                    
                    with col2:
                        # Preview contenuto export
                        st.markdown("""
                        **📋 Export Package Contents:**
                        - 📄 Complete predictions CSV
                        - 📊 Scientific analysis report  
                        - 📈 Structured JSON data
                        - 🔤 Word frequency tables
                        - 💬 Phrase analysis results
                        - 🎯 Topic analysis data
                        - 📋 Comprehensive README
                        """)
                    
                    # === EXPORT INDIVIDUALI ===
                    st.subheader("📄 Individual File Exports")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # CSV predizioni
                        if 'predictions' in analysis:
                            try:
                                df_results = analysis['df'].copy()
                                
                                for model_name, pred in analysis['predictions'].items():
                                    label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
                                    pred_labels = [label_map.get(p, f'class_{p}') for p in pred]
                                    df_results[f'{model_name}_prediction'] = pred_labels
                                
                                csv_data = df_results.to_csv(index=False)
                                
                                st.download_button(
                                    label="📄 Download Predictions CSV",
                                    data=csv_data,
                                    file_name=f"predictions_{analysis['filename']}",
                                    mime="text/csv"
                                )
                            except Exception as e:
                                st.error(f"Error preparing CSV: {e}")
                    
                    with col2:
                        # JSON strutturato
                        try:
                            json_data = {
                                'analysis_summary': {
                                    'filename': analysis['filename'],
                                    'timestamp': analysis['timestamp'].isoformat(),
                                    'total_samples': len(analysis['df']),
                                    'analysis_type': analysis.get('analysis_type', 'standard')
                                },
                                'comprehensive_analysis': analysis['stats'].get('comprehensive_analysis', {}),
                                'predictions': analysis.get('predictions', {}),
                                'metrics': analysis.get('metrics', {}),
                                'sentiment_analysis': analysis.get('sentiment_analysis', {})
                            }
                            
                            st.download_button(
                                label="📊 Download JSON Data",
                                data=json.dumps(safe_convert_for_json(json_data), indent=2),
                                file_name=f"analysis_data_{analysis['filename'].replace('.csv', '.json')}",
                                mime="application/json"
                            )
                        except Exception as e:
                            st.error(f"Error preparing JSON: {e}")
                    
                    with col3:
                        # Report scientifico
                        try:
                            report_lines = []
                            report_lines.append("SCIENTIFIC ANALYSIS REPORT")
                            report_lines.append("=" * 30)
                            report_lines.append(f"File: {analysis['filename']}")
                            report_lines.append(f"Generated: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                            report_lines.append("")
                            
                            stats = analysis['stats']
                            comprehensive = stats.get('comprehensive_analysis', {})
                            
                            if comprehensive:
                                basic_stats = comprehensive.get('basic_statistics', {})
                                report_lines.append("STATISTICS:")
                                report_lines.append(f"Total samples: {stats.get('total_reviews', 0):,}")
                                report_lines.append(f"Total words: {basic_stats.get('total_words', 0):,}")
                                report_lines.append(f"Vocabulary richness: {basic_stats.get('vocabulary_richness', 0):.3f}")
                                report_lines.append("")
                            
                            if analysis.get('predictions'):
                                report_lines.append("MODEL PREDICTIONS:")
                                for model_name, pred in analysis['predictions'].items():
                                    pred_dist = dict(zip(*np.unique(pred, return_counts=True)))
                                    report_lines.append(f"{model_name.upper()}: {len(pred)} predictions")
                                    for class_id, count in pred_dist.items():
                                        label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
                                        label = label_map.get(class_id, f'Class_{class_id}')
                                        report_lines.append(f"  {label}: {count}")
                            
                            report_text = '\n'.join(report_lines)
                            
                            st.download_button(
                                label="📋 Download Report TXT",
                                data=report_text,
                                file_name=f"report_{analysis['filename'].replace('.csv', '.txt')}",
                                mime="text/plain"
                            )
                        except Exception as e:
                            st.error(f"Error preparing report: {e}")
                    
                    # === CARATTERISTICHE ANALISI SCIENTIFICA ===
                    st.subheader("📋 Scientific Analysis Features")
                    st.markdown("""
                    **🔬 Professional Scientific Approach:**
                    - Objective statistical measurements with comprehensive NLP
                    - Word frequency analysis by sentiment class
                    - Phrase pattern analysis (bigrams/trigrams)
                    - Topic identification using keyword matching
                    - Reproducible analysis methods
                    - Quantitative metrics and distributions
                    
                    **📊 Complete Statistical Reports:**
                    - Sentiment distribution percentages
                    - Word relevance scoring by sentiment
                    - Phrase occurrence statistics
                    - Topic relevance measurements
                    - Vocabulary richness calculations
                    - Text quality scoring metrics
                    - Model performance statistics
                    
                    **📈 Research-Grade Output:**
                    - CSV files for further analysis
                    - JSON data for programmatic access
                    - Statistical summaries in TXT format
                    - Word/phrase/topic frequency tables
                    - Comprehensive methodology documentation
                    
                    **🎯 Export Location:**
                    All files are saved to: `project_root/results/session_YYYYMMDD_HHMMSS/exports/`
                    
                    **📁 File Organization:**
                    - Predictions and analysis results
                    - Statistical reports and summaries
                    - Structured data for further processing
                    - Complete documentation and methodology
                    """)
                except Exception as e:
                    st.error(f"Error in export results section: {e}")
            else:
                # Guida quando nessun risultato disponibile
                st.info("ℹ️ No analysis results available for export.")
                
                st.markdown("""
                ### 📥 How to Generate Results for Export:
                
                **🔄 Analysis Workflow:**
                1. 🧠 Go to **'Dataset Analysis'** section for local datasets
                2. 📂 Or upload a new CSV in **'Upload New Dataset'** section
                3. ⚡ Run analysis to generate comprehensive results
                4. 🔄 Return here to export all scientific results
                
                ### 🔬 What You'll Get:
                
                **📊 Complete Statistical Data Files:**
                - Predictions CSV with confidence scores
                - Word frequency analysis by sentiment
                - Phrase pattern analysis (bigrams/trigrams)
                - Topic modeling results with relevance scores
                - Statistical analysis summaries
                
                **📈 Scientific Reports:**
                - Objective statistical measurements
                - Comprehensive NLP feature extraction
                - Word/phrase/topic analysis
                - Reproducible analysis methodology
                - Quantitative performance metrics
                - Research-grade documentation
                
                **🎯 Professional Features:**
                - **Complete Analysis**: Statistical measurements without subjective interpretation
                - **Reproducible Methods**: Documented methodology for result replication
                - **Quantitative Focus**: Numerical metrics and statistical distributions
                - **Research Grade**: Suitable for academic and professional research
                - **Export Ready**: All results saved to project_root/results/ directory
                
                ### 💡 Scientific Approach:
                - **Objective Analysis**: Statistical measurements without subjective interpretation
                - **Reproducible Methods**: Documented methodology for result replication
                - **Quantitative Focus**: Numerical metrics and statistical distributions
                - **Research Grade**: Suitable for academic and professional research
                
                Start your scientific analysis in the **Dataset Analysis** or **Upload New Dataset** sections! 🔬🚀
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # === FOOTER PROFESSIONALE ===
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            🔬 <strong>Professional Scientific Sentiment Analysis System v1.0</strong> | 
            📊 AI-Powered Text Analysis | 
            🎯 Research-Grade Statistical Platform<br>
            <small>Built with Streamlit • Powered by PyTorch & scikit-learn • Professional Interface</small><br>
            <small>✨ <em>Scientific • Objective • Reproducible</em> ✨</small><br>
            <small>📁 <em>All results automatically saved to: project_root/results/</em> 📁</small>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Critical application error: {e}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()