#!/usr/bin/env python3
"""
Sistema di Analisi Sentiment - GUI Finale Scientifica - FIXED IMPORT VERSION
Interfaccia professionale con 3 sezioni principali per analisi rigorosa del sentiment.

🔧 FIXES APPLIED:
- ✅ Fixed import pipeline_runner functions with proper error handling
- ✅ Button callbacks now call REAL pipeline functions that exist
- ✅ Creates proper results/session_<timestamp>/ structure
- ✅ Enhanced error handling for missing dependencies
- ✅ Improved path detection and logging
- ✅ Real progress tracking and result feedback

CARATTERISTICHE FINALI:
- 📊 Dataset Analysis: Analisi completa unificata con pipeline REALE
- 📂 Upload New Dataset: Caricamento e analisi con pipeline REALE
- 📥 Export Results: Download completo di risultati e report professionali
- 🔬 Approccio scientifico con pipeline completa backend
- 📊 Grafici professionali + risultati autentici dalla pipeline
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

# === CONFIGURAZIONE PERCORSI CORRETTI ===
try:
    CURRENT_FILE = Path(__file__).resolve()
    PROJECT_ROOT = CURRENT_FILE.parent
except:
    PROJECT_ROOT = Path.cwd()

# Percorsi del progetto - CORRETTI
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed" 
EMBEDDINGS_DATA_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"  # CORRETTO: models è sotto results/
REPORTS_DIR = RESULTS_DIR / "reports"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Aggiungi scripts al path per import
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# === 🔧 FIXED: IMPORT PIPELINE RUNNER FUNCTIONS CON ERROR HANDLING ===
try:
    from scripts.pipeline_runner import (
        run_complete_csv_analysis, 
        run_dataset_analysis, 
        PipelineRunner
    )
    PIPELINE_AVAILABLE = True
    st.success("✅ Pipeline functions loaded successfully!")
except ImportError as e:
    PIPELINE_AVAILABLE = False
    st.error(f"❌ Pipeline import failed: {e}")
    st.info("🔧 Make sure pipeline_runner.py is in scripts/ directory")
    
    # Provide fallback functions to prevent crashes
    def run_complete_csv_analysis(csv_path, text_column='text', label_column='label'):
        return {
            'success': False,
            'error': 'Pipeline functions not available',
            'fallback': True
        }
    
    def run_dataset_analysis(csv_path):
        return {
            'success': False,
            'error': 'Pipeline functions not available',
            'fallback': True
        }

# === IMPORT SICURO DEGLI UTILS ===
try:
    from scripts.enhanced_utils_unified import *
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    # Fallback: definisci funzioni essenziali
    def clean_text(text):
        if not isinstance(text, str):
            text = str(text)
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

# Configurazione pagina
st.set_page_config(
    page_title="Scientific Sentiment Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS scientifico e professionale
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: linear-gradient(90deg, #f8f9fa, #e9ecef);
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    
    .scientific-section {
        background: #ffffff;
        padding: 2rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1.5rem 0;
    }
    
    .analysis-header {
        font-size: 1.4rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    
    .results-table {
        background: #fafafa;
        border-radius: 6px;
        padding: 1rem;
        border: 1px solid #ddd;
    }
    
    .export-section {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 2rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }

    .pipeline-progress {
        background: #e3f2fd;
        color: #0d47a1;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# === DEFINIZIONE ARCHITETTURA MLP ===
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

# === CARICAMENTO SICURO MODELLI ===
@st.cache_resource
def load_embedding_model():
    """Carica il modello di embedding con fallback"""
    try:
        # Prima prova modello locale
        local_model_dir = PROJECT_ROOT / "models" / "minilm-l6-v2"
        if local_model_dir.exists():
            model = SentenceTransformer(str(local_model_dir))
            return model, "Local model loaded successfully"
        else:
            # Fallback a modello online
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model, "Online model loaded successfully"
    except Exception as e:
        return None, f"Error loading embedding model: {e}"

@st.cache_resource
def load_trained_models():
    """Carica i modelli addestrati da /results/models/ con gestione graceful"""
    models = {
        'mlp': None,
        'svm': None,
        'status': {
            'mlp': 'not_found',
            'svm': 'not_found'
        },
        'messages': []
    }
    
    # === CARICAMENTO MLP ===
    mlp_paths = [
        MODELS_DIR / "mlp_model.pth",
        MODELS_DIR / "mlp_model_complete.pth"
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
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                
                models['mlp'] = model
                models['status']['mlp'] = 'loaded'
                models['messages'].append(f"MLP model loaded from {mlp_path.name}")
                break
                
            except Exception as e:
                models['messages'].append(f"Warning: Could not load MLP from {mlp_path.name}")
                continue
    
    # === CARICAMENTO SVM ===
    svm_paths = [
        MODELS_DIR / "svm_model.pkl"
    ]
    
    for svm_path in svm_paths:
        if svm_path.exists():
            try:
                models['svm'] = joblib.load(svm_path)
                models['status']['svm'] = 'loaded'
                models['messages'].append(f"SVM model loaded from {svm_path.name}")
                break
            except Exception as e:
                models['messages'].append(f"Warning: Could not load SVM from {svm_path.name}")
                continue
    
    # Messaggio finale
    loaded_count = sum(1 for status in models['status'].values() if status == 'loaded')
    if loaded_count == 0:
        models['messages'].append("No models found - analysis will continue with embeddings only")
    elif loaded_count == 1:
        models['messages'].append("Single model available - analysis will continue normally")
    else:
        models['messages'].append("All models loaded successfully")
    
    return models

# === CARICAMENTO DATASET CON VALIDAZIONE ===
@st.cache_data
def load_main_dataset():
    """Carica il dataset principale con validazione"""
    dataset_paths = [
        PROCESSED_DATA_DIR / "train.csv",
        PROCESSED_DATA_DIR / "test.csv", 
        PROCESSED_DATA_DIR / "val.csv",
        DATA_DIR / "raw" / "imdb_raw.csv"
    ]
    
    for path in dataset_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if validate_dataset(df)[0]:  # Se validazione passa
                    return df, str(path), None
                else:
                    continue
            except Exception as e:
                continue
    
    return None, None, "No valid dataset found in standard locations"

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, str]:
    """Valida che il dataset abbia le colonne necessarie"""
    if df is None or df.empty:
        return False, "Dataset is empty"
    
    # Colonne di testo accettate
    text_columns = ['review', 'text', 'content', 'comment', 'message', 'description', 'body', 'post']
    
    # Trova colonna di testo
    text_col = None
    for col in text_columns:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        return False, f"No text column found. Required: {text_columns}. Found: {list(df.columns)}"
    
    # Verifica che ci siano dati validi
    valid_texts = df[text_col].fillna('').astype(str).str.strip()
    if (valid_texts == '').all():
        return False, f"Column '{text_col}' contains no valid text data"
    
    return True, f"Valid dataset with text column: '{text_col}'"

# === 🔧 FIXED: FUNZIONI WRAPPER PER CHIAMARE PIPELINE REALE ===
def run_real_pipeline_analysis(dataset_path: str) -> Dict:
    """
    🚀 FIXED: Chiama la pipeline REALE invece di analisi fake inline
    """
    if not PIPELINE_AVAILABLE:
        return {
            'success': False,
            'error': 'Pipeline functions not available. Check import errors.',
            'fake_analysis': True
        }
    
    try:
        # Chiama la pipeline REALE
        st.info("🔄 Executing REAL pipeline analysis...")
        
        # Usa la funzione di pipeline_runner.py
        results = run_complete_csv_analysis(
            csv_path=dataset_path,
            text_column='text',
            label_column='label'
        )
        
        if results.get('success', False):
            st.success("✅ REAL pipeline completed successfully!")
            
            # Estrai informazioni dalla pipeline
            final_results = results.get('final_results', {})
            session_dir = final_results.get('session_directory')
            
            if session_dir:
                st.info(f"📁 Results saved to: {session_dir}")
                
                # Verifica struttura directory creata
                session_path = Path(session_dir)
                subdirs = ['processed', 'embeddings', 'models', 'plots', 'reports', 'logs']
                
                created_dirs = []
                for subdir in subdirs:
                    if (session_path / subdir).exists():
                        created_dirs.append(subdir)
                
                if created_dirs:
                    st.success(f"📂 Created directories: {', '.join(created_dirs)}")
                else:
                    st.warning("⚠️ Expected directory structure not found")
            
            return results
        else:
            st.error(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
            return results
            
    except Exception as e:
        st.error(f"❌ Pipeline execution error: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'fake_analysis': True
        }

def run_real_uploaded_csv_analysis(uploaded_file) -> Dict:
    """
    🚀 FIXED: Analisi CSV caricato con pipeline REALE
    """
    if not PIPELINE_AVAILABLE:
        return {
            'success': False,
            'error': 'Pipeline functions not available',
            'fake_analysis': True
        }
    
    try:
        # Salva temporaneamente il file
        temp_path = PROJECT_ROOT / f"temp_upload_{uploaded_file.name}"
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        st.info("🔄 Executing REAL pipeline for uploaded CSV...")
        
        # Chiama pipeline REALE
        results = run_complete_csv_analysis(
            csv_path=str(temp_path),
            text_column='text',  # Auto-detect in pipeline
            label_column='label'  # Auto-detect in pipeline
        )
        
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()
        
        if results.get('success', False):
            st.success("✅ REAL pipeline completed for uploaded CSV!")
            
            final_results = results.get('final_results', {})
            session_dir = final_results.get('session_directory')
            
            if session_dir:
                st.info(f"📁 Results saved to: {session_dir}")
            
            return results
        else:
            st.error(f"❌ Uploaded CSV pipeline failed: {results.get('error', 'Unknown error')}")
            return results
            
    except Exception as e:
        st.error(f"❌ Uploaded CSV analysis error: {str(e)}")
        
        # Cleanup temp file on error
        temp_path = PROJECT_ROOT / f"temp_upload_{uploaded_file.name}"
        if temp_path.exists():
            temp_path.unlink()
        
        return {
            'success': False,
            'error': str(e),
            'fake_analysis': True
        }

# === FUNZIONI VISUALIZZAZIONE RISULTATI PIPELINE ===
def display_pipeline_results(pipeline_results: Dict):
    """
    Mostra i risultati REALI dalla pipeline invece di analisi fake
    """
    if not pipeline_results.get('success', False):
        st.error("❌ No valid pipeline results to display")
        return
    
    final_results = pipeline_results.get('final_results', {})
    
    # === METRICHE MODELLI ===
    metrics = final_results.get('metrics', {})
    if metrics:
        st.markdown('<div class="analysis-header">Model Performance Metrics</div>', unsafe_allow_html=True)
        
        cols = st.columns(len(metrics))
        for i, (model_name, model_metrics) in enumerate(metrics.items()):
            with cols[i]:
                st.subheader(f"{model_name.upper()} Model")
                
                accuracy = model_metrics.get('accuracy', 0)
                f1_score = model_metrics.get('f1_score', 0)
                precision = model_metrics.get('precision', 0)
                recall = model_metrics.get('recall', 0)
                
                st.metric("Accuracy", f"{accuracy:.3f}")
                st.metric("F1-Score", f"{f1_score:.3f}")
                st.metric("Precision", f"{precision:.3f}")
                st.metric("Recall", f"{recall:.3f}")
    
    # === PREDIZIONI ===
    predictions = final_results.get('predictions', {})
    if predictions:
        st.markdown('<div class="analysis-header">Prediction Results</div>', unsafe_allow_html=True)
        
        for model_name, pred_list in predictions.items():
            if pred_list:
                # Converti predictions in distribution
                unique_preds, counts = np.unique(pred_list, return_counts=True)
                total = len(pred_list)
                
                label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
                sentiment_df = pd.DataFrame({
                    'Sentiment': [label_map.get(pred, f'Class_{pred}') for pred in unique_preds],
                    'Count': counts,
                    'Percentage': [(count/total)*100 for count in counts]
                })
                
                st.subheader(f"{model_name.upper()} Predictions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_bar = px.bar(
                        sentiment_df,
                        x='Sentiment',
                        y='Count',
                        title=f"{model_name.upper()} Distribution",
                        text='Percentage',
                        color='Sentiment',
                        color_discrete_map={'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#6c757d'}
                    )
                    fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    fig_pie = px.pie(
                        sentiment_df,
                        values='Count',
                        names='Sentiment',
                        title=f"{model_name.upper()} Distribution",
                        color_discrete_map={'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#6c757d'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
    
    # === INSIGHTS ===
    insights = final_results.get('insights', [])
    if insights:
        st.markdown('<div class="analysis-header">Intelligent Insights</div>', unsafe_allow_html=True)
        
        for i, insight in enumerate(insights, 1):
            st.info(f"💡 **Insight {i}**: {insight}")
    
    # === INFO SESSIONE ===
    session_dir = final_results.get('session_directory')
    if session_dir:
        st.markdown('<div class="analysis-header">Session Information</div>', unsafe_allow_html=True)
        
        st.success(f"📁 **Session Directory**: `{session_dir}`")
        
        # Verifica directory create
        session_path = Path(session_dir)
        if session_path.exists():
            subdirs = ['processed', 'embeddings', 'models', 'plots', 'reports', 'logs']
            created_dirs = [d for d in subdirs if (session_path / d).exists()]
            
            if created_dirs:
                st.info(f"📂 **Created Directories**: {', '.join(created_dirs)}")
            
            # Lista file creati
            all_files = []
            for root, dirs, files in os.walk(session_path):
                for file in files:
                    rel_path = Path(root).relative_to(session_path) / file
                    all_files.append(str(rel_path))
            
            if all_files:
                st.info(f"📄 **Files Generated**: {len(all_files)} files")
                
                with st.expander("View generated files"):
                    for file_path in sorted(all_files):
                        st.text(f"• {file_path}")

# === FUNZIONI LEGACY (mantenute per compatibilità) ===
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

def ensure_result_directories():
    """Crea automaticamente le directory mancanti per i risultati"""
    try:
        base_dirs = [
            RESULTS_DIR,
            MODELS_DIR,
            REPORTS_DIR,
            RESULTS_DIR / "plots",
            RESULTS_DIR / "exports"
        ]
        
        for directory in base_dirs:
            directory.mkdir(parents=True, exist_ok=True)
        
        return True, "Result directories verified"
    except Exception as e:
        return False, f"Error creating directories: {e}"

def create_timestamp_session():
    """Crea un timestamp unico per la sessione"""
    if 'session_timestamp' not in st.session_state:
        st.session_state.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return st.session_state.session_timestamp

def main():
    """Applicazione principale Streamlit - Versione FIXED con pipeline reale"""
    
    try:
        # Header principale
        st.markdown('<div class="main-header">Scientific Sentiment Analysis System - REAL PIPELINE VERSION</div>', 
                    unsafe_allow_html=True)
        
        # === CARICAMENTO INIZIALE SICURO ===
        with st.spinner("Loading system resources..."):
            # Verifica directory
            dir_ok, dir_msg = ensure_result_directories()
            
            # Carica embedding model
            embedding_model, embed_msg = load_embedding_model()
            
            # Carica modelli ML
            models = load_trained_models()
            
            # Carica dataset principale
            main_df, main_dataset_path, dataset_error = load_main_dataset()

            # Analisi automatica del test set all'avvio
            auto_pipeline = None
            auto_analysis = None
            default_test = PROCESSED_DATA_DIR / "test.csv"
            if PIPELINE_AVAILABLE and default_test.exists():
                auto_pipeline = run_complete_csv_analysis(str(default_test))
                auto_analysis = run_dataset_analysis(str(default_test))
                st.session_state['auto_results'] = {
                    'pipeline': auto_pipeline,
                    'analysis': auto_analysis,
                    'path': str(default_test)
                }
        
        # === SIDEBAR INFORMAZIONI ===
        with st.sidebar:
            st.header("System Information")
            
            # Pipeline status
            st.subheader("Pipeline Status")
            if PIPELINE_AVAILABLE:
                st.success("✅ Real Pipeline Available")
            else:
                st.error("❌ Pipeline Not Available")
                st.info("Check pipeline_runner.py import")
            
            # Utils status
            st.subheader("Utils Status")
            if UTILS_AVAILABLE:
                st.success("✅ Enhanced Utils Available")
            else:
                st.warning("⚠️ Enhanced Utils Not Available")
            
            # Percorsi
            st.subheader("Project Structure")
            st.text(f"Root: {PROJECT_ROOT}")
            st.text(f"Models: {MODELS_DIR}")
            st.text(f"Results: {RESULTS_DIR}")
            
            # Status modelli
            st.subheader("Model Status")
            for model_name, status in models['status'].items():
                if status == 'loaded':
                    st.success(f"{model_name.upper()}: Available")
                else:
                    st.warning(f"{model_name.upper()}: Not Found")
            
            # Messaggi sistema
            st.subheader("System Messages")
            if embedding_model:
                st.info(embed_msg)
            else:
                st.error(embed_msg)
            
            for msg in models['messages']:
                st.info(msg)
            
            if not dir_ok:
                st.error(dir_msg)
            
            # Quick actions
            st.subheader("Quick Actions")
            if st.button("Refresh System"):
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()
        
        # === TABS PRINCIPALI ===
        tab1, tab2, tab3 = st.tabs([
            "📊 Dataset Analysis", 
            "🆕 Upload New Dataset", 
            "📥 Export Results"
        ])
        
        # === TAB 1: DATASET ANALYSIS - FIXED ===
        with tab1:
            st.markdown('<div class="scientific-section">', unsafe_allow_html=True)
            st.header("Comprehensive Dataset Analysis with REAL Pipeline")

            if 'auto_results' in st.session_state:
                auto_pipeline = st.session_state['auto_results']['pipeline']
                auto_analysis = st.session_state['auto_results']['analysis']
                st.subheader("Automatic Analysis of test.csv")
                display_pipeline_results(auto_pipeline)

                try:
                    if auto_analysis and 'text_analysis' in auto_analysis:
                        col_used = auto_analysis['text_analysis']['column_used']
                        words = ' '.join(main_df[col_used].astype(str)).lower().split()
                        common_words = Counter(words).most_common(20)
                        word_df = pd.DataFrame(common_words, columns=['Word', 'Count'])
                        fig_words = px.bar(word_df, x='Word', y='Count', title='Most Used Words')
                        st.plotly_chart(fig_words, use_container_width=True)
                except Exception:
                    pass
            
            if not PIPELINE_AVAILABLE:
                st.error("❌ Real pipeline not available. Check pipeline_runner.py import.")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            
            if main_df is not None:
                st.markdown(f'<div class="success-box">Main dataset loaded successfully from: {main_dataset_path}</div>', unsafe_allow_html=True)
                
                # Validazione dataset
                is_valid, validation_msg = validate_dataset(main_df)
                if not is_valid:
                    st.error(f"Dataset validation failed: {validation_msg}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return
                
                # Trova colonna di testo
                text_columns = ['review', 'text', 'content', 'comment', 'message', 'description']
                text_col = None
                for col in text_columns:
                    if col in main_df.columns:
                        text_col = col
                        break
                
                if text_col:
                    # === TABELLA DATASET COMPLETA ===
                    st.subheader("Complete Dataset Table")
                    
                    # Controlli ricerca
                    col_search, col_filter = st.columns([2, 1])
                    with col_search:
                        search_term = st.text_input("Search in dataset:", placeholder="Enter search term...")
                    with col_filter:
                        show_rows = st.selectbox("Rows to display:", [50, 100, 200, 500, "All"], index=1)
                    
                    display_df = main_df
                    if search_term:
                        try:
                            mask = main_df[text_col].astype(str).str.contains(search_term, case=False, na=False)
                            if mask.any():
                                display_df = main_df[mask]
                                st.info(f"Found {len(display_df)} matching records")
                            else:
                                st.warning("No matches found")
                        except Exception as e:
                            st.warning("Search error occurred")
                    
                    # Gestione visualizzazione righe
                    if show_rows != "All":
                        display_df = display_df.head(int(show_rows))
                    
                    # Tabella scrollabile
                    st.dataframe(
                        display_df, 
                        use_container_width=True,
                        height=600
                    )
                    
                    # === 🚀 FIXED: AVVIA PIPELINE REALE ===
                    if st.button("🚀 Start Complete Analysis", type="primary"):
                        st.markdown('<div class="pipeline-progress">🔄 <strong>Executing REAL Scientific Pipeline...</strong><br>This will invoke: embed_dataset.py → train_mlp.py → train_svm.py → report.py</div>', unsafe_allow_html=True)
                        
                        # Chiama la pipeline REALE
                        pipeline_results = run_real_pipeline_analysis(main_dataset_path)
                        
                        if pipeline_results.get('success', False):
                            st.markdown('<div class="success-box">✅ <strong>Analysis complete!</strong> Real pipeline executed successfully.</div>', unsafe_allow_html=True)
                            
                            # Salva risultati in sessione
                            st.session_state['current_analysis'] = {
                                'pipeline_results': pipeline_results,
                                'df': main_df,
                                'filename': Path(main_dataset_path).name,
                                'timestamp': datetime.now(),
                                'analysis_type': 'real_pipeline_unified',
                                'session_directory': pipeline_results.get('final_results', {}).get('session_directory')
                            }
                            
                            # Mostra risultati REALI
                            display_pipeline_results(pipeline_results)
                            
                        else:
                            st.error("❌ Real pipeline execution failed")
                            if 'error' in pipeline_results:
                                st.error(f"Error: {pipeline_results['error']}")
                            
                            # Fallback message
                            st.warning("💡 Check logs and ensure all dependencies are installed")
                else:
                    st.error(f"Text column not found. Available columns: {list(main_df.columns)}")
                    st.info("Required: 'review', 'text', 'content', 'comment', 'message', or 'description'")
            else:
                if dataset_error:
                    st.error(dataset_error)
                else:
                    st.info("No main dataset found.")
                
                st.markdown("""
                ### Required Dataset Format:
                
                Your CSV should contain:
                - **Text column**: 'review', 'text', 'content', 'comment', 'message', or 'description'  
                - **Label column** (optional): 'sentiment', 'label', 'class', or 'target'
                
                Example:
                ```
                text,sentiment
                "This movie is fantastic!",positive
                "I didn't like it",negative
                "It was okay",neutral
                ```
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # === TAB 2: UPLOAD NEW DATASET - FIXED ===
        with tab2:
            st.markdown('<div class="scientific-section">', unsafe_allow_html=True)
            st.header("Upload New Dataset (CSV) with REAL Pipeline")
            
            if not PIPELINE_AVAILABLE:
                st.error("❌ Real pipeline not available. Check pipeline_runner.py import.")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            
            # Upload file
            st.subheader("CSV File Upload")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Choose a CSV file for analysis",
                    type=['csv'],
                    help="CSV must contain a text column"
                )
            
            with col2:
                if uploaded_file:
                    try:
                        file_size = len(uploaded_file.getvalue()) / (1024**2)
                        st.info(f"File: {uploaded_file.name}")
                        st.info(f"Size: {file_size:.2f} MB")
                    except Exception:
                        st.info(f"File: {uploaded_file.name}")
            
            if uploaded_file is not None:
                st.success(f"File uploaded: {uploaded_file.name}")
                
                # === 🚀 FIXED: PIPELINE REALE PER CSV CARICATO ===
                if st.button("🚀 Start Analysis", type="primary"):
                    st.markdown('<div class="pipeline-progress">🔄 <strong>Executing REAL Pipeline for Uploaded CSV...</strong><br>Processing through complete sentiment analysis pipeline</div>', unsafe_allow_html=True)
                    
                    # Chiama pipeline REALE per CSV caricato
                    pipeline_results = run_real_uploaded_csv_analysis(uploaded_file)
                    
                    if pipeline_results.get('success', False):
                        st.markdown('<div class="success-box">✅ <strong>Analysis complete!</strong> Uploaded CSV processed through real pipeline.</div>', unsafe_allow_html=True)
                        
                        # Salva risultati in sessione
                        st.session_state['current_analysis'] = {
                            'pipeline_results': pipeline_results,
                            'filename': uploaded_file.name,
                            'timestamp': datetime.now(),
                            'analysis_type': 'real_pipeline_uploaded',
                            'session_directory': pipeline_results.get('final_results', {}).get('session_directory')
                        }
                        
                        # Mostra risultati REALI
                        display_pipeline_results(pipeline_results)
                        
                    else:
                        st.error("❌ Real pipeline execution failed for uploaded CSV")
                        if 'error' in pipeline_results:
                            st.error(f"Error: {pipeline_results['error']}")
                        
                        st.warning("💡 Ensure CSV has proper text columns and format")
            else:
                st.markdown("""
                ### CSV Analysis Features (REAL PIPELINE):
                
                **Complete Analysis**
                - Real text processing and embedding generation
                - Authentic sentiment model training (MLP + SVM)
                - Comprehensive statistical analysis and insights
                - Scientific visualization generation
                - Results saved to `results/session_<timestamp>/`
                
                ### Supported File Formats:
                
                **Required Columns:**
                - Text data: `review`, `text`, `content`, `comment`, `message`, `description`
                - Labels (optional): `sentiment`, `label`, `class`, `target`
                
                **Example CSV Structure:**
                ```
                text,sentiment
                "Great product!",positive
                "Poor quality",negative
                "Average product",neutral
                ```
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # === TAB 3: EXPORT RESULTS - UPDATED ===
        with tab3:
            st.markdown('<div class="export-section">', unsafe_allow_html=True)
            st.header("Export Results")
            
            if 'current_analysis' in st.session_state:
                try:
                    analysis = st.session_state['current_analysis']
                    
                    st.success(f"Analysis results available for: **{analysis['filename']}**")
                    
                    # Panoramica risultati
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.info(f"Generated: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.info(f"Analysis Type: {analysis.get('analysis_type', 'Standard').title()}")
                        
                        # Session directory se disponibile
                        session_dir = analysis.get('session_directory')
                        if session_dir:
                            st.info(f"Session Directory: {session_dir}")
                        else:
                            st.info(f"Export Location: {RESULTS_DIR}")
                        
                        # Pipeline results
                        pipeline_results = analysis.get('pipeline_results', {})
                        if pipeline_results.get('success', False):
                            st.success("✅ Real pipeline results available")
                            
                            final_results = pipeline_results.get('final_results', {})
                            features = []
                            if final_results.get('predictions'):
                                features.append("Model Predictions")
                            if final_results.get('metrics'):
                                features.append("Performance Metrics")
                            if final_results.get('insights'):
                                features.append("Intelligent Insights")
                            
                            if features:
                                st.info(f"Features: {', '.join(features)}")
                        else:
                            st.warning("⚠️ Pipeline results incomplete")
                    
                    with col2:
                        pipeline_results = analysis.get('pipeline_results', {})
                        final_results = pipeline_results.get('final_results', {})
                        
                        # Metriche
                        predictions = final_results.get('predictions', {})
                        models_count = len(predictions) if predictions else 0
                        
                        total_predictions = 0
                        for model_preds in predictions.values():
                            if isinstance(model_preds, list):
                                total_predictions += len(model_preds)
                        
                        st.metric("Models Used", models_count)
                        st.metric("Total Predictions", total_predictions)
                        
                        insights_count = len(final_results.get('insights', []))
                        st.metric("Insights", insights_count)
                    
                    # Session directory info
                    session_dir = analysis.get('session_directory')
                    if session_dir and Path(session_dir).exists():
                        st.subheader("Session Directory Contents")
                        
                        session_path = Path(session_dir)
                        
                        # List directories created
                        subdirs = []
                        for item in session_path.iterdir():
                            if item.is_dir():
                                subdirs.append(item.name)
                        
                        if subdirs:
                            st.info(f"📂 Directories: {', '.join(sorted(subdirs))}")
                        
                        # List files created
                        all_files = []
                        for root, dirs, files in os.walk(session_path):
                            for file in files:
                                rel_path = Path(root).relative_to(session_path) / file
                                all_files.append(str(rel_path))
                        
                        if all_files:
                            st.info(f"📄 Total files: {len(all_files)}")

                            with st.expander("View all generated files"):
                                for file_path in sorted(all_files):
                                    st.text(f"• {file_path}")

                        # Download buttons for key results
                        final_results = pipeline_results.get('final_results', {})
                        download_map = {
                            'Predictions CSV': final_results.get('predictions_file'),
                            'Report PDF': final_results.get('report_pdf'),
                            'Summary TXT': final_results.get('summary_file')
                        }
                        log_file = session_path / 'pipeline.log'
                        if log_file.exists():
                            download_map['Log File'] = str(log_file)

                        if download_map:
                            st.subheader("Download Files")
                            for label, path in download_map.items():
                                if path and Path(path).exists():
                                    with open(path, 'rb') as f:
                                        st.download_button(
                                            label=f"Download {label}",
                                            data=f.read(),
                                            file_name=Path(path).name
                                        )
                    
                    # Export caratteristiche
                    st.subheader("Export Package Features")
                    st.markdown("""
                    **Real Pipeline Export Contents:**
                    - Complete model training results and checkpoints  
                    - Embedding matrices and metadata
                    - Performance metrics and evaluation reports
                    - Intelligent insights and analysis summaries
                    - Scientific visualization plots
                    - Structured session directory with proper organization
                    
                    **Professional Features:**
                    - Research-grade documentation and methodology
                    - Quantitative metrics with statistical significance
                    - Reproducible analysis with full traceability
                    - Export-ready for academic and industry presentation
                    
                    **Directory Structure:** `results/session_YYYYMMDD_HHMMSS/`
                    """)
                    
                except Exception as e:
                    st.error(f"Error in export section: {str(e)}")
            else:
                st.info("No analysis results available for export.")
                
                st.markdown("""
                ### How to Generate REAL Results:
                
                1. Go to **'📊 Dataset Analysis'** for local datasets
                2. Or upload a CSV in **'🆕 Upload New Dataset'**
                3. Click **'🚀 Start Complete Analysis'** or **'🚀 Start Analysis'**
                4. Wait for real pipeline execution (embed → train → predict → report)
                5. Return here for complete export package
                
                ### What You'll Get (REAL PIPELINE):
                
                - Actual model training and performance results
                - Real embeddings and model checkpoints
                - Authentic prediction accuracy and metrics
                - Scientific analysis with validated methodology
                - Complete session directory with all artifacts
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown(f"""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <strong>Scientific Sentiment Analysis System v2.0 - REAL PIPELINE</strong><br>
            <small>Research-Grade • Objective • Reproducible • {'✅ Pipeline Connected' if PIPELINE_AVAILABLE else '❌ Pipeline Disconnected'}</small><br>
            <small>Results automatically saved to: {PROJECT_ROOT}/results/session_&lt;timestamp&gt;/</small>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
