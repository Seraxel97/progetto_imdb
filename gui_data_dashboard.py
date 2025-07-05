#!/usr/bin/env python3
"""
Sistema di Analisi Sentiment - GUI Finale Scientifica - FIXED SESSION-BASED LOADING
Interfaccia professionale con caricamento automatico dei risultati dalle sessioni più recenti.

🔧 CRITICAL FIXES APPLIED:
- ✅ Fixed model loading to search in session directories (results/auto_analysis_*)
- ✅ Added automatic latest session detection and loading
- ✅ Enhanced result visualization from pipeline executions
- ✅ Fixed path resolution for session-based structure
- ✅ Improved error handling and fallback mechanisms

CARATTERISTICHE FINALI:
- 📊 Dataset Analysis: Analisi completa unificata con pipeline REALE
- 📂 Upload New Dataset: Caricamento e analisi con pipeline REALE
- 📥 Export Results: Download completo di risultati e report professionali
- 🔬 Approccio scientifico con pipeline completa backend
- 📊 Grafici professionali + risultati autentici dalla pipeline
- 🆕 Automatic session-based result loading
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
MODELS_DIR = RESULTS_DIR / "models"  # Legacy fallback
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
    
    .session-info {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# === 🔧 NEW: SESSION MANAGEMENT FUNCTIONS ===

def find_latest_session_directory():
    """🔧 NEW: Find the latest auto_analysis session directory"""
    try:
        results_dir = RESULTS_DIR
        
        if not results_dir.exists():
            return None
        
        # Find all auto_analysis directories
        session_dirs = []
        for item in results_dir.iterdir():
            if item.is_dir() and item.name.startswith("auto_analysis_"):
                session_dirs.append(item)
        
        if not session_dirs:
            return None
        
        # Sort by modification time (most recent first)
        session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return session_dirs[0]
        
    except Exception as e:
        st.error(f"Error finding latest session: {e}")
        return None

def find_session_models_and_data(session_dir=None):
    """🔧 NEW: Find models and data in session directory"""
    if session_dir is None:
        session_dir = find_latest_session_directory()
    
    if session_dir is None:
        return None, None, None, None
    
    session_dir = Path(session_dir)
    
    # Look for models directory
    models_candidates = [
        session_dir / "models",
        session_dir  # Sometimes models are in session root
    ]
    
    models_dir = None
    for candidate in models_candidates:
        if candidate.exists():
            # Check if it contains model files
            if any(candidate.glob("*.pth")) or any(candidate.glob("*.pkl")):
                models_dir = candidate
                break
    
    # Look for embeddings data
    embeddings_candidates = [
        session_dir / "embeddings",
        session_dir / "data" / "embeddings"
    ]
    
    embeddings_dir = None
    for candidate in embeddings_candidates:
        if candidate.exists() and any(candidate.glob("*.npy")):
            embeddings_dir = candidate
            break
    
    # Look for reports
    reports_candidates = [
        session_dir / "reports",
        session_dir / "models" / "reports",
        session_dir
    ]
    
    reports_dir = None
    for candidate in reports_candidates:
        if candidate.exists() and (any(candidate.glob("*.json")) or any(candidate.glob("*.txt"))):
            reports_dir = candidate
            break
    
    # Look for plots
    plots_candidates = [
        session_dir / "plots",
        session_dir / "models" / "plots",
        session_dir
    ]
    
    plots_dir = None
    for candidate in plots_candidates:
        if candidate.exists() and any(candidate.glob("*.png")):
            plots_dir = candidate
            break
    
    return models_dir, embeddings_dir, reports_dir, plots_dir

def load_session_info(session_dir):
    """🔧 NEW: Load information about a session"""
    if session_dir is None:
        return None
    
    session_dir = Path(session_dir)
    
    info = {
        'session_path': str(session_dir),
        'session_name': session_dir.name,
        'creation_time': datetime.fromtimestamp(session_dir.stat().st_ctime),
        'modification_time': datetime.fromtimestamp(session_dir.stat().st_mtime),
        'models_found': [],
        'reports_found': [],
        'plots_found': []
    }
    
    models_dir, embeddings_dir, reports_dir, plots_dir = find_session_models_and_data(session_dir)
    
    # Check for models
    if models_dir:
        for model_file in models_dir.glob("*.pth"):
            info['models_found'].append(f"MLP: {model_file.name}")
        for model_file in models_dir.glob("*.pkl"):
            info['models_found'].append(f"SVM: {model_file.name}")
    
    # Check for reports
    if reports_dir:
        for report_file in reports_dir.glob("*.json"):
            info['reports_found'].append(report_file.name)
        for report_file in reports_dir.glob("*.txt"):
            info['reports_found'].append(report_file.name)
    
    # Check for plots
    if plots_dir:
        for plot_file in plots_dir.glob("*.png"):
            info['plots_found'].append(plot_file.name)
    
    return info

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

# === 🔧 FIXED: CARICAMENTO SICURO MODELLI DA SESSIONI ===
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
def load_trained_models_from_session():
    """🔧 FIXED: Carica i modelli addestrati dalla sessione più recente"""
    models = {
        'mlp': None,
        'svm': None,
        'status': {
            'mlp': 'not_found',
            'svm': 'not_found'
        },
        'messages': [],
        'session_info': None
    }
    
    # Find latest session
    latest_session = find_latest_session_directory()
    if latest_session is None:
        models['messages'].append("No session directories found")
        return models
    
    models['session_info'] = load_session_info(latest_session)
    models['messages'].append(f"Using session: {latest_session.name}")
    
    # Find models in session
    models_dir, embeddings_dir, reports_dir, plots_dir = find_session_models_and_data(latest_session)
    
    if models_dir is None:
        models['messages'].append("No models directory found in session")
        return models
    
    models['messages'].append(f"Models directory: {models_dir}")
    
    # === CARICAMENTO MLP ===
    mlp_files = list(models_dir.glob("mlp_model*.pth"))
    
    for mlp_path in mlp_files:
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
            models['messages'].append(f"Warning: Could not load MLP from {mlp_path.name}: {str(e)}")
            continue
    
    # === CARICAMENTO SVM ===
    svm_files = list(models_dir.glob("svm_model*.pkl"))
    
    for svm_path in svm_files:
        try:
            models['svm'] = joblib.load(svm_path)
            models['status']['svm'] = 'loaded'
            models['messages'].append(f"SVM model loaded from {svm_path.name}")
            break
        except Exception as e:
            models['messages'].append(f"Warning: Could not load SVM from {svm_path.name}: {str(e)}")
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

def load_session_results():
    """🔧 NEW: Load results from the latest session"""
    latest_session = find_latest_session_directory()
    
    if latest_session is None:
        return None
    
    models_dir, embeddings_dir, reports_dir, plots_dir = find_session_models_and_data(latest_session)
    
    results = {
        'session_dir': str(latest_session),
        'session_name': latest_session.name,
        'models': {},
        'reports': {},
        'plots': {},
        'metrics': {}
    }
    
    # Load model metrics
    if reports_dir:
        # Load MLP metrics
        mlp_metrics_file = reports_dir / "mlp_metrics.json"
        if mlp_metrics_file.exists():
            try:
                with open(mlp_metrics_file, 'r', encoding='utf-8') as f:
                    results['metrics']['mlp'] = json.load(f)
            except Exception as e:
                st.warning(f"Could not load MLP metrics: {e}")
        
        # Load SVM metrics
        svm_metrics_file = reports_dir / "svm_metrics.json"
        if svm_metrics_file.exists():
            try:
                with open(svm_metrics_file, 'r', encoding='utf-8') as f:
                    results['metrics']['svm'] = json.load(f)
            except Exception as e:
                st.warning(f"Could not load SVM metrics: {e}")
        
        # Load evaluation report
        eval_report_file = reports_dir / "evaluation_report.json"
        if eval_report_file.exists():
            try:
                with open(eval_report_file, 'r', encoding='utf-8') as f:
                    results['reports']['evaluation'] = json.load(f)
            except Exception as e:
                st.warning(f"Could not load evaluation report: {e}")
    
    # Find plot files
    if plots_dir:
        for plot_file in plots_dir.glob("*.png"):
            model_type = "unknown"
            if "mlp" in plot_file.name.lower():
                model_type = "mlp"
            elif "svm" in plot_file.name.lower():
                model_type = "svm"
            
            if model_type not in results['plots']:
                results['plots'][model_type] = {}
            
            if "confusion" in plot_file.name.lower():
                results['plots'][model_type]['confusion_matrix'] = str(plot_file)
            elif "metrics" in plot_file.name.lower() or "performance" in plot_file.name.lower():
                results['plots'][model_type]['performance'] = str(plot_file)
            elif "history" in plot_file.name.lower():
                results['plots'][model_type]['training_history'] = str(plot_file)
    
    return results

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

# === 🔧 NEW: VISUALIZZAZIONE RISULTATI SESSIONE ===
def display_session_results():
    """🔧 NEW: Display results from the latest session"""
    session_results = load_session_results()
    
    if session_results is None:
        st.warning("⚠️ No session results found")
        return
    
    st.markdown('<div class="analysis-header">Latest Session Results</div>', unsafe_allow_html=True)
    
    # Session info
    st.markdown(f'<div class="session-info"><strong>Session:</strong> {session_results["session_name"]}<br><strong>Directory:</strong> {session_results["session_dir"]}</div>', unsafe_allow_html=True)
    
    # Model performance metrics
    if session_results.get('metrics'):
        st.markdown('<div class="analysis-header">Model Performance</div>', unsafe_allow_html=True)
        
        cols = st.columns(len(session_results['metrics']))
        
        for i, (model_name, metrics) in enumerate(session_results['metrics'].items()):
            with cols[i]:
                st.subheader(f"{model_name.upper()} Model")
                
                if isinstance(metrics, dict):
                    accuracy = metrics.get('validation_accuracy', metrics.get('accuracy', 0))
                    f1_score = metrics.get('validation_f1', metrics.get('f1_score', 0))
                    training_time = metrics.get('training_time', 0)
                    training_samples = metrics.get('training_samples', 0)
                    
                    st.metric("Accuracy", f"{accuracy:.3f}")
                    st.metric("F1-Score", f"{f1_score:.3f}")
                    if training_time > 0:
                        st.metric("Training Time", f"{training_time:.1f}s")
                    if training_samples > 0:
                        st.metric("Training Samples", f"{training_samples:,}")
    
    # Display plots
    if session_results.get('plots'):
        st.markdown('<div class="analysis-header">Performance Visualizations</div>', unsafe_allow_html=True)
        
        for model_name, plots in session_results['plots'].items():
            if plots:
                st.subheader(f"{model_name.upper()} Visualizations")
                
                plot_cols = st.columns(len(plots))
                
                for i, (plot_type, plot_path) in enumerate(plots.items()):
                    with plot_cols[i]:
                        try:
                            if Path(plot_path).exists():
                                st.image(plot_path, caption=f"{plot_type.replace('_', ' ').title()}")
                            else:
                                st.warning(f"Plot not found: {plot_path}")
                        except Exception as e:
                            st.error(f"Error loading plot: {e}")
    
    # Display insights
    if session_results.get('reports', {}).get('evaluation'):
        eval_report = session_results['reports']['evaluation']
        
        if 'summary' in eval_report and 'insights' in eval_report['summary']:
            st.markdown('<div class="analysis-header">Intelligent Insights</div>', unsafe_allow_html=True)
            
            insights = eval_report['summary']['insights']
            for i, insight in enumerate(insights, 1):
                st.info(f"💡 **Insight {i}**: {insight}")

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
    """Applicazione principale Streamlit - Versione FIXED con session-based loading"""
    
    try:
        # Header principale
        st.markdown('<div class="main-header">Scientific Sentiment Analysis System - FIXED SESSION-BASED VERSION</div>', 
                    unsafe_allow_html=True)
        
        # === CARICAMENTO INIZIALE SICURO ===
        with st.spinner("Loading system resources..."):
            # Verifica directory
            dir_ok, dir_msg = ensure_result_directories()
            
            # Carica embedding model
            embedding_model, embed_msg = load_embedding_model()
            
            # 🔧 FIXED: Carica modelli da sessioni
            models = load_trained_models_from_session()
            
            # Carica dataset principale
            main_df, main_dataset_path, dataset_error = load_main_dataset()

            # 🔧 FIXED: Carica risultati della sessione automaticamente
            session_results = load_session_results()
        
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
            
            # 🔧 NEW: Session status
            st.subheader("Session Status")
            if models.get('session_info'):
                session_info = models['session_info']
                st.success(f"✅ Latest Session: {session_info['session_name']}")
                st.info(f"Created: {session_info['creation_time'].strftime('%Y-%m-%d %H:%M')}")
                st.info(f"Models: {len(session_info['models_found'])}")
                st.info(f"Reports: {len(session_info['reports_found'])}")
                st.info(f"Plots: {len(session_info['plots_found'])}")
            else:
                st.warning("⚠️ No sessions found")
            
            # Percorsi
            st.subheader("Project Structure")
            st.text(f"Root: {PROJECT_ROOT}")
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

            # 🔧 NEW: Display session results first
            if session_results:
                display_session_results()
                st.markdown("---")

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
                            
                            # 🔧 NEW: Refresh models and session info
                            st.success("🔄 Pipeline completed! Refreshing models...")
                            st.cache_resource.clear()
                            st.rerun()
                            
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
                        
                        # 🔧 NEW: Refresh models and session info
                        st.success("🔄 Pipeline completed! Refreshing models...")
                        st.cache_resource.clear()
                        st.rerun()
                        
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
                - Results saved to `results/auto_analysis_<timestamp>/`
                
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
            
            # 🔧 NEW: Check for session results first
            if session_results:
                st.success(f"Latest session results available: **{session_results['session_name']}**")
                
                # Show session info
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.info(f"Session Directory: {session_results['session_dir']}")
                    
                    # Model metrics summary
                    if session_results.get('metrics'):
                        st.info("📊 Models with results:")
                        for model_name, metrics in session_results['metrics'].items():
                            if isinstance(metrics, dict):
                                accuracy = metrics.get('validation_accuracy', metrics.get('accuracy', 0))
                                st.info(f"   • {model_name.upper()}: {accuracy:.3f} accuracy")
                    
                    # Plots available
                    if session_results.get('plots'):
                        plot_count = sum(len(plots) for plots in session_results['plots'].values())
                        st.info(f"📈 {plot_count} visualization plots available")
                    
                with col2:
                    # Quick metrics
                    total_models = len(session_results.get('metrics', {}))
                    total_plots = sum(len(plots) for plots in session_results.get('plots', {}).values())
                    total_reports = len(session_results.get('reports', {}))
                    
                    st.metric("Models", total_models)
                    st.metric("Plots", total_plots)
                    st.metric("Reports", total_reports)
                
                # Download options for session results
                st.subheader("Download Session Files")
                
                session_path = Path(session_results['session_dir'])
                
                # Generate file lists
                available_files = {}
                
                # Model files
                models_dir = session_path / "models"
                if models_dir.exists():
                    for model_file in models_dir.glob("*.pth"):
                        available_files[f"MLP Model ({model_file.name})"] = model_file
                    for model_file in models_dir.glob("*.pkl"):
                        available_files[f"SVM Model ({model_file.name})"] = model_file
                
                # Report files
                reports_dir = session_path / "reports"
                if reports_dir.exists():
                    for report_file in reports_dir.glob("*.json"):
                        available_files[f"Report ({report_file.name})"] = report_file
                    for report_file in reports_dir.glob("*.txt"):
                        available_files[f"Text Report ({report_file.name})"] = report_file
                
                # Plot files
                plots_dir = session_path / "plots"
                if plots_dir.exists():
                    for plot_file in plots_dir.glob("*.png"):
                        available_files[f"Plot ({plot_file.name})"] = plot_file
                
                # Create download buttons
                if available_files:
                    cols = st.columns(3)
                    col_idx = 0
                    
                    for file_label, file_path in available_files.items():
                        with cols[col_idx % 3]:
                            try:
                                with open(file_path, 'rb') as f:
                                    st.download_button(
                                        label=f"📄 {file_label}",
                                        data=f.read(),
                                        file_name=file_path.name,
                                        mime="application/octet-stream"
                                    )
                                col_idx += 1
                            except Exception as e:
                                st.error(f"Error reading {file_label}: {e}")
                
                # Export features description
                st.subheader("Session Export Contents")
                st.markdown("""
                **Latest Session Export Contains:**
                - Complete model training results and checkpoints  
                - Embedding matrices and metadata
                - Performance metrics and evaluation reports
                - Intelligent insights and analysis summaries
                - Scientific visualization plots
                - Structured session directory with full organization
                
                **Professional Features:**
                - Research-grade documentation and methodology
                - Quantitative metrics with statistical significance
                - Reproducible analysis with full traceability
                - Export-ready for academic and industry presentation
                """)
                
            elif 'current_analysis' in st.session_state:
                # Legacy current analysis support
                try:
                    analysis = st.session_state['current_analysis']
                    
                    st.success(f"Analysis results available for: **{analysis['filename']}**")
                    
                    # Show analysis info (legacy support)
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.info(f"Generated: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.info(f"Analysis Type: {analysis.get('analysis_type', 'Standard').title()}")
                        
                        # Session directory if available
                        session_dir = analysis.get('session_directory')
                        if session_dir:
                            st.info(f"Session Directory: {session_dir}")
                    
                    with col2:
                        st.metric("Analysis Type", "Legacy")
                        
                except Exception as e:
                    st.error(f"Error in legacy export section: {str(e)}")
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
            <strong>Scientific Sentiment Analysis System v3.0 - FIXED SESSION-BASED</strong><br>
            <small>Research-Grade • Objective • Reproducible • {'✅ Pipeline Connected' if PIPELINE_AVAILABLE else '❌ Pipeline Disconnected'}</small><br>
            <small>Results automatically loaded from: {PROJECT_ROOT}/results/auto_analysis_&lt;timestamp&gt;/</small>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
