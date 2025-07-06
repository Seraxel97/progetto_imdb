#!/usr/bin/env python3
"""
🤖 SENTIMENT ANALYSIS PIPELINE - GUI COMPLETA FINALE 🤖

GUI perfetta che integra tutti gli script esistenti senza errori.
Sostituisce gui_data_dashboard.py con versione finale robusta.

SALVA QUESTO FILE COME: gui_data_dashboard.py
AVVIA CON: streamlit run gui_data_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import time
import zipfile
import io
import sys
import os
import logging
import subprocess
import warnings

warnings.filterwarnings('ignore')

# Setup paths dinamico
CURRENT_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
SCRIPTS_DIR = CURRENT_DIR / "scripts"
PROJECT_ROOT = CURRENT_DIR

# Aggiungi scripts al path
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Configurazione Streamlit
st.set_page_config(
    page_title="🤖 Sentiment Analysis Pipeline", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS per stile professionale
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .step-box {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .success-box {
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    .warning-box {
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    }
    .error-box {
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    }
    .metric-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# FUNZIONI DI IMPORT SICURE
# ================================

def safe_import_pipeline():
    """Import sicuro del pipeline runner"""
    try:
        from pipeline_runner import run_complete_csv_analysis
        return run_complete_csv_analysis, None
    except ImportError:
        try:
            from scripts.pipeline_runner import run_complete_csv_analysis
            return run_complete_csv_analysis, None
        except ImportError as e:
            return None, f"Errore import pipeline_runner: {e}"

def safe_import_utils():
    """Import sicuro delle utilities"""
    try:
        from enhanced_utils_unified import validate_and_process_csv
        return validate_and_process_csv, None
    except ImportError:
        try:
            from scripts.enhanced_utils_unified import validate_and_process_csv
            return validate_and_process_csv, None
        except ImportError as e:
            return None, f"Errore import enhanced_utils_unified: {e}"

# ================================
# INIZIALIZZAZIONE STATO SESSIONE
# ================================

def init_session_state():
    """Inizializza lo stato della sessione"""
    defaults = {
        'uploaded_file_path': None,
        'file_analyzed': False,
        'analysis_results': None,
        'pipeline_executed': False,
        'pipeline_results': None,
        'logs': [],
        'current_step': 'upload'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ================================
# FUNZIONI UTILITY
# ================================

def load_csv_robust(file_path):
    """Carica CSV con encoding detection"""
    try:
        # Prova UTF-8 prima
        df = pd.read_csv(file_path, encoding='utf-8')
        return df, "utf-8"
    except UnicodeDecodeError:
        try:
            # Prova Latin-1
            df = pd.read_csv(file_path, encoding='latin-1')
            return df, "latin-1"
        except Exception:
            try:
                # Prova con errori sostituiti
                df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
                return df, "utf-8 (with errors replaced)"
            except Exception as e:
                raise Exception(f"Impossibile leggere il file CSV: {e}")

def detect_text_column(df):
    """Rileva automaticamente la colonna di testo"""
    possible_names = ['text', 'review', 'comment', 'content', 'description', 'message']
    
    # Cerca per nome esatto
    for col in df.columns:
        if col.lower().strip() in possible_names:
            return col
    
    # Cerca per nome parziale
    for col in df.columns:
        for name in possible_names:
            if name in col.lower():
                return col
    
    # Prendi la prima colonna stringa con contenuto
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_text = str(df[col].iloc[0]) if len(df) > 0 else ""
            if len(sample_text) > 10:  # Almeno 10 caratteri
                return col
    
    return df.columns[0] if len(df.columns) > 0 else None

def detect_label_column(df):
    """Rileva automaticamente la colonna delle etichette"""
    possible_names = ['label', 'sentiment', 'class', 'target', 'rating']
    
    # Cerca per nome esatto
    for col in df.columns:
        if col.lower().strip() in possible_names:
            return col
    
    # Cerca per nome parziale
    for col in df.columns:
        for name in possible_names:
            if name in col.lower():
                return col
    
    # Cerca colonna con valori binari o categorici
    for col in df.columns:
        unique_vals = df[col].unique()
        if len(unique_vals) <= 5:  # Massimo 5 valori unici
            unique_str = [str(v).lower() for v in unique_vals if pd.notna(v)]
            if any(val in unique_str for val in ['0', '1', 'positive', 'negative', 'pos', 'neg']):
                return col
    
    return None

# ================================
# HEADER E INTERFACCIA
# ================================

def show_header():
    """Mostra header principale"""
    st.markdown('<h1 class="main-header">🤖 Sentiment Analysis Pipeline</h1>', unsafe_allow_html=True)
    
    # Info rapida
    with st.expander("ℹ️ Come funziona", expanded=False):
        st.markdown("""
        **Pipeline completa in 3 step:**
        1. 📁 **Upload CSV** con colonna testo (e opzionalmente etichette)
        2. 🚀 **Esecuzione automatica** di tutto il processo
        3. 📊 **Visualizzazione risultati** e download report
        
        **Formati supportati:** qualsiasi CSV con colonna di testo
        """)
    
    st.markdown("---")

def show_sidebar():
    """Mostra sidebar con controlli"""
    with st.sidebar:
        st.header("🎛️ Controlli")
        
        # Stato attuale
        st.subheader("📊 Stato")
        
        if st.session_state.uploaded_file_path:
            st.success("✅ File caricato")
        else:
            st.info("📁 Carica file CSV")
        
        if st.session_state.file_analyzed:
            st.success("✅ File analizzato")
        else:
            st.info("🔍 Analisi in attesa")
        
        if st.session_state.pipeline_executed:
            st.success("✅ Pipeline completata")
        else:
            st.info("🚀 Pipeline in attesa")
        
        st.markdown("---")
        
        # Reset
        if st.button("🔄 Reset Completo"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        
        # Info
        st.markdown("**💡 Suggerimenti:**")
        st.markdown("- CSV con colonna 'text' o 'review'")
        st.markdown("- Encoding UTF-8 preferito")
        st.markdown("- Min 10 righe per training")

# ================================
# STEP 1: UPLOAD E ANALISI FILE
# ================================

def step_upload_file():
    """Step 1: Upload e analisi file"""
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("📁 Step 1: Caricamento e Analisi File")
    
    uploaded_file = st.file_uploader(
        "Seleziona il tuo file CSV",
        type=['csv'],
        help="Carica un CSV con una colonna di testo per l'analisi del sentiment"
    )
    
    if uploaded_file is not None:
        # Salva file temporaneo
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file_path = temp_dir / f"uploaded_{timestamp}.csv"
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.uploaded_file_path = str(temp_file_path)
        
        # Analizza il file
        with st.spinner("🔍 Analizzando il file..."):
            analyze_uploaded_file(temp_file_path)
    
    st.markdown('</div>', unsafe_allow_html=True)

def analyze_uploaded_file(file_path):
    """Analizza il file caricato"""
    try:
        # Carica CSV
        df, encoding = load_csv_robust(file_path)
        
        # Rileva colonne
        text_col = detect_text_column(df)
        label_col = detect_label_column(df)
        
        if text_col is None:
            st.error("❌ Nessuna colonna di testo trovata nel CSV!")
            return
        
        # Pulisci dati base
        df_clean = df.copy()
        df_clean = df_clean.dropna(subset=[text_col])
        df_clean = df_clean[df_clean[text_col].astype(str).str.len() > 5]
        
        # Analizza etichette se presenti
        has_labels = False
        if label_col is not None:
            unique_labels = df_clean[label_col].unique()
            has_labels = len(unique_labels) <= 10  # Massimo 10 etichette uniche
        
        # Salva risultati analisi
        analysis_results = {
            'dataframe': df_clean,
            'text_column': text_col,
            'label_column': label_col,
            'has_labels': has_labels,
            'total_samples': len(df_clean),
            'encoding': encoding,
            'original_samples': len(df)
        }
        
        st.session_state.analysis_results = analysis_results
        st.session_state.file_analyzed = True
        
        # Mostra risultati
        show_analysis_results(analysis_results)
        
    except Exception as e:
        st.error(f"❌ Errore nell'analisi del file: {str(e)}")

def show_analysis_results(results):
    """Mostra i risultati dell'analisi"""
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.success("✅ File analizzato con successo!")
    
    # Metriche
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Campioni", f"{results['total_samples']:,}")
    
    with col2:
        labels_text = "✅ Sì" if results['has_labels'] else "❌ No"
        st.metric("🏷️ Etichette", labels_text)
    
    with col3:
        st.metric("📄 Colonna testo", results['text_column'])
    
    with col4:
        st.metric("📝 Encoding", results['encoding'])
    
    # Preview dati
    st.subheader("👁️ Anteprima dati")
    df = results['dataframe']
    preview_cols = [results['text_column']]
    if results['label_column']:
        preview_cols.append(results['label_column'])
    
    st.dataframe(df[preview_cols].head(10), use_container_width=True)
    
    # Grafici analisi
    col1, col2 = st.columns(2)
    
    with col1:
        # Lunghezza testi
        text_lengths = df[results['text_column']].astype(str).str.len()
        fig = px.histogram(
            x=text_lengths, 
            nbins=30,
            title="📏 Distribuzione Lunghezza Testi",
            labels={"x": "Caratteri", "y": "Frequenza"}
        )
        fig.add_vline(x=text_lengths.mean(), line_dash="dash", 
                     annotation_text=f"Media: {text_lengths.mean():.0f}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if results['has_labels']:
            # Distribuzione etichette
            label_counts = df[results['label_column']].value_counts()
            fig = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                title="🏷️ Distribuzione Etichette"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Info modalità inferenza
            st.info("🔍 **Modalità Inferenza**\n\nIl dataset non ha etichette. La pipeline analizzerà i sentimenti automaticamente.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ================================
# STEP 2: ESECUZIONE PIPELINE
# ================================

def step_execute_pipeline():
    """Step 2: Esecuzione pipeline"""
    if not st.session_state.file_analyzed:
        st.warning("⚠️ Prima analizza un file CSV!")
        return
    
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("🚀 Step 2: Esecuzione Pipeline Completa")
    
    st.info("La pipeline eseguirà automaticamente: **Preprocessing** → **Embeddings** → **Training** → **Report**")
    
    if st.button("🚀 Avvia Pipeline Completa", type="primary"):
        execute_full_pipeline()
    
    st.markdown('</div>', unsafe_allow_html=True)

def execute_full_pipeline():
    """Esegue la pipeline completa"""
    # Import funzione pipeline
    run_pipeline_func, import_error = safe_import_pipeline()
    
    if run_pipeline_func is None:
        st.error(f"❌ Errore import pipeline: {import_error}")
        st.error("Assicurati che il file scripts/pipeline_runner.py esista")
        return
    
    # Containers per UI
    progress_container = st.container()
    log_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    with log_container:
        log_area = st.empty()
    
    # Lista logs
    logs = []
    
    def log_callback(message):
        """Callback per logs real-time"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"{timestamp} - {message}"
        logs.append(log_line)
        
        # Aggiorna UI
        log_area.text_area("📋 Log Esecuzione", "\n".join(logs[-15:]), height=300)
        
        # Aggiorna progress
        if "Step 1" in message or "preprocessing" in message.lower():
            progress_bar.progress(20)
            status_text.text("📊 Preprocessing dati...")
        elif "Step 2" in message or "embedding" in message.lower():
            progress_bar.progress(40)
            status_text.text("🧠 Generazione embeddings...")
        elif "Step 3" in message or "mlp" in message.lower():
            progress_bar.progress(60)
            status_text.text("🤖 Training modello MLP...")
        elif "svm" in message.lower():
            progress_bar.progress(80)
            status_text.text("⚡ Training modello SVM...")
        elif "Step 4" in message or "report" in message.lower():
            progress_bar.progress(90)
            status_text.text("📋 Generazione report...")
        elif "completed" in message.lower() or "success" in message.lower():
            progress_bar.progress(100)
            status_text.text("✅ Pipeline completata!")
    
    # Esegui pipeline
    try:
        with st.spinner("🔄 Esecuzione in corso..."):
            results = run_pipeline_func(
                csv_path=st.session_state.uploaded_file_path,
                log_callback=log_callback
            )
        
        # Salva risultati
        st.session_state.pipeline_results = results
        st.session_state.logs = logs
        st.session_state.pipeline_executed = True
        
        # Mostra risultato
        if results.get('success') or results.get('overall_success'):
            st.success("🎉 Pipeline completata con successo!")
            progress_bar.progress(100)
            status_text.text("✅ Tutti i passaggi completati!")
        else:
            error_msg = results.get('error', 'Errore sconosciuto')
            st.error(f"❌ Pipeline fallita: {error_msg}")
            
    except Exception as e:
        st.error(f"❌ Errore durante l'esecuzione: {str(e)}")
        st.error("Controlla che tutti gli script siano nella cartella 'scripts/'")

# ================================
# STEP 3: VISUALIZZAZIONE RISULTATI
# ================================

def step_show_results():
    """Step 3: Mostra risultati"""
    if not st.session_state.pipeline_executed:
        st.info("🚀 Esegui prima la pipeline per vedere i risultati")
        return
    
    results = st.session_state.pipeline_results
    
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.subheader("🎉 Step 3: Risultati Pipeline")
    
    # Metriche generali
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = results.get('status', 'unknown').upper()
        st.metric("📈 Status", status)
    
    with col2:
        duration = results.get('total_duration', 0)
        st.metric("⏱️ Durata", f"{duration:.1f}s")
    
    with col3:
        inference_mode = "🔍 Sì" if results.get('inference_only') else "🎯 No"
        st.metric("Solo Inferenza", inference_mode)
    
    with col4:
        session_dir = results.get('session_directory', '')
        if session_dir:
            session_name = Path(session_dir).name
            st.metric("📁 Sessione", session_name)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs per risultati dettagliati
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Panoramica", "🤖 Modelli", "📈 Grafici", "📥 Download"])
    
    with tab1:
        show_overview_tab(results)
    
    with tab2:
        show_models_tab(results)
    
    with tab3:
        show_charts_tab(results)
    
    with tab4:
        show_download_tab(results)

def show_overview_tab(results):
    """Tab panoramica risultati"""
    st.subheader("📊 Panoramica Generale")
    
    # Steps eseguiti
    steps = results.get('steps', {})
    if steps:
        st.subheader("🔄 Passi Eseguiti")
        
        for step_name, step_info in steps.items():
            status = step_info.get('status', 'unknown')
            duration = step_info.get('duration', 0)
            
            if status == 'completed':
                st.success(f"✅ {step_name.replace('_', ' ').title()}: {duration:.1f}s")
            elif status == 'skipped':
                reason = step_info.get('reason', 'N/A')
                st.info(f"⏭️ {step_name.replace('_', ' ').title()}: Saltato ({reason})")
            elif status == 'failed':
                st.error(f"❌ {step_name.replace('_', ' ').title()}: Fallito")
            else:
                st.warning(f"⚠️ {step_name.replace('_', ' ').title()}: {status}")
    
    # Errori e warnings
    errors = results.get('errors', [])
    warnings_list = results.get('warnings', [])
    
    if errors:
        st.subheader("❌ Errori")
        for error in errors:
            st.error(error)
    
    if warnings_list:
        st.subheader("⚠️ Avvisi")
        for warning in warnings_list:
            st.warning(warning)
    
    # Logs
    if st.session_state.logs:
        with st.expander("📋 Log Completi"):
            st.text("\n".join(st.session_state.logs))

def show_models_tab(results):
    """Tab risultati modelli"""
    st.subheader("🤖 Risultati Modelli")
    
    final_results = results.get('final_results', {})
    
    # Risultati MLP
    if 'mlp_status' in final_results:
        mlp_status = final_results['mlp_status']
        
        st.subheader("🧠 Modello MLP (Multi-Layer Perceptron)")
        
        if mlp_status.get('status') == 'completed':
            performance = mlp_status.get('performance', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                accuracy = performance.get('accuracy', 0)
                st.metric("🎯 Accuracy", f"{accuracy:.3f}")
            with col2:
                epochs = performance.get('total_epochs', 0)
                st.metric("🔄 Epochs", epochs)
            with col3:
                val_acc = performance.get('final_val_accuracy', 0)
                st.metric("✅ Val Accuracy", f"{val_acc:.3f}")
                
        elif mlp_status.get('status') == 'skipped':
            st.info("⏭️ Training MLP saltato (dati insufficienti o modalità inferenza)")
        else:
            st.error("❌ Training MLP fallito")
    
    # Risultati SVM
    if 'svm_status' in final_results:
        svm_status = final_results['svm_status']
        
        st.subheader("⚡ Modello SVM (Support Vector Machine)")
        
        if svm_status.get('status') == 'completed':
            performance = svm_status.get('performance', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                accuracy = performance.get('accuracy', 0)
                st.metric("🎯 Accuracy", f"{accuracy:.3f}")
            with col2:
                f1_score = performance.get('f1_score', 0)
                st.metric("📊 F1-Score", f"{f1_score:.3f}")
            with col3:
                training_time = performance.get('training_time', 0)
                st.metric("⏱️ Training Time", f"{training_time:.1f}s")
                
        elif svm_status.get('status') == 'skipped':
            st.info("⏭️ Training SVM saltato (dati insufficienti o modalità inferenza)")
        else:
            st.error("❌ Training SVM fallito")
    
    # Se nessun modello
    if 'mlp_status' not in final_results and 'svm_status' not in final_results:
        st.info("🔍 Nessun modello trainato. Probabilmente eseguito in modalità inferenza.")

def show_charts_tab(results):
    """Tab grafici e visualizzazioni"""
    st.subheader("📈 Grafici e Visualizzazioni")
    
    session_dir = results.get('session_directory')
    if not session_dir:
        st.warning("⚠️ Nessuna directory di sessione trovata.")
        return
    
    plots_dir = Path(session_dir) / "plots"
    if not plots_dir.exists():
        st.warning("⚠️ Cartella plots non trovata.")
        return
    
    # Trova tutti i grafici PNG
    plot_files = list(plots_dir.glob("*.png"))
    
    if not plot_files:
        st.warning("⚠️ Nessun grafico trovato.")
        return
    
    # Mostra grafici in griglia
    st.subheader("📊 Grafici Generati")
    
    # Organizza in colonne
    num_plots = len(plot_files)
    if num_plots == 1:
        st.image(str(plot_files[0]), caption=plot_files[0].name, use_column_width=True)
    elif num_plots == 2:
        col1, col2 = st.columns(2)
        with col1:
            st.image(str(plot_files[0]), caption=plot_files[0].name, use_column_width=True)
        with col2:
            st.image(str(plot_files[1]), caption=plot_files[1].name, use_column_width=True)
    else:
        # Per più di 2 grafici, usa griglia 2x2
        cols = st.columns(2)
        for i, plot_file in enumerate(plot_files):
            with cols[i % 2]:
                st.image(str(plot_file), caption=plot_file.name, use_column_width=True)

def show_download_tab(results):
    """Tab download risultati"""
    st.subheader("📥 Download Risultati")
    
    session_dir = results.get('session_directory')
    if not session_dir:
        st.warning("⚠️ Nessuna directory di sessione trovata.")
        return
    
    session_path = Path(session_dir)
    
    # Info sui file
    st.subheader("📋 File Disponibili")
    
    # Conta file per categoria
    file_counts = {}
    total_size = 0
    
    for subdir in ['plots', 'reports', 'models', 'processed', 'embeddings', 'logs']:
        subdir_path = session_path / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob('*'))
            file_counts[subdir] = len([f for f in files if f.is_file()])
            total_size += sum(f.stat().st_size for f in files if f.is_file())
        else:
            file_counts[subdir] = 0
    
    # Mostra conteggi
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Grafici", file_counts.get('plots', 0))
        st.metric("📋 Report", file_counts.get('reports', 0))
    with col2:
        st.metric("🤖 Modelli", file_counts.get('models', 0))
        st.metric("📄 Dati", file_counts.get('processed', 0))
    with col3:
        st.metric("🧠 Embeddings", file_counts.get('embeddings', 0))
        st.metric("📝 Log", file_counts.get('logs', 0))
    
    total_size_mb = total_size / (1024 * 1024)
    st.info(f"📦 Dimensione totale: {total_size_mb:.1f} MB")
    
    # Crea e offri download ZIP
    try:
        st.subheader("📦 Download Completo")
        
        if st.button("📦 Prepara ZIP per Download"):
            with st.spinner("📦 Creando archivio ZIP..."):
                zip_data = create_results_zip(session_path)
            
            if zip_data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sentiment_analysis_results_{timestamp}.zip"
                
                st.download_button(
                    label="📥 Scarica Report Completo (ZIP)",
                    data=zip_data,
                    file_name=filename,
                    mime="application/zip",
                    help="Scarica tutti i risultati in un file ZIP"
                )
                
                st.success("✅ Report pronto per il download!")
    
    except Exception as e:
        st.error(f"❌ Errore nella creazione del ZIP: {str(e)}")

def create_results_zip(session_path):
    """Crea ZIP con tutti i risultati"""
    try:
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in session_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(session_path)
                    zip_file.write(file_path, arcname)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    except Exception as e:
        st.error(f"Errore creazione ZIP: {e}")
        return None

# ================================
# MAIN FUNCTION
# ================================

def main():
    """Funzione principale"""
    # Inizializza sessione
    init_session_state()
    
    # Header
    show_header()
    
    # Sidebar
    show_sidebar()
    
    # Layout principale basato su stato
    if st.session_state.current_step == 'upload' or not st.session_state.file_analyzed:
        step_upload_file()
        
        if st.session_state.file_analyzed:
            st.session_state.current_step = 'pipeline'
    
    if st.session_state.current_step == 'pipeline' and st.session_state.file_analyzed:
        step_execute_pipeline()
        
        if st.session_state.pipeline_executed:
            st.session_state.current_step = 'results'
    
    if st.session_state.current_step == 'results' and st.session_state.pipeline_executed:
        step_show_results()
    
    # Footer
    st.markdown("---")
    st.markdown("🤖 **Sentiment Analysis Pipeline** - Powered by Streamlit")

if __name__ == "__main__":
    main()
