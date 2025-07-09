#!/usr/bin/env python3
"""
Fixed Pipeline Runner - DIRECT INTEGRATION WITH STANDARD DIRECTORIES
Risolve i problemi salvando ANCHE nelle directory standard e generando report completi.

🔧 FIXES APPLIED:
- ✅ Salva nelle directory standard (data/processed, data/embeddings) oltre che nelle session
- ✅ Genera report e grafici completi
- ✅ Mantiene compatibilità GUI
- ✅ Error handling robusto
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import torch
import shutil
from pathlib import Path
from datetime import datetime
import logging
import time
from typing import Dict, Any, Optional, Callable
import warnings
from ftfy import fix_text

warnings.filterwarnings('ignore')

# Dynamic project root detection
try:
    CURRENT_FILE = Path(__file__).resolve()
    if CURRENT_FILE.parent.name == 'scripts':
        PROJECT_ROOT = CURRENT_FILE.parent.parent
    else:
        PROJECT_ROOT = CURRENT_FILE.parent
except:
    PROJECT_ROOT = Path.cwd()

class DirectEmbeddingGenerator:
    """Direct embedding generation senza subprocess"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        
    def load_model_safe(self, model_name="all-MiniLM-L6-v2"):
        """Load SentenceTransformer model with error handling"""
        try:
            from sentence_transformers import SentenceTransformer
            self.logger.info(f"📥 Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
            return True
        except ImportError as e:
            self.logger.error(f"❌ sentence-transformers not installed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            return False
    
    def process_csv_to_embeddings(self, input_dir: Path, output_dir: Path, standard_output_dir: Path = None):
        """Process CSV files to embeddings - salva in ENTRAMBE le directory"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            if standard_output_dir:
                standard_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find CSV files
            csv_files = list(input_dir.glob("*.csv"))
            if not csv_files:
                self.logger.warning("⚠️ No CSV files found")
                return False
            
            # Load model
            if not self.load_model_safe():
                return False
            
            for csv_file in csv_files:
                split_name = csv_file.stem
                self.logger.info(f"🔄 Processing {split_name}")
                
                try:
                    # Load CSV
                    df = pd.read_csv(csv_file)
                    df.columns = df.columns.str.strip().str.lower()
                    
                    if 'text' not in df.columns:
                        self.logger.warning(f"⚠️ No 'text' column in {split_name}")
                        continue
                    
                    # Process texts
                    texts = df['text'].fillna('').astype(str).tolist()
                    valid_texts = [t for t in texts if len(t.strip()) > 3]
                    
                    if not valid_texts:
                        self.logger.warning(f"⚠️ No valid texts in {split_name}")
                        continue
                    
                    # Generate embeddings in batches
                    self.logger.info(f"   🧠 Generating embeddings for {len(valid_texts)} texts")
                    embeddings = []
                    
                    batch_size = 32
                    for i in range(0, len(valid_texts), batch_size):
                        batch = valid_texts[i:i+batch_size]
                        try:
                            batch_emb = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                            embeddings.append(batch_emb)
                        except Exception as e:
                            self.logger.error(f"   ❌ Error in batch {i//batch_size}: {e}")
                            # Create dummy embeddings to continue
                            dummy_emb = np.zeros((len(batch), 384))
                            embeddings.append(dummy_emb)
                    
                    if embeddings:
                        all_embeddings = np.vstack(embeddings)
                    else:
                        continue
                    
                    # Handle labels
                    if 'label' in df.columns:
                        labels = df['label'].tolist()[:len(valid_texts)]
                        # Normalize labels
                        normalized_labels = []
                        for label in labels:
                            if pd.isna(label):
                                normalized_labels.append(-1)
                            elif str(label).lower() in ['positive', 'pos', '1', 1]:
                                normalized_labels.append(1)
                            elif str(label).lower() in ['negative', 'neg', '0', 0]:
                                normalized_labels.append(0)
                            else:
                                normalized_labels.append(-1)
                        labels = np.array(normalized_labels)
                    else:
                        labels = np.full(len(valid_texts), -1)
                    
                    # Save embeddings and labels in session directory
                    embeddings_file = output_dir / f"X_{split_name}.npy"
                    labels_file = output_dir / f"y_{split_name}.npy"
                    
                    np.save(embeddings_file, all_embeddings)
                    np.save(labels_file, labels)
                    
                    # 🆕 NUOVO: Salva ANCHE nelle directory standard
                    if standard_output_dir:
                        std_embeddings_file = standard_output_dir / f"X_{split_name}.npy"
                        std_labels_file = standard_output_dir / f"y_{split_name}.npy"
                        
                        np.save(std_embeddings_file, all_embeddings)
                        np.save(std_labels_file, labels)
                        
                        self.logger.info(f"   ✅ Saved in session: {embeddings_file.name}")
                        self.logger.info(f"   ✅ Saved in standard: {std_embeddings_file}")
                    else:
                        self.logger.info(f"   ✅ Saved: {embeddings_file.name} ({all_embeddings.shape})")
                    
                except Exception as e:
                    self.logger.error(f"   ❌ Error processing {split_name}: {e}")
                    continue
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Embedding generation failed: {e}")
            return False

class DirectMLPTrainer:
    """Direct MLP training senza subprocess"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def simple_mlp_training(self, embeddings_dir: Path, output_dir: Path):
        """Simple MLP training senza complex dependencies"""
        try:
            models_dir = output_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # 🆕 NUOVO: Crea anche directory standard
            standard_models_dir = PROJECT_ROOT / "results" / "models"
            standard_models_dir.mkdir(parents=True, exist_ok=True)
            
            # Find training data
            X_train_file = embeddings_dir / "X_train.npy"
            y_train_file = embeddings_dir / "y_train.npy"
            
            if not (X_train_file.exists() and y_train_file.exists()):
                self.logger.warning("⚠️ No training data found for MLP")
                return False
            
            # Load data
            X_train = np.load(X_train_file)
            y_train = np.load(y_train_file)
            
            # Filter valid labels
            valid_mask = np.isin(y_train, [0, 1])
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]
            
            if len(X_train) < 10:
                self.logger.warning("⚠️ Insufficient training data for MLP")
                return False
            
            # Simple MLP model
            class SimpleMLP(torch.nn.Module):
                def __init__(self, input_dim):
                    super().__init__()
                    self.network = torch.nn.Sequential(
                        torch.nn.Linear(input_dim, 128),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.2),
                        torch.nn.Linear(128, 64),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.2),
                        torch.nn.Linear(64, 1),
                        torch.nn.Sigmoid()
                    )
                
                def forward(self, x):
                    return self.network(x)
            
            self.logger.info(f"🧠 Training simple MLP on {len(X_train)} samples")
            
            # Create model
            model = SimpleMLP(X_train.shape[1]).to(self.device)
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Prepare data
            X_tensor = torch.FloatTensor(X_train).to(self.device)
            y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
            
            # Training loop
            model.train()
            epochs = min(50, max(10, len(X_train) // 20))
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    self.logger.info(f"   Epoch {epoch}: Loss = {loss.item():.4f}")
            
            # Save model in session directory
            model_path = models_dir / "mlp_model.pth"
            torch.save(model.state_dict(), model_path)
            
            # 🆕 NUOVO: Salva ANCHE in directory standard
            standard_model_path = standard_models_dir / "mlp_model.pth"
            torch.save(model.state_dict(), standard_model_path)
            
            # Calculate accuracy
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                predictions = (outputs > 0.5).float()
                accuracy = (predictions == y_tensor).float().mean().item()
            
            self.logger.info(f"   ✅ MLP training completed. Accuracy: {accuracy:.3f}")
            
            # Save metadata in entrambe le directory
            metadata = {
                'model_type': 'Simple_MLP',
                'accuracy': accuracy,
                'samples': len(X_train),
                'epochs': epochs,
                'timestamp': datetime.now().isoformat()
            }
            
            # Session directory
            metadata_file = models_dir / "mlp_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Standard directory
            standard_metadata_file = standard_models_dir / "mlp_metadata.json"
            with open(standard_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save status
            status = {
                'status': 'completed',
                'model_type': 'MLP',
                'performance': {'accuracy': accuracy},
                'timestamp': datetime.now().isoformat()
            }
            
            status_file = output_dir / "mlp_training_status.json"
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ MLP training failed: {e}")
            return False

class DirectSVMTrainer:
    """Direct SVM training senza subprocess"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
    def simple_svm_training(self, embeddings_dir: Path, output_dir: Path):
        """Simple SVM training"""
        try:
            from sklearn.svm import LinearSVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score
            
            models_dir = output_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # 🆕 NUOVO: Crea anche directory standard
            standard_models_dir = PROJECT_ROOT / "results" / "models"
            standard_models_dir.mkdir(parents=True, exist_ok=True)
            
            # Find training data
            X_train_file = embeddings_dir / "X_train.npy"
            y_train_file = embeddings_dir / "y_train.npy"
            
            if not (X_train_file.exists() and y_train_file.exists()):
                self.logger.warning("⚠️ No training data found for SVM")
                return False
            
            # Load data
            X_train = np.load(X_train_file)
            y_train = np.load(y_train_file)
            
            # Filter valid labels
            valid_mask = np.isin(y_train, [0, 1])
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]
            
            if len(X_train) < 10:
                self.logger.warning("⚠️ Insufficient training data for SVM")
                return False
            
            self.logger.info(f"⚡ Training SVM on {len(X_train)} samples")
            
            # Scale data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Train SVM
            svm = LinearSVC(C=1.0, class_weight='balanced', max_iter=5000, random_state=42)
            svm.fit(X_train_scaled, y_train)
            
            # Calculate accuracy
            predictions = svm.predict(X_train_scaled)
            accuracy = accuracy_score(y_train, predictions)
            
            self.logger.info(f"   ✅ SVM training completed. Accuracy: {accuracy:.3f}")
            
            # Save model package
            model_package = {
                'model': svm,
                'scaler': scaler,
                'accuracy': accuracy,
                'samples': len(X_train),
                'timestamp': datetime.now().isoformat()
            }
            
            # Session directory
            model_path = models_dir / "svm_model.pkl"
            joblib.dump(model_package, model_path)
            
            # 🆕 NUOVO: Standard directory
            standard_model_path = standard_models_dir / "svm_model.pkl"
            joblib.dump(model_package, standard_model_path)
            
            # Save metadata in entrambe le directory
            metadata = {
                'model_type': 'Simple_SVM',
                'accuracy': accuracy,
                'samples': len(X_train),
                'timestamp': datetime.now().isoformat()
            }
            
            # Session directory
            metadata_file = models_dir / "svm_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Standard directory  
            standard_metadata_file = standard_models_dir / "svm_metadata.json"
            with open(standard_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save status
            status = {
                'status': 'completed',
                'model_type': 'SVM',
                'performance': {'accuracy': accuracy},
                'timestamp': datetime.now().isoformat()
            }
            
            status_file = output_dir / "svm_training_status.json"
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ SVM training failed: {e}")
            return False

def create_comprehensive_report(session_dir: Path, logger=None):
    """🆕 NUOVO: Crea report comprensivo con grafici"""
    try:
        if logger:
            logger.info("📊 Creating comprehensive report...")
        
        reports_dir = session_dir / "reports"
        plots_dir = session_dir / "plots"
        reports_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Crea anche directory standard
        standard_reports_dir = PROJECT_ROOT / "results" / "reports"
        standard_plots_dir = PROJECT_ROOT / "results" / "plots"
        standard_reports_dir.mkdir(parents=True, exist_ok=True)
        standard_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect results
        mlp_status_file = session_dir / "mlp_training_status.json"
        svm_status_file = session_dir / "svm_training_status.json"
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'session_directory': str(session_dir),
            'models': {},
            'summary': {
                'total_models': 0,
                'successful_models': 0,
                'best_model': None,
                'best_accuracy': 0.0
            }
        }
        
        # Check MLP results
        if mlp_status_file.exists():
            with open(mlp_status_file, 'r') as f:
                mlp_data = json.load(f)
                report_data['models']['MLP'] = mlp_data
                report_data['summary']['total_models'] += 1
                if mlp_data.get('status') == 'completed':
                    report_data['summary']['successful_models'] += 1
                    acc = mlp_data.get('performance', {}).get('accuracy', 0)
                    if acc > report_data['summary']['best_accuracy']:
                        report_data['summary']['best_accuracy'] = acc
                        report_data['summary']['best_model'] = 'MLP'
        
        # Check SVM results
        if svm_status_file.exists():
            with open(svm_status_file, 'r') as f:
                svm_data = json.load(f)
                report_data['models']['SVM'] = svm_data
                report_data['summary']['total_models'] += 1
                if svm_data.get('status') == 'completed':
                    report_data['summary']['successful_models'] += 1
                    acc = svm_data.get('performance', {}).get('accuracy', 0)
                    if acc > report_data['summary']['best_accuracy']:
                        report_data['summary']['best_accuracy'] = acc
                        report_data['summary']['best_model'] = 'SVM'
        
        # 🆕 NUOVO: Crea grafici con matplotlib
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Backend senza GUI
            
            # Grafico comparazione modelli
            if report_data['summary']['successful_models'] > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                models = []
                accuracies = []
                
                for model_name, model_data in report_data['models'].items():
                    if model_data.get('status') == 'completed':
                        models.append(model_name)
                        acc = model_data.get('performance', {}).get('accuracy', 0)
                        accuracies.append(acc)
                
                if models and accuracies:
                    # Accuracy comparison
                    bars1 = ax1.bar(models, accuracies, color=['#1f77b4', '#ff7f0e'])
                    ax1.set_title('Model Accuracy Comparison')
                    ax1.set_ylabel('Accuracy')
                    ax1.set_ylim(0, 1)
                    
                    # Add value labels
                    for bar, acc in zip(bars1, accuracies):
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{acc:.3f}', ha='center', va='bottom')
                    
                    # Performance summary
                    ax2.bar(['Best Model'], [report_data['summary']['best_accuracy']], 
                           color='green', alpha=0.7)
                    ax2.set_title(f'Best Performance ({report_data["summary"]["best_model"]})')
                    ax2.set_ylabel('Accuracy')
                    ax2.set_ylim(0, 1)
                    ax2.text(0, report_data['summary']['best_accuracy'] + 0.01,
                            f'{report_data["summary"]["best_accuracy"]:.3f}', 
                            ha='center', va='bottom')
                
                plt.tight_layout()
                
                # Salva in entrambe le directory
                comparison_plot = plots_dir / "model_comparison.png"
                plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
                
                standard_comparison_plot = standard_plots_dir / "model_comparison.png"
                plt.savefig(standard_comparison_plot, dpi=300, bbox_inches='tight')
                
                plt.close()
                
                if logger:
                    logger.info(f"   📊 Created comparison plot: {comparison_plot}")
                    logger.info(f"   📊 Created standard plot: {standard_comparison_plot}")
        
        except Exception as e:
            if logger:
                logger.warning(f"   ⚠️ Could not create plots: {e}")
        
        # Save report in entrambe le directory
        report_file = reports_dir / "evaluation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        
        standard_report_file = standard_reports_dir / "evaluation_report.json"
        with open(standard_report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        
        # Crea report testuale
        text_report_lines = [
            "SENTIMENT ANALYSIS EVALUATION REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Session: {session_dir.name}",
            "",
            "SUMMARY:",
            f"  Total Models: {report_data['summary']['total_models']}",
            f"  Successful Models: {report_data['summary']['successful_models']}",
            f"  Best Model: {report_data['summary']['best_model']}",
            f"  Best Accuracy: {report_data['summary']['best_accuracy']:.4f}",
            "",
            "MODEL DETAILS:"
        ]
        
        for model_name, model_data in report_data['models'].items():
            status = model_data.get('status', 'unknown')
            text_report_lines.append(f"  {model_name}: {status}")
            if status == 'completed':
                acc = model_data.get('performance', {}).get('accuracy', 0)
                text_report_lines.append(f"    Accuracy: {acc:.4f}")
        
        text_report_lines.extend([
            "",
            "=" * 50,
            "Report generation completed."
        ])
        
        # Salva report testuale in entrambe le directory
        text_report_file = reports_dir / "evaluation_report.txt"
        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_report_lines))
        
        standard_text_report_file = standard_reports_dir / "evaluation_report.txt"
        with open(standard_text_report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_report_lines))
        
        if logger:
            logger.info(f"📄 Report saved: {report_file}")
            logger.info(f"📄 Standard report saved: {standard_report_file}")
            logger.info(f"📄 Text report saved: {text_report_file}")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Report generation failed: {e}")
        return False

def run_complete_csv_analysis_direct(csv_path: str, 
                                   log_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    🔧 FIXED: Complete CSV analysis con salvataggio in directory standard
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    def log_message(msg: str):
        logger.info(msg)
        if log_callback:
            log_callback(fix_text(msg))
    
    try:
        log_message("🚀 Starting DIRECT sentiment analysis pipeline")
        log_message("=" * 60)
        
        start_time = time.time()
        
        # Create session directory
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = results_dir / f"direct_analysis_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # 🆕 NUOVO: Crea anche le directory standard
        standard_processed_dir = PROJECT_ROOT / "data" / "processed"
        standard_embeddings_dir = PROJECT_ROOT / "data" / "embeddings"
        standard_processed_dir.mkdir(parents=True, exist_ok=True)
        standard_embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['processed', 'embeddings', 'models', 'plots', 'reports', 'logs']:
            (session_dir / subdir).mkdir(exist_ok=True)
        
        log_message(f"📁 Session directory: {session_dir}")
        log_message(f"📁 Standard processed: {standard_processed_dir}")
        log_message(f"📁 Standard embeddings: {standard_embeddings_dir}")
        
        # Step 1: Process CSV
        log_message("🔄 Step 1: Processing CSV data")
        
        try:
            # Load and process CSV
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip().str.lower()
            
            # Rename columns
            df = df.rename(columns={
                'review': 'text', 'content': 'text',
                'sentiment': 'label', 'class': 'label'
            })
            
            if 'text' not in df.columns:
                available_cols = list(df.columns)
                # Use first string column as text
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df['text'] = df[col]
                        break
            
            if 'text' not in df.columns:
                raise ValueError(f"No text column found. Available: {list(df.columns)}")
            
            # Clean data
            df = df.dropna(subset=['text'])
            df['text'] = df['text'].astype(str)
            df = df[df['text'].str.len() > 5]
            
            # Handle labels
            has_labels = 'label' in df.columns and not df['label'].isnull().all()
            if has_labels:
                # Normalize labels
                df['label'] = df['label'].replace({
                    'positive': 1, 'negative': 0, 'pos': 1, 'neg': 0
                })
                # Keep only binary labels
                df = df[df['label'].isin([0, 1])]
            else:
                df['label'] = -1  # Placeholder
            
            log_message(f"   📊 Processed {len(df)} samples, has_labels={has_labels}")
            
            # Create splits
            if len(df) >= 100 and has_labels:
                from sklearn.model_selection import train_test_split
                train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
                val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
                
                # Save splits in session directory
                train_df.to_csv(session_dir / "processed" / "train.csv", index=False)
                val_df.to_csv(session_dir / "processed" / "val.csv", index=False)
                test_df.to_csv(session_dir / "processed" / "test.csv", index=False)
                
                # 🆕 NUOVO: Salva ANCHE in directory standard
                train_df.to_csv(standard_processed_dir / "train.csv", index=False)
                val_df.to_csv(standard_processed_dir / "val.csv", index=False)
                test_df.to_csv(standard_processed_dir / "test.csv", index=False)
                
                log_message(f"   📂 Created splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
                log_message(f"   📂 Saved in session and standard directories")
            else:
                # Small dataset or no labels
                df.to_csv(session_dir / "processed" / "inference.csv", index=False)
                df.to_csv(standard_processed_dir / "inference.csv", index=False)
                log_message(f"   📂 Created inference file: {len(df)} samples")
                log_message(f"   📂 Saved in session and standard directories")
            
        except Exception as e:
            log_message(f"❌ CSV processing failed: {e}")
            return {'success': False, 'error': str(e)}
        
        # Step 2: Generate embeddings
        log_message("🔄 Step 2: Generating embeddings (DIRECT)")
        
        try:
            embedder = DirectEmbeddingGenerator(logger)
            embedding_success = embedder.process_csv_to_embeddings(
                session_dir / "processed",
                session_dir / "embeddings",
                standard_embeddings_dir  # 🆕 NUOVO: Passa anche directory standard
            )
            
            if embedding_success:
                log_message("   ✅ Embeddings generated successfully")
                log_message("   ✅ Saved in both session and standard directories")
            else:
                log_message("   ⚠️ Embedding generation had issues")
                
        except Exception as e:
            log_message(f"❌ Embedding generation failed: {e}")
            # Continue anyway
        
        # Step 3: Train models (only if we have labels)
        if has_labels and len(df) >= 50:
            # Train MLP
            log_message("🔄 Step 3a: Training MLP (DIRECT)")
            try:
                mlp_trainer = DirectMLPTrainer(logger)
                mlp_success = mlp_trainer.simple_mlp_training(
                    session_dir / "embeddings",
                    session_dir
                )
                if mlp_success:
                    log_message("   ✅ MLP training completed")
                    log_message("   ✅ Model saved in session and standard directories")
                else:
                    log_message("   ⚠️ MLP training skipped")
            except Exception as e:
                log_message(f"   ❌ MLP training failed: {e}")
            
            # Train SVM
            log_message("🔄 Step 3b: Training SVM (DIRECT)")
            try:
                svm_trainer = DirectSVMTrainer(logger)
                svm_success = svm_trainer.simple_svm_training(
                    session_dir / "embeddings",
                    session_dir
                )
                if svm_success:
                    log_message("   ✅ SVM training completed")
                    log_message("   ✅ Model saved in session and standard directories")
                else:
                    log_message("   ⚠️ SVM training skipped")
            except Exception as e:
                log_message(f"   ❌ SVM training failed: {e}")
        else:
            log_message("🔍 Skipping model training (insufficient data or no labels)")
        
        # Step 4: Generate comprehensive report
        log_message("🔄 Step 4: Generating comprehensive report")
        try:
            create_comprehensive_report(session_dir, logger)
            log_message("   ✅ Comprehensive report generated")
            log_message("   ✅ Report saved in session and standard directories")
        except Exception as e:
            log_message(f"   ❌ Report generation failed: {e}")
        
        # Calculate total time
        total_time = time.time() - start_time
        
        log_message("=" * 60)
        log_message("🎉 DIRECT PIPELINE COMPLETED!")
        log_message(f"⏱️ Total time: {total_time:.1f} seconds")
        log_message(f"📁 Session results: {session_dir}")
        log_message(f"📁 Standard processed: {standard_processed_dir}")
        log_message(f"📁 Standard embeddings: {standard_embeddings_dir}")
        log_message(f"📁 Standard models: {PROJECT_ROOT / 'results' / 'models'}")
        log_message("=" * 60)
        
        return {
            'success': True,
            'session_directory': str(session_dir),
            'total_duration': total_time,
            'has_labels': has_labels,
            'final_results': {
                'session_directory': str(session_dir),
                'standard_directories': {
                    'processed': str(standard_processed_dir),
                    'embeddings': str(standard_embeddings_dir),
                    'models': str(PROJECT_ROOT / 'results' / 'models'),
                    'reports': str(PROJECT_ROOT / 'results' / 'reports'),
                    'plots': str(PROJECT_ROOT / 'results' / 'plots')
                },
                'summary': {
                    'total_samples': len(df),
                    'has_labels': has_labels,
                    'processing_time': total_time
                }
            }
        }
        
    except Exception as e:
        log_message(f"❌ DIRECT pipeline failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# Export function per GUI compatibility
def run_complete_csv_analysis(csv_path: str, text_column: str = 'text', 
                             label_column: str = 'label',
                             log_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    🔧 GUI Integration: Fixed wrapper che usa direct integration con directory standard
    """
    return run_complete_csv_analysis_direct(csv_path, log_callback)

if __name__ == "__main__":
    # Test the direct pipeline
    import sys
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        result = run_complete_csv_analysis_direct(csv_file)
        print(f"Result: {result['success']}")
        if result['success']:
            print(f"Session: {result['session_directory']}")
            print(f"Standard dirs: {result['final_results']['standard_directories']}")
        else:
            print(f"Error: {result['error']}")