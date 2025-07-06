#!/usr/bin/env python3
"""
Fixed Pipeline Runner - DIRECT INTEGRATION WITHOUT SUBPROCESS ISSUES
Risolve i problemi di subprocess che bloccano la GUI integrando tutto direttamente.

🔧 FIXES APPLIED:
- ✅ Direct integration instead of subprocess calls
- ✅ Better error handling and timeouts
- ✅ Simplified embedding generation
- ✅ Robust fallback mechanisms
- ✅ GUI-friendly progress reporting
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
    """Direct embedding generation without subprocess"""
    
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
    
    def process_csv_to_embeddings(self, input_dir: Path, output_dir: Path):
        """Process CSV files to embeddings directly"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
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
                    
                    # Save embeddings and labels
                    embeddings_file = output_dir / f"X_{split_name}.npy"
                    labels_file = output_dir / f"y_{split_name}.npy"
                    
                    np.save(embeddings_file, all_embeddings)
                    np.save(labels_file, labels)
                    
                    self.logger.info(f"   ✅ Saved: {embeddings_file.name} ({all_embeddings.shape})")
                    
                except Exception as e:
                    self.logger.error(f"   ❌ Error processing {split_name}: {e}")
                    continue
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Embedding generation failed: {e}")
            return False

class DirectMLPTrainer:
    """Direct MLP training without subprocess"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def simple_mlp_training(self, embeddings_dir: Path, output_dir: Path):
        """Simple MLP training without complex dependencies"""
        try:
            models_dir = output_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            
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
            
            # Save model
            model_path = models_dir / "mlp_model.pth"
            torch.save(model.state_dict(), model_path)
            
            # Calculate accuracy
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                predictions = (outputs > 0.5).float()
                accuracy = (predictions == y_tensor).float().mean().item()
            
            self.logger.info(f"   ✅ MLP training completed. Accuracy: {accuracy:.3f}")
            
            # Save metadata
            metadata = {
                'model_type': 'Simple_MLP',
                'accuracy': accuracy,
                'samples': len(X_train),
                'epochs': epochs,
                'timestamp': datetime.now().isoformat()
            }
            
            metadata_file = models_dir / "mlp_metadata.json"
            with open(metadata_file, 'w') as f:
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
    """Direct SVM training without subprocess"""
    
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
            
            model_path = models_dir / "svm_model.pkl"
            joblib.dump(model_package, model_path)
            
            # Save metadata
            metadata = {
                'model_type': 'Simple_SVM',
                'accuracy': accuracy,
                'samples': len(X_train),
                'timestamp': datetime.now().isoformat()
            }
            
            metadata_file = models_dir / "svm_metadata.json"
            with open(metadata_file, 'w') as f:
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

def create_simple_report(session_dir: Path, logger=None):
    """Create a simple report"""
    try:
        reports_dir = session_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect results
        mlp_status_file = session_dir / "mlp_training_status.json"
        svm_status_file = session_dir / "svm_training_status.json"
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'session_directory': str(session_dir),
            'models': {}
        }
        
        # Check MLP results
        if mlp_status_file.exists():
            with open(mlp_status_file, 'r') as f:
                mlp_data = json.load(f)
                report_data['models']['MLP'] = mlp_data
        
        # Check SVM results
        if svm_status_file.exists():
            with open(svm_status_file, 'r') as f:
                svm_data = json.load(f)
                report_data['models']['SVM'] = svm_data
        
        # Save report
        report_file = reports_dir / "evaluation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        if logger:
            logger.info(f"📄 Report saved: {report_file}")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Report generation failed: {e}")
        return False

def run_complete_csv_analysis_direct(csv_path: str, 
                                   log_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    🔧 FIXED: Complete CSV analysis with direct integration (no subprocess issues)
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
        
        # Create subdirectories
        for subdir in ['processed', 'embeddings', 'models', 'plots', 'reports', 'logs']:
            (session_dir / subdir).mkdir(exist_ok=True)
        
        log_message(f"📁 Session directory: {session_dir}")
        
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
                
                # Save splits
                train_df.to_csv(session_dir / "processed" / "train.csv", index=False)
                val_df.to_csv(session_dir / "processed" / "val.csv", index=False)
                test_df.to_csv(session_dir / "processed" / "test.csv", index=False)
                
                log_message(f"   📂 Created splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
            else:
                # Small dataset or no labels
                df.to_csv(session_dir / "processed" / "inference.csv", index=False)
                log_message(f"   📂 Created inference file: {len(df)} samples")
            
        except Exception as e:
            log_message(f"❌ CSV processing failed: {e}")
            return {'success': False, 'error': str(e)}
        
        # Step 2: Generate embeddings
        log_message("🔄 Step 2: Generating embeddings (DIRECT)")
        
        try:
            embedder = DirectEmbeddingGenerator(logger)
            embedding_success = embedder.process_csv_to_embeddings(
                session_dir / "processed",
                session_dir / "embeddings"
            )
            
            if embedding_success:
                log_message("   ✅ Embeddings generated successfully")
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
                else:
                    log_message("   ⚠️ SVM training skipped")
            except Exception as e:
                log_message(f"   ❌ SVM training failed: {e}")
        else:
            log_message("🔍 Skipping model training (insufficient data or no labels)")
        
        # Step 4: Generate report
        log_message("🔄 Step 4: Generating report")
        try:
            create_simple_report(session_dir, logger)
            log_message("   ✅ Report generated")
        except Exception as e:
            log_message(f"   ❌ Report generation failed: {e}")
        
        # Calculate total time
        total_time = time.time() - start_time
        
        log_message("=" * 60)
        log_message("🎉 DIRECT PIPELINE COMPLETED!")
        log_message(f"⏱️ Total time: {total_time:.1f} seconds")
        log_message(f"📁 Results: {session_dir}")
        
        return {
            'success': True,
            'session_directory': str(session_dir),
            'total_duration': total_time,
            'has_labels': has_labels,
            'final_results': {
                'session_directory': str(session_dir),
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

# Export function for GUI compatibility
def run_complete_csv_analysis(csv_path: str, text_column: str = 'text', 
                             label_column: str = 'label',
                             log_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    🔧 GUI Integration: Fixed wrapper that uses direct integration
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
        else:
            print(f"Error: {result['error']}")
