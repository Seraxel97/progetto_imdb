#!/usr/bin/env python3
"""
MLP Training Script - PIPELINE AUTOMATION COMPATIBLE
Trains Multi-Layer Perceptron models for sentiment analysis with full pipeline integration.

🔧 FIXES APPLIED:
- ✅ Fixed CLI parameter names to match expected pipeline calls
- ✅ Added proper --output-dir parameter instead of only --models-dir
- ✅ Enhanced auto-mode detection and path discovery
- ✅ Improved compatibility with pipeline_runner.py subprocess calls

FEATURES:
- Auto-mode by default when called without arguments
- Smart path detection for embeddings and output directories
- Full compatibility with report.py, GUI, and pipeline automation
- Robust model saving with structured output directories
- Professional logging system with UTF-8 support
- Early stopping and advanced optimization

USAGE:
  python scripts/train_mlp.py                                              # Auto-defaults
  python scripts/train_mlp.py --embeddings-dir data/embeddings --output-dir results
  python scripts/train_mlp.py --embeddings-dir data/embeddings --output-dir results --epochs 50
  python scripts/train_mlp.py --embeddings-dir data/embeddings --output-dir results --fast
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# Dynamic project root detection for flexible execution
try:
    CURRENT_FILE = Path(__file__).resolve()
    if CURRENT_FILE.parent.name == 'scripts':
        PROJECT_ROOT = CURRENT_FILE.parent.parent
    else:
        PROJECT_ROOT = CURRENT_FILE.parent
except Exception:
    PROJECT_ROOT = Path.cwd()

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_logging(log_dir):
    """Setup logging configuration with UTF-8 support"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"mlp_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure stream handler with UTF-8 encoding
    import sys
    stream_handler = logging.StreamHandler(sys.stdout)
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, Exception):
        pass  # Fallback for older Python versions
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            stream_handler
        ]
    )
    
    return logging.getLogger(__name__)

class HateSpeechMLP(nn.Module):
    """Modello MLP per classificazione hate speech"""
    
    def __init__(self, input_dim=384, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super(HateSpeechMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))
            # Activation
            layers.append(nn.ReLU())
            # Dropout (meno dropout negli ultimi layer)
            dropout_rate = dropout if i < len(hidden_dims) - 2 else dropout * 0.7
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def find_embedding_files(embeddings_dir=None, session_dir=None, logger=None):
    """Trova i file di embedding in modo intelligente"""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # Lista dei possibili percorsi
    search_paths = []
    
    # 1. Percorso specificato esplicitamente
    if embeddings_dir:
        search_paths.append(Path(embeddings_dir))
    
    # 2. Percorso nella sessione specifica
    if session_dir:
        search_paths.append(Path(session_dir) / "embeddings")
    
    # 3. Percorsi di default
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    search_paths.extend([
        project_root / "data" / "embeddings",
        project_root / "results" / "embeddings",
        Path("data") / "embeddings",
        Path("results") / "embeddings"
    ])
    
    # Cerca nei percorsi più recenti nelle sessioni
    results_dir = project_root / "results"
    if results_dir.exists():
        session_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("session_")]
        session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Più recenti prima
        
        for session_dir in session_dirs[:3]:  # Controlla le 3 sessioni più recenti
            search_paths.append(session_dir / "embeddings")
    
    required_files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"]
    
    logger.info(f"Cercando file di embedding in {len(search_paths)} percorsi...")
    
    for path in search_paths:
        logger.info(f"Controllo: {path}")
        if path.exists():
            missing_files = []
            for file in required_files:
                if not (path / file).exists():
                    missing_files.append(file)
            
            if not missing_files:
                logger.info(f"✅ Tutti i file trovati in: {path}")
                return path
            else:
                logger.info(f"❌ File mancanti in {path}: {missing_files}")
        else:
            logger.info(f"❌ Percorso non esistente: {path}")
    
    raise FileNotFoundError(
        f"File di embedding non trovati. Cercati in:\n" + 
        "\n".join(f"  - {p}" for p in search_paths) +
        f"\nFile richiesti: {required_files}"
    )

def determine_output_dir(output_dir=None, session_dir=None):
    """Determina la directory di output in modo intelligente"""
    if output_dir:
        return Path(output_dir)
    
    if session_dir:
        return Path(session_dir)
    
    # Crea una nuova sessione se non specificata
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    results_dir = project_root / "results"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = results_dir / f"session_{timestamp}"
    
    return session_dir

def load_data(embeddings_dir, logger):
    """Carica i dati di embedding preprocessati"""
    logger.info("Caricamento dati...")
    
    embeddings_dir = Path(embeddings_dir)

    # Debug: show available files for troubleshooting
    try:
        logger.info(f"Files in embeddings dir: {os.listdir(embeddings_dir)}")
    except Exception:
        logger.warning("Could not list embeddings directory contents")

    # Verifica che i file esistano
    required_files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"]
    missing_files = []
    
    for file in required_files:
        if not (embeddings_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        error_msg = f"File mancanti in {embeddings_dir}: {missing_files}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # Carica i file
        X_train = np.load(embeddings_dir / "X_train.npy")
        y_train = np.load(embeddings_dir / "y_train.npy") 
        X_val = np.load(embeddings_dir / "X_val.npy")
        y_val = np.load(embeddings_dir / "y_val.npy")
        
        logger.info(f"Shape X_train: {X_train.shape}")
        logger.info(f"Shape y_train: {y_train.shape}")
        logger.info(f"Shape X_val: {X_val.shape}")
        logger.info(f"Shape y_val: {y_val.shape}")
        
        # Verifica delle etichette
        logger.info(f"Etichette uniche train: {np.unique(y_train)}")
        logger.info(f"Etichette uniche val: {np.unique(y_val)}")
        
        # Verifica dimensioni attese
        if X_train.shape[1] != 384:
            raise ValueError(f"Dimensione embedding attesa: 384, trovata: {X_train.shape[1]}")
        
        if not set(np.unique(y_train)).issubset({0, 1}):
            raise ValueError(f"Etichette non binarie trovate: {np.unique(y_train)}")
        
        # Converti in tensori PyTorch
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        logger.info("Dati caricati con successo")
        return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
        
    except Exception as e:
        logger.error(f"Errore nel caricamento dei dati: {str(e)}")
        raise

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """Crea i DataLoader per training e validazione"""
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Training per una singola epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in tqdm(train_loader, desc="Training", leave=False):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistiche
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
    """Validazione per una singola epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(val_loader, desc="Validation", leave=False):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            # Salva predizioni per metriche dettagliate
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy, all_predictions, all_targets

def save_metrics_and_plots(history, final_predictions, final_targets, output_dir, logger):
    """Salva metriche e grafici"""
    output_dir = Path(output_dir)
    
    # Crea sottodirectory
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Set matplotlib backend for headless operation
        import matplotlib
        matplotlib.use('Agg')
        
        # 1. Salva metriche in JSON
        metrics_data = {
            'training_history': {
                'train_losses': history['train_losses'],
                'train_accuracies': history['train_accuracies'],
                'val_losses': history['val_losses'],
                'val_accuracies': history['val_accuracies']
            },
            'final_metrics': {
                'final_accuracy': accuracy_score(final_targets, final_predictions),
                'total_epochs': len(history['train_losses']),
                'best_val_loss': min(history['val_losses']),
                'best_val_accuracy': max(history['val_accuracies'])
            }
        }
        
        metrics_file = reports_dir / "mlp_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        logger.info(f"Metriche salvate in: {metrics_file}")
        
        # 2. Classification report
        class_report = classification_report(
            final_targets, final_predictions, 
            target_names=['Negative', 'Positive'],
            output_dict=True
        )
        
        class_report_file = reports_dir / "mlp_classification_report.json"
        with open(class_report_file, 'w') as f:
            json.dump(class_report, f, indent=2)
        
        # Salva anche come testo leggibile
        class_report_txt = reports_dir / "mlp_classification_report.txt"
        with open(class_report_txt, 'w') as f:
            f.write("MLP Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(classification_report(
                final_targets, final_predictions,
                target_names=['Negative', 'Positive']
            ))
        
        # 3. Confusion Matrix
        cm = confusion_matrix(final_targets, final_predictions)
        
        # Salva confusion matrix come immagine
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('MLP Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Negative', 'Positive'])
        plt.yticks(tick_marks, ['Negative', 'Positive'])
        
        # Aggiungi numeri nella matrice
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_plot_path = plots_dir / 'mlp_confusion_matrix.png'
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Training history plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(history['train_losses'], label='Train Loss', color='blue')
        ax1.plot(history['val_losses'], label='Val Loss', color='red')
        ax1.set_title('MLP Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(history['train_accuracies'], label='Train Accuracy', color='blue')
        ax2.plot(history['val_accuracies'], label='Val Accuracy', color='red')
        ax2.set_title('MLP Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salva il plot
        history_plot_path = plots_dir / 'mlp_training_history.png'
        plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Salva summary testuale
        summary_file = reports_dir / "mlp_training_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("MLP Training Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total epochs: {len(history['train_losses'])}\n")
            f.write(f"Final train accuracy: {history['train_accuracies'][-1]:.4f}\n")
            f.write(f"Final validation accuracy: {history['val_accuracies'][-1]:.4f}\n")
            f.write(f"Best validation loss: {min(history['val_losses']):.4f}\n")
            f.write(f"Best validation accuracy: {max(history['val_accuracies']):.4f}\n")
            f.write(f"Final test accuracy: {accuracy_score(final_targets, final_predictions):.4f}\n")
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))
        
        # 6. Salva GUI status file per integrazione
        gui_status_file = output_dir / "mlp_training_status.json"
        gui_status = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'model_type': 'MLP',
            'performance': {
                'accuracy': float(accuracy_score(final_targets, final_predictions)),
                'final_val_accuracy': float(history['val_accuracies'][-1]),
                'best_val_accuracy': float(max(history['val_accuracies'])),
                'total_epochs': len(history['train_losses'])
            },
            'files': {
                'model': 'models/mlp_model.pth',
                'confusion_matrix': 'plots/mlp_confusion_matrix.png',
                'training_history': 'plots/mlp_training_history.png',
                'classification_report': 'reports/mlp_classification_report.json',
                'metrics': 'reports/mlp_metrics.json',
                'summary': 'reports/mlp_training_summary.txt'
            }
        }
        
        with open(gui_status_file, 'w') as f:
            json.dump(gui_status, f, indent=2)
        
        logger.info(f"Grafici e report salvati in: {output_dir}")
        
        return {
            'metrics_file': str(metrics_file),
            'confusion_matrix_plot': str(cm_plot_path),
            'training_history_plot': str(history_plot_path),
            'classification_report': str(class_report_file),
            'summary_file': str(summary_file),
            'gui_status_file': str(gui_status_file)
        }
        
    except Exception as e:
        logger.error(f"Errore nel salvataggio di metriche e grafici: {str(e)}")
        raise

def train_model(embeddings_dir=None, output_dir=None, epochs=100, lr=0.001, batch_size=32, logger=None, session_dir=None):
    """Funzione principale di training con parametri flessibili"""
    if logger is None:
        # Setup logging temporaneo se non fornito
        temp_log_dir = Path("logs") if not output_dir else Path(output_dir) / "logs"
        logger = setup_logging(temp_log_dir)
    
    try:
        # Determina percorsi in modo intelligente
        if not embeddings_dir:
            embeddings_dir = find_embedding_files(session_dir=session_dir, logger=logger)
        else:
            # Verifica che il percorso specificato sia valido
            embeddings_dir = find_embedding_files(embeddings_dir=embeddings_dir, logger=logger)
        
        if not output_dir:
            output_dir = determine_output_dir(session_dir=session_dir)

        train_path = Path(embeddings_dir) / "X_train.npy"
        val_path = Path(embeddings_dir) / "X_val.npy"
        if not train_path.exists() or not val_path.exists():
            raise ValueError("Training data not found. Ensure preprocessing created valid train/val files.")
        
        output_dir = Path(output_dir)
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Usando device: {device}")
        logger.info(f"Directory embeddings: {embeddings_dir}")
        logger.info(f"Directory output: {output_dir}")
        logger.info(f"Parametri training: epochs={epochs}, lr={lr}, batch_size={batch_size}")
        
        # Hyperparameters
        PATIENCE = 10
        
        # Carica i dati
        X_train, y_train, X_val, y_val = load_data(embeddings_dir, logger)
        
        # Crea data loaders
        train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size)
        
        # Crea il modello
        model = HateSpeechMLP(input_dim=384).to(device)
        
        logger.info("Architettura del modello:")
        logger.info(str(model))
        logger.info(f"Parametri totali: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss e optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Tracking delle metriche
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("Inizio training...")
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validation
            val_loss, val_acc, val_predictions, val_targets = validate_epoch(model, val_loader, criterion, device)
            
            # Scheduler step con logging
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                logger.info(f"Learning rate aggiornato da {old_lr:.6f} a {new_lr:.6f}")
            
            # Salva metriche
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Salva il miglior modello
                model_path = models_dir / "mlp_model.pth"
                torch.save(model.state_dict(), model_path)
                logger.info(f"Nuovo miglior modello salvato (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping alla epoch {epoch+1}")
                break
        
        # Carica il miglior modello
        model.load_state_dict(torch.load(models_dir / "mlp_model.pth"))
        
        # Valutazione finale
        logger.info("Valutazione finale...")
        final_val_loss, final_val_acc, final_predictions, final_targets = validate_epoch(
            model, val_loader, criterion, device
        )
        
        logger.info(f"Final Val Loss: {final_val_loss:.4f}")
        logger.info(f"Final Val Accuracy: {final_val_acc:.4f}")
        
        # Converti predizioni per metriche
        final_predictions = np.array(final_predictions).flatten()
        final_targets = np.array(final_targets).flatten()
        
        logger.info("Classification Report:")
        logger.info("\n" + classification_report(final_targets, final_predictions, target_names=['Negative', 'Positive']))
        
        # Salva modello completo per backup
        torch.save(model, models_dir / "mlp_model_complete.pth")
        
        # Salva metadati del modello
        metadata = {
            'model_type': 'MLP',
            'input_dim': 384,
            'hidden_dims': [512, 256, 128, 64],
            'dropout': 0.3,
            'training_params': {
                'epochs': len(train_losses),
                'learning_rate': lr,
                'batch_size': batch_size,
                'optimizer': 'Adam',
                'loss_function': 'BCELoss'
            },
            'performance': {
                'final_val_loss': final_val_loss,
                'final_val_accuracy': final_val_acc,
                'best_val_loss': best_val_loss
            },
            'timestamp': datetime.now().isoformat(),
            'embeddings_dir': str(embeddings_dir),
            'output_dir': str(output_dir)
        }
        
        metadata_file = models_dir / "mlp_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Modello e metadati salvati in: {models_dir}")
        
        # Prepara history per salvataggio
        history = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
        
        # Salva metriche e grafici
        saved_files = save_metrics_and_plots(history, final_predictions, final_targets, output_dir, logger)
        
        logger.info("Training completato con successo!")
        
        return {
            'model_path': str(models_dir / "mlp_model.pth"),
            'metadata_path': str(metadata_file),
            'performance': metadata['performance'],
            'saved_files': saved_files,
            'output_dir': str(output_dir),
            'embeddings_dir': str(embeddings_dir)
        }
        
    except Exception as e:
        logger.error(f"Errore durante il training: {str(e)}")
        # Salva status di errore per la GUI
        if output_dir:
            error_status = {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error_message': str(e),
                'model_type': 'MLP'
            }
            try:
                with open(Path(output_dir) / "mlp_training_status.json", 'w') as f:
                    json.dump(error_status, f, indent=2)
            except:
                pass
        raise

def load_trained_model(model_path):
    """Carica un modello pre-addestrato"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    
    model = HateSpeechMLP(input_dim=384)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_text_embedding(model, embedding):
    """Predici una singola embedding"""
    model.eval()
    with torch.no_grad():
        if isinstance(embedding, np.ndarray):
            embedding = torch.FloatTensor(embedding)
        
        if len(embedding.shape) == 1:
            embedding = embedding.unsqueeze(0)
        
        embedding = embedding.to(device)
        output = model(embedding)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0
        
        return prediction, probability

def parse_arguments():
    """🔧 FIXED: Parse command line arguments with corrected parameter names"""
    parser = argparse.ArgumentParser(description='Train MLP model for sentiment analysis - PIPELINE COMPATIBLE')
    
    # 🔧 FIXED: Corrected parameter names to match pipeline expectations
    parser.add_argument('--embeddings-dir', type=str, default=None,
                       help='Directory containing embedding files (.npy). If not specified, will search automatically.')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save model, plots, and metrics. If not specified, will create session directory.')
    parser.add_argument('--session-dir', type=str, default=None,
                       help='Session directory (used for automatic path detection)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Directory for log files (default: output-dir/logs)')
    
    # Modalità speciali
    parser.add_argument('--auto-mode', action='store_true',
                       help='Run in automatic mode with default parameters')
    parser.add_argument('--fast', action='store_true',
                       help='Run in fast mode with reduced epochs')
    
    return parser.parse_args()

def run_training_auto_mode(session_dir=None, **kwargs):
    """Modalità automatica per integrazione con altri script"""
    # Setup logging minimal per auto mode
    if session_dir:
        log_dir = Path(session_dir) / "logs"
    else:
        log_dir = Path("logs")
    
    logger = setup_logging(log_dir)
    
    logger.info("🤖 Modalità automatica MLP training")
    
    # Parametri di default ottimizzati per auto mode
    default_params = {
        'epochs': kwargs.get('epochs', 50),  # Meno epoch per speed
        'lr': kwargs.get('lr', 0.001),
        'batch_size': kwargs.get('batch_size', 32),
        'session_dir': session_dir
    }
    
    logger.info(f"Parametri auto mode: {default_params}")
    
    try:
        result = train_model(logger=logger, **default_params)
        logger.info("✅ Training automatico completato!")
        return result
    except Exception as e:
        logger.error(f"❌ Errore in modalità automatica: {str(e)}")
        raise

def main():
    """Main function for CLI usage"""
    args = parse_arguments()
    
    # Modalità automatica
    if args.auto_mode:
        return run_training_auto_mode(
            session_dir=args.session_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size
        )
    
    # Fast mode adjustments
    if args.fast:
        args.epochs = min(args.epochs, 30)  # Reduce epochs in fast mode
    
    # Detect no arguments and apply defaults
    no_args_provided = len(sys.argv) == 1
    
    # Apply defaults when no arguments provided
    if no_args_provided or not args.embeddings_dir:
        # Auto-detect embeddings directory
        project_root = PROJECT_ROOT
        default_embeddings = project_root / "data" / "embeddings"
        if no_args_provided or not args.embeddings_dir:
            args.embeddings_dir = str(default_embeddings)
    
    if no_args_provided or not args.output_dir:
        # Auto-detect output directory
        default_output = PROJECT_ROOT / "results"
        if no_args_provided or not args.output_dir:
            args.output_dir = str(default_output)
    
    # Setup logging
    if args.output_dir:
        log_dir = args.log_dir if args.log_dir else Path(args.output_dir) / "logs"
    else:
        log_dir = args.log_dir if args.log_dir else Path("logs")
    
    logger = setup_logging(log_dir)
    
    # Show warnings for auto-applied defaults
    if no_args_provided:
        logger.warning("⚠️ No arguments provided. Using auto-detected directories.")
    else:
        if not args.embeddings_dir or args.embeddings_dir == str(PROJECT_ROOT / "data" / "embeddings"):
            logger.warning("⚠️ Using default embeddings directory: data/embeddings")
        if not args.output_dir or args.output_dir == str(PROJECT_ROOT / "results"):
            logger.warning("⚠️ Using default output directory: results")
    
    logger.info("=" * 60)
    logger.info("MLP TRAINING STARTED")
    logger.info("=" * 60)
    logger.info(f"Embeddings dir: {args.embeddings_dir or 'Auto-detect'}")
    logger.info(f"Output dir: {args.output_dir or 'Auto-create'}")
    logger.info(f"Session dir: {args.session_dir or 'None'}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Fast mode: {args.fast}")
    logger.info(f"Device: {device}")
    
    try:
        # Avvia il training
        result = train_model(
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            logger=logger,
            session_dir=args.session_dir
        )
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETATO CON SUCCESSO!")
        logger.info("=" * 60)
        logger.info(f"Modello salvato: {result['model_path']}")
        logger.info(f"Accuracy finale: {result['performance']['final_val_accuracy']:.4f}")
        logger.info(f"Output directory: {result['output_dir']}")
        logger.info(f"Files salvati: {list(result['saved_files'].keys())}")
        
        return result
        
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Controlla se è chiamato senza argomenti (modalità legacy)
        if len(sys.argv) == 1:
            print("🚀 Avvio MLP training in modalità automatica...")
            print(f"Working directory: {os.getcwd()}")
            print(f"Script location: {__file__}")
            
            result = run_training_auto_mode()
            print(f"\n✅ Training completato! Modello salvato in: {result['model_path']}")
            print(f"📊 Accuracy finale: {result['performance']['final_val_accuracy']:.4f}")
        else:
            # Usa il parser per CLI
            result = main()
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrotto dall'utente")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Errore: {str(e)}")
        sys.exit(1)
