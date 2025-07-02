#!/usr/bin/env python3
"""
🔧 FIXED - MLP Training Script - COMPLETE PIPELINE INTEGRATION
Trains MLP model for sentiment analysis with comprehensive pipeline integration.

🔧 FIXES APPLIED:
- ✅ Compatible with pipeline_runner.py parameter passing
- ✅ Added --fast parameter for quick testing
- ✅ Standardized [TRAIN_MLP] logging format for debugging  
- ✅ Explicit file saving verification with print statements
- ✅ Robust error handling with clear error messages
- ✅ Smart path handling for session directories
- ✅ Real-time progress reporting
- ✅ Fast mode with reduced epochs and larger batches
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

# Imposta device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🔧 FIXED: Logging functions with standardized format
def log_mlp(message: str):
    """Log message with [TRAIN_MLP] prefix for pipeline debugging."""
    print(f"[TRAIN_MLP] {message}")
    logging.info(f"[TRAIN_MLP] {message}")

def log_mlp_error(message: str):
    """Log error message with [TRAIN_MLP] prefix for pipeline debugging."""
    print(f"[TRAIN_MLP] ERROR: {message}")
    logging.error(f"[TRAIN_MLP] ERROR: {message}")

def log_mlp_success(message: str):
    """Log success message with [TRAIN_MLP] prefix for pipeline debugging."""
    print(f"[TRAIN_MLP] SUCCESS: {message}")
    logging.info(f"[TRAIN_MLP] SUCCESS: {message}")

def setup_logging(log_dir):
    """🔧 FIXED: Simplified logging setup for pipeline compatibility"""
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"mlp_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        log_mlp(f"Logging to file: {log_file}")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        log_mlp("Logging to console only")
    
    return logging.getLogger(__name__)

class HateSpeechMLP(nn.Module):
    """🔧 FIXED: Modello MLP per classificazione hate speech"""
    
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

def load_data(embeddings_dir):
    """🔧 FIXED: Carica i dati di embedding preprocessati"""
    log_mlp("Loading embedding data...")
    
    embeddings_dir = Path(embeddings_dir)
    log_mlp(f"Embeddings directory: {embeddings_dir}")
    
    # Verifica che i file esistano
    required_files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"]
    missing_files = []
    
    for file in required_files:
        file_path = embeddings_dir / file
        if not file_path.exists():
            missing_files.append(file)
            log_mlp_error(f"Missing file: {file_path}")
        else:
            log_mlp(f"Found file: {file_path}")
    
    if missing_files:
        error_msg = f"Missing files in {embeddings_dir}: {missing_files}"
        log_mlp_error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # Carica i file
        log_mlp("Loading numpy arrays...")
        X_train = np.load(embeddings_dir / "X_train.npy")
        y_train = np.load(embeddings_dir / "y_train.npy") 
        X_val = np.load(embeddings_dir / "X_val.npy")
        y_val = np.load(embeddings_dir / "y_val.npy")
        
        log_mlp(f"X_train shape: {X_train.shape}")
        log_mlp(f"y_train shape: {y_train.shape}")
        log_mlp(f"X_val shape: {X_val.shape}")
        log_mlp(f"y_val shape: {y_val.shape}")
        
        # Verifica delle etichette
        train_labels = np.unique(y_train)
        val_labels = np.unique(y_val)
        log_mlp(f"Unique train labels: {train_labels}")
        log_mlp(f"Unique val labels: {val_labels}")
        
        # Verifica dimensioni attese
        if X_train.shape[1] != 384:
            error_msg = f"Expected embedding dimension: 384, found: {X_train.shape[1]}"
            log_mlp_error(error_msg)
            raise ValueError(error_msg)
        
        if not set(np.unique(y_train)).issubset({0, 1}):
            error_msg = f"Non-binary labels found: {np.unique(y_train)}"
            log_mlp_error(error_msg)
            raise ValueError(error_msg)
        
        # Converti in tensori PyTorch
        log_mlp("Converting to PyTorch tensors...")
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        log_mlp_success("Data loaded successfully")
        return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
        
    except Exception as e:
        log_mlp_error(f"Error loading data: {str(e)}")
        raise

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """🔧 FIXED: Crea i DataLoader per training e validazione"""
    log_mlp(f"Creating data loaders with batch size: {batch_size}")
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    log_mlp(f"Train batches: {len(train_loader)}")
    log_mlp(f"Validation batches: {len(val_loader)}")
    
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

def save_metrics_and_plots(history, final_predictions, final_targets, output_dir):
    """🔧 FIXED: Salva metriche e grafici con verifica esplicita"""
    output_dir = Path(output_dir)
    
    # 🔧 FIXED: Non creare sottodirectory se output_dir è già la directory target
    # Se output_dir è già 'models', non creare models/models
    if output_dir.name != "models":
        plots_dir = output_dir / "plots"
        reports_dir = output_dir / "reports"
    else:
        # Siamo già nella directory models, usa la parent per plots/reports
        base_dir = output_dir.parent
        plots_dir = base_dir / "plots"
        reports_dir = base_dir / "reports"
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    log_mlp(f"Saving metrics to: {reports_dir}")
    log_mlp(f"Saving plots to: {plots_dir}")
    
    try:
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
        
        # 🔧 FIXED: Verify file was saved
        if metrics_file.exists():
            log_mlp_success(f"Metrics saved: {metrics_file}")
        else:
            log_mlp_error(f"Failed to save metrics: {metrics_file}")
        
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
        
        # 🔧 FIXED: Verify classification report files
        if class_report_file.exists():
            log_mlp_success(f"Classification report saved: {class_report_file}")
        if class_report_txt.exists():
            log_mlp_success(f"Classification report (txt) saved: {class_report_txt}")
        
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
        
        # 🔧 FIXED: Verify confusion matrix plot
        if cm_plot_path.exists():
            log_mlp_success(f"Confusion matrix plot saved: {cm_plot_path}")
        else:
            log_mlp_error(f"Failed to save confusion matrix plot: {cm_plot_path}")
        
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
        
        # 🔧 FIXED: Verify training history plot
        if history_plot_path.exists():
            log_mlp_success(f"Training history plot saved: {history_plot_path}")
        else:
            log_mlp_error(f"Failed to save training history plot: {history_plot_path}")
        
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
        
        # 🔧 FIXED: Verify summary file
        if summary_file.exists():
            log_mlp_success(f"Training summary saved: {summary_file}")
        else:
            log_mlp_error(f"Failed to save training summary: {summary_file}")
        
        log_mlp_success("All metrics and plots saved successfully")
        
        return {
            'metrics_file': str(metrics_file),
            'confusion_matrix_plot': str(cm_plot_path),
            'training_history_plot': str(history_plot_path),
            'classification_report': str(class_report_file),
            'summary_file': str(summary_file)
        }
        
    except Exception as e:
        log_mlp_error(f"Error saving metrics and plots: {str(e)}")
        raise

def get_training_params(fast_mode=False):
    """🔧 FIXED: Get training parameters based on mode"""
    if fast_mode:
        log_mlp("Using FAST mode parameters")
        return {
            'epochs': 10,
            'batch_size': 64,
            'lr': 0.005,
            'patience': 3
        }
    else:
        log_mlp("Using NORMAL mode parameters")
        return {
            'epochs': 100,
            'batch_size': 32,
            'lr': 0.001,
            'patience': 10
        }

def train_model(embeddings_dir, output_dir, epochs=None, lr=None, batch_size=None, fast_mode=False):
    """🔧 FIXED: Funzione principale di training con modalità fast"""
    
    try:
        # 🔧 FIXED: Get parameters based on fast mode
        if fast_mode or epochs is None:
            params = get_training_params(fast_mode)
            epochs = epochs or params['epochs']
            lr = lr or params['lr']
            batch_size = batch_size or params['batch_size']
            patience = params['patience']
        else:
            patience = 10
        
        log_mlp(f"Training parameters:")
        log_mlp(f"  Device: {device}")
        log_mlp(f"  Epochs: {epochs}")
        log_mlp(f"  Learning rate: {lr}")
        log_mlp(f"  Batch size: {batch_size}")
        log_mlp(f"  Patience: {patience}")
        log_mlp(f"  Fast mode: {fast_mode}")
        
        # 🔧 FIXED: Smart output directory handling
        output_dir = Path(output_dir)
        
        # If output_dir ends with 'models', use it directly
        # Otherwise, create a models subdirectory
        if output_dir.name == "models":
            models_dir = output_dir
            log_mlp(f"Using models directory directly: {models_dir}")
        else:
            models_dir = output_dir / "models"
            log_mlp(f"Creating models subdirectory: {models_dir}")
        
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # 🔧 FIXED: Verify models directory was created
        if not models_dir.exists():
            raise RuntimeError(f"Failed to create models directory: {models_dir}")
        log_mlp_success(f"Models directory ready: {models_dir}")
        
        # Carica i dati
        X_train, y_train, X_val, y_val = load_data(embeddings_dir)
        
        # Crea data loaders
        train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size)
        
        # Crea il modello
        log_mlp("Creating MLP model...")
        model = HateSpeechMLP(input_dim=384).to(device)
        
        log_mlp("Model architecture:")
        log_mlp(str(model))
        total_params = sum(p.numel() for p in model.parameters())
        log_mlp(f"Total parameters: {total_params:,}")
        
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
        
        log_mlp("Starting training...")
        
        for epoch in range(epochs):
            log_mlp(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validation
            val_loss, val_acc, val_predictions, val_targets = validate_epoch(model, val_loader, criterion, device)
            
            # Scheduler step con logging
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                log_mlp(f"Learning rate updated from {old_lr:.6f} to {new_lr:.6f}")
            
            # Salva metriche
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            log_mlp(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            log_mlp(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 🔧 FIXED: Salva il miglior modello con verifica
                model_path = models_dir / "mlp_model.pth"
                log_mlp(f"Saving best model to: {model_path}")
                torch.save(model.state_dict(), model_path)
                
                # Verify model was saved
                if model_path.exists():
                    model_size = model_path.stat().st_size / 1024  # KB
                    log_mlp_success(f"Best model saved (val_loss: {val_loss:.4f}, size: {model_size:.1f}KB)")
                else:
                    log_mlp_error(f"Failed to save model: {model_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                log_mlp(f"Early stopping at epoch {epoch+1}")
                break
        
        # 🔧 FIXED: Carica il miglior modello con verifica
        best_model_path = models_dir / "mlp_model.pth"
        if best_model_path.exists():
            log_mlp(f"Loading best model from: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path))
        else:
            log_mlp_error(f"Best model not found: {best_model_path}")
        
        # Valutazione finale
        log_mlp("Performing final evaluation...")
        final_val_loss, final_val_acc, final_predictions, final_targets = validate_epoch(
            model, val_loader, criterion, device
        )
        
        log_mlp(f"Final Val Loss: {final_val_loss:.4f}")
        log_mlp(f"Final Val Accuracy: {final_val_acc:.4f}")
        
        # Converti predizioni per metriche
        final_predictions = np.array(final_predictions).flatten()
        final_targets = np.array(final_targets).flatten()
        
        log_mlp("Classification Report:")
        report = classification_report(final_targets, final_predictions, target_names=['Negative', 'Positive'])
        print(report)  # Print full report for visibility
        
        # 🔧 FIXED: Salva modello completo per backup con verifica
        complete_model_path = models_dir / "mlp_model_complete.pth"
        log_mlp(f"Saving complete model to: {complete_model_path}")
        torch.save(model, complete_model_path)
        
        if complete_model_path.exists():
            complete_size = complete_model_path.stat().st_size / 1024  # KB
            log_mlp_success(f"Complete model saved: {complete_model_path} ({complete_size:.1f}KB)")
        else:
            log_mlp_error(f"Failed to save complete model: {complete_model_path}")
        
        # 🔧 FIXED: Salva metadati del modello con verifica
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
                'loss_function': 'BCELoss',
                'fast_mode': fast_mode
            },
            'performance': {
                'final_val_loss': final_val_loss,
                'final_val_accuracy': final_val_acc,
                'best_val_loss': best_val_loss
            },
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = models_dir / "mlp_metadata.json"
        log_mlp(f"Saving metadata to: {metadata_file}")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if metadata_file.exists():
            log_mlp_success(f"Metadata saved: {metadata_file}")
        else:
            log_mlp_error(f"Failed to save metadata: {metadata_file}")
        
        # Prepara history per salvataggio
        history = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
        
        # Salva metriche e grafici
        try:
            saved_files = save_metrics_and_plots(history, final_predictions, final_targets, output_dir)
            log_mlp_success("Metrics and plots saved successfully")
        except Exception as e:
            log_mlp_error(f"Error saving metrics and plots: {e}")
            saved_files = {}
        
        log_mlp_success("Training completed successfully!")
        
        # 🔧 FIXED: Verification summary
        files_to_check = [
            models_dir / "mlp_model.pth",
            models_dir / "mlp_model_complete.pth", 
            models_dir / "mlp_metadata.json"
        ]
        
        log_mlp("File verification:")
        for file_path in files_to_check:
            if file_path.exists():
                file_size = file_path.stat().st_size / 1024
                log_mlp_success(f"  ✓ {file_path.name} ({file_size:.1f}KB)")
            else:
                log_mlp_error(f"  ✗ {file_path.name} (MISSING)")
        
        return {
            'success': True,
            'model_path': str(models_dir / "mlp_model.pth"),
            'complete_model_path': str(complete_model_path),
            'metadata_path': str(metadata_file),
            'performance': metadata['performance'],
            'saved_files': saved_files,
            'training_epochs': len(train_losses),
            'fast_mode': fast_mode
        }
        
    except Exception as e:
        log_mlp_error(f"Training failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def load_trained_model(model_path):
    """🔧 FIXED: Carica un modello pre-addestrato"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    log_mlp(f"Loading trained model from: {model_path}")
    model = HateSpeechMLP(input_dim=384)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    log_mlp_success("Model loaded successfully")
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
    """🔧 FIXED: Parse command line arguments with fast mode support"""
    parser = argparse.ArgumentParser(description='🔧 FIXED - Train MLP model for sentiment analysis')
    
    # Required arguments
    parser.add_argument('--embeddings-dir', type=str, required=True,
                       help='Directory containing embedding files (.npy)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save model, plots, and metrics')
    
    # 🔧 FIXED: Add fast mode parameter
    parser.add_argument('--fast', action='store_true',
                       help='Use fast mode (fewer epochs, larger batch size)')
    
    # Optional arguments (can be overridden by fast mode)
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (default: 100 normal, 10 fast)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default: 0.001 normal, 0.005 fast)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (default: 32 normal, 64 fast)')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Directory for log files (default: output-dir/logs)')
    
    return parser.parse_args()

def main():
    """🔧 FIXED: Main function for CLI usage with enhanced error handling"""
    args = parse_arguments()
    
    # Setup logging
    log_dir = args.log_dir if args.log_dir else Path(args.output_dir) / "logs"
    logger = setup_logging(log_dir)
    
    log_mlp("=" * 60)
    log_mlp("MLP TRAINING STARTED")
    log_mlp("=" * 60)
    log_mlp(f"Embeddings dir: {args.embeddings_dir}")
    log_mlp(f"Output dir: {args.output_dir}")
    log_mlp(f"Fast mode: {args.fast}")
    log_mlp(f"Custom epochs: {args.epochs}")
    log_mlp(f"Custom learning rate: {args.lr}")
    log_mlp(f"Custom batch size: {args.batch_size}")
    log_mlp(f"Device: {device}")
    
    try:
        # 🔧 FIXED: Verifica che la directory di embeddings esista
        embeddings_dir = Path(args.embeddings_dir)
        if not embeddings_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
        
        log_mlp(f"Embeddings directory verified: {embeddings_dir}")
        
        # 🔧 FIXED: Avvia il training con parametri aggiornati
        result = train_model(
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            fast_mode=args.fast
        )
        
        if result['success']:
            log_mlp("=" * 60)
            log_mlp_success("TRAINING COMPLETED SUCCESSFULLY!")
            log_mlp("=" * 60)
            log_mlp_success(f"Model saved: {result['model_path']}")
            log_mlp_success(f"Final accuracy: {result['performance']['final_val_accuracy']:.4f}")
            log_mlp_success(f"Training epochs: {result['training_epochs']}")
            log_mlp_success(f"Fast mode: {result['fast_mode']}")
            
            if result['saved_files']:
                log_mlp_success(f"Additional files saved: {len(result['saved_files'])}")
                for file_type, file_path in result['saved_files'].items():
                    log_mlp_success(f"  {file_type}: {Path(file_path).name}")
            
            return 0
        else:
            log_mlp_error("Training failed!")
            log_mlp_error(f"Error: {result.get('error', 'Unknown error')}")
            return 1
        
    except Exception as e:
        log_mlp_error(f"Execution error: {str(e)}")
        return 1

if __name__ == "__main__":
    # 🔧 FIXED: Improved legacy compatibility
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # Se chiamato senza argomenti, usa il comportamento legacy
    if len(sys.argv) == 1:
        log_mlp("Running in LEGACY mode...")
        log_mlp(f"Working directory: {os.getcwd()}")
        log_mlp(f"Script location: {__file__}")
        log_mlp(f"Project root: {project_root}")
        
        # Verifica che i file di embedding esistano (comportamento legacy)
        embeddings_dir = project_root / "data" / "embeddings"
        required_files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"]
        
        missing_files = []
        for file in required_files:
            if not (embeddings_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            log_mlp_error(f"Missing files in {embeddings_dir}:")
            for file in missing_files:
                log_mlp_error(f"  - {file}")
            log_mlp("Generate embeddings first with:")
            log_mlp("python scripts/embed_dataset.py")
            sys.exit(1)
        
        log_mlp_success("All required files found.")
        log_mlp("Starting training with default parameters...")
        
        # Setup logging per legacy mode
        logger = setup_logging(project_root / "results" / "logs")
        
        # Avvia il training legacy
        try:
            result = train_model(
                embeddings_dir=str(embeddings_dir),
                output_dir=str(project_root / "results"),
                fast_mode=False
            )
            
            if result['success']:
                log_mlp_success(f"Training completed! Model saved: {result['model_path']}")
            else:
                log_mlp_error(f"Training failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        except Exception as e:
            log_mlp_error(f"Training error: {str(e)}")
            sys.exit(1)
    else:
        # Usa il nuovo comportamento CLI
        sys.exit(main())