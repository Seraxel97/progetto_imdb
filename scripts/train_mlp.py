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

def setup_logging(log_dir):
    """Setup logging configuration"""
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
        
        logger.info(f"Grafici e report salvati in: {output_dir}")
        
        return {
            'metrics_file': str(metrics_file),
            'confusion_matrix_plot': str(cm_plot_path),
            'training_history_plot': str(history_plot_path),
            'classification_report': str(class_report_file),
            'summary_file': str(summary_file)
        }
        
    except Exception as e:
        logger.error(f"Errore nel salvataggio di metriche e grafici: {str(e)}")
        raise

def train_model(embeddings_dir, output_dir, epochs=100, lr=0.001, batch_size=32, logger=None):
    """Funzione principale di training"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # Crea directory di output
        output_dir = Path(output_dir)
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Usando device: {device}")
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
            'timestamp': datetime.now().isoformat()
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
            'saved_files': saved_files
        }
        
    except Exception as e:
        logger.error(f"Errore durante il training: {str(e)}")
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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train MLP model for sentiment analysis')
    
    # Required arguments
    parser.add_argument('--embeddings-dir', type=str, required=True,
                       help='Directory containing embedding files (.npy)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save model, plots, and metrics')
    
    # Optional arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Directory for log files (default: output-dir/logs)')
    
    return parser.parse_args()

def main():
    """Main function for CLI usage"""
    args = parse_arguments()
    
    # Setup logging
    log_dir = args.log_dir if args.log_dir else Path(args.output_dir) / "logs"
    logger = setup_logging(log_dir)
    
    logger.info("=" * 60)
    logger.info("MLP TRAINING STARTED")
    logger.info("=" * 60)
    logger.info(f"Embeddings dir: {args.embeddings_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {device}")
    
    try:
        # Verifica che la directory di embeddings esista
        embeddings_dir = Path(args.embeddings_dir)
        if not embeddings_dir.exists():
            raise FileNotFoundError(f"Directory embeddings non trovata: {embeddings_dir}")
        
        # Avvia il training
        result = train_model(
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            logger=logger
        )
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETATO CON SUCCESSO!")
        logger.info("=" * 60)
        logger.info(f"Modello salvato: {result['model_path']}")
        logger.info(f"Accuracy finale: {result['performance']['final_val_accuracy']:.4f}")
        logger.info(f"Files salvati: {list(result['saved_files'].keys())}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione: {str(e)}")
        return 1

if __name__ == "__main__":
    # Determina il path base del progetto per compatibilità legacy
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # Se chiamato senza argomenti, usa il comportamento legacy
    if len(sys.argv) == 1:
        print(f"Working directory: {os.getcwd()}")
        print(f"Script location: {__file__}")
        print(f"Project root: {project_root}")
        
        # Verifica che i file di embedding esistano (comportamento legacy)
        embeddings_dir = project_root / "data" / "embeddings"
        required_files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"]
        
        missing_files = []
        for file in required_files:
            if not (embeddings_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"\n❌ File mancanti in {embeddings_dir}:")
            for file in missing_files:
                print(f"  - {file}")
            print("\n💡 Genera prima gli embeddings con:")
            print("python scripts/embed_dataset.py")
            sys.exit(1)
        
        print("\n✅ Tutti i file necessari sono presenti.")
        print("Avvio training con parametri di default...")
        
        # Setup logging per legacy mode
        logger = setup_logging(project_root / "results" / "logs")
        
        # Avvia il training legacy
        try:
            result = train_model(
                embeddings_dir=str(embeddings_dir),
                output_dir=str(project_root / "results"),
                logger=logger
            )
            print(f"\n✅ Training completato! Modello salvato in: {result['model_path']}")
        except Exception as e:
            print(f"\n❌ Errore durante il training: {str(e)}")
            sys.exit(1)
    else:
        # Usa il nuovo comportamento CLI
        sys.exit(main())
