# IMDb Sentiment Analysis

Il progetto IMDb Sentiment Analysis ha l'obiettivo di classificare automaticamente recensioni cinematografiche come positive, negative o neutre. Utilizza modelli MLP e SVM, embedding con MiniLM-L6-v2, e offre una GUI interattiva per analisi su dataset e file. Include:

* Preprocessing e generazione dataset CSV
* Generazione embedding con modelli NLP
* Addestramento MLP e SVM
* Generazione report PDF, grafici, classificazioni
* Analisi file singolo (txt, pdf, docx, csv)
* GUI completa per esecuzione pipeline e visualizzazione risultati
* Analisi avanzate con keyword, frasi, topics e metriche modello

---

## 📁 Struttura del Progetto

├── config.yaml  
├── main.py  
├── gui_data_dashboard.py  
├── requirements.txt  
├── README.md  
├── data/  
│ ├── raw/  
│ ├── processed/  
│ └── embeddings/  
├── models/  
├── results/  
├── logs/  
├── scripts/  
│ ├── preprocess.py  
│ ├── embed_dataset.py  
│ ├── train_mlp.py  
│ ├── train_svm.py  
│ ├── report.py  
│ ├── pipeline_runner.py  
│ ├── enhanced_utils_unified.py  
│ ├── unified_pipeline.py  
│ └── unified_preprocessing.py  

---

## ⚙️ Comandi principali

```bash
# Preprocessing del dataset IMDb
python scripts/preprocess.py

# Generazione embeddings
python scripts/embed_dataset.py

# Addestramento MLP
python scripts/train_mlp.py

# Addestramento SVM
python scripts/train_svm.py

# Report finale
python scripts/report.py

# GUI interattiva
streamlit run gui_data_dashboard.py

# Pipeline automatica
python main.py --file data/raw/miofile.csv
```

---

## 🧪 Dataset CSV – Requisiti e Formato

Il sistema accetta file `.csv` con recensioni.  
È possibile usare la GUI o il CLI per elaborare i file.

### ✅ Requisiti minimi per l'addestramento:
- Colonna testo (es. `text`, `review`, `comment`, ecc.)
- Colonna etichetta (es. `label`, `sentiment`, `class`, ecc.)
- Almeno 3 righe con testo
- Almeno 2 classi diverse (`positive`, `negative`)

### ⚠️ Se mancano le etichette:
- Il sistema passa in **modalità inferenza**
- I modelli MLP e SVM **non verranno addestrati**

### ✅ Etichette accettate:
| Etichetta nel file | Interpretazione |
|--------------------|------------------|
| `positive`, `good`, `true`, `1`, `like`, `5` | ➝ `1` |
| `negative`, `bad`, `false`, `0`, `dislike`, `1` | ➝ `0` |

Esempio valido:

```csv
text,label
"I loved this movie!",positive
"Terrible acting and boring.",negative
```

---

## 📊 Output generati

- `data/processed/*.csv`: file puliti
- `data/embeddings/*.npy`: vettorializzazione
- `results/models/`: modelli addestrati
- `results/plots/`: grafici di performance
- `results/reports/`: JSON/PDF di valutazione
- `results/direct_analysis_*/`: sessioni separate

---

## 👤 Autore

Samuele Losio · Università di Chieti-Pescara · 2025
