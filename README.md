# 📘 IMDb Sentiment Analysis - README

## 🧠 Descrizione del Progetto

Il progetto IMDb Sentiment Analysis ha l'obiettivo di classificare automaticamente recensioni cinematografiche come positive o negative. Utilizza modelli MLP e SVM, embedding con MiniLM-L6-v2, e offre una GUI interattiva per analisi su dataset e file. Include:

* Preprocessing e generazione dataset CSV
* Generazione embedding con modelli NLP
* Addestramento MLP e SVM
* Generazione report PDF, grafici, classificazioni
* Analisi file singolo (txt, pdf, docx, csv)
* GUI completa per esecuzione pipeline e visualizzazione risultati

---

## 📁 Struttura del Progetto

```
├── config.yaml                        # Parametri di configurazione
├── main.py                            # Avvio CLI pipeline principale
├── gui_data_dashboard.py              # GUI Streamlit
├── requirements.txt                   # Librerie richieste
├── README.md                          # Questo file
├── data/
│   ├── raw/                           # Dataset originale (es. imdb_raw.csv)
│   ├── processed/                     # CSV processati: train.csv, test.csv, val.csv
│   └── embeddings/                    # Dataset vettorializzati (.npy, metadata)
├── models/                            # Modello MiniLM pre-addestrato
├── results/                           # Risultati pipeline, modelli, report, grafici
├── logs/                              # Log generali
├── scripts/                           # Script principali
│   ├── preprocess.py                  # Pulizia e split dataset
│   ├── embed_dataset.py              # Generazione embedding vettoriali
│   ├── train_mlp.py                  # Addestramento classificatore MLP
│   ├── train_svm.py                  # Addestramento classificatore SVM
│   ├── report.py                     # Generazione PDF e visualizzazioni
│   ├── pipeline_runner.py            # Esecuzione pipeline automatizzata
│   ├── enhanced_utils_unified.py    # Funzioni avanzate (analisi file, salvataggi)
│   ├── unified_pipeline.py          # Pipeline coerente embedding → predizione
│   └── unified_preprocessing.py     # Preprocessing unificato
```
## ⚙️ Comandi principali
```bash
# Preprocessing del dataset IMDb
python scripts/preprocess.py

# Generazione embeddings (MiniLM-L6-v2)
python scripts/embed_dataset.py

# Addestramento MLP
python scripts/train_mlp.py

# Addestramento SVM (GridSearch o modalità fast)
python scripts/train_svm.py

# Generazione report PDF/JSON/PNG
python scripts/report.py

# Avvio GUI (Streamlit)
streamlit run gui_data_dashboard.py

# Pipeline automatizzata da file CSV
python main.py --file data/raw/miofile.csv
```

## 🧠 Modello per embedding
Il modello `MiniLM-L6-v2` si trova in `models/minilm-l6-v2/`, compatibile con `sentence-transformers`.

## 🧪 Dataset
- Inserire i file `.csv` in `data/raw/` o caricarli dalla GUI.
- Dopo preprocessing, i dati vengono suddivisi in `train.csv`, `val.csv`, `test.csv` in `data/processed/`.
---

## ⚙️ Requisiti

Installare le librerie richieste:

```bash
pip install -r requirements.txt
```



## 📄 Dataset

Inserire i file `.csv` nella cartella `data/raw/` con il seguente formato:

```
text,label
"This movie was great!",1
"Terrible plot and poor acting",0
```

Il file `imdb_raw.csv` sarà processato automaticamente.

---

## 📊 Output generati

* `data/processed/*.csv`: dataset puliti
* `data/embeddings/*.npy`: dataset vettoriali
* `results/models/`: modelli addestrati
* `results/plots/`: grafici performance
* `results/reports/`: classificazione, metriche, confusione
* `results/session_*/`: cartelle temporali con esperimenti

---

## 💡 Esempi GUI

* Analisi file .csv/.txt/.docx/.pdf
* Inserimento manuale recensioni
* Visualizzazione metriche, report e grafici
* Nuovo dataset → auto-preprocess → embed → train → salva
* Analisi avanzate con conteggio parole/frasi e topic extraction
* Confronto modelli e distribuzione classi nel tab "Advanced Analysis"

---

## 📚 Librerie principali

* `torch`, `scikit-learn`, `transformers`, `sentence-transformers`
* `nltk`, `pandas`, `numpy`, `matplotlib`, `streamlit`, `plotly`

---

## 👤 Autore

Samuele Losio · Università di Chieti-Pescara · 2025

---

## 📎 Note finali

* Tutti gli script sono indipendenti ma interoperabili
* Dataset supportati: solo `.csv`
* Tutte le analisi sono salvate automaticamente in `results/`
* Versione Codex/ChatGPT compatibile con repository pubblica
