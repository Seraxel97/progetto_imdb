<<<<<<< HEAD
# IMDb Sentiment Analysis – README

## 🌟 Obiettivo del progetto

Sistema completo per analisi automatica del sentiment su recensioni IMDb, con:

* Classificazione binaria (positivo/negativo)
* Embedding MiniLM o TF-IDF
* Modelli MLP e SVM
* GUI interattiva per input manuale o da file
* Generazione automatica di grafici, CSV e report PDF

 Attualmente:
Funziona con qualsiasi lingua, ma è stata addestrata principalmente su inglese (dataset IMDb).

Se scrive recensioni in italiano o spagnolo, il sistema le capisce, ma con accuratezza un po’ più bassa, soprattutto per il modello MLP.

L’embedding MiniLM è multilingua (accetta input in italiano, francese, spagnolo…)

Tuttavia, il modello di classificazione (MLP e SVM) è stato addestrato su recensioni in inglese

🔁 Soluzioni (facoltative):
⚠️ Se vuole migliorare l’accuratezza in italiano: serve riaddestre i modelli con un dataset italiano

✅ Per test in italiano, puoi comunque usarla: i modelli capiscono abbastanza bene grazie all’embedding multilingua

## 📁 Struttura del progetto

```
progetto_imdb/
├── main.py                  # Avvio CLI o GUI
├── gui_main.py             # GUI interattiva completa
├── config.yaml             # Configurazioni base
├── enhanced_utils.py       # Utility avanzate
├── scripts/
│   ├── preprocess.py       # Pulizia e split dataset
│   ├── embed_dataset.py    # Embedding MiniLM / TF-IDF
│   ├── train_mlp.py        # Addestramento MLP
│   ├── train_svm.py        # Addestramento SVM
│   ├── predictor.py        # Analisi e predizioni
│   ├── report.py           # Generazione report PDF e grafici
│   └── file_handler.py     # Gestione file input/output
├── data/
│   ├── raw/                # IMDb originale
│   ├── processed/          # train/test/val.csv
│   └── embeddings/         # X/y già trasformati
├── models/                 # MiniLM salvato
├── results/                # Cartelle generate per ogni analisi
├── logs/                   # Log automatici
└── requirements.txt        # Librerie necessarie
```

---

## 🧠 Modelli usati

* **Embedding**: MiniLM (`all-MiniLM-L6-v2`) o TF-IDF
* **MLP**: 2 layer + ReLU, addestrato con AdamW
* **SVM**: SVC kernel RBF + scaler

---

## 🚀 Come eseguire

### ▶️ Da terminale (CLI):

```bash
python main.py --cli
```

* Preprocessa
* Embedda
* Addestra MLP e SVM
* Valida

### 🖥️ Da interfaccia grafica (GUI):

```bash
python gui_main.py
```

* Tab per scrivere testo manuale ✍️
* Caricamento file `.txt`, `.csv`, `.pdf`, `.docx`, `.jpg`
* Salvataggio automatico output in `/results` con grafici, CSV, PDF

---

## 📈 Output generato

Ogni analisi (file o testo) crea una cartella unica:

```
results/YYYY-MM-DD_HHMMSS_nomefile/
├── report.pdf
├── predictions.csv
├── plots/*.png
├── metadata.json
└── logs.txt
```

---

## 📦 Requisiti

```txt
Python >= 3.10
transformers
scikit-learn
sentence-transformers
pandas, matplotlib, seaborn
python-docx, pillow, weasyprint, PyMuPDF, textract
```

Installa tutto con:

```bash
pip install -r requirements.txt
```

---

## 👨‍🏫 Per i professori

* Il progetto funziona sia in modalità CLI che GUI
* Analizza testo, documenti, immagini OCR
* Il sistema è modulare e documentato
* Tutti i risultati sono salvati automaticamente
* I modelli sono già addestrati e usabili

---

## 📬 Contatti

* Autore: Samuele Losio
* Corso: Reti Neurali / Scientific Programming
* Università: Università degli Studi di Chieti-Pescara
* Email: [samuele97losio@gmail.com](mailto:samuele97losio@gmail.com)
* Anno: 2025
=======
# progetto_imdb
"IMDb Sentiment Analysis – Embedding, MLP &amp; SVM training, GUI, PDF report, batch &amp; file analysis."
>>>>>>> 7accba77aa0576b1a81eb714703c225206669d7e
