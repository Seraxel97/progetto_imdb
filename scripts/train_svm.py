C:\Users\Admin\Desktop\progetto_imdb\scripts>python train_svm.py
2025-07-04 00:29:19,429 - WARNING - ⚠️ No arguments provided. Using default embeddings and output directories.
2025-07-04 00:29:19,430 - INFO - ============================================================
2025-07-04 00:29:19,430 - INFO - SVM TRAINING PIPELINE
2025-07-04 00:29:19,430 - INFO - ============================================================
2025-07-04 00:29:19,430 - INFO - Embeddings dir: C:\Users\Admin\Desktop\progetto_imdb\data\embeddings
2025-07-04 00:29:19,430 - INFO - Output dir: C:\Users\Admin\Desktop\progetto_imdb\results
2025-07-04 00:29:19,430 - INFO - Fast mode: True
2025-07-04 00:29:19,432 - INFO - Grid search: False
2025-07-04 00:29:19,433 - INFO - Parameters: C=1.0, max_iter=10000
2025-07-04 00:29:19,433 - INFO - Loading embeddings from: C:\Users\Admin\Desktop\progetto_imdb\data\embeddings
2025-07-04 00:29:19,433 - INFO - Files in embeddings dir: ['embedding_metadata.json', 'X_test.npy', 'X_train.npy', 'X_val.npy', 'y_test.npy', 'y_train.npy', 'y_val.npy']
2025-07-04 00:29:19,446 - INFO -    ✅ train: 30,000 samples, 384 features
2025-07-04 00:29:19,451 - INFO -    ✅ val: 10,000 samples, 384 features
2025-07-04 00:29:19,456 - INFO -    ✅ test: 10,000 samples, 384 features
2025-07-04 00:29:19,456 - INFO - Successfully loaded embeddings:
2025-07-04 00:29:19,456 - INFO -    📊 Total samples: 50,000
2025-07-04 00:29:19,456 - INFO -    📏 Feature dimension: 384
2025-07-04 00:29:19,457 - INFO -    📊 Train labels: {np.int64(0): np.int64(15000), np.int64(1): np.int64(15000)}
2025-07-04 00:29:19,457 - INFO -    📊 Val labels: {np.int64(0): np.int64(5000), np.int64(1): np.int64(5000)}
2025-07-04 00:29:19,457 - INFO -    📊 Test labels: {np.int64(0): np.int64(5000), np.int64(1): np.int64(5000)}
2025-07-04 00:29:19,458 - INFO - Preparing training data...
2025-07-04 00:29:19,467 - INFO -    📊 Combined train+val: 40,000 samples
2025-07-04 00:29:19,491 - INFO -    ⚡ Fast mode: reduced to 10,000 samples
2025-07-04 00:29:19,491 - INFO -    📊 Fast mode labels: {np.int64(0): np.int64(5000), np.int64(1): np.int64(5000)}
2025-07-04 00:29:19,493 - INFO - Using fast LinearSVC training...
2025-07-04 00:29:19,493 - INFO - Starting fast SVM training...
2025-07-04 00:29:19,493 - INFO -    📋 Parameters: {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 10000, 'random_state': 42}
2025-07-04 00:29:19,493 - INFO -    📊 Training samples: 10,000
2025-07-04 00:29:19,494 - INFO -    📊 Validation samples: 10,000
2025-07-04 00:29:19,494 - INFO -    📏 Features: 384
2025-07-04 00:29:19,494 - INFO -    🔧 Fitting scaler and label encoder...
2025-07-04 00:29:19,541 - INFO -    📊 Label mapping: {np.int64(0): np.int64(0), np.int64(1): np.int64(1)}
2025-07-04 00:29:19,541 - INFO -    🤖 Training LinearSVC...
2025-07-04 00:29:20,641 - INFO -    ✅ Training completed in 1.15 seconds
2025-07-04 00:29:20,641 - INFO -    🔮 Evaluating on validation set...
2025-07-04 00:29:20,656 - INFO -    📊 Validation Results:
2025-07-04 00:29:20,656 - INFO -       Accuracy: 0.8174
2025-07-04 00:29:20,656 - INFO -       F1-Score: 0.8174
2025-07-04 00:29:20,656 - INFO -       Training time: 1.15s
2025-07-04 00:29:20,658 - INFO - Saving SVM model package...
2025-07-04 00:29:20,659 - INFO -    📄 Model package: C:\Users\Admin\Desktop\progetto_imdb\results\models\svm_model.pkl
2025-07-04 00:29:20,659 - INFO -    📄 Metadata: C:\Users\Admin\Desktop\progetto_imdb\results\models\svm_metadata.json
2025-07-04 00:29:20,661 - INFO - Creating training plots and saving metrics...
2025-07-04 00:29:20,661 - INFO - Metrics saved to: C:\Users\Admin\Desktop\progetto_imdb\results\reports\svm_metrics.json
2025-07-04 00:29:21,064 - INFO - Plots and reports saved to: C:\Users\Admin\Desktop\progetto_imdb\results
2025-07-04 00:29:21,064 - INFO - ============================================================
2025-07-04 00:29:21,064 - INFO - SVM TRAINING COMPLETED SUCCESSFULLY!
2025-07-04 00:29:21,065 - INFO - ============================================================
2025-07-04 00:29:21,065 - INFO - Final validation accuracy: 0.8174
2025-07-04 00:29:21,065 - INFO - Final validation F1-score: 0.8174
2025-07-04 00:29:21,065 - INFO - Training time: 1.15 seconds
2025-07-04 00:29:21,065 - INFO - Model saved to: C:\Users\Admin\Desktop\progetto_imdb\results\models\svm_model.pkl
2025-07-04 00:29:21,068 - INFO - ============================================================
2025-07-04 00:29:21,068 - INFO - TRAINING COMPLETED SUCCESSFULLY!
2025-07-04 00:29:21,068 - INFO - ============================================================
2025-07-04 00:29:21,068 - INFO - Model saved: C:\Users\Admin\Desktop\progetto_imdb\results\models\svm_model.pkl
2025-07-04 00:29:21,069 - INFO - Accuracy: 0.8174
2025-07-04 00:29:21,069 - INFO - F1-Score: 0.8174
2025-07-04 00:29:21,069 - INFO - Training time: 1.15s
2025-07-04 00:29:21,069 - INFO - Files saved: ['model_path', 'metadata_path', 'metrics_file', 'confusion_matrix_plot', 'performance_metrics_plot', 'training_summary_plot', 'classification_report_csv', 'classification_report_json', 'summary_file']
