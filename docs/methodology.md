# Methodology

← [Back to README](../README.md)

## 1. Data

| Item | Value |
|---|---|
| Source | Kaggle — [*Sentiment Analysis for Mental Health*](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) (`Combined Data.csv`) |
| Raw rows | 53,043 statements (362 with empty text) |
| Columns used | `statement` (text), `status` (label) |
| Classes | Anxiety, Bipolar, Depression, Normal, Personality disorder, Stress, Suicidal |
| Imbalance | largest : smallest ≈ 15 : 1 (Normal 16,343 vs Personality disorder 1,077) |

Class counts after removing empty rows → [`results/metrics/dataset_class_distribution.csv`](../results/metrics/dataset_class_distribution.csv)

## 2. Text preprocessing

1. **Lower-casing** of every statement.
2. **Pattern removal** with regular expressions: URLs, markdown links (e.g. `[View Poll](…)`), handles, punctuation and special characters.
3. **Tokenisation** with NLTK `word_tokenize`.
4. **Stemming** with the Porter stemmer (classical pipeline only).
5. **Stop words retained.** In mental-health text, negations and pronouns (*not, never, I, me*) change meaning, so they were not removed.
6. **Engineered features:** number of characters and number of sentences per statement, appended to the TF-IDF matrix.

## 3. Train / test protocol

| Pipeline | Split | Train | Test |
|---|---|---|---|
| Classical ML, ensembles, deep learning | 80 / 20, `random_state=101` | 42,434 | 10,609 |
| BERT / RoBERTa | 80 / 20, stratified, `random_state=101` | 42,144 | 10,537 |

**Metrics:** accuracy, macro-averaged precision/recall/F1 (primary), weighted F1, per-class report and confusion matrix.

## 4. Models and settings

### 4.1 Classical ML (TF-IDF)

- `TfidfVectorizer(ngram_range=(1, 2), max_features=50000)` + 2 length features → 50,002 dimensions.
- Models: Bernoulli NB, Multinomial NB, Logistic Regression (L1, `max_iter=2000`), SGD (log loss), Decision Tree, Random Forest (`n_estimators=400`), Extra Trees (`n_estimators=500`), Gradient Boosting (`lr=0.05, n_estimators=300, subsample=0.9`), XGBoost (`lr=0.2, max_depth=7, n_estimators=500, subsample=0.9, tree_method=hist`), LinearSVC (calibrated).

### 4.2 Ensembles

- **Soft voting:** Logistic Regression (L1), LinearSVC (calibrated), XGBoost, Multinomial NB.
- **Stacking:** base learners Logistic Regression (L1), LinearSVC (calibrated), XGBoost, Random Forest; meta-learner Logistic Regression (`C=2.0`, class-weighted).

### 4.3 Deep learning (TensorFlow / Keras)

| Setting | Value |
|---|---|
| Vocabulary | 67,769 tokens (capped) |
| Sequence length | 500 |
| Loss weighting | balanced class weights (e.g. Personality disorder 6.31, Normal 0.46) |
| Optimiser | Adam, lr = 1e-3, `ReduceLROnPlateau` |
| Training | up to 20 epochs, batch 64, `EarlyStopping(patience=4, restore_best_weights=True)` |
| Hardware | Kaggle, 2 × NVIDIA Tesla T4 |

Architectures:

- **1D-CNN** — embedding → parallel `Conv1D` branches with different kernel sizes → pooling → dense.
- **BiLSTM** — embedding → BiLSTM(192, seq) → BiLSTM(128) → dense; dropout 0.2.
- **BiGRU** — same layout with GRU cells *(run interrupted, not reported)*.
- **CNN-BiLSTM** — `Conv1D(256, 5)` → BiLSTM(192) *(implemented, not reported)*.

### 4.4 Transformers (PyTorch / Hugging Face)

| Setting | Value |
|---|---|
| Checkpoint | `bert-base-uncased` (also `roberta-base`, run incomplete) |
| Max length | 256 tokens |
| Batch size | 16 |
| Epochs | 4 |
| Learning rate | 2e-5, weight decay 0.01 |
| Loss | cross-entropy with balanced class weights (custom `WeightedTrainer`) |
| Model selection | best epoch by macro-F1 on the evaluation set (`load_best_model_at_end=True`) |
| Hardware | Google Colab GPU |

> Note: model selection uses the same split that is reported as the test set, so BERT's score may be slightly optimistic. A separate validation split is listed under future work.
