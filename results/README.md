# 📦 Results

Every number and figure reported in the main [README](../README.md) lives here, so the evaluation can be checked without re-running the notebooks.

| Folder | Contents |
|---|---|
| [`metrics/`](metrics) | CSV tables: overall model comparison, per-class F1, per-class precision/recall/F1, dataset class distribution |
| [`classification_reports/`](classification_reports) | scikit-learn classification report for each of the 15 models |
| [`figures/eda/`](figures/eda) | exploratory analysis: class balance, text length, n-grams, polarity, lexical diversity, PCA |
| [`figures/wordclouds/`](figures/wordclouds) | one word cloud per class |
| [`figures/confusion_matrices/`](figures/confusion_matrices) | confusion matrix for each model (rows = actual, columns = predicted) |
| [`figures/model_comparison/`](figures/model_comparison) | cross-model charts: Macro-F1 ranking, per-class F1 heatmap, best-per-family comparison |

## Provenance

| Artefact | Produced by |
|---|---|
| Classical ML, ensembles, 1D-CNN, BiLSTM metrics and confusion matrices | `notebooks/01_eda_classical_ml_deep_learning.ipynb` |
| BERT metrics and confusion matrix | `notebooks/02_transformer_bert.ipynb` |
| EDA figures and word clouds | `notebooks/01_eda_classical_ml_deep_learning.ipynb` |
| `figures/model_comparison/*` and `metrics/*.csv` | compiled from the saved notebook outputs |

Test split: 20% hold-out, `random_state=101` (10,609 rows for notebook 01; 10,537 rows, stratified, for notebook 02).
