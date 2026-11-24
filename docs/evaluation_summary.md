# Evaluation summary

← [Back to README](../README.md)

## Overall ranking (Macro-F1)

| Rank | Model | Family | Accuracy | Macro-F1 |
|:---:|---|---|:---:|:---:|
| 1 | BERT-base-uncased | Transformer | 0.837 | **0.826** |
| 2 | XGBoost | Classical ML | 0.799 | 0.773 |
| 3 | Soft Voting Ensemble | Ensemble | 0.793 | 0.769 |
| 4 | Stacking Ensemble | Ensemble | 0.789 | 0.761 |
| 5 | Logistic Regression (L1) | Classical ML | 0.772 | 0.747 |
| 9 | 1D-CNN | Deep Learning | 0.740 | 0.691 |
| 10 | BiLSTM | Deep Learning | 0.705 | 0.675 |

Full list: [`results/metrics/model_comparison.csv`](../results/metrics/model_comparison.csv)

## Per-class F1 of the best model in each family

| Class | Test support | BERT | XGBoost | 1D-CNN |
|---|:---:|:---:|:---:|:---:|
| Anxiety | 768 | **0.89** | 0.82 | 0.81 |
| Bipolar | 556 | **0.87** | 0.82 | 0.75 |
| Depression | 3,081 | **0.77** | 0.75 | 0.63 |
| Normal | 3,269 | **0.95** | 0.93 | 0.91 |
| Personality disorder | 215 | **0.80** | 0.72 | 0.49 |
| Stress | 517 | **0.76** | 0.67 | 0.55 |
| Suicidal | 2,131 | **0.75** | 0.69 | 0.69 |

*(Support shown for the BERT test split.)*

## Findings

1. **Contextual embeddings matter most for small, semantically overlapping classes.** BERT's largest gains over XGBoost are on Stress (+0.09), Personality disorder (+0.08) and Anxiety (+0.07); on the easy, high-support Normal class the gap is only +0.02.
2. **Depression ↔ Suicidal is the main error mode.** For BERT, 687 Depression statements are predicted as Suicidal and 364 Suicidal as Depression — together over 60% of all of its errors. The 1D-CNN shows the same pattern more strongly (1,180 Depression → Suicidal).
3. **Strong sparse baselines remain competitive.** XGBoost reaches 94% of BERT's Macro-F1 at a fraction of the training cost and without a GPU.
4. **Recurrent and convolutional nets trained from scratch overfit.** The BiLSTM reached ≈0.87 training accuracy but ≈0.75 validation accuracy; validation loss plateaued around 0.79–0.86 while training loss kept falling.
5. **Class weighting trades precision for recall on minority classes.** The stacking ensemble (class-weighted meta-learner) has the highest minority recall among classical models (Stress 0.82, Personality disorder 0.84) but lower precision on them.
6. **Linear models without enough iterations fail.** SGD and LinearSVC did not converge and predicted almost only Normal/Suicidal/Depression — a reminder to check convergence warnings before reporting.

## Threats to validity

- Slightly different test sets between notebooks (10,609 vs 10,537 rows).
- BERT's best epoch is selected on the reported split.
- One random split and seed; no confidence intervals.
- Oversampling was configured but did not run in the saved classical-ML run.
- Labels reflect the source datasets, not clinical diagnosis; possible duplicate or near-duplicate statements across splits were checked in EDA but not removed.

## Future work

- A fixed train / validation / test split shared by every notebook, with 5-fold cross-validation and bootstrap confidence intervals.
- Complete the RoBERTa, BiGRU and CNN-BiLSTM runs; add domain-adapted checkpoints such as `mental/mental-bert-base-uncased`.
- Hierarchical or cost-sensitive handling of the Depression/Suicidal boundary.
- Calibration analysis and explainability (SHAP / attention attribution) for error review.
