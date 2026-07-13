# Credit Card Fraud Detection Pipeline

**DecodeLabs Data Science Industrial Training — Batch 2026 | Project 2**

An end-to-end supervised learning pipeline that detects fraudulent credit card transactions in a highly imbalanced dataset (~0.17% fraud), comparing Logistic Regression and Random Forest with SMOTE-based oversampling and hyperparameter tuning.

---

## 🎯 Goal

Build a classifier that reliably catches fraudulent transactions despite extreme class imbalance, without relying on misleading metrics like plain accuracy.

## 📊 Dataset

**Credit Card Fraud Detection** ([Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)) — 284,807 transactions made by European cardholders in September 2013.

- `Time`, `Amount` — the only human-interpretable features
- `V1`–`V28` — PCA-anonymized features (original data is confidential)
- `Class` — target (1 = fraud, 0 = legitimate)
- Only **492 frauds** out of 284,807 transactions (**0.173%**)

## 🛠️ Methodology

1. **EDA** — class distribution, Amount/Time distributions by class, duplicate and missing-value checks.
2. **Train/test split** — stratified 80/20 split performed **before** any scaling or oversampling, to prevent data leakage.
3. **Pipelines** (built with `imblearn.Pipeline`, not plain sklearn, since SMOTE must transform both X and y):
   - **Logistic Regression**: `StandardScaler → SMOTE → LogisticRegression`
   - **Random Forest**: `SMOTE → RandomForestClassifier` (tree-based, no scaling needed)
4. **Hyperparameter tuning** — `GridSearchCV` with `StratifiedKFold`, scored on ROC-AUC.
5. **Evaluation** — Precision, Recall, F1, ROC-AUC, and PR-AUC (Average Precision) on the held-out test set. Accuracy is intentionally **not** used as the headline metric — with 99.8% legitimate transactions, a model that predicts "not fraud" every time would still score ~99.8% accuracy while catching zero fraud.
6. **Visual diagnostics** — ROC and Precision-Recall curves for both models side-by-side, plus Random Forest feature importances.

## ⚡ A note on tuning speed

The original grid search setup (5-fold CV, full SMOTE rebalancing to 100%, and unbounded `max_depth=None` trees) could take hours on large hardware-constrained runs. This pipeline uses a lighter, verified-fast configuration instead:

- 2–3 fold CV instead of 5
- Partial SMOTE rebalancing (`sampling_strategy=0.2`) instead of full 50/50 balance
- Bounded tree depth (`max_depth` capped at 10–15, never `None`)
- Parallelism kept at a single level (`n_jobs=-1` on the classifier only, not nested with `GridSearchCV`)

This full grid search (both models) completes in well under 10 minutes on a single CPU core, and faster on multi-core machines.

## 📈 Results

| Model | ROC-AUC | PR-AUC (Avg Precision) | Fraud Precision | Fraud Recall | Fraud F1 |
|---|---|---|---|---|---|
| **Random Forest** | **0.970** | **0.797** | 0.849 | 0.768 | 0.807 |
| Logistic Regression | 0.962 | 0.669 | 0.053 | 0.874 | 0.100 |

**Random Forest is the stronger model overall** — much higher precision (fewer false alarms) at a comparable recall, and a substantially better PR-AUC, which matters most on this kind of severe imbalance.

Logistic Regression catches slightly more fraud cases (87.4% recall) but at the cost of a huge number of false positives (precision of just 5.3%) — impractical for a real fraud system where every flagged transaction needs manual review.

## 📁 Repository Structure

```
.
├── Fraud_Detection_Pipeline_Project2.ipynb   # full pipeline notebook
├── creditcard.csv                            # dataset (from Kaggle, not included if too large for repo)
└── README.md
```

## ▶️ How to Run

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the same folder as the notebook.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```
3. Run all cells top to bottom in Jupyter / Colab.

## 🧰 Tech Stack

`pandas` · `numpy` · `scikit-learn` (`LogisticRegression`, `RandomForestClassifier`, `GridSearchCV`, `StratifiedKFold`) · `imbalanced-learn` (`SMOTE`, `Pipeline`) · `matplotlib` · `seaborn`

## 🔑 Key Takeaways

- **Data leakage matters**: scaling and SMOTE are fit only on the training fold, never on the full dataset before splitting.
- **Accuracy is a trap** on imbalanced data — ROC-AUC and PR-AUC (especially PR-AUC, which is more informative than ROC-AUC under severe imbalance) are the metrics that actually reflect fraud-catching ability.
- **SMOTE doesn't need to fully balance the classes** — partial rebalancing (e.g. 20% of majority class size) captures most of the benefit at a fraction of the training cost.
- Random Forest's feature importances point to a handful of `V*` components as the most predictive signals, consistent with published analyses of this dataset.

---
*Part of the DecodeLabs Data Science Industrial Training Kit (Batch 2026).*
