"""Save the results that were captured from the first successful run."""
import json

results = [
    # APPROACH 1 — TF-IDF + OneVsRest (main.py)
    {
        'model': 'TF-IDF + LogisticRegression (OvR)',
        'approach': 'main.py',
        'roc_auc_macro': 0.9664,
        'f1_macro': 0.3481,
        'train_time_sec': 255,
    },
    {
        'model': 'TF-IDF + LinearSVC (OvR)',
        'approach': 'main.py',
        'roc_auc_macro': 0.9657,
        'f1_macro': 0.4804,
        'train_time_sec': 51,
    },
    {
        'model': 'TF-IDF + DecisionTree (OvR)',
        'approach': 'main.py',
        'roc_auc_macro': 0.7993,
        'f1_macro': 0.3516,
        'train_time_sec': 405,
    },
    # APPROACH 2 — Per-label CV (third_approch.py)
    {
        'model': 'TF-IDF + LogReg per-label (3-fold CV)',
        'approach': 'third_approch.py',
        'roc_auc_macro': 0.9786,
        'cv_auc_toxic': 0.9696,
        'cv_auc_severe_toxic': 0.9849,
        'cv_auc_obscene': 0.9841,
        'cv_auc_threat': 0.9822,
        'cv_auc_insult': 0.9759,
        'cv_auc_identity_hate': 0.9751,
    },
    {
        'model': 'TF-IDF + MultinomialNB (OvR)',
        'approach': 'third_approch.py',
        'roc_auc_macro': 0.9084,
        'f1_macro': 0.2775,
        'train_time_sec': 0,
    },
    {
        'model': 'TF-IDF + RandomForest (OvR)',
        'approach': 'third_approch.py',
        'roc_auc_macro': 0.9536,
        'f1_macro': 0.3986,
        'train_time_sec': 1133,
    },
    # APPROACH 3 — XGBoost Binary Relevance (my_approach_2.py / my_approch.py)
    {
        'model': 'TF-IDF + XGBoost Binary Relevance',
        'approach': 'my_approach_2.py / my_approch.py',
        'roc_auc_macro': 0.9002,
        'auc_toxic': 0.8704,
        'auc_severe_toxic': 0.9254,
        'auc_obscene': 0.9000,
        'auc_threat': 0.9133,
        'auc_insult': 0.8883,
        'auc_identity_hate': 0.9037,
    },
]

with open('metrics_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Saved results for approaches 1–3 to metrics_results.json")
