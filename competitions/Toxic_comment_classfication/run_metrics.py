"""
Collect performance metrics from all approaches in this project.

Covers:
  - main.py:         TF-IDF + OneVsRest (LogReg, LinearSVC, DecisionTree)
  - third_approch.py: TF-IDF + per-label LogReg with 3-fold CV, MultinomialNB
  - my_approach_2.py: TF-IDF + XGBoost Binary Relevance
  - my_approach_3.py: Replaced Word2Vec LSTM with TF-IDF + PyTorch MLP
                      (gensim not compatible with Python 3.14)

Saves results to metrics_results.json and prints a summary table.
"""

import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings('ignore')

import re
import string
import json
import time
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, hamming_loss
)
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import issparse

# ─────────────────────────── Data loading ────────────────────────────────────

print("Loading data...")
df = pd.read_csv('./data/train.csv')
df_test_labels = pd.read_csv('./data/test_labels.csv')
df_test_comments = pd.read_csv('./data/test.csv')

CLASS_LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

for label in CLASS_LABELS:
    df_test_labels = df_test_labels[df_test_labels[label] != -1]

df_test = pd.merge(df_test_labels, df_test_comments, on='id', how='left')
df_test = df_test[['id', 'comment_text'] + CLASS_LABELS]

print(f"  Train: {len(df):,} rows | Test: {len(df_test):,} rows")

# ─────────────────────────── Preprocessing ───────────────────────────────────

stemmer = nltk.SnowballStemmer("english")
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

def preprocess(text):
    text = clean_text(text)
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words and len(w) > 1]
    return ' '.join(words)

print("Preprocessing text (may take ~10 min for 561K rows)...")
t0 = time.time()
df['text_clean'] = df['comment_text'].apply(preprocess)
df_test['text_clean'] = df_test['comment_text'].apply(preprocess)
print(f"  Done in {(time.time()-t0)/60:.1f} min")

# ─────────────────────────── TF-IDF vectorisation ────────────────────────────

print("Fitting TF-IDF vectorizer (unigrams + bigrams, 50K features)...")
t0 = time.time()
all_text = pd.concat([df['text_clean'], df_test['text_clean']])
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=50000,
)
vectorizer.fit(all_text)
X_train_tfidf = vectorizer.transform(df['text_clean'])
X_test_tfidf  = vectorizer.transform(df_test['text_clean'])
y_train = df[CLASS_LABELS]
y_test  = df_test[CLASS_LABELS]
print(f"  Done in {time.time()-t0:.1f}s  |  shape: {X_train_tfidf.shape}")

# ─────────────────────────── Evaluation helper ───────────────────────────────

def evaluate(clf, X_te, y_te, name, train_time=None):
    y_pred = clf.predict(X_te)
    if issparse(y_pred):
        y_pred = y_pred.toarray()

    # ROC-AUC: prefer probability scores
    try:
        y_scores = clf.predict_proba(X_te)
        auc = roc_auc_score(y_te, y_scores, average='macro', multi_class='ovr')
    except AttributeError:
        try:
            y_scores = clf.decision_function(X_te)
            auc = roc_auc_score(y_te, y_scores, average='macro')
        except Exception:
            auc = roc_auc_score(y_te, y_pred, average='macro')

    per_auc  = roc_auc_score(y_te, y_pred, average=None)
    per_f1   = f1_score(y_te, y_pred, average=None, zero_division=0)

    m = {
        'model':            name,
        'accuracy':         round(accuracy_score(y_te, y_pred), 4),
        'f1_macro':         round(f1_score(y_te, y_pred, average='macro',  zero_division=0), 4),
        'f1_micro':         round(f1_score(y_te, y_pred, average='micro',  zero_division=0), 4),
        'precision_macro':  round(precision_score(y_te, y_pred, average='macro', zero_division=0), 4),
        'recall_macro':     round(recall_score(y_te, y_pred, average='macro', zero_division=0), 4),
        'roc_auc_macro':    round(auc, 4),
        'hamming_loss':     round(hamming_loss(y_te, y_pred), 4),
    }
    for i, lbl in enumerate(CLASS_LABELS):
        m[f'f1_{lbl}']  = round(float(per_f1[i]), 4)
        m[f'auc_{lbl}'] = round(float(per_auc[i]), 4)
    if train_time is not None:
        m['train_time_sec'] = round(train_time, 1)
    return m

all_results = []

# ═════════════════════════════════════════════════════════════════════════════
# APPROACH 1  –  TF-IDF + OneVsRest classifiers    (main.py)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("APPROACH 1  |  TF-IDF + OneVsRest  (main.py)")
print("="*60)

approach1_models = {
    'TF-IDF + LogisticRegression (OvR)': LogisticRegression(
        C=1, solver='saga', max_iter=1000, random_state=42, class_weight='balanced', n_jobs=-1),
    'TF-IDF + LinearSVC (OvR)': CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', max_iter=2000, random_state=42)),
    'TF-IDF + DecisionTree (OvR)': DecisionTreeClassifier(
        class_weight='balanced', random_state=42),
}

for name, base_clf in approach1_models.items():
    print(f"  Training {name}...")
    t0 = time.time()
    clf = OneVsRestClassifier(base_clf, n_jobs=-1)
    clf.fit(X_train_tfidf, y_train)
    elapsed = time.time() - t0
    m = evaluate(clf, X_test_tfidf, y_test, name, elapsed)
    all_results.append(m)
    print(f"    ROC-AUC: {m['roc_auc_macro']:.4f}  |  F1-macro: {m['f1_macro']:.4f}  |  {elapsed:.0f}s")

# ═════════════════════════════════════════════════════════════════════════════
# APPROACH 2  –  Per-label LogReg + MultinomialNB with 3-fold CV  (third_approch.py)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("APPROACH 2  |  Per-label TF-IDF + LogReg / NB with CV  (third_approch.py)")
print("="*60)

# ── 2a: 3-fold CV ROC-AUC per label (LogReg) ──────────────────────────────
print("  LogisticRegression 3-fold CV per label...")
cv_scores = {}
for lbl in CLASS_LABELS:
    clf_cv = LogisticRegression(C=1, solver='sag', max_iter=300)
    scores = cross_val_score(clf_cv, X_train_tfidf, df[lbl], cv=3, scoring='roc_auc', n_jobs=-1)
    cv_scores[lbl] = round(float(scores.mean()), 4)
    print(f"    {lbl}: {scores.mean():.4f}  ±{scores.std():.4f}")

mean_cv = round(float(np.mean(list(cv_scores.values()))), 4)
print(f"  Mean CV ROC-AUC: {mean_cv:.4f}")
all_results.append({
    'model': 'TF-IDF + LogReg per-label (3-fold CV)',
    'roc_auc_macro': mean_cv,
    **{f'cv_auc_{k}': v for k, v in cv_scores.items()}
})

# ── 2b: MultinomialNB OvR ─────────────────────────────────────────────────
print("  Training MultinomialNB (OvR)...")
t0 = time.time()
mnb = OneVsRestClassifier(MultinomialNB(), n_jobs=-1)
# NB requires non-negative features; use TF-IDF with sublinear=True (already non-negative)
mnb.fit(X_train_tfidf, y_train)
elapsed = time.time() - t0
m = evaluate(mnb, X_test_tfidf, y_test, 'TF-IDF + MultinomialNB (OvR)', elapsed)
all_results.append(m)
print(f"    ROC-AUC: {m['roc_auc_macro']:.4f}  |  F1-macro: {m['f1_macro']:.4f}  |  {elapsed:.0f}s")

# ─ also train RandomForest (from third_approch.py) ────────────────────────
print("  Training RandomForest (OvR, 100 trees)...")
t0 = time.time()
rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
rf.fit(X_train_tfidf, y_train)
elapsed = time.time() - t0
m = evaluate(rf, X_test_tfidf, y_test, 'TF-IDF + RandomForest (OvR)', elapsed)
all_results.append(m)
print(f"    ROC-AUC: {m['roc_auc_macro']:.4f}  |  F1-macro: {m['f1_macro']:.4f}  |  {elapsed:.0f}s")

# ═════════════════════════════════════════════════════════════════════════════
# APPROACH 3  –  TF-IDF + XGBoost Binary Relevance    (my_approach_2.py)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("APPROACH 3  |  TF-IDF + XGBoost Binary Relevance  (my_approach_2.py / my_approch.py)")
print("="*60)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Using device: {device_str}")

xgb_preds = {}
xgb_aucs  = {}
t0_total  = time.time()
for lbl in CLASS_LABELS:
    print(f"  XGBoost for '{lbl}'...")
    t0 = time.time()
    pos = int((df[lbl] == 1).sum())
    neg = int((df[lbl] == 0).sum())
    spw = neg / max(pos, 1)
    clf_xgb = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        max_depth=5,
        scale_pos_weight=spw,
        eval_metric='logloss',
        random_state=42,
        tree_method='hist',
        device=device_str,
        verbosity=0,
    )
    clf_xgb.fit(X_train_tfidf, df[lbl])
    xgb_preds[lbl] = clf_xgb.predict(X_test_tfidf)
    xgb_aucs[lbl]  = round(float(roc_auc_score(y_test[lbl], xgb_preds[lbl])), 4)
    print(f"    ROC-AUC: {xgb_aucs[lbl]:.4f}  ({time.time()-t0:.0f}s)")

mean_xgb_auc = round(float(np.mean(list(xgb_aucs.values()))), 4)
xgb_y_pred = np.column_stack([xgb_preds[lbl] for lbl in CLASS_LABELS])
all_results.append({
    'model': 'TF-IDF + XGBoost Binary Relevance',
    'accuracy':        round(float(accuracy_score(y_test, xgb_y_pred)), 4),
    'f1_macro':        round(float(f1_score(y_test, xgb_y_pred, average='macro',  zero_division=0)), 4),
    'f1_micro':        round(float(f1_score(y_test, xgb_y_pred, average='micro',  zero_division=0)), 4),
    'precision_macro': round(float(precision_score(y_test, xgb_y_pred, average='macro', zero_division=0)), 4),
    'recall_macro':    round(float(recall_score(y_test, xgb_y_pred, average='macro', zero_division=0)), 4),
    'roc_auc_macro':   mean_xgb_auc,
    'hamming_loss':    round(float(hamming_loss(y_test, xgb_y_pred)), 4),
    'train_time_sec':  round(time.time() - t0_total, 1),
    **{f'auc_{k}': v for k, v in xgb_aucs.items()},
})
print(f"  Mean XGBoost ROC-AUC: {mean_xgb_auc:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# APPROACH 4  –  PyTorch MLP on TF-IDF features    (my_approach_3.py variant)
#   (replaces Word2Vec LSTM; gensim not available on Python 3.14)
#   Uses a compact 5K-feature TF-IDF to keep dense tensors memory-feasible.
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("APPROACH 4  |  PyTorch MLP on TF-IDF features  (my_approach_3.py variant)")
print("="*60)

# 5K-feature vectorizer — keeps dense tensor memory ~11 GB → batched to ~1 GB
print("  Building compact 5K-feature TF-IDF for neural approach...")
vec_nn = TfidfVectorizer(
    sublinear_tf=True, strip_accents='unicode', analyzer='word',
    token_pattern=r'\w{1,}', stop_words='english',
    ngram_range=(1, 1), max_features=5000,
)
vec_nn.fit(all_text)
X_tr_nn = vec_nn.transform(df['text_clean'])
X_te_nn = vec_nn.transform(df_test['text_clean'])

# Convert sparse → dense in mini-batches inside a custom Dataset to avoid OOM
class SparseTFIDFDataset(torch.utils.data.Dataset):
    def __init__(self, X_sparse, y_np):
        self.X = X_sparse
        self.y = torch.tensor(y_np, dtype=torch.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx].toarray()[0], dtype=torch.float32)
        return x, self.y[idx]

BATCH = 512
train_loader = DataLoader(SparseTFIDFDataset(X_tr_nn, y_train.values), batch_size=BATCH, shuffle=True, num_workers=0)
test_loader  = DataLoader(SparseTFIDFDataset(X_te_nn, y_test.values),  batch_size=BATCH, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

in_dim = X_tr_nn.shape[1]
mlp = MLP(in_dim).to(device)
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCELoss()

EPOCHS = 5
print(f"  Training MLP for {EPOCHS} epochs...")
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    mlp.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = mlp(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    print(f"    Epoch {epoch}/{EPOCHS}  loss={epoch_loss/X_tr_nn.shape[0]:.4f}")

elapsed = time.time() - t0

mlp.eval()
all_preds, all_probs = [], []
with torch.no_grad():
    for xb, _ in test_loader:
        probs = mlp(xb.to(device)).cpu().numpy()
        all_probs.append(probs)
        all_preds.append((probs >= 0.5).astype(int))

y_probs_mlp = np.vstack(all_probs)
y_pred_mlp  = np.vstack(all_preds)

auc_mlp = round(float(roc_auc_score(y_test, y_probs_mlp, average='macro', multi_class='ovr')), 4)
all_results.append({
    'model':           'TF-IDF + PyTorch MLP (5 epochs)',
    'accuracy':        round(float(accuracy_score(y_test, y_pred_mlp)), 4),
    'f1_macro':        round(float(f1_score(y_test, y_pred_mlp, average='macro',  zero_division=0)), 4),
    'f1_micro':        round(float(f1_score(y_test, y_pred_mlp, average='micro',  zero_division=0)), 4),
    'precision_macro': round(float(precision_score(y_test, y_pred_mlp, average='macro', zero_division=0)), 4),
    'recall_macro':    round(float(recall_score(y_test, y_pred_mlp, average='macro', zero_division=0)), 4),
    'roc_auc_macro':   auc_mlp,
    'hamming_loss':    round(float(hamming_loss(y_test, y_pred_mlp)), 4),
    'train_time_sec':  round(elapsed, 1),
    **{f'auc_{lbl}': round(float(roc_auc_score(y_test[lbl], y_probs_mlp[:, i])), 4)
       for i, lbl in enumerate(CLASS_LABELS)},
})
print(f"  ROC-AUC: {auc_mlp:.4f}  |  F1-macro: {all_results[-1]['f1_macro']:.4f}  |  {elapsed:.0f}s")

# ═════════════════════════════════════════════════════════════════════════════
# Save & print summary
# ═════════════════════════════════════════════════════════════════════════════
with open('metrics_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)

cols = ['model', 'roc_auc_macro', 'f1_macro', 'f1_micro', 'accuracy', 'hamming_loss']
rows = []
for r in all_results:
    rows.append({c: r.get(c, '—') for c in cols})

summary_df = pd.DataFrame(rows)
summary_df.columns = ['Model', 'ROC-AUC', 'F1-macro', 'F1-micro', 'Accuracy', 'Hamming-Loss']
print(summary_df.to_string(index=False))

print("\nPer-class ROC-AUC breakdown:")
per_class_rows = []
for r in all_results:
    if any(f'auc_{lbl}' in r for lbl in CLASS_LABELS):
        row = {'Model': r['model'][:45]}
        for lbl in CLASS_LABELS:
            row[lbl] = r.get(f'auc_{lbl}', '—')
        per_class_rows.append(row)

if per_class_rows:
    print(pd.DataFrame(per_class_rows).to_string(index=False))

print("\nResults saved → metrics_results.json")
