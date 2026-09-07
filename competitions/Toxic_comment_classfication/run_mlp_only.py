"""Run only the PyTorch MLP approach and append to existing results."""
import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')

import re, string, json, time
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, hamming_loss
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

CLASS_LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print("Loading data...")
df = pd.read_csv('./data/train.csv')
df_test_labels = pd.read_csv('./data/test_labels.csv')
df_test_comments = pd.read_csv('./data/test.csv')
for lbl in CLASS_LABELS:
    df_test_labels = df_test_labels[df_test_labels[lbl] != -1]
df_test = pd.merge(df_test_labels, df_test_comments, on='id', how='left')
y_train = df[CLASS_LABELS]
y_test  = df_test[CLASS_LABELS]

stemmer = nltk.SnowballStemmer("english")
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words and len(w) > 1]
    return ' '.join(words)

print("Preprocessing...")
df['text_clean'] = df['comment_text'].apply(preprocess)
df_test['text_clean'] = df_test['comment_text'].apply(preprocess)

all_text = pd.concat([df['text_clean'], df_test['text_clean']])
vec_nn = TfidfVectorizer(
    sublinear_tf=True, strip_accents='unicode', analyzer='word',
    token_pattern=r'\w{1,}', stop_words='english',
    ngram_range=(1, 1), max_features=5000,
)
vec_nn.fit(all_text)
X_tr_nn = vec_nn.transform(df['text_clean'])
X_te_nn = vec_nn.transform(df_test['text_clean'])

class SparseTFIDFDataset(Dataset):
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
print(f"Device: {device}")

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 256),   nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, out_dim),  nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)

mlp = MLP(X_tr_nn.shape[1]).to(device)
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCELoss()

EPOCHS = 5
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    mlp.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(mlp(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    print(f"  Epoch {epoch}/{EPOCHS}  loss={epoch_loss/X_tr_nn.shape[0]:.4f}")
elapsed = time.time() - t0

mlp.eval()
all_probs, all_preds = [], []
with torch.no_grad():
    for xb, _ in test_loader:
        probs = mlp(xb.to(device)).cpu().numpy()
        all_probs.append(probs)
        all_preds.append((probs >= 0.5).astype(int))

y_probs = np.vstack(all_probs)
y_pred  = np.vstack(all_preds)

auc = round(float(roc_auc_score(y_test, y_probs, average='macro', multi_class='ovr')), 4)
result = {
    'model': 'TF-IDF (5K) + PyTorch MLP (5 epochs)',
    'accuracy':        round(float(accuracy_score(y_test, y_pred)), 4),
    'f1_macro':        round(float(f1_score(y_test, y_pred, average='macro', zero_division=0)), 4),
    'f1_micro':        round(float(f1_score(y_test, y_pred, average='micro', zero_division=0)), 4),
    'precision_macro': round(float(precision_score(y_test, y_pred, average='macro', zero_division=0)), 4),
    'recall_macro':    round(float(recall_score(y_test, y_pred, average='macro', zero_division=0)), 4),
    'roc_auc_macro':   auc,
    'hamming_loss':    round(float(hamming_loss(y_test, y_pred)), 4),
    'train_time_sec':  round(elapsed, 1),
    **{f'auc_{lbl}': round(float(roc_auc_score(y_test[lbl], y_probs[:, i])), 4)
       for i, lbl in enumerate(CLASS_LABELS)},
}
print(f"ROC-AUC: {auc:.4f}  |  F1-macro: {result['f1_macro']:.4f}  |  {elapsed:.0f}s")

# Combine with previous results
existing = []
try:
    with open('metrics_results.json') as f:
        existing = json.load(f)
except FileNotFoundError:
    pass

all_results = existing + [result]
with open('metrics_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("Saved to metrics_results.json")
