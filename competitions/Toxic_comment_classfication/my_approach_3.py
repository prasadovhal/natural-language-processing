import re
import string
import pandas as pd
import numpy as np
from itertools import chain
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score
import warnings
from gensim.models import Word2Vec
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim

warnings.filterwarnings("ignore")


df = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test_labels.csv')
df_testcomments = pd.read_csv('./data/test.csv')
df_test = pd.merge(df_test, df_testcomments, on='id', how='left')

class_labels = list(df.columns[2:])

for class_label in class_labels:
    df_test = df_test[df_test[class_label] != -1]

stemmer = nltk.SnowballStemmer("english")
stop_words = stopwords.words('english')

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\n', ' ', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

def preprocess_data(text):
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    # Remove stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    # Stemm all the words in the sentence
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df['comment_text_clean'] = df['comment_text'].apply(preprocess_data)
df_test['comment_text_clean'] = df_test['comment_text'].apply(preprocess_data)


comments = df['comment_text_clean'].tolist()
comments_test = df_test['comment_text_clean']
model = Word2Vec(sentences=comments, workers=12)
word_vectors = model.wv

def get_document_vector(document_tokens):
    document_vector = np.zeros((model.vector_size,))
    n_words = 0
    for token in document_tokens:
        if token in model.wv.key_to_index:
            document_vector += model.wv.get_vector(token)
            n_words += 1
    if n_words > 0:
        document_vector /= n_words # get average
    return document_vector


document_vectors_train = [get_document_vector(doc) for doc in comments]
X_train = np.array(document_vectors_train)

document_vectors_test = [get_document_vector(doc) for doc in comments_test]
X_test = np.array(document_vectors_test)

y_train= df[class_labels]['toxic']
y_test = df_test[class_labels]['toxic']


class LSTM_model(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.3)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        
    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        # x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x)
        out = self.linear(ht[-1])
        return out
    
batch_size = 512
lr = 2e-1
train_target = torch.tensor(y_train.values)
test_target = torch.tensor(y_test.values)
train = TensorDataset(torch.tensor(X_train), train_target)
test = TensorDataset(torch.tensor(X_test), test_target)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)


model = LSTM_model(vocab_size=len(X_train), embedding_dim=300, hidden_dim=5)
model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr)

def train_epoch(model, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    model.train()
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x, y in dataloader: 
        # Move tensor to the proper device
        x = x.to(device)
        y_pred = model(x.int())
        # Evaluate loss
        print(y_pred.cpu().data.numpy()[:,0])
        auc_val = roc_auc_score(y, y_pred.cpu().data.numpy()[:,0])
        # Backward pass
        optimizer.zero_grad()
        auc_val.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (auc_val.item()))
        train_auc_val+=auc_val.item()

    return train_loss / len(dataloader.dataset)

num_epochs = 3
for epoch in range(num_epochs):
    train_loss = train_epoch(model,device,train_loader,optim)
    print('\n EPOCH {}/{} \t train loss {:.3f}'.format(epoch + 1, num_epochs,train_loss))
