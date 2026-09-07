import re
import string
import pandas as pd
import numpy as np
from itertools import chain
import nltk
from nltk.corpus import stopwords
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost.sklearn import XGBClassifier
from transformers import pipeline, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification, AutoTokenizer
import warnings
from RAkEL import Rakel
import itertools
import random
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec

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
X = np.array(document_vectors_train)

document_vectors_test = [get_document_vector(doc) for doc in comments_test]
X_test = np.array(document_vectors_test)

y = df[class_labels]
y_test = df_test[class_labels]

params = {
    'max_depth' : [2,3,4,5],
    'min_child_weight' : [3,4,5,6,7,8],
    'gamma': list(chain.from_iterable((i, i/10) for i in range(5,20))),
    'subsample': [i/10.0 for i in range(5,10)],
    'colsample_bytree':[i/10.0 for i in range(5,10)],
    'reg_alpha': list(chain.from_iterable((i, i/5) for i in range(0,20))),
    'reg_lambda': list(chain.from_iterable((i, i/5) for i in range(0,20))),
    'colsample_bylevel':[i/10.0 for i in range(5,10)],
}


## Binary relevance data strategy
pred = dict()
auc = dict()
for label in class_labels:
    print(label)
    wt = sum(y[label] == 1) / sum(y[label] == 0)
    XGB = XGBClassifier(learning_rate=1, 
                        n_estimators=5,  
                        scale_pos_weight = 1/wt,
                        tree_method = "gpu_hist",
                        objective='binary:logistic')
    
    rs = RandomizedSearchCV(XGB, 
                            param_distributions=params, 
                            n_iter=200, 
                            scoring='average_precision')
    
    rs.fit(X, y[label])
    
    pred[label] = rs.predict(X_test)
    auc[label] = roc_auc_score(y_test[label].values, pred[label])

auc_final = np.mean(list(auc.values()))
