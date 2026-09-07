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


# TF-IDF word embedding
vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)

X = vectorizer.fit_transform(df['comment_text_clean'])
X_test = vectorizer.transform(df_test['comment_text_clean'])

#word 2 vec
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

## Label powerset data strategy : didnt work
def BinToDec(x):
    summation = 0
    for i in np.arange(len(x)):
        summation = summation + x[i] * 2**(len(x)-1-i)
    return summation

y_multiclass = [BinToDec(y.iloc[j,:]) for j in range(len(y))]
y_multiclass_test = [BinToDec(y_test.iloc[j,:]) for j in range(len(y_test))]

XGB = XGBClassifier(learning_rate=1, 
                    n_estimators=5,  
                    tree_method = "gpu_hist",
                    objective='binary:logistic')

rs = RandomizedSearchCV(XGB, 
                        param_distributions=params, 
                        n_iter=200, 
                        scoring='average_precision')

rs.fit(X, y_multiclass)


## classifier chains data strategy

X_train = pd.DataFrame(X.toarray())

pred = dict()
auc = dict()
for label in class_labels:
    print(label)
    y_train = y[label]
    
    wt = sum(y_train == 1) / sum(y_train == 0)
    XGB = XGBClassifier(learning_rate=1, 
                        n_estimators=5,  
                        scale_pos_weight = 1/wt,
                        tree_method = "gpu_hist",
                        objective='binary:logistic')
    
    rs = RandomizedSearchCV(XGB, 
                            param_distributions=params, 
                            n_iter=200, 
                            scoring='average_precision')
    
    rs.fit(X_train, y_train)
    
    pred[label] = rs.predict(X_test)
    auc[label] = roc_auc_score(y_test[label].values, pred[label])

    X_train = pd.concat([X_train, y_train], axis=1)
    X_test = pd.concat([X_test, y_test[label]], axis=1)


# RAKEL
k_lable_sets = 6
rakel_object = Rakel(X, y)
labels = rakel_object.createCombinantions(k_lable_sets)

def Multi_label_to_multi_class(y,i):
    y_multilabel = y.iloc[:,list(i)]
    y_multiclass = [BinToDec(y_multilabel.iloc[j,:]) for j in range(len(y_multilabel))]
    y_op = pd.Series(y_multiclass)
    return y_op

# Multi-class to Multi-label 
def Multi_class_Multi_lable(y_pred_multiclass,i):
    y_pred_multi_label = ['{0:06b}'.format(y_pred_multiclass[j])[-len(i):] for j in range(len(y_pred_multiclass))]
    y_pred_set = pd.DataFrame([list(j) for j in y_pred_multi_label])
    y_pred_set  = y_pred_set.apply(pd.to_numeric)
    y_pred_set.columns = i
    y_pred_df = pd.DataFrame(index=range(0,len(y_pred_set)),columns=range(y_test.shape[1]), dtype='int')
    return y_pred_df,y_pred_set
    
def Voting_function(y_pred_set,pred_list_all_sets,k_lable_sets):
    y_pred_final = pd.DataFrame(index=range(0,len(y_pred_set)),columns=range(y.shape[1]), dtype='int')
    for i in range(y.shape[1]):
        classes = pd.DataFrame(index=range(0,len(y_pred_set)),columns=[0], dtype='int')
        for j in range(k_lable_sets):    
            classes = pd.concat([classes,pd.DataFrame(pred_list_all_sets[j].iloc[:,i])],axis=1)
        classes = classes.iloc[:,1:]
        
        tp = [classes.iloc[j,:].mode().values[0] for j in range(len(classes))]
        y_pred_final.iloc[:,i] = tp
    return y_pred_final


combinations = []    
for i in range(2,y.shape[1]-1):
    combinations.append(list(itertools.combinations(range(0,y.shape[1]), i)))
flat_list = [item for sublist in combinations for item in sublist]
labels = random.choices(flat_list,k = k_lable_sets)

pred_list_all_sets = []
for i in labels:
    print(i)
    y_train = Multi_label_to_multi_class(y,i)
    y_test_con = Multi_label_to_multi_class(y_test,i)

    encode = LabelEncoder()
    y_train = encode.fit_transform(y_train)
    y_test_con= encode.transform(y_test_con)
    
    mod = XGB = XGBClassifier(learning_rate=1,
                            n_estimators=15,
                            tree_method = "gpu_hist")
    
    rs = RandomizedSearchCV(XGB, 
                            param_distributions=params, 
                            n_iter=200, 
                            scoring='average_precision')
    
    rs.fit(X, y_train)
    
    y_pred_encoded = rs.predict(X_test)
    y_pred_multiclass = encode.inverse_transform(y_pred_encoded)
    
    y_pred_df, y_pred_set = Multi_class_Multi_lable(y_pred_multiclass, i)
    y_pred_df.iloc[:,list(i)] = y_pred_set
    pred_list_all_sets.append(y_pred_df)

y_pred_final = Voting_function(y_pred_set, pred_list_all_sets,k_lable_sets)
y_pred_final = y_pred_final.astype('int')
y_pred_final.columns = class_labels

auc= dict()
for label in class_labels:
    auc[label] = roc_auc_score(y_test[label], y_pred_final[label])

auc_final = np.mean(list(auc.values()))

