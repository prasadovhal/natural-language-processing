import os
import pickle
import re
import zipfile
from collections import Counter
import string
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import dill
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', 50)


df = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test_labels.csv')
df_testcomments = pd.read_csv('./data/test.csv')

class_labels = list(df.columns[2:])

# remove rows with -1 from df_test as it is not used for scoring
print(f'Before removing -1: {df_test.shape}')
for class_label in class_labels:
    df_test = df_test[df_test[class_label] != -1]
print(f'After removing -1: {df_test.shape}')

# merge df_test and df_testcomments on id
df_test = pd.merge(df_test, df_testcomments, on='id', how='left')
# rearraange columns to be the same as df
df_test = df_test[['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

stemmer = nltk.SnowballStemmer("english")

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


df['comment_text_clean'] = df['comment_text'].apply(preprocess_data)
df['comment_text_clean_token'] = df['comment_text_clean'].apply(lambda x: x.split())

df_test['comment_text_clean'] = df_test['comment_text'].apply(preprocess_data)
df_test['comment_text_clean_token'] = df_test['comment_text_clean'].apply(lambda x: x.split())


def graph_multilabel(df1,
                    title='Number of comments having multiple labels',
                    offset=500):
    """
    Plot number of comments having multiple labels

    :param df1: dataframe
    :param title: string
    :param offset: int
    :return: series
    """

    # sum of all labels
    rowSums = df1.iloc[:, 2:].sum(axis=1)
    # count of comments having multiple labels
    multiLabel_counts = rowSums.value_counts()
    # remove count of comments having zero labels
    multiLabel_counts = multiLabel_counts[1:]
    # sort the index
    multiLabel_counts = multiLabel_counts.sort_index(ascending=True)

    sns.barplot(x=multiLabel_counts.index, y=multiLabel_counts.values)
    for i, v in enumerate(multiLabel_counts.values):
        plt.text(i - 0.2, v + offset, str(v), color='black', fontweight='bold')
    plt.title(title)
    plt.ylabel('Number of comments', fontsize=12)
    plt.xlabel('Number of labels', fontsize=12)
    plt.show()
    return multiLabel_counts


def get_class_info(df, df_combine, class_labels):
    """
    Get number of sentences and tokens for each class

    :param df: dataframe
    :param df_combine: dataframe
    :param class_labels: list of strings
    :return: dictionary
    """
    class_info = {}
    for class_label in class_labels:
        data = df[df[class_label] == 1]
        num_sentences = data.shape[0]
        num_tokens = df_combine[df_combine[class_label]
                                == 1]['comment'].apply(len).sum()
        class_info[class_label] = {
            'num_sentences': num_sentences, 'num_tokens': num_tokens}
    return class_info


def plot_sent(class_info,
            title='Number of sentences per class in training data',
            label_offset=500):
    """
    Plot number of sentences per class

    :param class_info: dictionary
    :param title: string
    :param label_offset: int
    :return: None
    """
    counts = [d['num_sentences'] for d in class_info.values()]
    plt.figure(figsize=(10, 8))
    sns.barplot(x=class_labels, y=counts)
    plt.title(title)
    for i, count in enumerate(counts):
        plt.text(i, count + label_offset, count, ha='center', va='top')
    plt.show()


def plot_tokens(class_info,
                title='Number of tokens per class in training data',
                label_offset=500):
    """
    Plot number of tokens per class

    :param class_info: dictionary
    :param title: string
    :param label_offset: int
    :return: None
    """
    counts = [d['num_tokens'] for d in class_info.values()]
    plt.figure(figsize=(10, 8))
    sns.barplot(x=class_labels, y=counts)
    plt.title(title)
    for i, count in enumerate(counts):
        plt.text(i, count + label_offset, count, ha='center', va='top')
    plt.show()


def plot_common_words(counts, title):
    """
    Plot most common words

    :param counts: list of tuples
    :param title: string
    :return: None
    """
    labels = [word for word, _ in counts]
    freqs = [count for _, count in counts]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, freqs, color='blue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()


label_count = graph_multilabel(df.iloc[:, :8],
                                title='Number of comments having multiple labels in train.csv',
                                offset=10)

graph_multilabel(
df_test, title='Number of comments having multiple labels in test.csv', offset=10)

df_combined = pd.DataFrame({'comment': df['comment_text_clean_token'],
                            'toxic': df['toxic'],
                            'severe_toxic': df['severe_toxic'],
                            'obscene': df['obscene'],
                            'threat': df['threat'],
                            'insult': df['insult'],
                            'identity_hate': df['identity_hate']})

df_combined_test = pd.DataFrame({'comment': df_test['comment_text_clean_token'],
                                'toxic': df_test['toxic'],
                                'severe_toxic': df_test['severe_toxic'],
                                'obscene': df_test['obscene'],
                                'threat': df_test['threat'],
                                'insult': df_test['insult'],
                                'identity_hate': df_test['identity_hate']})

df_combined.head()
train_info = get_class_info(df, df_combined, class_labels)
test_info = get_class_info(df_test, df_combined_test, class_labels)

plot_sent(train_info, title='Number of sentences per class in training data')
plot_tokens(train_info, title='Number of tokens per class in training data', label_offset=10000)

plot_sent(test_info, title='Number of sentences per class in test data', label_offset=200)
plot_tokens(test_info, title='Number of tokens per class in test data', label_offset=3000)

for label in class_labels:
    words = []
    for comment in df_combined[df_combined[label] == 1]['comment']:
        words.extend(comment)
    most_common_words = Counter(words).most_common(15)
    plot_common_words(most_common_words, f'Most common words in {label} class in training data')
    print(f'Most common words in {label} class: {most_common_words}')


# Feature Extraction, Model Development and Evaluation

def eval_clf(clf, X_test, y_test, class_labels, name):
    """
    Evaluate classifier performance

    :param clf: classifier
    :param X_test: array
    :param y_test: array
    :param class_labels: list of strings
    :param name: string
    :return: dictionary, array, array
    """
    y_pred = clf.predict(X_test)
    # calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    auc_roc = roc_auc_score(np.array(y_test), y_pred, average='macro')

    # output performance report
    out_dict = {'model': name, 'accuracy': accuracy, 'precision': precision,
                'recall': recall, 'f1': f1, 'auc_roc': auc_roc}

    # output confusion matrix
    conf_matrix = multilabel_confusion_matrix(y_test, y_pred)

    class_report = classification_report(y_test, y_pred,
                                        target_names=class_labels, zero_division=1)

    return out_dict, class_report, conf_matrix


def plot_multilabel_confusion_matrix(conf_matrix, class_labels, model_name):
    """
    Plot multilabel confusion matrix
    
    :param conf_matrix: array
    :param class_labels: list of strings
    :param model_name: string
    :return: None
    """
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

    for cm, label, ax in zip(conf_matrix, class_labels, axs.flatten()):
        sns.heatmap(cm, annot=True, fmt='d', cmap=None, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(label)

    plt.tight_layout()
    plt.suptitle(f'Multilabel Confusion Matrix ({model_name})', y=1.02)
    plt.show()


vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)

X = vectorizer.fit_transform(df_combined['comment'])
X_test = vectorizer.transform(df_combined_test['comment'])

y = df_combined[class_labels]
y_test = df_combined_test[class_labels]


svm = LinearSVC(class_weight='balanced', max_iter=1000, random_state=0)
logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=0)
dt = DecisionTreeClassifier(class_weight='balanced', random_state=0)

init_classifiers = {'SVM': svm, 'LogReg': logreg, 'DecisionTree': dt}
classifiers = {}

metrics = []
class_reports = {}
conf_matrices = {}

for key, classifier in init_classifiers.items():
    classifiers[key] = OneVsRestClassifier(classifier, n_jobs=-1).fit(X, y)
    out_dict, class_report, conf_matrix = eval_clf(classifiers[key], X_test, y_test, class_labels, key)
    metrics.append(out_dict)
    class_reports[key] = class_report
    conf_matrices[key] = conf_matrix

metrics_df = pd.DataFrame(metrics)
best_model_tfidf = classifiers['LogReg']

for model_name, conf_matrix in conf_matrices.items():
    plot_multilabel_confusion_matrix(conf_matrix, class_labels, model_name)


comments = df_combined['comment'].tolist()
comments_test = df_combined_test['comment'].tolist()

y = df_combined[class_labels]
y_test = df_combined_test[class_labels]

# Word Embeddings vectorization

model = Word2Vec(sentences=comments, workers=12)
# model.save('model/w2v.model')
word_vectors = model.wv


def get_document_vector(document_tokens):
    """
    Get document vector by averaging word vectors

    :param document_tokens: list of strings
    :return: array
    """
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

X = normalize(X)
X_test = normalize(X_test)


svm = LinearSVC(class_weight='balanced', max_iter=1000, random_state=0)
logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=0)
dt = DecisionTreeClassifier(class_weight='balanced', random_state=0)

# Train and test models
init_classifiers = {'SVM': svm, 'LogReg': logreg, 'DecisionTree': dt}
classifiers = {}

metrics = []
class_reports = {}
conf_matrices = {}

for key, classifier in init_classifiers.items():
    classifiers[key] = OneVsRestClassifier(classifier, n_jobs=-1).fit(X, y)
    out_dict, class_report, conf_matrix = eval_clf(classifiers[key], X_test, y_test, class_labels, key)
    metrics.append(out_dict)
    class_reports[key] = class_report
    conf_matrices[key] = conf_matrix
    
    
metrics_df = pd.DataFrame(metrics)
best_model_w2v = classifiers['SVM'] # has the best auc_roc score

for model_name, conf_matrix in conf_matrices.items():
    plot_multilabel_confusion_matrix(conf_matrix, class_labels, model_name)

X_test = vectorizer.transform(df_test['comment_text_clean_token'])
y_pred = best_model_tfidf.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, columns=class_labels)

