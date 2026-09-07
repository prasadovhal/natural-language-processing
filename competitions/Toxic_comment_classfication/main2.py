import pandas as pd
import tensorflow as tf
import pathlib
import random
import string
import re
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
import sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer 
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.stats import rankdata
import json
import keras_nlp

class Config:
    batch_size = 128
    validation_split = 0.15
    epochs = 10 # Number of Epochs to train
    model_path = "model.tf"
    output_dataset_path = "../input/toxicity-keras-nlp-model"
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    modes = ["training", "inference"]
    mode = modes[1]
    model_name = "distil_bert_base_en_uncased"

config = Config()


df = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test_labels.csv')
df_testcomments = pd.read_csv('./data/test.csv')

X_train, X_val, y_train, y_val = train_test_split(df["comment_text"], 
                                                df[config.labels], 
                                                test_size=config.validation_split)


def make_dataset(X, y, batch_size, mode):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if mode == "train":
       dataset = dataset.shuffle(batch_size * 4) 
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache().prefetch(tf.data.AUTOTUNE).repeat(1)
    return dataset

train_ds = make_dataset(X_train, y_train, batch_size=config.batch_size, mode="train")
valid_ds = make_dataset(X_val, y_val, batch_size=config.batch_size, mode="valid")

for batch in train_ds.take(1):
    print(batch)


def get_model(config):
    encoder = keras_nlp.models.DistilBertBackbone.from_preset(
        "distil_bert_base_en_uncased"
    )
    encoder.trainable = False
    preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset("distil_bert_base_en_uncased")
    inputs = keras.Input(shape=(), dtype=tf.string)
    x = preprocessor(inputs)
    x = encoder(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    output = layers.Dense(6, activation="sigmoid")(x)
    model = keras.Model(inputs, output, name="model")
    model.compile(
        "adam", loss="binary_crossentropy", metrics=["categorical_accuracy", keras.metrics.AUC()]
    )
    return model

model = get_model(config)
model.summary()

if config.mode == config.modes[0]:
    checkpoint = keras.callbacks.ModelCheckpoint(config.model_path, monitor="val_categorical_accuracy", save_best_only=True)
    early_stopping = keras.callbacks.EarlyStopping(patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=5, min_delta=1e-4, min_lr=1e-6)
    model.fit(train_ds, epochs=config.epochs, validation_data=valid_ds, callbacks=[checkpoint, reduce_lr])


if config.mode == config.modes[0]:
    from sklearn.metrics import classification_report
    y_pred = np.array(model.predict(valid_ds) > 0.5, dtype=int)
    cls_report = classification_report(y_val, y_pred)
    print(cls_report)
