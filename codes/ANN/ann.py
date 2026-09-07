import numpy as np
import pandas as pd

df = pd.read_csv('./ANN/Churn_Modelling.csv')
X = df.iloc[:,3:13]
y = df.iloc[:,13]

geo = pd.get_dummies(X['Geography'], drop_first=True)
gender = pd.get_dummies(X['Gender'], drop_first=True)

X.drop(['Geography', 'Gender'],axis=1,inplace=True)

X = pd.concat([X, geo, gender],axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, PReLU, ELU, Dropout

clf = Sequential()
clf.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu',input_dim=11))
clf.add(Dropout(0.3))
clf.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))
clf.add(Dropout(0.4))
clf.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_hist = clf.fit(X_train, y_train, validation_split=0.33, batch_size=10,epochs=100)

y_pred = clf.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)

import matplotlib.pyplot as plt

print(model_hist.history.keys())
plt.plot(model_hist.history['accuracy'])
plt.plot(model_hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(model_hist.history['loss'])
plt.plot(model_hist.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()