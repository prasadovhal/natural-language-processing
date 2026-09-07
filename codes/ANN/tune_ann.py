import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch

df = pd.read_csv('./Real_Combine.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# hyperparametes
# number of hidden layers
# number of neurons in each hidden layers
# learning rate

def build_model(hp):
    model = keras.Sequential()
    
    for i in range(hp.Int('num_layers', 2,10)):
        model.add(layers.Dense(units=hp.Int('units_'+str(i),
                                            min_value=32,
                                            max_value=256,
                                            step=32),
                               activation='relu',))
        model.add(layers.Dense(units=1, activation='linear'))
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2,1e-3, 1e-4])),
                      loss='mean_absolute_error',
                      metrics=['mean_absolute_error'])
        
    return model


tuner = RandomSearch(
    build_model, 
    objective='val_mean_absolute_error',
    max_trials=5,
    executions_per_trial=3,
    directory='NLP_ann_tune',
    project_name='Air Quality index'    
)

tuner.search_space_summary()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

tuner.search(X_train, y_train,
             epochs=5,
             validation_data=(X_test, y_test))

tuner.results_summary()