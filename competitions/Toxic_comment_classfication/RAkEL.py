#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 00:56:19 2019

@author: prasad
"""

import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import random

class Rakel:
    
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        
    # binary to decimal
    def BinToDec(self,x):
        summation = 0
        for i in np.arange(len(x)):
            summation = summation + x[i] * 2**(len(x)-1-i)
        return summation
    
    # Create model and get prediction
    def Model_create(self,X_train,y_train,X_test,y_test):
        model = SVC(kernel='linear', C = 1.0)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        return y_pred
    
    # Create combinantions of labels
    def createCombinantions(self,k_lable_sets):
        combinations = []    
        for i in range(2,self.Y.shape[1]-1):
            combinations.append(list(itertools.combinations(range(0,self.Y.shape[1]), i)))        
        flat_list = [item for sublist in combinations for item in sublist]
        labels = labels = random.choices(flat_list,k = k_lable_sets)
        return labels
     
    # Test train divide
    def Test_Train_Split(self,y,split_data):
        X_train = self.X.loc[np.where(split_data == 1)[0],:]
        X_test = self.X.loc[np.where(split_data != 1)[0],:]
        y_train = y.loc[np.where(split_data == 1)[0]]
        y_test = y.loc[np.where(split_data != 1)[0]]
        return X_train,X_test,y_train,y_test
    
    # Multi - label to multi-class
    def Multi_label_to_multi_class(self,i):
        y_multilabel = self.Y.iloc[:,list(i)]
        y_multiclass = [self.BinToDec(y_multilabel.iloc[j,:]) for j in range(len(y_multilabel))]
        y = pd.Series(y_multiclass)
        return y
    
    # Multi-class to Multi-label 
    def Multi_class_Multi_lable(self,y_pred_multiclass,i):
        y_pred_multi_label = ['{0:06b}'.format(y_pred_multiclass[j])[-len(i):] for j in range(len(y_pred_multiclass))]
        y_pred_set = pd.DataFrame([list(j) for j in y_pred_multi_label])
        y_pred_set  = y_pred_set.apply(pd.to_numeric)
        y_pred_set.columns = i
        y_pred_df = pd.DataFrame(index=range(0,len(y_pred_set)),columns=range(self.Y.shape[1]), dtype='int')
        return y_pred_df,y_pred_set
    
    # Voting function
    def Voting_function(self,y_pred_set,pred_list_all_sets,k_lable_sets):
        y_pred_final = pd.DataFrame(index=range(0,len(y_pred_set)),columns=range(self.Y.shape[1]), dtype='int')
        for i in range(self.Y.shape[1]):
            classes = pd.DataFrame(index=range(0,len(y_pred_set)),columns=[0], dtype='int')
            for j in range(k_lable_sets):    
                classes = pd.concat([classes,pd.DataFrame(pred_list_all_sets[j].iloc[:,i])],axis=1)
            classes = classes.iloc[:,1:]
            
            tp = [classes.iloc[j,:].mode().values[0] for j in range(len(classes))]
            y_pred_final.iloc[:,i] = tp
        return y_pred_final
    
    def CrossValidation(self,split_data,labels,k_lable_sets,Y_test):
        pred_list_all_sets = []
        for i in labels:
            print(i)
            y = self.Multi_label_to_multi_class(i)  
            X_train,X_test,y_train,y_test = self.Test_Train_Split(y,split_data)
            y_pred_multiclass = self.Model_create(X_train,y_train,X_test,y_test)
            y_pred_df, y_pred_set = self.Multi_class_Multi_lable(y_pred_multiclass,i)
            y_pred_df.iloc[:,list(i)] = y_pred_set
            pred_list_all_sets.append(y_pred_df)
        y_pred_final = self.Voting_function(y_pred_set,pred_list_all_sets,k_lable_sets)
        y_pred_final.columns = Y_test.columns
        return y_pred_final
    
#df =  pd.read_csv('emotions.csv')
#X = df.iloc[:,:72]
#X = pd.DataFrame(scale(X))
#Y = df.iloc[:,72:]
#k_lable_sets = 6
#split_data = np.random.choice(2,size=len(Y),replace=True,p=[0.7,0.3])
#
#Y_test = Y.iloc[np.where(split_data != 1)[0],:].reset_index().drop('index',axis=1)
#
#rakel_object = Rakel(X,Y)
#labels = rakel_object.createCombinantions(k_lable_sets)
#y_pred_final = rakel_object.CrossValidation(split_data,labels,k_lable_sets,Y_test)
#
#conf_mat = np.zeros(shape=(2,2))
#for i in range(Y_test.shape[1]):
#    conf_mat += confusion_matrix(Y_test.iloc[:,i],y_pred_final.iloc[:,i])
#    
#hamming_loss = (conf_mat[0,1] + conf_mat[1,0]) / conf_mat.sum()
