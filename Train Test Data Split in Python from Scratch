#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 15:24:57 2020

@author: lalita
"""

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression as LR
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#%%defining function which randomly shuffle the dataset and divide it in training and test in 0.75:0.25 ratio k times
#the function returns the errors across each split, mean and std across the splits using 3 methods
def my_train_test(method, X,y,pie,k):
#  
    err = []
    L = len(X)
    train_len = round(pie*L)
    test_len = round((1-pie)*L)
    
  
    for i in range(k):
        X = X.sample(frac=1).reset_index(drop=True) 
        y = y.sample(frac=1).reset_index(drop=True) 
        X_train = (X.iloc[:train_len,:]).values
        y_train = (y.iloc[:train_len,:]).values
        
        X_test = (X.iloc[train_len:,:]).values
        y_test = (y.iloc[train_len:,:]).values
        
        
        method.fit(X_train, y_train)
        y_test_pred = method.predict(X_test)
        c = 0
        for jj in range(len(y_test_pred)):
            if y_test[jj] != y_test_pred[jj]:
                c += 1
    
        E = c/(len(y_test_pred))
        err.append(E)
        mean = np.mean(err)
        std = np.mean(err)
    return err, mean, std

