#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:36:37 2020

@author: lalita
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:18:37 2020

@author: lalita
"""



import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import linalg as LA

#%%

class MyLogisticReg2:
    def __init__(self,d):
        
        self.d = d
        self.w = (np.random.uniform(-0.01,0.01,(self.d)+1)).reshape((self.d)+1,1) #randomly generates w vector which has w0 too
        
    def fit(self,X1,y1): #X1 and y1 are df
 
        # cost = []
        # w1 = [self.w]
        z=np.ones((len(X1),1))
        X1 = np.append(X1, z, axis=1) 

        scaler = MinMaxScaler()
        X1 = scaler.fit_transform(X1)
        y1 = np.asarray(y1)
                
        max_iter = int(10e5)
        
        step = 0.01       
        
        for k in range(1,max_iter):
            # self.w = w1[k-1]
            sigma = 1/ (1+np.exp(-np.dot(X1,self.w)))
            h = (sigma).reshape(sigma.shape[0],sigma.shape[1])
            # L = -np.sum(np.multiply(y1, np.log(h)) + np.multiply((1 - y1), np.log(1 - h)))
            L_grad = (np.dot(X1.T,(h-y1)))
            self.w = self.w - step*L_grad
            # cost.append(L)
        # self.w = w1
        # return (self.w)
    def predict(self,X1): #X1 is dataframe
        z=np.ones((len(X1),1))
        X1 = np.append(X1, z, axis=1) 

        scaler = MinMaxScaler()
        X1 = scaler.fit_transform(X1)
        # y1 = np.asarray(y1)
        y = []
        w = self.w
        sigma = 1/ (1+np.exp(-np.dot(X1,w)))
        for m in range(len(sigma)):
            if sigma[m] > 0.5:
                y.append(1)
            else:
                y.append(0)
        return y
        
        
        
      
    