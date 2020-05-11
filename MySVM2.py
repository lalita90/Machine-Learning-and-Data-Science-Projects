#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:22:33 2020

@author: lalita
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:56:37 2020

@author: lalita
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import linalg as LA
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
#%%

class MySVMv1:
    def __init__(self,d,m):
        
        self.d = d
        self.w = (np.random.uniform(-0.01,0.01,(self.d)+1)).reshape((self.d)+1,1) 
        self.m = m
    def fit(self,X1,y1): 
        cost = []
        z=np.ones((len(X1),1))
        X1 = np.append(X1, z, axis=1) 

        # scaler = MinMaxScaler()
        # X1 = scaler.fit_transform(X1)
        y1 = np.asarray(y1)
            
        data = pd.DataFrame(np.append(X1,y1,axis=1))
        data = data.sample(self.m)
        # print(len(data))
        X1 = data.iloc[:,:-1].values
        y1 = (data.iloc[:,-1].values).reshape(-1,1)
        
        
        N = X1.shape[0]
        distances = 1 - y1 * (np.dot(X1, self.w))
        
        distances[distances <= 0] = 0  # equivalent to max(0, distance)
       
        Y_batch = y1
        X_batch = X1
       
        dw = np.zeros(len(self.w)).reshape(-1,1)     
        
        max_epochs = int(10e3)
        weights = self.w
              
        for epoch in range(0, max_epochs): 
            
            learning_rate = 0.00001 #0.0001 for b75, iter=1000, 10e3 for b50 m=40 with 0.00001
                       
            distances = 1 - y1 * (np.dot(X1, weights))
                       
            for ind, d in enumerate(distances):
                
                if max(0, d) == 0:
                    di = 5*weights
                else:
                    di = 5*weights - ( Y_batch[ind] * X_batch[ind].reshape(-1,1))
                dw += di
        
            dw = dw/len(Y_batch)  # average
            ascent = dw
            
            weights= weights - (learning_rate * ascent)
            self.w = weights
        
            # epochs += 1  
       
        self.w = weights 
        
        
       
           
        
    def predict(self,X1): 
        z=np.ones((len(X1),1))
        X1 = np.append(X1, z, axis=1) 

        w = self.w
        
        yp = np.sign(np.dot(X1,w))
       
        return yp
        
        
        
      
    