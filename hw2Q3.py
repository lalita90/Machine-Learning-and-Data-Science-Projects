#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:45:46 2020

@author: lalita
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.datasets import load_boston
from math import sqrt
from my_cross_val import my_cross_val  #browse to particular folder where the file presents
from sklearn.linear_model import LogisticRegression as LR
from numpy import linalg as LA
from numpy.linalg import multi_dot
from MultiGaussClassify import MultiGaussClassify
#%%
#loading original boston dataset data and labels
boston = load_boston()

X_b = pd.DataFrame(boston.data)
y_b = pd.DataFrame(boston.target)
boston_data = np.hstack((X_b,y_b)) #original data without class lables coded
#np.random.shuffle(boston_data)


#loading original digits dataset data and labels
digits = load_digits()

X_d = pd.DataFrame(digits.data)
y_d = pd.DataFrame(digits.target)
digits_data = np.hstack((X_d,y_d)) ##original data with class lables that is required
#np.random.shuffle(digits_data)
#%%getting boston50 with binary classes; data is boston50 dataset with labels
df = pd.DataFrame(boston_data)
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD',
              'TAX','PTRATIO','B','LSTAT','MEDV']

df = df.convert_objects(convert_numeric=True)

median = np.median(df['MEDV'])
data = df.copy() #creating copy of df

#checking indexes in label where value >= median
C1_index = np.asarray(np.where(data['MEDV'] >= median)).ravel()

#2 class classification problem for Boston50
#creating two classes 0 and 1 based on the condition of median for Boston50
for i in range(len(data)):
    if i in C1_index:
        data.iloc[i,13] = 1
    else:
        data.iloc[i,13] = 0
        
        
#%%creating classes for boston75; data1 is boston 75 with labels
data1 = df.copy()
perc_75 = np.percentile(df['MEDV'],75)

C2_index = np.asarray(np.where(data1['MEDV'] >= perc_75)).ravel()
for j in range(len(data1)):
    if j in C2_index:
        data1.iloc[j,13] = 1
    else:
        data1.iloc[j,13] = 0


#boston50 data is data with X and y datapoints and labels

X = data.iloc[:,:data.shape[1]-1]
y = pd.DataFrame(data.iloc[:,-1]) 


#boston75 data is data1 with XX and yy datapoints and labels
data1 = data1.sample(frac=1).reset_index(drop=True) 
XX = data1.iloc[:,:data1.shape[1]-1]
yy = pd.DataFrame(data1.iloc[:,-1]) 

#%%initializing classification methods that we are going to use :change them

MG_boston = MultiGaussClassify(2,13)

MG_digits = MultiGaussClassify(10,64)



#methods = [method1, method2, method3]
#%%
#printing the summary of results for CV using 3 methods for 5 folds of each dataset
        
method_err, mean, std = my_cross_val(MG_boston,X,y,5,False)
print('Error for {} with {}: {} and mean :{}, std: {}'.format('Boston50_Full', 'Boston50',method_err,mean,std))

method_err, mean, std = my_cross_val(MG_boston,X,y,5,True)
print('Error for {} with {}: {} and mean :{}, std: {}'.format('Boston50_diag', 'Boston50',method_err,mean,std))

        
method_err, mean, std = my_cross_val(MG_boston,XX,yy,5,False)
print('Error for {} with {}: {} and mean :{}, std: {}'.format('Boston75_Full', 'Boston75',method_err,mean,std))

method_err, mean, std = my_cross_val(MG_boston,XX,yy,5,True)
print('Error for {} with {}: {} and mean :{}, std: {}'.format('Boston75_diag', 'Boston75',method_err,mean,std))


        
method_err, mean, std = my_cross_val(MG_digits,X_d,y_d,5,False)
print('Error for {} with {}: {} and mean :{}, std: {}'.format('Digits_Full', 'Digits',method_err,mean,std))

method_err, mean, std = my_cross_val(MG_digits,X_d,y_d,5,True)
print('Error for {} with {}: {} and mean :{}, std: {}'.format('Digits_diag', 'Digits',method_err,mean,std))
#%%
from my_cross_val1 import my_cross_val1

method7 = LR(penalty='l2',solver='lbfgs', multi_class='multinomial',
             max_iter=5000)
method_err, mean, std = my_cross_val1(method7,X,y,5)
print('Error for {} with {}: {} and mean :{}, std: {}'.format('Logistic Regression_B50', 'Boston50',method_err,mean,std))

method_err, mean, std = my_cross_val1(method7,XX,yy,5)
print('Error for {} with {}: {} and mean :{}, std: {}'.format('Logistic Regression_B75', 'Boston75',method_err,mean,std))

method_err, mean, std = my_cross_val1(method7,X_d,y_d,5)
print('Error for {} with {}: {} and mean :{}, std: {}'.format('Logistic Regression_digits', 'Digits',method_err,mean,std))




