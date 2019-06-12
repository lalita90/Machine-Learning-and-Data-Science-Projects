# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:15:04 2019

@author: LALITA
"""
import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.model_selection import cross_val_score
import time


#%%Q2 (a)

data1 = pd.read_csv('hw2_question3.csv')

data1.insert(2, "f2_-1", 0) 
data1.insert(3, "f2_0", 0) 
data1.insert(4, "f2_+1", 0) 
data1.loc[data1.f2 ==-1, 'f2_-1'] = 1   #df.loc[df.column_name condition, 'new column name'] = 'value if condition is met'
data1.loc[data1.f2 ==0, 'f2_0'] = 1   
data1.loc[data1.f2 ==1, 'f2_+1'] = 1   

data1.insert(10, "f7_-1", 0) 
data1.insert(11, "f7_0", 0) 
data1.insert(12, "f7_+1", 0) 
data1.loc[data1.f7 ==-1, 'f7_-1'] = 1   #df.loc[df.column_name condition, 'new column name'] = 'value if condition is met'
data1.loc[data1.f7 ==0, 'f7_0'] = 1   
data1.loc[data1.f7 ==1, 'f7_+1'] = 1   

data1.insert(14, "f8_-1", 0) 
data1.insert(15, "f8_0", 0) 
data1.insert(16, "f8_+1", 0) 
data1.loc[data1.f8 ==-1, 'f8_-1'] = 1   #df.loc[df.column_name condition, 'new column name'] = 'value if condition is met'
data1.loc[data1.f8 ==0, 'f8_0'] = 1   
data1.loc[data1.f8 ==1, 'f8_+1'] = 1 

data1.insert(23, "f14_-1", 0) 
data1.insert(24, "f14_0", 0) 
data1.insert(25, "f14_+1", 0) 
data1.loc[data1.f14 ==-1, 'f14_-1'] = 1   #df.loc[df.column_name condition, 'new column name'] = 'value if condition is met'
data1.loc[data1.f14 ==0, 'f14_0'] = 1   
data1.loc[data1.f14 ==1, 'f14_+1'] = 1 

data1.insert(27, "f15_-1", 0) 
data1.insert(28, "f15_0", 0) 
data1.insert(29, "f15_+1", 0) 
data1.loc[data1.f15 ==-1, 'f15_-1'] = 1   #df.loc[df.column_name condition, 'new column name'] = 'value if condition is met'
data1.loc[data1.f15 ==0, 'f15_0'] = 1   
data1.loc[data1.f15 ==1, 'f15_+1'] = 1 

data1.insert(31, "f16_-1", 0) 
data1.insert(32, "f16_0", 0) 
data1.insert(33, "f16_+1", 0) 
data1.loc[data1.f16 ==-1, 'f16_-1'] = 1   #df.loc[df.column_name condition, 'new column name'] = 'value if condition is met'
data1.loc[data1.f16 ==0, 'f16_0'] = 1   
data1.loc[data1.f16 ==1, 'f16_+1'] = 1 


data1.insert(44, "f26_-1", 0) 
data1.insert(45, "f26_0", 0) 
data1.insert(46, "f26_+1", 0) 
data1.loc[data1.f26 ==-1, 'f26_-1'] = 1   #df.loc[df.column_name condition, 'new column name'] = 'value if condition is met'
data1.loc[data1.f26 ==0, 'f26_0'] = 1   
data1.loc[data1.f26 ==1, 'f26_+1'] = 1 


data1.insert(50, "f29_-1", 0) 
data1.insert(51, "f29_0", 0) 
data1.insert(52, "f29_+1", 0) 
data1.loc[data1.f29 ==-1, 'f29_-1'] = 1   #df.loc[df.column_name condition, 'new column name'] = 'value if condition is met'
data1.loc[data1.f29 ==0, 'f29_0'] = 1   
data1.loc[data1.f29 ==1, 'f29_+1'] = 1 


data1.drop(['f2', 'f7', 
                'f8', 'f14', 'f15', 'f16', 'f26', 'f29'], axis=1, inplace=True)
#balanced classes for data 1
class0=data1.loc[data1['f31'] == -1]
class1=data1.loc[data1['f31'] == 1]

#x=data.iloc[:,0:9]
#y=data.iloc[:,9]
class0=class0.sample(frac=1).reset_index(drop=True)
class1=class1.sample(frac=1).reset_index(drop=True)
class0_1=class0.iloc[:3233,:]
class0_2=class0.iloc[3233:,:]
class1_1=class1.iloc[:4064,:]
class1_2=class1.iloc[4064:,:]

train=pd.concat([class0_1,class1_1],ignore_index=True)
X_train=train.iloc[:,0:46]
y_train=train.iloc[:,46]

test=pd.concat([class0_2,class1_2],ignore_index=True)
X_test=test.iloc[:,0:46]
y_test=test.iloc[:,46]
#%% Q2(b) Linear SVM

#import time
start = time.time()
scores=[]
mean_score1=[]
for i in range(1, 50):
    clf = svm.SVC(kernel='linear', C=i)
    scores.append(cross_val_score(clf, X_train, y_train, cv=3))
for i in range (0,49):
    mean_score1.append(np.mean(scores[i]))
best_C=np.argmax(mean_score1)
clf = svm.SVC(kernel='linear', C=best_C).fit(X_train, y_train)
#test_score=clf.score(X_test, y_test) 

end = time.time()
train_time1=(end - start) #334.598 sec

start_test1 = time.time()
test_score1=clf.score(X_test, y_test)
end_test1 = time.time()
test1_time=end_test1-start_test1

#%% Q2(c) kernel SVM
#poly kernel

start = time.time()
scores=[]
mean_score2=[]
for i in range(1, 50):
    clf = svm.SVC(kernel='poly',degree=8, C=i)
    scores.append(cross_val_score(clf, X_train, y_train, cv=3))
for i in range (0,49):
    mean_score2.append(np.mean(scores[i]))
best_C=np.argmax(mean_score2)



clf = svm.SVC(kernel='poly',degree=8, C=best_C).fit(X_train, y_train)  #168.32888 sec
end = time.time()
train_time2=(end - start)

start_test2 = time.time()
test_score2=clf.score(X_test, y_test) 
end_test2 = time.time()
test2_time=end_test2-start_test2


#rbf kernel
start = time.time()
scores=[]
mean_score3=[]
for i in range(1, 50):
    clf = svm.SVC(kernel='rbf',C=i)
    scores.append(cross_val_score(clf, X_train, y_train, cv=3))
for i in range (0,49):
    mean_score3.append(np.mean(scores[i]))
best_C=np.argmax(mean_score3) #giving index of best C



clf = svm.SVC(kernel='rbf',C=best_C).fit(X_train, y_train)  #81.9058530330658 sec
end = time.time()
train_time3=(end - start)
start_test3 = time.time()
test_score3=clf.score(X_test, y_test) 
end_test3 = time.time()
test3_time=end_test3-start_test3