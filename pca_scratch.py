# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:18:21 2019

@author: LALITA
"""

from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy
import ntpath
import re

from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
from numpy.core import multiarray
from numpy import linalg as LA

#%%
image_list = []
file=[]
for filename in glob.glob('G:\My Drive\Main\Lalita\Applications\Academics\CSCE633 ML\HW3\yalefaces-20190331T011626Z-001\yalefaces/*.gif'): #assuming gif
    file.append(ntpath.basename(filename))
    im=Image.open(filename)
    image_list.append(im)
for i in range(0,165):
    file[i]=re.sub("[^0-9]", "", file[i])
    file[i]=int(file[i][:2])
#%%
#np.array(image_list[0]) gives array of that image
img=[]
for i in range(0,len(image_list)):
    img.append(np.array(image_list[i]))

    img[i]=scipy.misc.imresize(img[i], (120,160), interp='bilinear', mode=None)

#%%normalization of row stacked images
for i in range(0,len(img)):
    img[i]=img[i].flatten().T
import numpy as np
# compute the mean face
mu = np.mean(img, 0)
# Subtract the mean face from each image before performing SVD and PCA
img_norm = img - mu

#%%

Cov = np.dot (img_norm ,img_norm .T)
w, v = LA.eig(Cov)

eigenvectors,eigenvalues,variance = np.linalg.svd(img_norm.T, full_matrices = False )

idx = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx] #eigenfaces
plt.plot(eigenvalues)
plt.ylabel('Eigenvalues')
plt.show()
    
#%%
list=range(0,165)
sum1=[]
var_exp=[]
for j in range(1,165):
    sum1.append(np.sum(eigenvalues[0:j]))
    
for i in range(0,164):    
    var_exp.append([(eigenvalues[i] / sum(eigenvalues))*100]) #energy of one eigenvalue
var=np.cumsum(var_exp)    
#var_50= #22 eigenvalues
plt.scatter(list[0:164],var,alpha=0.5)
plt.plot(var)
plt.ylabel('energy')
plt.show()

#%%
eigenvectors=eigenvectors.T
#eigenvectors=eigenvectors+

# project data
fig, ax = plt.subplots(1,1)
for i in range(0,10):
  ord = eigenvectors[:][i]
  proj_img = ord.reshape(120,160)
  plt.imshow(proj_img,cmap='gray')
  plt.show()
#%%Q2d
n_components=[1,10,20,30,40,50]

for i in n_components:
    for j in range(0,i):
        ord=eigenvectors[j]
        proj_img = ord.reshape(120,160)
        plt.imshow(proj_img,cmap='gray')
        plt.show()
#%%image classification
coeff=np.dot(img_norm,eigenvectors.T)
import pandas as pd

lab_img=[]
for i in range(0,165):
    lab_img.append(np.append(coeff[i],file[i]))

#%%
from sklearn.model_selection import cross_val_score
df=pd.DataFrame.from_dict(lab_img)
df=df.sample(frac=1)
from sklearn.model_selection import train_test_split
col=df.shape[1]-1
x=df.iloc[:,0:col]
y=df.iloc[:,col]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state =40,stratify=y)
xTrain=(xTrain.reset_index())
xTrain.drop(['index'], axis=1, inplace=True)
xTest=xTest.reset_index()
xTest.drop(['index'], axis=1, inplace=True)
yTrain=yTrain.reset_index()
yTrain.drop(['index'], axis=1, inplace=True)
yTest=yTest.reset_index()
yTest.drop(['index'], axis=1, inplace=True)
#%%
# creating odd list of K for KNN
from sklearn.neighbors import KNeighborsClassifier
myList=[]
for i in range(1,15):
    myList.append(i)

# subsetting just the odd ones
neighbors = []
for i in range(3,len(myList)):
    if(myList[i] % 2 !=0):
        neighbors.append(myList[i])

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, xTrain, yTrain, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=optimal_k)

# Train the model using the training sets
model.fit(xTrain,yTrain)
yTest_Pred=[]
#Predict Output
for i in range(0,33):
    yTest_Pred.append(int(model.predict([xTest.iloc[i,:]]))) 
count1=0
for i in range(0,33):
    if (yTest.iloc[i,0]==yTest_Pred[i]):
        count1=count1+1
accuracy_knn=(count1/33)*100  #max 87%

#%%CNN on data
train_data=[]
x_train=[]
y_train=[]
x_test=[]
y_test=[]
for i in range(0,165):
    train_data.append(np.append(img_norm[i],file[i]))
for i in range(0,132):
    x_train.append(train_data[i][0:19200])
    y_train.append(train_data[i][19200])
for i in range(132,165):
    x_test.append(train_data[i][0:19200])
    y_test.append(train_data[i][19200])
from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
model = Sequential()
K.set_image_dim_ordering('th')
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(1,120, 160)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(16, activation='sigmoid'))
#model.add(Dropout(0.3))
model.add(Dense(16, activation = 'softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(np.array(x_train).reshape(132,1,120,160),np.array(y_train),epochs=20,verbose=1)
score = model.evaluate(np.array(x_test).reshape(33,1,120,160), np.array(y_test), verbose=0) #dropouts=0.2#epochs:20:ADAM-96.97%, SGD-81.82,adagrad-0.9394;#dropouts=0.5#epochs:20:ADAM-0.8182%, SGD-0.4848,adagrad-0.8788
#epochs:10:ADAM-93.94%, SGD-41,adagrad-0.9091
#epochs:10:ADAM-93.94%, SGD-41,adagrad-0.9091
#epochs:30:ADAM-100%, SGD-81.82,adagrad-0.8788
#%%1 conv layers
import keras
input_shape=(1,120,160)
cnn1 = Sequential()
cnn1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Dropout(0.5))

cnn1.add(Flatten())

cnn1.add(Dense(128, activation='relu'))
cnn1.add(Dense(10, activation='softmax'))

cnn1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.sgd(),
              metrics=['accuracy'])
model.fit(np.array(x_train).reshape(132,1,120,160),np.array(y_train),epochs=20,verbose=1)#, validation_data=(np.array(x_test).reshape(33,1,120,160), np.array(y_test)))
score = model.evaluate(np.array(x_test).reshape(33,1,120,160), np.array(y_test), verbose=0)
#%%3 conv layers
cnn3 = Sequential()
cnn3.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
cnn3.add(MaxPooling2D((2, 2)))
cnn3.add(Dropout(0.25))

cnn3.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn3.add(MaxPooling2D(pool_size=(2, 2)))
cnn3.add(Dropout(0.25))

cnn3.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn3.add(Dropout(0.4))

cnn3.add(Flatten())

cnn3.add(Dense(128, activation='relu'))
cnn3.add(Dropout(0.3))
cnn3.add(Dense(10, activation='softmax'))

cnn3.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.fit(np.array(x_train).reshape(132,1,120,160),np.array(y_train),epochs=20,verbose=1)
score = model.evaluate(np.array(x_test).reshape(33,1,120,160), np.array(y_test), verbose=0)
#%%increased poolsize decrease number of layers
model = Sequential()
K.set_image_dim_ordering('th')
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(1,120, 160)))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(16, activation='sigmoid'))
#model.add(Dropout(0.3))
model.add(Dense(16, activation = 'softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(np.array(x_train).reshape(132,1,120,160),np.array(y_train),epochs=20,verbose=1, validation_data=(np.array(x_test).reshape(33,1,120,160), np.array(y_test)))
score = model.evaluate(np.array(x_test).reshape(33,1,120,160), np.array(y_test), verbose=0) 
#drop-0.2,epoch=20:adam-1,sgd-0.5455, adagrad-1
#%%trying diff no. of PCA components
for j in range(0,121):
       eigenvector=eigenvectors[0:j]
coeff=np.dot(img_norm,eigenvector.T)
#image classification
import pandas as pd

lab_img=[]
for i in range(0,len(coeff)):
    lab_img.append(np.append(coeff[i],file[i]))

#
from sklearn.model_selection import cross_val_score
df=pd.DataFrame.from_dict(lab_img)
df=df.sample(frac=1)
from sklearn.model_selection import train_test_split
col=df.shape[1]-1
x=df.iloc[:,0:col]
y=df.iloc[:,col]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state =40,stratify=y)
xTrain=(xTrain.reset_index())
xTrain.drop(['index'], axis=1, inplace=True)
xTest=xTest.reset_index()
xTest.drop(['index'], axis=1, inplace=True)
yTrain=yTrain.reset_index()
yTrain.drop(['index'], axis=1, inplace=True)
yTest=yTest.reset_index()
yTest.drop(['index'], axis=1, inplace=True)

# creating odd list of K for KNN
from sklearn.neighbors import KNeighborsClassifier
myList=[]
for i in range(1,15):
    myList.append(i)

# subsetting just the odd ones
neighbors = []
for i in range(3,len(myList)):
    if(myList[i] % 2 !=0):
        neighbors.append(myList[i])

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, xTrain, yTrain, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=optimal_k)

# Train the model using the training sets
model.fit(xTrain,yTrain)
yTest_Pred=[]
n=xTest.shape[0]
#Predict Output
for i in range(0,n):
    yTest_Pred.append(int(model.predict([xTest.iloc[i,:]]))) 
count1=0
for i in range(0,n):
    if (yTest.iloc[i,0]==yTest_Pred[i]):
        count1=count1+1
accuracy_knn=(count1/n)*100  #30:75%;60:78%; 120-81%; all-85%