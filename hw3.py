# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:39:42 2019

@author: LALITA
"""
#https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
#http://dilloncamp.com/projects/pca.html
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.decomposition import PCA
from PIL import Image
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.image as image
import scipy
import ntpath
import re

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
from numpy.core import multiarray
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

#%%normalization
scaler = StandardScaler()
img_norm=[]
for i in range(0,len(img)):
    scaler.fit(img[i])
    img_norm.append(scaler.transform(img[i]))
    
#%%row stacked matrix
for i in range(0,len(img_norm)):
    img_norm[i]=img_norm[i].flatten().T
m =img_norm[0]
for i in range(1,len(img_norm)):
    m=np.vstack((m, img_norm[i]))
pca = PCA(0.99)
img_pca=[]
pca.fit(m)
img_pca.append(pca.transform(m))
    
energy=(pca.explained_variance_ratio_)
print(pca.singular_values_)  
pc=pca.components_[0:10]  
all_pc=pca.components_   

#%%plot energy of eigenvectors
list=[1,2,3,4,5,6,7,8,9,10]
# Plot
plt.scatter(list, energy[0:10], alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('eigenvalues')
plt.ylabel('energy')
plt.show()
#%%for 50% energy
# Make an instance of the Model
m1 =img_norm[0]
for i in range(1,len(img_norm)):
    m1=np.vstack((m1, img_norm[i]))
pca1 = PCA(0.5)
img_pca1=[]
pca.fit(m1)
img_pca1.append(pca.transform(m1))
    
energy1=(pca.explained_variance_ratio_)
print(pca.singular_values_)  
pc_50=pca.components_   #where every row is a principal component in the p-dimensional space. Each of these rows is an Eigenvector of the centered covariance matrix XXT. 
#%%eigenfaces plot
#plt.imshow(m[164].reshape(120,160))  #gives img1 saved in the folder
#plt.imshow(approximation[164].reshape(120,160))  #gives img1 saved in the folder
#plotting eigenfaces

#plt.imshow(pc[3].reshape(120,160),cmap='gray')   #eigenface plot
plt.imshow(pc[0].reshape(120,160),cmap='gray')
plt.imshow(pc[1].reshape(120,160),cmap='gray')
plt.imshow(pc[2].reshape(120,160),cmap='gray')
plt.imshow(pc[3].reshape(120,160),cmap='gray')
plt.imshow(pc[4].reshape(120,160),cmap='gray')
plt.imshow(pc[5].reshape(120,160),cmap='gray')

fig4, axarr = plt.subplots(3,2,figsize=(8,8))
axarr[0,0].imshow(img_norm[0].reshape(120,160),cmap='gray')
axarr[0,0].set_title('Original Image')
axarr[0,0].axis('off')
axarr[0,1].imshow(pc[0].reshape(120,160),cmap='gray')
axarr[0,1].set_title('99% Variation')
axarr[0,1].axis('off')
axarr[1,0].imshow(img_norm[1].reshape(120,160),cmap='gray')
axarr[1,0].set_title('Original Image')
axarr[1,0].axis('off')
axarr[1,1].imshow(pc[1].reshape(120,160),cmap='gray')
axarr[1,1].set_title('99% Variation')
axarr[1,1].axis('off')
axarr[2,0].imshow(img_norm[2].reshape(120,160),cmap='gray')
axarr[2,0].set_title('Original Image')
axarr[2,0].axis('off')
axarr[2,1].imshow(pc[2].reshape(120,160),cmap='gray')
axarr[2,1].set_title('99% variation')
axarr[2,1].axis('off')
plt.show()

plt.imshow(img_norm[3].reshape(120,160),cmap='gray')
plt.imshow(pc[3].reshape(120,160),cmap='gray')

plt.show()

fig5, axarr = plt.subplots(3,2,figsize=(8,8))
axarr[0,0].imshow(img_norm[4].reshape(120,160),cmap='gray')
axarr[0,0].set_title('Original Image')
axarr[0,0].axis('off')
axarr[0,1].imshow(pc[4].reshape(120,160),cmap='gray')
axarr[0,1].set_title('99% Variation')
axarr[0,1].axis('off')
axarr[1,0].imshow(img_norm[5].reshape(120,160),cmap='gray')
axarr[1,0].set_title('Original Image')
axarr[1,0].axis('off')
axarr[1,1].imshow(pc[5].reshape(120,160),cmap='gray')
axarr[1,1].set_title('99% Variation')
axarr[1,1].axis('off')
axarr[2,0].imshow(img_norm[6].reshape(120,160),cmap='gray')
axarr[2,0].set_title('Original Image')
axarr[2,0].axis('off')
axarr[2,1].imshow(pc[6].reshape(120,160),cmap='gray')
axarr[2,1].set_title('99% variation')
axarr[2,1].axis('off')
plt.show()

fig4, axarr = plt.subplots(3,2,figsize=(8,8))
axarr[0,0].imshow(img_norm[7].reshape(120,160),cmap='gray')
axarr[0,0].set_title('Original Image')
axarr[0,0].axis('off')
axarr[0,1].imshow(pc[7].reshape(120,160),cmap='gray')
axarr[0,1].set_title('99% Variation')
axarr[0,1].axis('off')
axarr[1,0].imshow(img_norm[8].reshape(120,160),cmap='gray')
axarr[1,0].set_title('Original Image')
axarr[1,0].axis('off')
axarr[1,1].imshow(pc[8].reshape(120,160),cmap='gray')
axarr[1,1].set_title('99% Variation')
axarr[1,1].axis('off')
axarr[2,0].imshow(img_norm[9].reshape(120,160),cmap='gray')
axarr[2,0].set_title('Original Image')
axarr[2,0].axis('off')
axarr[2,1].imshow(pc[9].reshape(120,160),cmap='gray')
axarr[2,1].set_title('99% variation')
axarr[2,1].axis('off')
plt.show()

#%%

approximation = pca.inverse_transform(img_pca[0])
for i in range(0,m.shape[0]):
    m[i] = m[i].T
    approximation[i] = approximation[i].T

fig4, axarr = plt.subplots(3,2,figsize=(8,8))
axarr[0,0].imshow(img_norm[0].reshape(120,160),cmap='gray')
axarr[0,0].set_title('Original Image')
axarr[0,0].axis('off')
axarr[0,1].imshow(approximation[0].reshape(120,160),cmap='gray')
axarr[0,1].set_title('99% Variation')
axarr[0,1].axis('off')
axarr[1,0].imshow(img_norm[1].reshape(120,160),cmap='gray')
axarr[1,0].set_title('Original Image')
axarr[1,0].axis('off')
axarr[1,1].imshow(approximation[1].reshape(120,160),cmap='gray')
axarr[1,1].set_title('99% Variation')
axarr[1,1].axis('off')
axarr[2,0].imshow(img_norm[2].reshape(120,160),cmap='gray')
axarr[2,0].set_title('Original Image')
axarr[2,0].axis('off')
axarr[2,1].imshow(approximation[2].reshape(120,160),cmap='gray')
axarr[2,1].set_title('99% variation')
axarr[2,1].axis('off')
plt.show()

fig4, axarr = plt.subplots(3,2,figsize=(8,8))
axarr[0,0].imshow(img_norm[3].reshape(120,160),cmap='gray')
axarr[0,0].set_title('Original Image')
axarr[0,0].axis('off')
axarr[0,1].imshow(approximation[3].reshape(120,160),cmap='gray')
axarr[0,1].set_title('99% Variation')
axarr[0,1].axis('off')
axarr[1,0].imshow(img_norm[4].reshape(120,160),cmap='gray')
axarr[1,0].set_title('Original Image')
axarr[1,0].axis('off')
axarr[1,1].imshow(approximation[4].reshape(120,160),cmap='gray')
axarr[1,1].set_title('99% Variation')
axarr[1,1].axis('off')
axarr[2,0].imshow(img_norm[5].reshape(120,160),cmap='gray')
axarr[2,0].set_title('Original Image')
axarr[2,0].axis('off')
axarr[2,1].imshow(approximation[5].reshape(120,160),cmap='gray')
axarr[2,1].set_title('99% variation')
axarr[2,1].axis('off')
plt.show()

#%%image classification
import pandas as pd
#coeff=np.dot(approximation,pc.T)
coeff=np.dot(img_norm,all_pc.T)

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
for i in range(0,len(myList)):
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
    yTest_Pred.append(int(model.predict([xTest.iloc[i,:]]))) # 0:Overcast, 2:Mild
count1=0
for i in range(0,33):
    if (yTest.iloc[i,0]==yTest_Pred[i]):
        count1=count1+1
accuracy_knn=(count1/33)*100    #upto 85%
#10 components=87.87
#20=84.84
#30=81.8181
#40=72.72
#50=69.69
#%%

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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(np.array(x_train).reshape(132,1,120,160),np.array(y_train),epochs=20,verbose=1, validation_data=(np.array(x_test).reshape(33,1,120,160), np.array(y_test)))
score = model.evaluate(np.array(x_test).reshape(33,1,120,160), np.array(y_test), verbose=0)
#%%Q2 d
n_components=[1,10,20,30,40,50]
for i in n_components:
    pca = PCA(i)
    img_pca1=[]
    pca.fit(m1)
    img_pca1.append(pca.transform(m1))
    
    energy1=(pca.explained_variance_ratio_)
    approximation = pca.inverse_transform(img_pca1[0])
    for i in range(0,m.shape[0]):
        m[i] = m[i].T
        approximation[i] = approximation[i].T
    fig4, axarr = plt.subplots(3,2,figsize=(8,8))
    axarr[0,0].imshow(img_norm[0].reshape(120,160),cmap='gray')
    axarr[0,0].axis('off')
    axarr[0,1].imshow(approximation[0].reshape(120,160),cmap='gray')
    axarr[0,1].axis('off')
    axarr[1,0].imshow(img_norm[1].reshape(120,160),cmap='gray')
    axarr[1,0].axis('off')
    axarr[1,1].imshow(approximation[1].reshape(120,160),cmap='gray')
    axarr[1,1].axis('off')
    axarr[2,0].imshow(img_norm[2].reshape(120,160),cmap='gray')
    axarr[2,0].axis('off')
    axarr[2,1].imshow(approximation[2].reshape(120,160),cmap='gray')
    axarr[2,1].axis('off')
    plt.show()