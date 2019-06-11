# -*- coding: utf-8 -*-

"""
Created on Sat Jan 26 16:16:29 2019

@author: lalitameena
"""
#KNN using Euclidean Distances
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sc
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv', delimiter='|')

test = pd.read_csv('test.csv')
train.shape
test.shape
train.info()
desc_train=train.describe()
train_num=train.drop(['month','day'], axis=1)
test_num=test.drop(['month','day'], axis=1)
desc_test=test.describe()
train_num.shape
test_num.shape
#%%data visualization
sns.distplot(train['FFMC'])
sns.distplot(train['DMC'])
sns.distplot(train['DC'])
sns.distplot(train['ISI'])
sns.distplot(train['temp'])
sns.distplot(train['RH'])
sns.distplot(train['wind'])
sns.distplot(train['rain'])
sns.distplot(train['area'])
sns.distplot(train['month'])
sns.distplot(train['day'])

train.plot.scatter(x='FFMC',y='area',title='FFMC vs area')
train.plot.scatter(x='DMC',y='area',title='DMC vs area')
train.plot.scatter(x='DC',y='area',title='DC vs area')
train.plot.scatter(x='ISI',y='area',title='ISI vs area')
train.plot.scatter(x='temp',y='area',title='temp vs area')
train.plot.scatter(x='RH',y='area',title='RH vs area')
train.plot.scatter(x='wind',y='area',title='wind vs area')
train.plot.scatter(x='rain',y='area',title='rain vs area')
train.plot.scatter(x='month',y='area',title='month vs area')
train.plot.scatter(x='day',y='area',title='day vs area')

corr=train.corr()
# plot the correlation matrix in form of heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

#%%normalize train data features

#normalized training set
train_max=train_num.max()
train_min=train_num.min()
Dr=(train_max-train_min).to_frame(name=None).T
train_min_df=(train_min.to_frame(name=None)).T
num=(train_num-train_min_df).combine_first(train_num).reindex_like(train_num).astype(float)
concat_df=pd.concat([num, Dr],ignore_index=True)
norm_train=num.div(Dr.iloc[0],axis='columns')
#norm_train.loc[norm_train['area']>0,'area']=1
#%%
#shuffling train data

norm_train=norm_train.sample(frac=1).reset_index(drop=True)
#%%
#normalized test set
num1=(test_num-train_min_df).combine_first(test_num).reindex_like(test_num).astype(float)
concat_df1=pd.concat([num1, Dr],ignore_index=True)
norm_test=num1.div(Dr.iloc[0],axis='columns')



#%%
#shuffling test data

norm_test=norm_test.sample(frac=1).reset_index(drop=True)

   
#%%Cross validation: dividing training in 3 equal parts

#this section is fixed for dev1
dev1=norm_train.iloc[0:155,:]   #valid set part 1
dev1.loc[dev1['area']>0,'area']=1
train1_norm=norm_train.iloc[155:,:]
train1_norm=train1_norm.reset_index()
train1_norm=train1_norm.reset_index()
dist1=sc.spatial.distance.cdist(train1_norm.iloc[:,2:], dev1.iloc[:,0:],metric='euclidean')#valdi sample columns, train samples row nos.
dist_df1=pd.DataFrame(dist1)


#%%dev 2 dataset
#================================================================================
dev2=norm_train.iloc[155:310,:]   
dev2.loc[dev2['area']>0,'area']=1
train11_norm=norm_train.iloc[0:155,:]
train22_norm=norm_train.iloc[310:466,:]
trainnorm=pd.concat([train11_norm, train22_norm],ignore_index=True)

dist2=sc.spatial.distance.cdist(trainnorm.iloc[:,0:], dev2.iloc[:,0:],metric='euclidean')#test sample columns, train samples row nos.
dist_df2=pd.DataFrame(dist2)
trainnorm=trainnorm.reset_index()


#%%dev 3 dataset
#=============================================================================
dev3=norm_train.iloc[310:465,:]   
dev3.loc[dev3['area']>0,'area']=1
trainnorm1=norm_train.iloc[0:310,:]
trainnorm1=trainnorm1.reset_index()
dist3=sc.spatial.distance.cdist(trainnorm1.iloc[:,1:], dev3.iloc[:,0:],metric='euclidean')#test sample columns, train samples row nos.
dist_df3=pd.DataFrame(dist3)



#%%list of k values
k=[]
for i in range (1,50):
    k.append((2*i)+1)


#%%final function for first choice of k
def acc_cal1(k, dev, tr_norm, dist_df):
     
    ed1_dev=[]
    for i in range(0,155):
        ed1_dev.append(dist_df.nsmallest(k,dist_df.columns[i]))
    for i in range(0,155):
        ed1_dev[i]=ed1_dev[i].reset_index()
    new1=[]
    new=[]
    pos_area2_dev=[]
    correct=0
    incorrect=0
   
    for i in range (0,155):     
        new=tr_norm.iloc[:,0].isin(ed1_dev[i]['index'])
        new1.append(tr_norm.iloc[np.where(new)]) #predicted class
        pos_area2_dev.append(((new1[i]['area']>0).value_counts()).to_frame(name=None))
        pos_area2_dev[i]=pos_area2_dev[i].reset_index()
 
        if (pos_area2_dev[i].iloc[0,0]==True & (pos_area2_dev[i].iloc[0,1]>0.5*k)):
            pos_area2_dev[i]['class']=1
        else:
            pos_area2_dev[i]['class']=0

        if (dev.iloc[i,10]==pos_area2_dev[i].iloc[0,2]):
            correct=correct+1
        else:
            incorrect=incorrect+1
    accuracy21=(correct/ (correct+incorrect))*100
    return(accuracy21)


dev1_acc=[]    
dev2_acc=[] 
dev3_acc=[] 
#for j in range(0,49):
dev1_acc.append(acc_cal1(k[0],dev1,train1_norm,dist_df1)) 
dev2_acc.append(acc_cal1(k[0],dev2,trainnorm,dist_df2))
dev3_acc.append(acc_cal1(k[0],dev3,trainnorm1,dist_df3))
#%%function for second choice of k
def acc_cal2(k,k1, dev, tr_norm, dist_df):


##KNN for K=79
    ed2_dev=[]
    ed22=[]
    for i in range(0,155):
        ed2_dev.append(dist_df.nsmallest(k+k1,dist_df.columns[i]))
    for i in range(0,155):
        ed2_dev[i]=ed2_dev[i].reset_index()

    for i in range(0,155):
#        ed22.append(ed2_dev[i].iloc[k+1:k+k1+1,:]) #final df for k=40 to 79 ie for k =79(set diff)
         ed22.append(ed2_dev[i].iloc[k+1:k+k1+1,:])

    new1=[]
    new=[]
    pos_area2_dev=[]
    correct=0
    incorrect=0
    for i in range (0,155):     
        new=tr_norm.iloc[:,0].isin(ed22[i]['index'])
        new1.append(tr_norm.iloc[np.where(new)]) #predicted class
  
 
        pos_area2_dev.append(((new1[i]['area']>0).value_counts()).to_frame(name=None))
        pos_area2_dev[i]=pos_area2_dev[i].reset_index()

        if (pos_area2_dev[i].iloc[0,0]==True & (pos_area2_dev[i].iloc[0,1]>0.5*k1)):
            pos_area2_dev[i]['class']=1
        else:
            pos_area2_dev[i]['class']=0

        if (dev.iloc[i,10]==pos_area2_dev[i].iloc[0,2]):
            correct=correct+1
        else:
            incorrect=incorrect+1
    accuracy12=(correct/ (correct+incorrect))*100
    return(accuracy12)


dev11_acc=[]    
dev22_acc=[] 
dev33_acc=[] 
for j in range(0,48):
    dev11_acc.append(acc_cal2(k[j],k[j+1],dev1,train1_norm,dist_df1))
    dev22_acc.append(acc_cal2(k[j],k[j+1],dev2,trainnorm,dist_df2))
    dev33_acc.append(acc_cal2(k[j],k[j+1],dev3,trainnorm1,dist_df3))
#%%finding k with highest performance
k3=(dev1_acc[0]+dev2_acc[0]+dev3_acc[0])/3
k_p=[]
for p in range(0,48):
    k_p.append((dev11_acc[p]+dev22_acc[p]+dev33_acc[p])/3)
max_kp=max(k_p)
max_perf=max(k3,max_kp)
if max_perf==max_kp:

    max_perfk=k[k_p.index(max_perf)+1]
else:
    max_perfk=3
#%%plotting classif accuracy on validation set
k_p.insert(0,k3)
plt.scatter(k,k_p)
plt.title('Scatter plot K vs Avg Classification Accuracy on Development Set')
plt.xlabel('K')
plt.ylabel('Avg Classification Accuracy on Development Set')
plt.show()
#%%getting euclidean dist matrix for test data
norm_train=num.div(Dr.iloc[0],axis='columns')
norm_train=norm_train.reset_index()
dist_test=sc.spatial.distance.cdist(norm_train.iloc[:,1:], norm_test.iloc[:,0:],metric='euclidean')#test sample columns, train samples row nos.
dist_test_df=pd.DataFrame(dist_test)
norm_test.loc[norm_test['area']>0,'area']=1

#%%testing KNN for max_perfk best performance 

k_test=max_perfk

ed1_dev1=[]
for i in range(0,51):
    ed1_dev1.append(dist_test_df.nsmallest(k_test,dist_test_df.columns[i]))
for i in range(0,51):
    ed1_dev1[i]=ed1_dev1[i].reset_index()

new1=[]
new=[]
pos_area2_dev1=[]
correct=0
incorrect=0

for i in range (0,51):     
    new=norm_train.iloc[:,0].isin(ed1_dev1[i]['index'])
    new1.append(norm_train.iloc[np.where(new)]) #predicted class
  
 
    pos_area2_dev1.append(((new1[i]['area']>0).value_counts()).to_frame(name=None))
    pos_area2_dev1[i]=pos_area2_dev1[i].reset_index()

    if (pos_area2_dev1[i].iloc[0,0]==True & (pos_area2_dev1[i].iloc[0,1]>0.5*k_test)):
        pos_area2_dev1[i]['class']=1
    else:
        pos_area2_dev1[i]['class']=0

    if (norm_test.iloc[i,10]==pos_area2_dev1[i].iloc[0,2]):
        correct=correct+1
    else:
        incorrect=incorrect+1
accuracy11=(correct/ (correct+incorrect))*100




