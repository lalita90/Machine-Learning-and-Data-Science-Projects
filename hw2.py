# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 23:44:20 2019

@author: LALITA
"""
#https://www.python-course.eu/Decision_Trees.php
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math
eps = np.finfo(float).eps
from numpy import log2 as log
import pydotplus
import pydot
import graphviz  #Type conda install pydot graphviz in cmd, and then add the executables location directory C:\Anaconda3\pkgs\graphviz-2.38-hfd603c8_2\Library\bin\graphviz to your system path variable. That works!
from IPython.display import Image
from sklearn.tree import export_graphviz
import os

#%%Q1 b(i)
data = pd.read_csv('hw2_question1.csv')
class2=data.loc[data['class'] == 2]
class4=data.loc[data['class'] == 4]

class2=class2.sample(frac=1).reset_index(drop=True)
class4=class4.sample(frac=1).reset_index(drop=True)
class2_1=class2.iloc[:294,:]
class2_2=class2.iloc[294:,:]
class4_1=class4.iloc[:158,:]
class4_2=class4.iloc[158:,:]

train=pd.concat([class2_1,class4_1],ignore_index=True)

test=pd.concat([class2_2,class4_2],ignore_index=True)
X_train=train.iloc[:,0:9]
Y_train=train.iloc[:,9]
X_test=test.iloc[:,0:9]
Y_test=test.iloc[:,9]

#%%  https://medium.com/@rakendd/decision-tree-from-scratch-9e23bcfb4928

def find_entropy(df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy
  
  
def find_entropy_attribute(df,attribute):
  Class = df.keys()[-1]   #To make the code generic, changing target variable class name
  target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps)
      fraction2 = den/len(df)
      entropy2 += -fraction2*entropy
  return abs(entropy2)

   
def find_winner(df):
#    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
#         Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    return df.keys()[:-1][np.argmax(IG)]
  
  
def get_subtable(df, node,value):   #node=attribute name
  return df[df[node] == value].reset_index(drop=True)


def buildTree(df,tree=None): 
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    
    #Here we build our decision tree

    #Get attribute with maximum information gain
    node = find_winner(df)
    
    #Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    attValue = np.unique(df[node])
    
    #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
   #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 

    for value in attValue:
        
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable['class'],return_counts=True)                        
        
        if len(counts)==1:#Checking purity of subset
            tree[node][value] = clValue[0]                                                    
        else:        
            tree[node][value] = buildTree(subtable) #Calling the function recursively 
                   
    return tree
  
def predict(inst,tree):
    #This function is used to predict for any input variable 
    
    #Recursively we go through the tree that we built earlier

    for nodes in tree.keys():        
        
        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = 0
            
        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break;                            
        
    return prediction
#%%
# Q1 b(ii)
#entropy_node for train
#entropy_node = find_entropy(train)
entropy_node = find_entropy(train)
#entropy attribute for train    
   
c=[]
list=train.columns
for i in range(0,9):
    attribute=list[i]
    c.append(find_entropy_attribute(train,attribute))

#IG_Taste = entropy_node — entropy_attribute 
IG=[]
for k in range(0,9):
    info_gain=entropy_node-c[k]
    IG.append(info_gain)
    

tree=buildTree(train)


pred_train=[]
for kk in range(0,452):
    inst=train.iloc[kk,:]
    pred_train.append(predict(inst,tree))

train['pred_class'] = pred_train
#%%


tree

def walk_dictionaryv2(graph, dictionary, parent_node=None):
    '''
    Recursive plotting function for the decision tree stored as a dictionary
    '''

    for k in dictionary.keys():

        if parent_node is not None:

            from_name = parent_node.get_name().replace("\"", "") + '_' + str(k)
            from_label = str(k)

            node_from = pydot.Node(from_name, label=from_label)

            graph.add_edge( pydot.Edge(parent_node, node_from) )

            if isinstance(dictionary[k], dict): # if interim node


                walk_dictionaryv2(graph, dictionary[k], node_from)

            else: # if leaf node
                to_name = str(k) + '_' + str(dictionary[k]) # unique name
                to_label = str(dictionary[k])

                node_to = pydot.Node(to_name, label=to_label, shape='box')
                graph.add_edge(pydot.Edge(node_from, node_to))

                #node_from.set_name(to_name)

        else:

            from_name =  str(k)
            from_label = str(k)

            node_from = pydot.Node(from_name, label=from_label)
            walk_dictionaryv2(graph, dictionary[k], node_from)


def plot_tree(tree, name):

    # first you create a new graph, you do that with pydot.Dot()
    graph = pydot.Dot(graph_type='graph')

    walk_dictionaryv2(graph, tree)

    graph.write_png(name+'.png')



plot_tree(tree,'name')
#%%test data

#entropy_node for test
entropy_node1=find_entropy(test)

#entropy and IG cal for test data
c1=[]
list1=test.columns
for i1 in range(0,9):
    attribute1=list1[i1]
    c1.append(find_entropy_attribute(test,attribute1))


IG1=[]
for k1 in range(0,9):
    info_gain=entropy_node1-c1[k1]
    IG1.append(info_gain)
    
tree1=buildTree(test)
plot_tree(tree1,'name1')

pred_test=[]
for jj in range(0,231):
    inst=test.iloc[jj,:]
    pred_test.append(predict(inst,tree1))

test['pred_class'] = pred_test
cc=(test['class']==test['pred_class'])
test11=test.iloc[np.where(cc)]
accuracy_test=(len(test11.index)/len(test.index))*100