#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:18:35 2019

@author: lalita
"""

import numpy as np
#problem dimension
n=10 #less problem dim more cond number results in later converg 10
#condition number
c=1000   #5000
#generating data matrix Q
A=np.random.randn(n,n)
#[V,D]: V=u; D=s
u, s, vh = np.linalg.svd(A, full_matrices=True)
s_diag=np.diag(s)
alpha = (c*s_diag[n-1,n-1]/s_diag[0,0])**(1/(n-1)) #matlab index starts from 1 and python from 0
a=[]
for i in range(0,n):
    a.append(alpha**(n-i))
s_diag=s_diag*np.diag(a)
Q=u.T*s_diag*u
#generating vector b

b=np.random.randn(n,1)
#%%

x=np.ones((n), dtype=int)
x=x.reshape(n,1)
x_T=x.reshape((n,1)).T #need to convert it to (1,50) array

#%%
# the objective function
def func(x):
    return 0.5*(np.matmul(np.matmul(x.T,Q),x))+np.matmul(b.T,x) #need to take transpose for both dot and matmul

# first order derivatives of the function
def dfunc(x):
    df = Q.dot(x.reshape(n,1))+b
    return (df)

#Armijo Stepsize Selection Algorithm
def armijo(old,beta, sigma,niters):
    #beta = random.random()
    #sigma = random.uniform(0, .5)
#    beta = beta
#    sigma = sigma
    S = 1
    leftf=[]
    rightf=[]
    
#    max_iter = 30
    for i in range(0,niters):
        new = old+(beta**i)*S*(-dfunc(old))
        leftf.append(func(old) - func(new))
        rightf.append(-sigma*(beta**i)*S*(np.matmul((dfunc(old)).T,-dfunc(old))))
    return leftf, rightf

        
#%%
    
# initialization
old = x.reshape(n,1)
#direction = -dfunc(x)
maximum_iterations =20
leftf,rightf=armijo(old,0.25,0.25, maximum_iterations)
index=[]
for i in range(0,len(leftf)):
    if (leftf[i] >= rightf[i]):
        index.append(i)
    
#beta = 0.25
#sigma = 0.25
S = 1
#S=1
x_new = old+(0.5**index[0])*S*(-dfunc(old))
#
#%%Diagonal Scaling Gradient Descent
Hess = Q
epsilon = 0.5     #n=10,c=1000
hess_inv = np.linalg.inv(Q)
df = Q.dot(x.reshape(n,1))+b
s_d =np.arange(0.0001,1,0.0001)     #constant step sizes' range                #0.001 true
x_new_diagscal = []
step=[]
for i in range(0,len(s_d)):
    x_new_diagscal.append(x - s_d[i]*np.matmul(hess_inv,df))
    if (np.linalg.norm(x_new_diagscal[i]-x) < epsilon):
        step.append(s_d[i])
x_new_diagscal = x - step[0]*np.matmul(hess_inv,df)
#%%
#algorithm for conjugate gradient
x0=x
g0=dfunc(x0)
d0=-g0
Nr_alpha = np.matmul(d0.T,(b-np.matmul(Q,x0)))
Dr_alpha = np.matmul(np.matmul(d0.T,Q),d0)
alpha0=Nr_alpha/Dr_alpha

#x1 = x0+alpha0*d0
#g1 = dfunc(x1)
#Nr = np.matmul(np.matmul(g1.T,Q),d0)       #np.matmul(np.matmul(G[r].T,Q),D[j])
#Dr=np.matmul(np.matmul(d0.T,Q),d0)
#term2= np.sum((Nr/Dr)*d0)
#d1 = -g1+term2
#alpha1 = (np.matmul(d1.T,(b-np.matmul(Q,x1))))/np.matmul(np.matmul(d1.T,Q),d1)


#%%
#X=[x0]
#alpha=[alpha0]
#D=[d0]
#G=[g0]
##for r in range(1,2):
#r=1
#if D[r-1].all() !=0:
#    for j in range(0,r):
#        X.append(X[r-1]+alpha[r-1]*D[r-1])   #correct calculation
#        G.append(dfunc(X[r]))  #correct calculation
#        Nr = np.matmul(np.matmul(G[r].T,Q),D[j])       #np.matmul(np.matmul(G[r].T,Q),D[j])
#        Dr=np.matmul(np.matmul(D[j].T,Q),D[j])
#        term2= np.sum((Nr/Dr)*D[j])
#        D.append(-G[r]+term2)  #correct cal
#        Nr_alpha = np.matmul(D[r].T,(b-np.matmul(Q,X[r])))
#        Dr_alpha = np.matmul(np.matmul(D[r].T,Q),D[r])
#        alpha.append(Nr_alpha/Dr_alpha) #correct cal
#    r=r+1
#        
        


#%% New

#X = [x0]
#alpha=[alpha0]
#D=[d0]
#G=[g0]
#
#r=1
#
##for r in range(1,N):
##while np.any(np.abs(np.array(G[r-1])>0.1)):
#while np.any(np.array(G[r-1])>5):
##    if G[r-1].all() !=0:
#    summand = 0
#    X.append((X[r-1]+(alpha[r-1]*D[r-1])))
#    G.append(dfunc(X[r]))
#    for j in range(0,r):
#        summand = summand + ((np.linalg.multi_dot([G[r].T,Q,D[j]]))/(np.linalg.multi_dot([D[j].T,Q,D[j]])))*D[j]
#    D.append(-G[r]+ summand)
##    alpha.append(((np.linalg.multi_dot([D[r].T,(b-np.dot(Q,X[r]))])))/((np.linalg.multi_dot([D[r].T,Q,D[r]])))
#    alpha.append(np.linalg.multi_dot([D[r].T,(b-np.dot(Q,X[r]))])/np.linalg.multi_dot([D[r].T,Q,D[r]]))
#    r = r+1
#    
#    
    
    
    
    
#%% New1: With different stopping criteria
    
X = [x0]
alpha=[alpha0]
D=[d0]
G=[g0]

X.append(X[0]+alpha[0]*D[0])
G.append(dfunc(X[1]))
Nr = np.matmul(np.matmul(G[1].T,Q),D[0])       #np.matmul(np.matmul(G[r].T,Q),D[j])
Dr=np.matmul(np.matmul(D[0].T,Q),D[0])
term2= (Nr/Dr)*D[0]
D.append(-G[1]+term2)
alpha.append((np.matmul(D[1].T,(b-np.matmul(Q,X[1]))))/np.matmul(np.matmul(D[1].T,Q),D[1]))

r=2


#for r in range(1,N):
#while np.any(np.abs(np.array(G[r-1])>0.1)):
while (np.linalg.norm(X[r-1]-X[r-2]) > 0.001):
#    if G[r-1].all() !=0:
    summand = 0
    X.append((X[r-1]+(alpha[r-1]*D[r-1])))
    G.append(dfunc(X[r]))
    for j in range(0,r):
        summand = summand + ((np.linalg.multi_dot([G[r].T,Q,D[j]]))/(np.linalg.multi_dot([D[j].T,Q,D[j]])))*D[j]
    D.append(-G[r]+ summand)
#    alpha.append(((np.linalg.multi_dot([D[r].T,(b-np.dot(Q,X[r]))])))/((np.linalg.multi_dot([D[r].T,Q,D[r]])))
    alpha.append(np.linalg.multi_dot([D[r].T,(b-np.dot(Q,X[r]))])/np.linalg.multi_dot([D[r].T,Q,D[r]]))
    
    r = r+1
    
print(X[r-1])    
    
#%%
    
#G = [g0]
#
#X = [x0]
#D = [d0]
#N = len(X)
#for r in range(0,N): 
#    X.append(X[r]+alpha[r]*D[r])
#    for j in range(0,r):
#        X.append(x0+alpha0*d0)
#        D.append(-dfunc(X[r]))
#        Nr_alpha = np.matmul(D[r].T,(b-np.matmul(Q,X[r])))
#        Dr_alpha = np.matmul(np.matmul(D[r].T,Q),D[r])
#        alpha = Nr_alpha/Dr_alpha
#        X[r]=X[r-1]+alpha*D[r]
#        X.append(X[r])
#        G.append(-dfunc(X[r]))
#        
#        Nr = np.matmul(np.matmul(G[r].T,Q),D[j])       #np.matmul(np.matmul(G[r].T,Q),D[j])
#        Dr=np.matmul(np.matmul(D[j].T,Q),D[j])
#        term2= np.sum((Nr/Dr)*D[j])
#        D.append(-G[r]+term2)
#    
