
"""
#Created on Tue Mar  3 15:10:17 2020
#@author: lalita
"""
import numpy as np
import pandas as pd

from numpy import linalg as LA

#%%

#create instance MG=classname(2,13); MG.fit(x,y,false)
class MultiGaussClassify:
    def __init__(self,k,d):
        self.k = k #no of classes
        self.d = d
        self.P_c = []  #prior
        self.mean = []
        self.cov = []
        self.det_cov = []
        self.i_cov = []
        for i in range((self.k)):
            (self.P_c).append(1/self.k)
            (self.mean).append(np.zeros((1,self.d)))
            (self.cov).append(np.eye(self.d))
            
    def fit(self,X1,y1,diag):
        #===============================
        #separate by class
        dataset = np.hstack((X1,y1))
        sep = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if (class_value not in sep):
                sep[class_value] = list()
            sep[class_value].append(vector)
            
        #=======================================    
        #class probability
#        sep_data = separate_by_class(X1,y1)
        l = []
        self.P_c = []
        self.k = len(sep)
        for i in range(self.k):
            l.append(len(sep[i]))
        
        s = sum(l)
        for i in range(0,self.k):
            self.P_c.append(l[i]/(s))
        self.P_c = np.log(self.P_c)
        
        #==========================
        #calculating mean
        class_data = []
        self.mean = []
        self.d = X1.shape[1]
#        sep = separate_by_class(X1,y1)
        for i in range(0,self.k):
            cd = pd.DataFrame(sep[i])
            m = np.mean(cd.iloc[:,:self.d])
            class_data.append(cd)
            self.mean.append(pd.DataFrame(m).T)
            
        #======================================
        #cal mean sub data
        class_data = []
#        sep = separate_by_class(X1,y1)
#        d = X1.shape[1]
        mux = []
#        mean = mean1(X1,y1)
        for i in range(0,self.k):
            cd = pd.DataFrame(sep[i])
            
            class_data.append(cd)
            for j in range(len(cd)):
                mux.append(cd.iloc[j,:self.d] - self.mean[i].iloc[:,:])
                
        mux = pd.concat(mux)
        
        mux.reset_index(drop=True, inplace=True)
        #=======================================
        #cal covariance
        epsilon = 0.5
        self.det_cov = []
        self.i_cov = []
        self.cov = []
        mu_x = []
        l = []
      
        mux_prod = np.zeros((self.d,self.d))
#        sep = separate_by_class(X1,y1)
        for i in range(self.k):
            l.append(len(sep[i]))
        for n in range(1,len(l)):
            l[n] = l[n-1]+l[n]
            
            
        for j in range(len(l)):
            if j == 0:
                mu_x.append(mux.iloc[0:l[j],:])
            else:
                mu_x.append(mux.iloc[l[j-1]:l[j]]) #class sep mux=mu_x
        #cal mean sub class wise data in mu_x        
          
    
            
        self.cov = []
        for t in range(len(mu_x)):
            x1 = mu_x[t]
            Dr = len(x1)
#            d = x1.shape[1]
            mux_prod=np.zeros((self.d,self.d))
            
            for u in range(Dr):
    
                x2 = np.asarray(x1.iloc[u,:]).reshape(self.d,1)
                mux_prod += (1/Dr)*np.matmul(x2,x2.T)
            self.cov.append(mux_prod)   
                
          
        #converting singular mat to non sing
        for p in range(len(self.cov)):
    #        for u in range(len(cv)):
            df = pd.DataFrame(self.cov[p])
            for tt in range(len(df)):
                if (df.iloc[:,tt]).all()==0:
                    df += epsilon*np.eye(self.d)
            self.cov[p] = df
       
         #converting to diag cov       
            if diag:
                   
                for ttt in range(len(self.cov[p])):
                    for pp in range(len(self.cov[p])):
                        if ttt!=pp:
                            self.cov[p].iloc[ttt,pp] = 0
    #        cov[p] = cov[p]/LA.norm(cov[p])
            self.i_cov.append(LA.inv((self.cov[p]).values))
            self.det_cov.append(LA.det((self.cov[p]).values))
        #==================================================
        #discriminant function
        sum1=[]
        
        for i in range(self.k):
            for n in range(len(sep[i])):
        
                S = -0.5*np.log(self.det_cov[i])
                
                mux= pd.DataFrame(sep[i]).iloc[n,:self.d]-self.mean[i]
                prod1 = -0.5*np.dot(mux,self.i_cov[i])
                prod2 = np.dot(prod1,mux.T)
                for j in range(self.k):
                    pc = self.P_c[j]
                    sum1.append(S+prod2[0][0]+pc)
        sum1=(np.asarray(sum1)) .reshape(self.k,len(X1)) 
        ind = []
        for ii in range(sum1.shape[1]):
            ind.append(np.argmax(sum1[:,ii]))
                    
            
            

    def predict(self,X1):
        sum1_p=[]

        for n in range(len(X1)):
            for i in range(self.k):
            
        
                S_p = -0.5*np.log(self.det_cov[i])
                
                mux_p=((np.asarray(X1.iloc[n,:self.d])).reshape(self.d,1)).T - self.mean[i].iloc[:,:self.d]
                prod1_p = -0.5*np.dot(mux_p,self.i_cov[i])
                prod2_p = np.dot(prod1_p,mux_p.T)
    #            for j in range(len(P_c)):
                pc_p = self.P_c[i]
                sum1_p.append(S_p+prod2_p[0][0]+pc_p)
        sum1_p=(np.asarray(sum1_p)).reshape(len(self.P_c),len(X1)) 
        ind_p = []
        for ii in range(sum1_p.shape[1]):
            ind_p.append(np.argmax(sum1_p[:,ii]))
        return ind_p
                  
                
        
            
    
    
    
        
        
        
    
        
