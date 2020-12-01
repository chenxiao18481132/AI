# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:28:07 2019

@author: jianfal
"""
import matplotlib.pyplot as plt
from math import sqrt
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.utils import to_categorical

from sklearn import preprocessing 



#c_training = pd.read_csv('c_training.csv')
m_training = pd.read_csv('m_training.csv')
#c_training=pd.concat([c_training,m_training])
c_training=m_training


#############################
factor_list=['Hour','Position','Weight','Age','Percent','PpgPkValAcMedian','PpgVlValAcMedian',
             'RatioPpgMusAcValMedian','RatioB_A_Median','PpgStart2MusIntervalMedian',
             'RatioPpgStart2MusIntervalMedian','PpgStart2PpgEndIntervalMedian','PpgStart2MusAreaMedian','Mus2SecMaxAreaMedian','Pk2SecMaxIntervalMedian','PpgMus2PkLpAcAreaMedian','RatioPk2SecMaxIntervalMedian','PpgStart2ThdBIntervalMedian','RatioPpgStart2ThdBIntervalMedian','RatioPpgStart2MusLpAcAreaMedian','RatioPpgPk2EndLpAcAreaMedian','RatioPpgSecMax2PkLpAcAreaMedian','RatioThdBpoint2StartLpAcAreaMedian','RatioPk2PRMedian','logRelativePpgMusAcValMedian','logRatioPpgMusAcValMedian','logRelativeThdBpointValMedian','logRatioThdFpointValMedian','logRatioB_A_Median','logRatioBE_A_Median','logPpgStart2MusIntervalMedian','logRatioPpgStart2MusIntervalMedian','logPpgStart2PpgEndIntervalMedian','logRatioMus2PpgEndIntervalMedian','logPk2SecMaxIntervalMedian','logRatioPk2SecMaxIntervalMedian','logPk2PpgStartAreaMedian','logPpgStart2ThdBIntervalMedian','logRatioPpgStart2ThdBIntervalMedian','logRatioThdB2EndIntervalMedian','logPpgStart2MusLpAcAreaMedian','logRatioPpgStart2MusLpAcAreaMedian','logRatioPpgPk2EndLpAcAreaMedian','logRatioPpgSecMax2PkLpAcAreaMedian','logRatioThdBpoint2StartLpAcAreaMedian','logRatioEpointPre2PostLpAreaMedian','logPulseRateMedian']
factor_list_without_ppg=['Hour','Position','Weight','Age']
#########################
z=c_training[["Folder"]]
y=c_training[['Sbp','Dbp']]
x=c_training[factor_list]
#x=c_training[factor_list_without_ppg]
x=np.array(x)
x = preprocessing.scale(x)#标准化
x=np.concatenate((np.ones((x.shape[0],1)),x),axis=1).astype(float)
#x=np.ones((y.shape[0],1))
y=np.array(y)


#z=np.array(z,dtype=int)



y_class=np.zeros((len(z),1),dtype=int)
z_len=len(z)




for i in range(z_len):
#    if i%1000==0:
#        print(i)
    if y[i][0]>140 or y[i][1]>90:
        y_class[i][0]=1

#test_size=0.3
x_train, x_test, y_class_train, y_class_test, z_train, z_test = train_test_split(x, y_class,z, test_size = test_size)
x_train=x[:30000]
y_class_train=y_class[:30000]
z_train=z[:30000]
class_mapping = {label:idx for idx,label in enumerate(set(z_train['Folder']))}
z_train['Folder'] = z_train['Folder'].map(class_mapping)
z_train=np.array(to_categorical(z_train),dtype=int)
#z=np.array(to_categorical(z),dtype=int)

y_class_test=y_class[30000:35000]
x_test=x[30000:35000]
z_test=np.zeros((len(x_test),np.shape(z_train)[1]),dtype=int)
####cov
for i in range(len(x_test)):
    cov_result=[]
    print(i)
    for j in range(len(x_train)):
        result=np.dot(x_test[i],x_train[j])/(np.dot(x_train[j],x_train[j])*np.dot(x_test[i],x_test[i]))**0.5
        
        cov_result.append(result)
    z_test[i]=z_train[cov_result.index(max(cov_result))]
####distance
for i in range(len(x_test)):
    cov_result=[]
    print(i)
    for j in range(len(x_train)):
        result=np.dot(x_test[i]-x_train[j],x_test[i]-x_train[j])
        cov_result.append(result)
    z_test[i]=z_train[cov_result.index(min(cov_result))]  
#z_test=z[30000:35000]


#y_class_train, y_class_test = train_test_split(y_class, test_size = test_size)

#x_train, x_test, y_train, y_test,z_train, z_test = train_test_split(x_test, y_class_test,z_test, test_size = 0.3)
def lamda(eita):
    s=1/(1+np.exp(-eita))
    lamda=(1/(2*eita))*(s-0.5)
    return lamda


def Estep(beta_old,sigma_old,Y,X,Z,epsilon):
    N=Y.shape[0]
    #L=X.shape[1]
    K=Z.shape[1]
    Sigma=(1/sigma_old)*(np.ones(K))
    
    for i in range(N):
        Sigma=Sigma+2*lamda(np.sqrt(epsilon[i]))*Z[i,]#(np.outer(Z[i,],Z[i,]).diagonal())
    
    Sigma1=1/(Sigma)
    mu=np.zeros(K)
    for i in range(N):
        if Y[i]==1:
            mu=mu+Z[i,]-2*Y[i]*Z[i,]+4*lamda(np.sqrt(epsilon[i]))*(np.dot(X[i,],beta_old))*Z[i,]
    p=-0.5*Sigma1*mu
    list1=[p,Sigma1]
    return list1

def Mstep(beta_old,sigma_old,Y,X,Z,epsilon,p,Sigma):
    epsilon_new=epsilon
    N=Y.shape[0]
    L=X.shape[1]
    K=Z.shape[1]    
    for i in range(N):
        epsilon_new[i]=np.dot(Sigma,Z[i,])+(np.dot(p,Z[i,]))**2+2*(np.dot(X[i,],beta_old))*(np.dot(Z[i,],p))+(np.dot(X[i,],beta_old))**2
    S=np.zeros((L,L))
    M=np.zeros(L)
    for i in range(N):
        S=S+2*lamda(np.sqrt(epsilon_new[i]))*np.outer(X[i,],X[i,])
        M=M+Y[i]*X[i,]-2*lamda(np.sqrt(epsilon_new[i]))*(np.dot(Z[i,],p))*X[i,]-0.5*X[i,]
    beta_new=np.matmul(np.linalg.inv(S),M)
    sigma_new=(np.dot(p,p)+np.sum(Sigma))/K
    list2=[beta_new,sigma_new,epsilon_new]
    return list2

beta_old=np.random.random(size=48)

N=len(z_train)
Y=y_class_train
Z=z_train
X=x_train

sigma_old=1
#beta_old=[0.1,0.8]
#N=n
#Y=y
#Z=z
epsilon=np.zeros(N)
for i in range(N):
    epsilon[i]=sigma_old*np.dot(Z[i,],Z[i,])+(np.dot(X[i,],beta_old))**2

"""Variational EM"""

K=Z.shape[1] 
epochs=200
g=-100000000
g_list=[]
for e in range(epochs):
    p,Sigma=Estep(beta_old,sigma_old,Y,X,Z,epsilon)
    beta_old,sigma_old,epsilon=Mstep(beta_old,sigma_old,Y,X,Z,epsilon,p,Sigma)
    g_old=g
    g=-0.5*K*np.log(sigma_old)+0.5*np.sum(np.log(Sigma))#lower bound
    for i in range(N):
        g=g+Y[i]*(np.dot(Z[i,],p)+np.dot(X[i,],beta_old))-lamda(np.sqrt(epsilon[i]))*(np.dot(Sigma,Z[i,])+(np.dot(p,Z[i,]))**2+2*(np.dot(X[i,],beta_old))*(np.dot(Z[i,],p))+(np.dot(X[i,],beta_old))**2-epsilon[i])-0.5*(np.dot(Z[i,],p))-0.5*(np.dot(X[i,],beta_old))-0.5*np.sqrt(epsilon[i])+np.log(1/(1+np.exp(-np.sqrt(epsilon[i]))))
    if abs(g-g_old)<0.01:
        print("finished")
        break
    g_list.append(g)
    print(beta_old,sigma_old,g)



def prediction(x,beta,p,Sigma,z):#  pi
    mu=np.dot(z,p)
    sigma=np.dot(Sigma,z)# posterior mean and sigma of random effects
    def w(u):#pi
        b=1/(1+np.exp(-np.dot(x,beta)-u))
        return b
    W=np.zeros(2000)
    for i in range(2000):
        u=np.random.normal(mu, sigma, 1)# Monte carlo method
        W[i]=w(u)
    return np.mean(W)

"""PREDICTION"""

def accuracy(X,Y,Z,t=0.5):
    N=X.shape[0]
    #print(N)
    Y_P=np.zeros(N)
    acc=0
    for i in range(N):
        if i%1000==0:
            print(i)
        if prediction(X[i,],beta_old,p,Sigma,Z[i,])>t:
            Y_P[i]=1
        if Y_P[i]==Y[i]:
            acc+=1
    return round(acc/N,2),Y_P,Y
h_acc_percent=[]
n_acc_percent=[]
for t in range(10):
    threshold=t*0.02+0.05
    acc_1,y_p,y_true=accuracy(x_test[:5000],y_class_test[:5000],z_test[:5000],threshold)
    
    h_acc=0
    h_num=0
    n_acc=0
    n_num=0
    print(len(y_p))
    for i in range(len(y_true)):
        if y_true[i]==1:
            h_num+=1
            if y_p[i]==y_true[i]:
                h_acc+=1
        else:
            n_num+=1
            if y_p[i]==y_true[i]:
                n_acc+=1
    
    print(acc_1,round(h_acc/h_num*100,2),round(n_acc/n_num*100,2))
    h_acc_percent.append(round(h_acc/h_num*100,2))
    n_acc_percent.append(round(n_acc/n_num*100,2))

plt.scatter(h_acc_percent, n_acc_percent, marker='x',c="r")
#plt.scatter(h_test_list[0:50], n_test_list[0:50], marker='x',c="r")
#plt.scatter(h_train_list, n_train_list, marker='x')
plt.ylabel('hyper_acc')
plt.xlabel('norm_acc')
plt.show()

####No feature:
'''
[-2.90111237  0.22647864 -0.07007777  0.20588948  0.35411964] 
sigma: 1.787226104814455 
[-10158.04501572]

0.72 92.75 68.92
0.81 89.05 79.33
0.85 85.19 84.81
0.87 81.32 88.26
0.89 77.78 90.36
0.9 73.75 91.87
0.91 68.28 94.11

[-2.83386054] 
sigma: 1.810854718298414 
[-10267.30730708]

0.13 100.0 0.0
0.62 97.03 56.62
0.81 90.61 79.43
0.84 86.38 84.18
0.87 81.06 88.01
0.89 77.46 90.35
0.9 73.55 92.07
0.9 70.11 93.3
0.91 65.88 94.57
'''







