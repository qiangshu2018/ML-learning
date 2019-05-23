# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:21:54 2019

@author: qiangshu
"""

import numpy as np
import pandas as pd
import random
from sklearn.datasets import load_iris
data = load_iris()

Dataset = pd.DataFrame(data = data.data,columns = data.feature_names)
Dataset['class'] = data.target

def rand_split(dataset, rate):
    dataset_len = dataset.shape[0]
    idx = list(range(dataset_len))
    random.shuffle(idx)
    train_num = int(dataset_len*rate)
    train_data = dataset.iloc[idx][:train_num]
    test_data  = dataset.iloc[idx][train_num:]
    return train_data,test_data

def gnb_classfy(train,test):
    labels = train.iloc[:,-1].value_counts().index
    mean = [] #存放每个类别的均值
    std  = [] #存放每个类别的方差
    result = [] #存放每个类别的预测结果
    for i in labels:
        item = train.loc[train.iloc[:,-1]==i]
        m = item.iloc[:,:-1].mean().values
        s = np.sum((item.iloc[:,:-1]-m)**2)/item.shape[0]
        mean.append(m)
        std.append(s)
    stds  = pd.DataFrame(std,index=labels)
    means = pd.DataFrame(mean,index=labels,columns = stds.columns)
#    return means,stds
    for j in range(test.shape[0]):
        iset = test.iloc[j,:-1].tolist()
        iprob = np.exp(-1*(iset-means)**2/(2*stds))/(np.sqrt(2*np.pi*stds))
        prob = 1
        for k in range(test.shape[1]-1):
            prob *= iprob.iloc[:,k]
            cla = prob.index[np.argmax(prob.values)]
        result.append(cla)
    test['predict'] = result
    acc = sum(test.iloc[:,-1] == test.iloc[:,-2])/test.shape[0]
    return acc, test
    
if __name__ == '__main__':
    Train_data,Test_data = rand_split(Dataset,0.75)
    acc,test = gnb_classfy(Train_data,Test_data)
    
    
    
    
    
    
    
    