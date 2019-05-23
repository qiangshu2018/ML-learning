# -*- coding: utf-8 -*-
"""
Created on Fri May  3 19:39:33 2019

@author: qiangshu
"""

import numpy as np
import pandas as pd

def createDataSet():
    row_data = {'no surfacing': [1,1,1,0,0],
                'fish': ['yes','yes','no','no','no'],
                'flippers': [1,1,0,1,1]
                }
    dataset = pd.DataFrame(row_data,columns=['no surfacing','flippers','fish'])
    return dataset

def calc_entropy(dataset):
    label = dataset.iloc[:,-1]
    label_count = label.value_counts().values
    entropy = sum([-n/sum(label_count)*np.log2(n/sum(label_count)) for n in label_count])
    return entropy 
    
def calc_entropy_gain(dataset):
    maxgain_axis = -1
    best_sub_HD_entropy = 1000
    sub_HD_entropy = 0
    for i in range(dataset.shape[1]-1):
        labels = dataset.iloc[:,i].value_counts().index
        for j in (labels):
            sub_HD = dataset[dataset.iloc[:,i]==j]
            sub_HD_entropy += sub_HD.shape[0]/dataset.shape[0]*calc_entropy(sub_HD)
        if sub_HD_entropy < best_sub_HD_entropy:
            best_sub_HD_entropy = sub_HD_entropy
            maxgain_axis = i
            
    return maxgain_axis, calc_entropy(dataset)-best_sub_HD_entropy
 
def split_dataset(axis,dataset,value):
    col = dataset.columns[axis]
    redataset = dataset.loc[dataset[col]==value,:].drop(col,axis=1)
    return redataset
    
def createtree(dataset):
    featlist = list(dataset.columns)
    classlist = dataset.iloc[:,-1].value_counts()
    if classlist[0]==dataset.shape[0] or len(featlist)==1:
        return classlist.index[0]
    split_axis, Best_sub_Entropy = calc_entropy_gain(dataset)
    bestfeat = featlist[split_axis]
    mytree = {bestfeat:{}}
    del featlist[split_axis]
    valuelist = set(dataset.iloc[:,split_axis])
    for value in valuelist:
        mytree[bestfeat][value] = createtree(split_dataset(split_axis,dataset,value))
    return mytree

def classify(input_tree,labels,testvec):
    firststr = next(iter(input_tree))
    second_dict = input_tree[firststr]
    featIndex = labels.index(firststr)
    for key in second_dict.key():
        if testvec[featIndex]==key:
            if type(second_dict[key])==dict:
                class_label = classify(second_dict[key],labels,testvec)
            else:
                class_label = second_dict[key]
    return class_label
    
if __name__ == '__main__':
    DataSet = createDataSet()
    Entropy = calc_entropy(DataSet)
    createtree(DataSet)
#    split_axis, Best_sub_Entropy = calc_entropy_gain(DataSet)