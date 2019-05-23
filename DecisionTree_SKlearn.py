# -*- coding: utf-8 -*-
"""
Created on Sat May  4 21:41:08 2019

@author: qiangshu
"""

import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz 

def createDataSet():
    row_data = {'no surfacing': [1,1,1,0,0],
                'fish': ['yes','yes','no','no','no'],
                'flippers': [1,1,0,1,1]
                }
    dataset = pd.DataFrame(row_data,columns=['no surfacing','flippers','fish'])
    return dataset
    
DataSet = createDataSet()
xtrain = DataSet.iloc[:,:-1]
ytrain = DataSet.iloc[:,-1]
labels = ytrain.unique().tolist()
ytrain = ytrain.apply(lambda x: labels.index(x))
#ytrain = [1 if i=='yes' else 0 for i in ytrain]

clf = DecisionTreeClassifier()
clf = clf.fit(xtrain,ytrain)
tree.export_graphviz(clf)

dot_data = tree.export_graphviz(clf,out_file=None)
graphviz.Source(dot_data)

#dot_data = tree.export_graphviz(clf,out_file=None,feature_names=['no surfacing', 'flippers'],
#                                clas_name=['fish','notfish'],filled=True,rouded=True,)