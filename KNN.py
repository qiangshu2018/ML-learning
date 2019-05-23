# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:07:13 2019

@author: qiangshu
"""

import pandas as pd
import numpy as np
rowdata = {'name':['dont-ask','us','fromer','sea-action','town','wolf2'],
           'pk':[1,5,12,108,112,115],
            'kiss':[101,89,97,5,9,8],
            'genre':['l','l','l','a','a','a']}

DataSet = pd.DataFrame(rowdata,columns = ['name','pk','kiss','genre'])


def classify(k,dataset,test_data):
    distance = np.sum((dataset.iloc[:,1:3].values-[test_data])**2,axis=1)
    dataset['distance'] = distance
    new_data = dataset.sort_values(by='distance')
    ret_labels = new_data['genre'].iloc[0:k].value_counts().index[0]
    return ret_labels

classify(4,DataSet,[3.67])
    