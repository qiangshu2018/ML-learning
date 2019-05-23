# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:55:02 2019

@author: qiangshu
"""

import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10,10,1000)
u = 0
std = 1
y = np.exp(-(x-u)**2/(2*std))/np.sqrt(std*2*np.pi)
plt.plot(x,y)