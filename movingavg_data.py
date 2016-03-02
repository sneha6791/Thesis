# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 17:42:41 2016

@author: sneha
"""
import numpy
window_width = 8

data = []
for line in file('movingavg_data'):
    data.append(float(line.strip().split('\n')[0]))
cumsum_vec = numpy.cumsum(numpy.insert(data, 0, 0))
ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
for x in ma_vec:
        print x
