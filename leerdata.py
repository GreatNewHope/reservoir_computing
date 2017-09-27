#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:54:04 2017

@author: marcos
"""

import scipy.io
import numpy

# Solo sacas los targets positivos para los sanos, que son los primeros 14 * 40
data = scipy.io.loadmat('data.mat')
inputs = numpy.insert(data['inputs'], 0, 1, axis = 0)
targets = data['targets'][:, 0]
targets = numpy.reshape(targets,(1680,1))