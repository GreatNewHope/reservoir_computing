#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:44:18 2017

@author: marcos
"""

import numpy as np
import matplotlib.pyplot as plt

def reservoir1(X, t, D, d):
    """
    Este va a ser el reservorio con regresión lineal y crossvalidation
        
        X = inputs; t = targets; D = número neuronas en hidden layer
        d = número de neuronas en output layer; n = número de ejemplos
        
    """
    n = X.shape[1]
    predicciones = []
    
    #Creación del reservorio
    np.random.seed(61950)
    Win = np.random.rand(X.shape[0], D) - 0.5
    
    for i in range(41):
        
        # Separación entre training y test data
        rangotrain = list(range(i * 40)) + list(range((i + 1) * 40, 1680))
        rangotest = range(i * 40, (i + 1) * 40)
        Xtrain = X[:, rangotrain] 
        Xtest = X[:, rangotest]
        ttrain = np.reshape(t[rangotrain, 0], (1, n - 40))
        
        # Optimización del reservorio
        ztrain = np.matmul(np.matrix.transpose(Win), Xtrain)
        rtrain = np.sin(0.1 * ztrain + 2.1) ** 2
        rtrain = np.insert(rtrain, 0, 1, axis = 0)
        Weights = np.matrix.transpose(np.matmul(ttrain, np.linalg.pinv(rtrain)))
    
        #Prueba en el test
        ztest = np.matmul(np.matrix.transpose(Win), Xtest)
        rtest = np.sin(0.1 * ztest + 2.1) ** 2
        rtest = np.insert(rtest, 0, 1, axis = 0)
        otest = np.matmul(np.matrix.transpose(Weights), rtest)
        
        #Decidir si el paciente es epiléptico o no
        predicciones.append(np.median(otest))
        
    return (predicciones, Weights)

def imprimir(otrain, otest, targets):
    """
    Usado para sacar las gráficas
    """
    plt.figure().add_subplot(111).plot(np.matrix.transpose(otrain))
    plt.figure().add_subplot(111).plot(np.matrix.transpose(otest))
    plt.figure().add_subplot(111).plot(targets)