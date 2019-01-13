# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 00:42:24 2018

@author: Razer
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from sklearn.preprocessing import PolynomialFeatures
    
def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # Get indexes for class 0 and class 1
    neg = data[:, 2] == 0
    pos = data[:, 2] == 1
    
    if axes == None:
        axes=plt.gca()
    #axes.scatter(0.4656 ,-0.5132 ,marker='v', c='r', label=('85,45'))
    axes.scatter(data[neg][:, 0], data[neg][:, 1], c='k', marker='x', label=label_neg)
    axes.scatter(data[pos][:, 0], data[pos][:, 1], c='y', marker='o', label=label_pos)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend()
    
def sigmoid(z):
    return(1 / (1 + np.exp(-z)))
    
def predict(theta, X, threshold=0.5):
    p = sigmoid(np.dot(X, theta)) >= threshold
    return p


def costFunctionReg(theta, reg, XX, y):
    m = len(y)
    h = sigmoid(np.dot(XX, theta))
    class_1 = np.sum(np.dot(np.transpose(y), np.log(h)))
    class_2 = np.sum(np.dot((1-np.transpose(y)), np.log(1-h)))
    reg = np.sum(np.square(theta)) * reg/(2*m)
    J = ((class_1 + class_2)/-m) + reg
    return J

def gradientReg(theta, reg, XX, y):
    m = len(y)
    h = sigmoid(np.dot(XX, theta.reshape((-1,1))))
    predictions = np.dot(np.transpose(XX), (h-y)) * 1/m
    reg = reg/m * theta.reshape((-1,1))
    predictions = (predictions + reg)
    return predictions.flatten()

if __name__ == '__main__':
    data2 = pd.read_csv('ex2data2.txt', header=None)
    y = data2.iloc[:,2:].values
    X = data2.iloc[:,0:2].values
    plotData(data2.values, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0')
    
    poly = PolynomialFeatures(8)
    XX = poly.fit_transform(X)
    
    initial_theta = np.zeros((XX.shape[1], 1))
    print(costFunctionReg(initial_theta, 0, XX, y))
    fig, axes = plt.subplots(1,3, sharey = True, figsize=(14,5))
    
    # Decision boundaries
    # Lambda = 0 : No regularization --> too flexible, overfitting the training data
    # Lambda = 1 : Looks about right
    # Lambda = 100 : Too much regularization --> high bias
    for i, C in enumerate([0, 1, 100]):
        # Optimize costFunctionReg
        res2 = minimize(costFunctionReg, initial_theta, args=(C, XX, y), method='BFGS', jac=gradientReg, options={'maxiter':3000})
        
        # Accuracy
        accuracy = 100*sum(predict(res2.x, XX) == y.ravel())/y.size    
    
        # Scatter plot of X,y
        plotData(data2.values, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes.flatten()[i])
        
        # Plot decisionboundary
        x1_min, x1_max = X[:,0].min(), X[:,0].max()
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
        X1, X2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
        h = sigmoid(np.dot(poly.fit_transform(np.c_[X1.ravel(), X2.ravel()] ), res2.x))
        h = h.reshape(X1.shape)
        axes.flatten()[i].contour(X1, X2, h, [0.5], linewidths=1, colors='g');       
        axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))