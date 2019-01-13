# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 21:50:43 2018

@author: Razer
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def declare_var():
    data = pd.read_csv("ex2data1.txt", header=None)
    X = data.iloc[:, [0,1]].values
    X = np.append(np.ones((len(X), 1)), X, axis=1)
    y = data.iloc[:, -1:].values
    theta = np.zeros((X.shape[1],1))
    return X, y, theta

def hypothesis(z):
    return 1 / (1 + np.exp(-z))

def cost_function(theta, X, y):
    reg = 1
    m = len(X)
    h = hypothesis(np.dot(X, theta.reshape((-1,1))))
    class_1 = np.sum( np.dot(np.transpose(y), np.log(h)) )
    class_0 = np.sum( np.dot(1-np.transpose(y), np.log(1-h)) )
    #reg_cost = reg/(2*m) * np.sum(np.square(theta))
    cost = ((class_1 + class_0)/-m) #+ reg_cost
    return cost

def gradient(theta, X, y, alpha=1):
    reg = 1
    m = len(X)
    h = hypothesis(np.dot(X, theta.reshape((-1,1))))
    update = np.dot(np.transpose(X), (h-y))
    #reg_grad = (reg/(m)) *  theta.reshape((-1,1))
    update = ((1/m) * update) #+ reg_grad
    return update.flatten()

def predict(theta, X):
    p = hypothesis(np.dot(X, theta)) >= 0.5
    return(p.astype('int'))
    
def plot(X, y, res=0):
#    data = np.loadtxt('ex2data1.txt', delimiter=',')
    X = X[:, [1,2]]
    y = y[:, 0]
    
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.scatter(X[pos, 0], X[pos, 1], marker='x', c='k', label = 'Admited')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='y', label = 'Not Admited')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(frameon= True, fancybox = True)
    plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')

    x1_min, x1_max = X[:,0].min(), X[:,0].max(),
    x2_min, x2_max = X[:,1].min(), X[:,1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = hypothesis( np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b');
    plt.show()

if __name__ == "__main__":
    X, y, theta = declare_var()
    
    cost = cost_function(theta, X, y)
    grad = gradient(theta, X, y)
    res = minimize(fun = cost_function, 
                                 x0 = theta, 
                                 args = (X, y),
                                 method = None,
                                 jac = gradient,
                                 options={'maxiter':3000})
    p = predict(res.x, X)
    print('Train accuracy {}%'.format(100*np.mean(p==y.flatten())))
    plot(X, y, res)

    
