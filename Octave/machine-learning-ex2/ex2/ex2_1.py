# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 20:27:04 2018

@author: Razer
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
def get_data():
    data = pd.read_csv("ex2data1.txt", header=None)
    X = data.iloc[:, [0,1]].values
    X = np.append(np.ones((len(X), 1)), X, axis=1)
    y = data.iloc[:, 2:].values
    theta = np.zeros((X.shape[1], 1))
    return X, y, theta

def plot(X, y, theta, xlabel, ylabel, pos_label, neg_label, axes=None):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    
    if axes == None:
        axes = plt.gca()
    axes.scatter(X[pos, 1], X[pos, 2], c='k', linewidths=1, marker="+", label=pos_label)
    axes.scatter(X[neg, 1], X[neg, 2], c='y', linewidths=1, marker="o", label=neg_label)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.legend(loc=0)
    draw_boundary(X, y, theta, axes)
    
def sigmoid(X, theta, is_vec=True):
    if is_vec:
        z = np.dot(X, theta)
    else:
        z = X
    return 1 / (1 + np.exp(-z))

def costFunction(theta, X, y):
    h = sigmoid(X, theta)
    m = X.shape[0]
    class_1 = np.sum(np.dot(np.transpose(y), np.log(h)))
    class_0 = np.sum(np.dot(1-np.transpose(y), np.log(1-h)))
    cost = (class_1 + class_0) / -m
    return cost

def gradient(theta, X, y):
    m = X.shape[0]
    h = sigmoid(X, theta.reshape((-1,1)))
    gradient = np.dot(np.transpose(X), (h-y))
    gradient = (1/m) * gradient
    return gradient.flatten()

def draw_boundary(X, y, theta, axes):
    min_cost = minimize(costFunction, theta, args=(X, y), jac=gradient, options={"maxiter":400}, method=None)
    X1_min, X1_max = X[:, 1].min(), X[:, 1].max()
    X2_min, X2_max = X[:, 2].min(), X[:, 2].max()
    xx1, xx2 = np.meshgrid(np.linspace(X1_min, X1_max), np.linspace(X2_min, X2_max))
    bias = np.ones((xx1.ravel().shape[0], 1))
    h = sigmoid(np.c_[bias, xx1.ravel(), xx2.ravel()], min_cost.x)
    h = h.reshape(xx1.shape)
    axes.contour(xx1, xx2, h, [0.5], colors='blue', linewidths=1)
    
def predict(X, theta):
    p = sigmoid(X, theta) >= 0.5
    return p    

def magic_odd(n):
    if n % 2 == 0:
        raise ValueError('n must be odd')
    return np.mod((np.arange(n)[:, None] + np.arange(n)) + (n-1)//2+1, n)*n + \
          np.mod((np.arange(1, n+1)[:, None] + 2*np.arange(n)), n) + 1
          
if __name__ == '__main__':
    X, y, theta = get_data()
#    X = np.array([[1, 1] , [1, 2.5] , [1 ,3] , [1, 4]])
#    theta = np.array([[-3.5] , [1.3]])

#    X = np.append(np.ones((3,1)), magic_odd(3), axis=1)
#    y = np.array([1, 0, 1]).T
#    theta = np.array([-2, -1, 1, 2]).T
    
    J = costFunction(theta, X, y)
    g = gradient(theta, X, y)
    min_cost = minimize(costFunction, x0=theta, args=(X, y),
                        jac=gradient, options={"maxiter":400}, method=None)
    predictions = predict(X, min_cost.x)
    plot(X, y, min_cost.x, "Exam Score 1", "Exam Score 1", "Admitted", "Not Admitted", axes=None)
    plt.title("Train Accuracy {0:.2f}%".format(100*np.mean(predictions == y.flatten())))
    