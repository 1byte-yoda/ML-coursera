# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 21:20:47 2018

@author: Razer
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("ex1data1.txt", header=None)
X = data.iloc[:, 0:1].values
X = np.append(np.ones((len(X), 1)), X, axis=1)
y = data.iloc[:, 1:].values
theta = np.zeros((X.shape[1], 1))

def cost_function(X, y, theta):
    m = len(X)
    hypothesis = np.dot(X, theta)
    squared_error = np.sum((hypothesis-y)**2)
    cost = 1/(2*m) * squared_error
    return cost

def update_theta(X, y, theta, alpha):
    m = len(X)
    hypothesis = np.dot(X, theta)
    gradient = np.dot(np.transpose(X), (hypothesis-y))
    gradient = (1/m * gradient) * alpha
    theta = theta-gradient
    return theta

def train(X, y, theta, alpha, iterations):
    cost_history = []
    
    for i in range(iterations):
        temp_theta = update_theta(X, y, theta, alpha)
        cost = cost_function(X, y, theta)
        if i % 10 == 0:
            print("Iterations: {0}, Cost: {1:0.3f}".format( i, cost))
            cost_history.append(cost)
        if np.abs(np.sum(temp_theta - theta)) < 0.001:
            print("Gradient Descent has converged!, Iterations: %d" % i)
            break
        theta = temp_theta
    return theta

theta = train(X, y, np.array([[0],[0]]),
                                 0.01, 30000) 
plt.figure(1)
plt.scatter(X[:,1], y, color = 'red', marker="+", alpha=0.85)
plt.plot(X[:,1], np.dot(X, theta), color='green', alpha=2)
plt.xlabel("Profit in $10,000s")
plt.ylabel("Population of the City in 10,000s")
plt.legend(["Training data", "Linear Regression"])
plt.show()