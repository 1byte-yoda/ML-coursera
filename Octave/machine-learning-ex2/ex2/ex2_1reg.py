import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
def load_data():
    data = pd.read_csv("ex2data2.txt", header=None)
    X = data.iloc[:, [0, 1]].values;
#    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    y = data.iloc[:, 2:].values

    return X, y

def visualize(X, y, xlabel, ylabel, pos_label, neg_label, axis=None):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    
    if axis == None:
        axis = plt.gca()
        
    axis.scatter(X[pos, 0], X[pos, 1], label = pos_label, c='black', marker="+")
    axis.scatter(X[neg, 0], X[neg, 1], label = neg_label, c='y', marker="o")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(loc='best')

def sigmoid(z):
    return 1 / (1+np.exp(-z))


def magic_odd(n):
    if n % 2 == 0:
        raise ValueError('n must be odd')
    return np.mod((np.arange(n)[:, None] + np.arange(n)) + (n-1)//2+1, n)*n + \
          np.mod((np.arange(1, n+1)[:, None] + 2*np.arange(n)), n) + 1

def lrCostFunction(theta, lambda_, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    class_0 = np.sum(np.dot(np.transpose(y), (np.log(h))))
    class_1 = np.sum(np.dot(1-np.transpose(y), np.log(1-h)))
    J = (class_0 + class_1) / -m
#    theta[0] = 0
    reg = lambda_/(2*m) * np.sum(np.square(theta))
    J_reg = J + reg
    return J_reg

def lrGradient(theta, lambda_, X, y):
    m = len(y)
    h = sigmoid(X.dot(theta.reshape((-1,1))))
    gradient = (np.dot(np.transpose(X), (h-y))) / m
#    theta[0] = 0
    reg = (lambda_/m) * theta.reshape((-1,1))
    gradient_reg = gradient + reg
    return gradient_reg.flatten()

def predict(X, theta):
    p = sigmoid(X.dot(theta)) >= 0.5
    return p
if __name__ == "__main__":
    X, y = load_data()
    poly = PolynomialFeatures(6)
    XX = poly.fit_transform(X)
    lambda_ = 0.1
    theta = np.zeros((XX.shape[1], 1))
#    visualize(X, y, "Microchip 1", "Microchip 2", "y=1", "y=0", axis=None)
    fig, axes = plt.subplots(1,3, figsize=(14,5))
    for i, C in enumerate([0,1,100]):
        fminunc = minimize(fun=lrCostFunction, jac=lrGradient, x0=theta,
                       args=(C, XX, y), method="BFGS", options={"maxiter":3000})
        visualize(X, y, "Microchip 1", "Microchip 2", "y=1", "y=0", axis=axes[i])
        accuracy = np.mean(predict(XX, fminunc.x)==y.ravel()) * 100
        
        x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
        X1, X2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
        z = poly.fit_transform(np.c_[ X1.ravel(), X2.ravel()]).dot(fminunc.x)
        h = sigmoid(z)
        h = h.reshape(X1.shape)
        axes[i].contour(X1, X2, h, [0.5], colors='g')
        axes[i].set_title(accuracy)
#    X = np.append(np.ones((3,1)), magic_odd(3), axis=1)
#    y = np.array([1, 0, 1]).T
#    theta = np.array([-2, -1, 1, 2]).T
#    print(lrGradient(theta, 4, X, y))
