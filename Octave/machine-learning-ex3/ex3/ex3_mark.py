import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

def load_data():
    data = loadmat("ex3data1.mat");
    X = data["X"];
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    y = data["y"];
    weights = loadmat("ex3weights.mat");
    theta1, theta2 = weights["Theta1"], weights["Theta2"];
    return X, y, theta1, theta2;

def visualize(X):
    fig, axes = plt.subplots(10, 10, figsize=(10,10));
    rows, cols = 10, 10;
    for i in range(rows):
        for j in range(cols):
            rand_index = np.random.randint(1, X.shape[0]);
            axes[i, j].imshow(X[rand_index].reshape((20,20), order='F'));
            axes[i,j].axis("off");

def sigmoid(z):
    return 1 / (1+np.exp(-z));

def lrCostfunctionReg(theta, lambda_, X, y):
    m = y.size
    h = sigmoid(np.dot(X, theta))
    class_0 = np.sum(np.dot(np.transpose(y), np.log(h)))
    class_1 = np.sum(np.dot(1-np.transpose(y), np.log(1-h)))
    J = (class_0 + class_1) / -m
    reg = lambda_/(2*m) * np.sum(np.square(theta[1:].reshape((-1,1))))
    J = J + reg
    return J

def lrGradientReg(theta, lambda_, X, y):
    m = y.size
    h = sigmoid(np.dot(X, theta.reshape((-1,1))))
    gradient = np.dot(np.transpose(X), (h-y)) * (1/m)
    theta[: 0] = 0
    reg = (lambda_/m) * theta.reshape((-1,1))
    gradient = gradient + reg    
    return gradient.flatten()

def oneVsAll(X, y, n_labels, lambda_):
    init_theta = np.zeros((X.shape[1], 1));
    all_opt_theta = np.zeros((n_labels, X.shape[1]))
    
    for i in range(1, n_labels+1):
        fminunc = minimize(fun=lrCostfunctionReg, jac=lrGradientReg, x0=init_theta, 
                           args=(lambda_, X, (y==i)*1), method=None, options={"maxiter": 50})
        all_opt_theta[i-1] = fminunc.x
    return all_opt_theta

def predictOneVsAll(X, theta):
    p = sigmoid(np.dot(X, np.transpose(theta)))
    return np.argmax(p, axis=1) + 1

def predictNN(X, theta1, theta2):
    a1 = X
    z2 = np.dot(a1, np.transpose(theta1))
    a2 = np.append(np.ones((z2.shape[0], 1)), sigmoid(z2), axis=1)
    z3 = np.dot(a2, np.transpose(theta2))
    a3 = sigmoid(z3)
    return np.argmax(a3, axis=1) + 1

if __name__ == "__main__":
    X, y, theta1, theta2 = load_data()
    visualize(X[:, 1:])
    theta = oneVsAll(X, y, 10, 0.1)
    pred_lr = predictOneVsAll(X, theta)
    pred_nn = predictNN(X, theta1, theta2)
    print("Logistic Regression Accuracy: {0:.2f}".format(np.mean(pred_lr == y.ravel()) * 100))
    print("Nerual Network Accuracy: {0:.2f}".format(np.mean(pred_nn == y.ravel()) * 100))