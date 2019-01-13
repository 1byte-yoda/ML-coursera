import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from scipy.optimize import fmin_cg

def displayData(X, y):
    plt.scatter(X, y, marker='x', c='red')
    plt.xlabel("Change in water level (x)")    
    plt.ylabel("Water flowing out of the dam (y)")
    plt.legend(["Data", "Hypothesis"])
    plt.show()

def linearRegCostFunction(theta, X, y, lambda_):
    h = np.dot(X, theta)
    m = y.size
    cost = np.sum(np.square(h-y))/(2*m)
    reg = lambda_/(2*m) * np.sum(np.square(theta[1:]))
    return cost + reg

def linearRegGradient(theta, X, y, lambda_):
    h = np.dot(X, theta)
    m = y.size
    grad = np.dot(X.T, (h-y))/m
    reg = (lambda_/m) * theta
    return grad + reg

def trainLinearReg(X, y, lambda_):
    theta = np.zeros(X.shape[1])
    return fmin_cg(f=linearRegCostFunction, x0=theta, 
                   args=(X, y, lambda_), fprime=linearRegGradient, disp=False)
def insertOnes(X):
    ones = np.ones(shape=(X.shape[0], X.shape[1]))
    X_with_ones = np.hstack((ones, X))
    return X_with_ones

def learningCurve(Xtrain, ytrain, Xval, yval, lambda_):
    m = ytrain.size
    train_error = np.zeros(m)
    cv_error = np.zeros(m)
    for i in range(1,m):
        theta = trainLinearReg(Xtrain[:i+1, :], ytrain[:i+1], lambda_)
        train_error[i] = linearRegCostFunction(theta, Xtrain[:i+1, :], ytrain[:i+1], 0)
        cv_error[i] = linearRegCostFunction(theta, Xval, yval, 0)
    plt.plot(range(2, m+1), train_error[1:])
    plt.plot(range(2, m+1), cv_error[1:])
    plt.axis([2, m, 0, 140])
    plt.show()

def poly_features(X, degree):
    X_poly = np.zeros((len(X), degree))
    for i in range(0, degree):
        X_poly[:, i] = X.squeeze() ** (i+1)
    return X_poly

def poly_fit(min_x, max_x, means, stds, theta, degree):
    X =np.linspace(min_x -5, max_x + 5, 1000)
    X_poly = poly_features(X, degree)
    X_poly = (X_poly-means) / stds
    X_poly = insertOnes(X_poly)
    plt.plot(X, X_poly.dot(theta))
    plt.show()
def opt_lambda(theta, X_train, y_train, Xval, yval):
    lambda_values = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];
    val_err = []
    for lambda_ in lambda_values:
        theta = trainLinearReg(X_train, y_train, lambda_)
        val_err.append(linearRegCostFunction(theta, Xval, yval, 0))
    plt.plot(lambda_values, val_err, c='b')
    plt.axis([0, len(lambda_values), 0, val_err[-1] + 1])
    plt.xlabel("Lambda Values")
    plt.ylabel("Validation Error")
    plt.show()
    
if __name__ == "__main__":
    dataset = loadmat("ex5data1.mat")
    X_train, y_train = dataset['X'], dataset['y'].squeeze()
    X_test, y_test = dataset['Xtest'], dataset['ytest'].squeeze()
    X_val, y_val = dataset['Xval'], dataset['yval'].squeeze()
#    theta = np.ones((1, 1))
#    displayData(X_train, y_train)
#    X_train = insertOnes(X_train)
#    theta = trainLinearReg(X_train, y_train, 0)
#    plt.plot(X_train[:, 1:], X_train.dot(theta))
    
    
#    X_train = insertOnes(X_train)
#    X_val = insertOnes(X_val)
#    learningCurve(X_train, y_train, X_val, y_val, 0)
    
    X_train_poly = poly_features(X_train, 8)
    X_test_poly = poly_features(X_test, 8)
    X_val_poly = poly_features(X_val, 8)
    
    X_train_poly_means = X_train_poly.mean(axis=0)
    X_train_poly_std = X_train_poly.std(axis=0, ddof=1)
    
    X_train_poly = (X_train_poly - X_train_poly_means) / X_train_poly_std
    X_test_poly = (X_test_poly - X_train_poly_means) / X_train_poly_std
    X_val_poly = (X_val_poly - X_train_poly_means) / X_train_poly_std
    
    X_train_poly = insertOnes(X_train_poly)
    X_test_poly = insertOnes(X_test_poly)
    X_val_poly = insertOnes(X_val_poly)
    
    theta = trainLinearReg(X_train_poly, y_train, 0)
    plt.scatter(X_train, y_train, marker='x', s=40, c='r')
#    poly_fit(np.min(X_train), np.max(X_train), X_train_poly_means, X_train_poly_std, theta, 8)
#    learningCurve(X_train_poly, y_train, X_val_poly, y_val, 1)
    opt_lambda(theta, X_train_poly, y_train, X_val_poly, y_val)
