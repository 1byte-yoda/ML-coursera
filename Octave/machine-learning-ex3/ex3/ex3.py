import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.io import loadmat


def load_data():
    data = loadmat("ex3data1.mat")
    weights = loadmat("ex3weights.mat")
    X = data["X"]
    y = data["y"]
    theta1, theta2 = weights["Theta1"], weights["Theta2"]
    return X, y, theta1, theta2

def visualize_data(X):
    _, axes = plt.subplots(10,10,figsize=(20,20))
    for i in range(10):
        for j in range(10):
           rand_index = np.random.randint(X.shape[0])
           axes[i][j].imshow(X[rand_index].reshape((20,20), order="F"))        
           axes[i,j].axis('off')  
def sigmoid(z):

    return 1/(1+np.exp(-z))

def lrcostFunctionReg(theta, reg, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    
    J = -1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + (reg/(2*m))*np.sum(np.square(theta[1:]))
    
    if np.isnan(J[0]):
        return(np.inf)
    return(J) 
    
def lrgradientReg(theta, reg, X,y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
      
    grad = (1/m)*X.T.dot(h-y) + (reg/m)* np.append([[0]], theta[1:].reshape(-1,1), axis=0)
        
    return(grad.flatten())
    
def oneVsAll(X, y, n_labels, reg):
    initial_theta = np.zeros((X.shape[1],1))  # 401x1
    all_theta = np.zeros((n_labels, X.shape[1])) #10x401

    for c in np.arange(1, n_labels+1):
        res = minimize(lrcostFunctionReg, initial_theta, args=(reg, X, (y == c).astype(int)), method=None,
                       jac=lrgradientReg, options={'maxiter':50})
        all_theta[c-1] = res.x
    return(all_theta)        
def predictOneVsAll(all_theta, features):
    probs = sigmoid(X.dot(all_theta.T))
        
    # Adding one because Python uses zero based indexing for the 10 columns (0-9),
    # while the 10 classes are numbered from 1 to 10.
    return(np.argmax(probs, axis=1)+1)

if __name__ == "__main__":
    X, y, theta1, theta2 = load_data()
#    visualize_data(X, pred[random_number])
#    X = np.c_[np.ones((X.shape[0],1)), X]
    
    X = np.append(np.ones((X.shape[0],1)), X, axis=1)
#    theta = oneVsAll(X, y, 10, 0.1)
#    pred = predictOneVsAll(theta, X)
#    print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel())*100))
    theta = np.zeros((X.shape[1], 1))
    x=lrgradientReg(theta, 0, X, y)
    
#    visualize_data(X[:, 1:])
#    fig, axes = plt.subplots(2, 5, figsize=(10, 2))
#    fig.suptitle('Handwriting Digit Recognizer (Accuracy: {0:.2f}% @ 500 epochs)'.format(np.mean(pred == y.ravel())*100))
#    for i in range(2):
#        for j in range(5):
#            random_number = np.random.randint(1, X.shape[0])
#            axes[i,j].imshow(X[random_number, 1:].reshape((20,20), order="F"))
#            if pred[random_number] == 10:
#                pred[random_number] = 0
#            axes[i,j].set_title("Output:%d" % pred[random_number])
#            axes[i,j].axis("off")
        

    
        