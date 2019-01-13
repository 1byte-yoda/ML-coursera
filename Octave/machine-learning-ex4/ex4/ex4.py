import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import fmin_cg
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

def displayData(rows, cols, X):
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10,5))
    
    for i in range(rows):
        for j in range(cols):
            rand_index = np.random.randint(1, X.shape[0])
            ax[i, j].imshow(X[rand_index].reshape((20,20), order='F'), cmap='gray')
            ax[i, j].axis("off")
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoidGrad(z):
    return np.multiply(sigmoid(z), (1-sigmoid(z)))

def randInit_Theta(nrows, ncols):
    epsilon = 0.12
    return np.random.uniform(size=(nrows, ncols+1)) * 2 * epsilon - epsilon

def nnCostFunction(nn_unrolled_theta, X, y, lambda_, input_layer_size,
                   hidden_layer_size, output_layer_size):
    Theta1 = nn_unrolled_theta[:hidden_layer_size*(input_layer_size+1)]\
                              .reshape((hidden_layer_size, (input_layer_size+1)), order='F')
                              
    Theta2 = nn_unrolled_theta[hidden_layer_size*(input_layer_size+1):]\
                              .reshape((output_layer_size, (hidden_layer_size+1)), order='F')
                              
    m = len(y)
    ones = np.ones((m, 1))                
    a1 = np.hstack((ones, X))
    z2 = a1.dot(Theta1.T)
    a2 = np.hstack((ones, sigmoid(z2)))
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)

    y_vec = pd.get_dummies(y.flatten())
    class_0 = np.multiply(y_vec, np.log(a3))
    class_1 = np.multiply((1-y_vec), np.log(1-a3))
    cost = np.sum(np.sum(class_0 + class_1 )/-m)
    sum_theta1 = np.sum(np.sum(np.square(Theta1[:, 1:]), axis = 1))
    sum_theta2 = np.sum(np.sum(np.square(Theta2[:, 1:]), axis = 1))
    reg_cost = lambda_/(2*m) * (sum_theta1+sum_theta2)
    cost = cost + reg_cost
    
    return cost

def nnGradient(nn_unrolled_theta, X, y, lambda_, input_layer_size,
                   hidden_layer_size, output_layer_size):
    Theta1 = nn_unrolled_theta[:hidden_layer_size*(input_layer_size+1)]\
                              .reshape((hidden_layer_size, (input_layer_size+1)), order='F')
                              
    Theta2 = nn_unrolled_theta[hidden_layer_size*(input_layer_size+1):]\
                              .reshape((output_layer_size, (hidden_layer_size+1)), order='F')
    m = len(y)
    y_vec = pd.get_dummies(y.flatten())
    delta1 = np.zeros(Theta1.shape)
    delta2 = np.zeros(Theta2.shape)
    
    for i in range(m):
        ones = np.ones(1)
        a1 = np.hstack((ones, X[i].T))
        z2 = Theta1.dot(a1)
        a2 = np.hstack((ones, sigmoid(z2)))
        z3 = Theta2.dot(a2)
        a3 = sigmoid(z3)
        
        d3 = a3 - np.array([y_vec.loc[i]])
        z2 = np.hstack((ones, z2))
        d2 = np.multiply(np.dot(Theta2.T, d3.T), np.array([sigmoidGrad(z2)]).T)
        
        delta1 = delta1 + np.dot(d2[1:, :], np.array([a1]))
        delta2 = delta2 + np.dot(d3.T, np.array([a2]))
    delta1 /= m
    delta2 /= m
    
    delta1[:, 1:] = delta1[:, 1:] + (Theta1[:, 1:] * (lambda_/m))
    delta2[:, 1:] = delta2[:, 1:] + (Theta2[:, 1:] * (lambda_/m))
    
    return np.hstack((delta1.ravel('F'), delta2.ravel('F')))

def checkGradient(nn_unrolled_theta, X, y, lambda_, input_layer_size,
                  hidden_layer_size, output_layer_size, nnbackpropgrad):
    epsilon = 10**-4
    n_elems = len(nn_unrolled_theta)
    for i in range(10):
        rand_index = np.random.randint(n_elems)
        epsilon_vector = np.zeros((n_elems, 1))
        epsilon_vector[rand_index] = epsilon
        
        left_hand = nnCostFunction(nn_unrolled_theta+epsilon_vector.flatten(), X, y, lambda_,
                                   input_layer_size, hidden_layer_size, output_layer_size)
        right_hand = nnCostFunction(nn_unrolled_theta-epsilon_vector.flatten(), X, y, lambda_, 
                                    input_layer_size, hidden_layer_size, output_layer_size)
        numerical_grad = (left_hand - right_hand) / (2*epsilon)
        
        print("Element {0}, Numerical Grad: {1:.9f}, BackPropagation Grad: {2:.9f}".format(rand_index, numerical_grad, 
              nnbackpropgrad[rand_index]))

def predict(X, y, theta1, theta2):
    ones = np.ones((len(y), 1))
    a1 = np.hstack((ones, X))
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack((ones, sigmoid(z2)))
    z3 = np.dot(a2,theta2.T)
    a3 = sigmoid(z3)
    
    return np.argmax(a3, axis=1) + 1
def getDatumImg(row):
    """
    Function that is handed a single np array with shape 1x400,
    crates an image object from it, and returns it
    """
    width, height = 20, 20
    square = row[1:].reshape(width,height)
    return square.T

def showNumbers(theta):
    fig, ax = plt.subplots(5, 5, figsize=(5,5))
    
    for i in range(5):
        for j in range(5):
            ax[i,j].imshow(theta)


    
if __name__ == '__main__':
    dataset = loadmat("ex4data1.mat")
    X = dataset['X']
    y = dataset['y']
    weights = loadmat("ex4weights.mat")
    theta1 = weights['Theta1']
    theta2 = weights['Theta2']
#    displayData(10, 10, X)
    unrolled_theta = np.hstack((theta1.ravel(order='F'), theta2.ravel(order='F')))
    input_layer_size = theta1.shape[1] - 1
    hidden_layer_size = theta2.shape[1] - 1
    output_layer_size = np.unique(y).size
#    nnbackpropgrad = nnGradient(unrolled_theta, X, y, 1, input_layer_size,
#                   hidden_layer_size, output_layer_size)
#    checkGradient(unrolled_theta, X, y, 1, input_layer_size,
#                  hidden_layer_size, output_layer_size, nnbackpropgrad)
#    
    rand_theta1 = randInit_Theta(hidden_layer_size, input_layer_size)
    rand_theta2 = randInit_Theta(output_layer_size, hidden_layer_size)
    unrolled_rand_theta = np.hstack((rand_theta1.ravel('F'), rand_theta2.ravel('F')))
    
    #Change lambda_ value into 10 to regularized
    opt_theta = fmin_cg(f = nnCostFunction, x0 = unrolled_rand_theta, args = (X, y, 1, input_layer_size,
                       hidden_layer_size, output_layer_size), fprime = nnGradient, maxiter = 50)
#    
    opt_theta1 = opt_theta[:hidden_layer_size*(input_layer_size+1)].reshape((hidden_layer_size,
                           (input_layer_size+1)), order='F')
    
    opt_theta2 = opt_theta[hidden_layer_size*(input_layer_size+1):].reshape((output_layer_size,
                           (hidden_layer_size+1)), order='F')
    
    y_pred = predict(X, y, opt_theta1, opt_theta2)
    
    print("Accuracy: {0:.2f}".format(np.mean(y_pred==y.flatten()) * 100))
#    t1, t2 = nnCostFunction(unrolled_theta, X, y, 1, input_layer_size,
#                   hidden_layer_size, output_layer_size)
    displayData(5,5, opt_theta1[:, 1:])

    