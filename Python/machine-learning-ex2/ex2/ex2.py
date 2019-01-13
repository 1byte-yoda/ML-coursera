import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def pred(X, theta):
    z = np.dot(X, theta)
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(X)
    hypothesis = pred(X, theta)
    class_1 = np.sum(np.dot(np.transpose(y) , np.log(hypothesis)))
    class_0 = np.sum(np.dot(1-np.transpose(y) , np.log(1-hypothesis)))
    J_theta = class_1 + class_0
    J_theta = J_theta/-m
    return J_theta

def update(X, y, theta, alpha):
    m = len(X)
    hypothesis = pred(X, theta)
    gradient = np.dot(np.transpose(X), (hypothesis-y))
    gradient = ((1/m) * gradient) * alpha
    theta = theta - gradient
    return theta

def train(X, y, theta, aplha, iterations):
    cost_history = []
    for i in range(iterations):
        temp_theta = update(X, y, theta, alpha)
        cost = cost_function(X, y, theta)
        if i % 1000== 0:
#            print("Iterations: {0}, Cost: {1}".format(i, cost))
            cost_history.append(cost)
            print("%.8f" % cost)
        if np.abs(np.sum(temp_theta - theta)) < 0.001:
            print("%.8f" % cost)
            print("Gradient Descent has converged!!")
            break
        theta = temp_theta
        
    return theta, cost_history

def predict(y_pred):
    prediction = []
    for i,p in enumerate(y_pred):
        if p >= 0.5:
            prediction.append(1)  
        else:
             prediction.append(0)  
    return prediction
def plot(X, y, res=0):
#    data = np.loadtxt('ex2data1.txt', delimiter=',')
    X = X[:, [1,2]]
    y = y[:, 0]
    
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='k', label = 'Admited')
    plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='y', label = 'Not Admited')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(frameon= True, fancybox = True)
    plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
#
    x1_min, x1_max = X[:,0].min(), X[:,0].max(),
    x2_min, x2_max = X[:,1].min(), X[:,1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = pred( np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()],(res))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b');
    plt.show()
if __name__ == '__main__':
    data = pd.read_csv("ex2data1.txt", header=None)
    #print(data)
    X = data.iloc[:, [0,1]]
    y = data.iloc[:, 2:].values
    X = np.append(np.ones((100, 1)), X, axis = 1)
    
    theta = np.array([[0],[0],[0]])
    iterations = 3000000
    alpha = 0.1
    
    opt_theta = train(X, y, theta, alpha, iterations)
##    print(cost_function(X, y, theta))
##    print(update(X, y, theta, alpha))
    y_pred = np.dot(X, opt_theta[0])
    y_pred = np.array(predict(y_pred))
    print(np.mean(y_pred == y[:, 0]) * 100)
    cm = confusion_matrix(y, y_pred)
#    plot(X, y, np.array([-25.1613,
#                         0.206232,
#                         0.201471,
#                         ]))
    plot(X, y, opt_theta[0])
    