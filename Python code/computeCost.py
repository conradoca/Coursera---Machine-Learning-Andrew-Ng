# Computes the cost for linear regression
# computes the cost of using theta as the
# parameter for linear regression to fit the data points in X and y
import numpy as np

def computeCost(X, y, theta):
    J = 0
    # number of training examples
    m = len(y)

    J= (1/(2*m))*np.sum(np.power((X.dot(theta)-y), 2), axis=0)

    return J
