# Performs gradient descent to learn theta
# updates theta by taking num_iters gradient steps with learning rate alpha
import numpy as np
from computeCost import *

def gradientDescent(X, y, theta, alpha, num_iters):
    # Determine X size
    m = np.size(X,0)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha/m)*(X.T.dot(X.dot(theta)-y))
        J_history[i]=computeCost(X, y, theta)

    return theta, J_history
