import numpy as np

# Computes the cost for linear regression
# computes the cost of using theta as the
# parameter for linear regression to fit the data points in X and y
def computeCost(X, y, theta):
    J = 0
    # number of training examples
    m = len(y)

    J= (1/(2*m))*np.sum(np.power((X.dot(theta)-y), 2), axis=0)

    return J


# Performs gradient descent to learn theta
# updates theta by taking num_iters gradient steps with learning rate alpha
def gradientDescent(X, y, theta, alpha, num_iters):
    # Determine X size
    m = np.size(X,0)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha/m)*(X.T.dot(X.dot(theta)-y))
        J_history[i]=computeCost(X, y, theta)

    return theta, J_history

# Normalize rescales the features to a range of [0,1]
# MinMaxScaler processing from sklearn.preprocessing should do the same effect
def normalizeData(X):
    xmin = np.min(X, 0)
    xmax = np.max(X, 0)
    X = (X - xmin)/(xmax - xmin)
    return X


# Standardize rescales the features to a range of [0,1]
# MinMaxScaler processing from sklearn.preprocessing should do the same effect
def standardizeData(X):
    xmean = np.mean(X, 0)
    xstd = np.std(X, 0)
    X = (X - xmean)/xstd
    return X
