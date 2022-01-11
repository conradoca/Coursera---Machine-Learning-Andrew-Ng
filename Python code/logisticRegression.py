import numpy as np

# Computing the sigmoid function
# It only works with x being an array.
# you can convert any value or Python's list to an array using numpy.array(list)
def sigmoid(x):
    z = np.exp(-x)
    g=(1/(1+z))
    return g


# Compute cost and gradient for logistic regression
# using theta as the parameter for regularized logistic regression
# the regularization is defaulted to 0.
# Lambda is a reserved word in Python. We use lmbd instead
def costFunction(theta, X, y, lmbd=0):
    # number of training examples
    m = np.size(X,0)
    # n = np.size(X,1)
    J=0
    grad = np.zeros(np.size(theta,0))

    J=(-1/m)*(y.T.dot(np.log(sigmoid(X.dot(theta))))+((1-y).T.dot(np.log(1-sigmoid(X.dot(theta)))))) + ((lmbd/(2*m))*np.sum((np.power(theta[1:],2))))

    grad = (1/m)*(X.T.dot(sigmoid(X.dot(theta))-y))
    grad[1:] = grad[1:] + ((lmbd/m)*theta[1:]);

    return J, grad


# Predict whether the label is 0 or 1 using learned logistic regression parameters theta
def predict(theta, X):
    pred= [sigmoid(X.dot(theta)) >= 0.5]
    return pred


# Feature mapping function to polynomial features
# maps the two input features to quadratic features used in the regularization exercise.

# Returns a new feature array with more features, comprising of X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
# Inputs X1, X2 must be the same size
# Degree determines the highest degree of combination of the two variables
def mapFeature(X1, X2, degree=1):
    out= np.ones(X1.shape[0])[:,np.newaxis]
    for i in range(1, degree+1):
        for j in range(0, i+1):
            out = np.hstack((out, (np.multiply(np.power(X1, i-j), np.power(X2, j))[:,np.newaxis])))
    return out


def mapFeatureForPlotting(X1, X2, degree=1):
    out= np.ones(1)
    for i in range(1, degree+1):
        for j in range(0, i+1):
            out = np.hstack((out, (np.multiply(np.power(X1, i-j), np.power(X2, j)))))
    return out
