import numpy as np

# Optimization module in scipy
from scipy import optimize


class linearRegression():
    # Parameters
    # epochs: number of iterations in the training set to minimize the cost function
    # lmbd: lambda value used for regularization (default is no regularization)
    def __init__(self, epochs=50, lmbd=0):
        self.epochs = epochs
        self.lmbd = lmbd

    @staticmethod
    def _initializeTheta(m):

        # We are adding 1 row to take in consideration the bias unit
        W = np.zeros((1, m))

        # The random function provides random values uniformly distributed between [0,1)
        # We'll multiply by a very small value (epsilon) to make sure the output is made
        # of very small values
        eps = 0.12

        # We get a testCases random datapoints. First we get a number of testCases indices out of m
        W = np.random.rand(1, m)*(2*eps)-eps
        return W

    def _CostFunction(self, theta, X, y, lmbd):
        J = 0
        # m = number of training examples, n = number of features
        (m, n) = X.shape

        J = (1/(2*m))*np.sum(np.power((X.dot(theta)-y), 2), axis=0) + ((lmbd/(2*m))*np.sum((np.power(theta[1:], 2))))

        grad = (1/m)*(X.T.dot(X.dot(theta)-y))
        grad[1:] = grad[1:] + ((lmbd/m)*theta[1:])

        return J, grad

    def fit(self, X_train, y_train):
        # Learn weights from the training data
        # X_train: input layer with original features
        # y_train: target class labels
        #
        # Returns: self

        (m, n) = X_train.shape

        # Weights
        # We create a Theta array of arrays
        Theta = np.zeros((1, n))
        # We initialize Theta with random values
        Theta = self._initializeTheta(n)

        # Minimizing the cost function

        # set options for optimize.minimize
        options = {'maxiter': self.epochs}

        # The function returns an object `OptimizeResult`
        # We use truncated Newton algorithm for optimization which is
        # equivalent to MATLAB's fminunc
        # See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
        res = optimize.minimize(self._CostFunction,
                                Theta,
                                (X_train, y_train.flatten(), self.lmbd),
                                jac=True,
                                method='TNC',
                                options=options)

        # the fun property of `OptimizeResult` object returns
        # the value of costFunction at optimized theta
        self.cost = res.fun

        # the optimized theta is in the x property
        self.theta = res.x

        return self

    def predict(self, X):
        # X: input layer with original features
        # Returns
        # y_pred: predicted class labels

        y_pred = X.dot(self.theta)
        return y_pred

    def score(self, X, y):
        u = np.power(y - self.predict(X), 2).sum()
        v = np.power(y - np.mean(y), 2).sum()
        scr = 1 - u/v
        return scr
