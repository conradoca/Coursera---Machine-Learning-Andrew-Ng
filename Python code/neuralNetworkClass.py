import numpy as np
# Optimization module in scipy
from scipy import optimize


class NNClassifier():
    # Parameters:
    # nnDef: NN definition. Array that contains the nodes per layer
    # epochs: number of iterations in the training set to minimize the cost function
    # lmbd: lambda value used for regularization (default is no regularization)
    def __init__(self, nnDef, epochs=50, lmbd=0):
        self.nnDef = nnDef
        self.epochs = epochs
        self.lmbd = lmbd

    # Randomly initialize the weights of a layer with L_in
    # incoming connections and L_out outgoing connections
    @staticmethod
    def _randInitializeWeights(L_in, L_out):
        # We are adding 1 row to take in consideration the bias unit
        W = np.zeros((L_out, 1 + L_in))

        # The random function provides random values uniformly distributed between [0,1)
        # We'll multiply by a very small value (epsilon) to make sure the output is made
        # of very small values
        eps = 0.12

        # We get a testCases random datapoints. First we get a number of testCases indices out of m
        W = np.random.rand(L_out, 1 + L_in)*(2*eps)-eps
        return W

    # Randomly initialize the Theta array of arrays
    # Remember that to maintain coherence with the notation we start
    # with Theta[1], which represents the input  to first hidden layer
    def _initializeTheta(self, nnDef):

        Theta = np.zeros(nnDef.size, dtype=np.ndarray)

        for i in range(1, nnDef.shape[0]):
            Theta[i] = self._randInitializeWeights(nnDef[i-1], nnDef[i])
        return Theta

    @staticmethod
    def _sigmoid(x):
        z = np.exp(-x)
        g = (1/(1+z))
        return g

    # Function that converts the Theta array of arrays
    # in a 1D vector. This is required to be sent to
    # the minimize sCiPy function
    @staticmethod
    def _ThetaTo1D(Theta, nnDef):
        # Useful variables
        num_layers = nnDef.size            # Number of layers including the Input Layer

        Theta1D = Theta[1].reshape(-1)
        for i in range(2, num_layers):
            Theta1D = np.concatenate((Theta1D, Theta[i].reshape(-1)), axis=None)

        return Theta1D

    # Function that converts back Theta from a 1D vector
    # into an array of arrays
    @staticmethod
    def _ThetaFrom1D(Theta1D, nnDef):
        # Useful variables
        num_layers = nnDef.size            # Number of layers including the Input Layer

        Theta = np.zeros(num_layers, dtype=np.ndarray)
        start = 0
        for i in range(1, num_layers):
            Theta[i] = Theta1D[start:(start + (nnDef[i] * (nnDef[i-1]+1)))].reshape(nnDef[i], nnDef[i-1]+1)
            start = start + (nnDef[i] * (nnDef[i-1]+1)+1) - 1
        return Theta

    def _feedForwardPropagation(self, Theta, X, nnDef):
        # Useful variables
        (m, n) = X.shape    # m = number of training examples, n = number of features
        num_layers = nnDef.size            # Number of layers including the Input Layer
        # layerActiv= Array containing the activation arrays
        layerActiv = np.zeros((nnDef.size+1,), dtype=np.ndarray)

        layerActiv[1] = X             # The activation for the Input layer is X

        for i in range(1, num_layers):
            # Add the bias unit to the activation
            mLayer = layerActiv[i].shape[0]
            layerActiv[i] = np.append(np.ones((mLayer, 1)), layerActiv[i], axis=1)
            layerActiv[i+1] = self._sigmoid(np.dot(layerActiv[i], Theta[i].T))

        return layerActiv

    # NNCOSTFUNCTION Implements the neural network cost function for the
    # neural network which performs classification
    # J, grad = nnCostFunction(Theta, nnDef, X, y, lmbd)
    # computes the cost and gradient of the neural network
    # We are passing the Theta values as a single vector
    # because the minimize function only works with 1-D vectors.
    # That means that the function must reconstruct Theta as a first step
    def _nnCostFunction(self, Theta1D, nnDef, X, y, lmbd):
        # Useful variables
        (m, n) = X.shape                   # m = number of training examples, n = number of features
        num_labels = nnDef[-1]             # Output Layer units
        num_layers = nnDef.size            # Number of layers including the Input Layer
        Theta = self._ThetaFrom1D(Theta1D, nnDef)

        # FeedForward Propagation
        # a array containing the activation arrays
        # (using numbering from 1 to be coerent with notation)
        a = np.zeros((nnDef.size+1,), dtype=np.ndarray)

        a = self._feedForwardPropagation(Theta, X, self.nnDef)

        # Extending the y vector into an array where 1 representents the label
        y10 = np.zeros((m, num_labels))
        y = y[:, np.newaxis]
        for i in range(num_labels):
            y10[:, i][:, np.newaxis] = np.where(y == i, 1, 0)

        # Cost Function
        J = (-1/m)*np.sum((np.multiply(np.log(a[num_layers]), y10) + np.multiply((1 - y10), np.log(1-a[num_layers]))))
        # Cost adding regularization
        for i in range(1, num_layers):
            J = J + (lmbd/(2*m))*(np.sum((np.power(Theta[i][:, 1:], 2))))

        # Getting the gradient
        grad = np.zeros((Theta.shape), dtype=np.ndarray)
        delta = np.zeros((nnDef.size+1,), dtype=np.ndarray)

        delta[num_layers] = (a[num_layers] - y10)
        for i in reversed(range(2, num_layers)):
            delta[i] = (np.dot(delta[i+1], Theta[i]))*(a[i]*(1-a[i]))
            delta[i] = delta[i][:, 1:]

        # Regularization part of the gradient
        for i in reversed(range(1, num_layers)):
            grad[i] = ((1/m)*np.dot(delta[i+1].T, a[i])) + ((lmbd/m)*np.hstack((np.zeros((Theta[i].shape[0], 1)), Theta[i][:,1:])))

        grad1D = self._ThetaTo1D(grad, nnDef)

        return J, grad1D

    def fit(self, X_train, y_train):
        # Learn weights from the training data
        # X_train: input layer with original features
        # y_train: target class labels
        #
        # Returns: self

        # Weights
        # We create a Theta array of arrays
        Theta = np.zeros(self.nnDef.size, dtype=np.ndarray)
        # We initialize Theta with random values
        Theta = self._initializeTheta(self.nnDef)
        # We convert Theta into a 1D vector
        Theta1D = self._ThetaTo1D(Theta, self.nnDef)

        # Minimizing the cost function

        # set options for optimize.minimize
        options = {'maxiter': self.epochs}

        # The function returns an object `OptimizeResult`
        # We use truncated Newton algorithm for optimization which is
        # equivalent to MATLAB's fminunc
        # See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
        res = optimize.minimize(self._nnCostFunction,
                                Theta1D,
                                (self.nnDef, X_train, y_train.flatten(), self.lmbd),
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

        Theta = self._ThetaFrom1D(self.theta, self.nnDef)
        a = self._feedForwardPropagation(Theta, X, self.nnDef)
        y_pred = np.argmax(a[-1], axis=1)

        return y_pred
