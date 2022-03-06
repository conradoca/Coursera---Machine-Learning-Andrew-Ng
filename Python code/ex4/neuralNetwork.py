import numpy as np


# Computing the sigmoid function
# It only works with x being an array.
# you can convert any value or Python's list to an array using numpy.array(list)
def sigmoid(x):
    z = np.exp(-x)
    g = (1/(1+z))
    return g


# Compute cost and gradient for logistic regression
# using theta as the parameter for regularized logistic regression
# the regularization is defaulted to 0.
# Lambda is a reserved word in Python. We use lmbd instead
def costFunction(theta, X, y, lmbd=0):
    # number of training examples
    m = np.size(X, 0)
    # n = np.size(X,1)
    J = 0
    grad = np.zeros(np.size(theta, 0))

    J = (-1/m)*(y.T.dot(np.log(sigmoid(X.dot(theta))))+((1-y).T.dot(np.log(1-sigmoid(X.dot(theta)))))) + ((lmbd/(2*m))*np.sum((np.power(theta[1:], 2))))

    grad = (1/m)*(X.T.dot(sigmoid(X.dot(theta))-y))
    grad[1:] = grad[1:] + ((lmbd/m)*theta[1:])

    return J, grad


# Feedforward propagation algorithm
# Generic function that can work with any shape of NN
# Theta = an array that contains the Theta[i] arrays for the various layers i
# nnDef = NN definition. Vector that defines the number of nodes per each layer
# Predict the label of an input given a trained neural network
# p outputs the predicted label of X given the trained weights of a neural network (Theta)
# Pred outputs the array with the probabilities for being each value
def feedForwardPropagation(Theta, X, nnDef):
    # Useful variables
    (m, n) = X.shape    # m = number of training examples, n = number of features
    num_labels = nnDef[-1]             # Output Layer units
    num_layers = nnDef.size            # Number of layers including the Input Layer
    p = np.zeros((1, num_labels))       # Classification vector
    Prob = np.zeros((m, num_labels))    # Matrix contaning all the probabilities per label
    # layerActiv= Array containing the activation arrays
    layerActiv = np.zeros((nnDef.size,), dtype=np.ndarray)

    layerActiv[0] = X             # The activation for the Input layer is X

    for i in range(num_layers-1):
        # Add the bias unit to the activation
        mLayer = layerActiv[i].shape[0]
        layerActiv[i] = np.append(np.ones((mLayer, 1)), layerActiv[i], axis=1)
        layerActiv[i+1] = sigmoid(np.dot(layerActiv[i], Theta[i+1].T))

    Prob = layerActiv[-1]
    p = np.argmax(layerActiv[-1], axis=1)

    return Prob, p


# Randomly initialize the weights of a layer with L_in
# incoming connections and L_out outgoing connections
def randInitializeWeights(L_in, L_out):
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
def initializeTheta(nnDef):

    Theta = np.zeros(nnDef.size, dtype=np.ndarray)

    for i in range(1, nnDef.shape[0]):
        Theta[i] = randInitializeWeights(nnDef[i-1], nnDef[i])
    return Theta


# Function that converts the Theta array of arrays
# in a 1D vector. This is required to be sent to
# the minimize sCiPy function
def ThetaTo1D(Theta, nnDef):
    # Useful variables
    num_layers = nnDef.size            # Number of layers including the Input Layer

    Theta1D = Theta[1].reshape(-1)
    for i in range(2, num_layers):
        Theta1D = np.concatenate((Theta1D, Theta[i].reshape(-1)), axis=None)

    return Theta1D


# Function that converts back Theta from a 1D vector
# into an array of arrays
def ThetaFrom1D(Theta1D, nnDef):
    # Useful variables
    num_layers = nnDef.size            # Number of layers including the Input Layer

    Theta = np.zeros(nnDef.size, dtype=np.ndarray)
    start = 0
    for i in range(1, num_layers):
        Theta[i] = Theta1D[start:(start + (nnDef[i] * (nnDef[i-1]+1)))].reshape(nnDef[i], nnDef[i-1]+1)
        start = start + (nnDef[i] * (nnDef[i-1]+1)+1) - 1
    return Theta

# NNCOSTFUNCTION Implements the neural network cost function for the
# neural network which performs classification
# J, grad = nnCostFunction(Theta, nnDef, X, y, lmbd)
# computes the cost and gradient of the neural network
# We are passing the Theta values as a single vector
# because the minimize function only works with 1-D vectors.
# That means that the function must reconstruct Theta as a first step
def nnCostFunction(Theta1D, nnDef, X, y, lmbd):
    # Useful variables
    (m, n) = X.shape                   # m = number of training examples, n = number of features
    num_labels = nnDef[-1]             # Output Layer units
    num_layers = nnDef.size            # Number of layers including the Input Layer
    Theta = ThetaFrom1D(Theta1D, nnDef)

    # a array containing the activation arrays
    # (using numbering from 1 to be coerent with notation)
    a = np.zeros((nnDef.size+1,), dtype=np.ndarray)

    a[1] = X                # The activation for the Input layer is X

    # Extending the y vector into an array where 1 representents the label
    y10 = np.zeros((m, num_labels))
    y = y[:, np.newaxis]
    for i in range(num_labels):
        y10[:, i][:, np.newaxis] = np.where(y == i, 1, 0)

    # Forward Propagation
    for i in range(1, num_layers):
        # Add the bias unit to the a layer
        mLayer = a[i].shape[0]
        a[i] = np.append(np.ones((mLayer, 1)), a[i], axis=1)
        a[i+1] = sigmoid(np.dot(a[i], Theta[i].T))

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

    grad1D = ThetaTo1D(grad, nnDef)

    return J, grad1D
