# Importing the needed libraries
import numpy as np
from logisticRegression import costFunction, sigmoid

# Optimization module in scipy
from scipy import optimize


# oneVsAll trains multiple logistic regression classifiers and returns all
# the classifiers in a matrix all_theta, where the i-th row of all_theta
# corresponds to the classifier for label i
#   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
#   logistic regression classifiers and returns each of these classifiers
#   in a matrix all_theta, where the i-th row of all_theta corresponds
#   to the classifier for label i
def oneVsAll(X, y, num_labels, lmbd):


    # m = number of training examples
    # n = number of features
    (m, n) = X.shape

    # Add a ones column to X
    X = np.append(np.ones((m, 1)), X, axis=1)

    initial_theta = np.zeros((n + 1, 1))
    all_theta = np.zeros((num_labels, n + 1))
    all_cost = np.array([])

    # set options for optimize.minimize
    options = {'maxiter': 400}

    for i in range(num_labels):
        yi = (y == i)
        res = optimize.minimize(costFunction,
                        initial_theta.flatten(),
                        (X, yi.flatten(), lmbd),
                        jac=True,
                        method='TNC',
                        options=options)

        all_cost = np.append(all_cost, res.fun)

        # the optimized theta is in the x property
        all_theta[i, :] = res.x

    return all_theta, all_cost

# Predict the label for a trained one-vs-all classifier. The labels
# are in the range 1..K, where K = all_theta.shape[0].
# It will return a vector of predictions
# for each example in the matrix X. Note that X contains the examples in
# rows. all_theta is a matrix where the i-th row is a trained logistic
# regression theta vector for the i-th class. You should set p to a vector
# of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
# for 4 examples)
def predictOneVsAll(all_theta, X):
    num_labels = all_theta.shape[0]
    (m, n) = X.shape
    p = np.zeros((m, 1))
    pTemp = np.zeros((m, num_labels))

    # Add a ones column to X
    X = np.append(np.ones((m, 1)), X, axis=1)

    pTemp = sigmoid(np.dot(X, all_theta.T))
    p = np.argmax(pTemp, axis=1)

    return p
