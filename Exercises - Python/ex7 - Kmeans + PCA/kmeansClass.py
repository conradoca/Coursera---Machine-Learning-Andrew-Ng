import numpy as np


class Kmeans():

    def __init__(self, K, epochs=10):
        self.epochs = epochs
        self.km = K
        self.centroids = None
        self.idCentroids = None

    # km -> number of classes/centroids
    # n -> The number of attributes define the dimension of these centroids
    # To init Knn we pick up K random training points and we set them as centroids
    @staticmethod
    def _initCentroids(X, km):

        (m, n) = X.shape
        centroids = np.zeros((km, n))
        centroids = X[np.random.randint(m, size=km)]
        return centroids

    @staticmethod
    def _findClosestCentroids(X, centroids):
        m = X.shape[0]
        idCentroids = np.zeros((m))

        for i in range(m):
            dist = np.sum((np.square(X[i, :]-centroids)), axis=1)
            idCentroids[i] = np.argmin(dist)

        return idCentroids

    @staticmethod
    def _computeCentroid(X, idCentroids, km):

        n = X.shape[1]
        centroids = np.zeros((km, n))

        for i in range(km):
            centroids[i] = np.mean(X[idCentroids == i], axis=0)

        return centroids

    def fit(self, X):

        (m, n) = X.shape

        self.centroids = np.zeros((self.epochs+1, self.km, n))
        self.centroids[0] = self._initCentroids(X, self.km)

        for i in range(self.epochs):
            self.idCentroids = self._findClosestCentroids(X, self.centroids[i])
            self.centroids[i+1] = self._computeCentroid(X, self.idCentroids, self.km)

        return self
