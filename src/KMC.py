"""
File containg the k-means clustering algorithm
"""

__author__ = 'Daniel Elisabethsønn Antonsen'

# Importing libraries and modules
import numpy as np

## OBS! THIS CODE HAS BEEN WRITTEN USING GITHUB CO-PILOT (SKRIV OM FØR INNLEVERING!)
class K_means:

    def __init__(self, X:np.ndarray, k:int):
        """
        Args:
            X: np.ndarray, the data matrix
            k: int, the number of clusters
        """
        self.__X = X
        self.__k = k
        self.__N = X.shape[0]
        self.__d = X.shape[1]
        self.__centroids = np.random.rand(k, self.__d)
        self.__clusters = np.zeros(self.__N)
        self.__distances = np.zeros((self.__N, self.__k))
        self.__new_centroids = np.zeros((self.__k, self.__d))
        self.__old_centroids = np.zeros((self.__k, self.__d))
        self.__converged = False

    def __distance(self, X:np.ndarray, centroids:np.ndarray):
        """
        Method for computing the distance between the data points and the centroids

        Args:
            X: np.ndarray, the data matrix
            centroids: np.ndarray, the centroids

        Output:
            np.ndarray, array containing the distances
        """
        return np.sqrt(np.sum((X - centroids[:, np.newaxis])**2, axis=2))

    def __update_centroids(self):
        """
        Method for updating the centroids
        """
        for i in range(self.__k):
            self.__new_centroids[i] = np.mean(self.__X[self.__clusters == i], axis=0)

    def __update_clusters(self):
        """
        Method for updating the clusters
        """
        self.__clusters = np.argmin(self.__distances, axis=1)

    def __convergence(self):
        """
        Method for checking if the algorithm has converged
        """
        if np.all(self.__new_centroids == self.__old_centroids):
            self.__converged = True

    def __update(self):
        """
        Method for updating the centroids and clusters
        """
        self.__update_centroids()
        self.__update_clusters()
        self.__convergence()

    def __assign_clusters(self):
        """
        Method for assigning the clusters
        """
        self.__clusters = np.argmin(self.__distances, axis=1)

    def __assign_centroids(self):
        """
        Method for assigning the centroids
        """
        self.__centroids = self.__X[np.random.choice(self.__N, self.__k, replace=False)]

    def __assign(self):
        """
        Method for assigning the clusters and centroids
        """
        self.__assign_clusters()
        self.__assign_centroids()

    def __compute(self):
        """
        Method for computing the distances between the data points and the centroids
        """
        self.__distances = self.__distance(self.__X, self.__centroids)
    
    def run(self):
        """
        Method for running the k-means clustering algorithm
        """
        self.__assign()
        while not self.__converged:
            self.__compute()
            self.__update()
        return self.__clusters, self.__centroids
