"""
File containg the k-means clustering algorithm
"""

__author__ = 'Daniel Elisabethsønn Antonsen'

# Importing libraries and modules
import numpy as np
import matplotlib.pyplot as plt

class k_mean:

    def __init__(self, X:np.ndarray, k:int):
        """
        Args: 
            X: np.ndarray, matrix containg unlabeld dataset
            k: int, number of clusters
        """
        self.__X = X
        self._k = k
        self._centroids = X[np.random.choice(X.shape[0], k, replace=False)]
        self._labels = np.zeros(X.shape[0])
        self._distance = np.zeros((X.shape[0], k))

    def _comp_distance(self):
        """
        Method for computing the Euclidean distance matrix
        """
        for i in range(self._k):
            self._distance[:, i] = np.linalg.norm(self.__X - self._centroids[i, :], axis=1)
    
    def _assign(self):
        """
        Method for assigning points to clusters
        """
        self._labels = np.argmin(self._distance, axis=1)

    def _update_centroids(self):
        """
        Method for updating the centroids
        """
        for i in range(self._k):
            self._centroids[i] = np.mean(self.__X[self._labels == i], axis=0)

    def run(self):
        """
        Method for running the algorithm
        """
        while True:
            self._comp_distance()
            self._assign()
            self._update_centroids()
            old_distance = self._distance
            if np.all(np.linalg.norm(self._distance - old_distance)) < 1E-5:
                break
        return self._labels, self._centroids, np.argsort(self._labels)

    def plot(self, imgs:np.ndarray, centroids:np.ndarray):
        """
        Method for plotting the clusters
        
        Args:
            img: np.ndarray, array containing the images to be plotted
            centroids: np.ndarray, array containing the centroids
        """
        fig, ax = plt.subplots(self._k, imgs.shape[1], figsize=(12, 12), tight_layout=True \
                  , subplot_kw={'xticks': [], 'yticks': []}, sharex=True, sharey=True)
        for j in range(self._k):
            im_centroid = centroids[j].reshape(28, 20)
            ax[j, 0].imshow(im_centroid, cmap='gray')
            for i in range(1, imgs.shape[1]):
                im = imgs[j, i].reshape(28, 20)
                ax[j, i].imshow(im, cmap='gray')
        fig.suptitle(f"K-means clustering with k = {str(self._k)}")



