"""
File containg the Multidimensional scaling algorithm 
"""

__author__ = 'Daniel Elisabethsønn Antonsen'

# Importing libraries and modules
import numpy as np
import matplotlib.pyplot as plt

class Multidimensional_scaling:

    def __init__(self, B:np.ndarray, labels:np.ndarray):
        """
        Args:
            B: np.ndarray, the distance matrix
            labels: np.ndarray, the labels for each city
        """
        self.__B = B
        self.__labels = labels
    
    def __eigenvectors(self):
        """
        Method for computing the eigenvectors and eigenvalues 

        Output:
            eigenvalues: np.ndarray, array containing eigenvalues
            eigenvectors: np.ndarray, array containing corrresponding eigenvectors
        """
        # Computing eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(self.__B)
        return eigenvalues, eigenvectors
    
    def __coordinates(self, eigenvectors:np.ndarray, eigenvalues:np.ndarray, dim:int):
        """
        Method for computing the new coordinates for the points
        
        Args: 
            eigenvalues: np.ndarray, array containing the eigenvalues for matrix B
            eigenvectors: np.ndarray, array containing the corresponding eigenvectors for matrix B
            dim: int, spesifying the dimensions
        Output:
            np.ndarray, array containing the new coordinates
        """
        return np.sqrt(eigenvalues[:dim]) * eigenvectors[:, :dim] 


    def _plot(self, coordinates:np.ndarray, labels:np.ndarray):
        """
        Method for plotting the points
        
        Args:
            coordinates: np.ndarray, array containing the new points
            labels: np.ndarray, array containing the labels for for each cities
        """
        plt.scatter(coordinates[:, 1], coordinates[:, 0])
        for i, txt in enumerate(labels[:, 0]):
            plt.annotate(txt, (coordinates[i, 1] + 3, coordinates[i, 0] + 3))
        plt.show()

    def run(self):
        """
        Method for running the MDS algorithm
        """
        __eigenvalues , __eigenvectors = self.__eigenvectors()
        __coordinates = self.__coordinates(__eigenvectors, __eigenvalues, 2)
        self._plot(__coordinates, self.__labels)


