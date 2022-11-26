"""
File containg the Multidimensional scaling algorithm 
"""

__author__ = 'Daniel Elisabethsønn Antonsen'

# Importing libraries and modules
import numpy as np
import matplotlib.pyplot as plt

class Multidimensional_scaling:

    def __init__(self, B:np.ndarray):
        """
        Args:
            B: np.ndarray, the distance matrix
        """
        self.__B = B 
    
    def __eigenvectors(self):
        """
        Method for computing the eigenvectors and eigenvalues 

        Output:
            eigenvalues: np.ndarray, array containing eigenvalues
            eigenvectors: np.ndarray, array containing corrresponding eigenvectors
        """
        # Computing eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(self.__B)
        # Sort arrays in descending order
        index = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[index]
        eigenvectors = eigenvectors[:, index]

        return eigenvalues, eigenvectors
    
    def __coordinates(self, eigenvectors:np.ndarray, eigenvalues:np.ndarray):
        """
        Method for computing the new coordinates for the points
        
        Args: 
            eigenvalues: np.ndarray, array containing the eigenvalues for matrix B
            eigenvectors: np.ndarray, array containing the corresponding eigenvectors for matrix B
        """
        return np.sqrt(eigenvalues) * eigenvectors 


    def _plot(self, coordinates:np.ndarray):
        """
        Method for plotting the points
        
        Args:
            coordinates: np.ndarray, array containing the new points
        """
        plt.scatter(coordinates[:, 0], coordinates[:, 1], label=r"New coordinates")
        plt.show()

    def run(self):
        """
        Method for running the MDS algorithm
        """
        __eigenvalues , __eigenvectors = self.__eigenvectors()
        __coordinates = self.__coordinates(__eigenvectors, __eigenvalues)
        self._plot(__coordinates)



