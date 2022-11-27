"""
Main file for the course FYS-2021 Machine Learning, UiT The Arctic University.
"""

__author__ = 'Daniel Elisabethsønn Antonsen' 

# Importing modules and libraries
import numpy as np
import os 
from MDS import Multidimensional_scaling
import pandas as pd
from KMC import k_mean
import matplotlib.pyplot as plt

## Problem 1
# Opening data containing the B-matrix
data_inner_sweden = np.loadtxt(os.path.join("resources", "city-inner-sweden.csv"))
names_inner_sweden = pd.read_csv(os.path.join("resources", "city-names-sweden.csv"), header=None).to_numpy()
faces_data = np.loadtxt(os.path.join("resources", "frey-faces.csv"))

# Running multidimensional scaling algorithm
#Multidimensional_scaling(data_inner_sweden, names_inner_sweden).run()

labels_2, centroids_2 = k_mean(faces_data, 2).run()

im = centroids_2[0, :]
im2 = centroids_2[1, :]
im = np.reshape(im, (28, 20))
im2 = np.reshape(im2, (28, 20))
plt.gray()
plt.figure()
plt.imshow(im)
plt.figure()
plt.imshow(im2)
plt.show()
if __name__ == "__main__":
    pass   

