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
# Multidimensional_scaling(data_inner_sweden, names_inner_sweden).run()

# Running k-mean clustering algorithm for k = 2
mean_2 = k_mean(faces_data, 2)
labels_2, centroids_2, sorted_2 = mean_2.run()
# Sorte the images and choose the once closest to the centroids
sorted_2_0 = faces_data[sorted_2[labels_2 == 0], :]
sorted_2_1 = faces_data[sorted_2[labels_2 == 1], :]
im_2 = np.array([sorted_2_0[:5, :], sorted_2_1[:5, :]])

# Plotting the images for k = 2
mean_2.plot(im_2, centroids_2)

# Running k-mean clustering for k = 4
mean_4 = k_mean(faces_data, 4)
labels_4, centroids_4, sorted_4 = mean_4.run()
# Sorte the images and choose the once closest to the centroids
sorted_4_0 = faces_data[sorted_4[labels_4 == 0], :]
sorted_4_1 = faces_data[sorted_4[labels_4 == 1], :]
sorted_4_2 = faces_data[sorted_4[labels_4 == 2], :]
sorted_4_3 = faces_data[sorted_4[labels_4 == 3], :]

# Plotting the images for k = 4
im_4 = np.array([sorted_4_0[:5, :], sorted_4_1[:5, :], sorted_4_2[:5, :], sorted_4_3[:5, :]])
mean_4.plot(im_4, centroids_4)

# Runnning k-mean clustering for k = 10
mean_10 = k_mean(faces_data, 10)
labels_10, centroids_10, sorted_10 = mean_10.run()

# Sorte the images and choose the once closest to the centroids
sorted_10_0 = faces_data[sorted_10[labels_10 == 0], :]
sorted_10_1 = faces_data[sorted_10[labels_10 == 1], :]
sorted_10_2 = faces_data[sorted_10[labels_10 == 2], :]
sorted_10_3 = faces_data[sorted_10[labels_10 == 3], :]
sorted_10_4 = faces_data[sorted_10[labels_10 == 4], :]
sorted_10_5 = faces_data[sorted_10[labels_10 == 5], :]
sorted_10_6 = faces_data[sorted_10[labels_10 == 6], :]
sorted_10_7 = faces_data[sorted_10[labels_10 == 7], :]
sorted_10_8 = faces_data[sorted_10[labels_10 == 8], :]
sorted_10_9 = faces_data[sorted_10[labels_10 == 9], :]

# Plotting the images for k = 10
im_10 = np.array([sorted_10_0[:5, :], sorted_10_1[:5, :], sorted_10_2[:5, :], sorted_10_3[:5, :],
                  sorted_10_4[:5, :], sorted_10_5[:5, :], sorted_10_6[:5, :], sorted_10_7[:5, :],
                  sorted_10_8[:5, :], sorted_10_9[:5, :]])
mean_10.plot(im_10, centroids_10)

if __name__ == "__main__":
    plt.show()   

