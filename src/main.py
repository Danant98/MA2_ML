"""
Main file for the course FYS-2021 Machine Learning, UiT The Arctic University.
"""

__author__ = 'Daniel Elisabethsønn Antonsen' 

# Importing modules and libraries
import numpy as np
import os 
from MDS import Multidimensional_scaling
import pandas as pd

## Problem 1
# Opening data containing the B-matrix
data_inner_sweden = np.loadtxt(os.path.join("resources", "city-inner-sweden.csv"))
names_inner_sweden = pd.read_csv(os.path.join("resources", "city-names-sweden.csv"))
print(names_inner_sweden)
exit()
# names_inner_sweden = names_inner_sweden[names_inner_sweden.argsort()]


coordinates = Multidimensional_scaling(data_inner_sweden, names_inner_sweden).run()
dist_umeå = coordinates[-1] - coordinates[np.where(names_inner_sweden == "Umeå")]
print(dist_umeå@dist_umeå.T)




if __name__ == "__main__":
    pass   

