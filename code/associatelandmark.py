
"""
@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Associate Landmark IDs
"""

# Import libraries
import numpy as np
import os

# Set working directory
directory = os.getcwd()

# Set directory with dataset
dataset_dir = os.path.join(directory, "dataset")


# This function gets the landmark index
#inputs:
#  observations:
#            a structure containing n observations of landmarks
#            for each observation we have
#            - the appearance of each landmark observed
#outputs:
#  index of landmark
#  [observations]: updated observation info with correct landmark id

def associateLandmarkIDs(world_data, observations):

    # Get how many landmarks have been seen in current step
    M = len(observations)

    # First of all we are interested in REOBSERVED landmark
    for i in range(M):

        # Get the appearance given by known info
        l_appear = world_data[:,4:14]

        # Difference between the known and measured appearances
        delta = l_appear - observations[i][2][:]
        # Get the error of the Euclidean distance of each option
        error = np.linalg.norm(delta, ord=2, axis=1)
        # Get the index of the smallest error
        land_index = np.argmin(error)

        # Test: to compare that the landmark id is correct
        # measurement = observations[i]
        # id = int(measurement[0][1]) # current landmark number
        # if land_index == id:
        #     print('Same landmark!')
        # else: print('NO!')

        # Update the observation info with id info to be used later
        observations[i][0] = np.append(observations[i][0],land_index)
        # print('updated observations \n', observations[i][0])

    return (observations)


###############################################################################

#                          END ASSOCIATE LANDMARK SETUP

###############################################################################
