#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 17:21:44 2019

@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Add New Landmark Function Setup
"""

# Import libraries
import numpy as np
import math
import os

from Planar_Monocular_SLAM_data import get_all_data

########################## ERASE LATER 
# Set working directory
directory = os.getcwd()

# Set directory with dataset
dataset_dir = os.path.join(directory, "dataset")


###############################################################################

#                           ADD NEW LANDMARK SETUP

###############################################################################


# This function computes gets the new landmark that has not been previously seen
#inputs:
#  mu: mean, 
#  sigma: covariance of the robot-landmark set (x,y, theta, l_1, ..., l_N)
#
#  observations:
#            a structure containing n observations of landmarks
#            for each observation we have
#            - the index of the landmark seen
#            - the location where we have seen the landmark (x,y) w.r.t the robot
#
#  id_to_state_map:
#            mapping that given the id of the measurement, returns its position in the mu vector
#  state_to_id_map:
#            mapping that given the index of mu vector, returns the id of the respective landmark
#
#outputs:
#  [mu, sigma]: the updated mean and covariance
#  [id_to_state_map, state_to_id_map]: the updated mapping vector between landmark position in mu vector and its id

def newlandmark(mu, sigma, observations, id_to_state_map, state_to_id_map):
    
    # Get how many landmarks have been seen in current step
    M = len(observations)
    #current number of landmarks in the state
    n = int((len(mu)-3)/2) -1

    # Dimension of the entire state (robot pose + landmark positions)
    mu_t            = mu[0:2] # translational part of the robot pose
    mu_theta        = mu[2] # rotation of the robot
    
    # Precomputed variables
    c   = math.cos(mu_theta)
    s   = math.sin(mu_theta)
    R   = np.matrix([[c, -s], [s, c]])  #rotation matrix
    

    # First of all we are interested in REOBSERVED landmark 
    for i in range(M):
  
        # Get info about each observed landmark
        measurement = observations[i]

        #fetch the position in the state vector corresponding to the actual measurement
        state_pos_landmark = id_to_state_map[measurement[0][1], 0]
#        print('state_pos_landmark', state_pos_landmark)
    
        # Temp for correct location of landmark from world_data
        world_data, _, _ = get_all_data(dataset_dir)
        l_loc = world_data[measurement[0][1]][1:4]
        #l_appear = world_data[n][4:14]

        # If current landmark is observed by the first time
        if state_pos_landmark == -1:
            
            # Get new values for landmark id and set them to the mappings of id and state
            n += 1
#            print('n', n)
#            print('measurement[0][1]', measurement[0][1])
#            print(' id_to_state_map[measurement[0][1]][0]',  id_to_state_map[measurement[0][1]][0])
            id_to_state_map[measurement[0][1], 0] = n
#            print(' id_to_state_map[measurement[0][1]][0]',  id_to_state_map[measurement[0][1]][0])
#            print('state_to_id_map[n, 0]', state_to_id_map[n, 0])
            state_to_id_map[n, 0] = measurement[0][1]
#            print('state_to_id_map[n, 0]', state_to_id_map[n, 0])

            #compute landmark position in the world
            landmark_pos_robot = np.array([[l_loc[0]], [l_loc[1]]])
            landmark_pos_world = mu_t + (R*landmark_pos_robot)


            #retrieve from the index the position of the landmark block in the state
            id_state = 3+2*(n-1)

            #adding the landmark state to the full state
            mu = np.vstack([mu, landmark_pos_world])
#            mu[id_state:id_state+1,1] = landmark_pos_world

            #initial noise assigned to a new landmark
            #for simplicity we put a high value only in the diagonal.
            #A more deeper analysis on the initial noise should be made.
            initial_landmark_noise = 0.001
            sigma_landmark         = np.eye(2)*initial_landmark_noise

            # Dimension of the entire state (robot pose + landmark positions)
            dimension_state = len(mu) 

            #adding the landmark covariance to the full covariance
            sigma = np.vstack([sigma, np.zeros((2, dimension_state - 2))])
            sigma = np.hstack([sigma, np.zeros((dimension_state, 2))])

            #set the covariance block
            sigma[id_state:id_state+2, id_state:id_state+2] = sigma_landmark

#            print("observed new landmark with identifier: ", measurement[0][1])
    return (mu, sigma, id_to_state_map, state_to_id_map)


###############################################################################

#                          END CORRECTION SETUP

###############################################################################





