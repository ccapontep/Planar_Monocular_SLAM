#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 17:21:44 2019

@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Correction Function Setup
"""

# Import libraries
import numpy as np
import math
import os

from Planar_Monocular_SLAM_data import 
from Planar_Monocular_SLAM_functools import box_plus

########################## ERASE LATER 
# Set working directory
directory = os.getcwd()

# Set directory with dataset
dataset_dir = os.path.join(directory, "dataset")


###############################################################################

#                           CORRECTION SETUP

###############################################################################


# This function computes the correction step of the filter
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

def correction(mu, sigma, observations, id_to_state_map, state_to_id_map):
    
    # Get how many landmarks have been seen in current step
    M = len(observations)
    
    # If seen no landmarks, nothing is done
    if M == 0: pass

    # Dimension of the entire state (robot pose + landmark positions)
    dimension_state = len(mu)
    mu_t            = mu[0:2] # translational part of the robot pose
    mu_theta        = mu[2] # rotation of the robot
    
    # Precomputed variables
    c   = math.cos(mu_theta)
    s   = math.sin(mu_theta)
    Rt  = np.matrix([[c,s], [-s, c]])    #transposed rotation matrix
    Rtp = np.matrix([[-s,c], [-c,-s]]) #derivative of transposed rotation matrix
    
    # Count for amount of observations of old landmarks known
    number_known_landmarks = 0

    # Here two cases arise, the current landmark has been already seen, i.e. REOBSERVED landmark,
    # or the current landmark is completely new, i.e. NEW landmark.
    #
    # analyze only the reobserved landmark
    # and work with them as in a localization procedure (of course, with full Jacobian now).
    # With this reobserved landmark we compute the correction/update of mean and covariance.
    # Then, after the correction is done, we can simply add the new landmark expanding the
    # mean and the covariance.
    #
    # First of all we are interested in REOBSERVED landmark 
    for i in range(M):
  
        # Get info about each observed landmark
        measurement = observations[i]
#        print('mu', mu.shape)
        # Get the position in the state vector corresponding to the actual measurement
        n = id_to_state_map[measurement[0][1], 0]

        # Temp for correct location of landmark from world_data
        world_data, _, _ = get_all_data(dataset_dir)
        l_loc = world_data[measurement[0][1]][1:4]
        #l_appear = world_data[n][4:14]

    #IF current landmark is a REOBSERVED LANDMARK
        if n != -1:

            # Compute index in the state vector corresponding to the pose of the landmark	
            id_state = 3+2*(n-1)
        
            # initialize data structures
            z_out = np.zeros((2, 1))
            h_pred = np.zeros((2, 1))

            # Increment the counter of observations from already known landmarks
            number_known_landmarks += 1
            # Add landmark measurement -- add to end of array two rows, one for x and y
            z_out = np.array([[l_loc[0]], [l_loc[1]]]) ######## change both later 

            # Get the position (x,y) of the landmark in the state
            landmark_mu = mu[id_state:id_state+2, :]

            # Prediction of where that landmark would be seen
            delta_t            = landmark_mu - mu_t
            measure_prediction = Rt * delta_t

            # Add landmark measurement prediction
            h_pred = measure_prediction
            
            # Jacobian piece w.r.t. robot
            C_m          = np.zeros((2, dimension_state))
            C_m[0:2,0:2] = -Rt
            C_m[0:2,2:3]   = Rtp*delta_t

            #jacobian piece w.r.t. landmark
            C_m[:,id_state:id_state+2] = Rt


            #add jacobian piece to main jacobian
            if number_known_landmarks == 1: # To initialize the matrices
                C_t = C_m 
                z_t = z_out
                h_t = h_pred
            else: 
                C_t = np.vstack([C_t, C_m])
                z_t = np.vstack([z_t, z_out])
                h_t = np.vstack([h_t, h_pred])
            
#            print('h_all', h_all.shape)
    #if I have seen again at least one landmark
    #I need to update, otherwise I jump to the new landmark case
    if (number_known_landmarks > 0):
      
        #observation noise
        noise   = 0.001
        sigma_z = np.identity(2*number_known_landmarks)*noise

        #Kalman gain
        K = sigma * np.transpose(C_t)*(np.linalg.inv(C_t*sigma*np.transpose(C_t) + sigma_z))

        #update mu
        innovation = z_t - h_t
        mu         = mu + K*innovation        
        
        #update sigma
        sigma = (np.eye(dimension_state) - K*C_t)*sigma	
      
    return (mu, sigma, id_to_state_map, state_to_id_map)


###############################################################################

#                          END CORRECTION SETUP

###############################################################################




