#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 17:21:44 2019

@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Prediction Function Setup
"""

# Import libraries
import math
import numpy as np

from Planar_Monocular_SLAM_transition import transition_model




###############################################################################

#                           PREDICTION SETUP

###############################################################################


# This function implements the kalman prediction step of the SLAM system
# Inputs: 
#   control input: previous and current odometry poses in the form:
#                  [  [x, y, theta], [x', y', theta']  ]
#  considering as 'm' the number of seen landmarks
#  mu_state: the previous mean of (x,y,theta, l1, l2, ..., lm), i.e. estimated robot pose
#      and the m landmark positions
#  sigma_state: the covariance of the previously estimated robot pose and landmark 
#         positions ((3+m)x(3+m) matrix)

# outputs 
# [mu_state, sigma_state] are mean and covariance of the estimate after transition


def prediction_model(mu_state, sigma_state, control_input):

    #domain spaces
    dimension_mu = mu_state.shape
    dimension_u  = 3
    
    # Control input is u_t = [x_t-1   x_t]
    theta = control_input[0][2]
    
    # Predict the robot motion
    # The transition model only affects the robot pose not the landmarks
    mu_r, delta_rot1, delta_trans, delta_rot2 = transition_model(mu_state[0:3], control_input)
    # Update the robot state
    mu_state[0:3] = mu_r
   
    # Jacobian A
    # Initialize A as an identity and fill only the robot block
    A = np.identity(dimension_mu[0])
    A[0:3,0:3] = np.matrix([[1, 0, -delta_trans*math.sin(theta + delta_rot1)],
                             [0, 1, delta_trans*math.cos(theta + delta_rot1)],
                             [0, 0, 1]])
    
    # Jacobian B
    # For each state variable we have to associate the available control inputs
    # Since the input u = [x_t-1   x_t] is now in the form delta_rot1, delta_trans, 
    # delta_rot2, the derivative will be based on delta_trans, delta_rot2
    B = np.zeros((dimension_mu[0], dimension_u))
    B[0:3,:] = np.matrix([
                [-delta_trans*math.sin(theta + delta_rot1), math.cos(theta + delta_rot1), 0],
                [delta_trans*math.cos(theta + delta_rot1), math.sin(theta + delta_rot1), 0],
                [1,                                              0,                      1]])
            
    # Control noise u
    sigma_u = 0.001           # constant part
    sigma_R1 = abs(delta_rot1) + abs(delta_trans)      #rotational velocity2 dependent part
    sigma_T = abs(delta_trans) + abs(delta_rot1 + delta_rot2)    #translational velocity dependent part
    sigma_R2 = abs(delta_rot2) + abs(delta_trans)      #rotational velocity2 dependent part
    
    #compose control noise covariance sigma_u
    sigma_u = np.matrix([
            [sigma_u+sigma_R1,                0,                            0], 
            [0,                          sigma_u+sigma_T,                   0],
            [0,                              0,                sigma_u+sigma_R2]  ])
    
    #predict sigma
    sigma_state = A*sigma_state*(np.transpose(A)) + B*sigma_u* (np.transpose(B))
    
    return([mu_state, sigma_state])


###############################################################################

#                          END PREDICTION SETUP

###############################################################################





