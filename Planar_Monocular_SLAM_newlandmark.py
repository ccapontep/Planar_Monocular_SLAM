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
from mpmath import mp
mp.dps = 6 # precision for rounding the trig functions
mp.pretty = True

from Planar_Monocular_SLAM_data import get_all_data
#from Planar_Monocular_SLAM_functools import v2t
from Planar_Monocular_SLAM_uv_to_xy import uv_to_xy
from Planar_Monocular_SLAM_CamL_CoorL import ray_reverse


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

def newlandmark(t, mu, sigma, observations, id_to_state_map, state_to_id_map, robot_pose_map, robot_gt_pose_map):
    
    _, trajectory_data, camera_data = get_all_data(dataset_dir)
    
    # Get how many landmarks have been seen in current step
    M = len(observations)

    # To check that the id_to_state_map is being set correctly (not
    # repeating the same id number)
    from collections import Counter
    a = Counter(list(id_to_state_map[:,0]))
    b = Counter(list(id_to_state_map[:,4]))

    # Current number of landmarks in the state
    # if the first time, start at n = 0, else check the id map
    if t == 0:
        n = int((len(mu)-3)/3) -1
    else: n = max(list(a.items()))[0]

    # Current robot pose
#    mu_t            = mu[0:2] # translational part of the robot pose
#    mu_theta        = mu[2] # rotation of the robot
    
    # check with the correct robot trajectory:
    mu_t = robot_gt_pose_map[t, 0:2].reshape(2,1)
    mu_t = np.insert(mu_t, 2, 0, axis = 0)
    mu_theta = robot_gt_pose_map[t, 2]
    
    # Precomputed variables
    c   = mp.cos(mu_theta)
    s   = mp.sin(mu_theta)
    R   = np.matrix([[c, -s, 0], [s, c, 0], [0,0,1]], dtype = 'float')  #rotation matrix
    

    # First of all we are interested in REOBSERVED landmark 
    for i in range(M):
  
        # Get info about each observed landmark
        measurement = observations[i]
        meas_land = int(measurement[0][1]) # current landmark number
        

        q = int((len(mu)-3)/3) -1
        
        #fetch the value in the state vector corresponding to the actual measurement
        state_pos_landmark_1 = id_to_state_map[meas_land, 0]
        state_pos_landmark_2 = id_to_state_map[meas_land, 4]
    
        # Temp for correct location of landmark from world_data
        world_data, _, camera_data = get_all_data(dataset_dir)
        l_loc = world_data[meas_land][1:4]
        landmark_loc = np.array([l_loc[0], l_loc[1], l_loc[2]])
        #l_appear = world_data[n][4:14]       
        
        # Get the measurement col, row of each landmark (u, v, 1)
        u = measurement[1][0]
        v = measurement[1][1] 


### For the landmark seen for the first time ###########
        
        # If current landmark is observed by the first time
        if state_pos_landmark_1 == -1 and state_pos_landmark_2 == -1:
        
            # Get new values for landmark id and set them to the mappings of id and state
            n += 1
            id_to_state_map[meas_land, 0] = n


            # If it is the first time seeing this n landmark:
            id_to_state_map[meas_land, 1] = t # add the time step of when it was seen
            id_to_state_map[meas_land, 2] = u # add the corresponding u
            id_to_state_map[meas_land, 3] = v # add the corresponding v

        # If haven't seen a landmark and creating enough paralax, will erase
        # all data so the landmark will be initiated for the first time again
        elif id_to_state_map[meas_land, 11] >= 4:
            id_to_state_map[meas_land, :] = -1
#            print('************')
#            print('Removed all data for landmark ', meas_land, 'since it did not create enough paralax')
#            print('************')
        
############# DELAYED INITIALIZATION DUE TO BEARING-ONLY ######################
# This section is for the collection of information before a landmark is 
# initialized, since one angle is not enough to describe the position of the 
# landmark. Therefore we collect two bearing measurements to calculate the 
# landmark location and initialize it in the state vector.
# http://www-personal.acfr.usyd.edu.au/tbailey/papers/icra03.pdf
# For three landmarks before initialization:
# http://mars.cs.umn.edu/tr/reports/Trawny05a.pdf

        
        # Else, if it is the second time observing the same landmark: 
        elif state_pos_landmark_2 == -1 and state_pos_landmark_1 != -1:  
            

            id_to_state_map[meas_land, 4] = q
#            print('t', t)
            # Convert uv coordinate to xy
            landmark_pos_rob, id_to_state_map, alpha, beta, b, row, theta, Rq_curr, Rq_der, h_c_reverse, obs_landmark_w = uv_to_xy(t, u, v, meas_land, id_to_state_map, robot_pose_map)

# &&&&&&&&&&&&&&&&&&&& CHANGE GT TO ROBOT_POSE_MAP!!!!!!!!!

            # If alpha is less than 4 degrees, erase the data of observation.
            # Will have to wait to the next time to reobserve the landmark in 
            # order to be initialized.
            # Beta also has to be more than 20 degrees, so that if it is less, 
            # the landmark is located in front of the direction of the camera and 
            # will be discarded for beta < 0.34906
            # Also need a minimum translation of the camera is set to have about 
            # 6 degrees of paralax. After 28 time steps the average was taken to 
            # be b_min of 0.68454 and not too far since the system cannot make a 
            # good estimate when b > 2.5
            # https://core.ac.uk/download/pdf/41760856.pdf
            
                
                    
            # Landmark with has to have a baseline distance 'b' that is not too far
            # and not located in front of the direction of the camera movement 'beta'
            if alpha <= 0.1 or beta <= 0.35 or b >= 2 : #     or b <= 1.2 
                # Erase data of this information, so it can get new ones until initiation.
                id_to_state_map[meas_land, 4:11] = -1
                id_to_state_map[meas_land, 11] += 1
                

            
            # If alpha is greater than 5 degrees, initiate the landmark.
            elif alpha > 0.1 and beta > 0.35 and b < 2 : #    and b > 1.2 
                # Add the landmark to the state map
                q += 1
                state_to_id_map[q, 0] = meas_land 
                
                # To find the minimum base-line b for paralax about 6 deg
#                if alpha > 0.105 and alpha < 0.122:
##                    print('************')
#                    bb.append(b)
#                    print('bb', bb)

                # Get landmark x, y position in the world
#                landmark_pos_robot = np.array([[landmark_pos_rob[0,0]], [landmark_pos_rob[1,0]], landmark_pos_rob[2,0]])  
                landmark_pos_world = mu_t + (R*landmark_pos_rob)
#                landmark_pos_world = landmark_pos_rob
#                h_c_reverse = ray_reverse(landmark_pos_world, mu_t, Rq_curr)
                
#                landmark_pos_world = np.insert(landmark_pos_world, 2, landmark_pos_rob[2,0], axis = 0)

#                print('Alpha is higher than 5 degrees, enough to initiate this landmark.')
#                print('Alpha: ', alpha)
#                print('Beta: ', beta)
#                print('Landmark initiated: ', meas_land) 
#                print('Depth of landmark (row): ', row)
#                print('Baseline distance', b)
#                print('landmark_pos_rob \n', landmark_pos_rob)
##                print('mu_t \n', mu_t)
#                print('landmark_pos_world \n', landmark_pos_world)
#                print('gt landmark_loc \n', landmark_loc)
#                print('obs_landmark_w \n', obs_landmark_w)

#                if beta < 0.34906:
#                    print('Beta is smaller than 20 degrees, landmark is not located in front of the direction of the camera movement.')
#                    print('Beta: ', beta)
    
                # Add the reverce of the ray to later use for prediction
#                id_to_state_map[meas_land, 8] = Rq_curr
                
#                id_to_state_map[meas_land, 8] = h_c_reverse[0,0]
#                id_to_state_map[meas_land, 9] = h_c_reverse[1,0]
#                id_to_state_map[meas_land, 10] = h_c_reverse[2,0]

                
                #retrieve from the index the position of the landmark block in the state
                g = id_to_state_map[meas_land, 4] # get the index in state
                id_state = int(3+2*g)
    
                #adding the landmark state to the full state
                mu = np.vstack([mu, landmark_pos_world])
    
    
                #initial noise assigned to a new landmark
                #for simplicity we put a high value only in the diagonal.
                #A more deeper analysis on the initial noise should be made.
                initial_landmark_noise = 2
                sigma_landmark         = np.eye(3)*initial_landmark_noise
    
                # Dimension of the entire state (robot pose + landmark positions)
                dimension_state = len(mu) 
    
                #adding the landmark covariance to the full covariance
                sigma = np.vstack([sigma, np.zeros((3, dimension_state - 3))])
                sigma = np.hstack([sigma, np.zeros((dimension_state, 3))])
    
                #set the covariance block
                sigma[id_state:id_state+3, id_state:id_state+3] = sigma_landmark
                print('---------------------------')

#               print("observed new landmark with identifier: ", meas_land)
    return (mu, sigma, id_to_state_map, state_to_id_map)


###############################################################################

#                          END CORRECTION SETUP

###############################################################################





