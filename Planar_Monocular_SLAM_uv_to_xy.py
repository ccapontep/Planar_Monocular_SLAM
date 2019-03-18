#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 17:21:44 2019

@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Convert uv coordinate to xy Setup
"""

# Import libraries
import numpy as np

#from Planar_Monocular_SLAM_functools import v2t
from Planar_Monocular_SLAM_CamL_CoorL import beta_calc, gamma_calc, alpha_calc, landmark_pos, cam_coor


###############################################################################

#                           CONVERT UV TO XY SETUP

###############################################################################


# This function computes gets the new landmark that has not been previously seen
#inputs:
#  t, u, v: time, u camera coord, v camera coord
#  meas_land: landmark point number
#
#  id_to_state_map:
#            mapping that given the id of the measurement, returns its position in the mu vector
#  state_to_id_map:
#            mapping that given the index of mu vector, returns the id of the respective landmark
#
#outputs:
#  [mu, sigma]: the updated mean and covariance
#  [id_to_state_map, state_to_id_map]: the updated mapping vector between landmark position in mu vector and its id

def uv_to_xy(t, u, v, meas_land, id_to_state_map, robot_pose_map):
    

    id_to_state_map[meas_land, 5] = t # add new time step
    id_to_state_map[meas_land, 6] = u # add new u
    id_to_state_map[meas_land, 7] = v # add new v
    
      
    # For the first observation get info of robot location:
    rob_time_1 = int(id_to_state_map[meas_land, 1])
    # pose of robot when the landmark was first seen
    x1 = robot_pose_map[rob_time_1,0]
    y1 = robot_pose_map[rob_time_1,1]
    theta1 = robot_pose_map[rob_time_1,2]
    rob_pose1 = np.array([x1, y1, theta1]).reshape(3,1) 
            
    # Pose of robot during the second observation
    rob_time_2 = int(id_to_state_map[meas_land, 5])
    # pose of when the landmark was secondly seen
    x2 = robot_pose_map[rob_time_2,0]
    y2 = robot_pose_map[rob_time_2,1]
    theta2 = robot_pose_map[rob_time_2,2]
    rob_pose2 = np.array([x2, y2, theta2]).reshape(3,1)
    
    # Get the measurement col, row of each time the landmark is seen (u, v, 1)
    u_v1 = np.array([id_to_state_map[meas_land, 2], id_to_state_map[meas_land, 3], 1]).reshape(3,1)
    u_v2 = np.array([id_to_state_map[meas_land, 6], id_to_state_map[meas_land, 7], 1]).reshape(3,1)
    
    # Get current Camera position
    Cam_pose_1, Cam_pose_2, Cam_pose_2_der = cam_coor(rob_pose1, rob_pose2)
    # For beta, use the camera coordinates of the observed point the first time
    beta, b, Rq_1 = beta_calc(u_v1[0,0], u_v1[1,0], Cam_pose_1, Cam_pose_2)
    # For gamma, use the camera coordinates of the observed point the current time
    gamma, H_C_2, Rq_curr, Rq_der = gamma_calc(u_v2[0,0], u_v2[1,0], Cam_pose_1, Cam_pose_2, Cam_pose_2_der)
    alpha = alpha_calc(beta, gamma)
    
    from mpmath import mp
    mp.dps = 6 # precision for rounding the trig functions
    mp.pretty = True
    
    mu_t = rob_pose1[0:2,0].reshape(2,1)
    mu_t = np.insert(mu_t, 2, 0, axis = 0)
    mu_theta = rob_pose1[2,0]
    
    # Precomputed variables
    c   = mp.cos(mu_theta)
    s   = mp.sin(mu_theta)
    R   = np.matrix([[c, -s, 0], [s, c, 0], [0,0,1]], dtype = 'float')  #rotation matrix
    
    landmark_pos_rob, row, b, theta, Rq_curr, h_c_reverse, obs_landmark_w = landmark_pos(beta, alpha, gamma, H_C_2, b, Cam_pose_1, Cam_pose_2, Rq_curr, mu_t, R)

    return (landmark_pos_rob, id_to_state_map, alpha, beta, b, row, theta, Rq_curr, Rq_der, h_c_reverse, obs_landmark_w)


###############################################################################

#                          END CONVERT UV TO XY  SETUP

###############################################################################





