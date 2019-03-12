#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 17:21:44 2019

@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Transition Function Setup
"""

# Import libraries
import math
from mpmath import mp
mp.dps = 8 # precision for rounding the trig functions



###############################################################################

#                           TRANSITION SETUP

###############################################################################



# Transition function computes the transition of the robot after incorportating
# a movement traj_odo_prev (u in slides) = [traj_odo_prev_x, traj_odo_prev_y, traj_odo_prev_theta]
# f(x_t-1, u_t-1) = p(x_t | x_t-1, u_t-1)
#                 = A_t * x_t-1  + B_t * (u_t-1 + n_u)  where n_u is the control noise

# ODOMETRY MODEL
# Robot moves from (x, y, theta) to (x', y', theta') 
# Sequence is done as a rotation delta_rot1, then translation delta_trans, and
# a final rotation delta_rot2


# Odometry info: PREVIOUS/CURRENT
#   traj_odo_prev(or curr)[0]: movement on x
#   traj_odo_prev(or curr)[1]: movement on y 
#   traj_odo_prev(or curr)[2]: movement on theta


# Mean of current location of robot (mu in slides)
#   mu_pose_curr[0]: x coord of robot w.r.t world
#   mu_pose_curr[1]: y coord of robot w.r.t world
#   mu_pose_curr[2]: angle of robot w.r.t world
#

# Returns: Mean of location of robot after control input movement (mu_prime in slides)
#   mu_state_next[0]: x coord of robot w.r.t world, after transition
#   mu_state_next[1]: y coord of robot w.r.t world, after transition
#   mu_state_next[2]: angle of robot w.r.t world, after transition



def transition_model(mu_state_curr, control_input):
    
    # Where control_input = [  [traj_odo_prev] [traj_odo_prev]  ]
    
    ## Get the deltas for rotation and translation given the control info from
    # odometry as [  [x, y, theta] [x', y', theta']  ]
    
    #   Get the previous trajectory parameters
    x = control_input[0][0]
    y = control_input[0][1]
    theta = control_input[0][2]
#   Get the current trajectory parameters
    x_prime = control_input[1][0]
    y_prime = control_input[1][1]
    theta_prime = control_input[1][2]

#   Calculate the change in the parameters using the odometry model
    delta_rot1 = math.atan2(y_prime - y, x_prime - x) - theta
    delta_trans = math.sqrt((x - x_prime)**2 + (y-y_prime)**2)
    delta_rot2 = theta_prime - theta - delta_rot1
#   Set the current robot state as the next state (to save the landmark info)   
    mu_state_next = mu_state_curr
#   Get the x, y, theta compoments of the current state of robot
    mu_state_curr_x = mu_state_curr[0]
    mu_state_curr_y = mu_state_curr[1]   
    mu_state_curr_theta = mu_state_curr[2]
#   For ease of use
    c = mp.cos(theta + delta_rot1)
    s = mp.sin(theta + delta_rot1)

#    (x' ) = (    x + delta_trans * cos(th + delta_rot1)     )
#    (y' ) = (    y + delta_trans * sin(th + delta_rot1)     )
#    (th') = (    th + delta_rot1 + delta_rot2               )
    

    mu_state_next[0] = mu_state_curr_x + delta_trans * c
    mu_state_next[1] = mu_state_curr_y + delta_trans * s
    mu_state_next[2] = mu_state_curr_theta + delta_rot1 + delta_rot2 
    
    return(mu_state_next, delta_rot1, delta_trans, delta_rot2)


###############################################################################

#                          END TRANSITION SETUP

###############################################################################





