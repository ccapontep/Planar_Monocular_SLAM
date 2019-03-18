#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 17:21:44 2019

@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Main Setup
"""

# Import libraries
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
#import matplotlib
#matplotlib.use("TkAgg") 
#from matplotlib import pyplot as plt
#import pylab
#ipython --pylab
#plt.ion()
#plt.switch_backend("TkAgg")
#plt.switch_backend("GTK3Agg")
plt.ion()
#plt.rcParams["figure.figsize"] = (15,10) # set the size of the figures (increase)

# Import files
from Planar_Monocular_SLAM_data import get_all_data, get_new_seqdata
from Planar_Monocular_SLAM_prediction import prediction_model
from Planar_Monocular_SLAM_error_ellipse import error_ellipse
from Planar_Monocular_SLAM_correction import correction
from Planar_Monocular_SLAM_newlandmark import newlandmark

# Set working directory
directory = os.getcwd()

# Set directory with dataset
dataset_dir = os.path.join(directory, "dataset")

## Get the data info
world_data, trajectory_data, camera_data = get_all_data(dataset_dir)


###############################################################################

#                           MAIN SETUP

###############################################################################


# Initialize variables
# Initial robot pose has already been set in Planar_Monocular_SLAM_data file
mu = np.array([[0], [0], [0]], dtype='float64')
# Initilize sigma
sigma = np.identity(3, dtype='float64')
mu_sum = np.array([[0], [0], [0]], dtype='float64')

id_to_state_map = np.ones((1000, 12), dtype='float64')*-1
state_to_id_map = np.ones((1000, 1), dtype='int32')*-1
# will retain the pose of the robot for each time sequence
robot_pose_map = np.zeros((336, 3)) 
robot_gt_pose_map = np.zeros((336, 3)) 

bbb = []

# Main simulation cycle
# for testing only do first 3 sequences, later return to 336
for t in range(336):
    print('t = ', t)
    
 
    # Get the data from the correct sequence
#    _, _, _, traj_odo_prev, traj_odo_curr, _, _ = get_new_seqdata(dataset_dir, t)
    meas_gt_curr, meas_odo_curr, meas_lpoint, control_input, robot_pose_map, robot_gt_pose_map = get_new_seqdata(dataset_dir, t, robot_pose_map, robot_gt_pose_map)
    
    
    # EKF predict
    mu, sigma = prediction_model(mu, sigma, control_input)
    
    # Get the error along the mean, to be used for the error 2D ellipse drawing
    eigen_error = error_ellipse(sigma)

    # Save current robot pose to the map for landmark initialization below
#    robot_pose_map[t,0] = mu[0]
#    robot_pose_map[t,1] = mu[1]
#    robot_pose_map[t,2] = mu[2]

#    print('mu', mu)
#    print('sigma', sigma)
#    print('error along mu', eigen_error)

    # bookkeeping: to and from mapping between robot pose (x,y, theta) and landmark indices (i)
    # all mappings are initialized with invalid value -1 (meaning that the index is not mapped)
    # 1000 landmarks are known to appear so this is the size of the bookkeeping matrix


##   Obtain current observation using the data association
#    observations_t = associateLandmarkIDs(mu, sigma, observations(t), state_to_id_map);
#
#    EKF correct
    # Test using the world map with the landmarks already known. later will replace
    # world_data for meas_XXXXX.
#    mu, sigma, id_to_state_map, state_to_id_map = correction(t, mu, sigma, meas_lpoint, id_to_state_map, state_to_id_map, robot_pose_map);
#    print('correction shape mu', mu.shape)

##    ADD new landmarks to the state
    mu, sigma, id_to_state_map, state_to_id_map = newlandmark(t, mu, sigma, meas_lpoint, id_to_state_map, state_to_id_map, robot_pose_map, robot_gt_pose_map) 
    print('add new landmark shape mu', mu.shape)
#    bbb.append(bb)
    # mean of the value of b over 28 time steps to get a paralax of 6 deg was 0.68454
#    print('************* bb *********** \n', bbb)
#    print('mu', mu[0,0], mu[1,0])
    

    # Separate landmark x and y for each
    l_x = np.array(mu[3:len(mu+1):3])
    l_y = np.array(mu[4:len(mu+1):3])

    keyboardClick=False
    while keyboardClick != True:
        keyboardClick=plt.waitforbuttonpress()
        state_items = np.array(list(set(list(state_to_id_map.flatten())))).size
        if state_items > 1:
            for k, val in enumerate(state_to_id_map[:,0]):
                if val != -1:
                    gt_l = plt.scatter(world_data[val,1], world_data[val,2], color='red', marker = '+', s =3)
                    ann_lgt = plt.annotate(val, (world_data[val,1], world_data[val,2]))
                    ann_lgt.set_fontsize(6)
                    ann_lgt.set_color('red')
                    ann_lpred = plt.annotate(val, (l_x[k,0], l_y[k,0]))
                    ann_lpred.set_fontsize(6)
                    ann_lpred.set_color('purple')
        # Predicted landmarks
        pred_l = plt.scatter(l_x[:,0], l_y[:,0], color='purple', marker = 'o', s=2)
        # Plot actual robot tranjectory
        plt.scatter(mu[0,0], mu[1,0], color='blue', marker = 'o', s=1, zorder=1)
        # Plot ground truth trajectory of robot
        plt.scatter(robot_gt_pose_map[t, 0], robot_gt_pose_map[t,1], color='green', marker = 'o', s=1, zorder=1) # traj ground truth
    #    plt.plot(mu[0, 0], mu[1, 0], linestyle='-', linewidth= 1, zorder=2)
        ellipse = Ellipse(mu, eigen_error[0], eigen_error[1], angle = eigen_error[2],
                          edgecolor = 'b', facecolor = 'none', lw=1)
        plt.axes().add_artist(ellipse)
        plt.scatter(mu_sum[:,0], mu_sum[:,1], color='blue', marker = 'o', s=1, zorder=1)
    #    plt.plot(mu_sum[:,0], mu_sum[:,1], linestyle='-', linewidth= 1, zorder=2)
    #    plt.draw()
        plt.show()
        plt.pause(0.0001)
        ellipse.remove() # remove previous ellipse so a new one can be drawm with new movement
        pred_l.remove()
        if state_items > 1: 
            gt_l.remove()    
            ann_lgt.remove()
            ann_lpred.remove()
    


    
"""
#--------------------------------- VISUALIZATION-------------------------------
    #display current state - transform data
    N = (rows(mu)-2)/2
    if N > 0:
        landmarks = landmark(state_to_id_map(1), [mu(4), mu(5)])
        for u  in range(2, N):
            landmarks(i+1) = landmark(state_to_id_map(u), [mu(3+2*u-1), mu(3+2*u)])
            
        print("current pose: [%f, %f, %f], map size (landmarks): %u\n", mu(1), mu(2), mu(3), N)
        
        trajectory = [trajectory; mu(1), mu(2)]
        plotState(landmarks, mu, sigma, observations_t, trajectory)

  pause(0.1)
  fflush(stdout)
  
#----------------------------- VISUALIZATION  ---------------------------------


# Test that the landmarks with their correct pose are drawm on the map

#Creates just a figure and only one subplot
#fig, aux = plt.subplots()

x_w=world_data[:,1]
y_w=world_data[:,2]


plt.scatter(x_w, y_w, color='red', marker = '*')
plt.scatter(x_t, y_t, color='blue', marker = 'o', s=1, zorder=1)
plt.plot(x_t, y_t, linestyle='-', linewidth= 1, zorder=2)
plt.show()







    #calculate the trajectory odometry pose
    # we precompute some quantities that come in handy later on
    cont_od_x = traj_curr[1]
    cont_od_y = traj_curr[2]
    cont_od_theta = traj_curr[3]
    c=cos(cont_od_theta)
    s=sin(cont_od_theta)
    R = [c, -s][s, c] # rotation matrix
    

  

# Start testing with only 3 instances of the data
world_data = world_data[0:3, :]
trajectory_data = trajectory_data[0:3,:]

"""





