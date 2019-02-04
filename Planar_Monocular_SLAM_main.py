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
mu = np.array([0, 0, 0], dtype='float64')
# Initilize sigma
sigma = np.identity(3, dtype='float64')
mu_sum = np.array([0, 0, 0], dtype='float64')

# Main simulation cycle
for t in range(336):
    print('t = ', t)
    # Get the data from the correct sequence
#    _, _, _, traj_odo_prev, traj_odo_curr, _, _ = get_new_seqdata(dataset_dir, t)
    meas_gt_curr, meas_odo_curr, meas_lpoint, control_input, traj_gt_prev, traj_gt_curr= get_new_seqdata(dataset_dir, t)
    
    
    # EKF predict
    mu, sigma = prediction_model(mu, sigma, control_input)
    
    # Sort the eigenvalue by highest to lowest value to draw correctly
#    https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
#    https://www.soest.hawaii.edu/martel/Courses/GG303/Lab.09.2017.pptx.pdf -- slide 20
    '''
        def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        vals_sor, vecs_sor = np.linalg.eigh(cov)
        vals_sor, vecs_sor = vals_sor[0:2], vecs_sor[0:2, 0:2]
        order = vals.argsort()[::-1]
        vals_sor, vecs_sor = vals[order], vecs[:, order] 
        return vals, vecs, vals_sor, vecs_sor
    
    eigenvalues, vecs, vals_sor, vecs_sor = eigsorted(sigma)
    print('eigenvalues, vecs, vals_sor, vecs_sor', eigenvalues, vecs, vals_sor, vecs_sor)
#   eigen function returns vectors as: column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    theta = np.degrees(np.arctan2(vecs_sor[1,0], vecs_sor[0,0])) # angle direction of error
    '''
    
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        vals_sor, vecs_sor = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals_sor, vecs_sor = vals[order], vecs[:, order] 
        return vals, vecs, vals_sor, vecs_sor
    
    eigenvalues, vecs, vals_sor, vecs_sor = eigsorted(sigma[0:2, 0:2])
    print('eigenvalues, vecs, vals_sor, vecs_sor', eigenvalues, vecs, vals_sor, vecs_sor)
#   eigen function returns vectors as: column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    theta = np.degrees(np.arctan2(vecs_sor[1,0], vecs_sor[0,0])) # angle direction of error
#   To find the error of mu to draw ellipse by: 2 * nstd * np.sqrt(vals)
#   where nstd is the std deviation, in this case 1 std
    eigen_error = np.array([2*math.sqrt(eigenvalues[0]), 2*math.sqrt(eigenvalues[1]), 
                      theta])
#    math.sqrt(abs(mu[2]*180/math.pi))
    print('mu', mu)
    print('sigma', sigma)
    print('error along mu', eigen_error)
    
##    Obtain current observation using the data association
#    observations_t = associateLandmarkIDs(mu, sigma, observations(t), state_to_id_map);
#
##    EKF correct
#    mu, sigma, id_to_state_map, state_to_id_map = correction(mu, sigma, observations_t, id_to_state_map, state_to_id_map);
#
##    ADD new landmarks to the state
#    mu, sigma, id_to_state_map, state_to_id_map, last_landmark_id = addNewLandmarks(mu, sigma, observations_t, id_to_state_map, state_to_id_map, last_landmark_id);

#    mu_sum = np.vstack([mu_sum, mu])
#    print('mu_sum', mu_sum)
    # Plot landmarks, robot trajectory, error in each
    plt.scatter(world_data[:,1], world_data[:,2], color='red', marker = 'o', s =1)
    plt.scatter(mu[0], mu[1], color='blue', marker = 'o', s=1, zorder=1)
    plt.scatter(traj_gt_curr[0], traj_gt_curr[1], color='green', marker = 'o', s=1, zorder=1) # traj ground truth
    plt.plot(mu[0], mu[1], linestyle='-', linewidth= 1, zorder=2)
    ellipse = Ellipse(mu, eigen_error[0], eigen_error[1], angle = theta,
                      edgecolor = 'b', facecolor = 'none', lw=1)
    plt.axes().add_artist(ellipse)
#    plt.scatter(mu_sum[:,0], mu_sum[:,1], color='blue', marker = 'o', s=1, zorder=1)
#    plt.plot(mu_sum[:,0], mu_sum[:,1], linestyle='-', linewidth= 1, zorder=2)
#    plt.draw()
    plt.show()
    plt.pause(0.0001)
    ellipse.remove() # remove previous ellipse so a new one can be drawm with new movement
    
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





