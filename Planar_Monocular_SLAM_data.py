#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 17:21:44 2019

@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Data Aquisition
"""

# Import libraries
import numpy as np
import os, codecs
import re


# Check the environment of python is set correctly
import sys
print('\n'.join(sys.path))


# Set working directory
directory = os.getcwd()

# Set directory with dataset
dataset_dir = os.path.join(directory, "dataset")

# Function to get all initial data
def get_all_data(dataset_dir):
    # Initiate the information about the map, trajectory
    world_data = np.loadtxt(os.path.join(dataset_dir, "world.dat"))
    #w_columns=['point', 'l_x', 'l_y', 'l_z', 'l_a1', 'l_a2', 'l_a3', 'l_a4', 
    #'l_a5', 'l_a6', 'l_a7', 'l_a8', 'l_a9', 'l_a10']
    trajectory_data = np.genfromtxt(os.path.join(dataset_dir, "trajectory.dat"))[:,1:]
    
    #t_columns=['poseID', 'odo_pose_x', 'odo_pose_y', 'odo_pose_theta', 'truth_pose_x',
    #'truth_pose_y', 'truth_pose_theta']
    camera_data = [['cam_matrix', np.matrix([[180, 0, 320], [0, 180, 240], [0, 0, 1]])],
                   ['cam_transform', np.matrix([[0, 0, 1, 0.2], [-1, 0, 0, 0], [0, 
                                               -1, 0, 0], [0, 0, 0, 0]])],
                   ['z_near', 0],
                   ['z_far', 5],
                   ['width', 640],
                   ['height', 480]]
    
    return(world_data, trajectory_data, camera_data)


# Function to acquire the data for each measurement and clean it to use it
def get_meas_data(doc_name, dataset_dir):
    meas = []
    with codecs.open(os.path.join(dataset_dir, doc_name), "r") as file: 
        for line in file.readlines()[:-1]: 
            line = re.sub('point ', 'point,', line) 
            line = re.sub(': +', ',', line) 
            line = re.sub(' +', ' ', line) 
            line = re.sub(' ', ',', line) 
            line = re.sub('\n', '', line)
#            line = line, end=''
            split = np.array(line.split(','))
            meas += [split] 
    return(meas)

# Acquire the next measurement after robot moves and keep info about previous measurement
#traj_prev = [0, 0]
#while robot_moved == True:
def get_new_seqdata(dataset_dir, i):
    _, trajectory_data, _ = get_all_data(dataset_dir)
    #    for i in range(336): # control has t+1 compared to meas info since control is u_t-1
    #        meas_prev = meas_curr
    if i < 10: meas_curr = get_meas_data('meas-0000' + str(i) + '.dat', dataset_dir) 
    elif i < 100: meas_curr = get_meas_data('meas-000' + str(i) + '.dat', dataset_dir)
    #        elif i < 336: meas_curr = get_meas_data('meas-00' + str(i) + '.dat', dataset_dir)
    else: meas_curr = get_meas_data('meas-00' + str(i) + '.dat', dataset_dir)
    
    # Call each item in measurement
    meas_gt_curr = np.array([meas_curr[1][1], meas_curr[1][2], meas_curr[1][3]])
    meas_odo_curr = np.array([meas_curr[2][1], meas_curr[2][2], meas_curr[2][3]])
    
    
    # Get all landmark points
    for j in range(3, len(meas_curr)):      
        meas_lpoint = [np.matrix([meas_curr[j][1], meas_curr[j][2]]), 
                       np.matrix([meas_curr[j][3], meas_curr[j][4]]),
                       np.matrix([meas_curr[j][5], meas_curr[j][6], meas_curr[j][7],
                        meas_curr[j][8], meas_curr[j][9], meas_curr[j][10],
                        meas_curr[j][11], meas_curr[j][12], meas_curr[j][13],
                        meas_curr[j][14]])]
    
    # Get previous and current trajectory data -- for odometry and ground truth
    #    traj_prev = [trajectory_data[i-1,1], trajectory_data[i-1,2], trajectory_data[i-1,3]]
    #       The control starts at t-1 
    if i == 0:
        traj_odo_prev = [0, 0, 0] # Initial odometry control robot pose at 0
        traj_gt_prev = [0, 0, 0] # Initial ground truth of control robot pose at 0
    else:
        traj_odo_prev = [trajectory_data[i-1,1], trajectory_data[i-1,2], trajectory_data[i-1,3]]
        traj_gt_prev = [trajectory_data[i-1,4], trajectory_data[i-1,5], trajectory_data[i-1,6]]
    
    traj_odo_curr = [trajectory_data[i,1], trajectory_data[i,2], trajectory_data[i,3]]
    traj_gt_curr = [trajectory_data[i,4], trajectory_data[i,5], trajectory_data[i,6]]

    control_input = np.array([traj_odo_prev, traj_odo_curr])
    return(meas_gt_curr, meas_odo_curr, meas_lpoint, control_input, traj_gt_prev, traj_gt_curr)
#    return(meas_gt_curr, meas_odo_curr, meas_lpoint, traj_odo_prev, traj_gt_prev)


##############################################################################

#                    END ACQUIRING DATA

###############################################################################

