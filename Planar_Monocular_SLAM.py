#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 17:21:44 2019

@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM
"""

# Import libraries
import numpy as np
import os, codecs
import pandas as pd
import matplotlib.pyplot as plt
import re
import math

# Check the environment of python is set correctly
import sys
print('\n'.join(sys.path))


# Set working directory
directory = os.getcwd()

# Set directory with dataset
dataset_dir = os.path.join(directory, "dataset")


# Initiate the information about the map, trajectory
world_data = np.loadtxt(os.path.join(dataset_dir, "world.dat"))
#w_columns=['point', 'l_x', 'l_y', 'l_theta', 'l_a1', 'l_a2', 'l_a3', 'l_a4', 
#'l_a5', 'l_a6', 'l_a7', 'l_a8', 'l_a9', 'l_a10']
trajectory_data = np.genfromtxt(os.path.join(dataset_dir, "trajectory.dat"))[:,1:]
#t_columns=['poseID', 'odo_pose_x', 'odo_pose_y', 'odo_pose_theta', 'truth_pose_x',
#'truth_pose_y', 'truth_pose_theta']
camera_data = [['cam_matrix', np.array([[180, 0, 320], [0, 180, 240], [0, 0, 1]])],
               ['cam_transform', np.array([[0, 0, 1, 0.2], [-1, 0, 0, 0], [0, 
                                           -1, 0, 0], [0, 0, 0, 0]])],
               ['z_near', 0],
               ['z_far', 5],
               ['width', 640],
               ['height', 480]]


# Function to acquire the data for each measurement and clean it to use it
def get_meas_data(doc_name):
    meas = []
    with codecs.open(os.path.join(dataset_dir, doc_name), "r") as file: 
        for line in file.readlines(): 
            line = re.sub('point ', 'point,', line) 
            line = re.sub(': +', ',', line) 
            line = re.sub(' +', ' ', line) 
            line = re.sub(' ', ',', line) 
            split = np.array(line.split(','))
            meas += [split] 
    return(meas)

# Acquire the next measurement after robot moves and keep info about previous measurement
#while robot_moved == True:
for i in range(336):
    meas_prev = meas_curr
    if i < 10: meas_curr = get_meas_data('meas-0000' + str(i) + '.dat') 
    elif i < 100: meas_curr = get_meas_data('meas-000' + str(i) + '.dat')
    else: meas_curr = get_meas_data('meas-00' + str(i) + '.dat')
    

# Start testing with only 3 instances of the data
world_data = world_data[0:3]
trajectory_data = trajectory_data[0:3]

"""
w_columns=['point', 'x', 'y', 'theta', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 
         'a8', 'a9', 'a10']

world_data_df = pd.DataFrame(world_data, columns=w_columns)

x_world = world_data.tolist()[:,1]
    x_world = i

x_world = world_data[:,1] + math.cos()


for i, col in zip(world_data):
    if col == 1:
        x_world = col 
"""




# Test that the landmarks with their correct pose are drawm on the map
plt.scatter(x=world_data[:,1], y=world_data[:,2])
plt.show()
