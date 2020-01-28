
"""
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
    trajectory_data = np.genfromtxt(os.path.join(dataset_dir, "trajectory.dat"))[:,1:]

    camera_data = [['cam_matrix', np.matrix([[180, 0, 320], [0, 180, 240], [0, 0, 1]])],
                   ['cam_transform', np.matrix([[0, 0, 1, 0.2],
                                                [-1, 0, 0, 0],
                                                [0, -1, 0, 0],
                                                [0, 0, 0, 1]])],
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
            split = np.array(line.split(','))
            meas += [split]
    return(meas)

# Acquire the next measurement after robot moves and keep info about previous measurement
def get_new_seqdata(dataset_dir, i, all_obs, robot_pose_map, robot_gt_pose_map):
    _, trajectory_data, _ = get_all_data(dataset_dir)

    if i < 10: meas_curr = get_meas_data('meas-0000' + str(i) + '.dat', dataset_dir)
    elif i < 100: meas_curr = get_meas_data('meas-000' + str(i) + '.dat', dataset_dir)
    else: meas_curr = get_meas_data('meas-00' + str(i) + '.dat', dataset_dir)

    # Call each item in measurement
    meas_gt_curr = np.array([meas_curr[1][1], meas_curr[1][2], meas_curr[1][3]])
    meas_odo_curr = np.array([meas_curr[2][1], meas_curr[2][2], meas_curr[2][3]])

    # Get all landmark points
    meas_lpoint = []
    for j in range(3, len(meas_curr)):
        meas_lpoint_item = [np.array([meas_curr[j][1], meas_curr[j][2]], dtype=int),
                       np.array([meas_curr[j][3], meas_curr[j][4]], dtype='float'),
                       np.array([meas_curr[j][5], meas_curr[j][6], meas_curr[j][7],
                        meas_curr[j][8], meas_curr[j][9], meas_curr[j][10],
                        meas_curr[j][11], meas_curr[j][12], meas_curr[j][13],
                        meas_curr[j][14]], dtype='float')]
        meas_lpoint += [meas_lpoint_item]

    all_obs.append(meas_lpoint)

    x = trajectory_data[i, 1]
    y = trajectory_data[i, 2]
    theta = trajectory_data[i, 3]

    x_gt = trajectory_data[i, 4]
    y_gt = trajectory_data[i, 5]
    theta_gt = trajectory_data[i, 6]

    # Save the pose for actual and ground truth on their respective maps
    robot_pose_map[i,0] = x
    robot_pose_map[i,1] = y
    robot_pose_map[i,2] = theta

    robot_gt_pose_map[i,0] = x_gt
    robot_gt_pose_map[i,1] = y_gt
    robot_gt_pose_map[i,2] = theta_gt

    control_input = np.array([robot_pose_map[i,:], robot_gt_pose_map[i,:]]).reshape(6,1)

    return(meas_gt_curr, meas_odo_curr, meas_lpoint, all_obs, control_input, robot_pose_map, robot_gt_pose_map)
