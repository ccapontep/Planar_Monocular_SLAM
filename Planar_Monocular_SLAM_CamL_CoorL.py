#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 17:21:44 2019

@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Landmark camera to Coordinate
"""

# Import libraries
import numpy as np
import os
import math

# Import files
from Planar_Monocular_SLAM_data import get_all_data
from Planar_Monocular_SLAM_functools import v2t

# Set working directory
directory = os.getcwd()

# Set directory with dataset
dataset_dir = os.path.join(directory, "dataset")

_, _, camera_data = get_all_data(dataset_dir)

# Get useful info about camera
cam_matrix = camera_data[0][1]
cam_transform = camera_data[1][1]
cam_rot = np.copy(cam_transform)
cam_rot[0,3] = 0
cam_transl_h = np.eye(4)
cam_transl_h[0, 3] = cam_transform[0, 3]
cam_transl = (cam_transl_h[0:3,3]).reshape(3,1)

# World to camera transform
cam_transl_T = -np.matrix(np.transpose(cam_rot)[0:3,0:3]) * cam_transl
cam_rot_T = np.matrix(np.transpose(cam_rot)[0:3,0:3])
cam_world_cam_h = np.eye(4)
cam_world_cam_h[0:3,0:3] = cam_rot_T
cam_world_cam_h[0:3,3] = cam_transl_T.reshape(3,)


# Get initial locations of Principal Point Offset
u_0 =  cam_matrix[0,2]
v_0 = cam_matrix[1,2]

# Get focal length
f = cam_matrix[0,0]

# Distortion coefficient -- quantity that camera optics distorts the images
# approximate based on an assumption of a R-T Model with division correction as  
# seen in Section 2.1.4 in paper Lens Correction and Gamma Correction
# file:///home/ccapontep/Downloads/9789401790741-c2.pdf
# because k2 << k1, k2 is often ignored
d = 9.46804e-8 # distortion in the radial direction
#d2 =  -7.1974e-14  # distortion in the tangental direction

# Another approximation, has a greater distortion in radial direction by 10 pixels
## seen in Table 1 in paper A new calibration model of camera lens distortion
## http://www.dia.fi.upm.es/~lbaumela/Vision13/PapersCalibracion/wang-PR208.pdf
#d = -12.215e-7 # distortion in the radial direction
#d2 = 20.098e-13  # distortion in the tangental direction

###############################################################################

#               LANDMARK CAMERA TO COORDINATE SETUP

###############################################################################

# These series of functions change the camera coordinate information from the camera to
# x, y, z coordinate system for each landmark



# This function gets the position of the camera during each obsercvations
# and by transforming (rot + trans) the robot location it into camera coordinate
def cam_coor(rob_pose1, rob_pose2):

    # Transpose camera rotation and translation matrices
    cam_rot_T = np.matrix(np.transpose(np.copy(cam_rot))[0:3,0:3])
#    cam_trans_T = np.array(cam_rot_T * -cam_transl)
    
    # Convert robot pose to homogeneous coordinates
    Rob_pose1 = v2t(np.array(rob_pose1[0:2,0]).reshape(2,1), 0, rob_pose1[2,0])
    Rob_pose2 = v2t(np.array(rob_pose2[0:2,0].reshape(2,1)), 0, rob_pose2[2,0]) 
    print('Rob_pose1 \n', Rob_pose1)
    
    # Create base for homogeneous coordinates
    Cam_pose_1 = np.eye(4)
    Cam_pose_2 = np.eye(4)

    # Add rotation
    Cam_pose_1[0:3,0:3] = cam_rot_T * np.matrix(Rob_pose1[0:3,0:3])
    Cam_pose_2[0:3,0:3] = cam_rot_T * np.matrix(Rob_pose2[0:3,0:3])
    
    # Add translation
    Cam_pose_1[0:3,3] = (cam_rot_T * (cam_transl + np.array(Rob_pose1[0:3,3]))).reshape(3,)
    Cam_pose_2[0:3,3] = (cam_rot_T * (cam_transl + np.array(Rob_pose2[0:3,3]))).reshape(3,)
    print('Cam_pose_1 \n', Cam_pose_1)
    
#    # Get camera rotation from robot coordinate
#    Cam_rot_1 = cam_rot_T * np.matrix(Rob_pose1[0:3,0:3])
#    Cam_rot_2 = cam_rot_T * np.matrix(Rob_pose2[0:3,0:3])
#
#    # Get camera translation from robot coordinate
#    cam_trans_T_1 = (-Cam_rot_1 * (cam_transl + np.array(Rob_pose1[0:3,3]))).reshape(3,)
#    cam_trans_T_2 = (-Cam_rot_2 * (cam_transl + np.array(Rob_pose2[0:3,3]))).reshape(3,)
#
#    # Create base for homogeneous coordinates
#    Cam_pose_1 = np.eye(4)
#    Cam_pose_2 = np.eye(4)
#
#    # Add rotation
#    Cam_pose_1[0:3,0:3] = Cam_rot_1
#    Cam_pose_2[0:3,0:3] = Cam_rot_2
#    
#    # Add translation
#    Cam_pose_1[0:3,3] = cam_trans_T_1
#    Cam_pose_2[0:3,3] = cam_trans_T_2
    
#    print('rob_pose2 \n', rob_pose2)
#    print('Rob_pose2 \n', Rob_pose2)    
#    print('Cam_rot_2 \n', Cam_rot_2)
#    print('cam_trans_T_2 \n', cam_trans_T_2)
#    print('Cam_pose_2 \n', Cam_pose_2)
    
    # Get camera rotation from robot coordinate
#    Cam_pose_1 = cam_world_cam_h * Rob_pose1
#    Cam_pose_2 = cam_world_cam_h * Rob_pose2
    
    return(Cam_pose_1, Cam_pose_2)


##############################################################################


# This function calculates the current quaternion and R(q) rotation of the camera
def Rot_quaternion(Cam_pose):

    # Get the quaternion using the current rotation matrix of the camera transform
    # from eqns A.2 and A.3 in https://pdfs.semanticscholar.org/3a02/048014b570c3081cf28b444c420af025d785.pdf
#    q = np.array([1, 0.25, -0.25, 0.25]).reshape(4,1)
#    q1 = math.sqrt(1 + Cam_pose[0,0] + Cam_pose[1,1] + Cam_pose[2,2])
#    q2 = (Cam_pose[2,1] - Cam_pose[1,2]) / (4 * q1)
#    q3 = (Cam_pose[0,2] - Cam_pose[2,0]) / (4 * q1)
#    q4 = (Cam_pose[1,0] - Cam_pose[0,1]) / (4 * q1)
#    
#    # Get the quaternion rotation
#    R_col1_1= q1**2 + q2**2 - q3**2 - q4**2
#    R_col1_2 = 2 * (q2 * q3 + q1 * q4)
#    R_col1_3 = 2 * (q2 * q4 - q1 * q3)
#    
#    R_col2_1 = 2 * (q2 * q3 - q1 * q4)
#    R_col2_2= q1**2 - q2**2 + q3**2 - q4**2
#    R_col2_3 = 2 * (q1 * q2 + q3 * q4)
#    
#    R_col3_1 = 2 * (q1 * q3 + q2 * q4)
#    R_col3_2 = 2 * (q3 * q4 - q1 * q2)
#    R_col3_3= q1**2 - q2**2 - q3**2 + q4**2
#    
#
#    Rq = np.matrix([[R_col1_1, R_col2_1, R_col3_1], 
#                    [R_col1_2, R_col2_2, R_col3_2], 
#                    [R_col1_3, R_col2_3, R_col3_3]])
    
#    Rq = np.matrix([[R_col1_1, R_col2_1, R_col3_1], 
#                    [R_col1_2, R_col1_1, R_col3_2], 
#                    [R_col1_3, R_col2_3, R_col1_1]])
    
    
    # Quaternion rotation has been tested where Rq*Rq_T is equal to the identity
    # http://orion.lcg.ufrj.br/games/ArcBall/Rotation-formalisms-in-three-dimensions.pdf
    
    q4 = 0.5 * math.sqrt(1 + Cam_pose[0,0] + Cam_pose[1,1] + Cam_pose[2,2])
    q1 = (Cam_pose[2,1] - Cam_pose[1,2]) / (4 * q4)
    q2 = (Cam_pose[0,2] - Cam_pose[2,0]) / (4 * q4)
    q3 = (Cam_pose[1,0] - Cam_pose[0,1]) / (4 * q4)
    
    # Get the quaternion rotation
    R_col1_1= 1 - 2*q2**2 - 2*q3**2
    R_col1_2 = 2 * (q1 * q2 + q3 * q4)
    R_col1_3 = 2 * (q1 * q3 - q2 * q4)
    
    R_col2_1 = 2 * (q1 * q2 - q3 * q4)
    R_col2_2= 1 - 2*q1**2 - 2*q3**2
    R_col2_3 = 2 * (q1 * q4 + q2 * q3)
    
    R_col3_1 = 2 * (q1 * q3 + q2 * q4)
    R_col3_2 = 2 * (q2 * q3 - q1 * q4)
    R_col3_3= 1 - 2*q1**2 - 2*q2**2
    

    Rq = np.matrix([[R_col1_1, R_col2_1, R_col3_1], 
                    [R_col1_2, R_col2_2, R_col3_2], 
                    [R_col1_3, R_col2_3, R_col3_3]])
    
    
    return(Rq)

# This function calculates the directional projection ray vector
def proj_ray_calc(u, v, Cam_pose):
    
    # Get the current rotation of camera in quaternion terms
    Rq = Rot_quaternion(Cam_pose)
    
    # Calculate the Euclidean distance between the initial position and current landmark
    r = math.sqrt((u - u_0)**2 + (v - v_0)**2)
    
    # Calculate the undistored pixel locations
    u_u = ((u - u_0)/math.sqrt(1-(2*d*r**2))) + u_0
    v_u = ((v - v_0)/math.sqrt(1-(2*d*r**2))) + v_0
    
    # Directional vector from camera location to the feature seen
    h_c = np.array([[(u_0 - u_u)/f], [(v_0 - v_u)/f], [1]]).reshape(3,1)

    # Calculate the directional projection ray vector
    H_C = Rq * h_c
    
    return(H_C, Rq)
    

##############################################################################

# This function calculates the angle beta for defining the base-line b in the 
# direction of the camera trajectory
def beta_calc(u, v, Cam_pose_1, Cam_pose_2):
    
### Calculate the beta value for observation of the first about the second 
# observation of this feature:
    
    # Get projection ray vector -- use the camera position when it was first observed
    H_C_1, Rq_1 =  proj_ray_calc(u, v, Cam_pose_1)
    
    # return only camera location, not ratation matrix
    Cam_pos_1 = Cam_pose_1[0:3,3].reshape(3,1)
    Cam_pos_2 = Cam_pose_2[0:3,3].reshape(3,1)
    
    # Calculate the vector for camera baseline between camera optimal center
    # position at first obs of feature and the second time
    b1 = (Cam_pos_2 - Cam_pos_1)
    b = np.linalg.norm(b1)
    
    # Make arrays of size (3,)
    H_C_1 =  np.array([H_C_1[0,0], H_C_1[1,0], H_C_1[2,0]])
    b1 = np.array([b1[0,0], b1[1,0], b1[2,0]])

    # Calculate beta 
    beta = math.acos(np.dot(H_C_1, b1) / (np.linalg.norm(H_C_1) * np.linalg.norm(b1)))
#    print('beta \n', beta)
    return(beta, b, Rq_1)
    
##############################################################################
    
# This function calculates the angle gamma determined in a similar way as beta
# but using instead the directional projection ray vector h2 and the vector b2
def gamma_calc(u, v, Cam_pos_1, Cam_pos_2):

### Calculate the gamma value for observation of the first about the second 
# observation of this feature:

    # Get projection ray vector
    H_C_2, Rq =  proj_ray_calc(u, v, Cam_pos_2)

    # return only camera location, not ratation matrix
    Cam_pos_1 = Cam_pos_1[0:3,3].reshape(3,1) 
    Cam_pos_2 = Cam_pos_2[0:3,3].reshape(3,1)

    # Calculate the vector for camera baseline between camera optimal center
    # position at first obs of feature and the second time
    b2 = (Cam_pos_1 - Cam_pos_2)
    
    # Make arrays of size (3,)
    H_C_2 =  np.array([H_C_2[0,0], H_C_2[1,0], H_C_2[2,0]])
    b2 = np.array([b2[0,0], b2[1,0], b2[2,0]])
    
    # Dot product of 3x1 vectors divided by their Euclidean distance of each 
    # multiplied to obtain the angle of the unit vector norm
    gamma = math.acos(np.dot(H_C_2, b2) / (np.linalg.norm(H_C_2) * np.linalg.norm(b2)))
#    print('gamma \n', gamma)
    return(gamma, H_C_2, Rq)
    
##############################################################################

# This function calculates the angle alpha which is the parallax angle
def alpha_calc(beta, gamma):
    
    alpha = math.pi - (beta + gamma)
    
    return(alpha)

##############################################################################
    
# This function calculates the location of the landmark after the second observation
def landmark_pos(beta, alpha, gamma, H_C_2, b, Cam_pose_1, Cam_pose_2, Rq_curr, Rq_1):
    
    # Calculate the value 'row': inverse depth between the feature and the robot

    row = math.sin(alpha) / (b * math.sin(beta))
    
    # theta, phi represent the azimuth and the elevation respectively respect 
    # to the world reference 
    
# works prev
    theta = math.atan2(-H_C_2[1], -H_C_2[0])
    phi = math.acos(H_C_2[2] / (math.sqrt((H_C_2[0])**2 + (H_C_2[1])**2) + (H_C_2[2])**2))
    
#    theta = math.atan2(-H_C_2[1], -H_C_2[0])
#    phi = math.acos(H_C_2[2] / (math.sqrt((H_C_2[0])**2 + (H_C_2[1])**2) + (H_C_2[2])**2))
  



# Works prev
    m_dir_vec = np.array([[math.cos(theta) * math.sin(phi)], 
                          [math.sin(theta) * math.sin(phi)], 
                          [math.cos(phi)]]).reshape(3,1)
    
#    m_dir_vec = np.array([[math.cos(theta) * math.sin(phi)], 
#                          [-math.cos(phi)], 
#                          [-math.sin(theta) * math.sin(phi)]]).reshape(3,1)

#    m_dir_vec = np.array([[math.sin(theta) * math.sin(phi)], 
#                          [-math.cos(theta) * math.sin(phi)], 
#                          [-math.cos(phi)]]).reshape(3,1)
#    m_dir_vec = np.array([[(math.cos(phi))], 
#                          [-(math.sin(theta) * math.sin(phi))], 
#                          [-(math.cos(theta) * math.sin(phi))]]).reshape(3,1)
#    m_dir_vec = np.array([[(math.cos(theta) * math.sin(phi))], 
#                           [(math.sin(theta) * math.sin(phi))], 
#                           [(math.cos(phi))]]).reshape(3,1)

    
    # Calculate the landmark position in camera coordinate
    obs_landmark_pos = (Cam_pose_1[0:3,3]).reshape(3,1) + (m_dir_vec / row)
#    obs_landmark_pos_h = v2t(obs_landmark_pos[0:2,0], obs_landmark_pos[2,0], 0)
#    Rq_curr_h_T = np.eye(3)
#    Rq_curr_h_T[0:3,0:3] = np.transpose(Rq_curr)
#    Rq_curr_h_T[0:3,3] = cam_transl.reshape(3,)
#    

#    obs_landmark_pos = np.transpose(Rq_1) * obs_landmark_pos
    # Get homogeneous coordinate
#    obs_landmark_pos_h = v2t(obs_landmark_pos, 0)
#    obs_landmark_pos_h[2,3] = obs_landmark_pos[2,0]
    
    # Calculate the landmark position in world coordinate
    # http://ksimek.github.io/2012/08/22/extrinsic/
#    landmark_pos_world = np.transpose(Rq_curr) * (obs_landmark_pos - cam_transl)
#    landmark_pos_cam = offset_coord * (Rq_curr_h_T * cam_transform * obs_landmark_pos_h)[0:3, 3]
#    landmark_pos_world = obs_landmark_pos
#    R_curr_T = Rot_quaternion(np.transpose(np.matrix(Cam_pose_2[0:3,0:3])))
    
    h_c_reverse = (np.transpose(Rq_curr) * (obs_landmark_pos - (Cam_pose_2[0:3,3]).reshape(3,1)))
#    h_c_reverse = R_curr_T * (obs_landmark_pos - (Cam_pose_2[0:3,3]).reshape(3,1))

    
    return(obs_landmark_pos, b, h_c_reverse, theta)


def dist_uv(u_rev, v_rev):

    # Get undistorted pixel coordinates
    u_und = u_0 - f*u_rev
    v_und = v_0 - f*v_rev
    
    # Calculate the Euclidean distance between the initial position and current landmark
    r = math.sqrt((u_und - u_0)**2 + (v_und - v_0)**2)
    
    # Calculate the undistored pixel locations
    u_dist = ((u_und - u_0)/math.sqrt(1+(2*d*r**2))) + u_0
    v_dist = ((v_und - v_0)/math.sqrt(1+(2*d*r**2))) + v_0
    
    
    return(u_dist, v_dist)
    
def undist_uv(u, v):
    
    # Calculate the Euclidean distance between the initial position and current landmark
    r = math.sqrt((u - u_0)**2 + (v - v_0)**2)
    
    # Calculate the undistored pixel locations
    u_u = ((u - u_0)/math.sqrt(1-(2*d*r**2))) + u_0
    v_u = ((v - v_0)/math.sqrt(1-(2*d*r**2))) + v_0
    
#    # Theta and phi angles from camera location to the feature seen
#    angles = np.array([[math.atan2((u_u - u_0)/f)], [math.atan2((v_u - v_0)/f)]]).reshape(2,1)

    return(angles)





###############################################################################

#               END LANDMARK CAMERA TO COORDINATE SETUP

###############################################################################
