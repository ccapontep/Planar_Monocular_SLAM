"""
@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Landmark camera to Coordinate
"""

# Import libraries
import numpy as np
import math

# Import files
from data import get_all_data
from funcTools import *

# These series of functions change the camera coordinate information from the
# camera to x, y, z coordinate system for each landmark

###############################################################################

# This function gets the position of the camera during each obsercvations
# and by transforming (rot + trans) the robot location it into camera coordinate
def cam_coor(rob_pose1, rob_pose2):
    rob_pose1 = np.array([rob_pose1[0], rob_pose1[1], 0, 0, 0, rob_pose1[2]])
    rob_pose2 = np.array([rob_pose2[0], rob_pose2[1], 0, 0, 0, rob_pose2[2]])

    # Convert robot pose to homogeneous coordinates
    rob_pose1 = v2t_6(rob_pose1)
    rob_pose2 = v2t_6(rob_pose2)

    # # Convert robot pose to homogeneous coordinates
    # rob_pose1 = v2t(rob_pose1[0:2], 0, rob_pose2[2])
    # rob_pose2 = v2t(rob_pose2[0:2], 0, rob_pose2[2])

    Cam_pose_1 = rob_pose1 * cam_transform
    Cam_pose_2 = rob_pose2 * cam_transform

    return(Cam_pose_1, Cam_pose_2)

###############################################################################

# This function calculates the directional projection ray vector
def proj_ray_calc(u, v, Rq):

    # Directional vector from camera location to the feature seen
    h_c = np.array([[(u - u_0)/f], [(v - v_0)/f], [1]]).reshape(3,1)

    # Calculate the directional projection ray vector
    H_C = Rq * h_c

    return(H_C)


##############################################################################

# This function calculates the angle beta for defining the base-line b in the
# direction of the camera trajectory
def beta_calc(u, v, Cam_pose_1, Cam_pose_2):


    # Get projection ray vector -- use the camera position when it was first observed
    H_C_1 =  proj_ray_calc(u, v, Cam_pose_1[0:3,0:3])

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
    beta = (math.acos(np.dot(H_C_1, b1) / (np.linalg.norm(H_C_1) * np.linalg.norm(b1))))

    return(beta, b)

##############################################################################

# This function calculates the angle gamma determined in a similar way as beta
# but using instead the directional projection ray vector h2 and the vector b2
def gamma_calc(u, v, Cam_pos_1, Cam_pos_2):

    # Get projection ray vector
    H_C_2 =  proj_ray_calc(u, v, Cam_pos_2[0:3,0:3])

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
    gamma = (math.acos(np.dot(H_C_2, b2) / (np.linalg.norm(H_C_2) * np.linalg.norm(b2))))

    return(gamma, H_C_2)

##############################################################################

# This function calculates the angle alpha which is the parallax angle
def alpha_calc(beta, gamma):

    alpha = (math.pi - (beta + gamma))

    return(alpha)

##############################################################################

# This function calculates the location of the landmark after the second observation
def landmark_pos(beta, alpha, gamma, b, H_C_2, Cam_pose_2):

    # Calculate the value 'rho': inverse depth between the feature and the robot
    rho = (math.sin(alpha) / (b * math.sin(beta)))

    # theta, phi represent the azimuth and the elevation respectively respect
    # to the world reference
    theta = math.atan2(H_C_2[2], H_C_2[1])
    phi = math.atan2((math.sqrt(H_C_2[1]**2 + H_C_2[2]**2)), H_C_2[0])
    # Directional vector of where the landmark is located relative to the 2nd time
    # the robot is seen
    m_dir_vec = np.array([[math.cos(phi)],
                          [math.cos(theta) * math.sin(phi)],
                          [math.sin(theta) * math.sin(phi)]]).reshape(3,1)

    # Calculate the final location of the landmark
    dist2land = np.insert((m_dir_vec / rho),3,0,axis=0)
    landmark_pos_world = (Cam_pose_2[0:4,3]).reshape(4,1) + dist2land

    return(landmark_pos_world, rho, m_dir_vec, dist2land)
