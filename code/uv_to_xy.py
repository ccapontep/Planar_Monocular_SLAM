
"""
@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Convert uv coordinate to xy Setup
"""

# Import libraries
import numpy as np
import math

from CamL_CoorL import *


# This function computes gets the new landmark that has not been previously seen
#inputs:
#  t, u, v: time, u camera coord, v camera coord
#  meas_land: landmark point number
#
#  id_to_state_map:
#            mapping that given the id of the measurement, returns its position in the mu vector
#
#outputs:
#  [landmark_pos_world]: the location of the landmark in world coordinate
#  [id_to_state_map]: the updated mapping vector between landmark position in mu vector and its id

def uv_to_xy(t, u, v, meas_land, id_to_state_map, rob_poseH, robot_gt_pose_map, robot_pose_map, rob_update):

    # For the first observation get info of robot location:
    rob_time_1 = int(id_to_state_map[meas_land, 1])
    angle1 = math.atan2(rob_poseH[1,0,rob_time_1], rob_poseH[0,0,rob_time_1])
    # rob_pose1 = robot_gt_pose_map[rob_time_1, :].reshape(3,1) # to check for gt behavior
    rob_pose1 = robot_pose_map[rob_time_1, :].reshape(3,1)
    # rob_pose1 = rob_update[rob_time_1, :].reshape(6,1) # for continuous SLAM


    # Pose of robot during the second observation
    rob_time_2 = int(id_to_state_map[meas_land, 5])
    angle2 = math.atan2(rob_poseH[1,0,rob_time_2], rob_poseH[0,0,rob_time_2])
    # rob_pose2 = robot_gt_pose_map[rob_time_2, :].reshape(3,1) # to check for gt behavior
    rob_pose2 = robot_pose_map[rob_time_2, :].reshape(3,1)
    # rob_pose2 = rob_update[rob_time_2, :].reshape(6,1) # for continuous SLAM


    # Get the measurement col, rho of each time the landmark is seen (u, v, 1)
    u_v1 = np.array([id_to_state_map[meas_land, 2], id_to_state_map[meas_land, 3], 1]).reshape(3,1)
    u_v2 = np.array([id_to_state_map[meas_land, 6], id_to_state_map[meas_land, 7], 1]).reshape(3,1)


    # Get current Camera position
    Cam_pose_1, Cam_pose_2 = cam_coor(rob_pose1, rob_pose2)

    # For beta, use the camera coordinates of the observed point the first time
    beta, b = beta_calc(u_v1[0,0], u_v1[1,0], Cam_pose_1, Cam_pose_2)

    # For gamma, use the camera coordinates of the observed point the current time
    gamma, H_C_2 = gamma_calc(u_v2[0,0], u_v2[1,0], Cam_pose_1, Cam_pose_2)


    alpha = alpha_calc(beta, gamma)
    # Calculates the location of the landmark
    landmark_pos_world, rho, m_dir_vec, dist2land = landmark_pos(beta, alpha, gamma, b, H_C_2, Cam_pose_2)


    return (landmark_pos_world, id_to_state_map, alpha, beta, b, rho, gamma, dist2land)
