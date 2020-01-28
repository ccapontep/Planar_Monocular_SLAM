
"""
@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Prediction Function Setup
"""

# Import libraries
import numpy as np
from funcTools import v2t_6


# This function implements the kalman prediction step of the SLAM system
# Inputs:
#   control input: previous and current odometry poses in the form:
#                  [  [x, y, theta], [x', y', theta']  ]
#  considering as 'm' the number of seen landmarks
#  mu_state: the previous mean of (x,y,theta, l1, l2, ..., lm), i.e. estimated robot pose
#      and the m landmark positions
#  sigma_state: the covariance of the previously estimated robot pose and landmark
#         positions ((3+m)x(3+m) matrix)

# outputs
# [mu_state, sigma_state] are mean and covariance of the estimate after transition

def pose_model(t, rob_poseH, rob_poseH_gt, control_input):

    # initiate output trajectory - needs to be updated each time since info
    # about robot location is constantly changing
    rob_traj = np.zeros([4, 4, 1])
    rob_traj_gt = np.zeros([4, 4, 1])
    pose_associations = np.zeros([2, 1]).astype(int)

    # initiate to later concatenate
    new_pose = np.zeros([4, 4, 1])
    new_pose_gt = np.zeros([4, 4, 1])
    new_traj = np.zeros([4, 4, 1])
    new_traj_gt = np.zeros([4, 4, 1])

    pose_odom = np.array([control_input[0], control_input[1], 0, 0, 0, control_input[2]])
    pose_gt = np.array([control_input[3], control_input[4], 0, 0, 0, control_input[5]])

    # create map for robot odometry and ground truth in homogeneous matrix
    new_pose[:, :, 0] = v2t_6(pose_odom)
    new_pose_gt[:, :, 0] = v2t_6(pose_gt)

    if t == 0:
        rob_poseH[:, :, 0] = v2t_6(pose_odom)
        rob_poseH_gt[:, :, 0] = v2t_6(pose_gt)
    else:
        rob_poseH = np.concatenate((rob_poseH, new_pose), axis= 2)
        rob_poseH_gt = np.concatenate((rob_poseH_gt, new_pose_gt), axis= 2)

    # get current and previous homog. poses
    num_poses = rob_poseH.shape[2]

    # Calculate the trajectory of the robot
    if num_poses > 1:
        for pose_num in range(num_poses-1): # one edge per two nodes, so one less pose
            rob_prev = rob_poseH[:, :, pose_num]
            rob_curr = rob_poseH[:, :, pose_num+1]

            rob_prev_gt = rob_poseH_gt[:, :, pose_num]
            rob_curr_gt = rob_poseH_gt[:, :, pose_num+1]

            new_traj[:, :, 0] = np.linalg.inv(rob_prev) @ rob_curr
            new_traj_gt[:, :, 0] = np.linalg.inv(rob_prev_gt) @ rob_curr_gt
            if pose_num == 0:
                rob_traj = new_traj
                rob_traj_gt = new_traj_gt
                pose_associations[:, 0] = [pose_num, pose_num+1]
            else:
                rob_traj = np.concatenate(( rob_traj, new_traj), axis= 2)
                rob_traj_gt = np.concatenate(( rob_traj_gt, new_traj_gt), axis= 2)
                pose_associations = np.concatenate(( pose_associations, np.array([[pose_num], [pose_num+1]])), axis= 1)


    return(rob_poseH, rob_poseH_gt, rob_traj, rob_traj_gt, pose_associations)
