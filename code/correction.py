
"""
@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Correction Function Setup
"""

# Import libraries
import numpy as np

from data import get_all_data
from funcTools import *
from pose_error_linearization import linearizePoses
from projection_error_linearization import linearizeProjections


# Implementation of the optimization loop with robust kernel
# applies a perturbation to a set of landmarks and robot poses
# Input:
#   rob_poseH: the initial robot poses (4x4xnum_poses: array of homogeneous matrices)
#   land_triang: the initial landmark estimates (3xnum_landmarks matrix of landmarks)
#   Z:  the measurements (3xnum_measurements)
#   associations: 2xnum_measurements.
#                 associations(:,k)=[p_idx,l_idx]' means the kth measurement
#                 refers to an observation made from pose p_idx, that
#                 observed landmark l_idx
#   num_poses: number of poses in rob_poseH (added for consistency)
#   num_landmarks: number of landmarks in land_triang (added for consistency)
#   max_iterations: the number of iterations of least squares
#   damping:      damping factor (in case system not spd)
#   kernel_threshod: robust kernel threshold
# Output:
#   rob_poseH: the robot poses after optimization
#   land_triang: the landmarks after optimization
#   chi_stats_{l,p,r}: array 1:max_iterations, containing evolution of chi2 for landmarks, projections and poses
#   num_inliers{l,p,r}: array 1:max_iterations, containing evolution of inliers landmarks, projections and poses
def correction(t, rob_poseH, rob_poseH_gt, land_triang, land_triang_gt, num_poses, num_landmarks, land_uv, state_to_id_map, pose_associations, projection_associations, landmark_associations, rob_traj):

    # Parameters with landmark visual odometry
    damping = 1e-4
    kernel_threshold_proj = 50
    kernel_threshold_pose = 3
    max_iterations = 200

    # For landmark Ground Truth
    # damping = 1e4
    # kernel_threshold_proj = 50
    # kernel_threshold_pose = 0.01
    # max_iterations = 200

    # Chi and Inliers
    chi_stats_p = np.zeros(max_iterations)
    num_inliers_p = np.zeros(max_iterations)
    chi_stats_l = np.zeros(max_iterations)
    num_inliers_l = np.zeros(max_iterations)
    chi_stats_r = np.zeros(max_iterations)
    num_inliers_r = np.zeros(max_iterations)


    # Size of the linear system
    state_size = pose_dim * num_poses + landmark_dim * num_landmarks

    iteration = 0
    error = 1e6
    while iteration < max_iterations and error > 1e-6: # and (curr2correct % step2correct == 0):
        print('Iteration: ' + str(iteration))

        # Poses
        H_poses, b_poses, chi_, num_inliers_ = linearizePoses(rob_poseH, rob_traj, state_to_id_map, pose_associations, num_poses, num_landmarks, kernel_threshold_pose)
        chi_stats_r[iteration] += chi_
        num_inliers_r[iteration] = num_inliers_


        # Projections
        if num_landmarks != 0 :
            H_proj, b_proj, chi_, num_inliers_ = linearizeProjections(rob_poseH, land_triang, land_uv, projection_associations, num_poses, num_landmarks, kernel_threshold_proj)
            chi_stats_p[iteration] += chi_
            num_inliers_p[iteration] = num_inliers_

            H = H_poses + H_proj
            b = b_poses + b_proj

        else:
            H = H_poses
            b = b_poses
            chi_stats_p[iteration] = 0
            num_inliers_p[iteration] = 0

            chi_stats_l[iteration] = 0
            num_inliers_l[iteration] = 0


        # Add damping
        H += np.eye(state_size) * damping

        # Solve linear system (block first pose)
        dx = np.zeros([state_size, 1])
        dx[pose_dim:] = -np.linalg.solve(H[pose_dim:, pose_dim:], b[pose_dim:,0]).reshape([-1,1])

        # Box plus
        rob_poseH, land_triang = boxPlus(rob_poseH, land_triang, num_poses, num_landmarks, dx)

        iteration += 1
        error = np.sum(np.absolute(dx))
        print("Error: " + str(error))

    return(rob_poseH, land_triang, chi_stats_p, num_inliers_p,chi_stats_l, num_inliers_l,chi_stats_r, num_inliers_r, iteration)
