#!/usr/bin/env python3

"""
@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Main Setup
"""

# Import libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import math
# plt.ion()

# Import files
from data import get_all_data, get_new_seqdata
from prediction import pose_model
from error_ellipse import error_ellipse
from correction import correction
from newlandmark import newlandmark
from associatelandmark import associateLandmarkIDs
from landmarks_model import landmark_model
from funcTools import *


directory = os.getcwd() # Set working directory
dataset_dir = os.path.join(directory, "dataset") # Set directory with dataset

world_data, trajectory_data, camera_data = get_all_data(dataset_dir) # Get the data info

# Initialize variables
rob_poseH = np.zeros([4, 4, 1])
rob_poseH_gt = np.zeros([4, 4, 1])
rob_update = np.zeros([4, 4, 1]) # updated pose after correction


id_to_state_map = np.ones((1000, 14), dtype='float64')*-1
state_to_id_map = np.ones((1000, 1), dtype='int32')*-1
# will retain the pose of the robot for each time sequence
robot_pose_map = np.zeros((336, 3))
robot_gt_pose_map = np.zeros((336, 3))

land_triang = np.ones([3,1])*-1
land_triang_gt = np.ones([3,1])*-1
Land_TriangPrev = np.ones([3,1])*-1
land_id = list()

projection_associations = np.ones([3,1]).astype(int)*-1
landmark_associations = np.ones([2,1]).astype(int)*-1
land_uv = np.ones([3, 1])*-1e7
land_index = -1

all_obs = []


for t in range(335):
    print('***********   t = ', t, ' *************')

    # Get the data from the correct sequence
    meas_gt_curr, meas_odo_curr, meas_lpoint, all_obs, control_input, robot_pose_map, robot_gt_pose_map = get_new_seqdata(dataset_dir, t, all_obs, robot_pose_map, robot_gt_pose_map)

    # Create the robot poses
    rob_poseH, rob_poseH_gt, rob_traj, rob_traj_gt, pose_associations = pose_model(t, rob_poseH, rob_poseH_gt, control_input)
    rob_traj_pred = rob_traj
    # return the rotation angle
    angle = math.acos(rob_poseH[0,0,t])
    pred_pose = np.insert(rob_poseH[0:2,3,t].reshape(2,1), 2, angle, axis=0)

    print('Prediction of robot pose: \n', pred_pose)

#   Obtain current observation using the data association
    meas_lpoint = associateLandmarkIDs(world_data, meas_lpoint)

    # Get all landmark information
    num_poses, num_landmarks, land_uv, projection_associations, landmark_associations = landmark_model(t, rob_poseH, land_triang, land_uv, meas_lpoint, state_to_id_map, projection_associations, landmark_associations)

#    ADD new landmarks to the state
    land_triang, land_triang_gt, id_to_state_map, state_to_id_map, land_id, Land_TriangPrev = newlandmark(t, world_data, rob_poseH, land_triang, land_triang_gt, meas_lpoint, id_to_state_map, state_to_id_map, rob_update, robot_pose_map, robot_gt_pose_map, land_id, Land_TriangPrev)

Land_TriangPrev = land_triang.copy()


# Correction
rob_poseH, land_triang, chi_stats_p, num_inliers_p, chi_stats_l, num_inliers_l, chi_stats_r, num_inliers_r, iteration = correction(t, rob_poseH, rob_poseH_gt, land_triang, land_triang_gt, num_poses, num_landmarks, land_uv, state_to_id_map, pose_associations, projection_associations, landmark_associations, rob_traj)
for time_step in range(rob_poseH.shape[2]):
    angle = math.atan2(rob_poseH[1,0,time_step], rob_poseH[0,0,time_step])
    corr_pose = np.insert(rob_poseH[0:2,3,time_step].reshape(2,1), 2, angle, axis=0)

# Keep track of robot pose at correction, to use for new_landmark
# rob_update = np.vstack([ rob_update, t2v(rob_poseH[:,:,t]) ])
print('Correction of robot pose: \n', corr_pose)
print('Ground Truth of robot pose: \n', robot_gt_pose_map[t, 0:3].reshape(3,1))

############################      PLOT       ###############################
state_items = len(list(set(list(state_to_id_map.flatten()))))
if t > 0 and  state_items > 1:
    # Separate landmark x and y for each
    l_x = np.array(land_triang[0,:])
    l_y = np.array(land_triang[1,:])

    ann_list_gt = []
    ann_list_pred = []

    for k, val in enumerate(state_to_id_map[:,0]):
        if val != -1:
            gt_l = plt.scatter(world_data[val,1], world_data[val,2], color='red', marker = '+', s =3)
            ann_lgt = plt.annotate(val, (world_data[val,1], world_data[val,2]))
            ann_lgt.set_fontsize(6)
            ann_lgt.set_color('red')
            ann_lpred = plt.annotate(val, (l_x[0,k], l_y[0,k]))
            ann_lpred.set_fontsize(6)
            ann_lpred.set_color('purple')
            ann_list_gt.append(ann_lgt)
            ann_list_pred.append(ann_lpred)
    # Predicted landmarks
    pred_l = plt.scatter(l_x[:], l_y[:], color='purple', marker = 'o', s=2)

# Plot odometry trajectory of robot
plt.scatter(robot_pose_map[:, 0], robot_pose_map[:,1], color='orange', marker = 'o', s=4)

# Plot actual robot tranjectory
rob_ = plt.scatter(rob_poseH[0,3,:], rob_poseH[1,3,:], color='blue', marker = 'o', s=4)

# Plot ground truth trajectory of robot
plt.scatter(robot_gt_pose_map[:, 0], robot_gt_pose_map[:,1], color='green', marker = 'o', s=4)

plt.savefig("images/full_pose_landmarks.png", dpi=100)
plt.show()


################################# CREATE GRAPHS  #################################

# Landmark and poses
fig1 = plt.figure(1)
fig1.set_size_inches(16, 12)
fig1.suptitle("Landmark and Poses", fontsize=16)

ax1 = fig1.add_subplot(221)
ax1.plot(rob_poseH[0,3,:],rob_poseH[1,3,:], 'o', mfc='none', color='b', markersize=3)
ax1.plot(Land_TriangPrev[0, :], Land_TriangPrev[1, :], 'x', color='orange', markersize=3)
# ax1.axis([-15,15,-15,15])
# ax1.set_zlim([-3,3])
ax1.set_title("Landmark ground truth and triangulation", fontsize=10)

ax2 = fig1.add_subplot(222)
ax2.plot(rob_poseH[0,3,:],rob_poseH[1,3,:],'o', mfc='none', color='b', markersize=3)
ax2.plot(land_triang[0,:],land_triang[1,:], 'x', color='g', markersize=3)
# ax2.axis([-15,15,-15,15])
# ax2.set_zlim([-3,3])
ax2.set_title("Landmark ground truth and optimization", fontsize=10)

ax3 = fig1.add_subplot(223)
ax3.plot(robot_gt_pose_map[:,0],robot_gt_pose_map[:,1], 'o', mfc='none', color='g', markersize=3)
ax3.plot(robot_pose_map[:,0],robot_pose_map[:,1], 'x', color='orange', markersize=3)
ax3.axis([-10,10,-10,10])
ax3.set_title("Robot ground truth and odometry values", fontsize=10)

# Estimated trajectory
traj_estimated = np.zeros([3, num_poses])
traj_true = np.zeros([3, num_poses])

for i in range(num_poses):
    traj_estimated[:,i] = t2v(rob_poseH[:,:,i])[0:3]
    traj_true[:,i] = t2v(rob_poseH_gt[:,:,i])[0:3]


ax4 = fig1.add_subplot(224)
ax4.plot(traj_true[0,:],traj_true[1,:], 'o', mfc='none', color='g', markersize=3)
ax4.plot(traj_estimated[0,:],traj_estimated[1,:], 'x', color='b', markersize=3)
ax4.axis([-10,10,-10,10])
ax4.set_title("Robot ground truth and optimization", fontsize=10)

# Chi and inliers
fig2 = plt.figure(2)
fig2.set_size_inches(16, 12)
fig2.suptitle("Chi and Inliers", fontsize=16)

ax3 = fig2.add_subplot(221)
ax3.plot(chi_stats_r[0:iteration])
ax3.set_title("Chi poses", fontsize=10)
ax4 = fig2.add_subplot(222)
ax4.plot(num_inliers_r[0:iteration])
ax4.set_title("Inliers poses", fontsize=10)

ax5 = fig2.add_subplot(223)
ax5.plot(chi_stats_p[0:iteration])
ax5.set_title("Chi projections", fontsize=10)
ax6 = fig2.add_subplot(224)
ax6.plot(num_inliers_p[0:iteration])
ax6.set_title("Inliers projections", fontsize=10)

# Chi and inliers
fig3 = plt.figure(3)
fig3.set_size_inches(16, 6)
fig3.suptitle("Landmarks (without outliers)", fontsize=16)

ax7 = fig3.add_subplot(121)
ax7.plot(world_data[:,1],world_data[:,2], 'o', mfc='none', color='purple', markersize=3)
ax7.plot(Land_TriangPrev[0, :], Land_TriangPrev[1, :], 'x', color='r', markersize=3)
ax7.set_title("Landmark ground truth and triangulation", fontsize=10)
ax7.axis([-15,15,-15,15])
ax8 = fig3.add_subplot(122)
ax8.plot(world_data[:,1],world_data[:,2], 'o', mfc='none', color='purple', markersize=3)
ax8.plot(land_triang[0,:],land_triang[1,:], 'x', color='r', markersize=3)
ax8.axis([-15,15,-15,15])
ax8.set_title("Landmark ground truth and optimization", fontsize=10)

# Save figures
fig1.savefig("images/landmark_and_pose.png", dpi=100)
fig2.savefig("images/chi_and_inliers.png", dpi=100)
fig3.savefig("images/landmarks_optimized.png", dpi=100)

plt.show()
