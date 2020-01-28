
"""
@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Helpful Tool Functions Setup
"""

# Import libraries
from data import get_all_data

import numpy as np
import os
import math

# Set working directory
directory = os.getcwd()
# Set directory with dataset
dataset_dir = os.path.join(directory, "dataset")

_, _, camera_data = get_all_data(dataset_dir)

# Get useful info about camera
cam_matrix = camera_data[0][1]
cam_transform = camera_data[1][1]

# Dimensions
projection_dim = 2
pose_dim = 6
landmark_dim = 3

# Get initial locations of Principal Point Offset, focal length, and z far/near
u_0 =  cam_matrix[0,2]
v_0 = cam_matrix[1,2]
f = cam_matrix[0,0]
z_near = camera_data[2][1]
z_far = camera_data[3][1]
cam_width = camera_data[4][1]
cam_height = camera_data[5][1]

# This function converts a perturbation into a matrix
# computes the homogeneous transform matrix H of the pose vector v
# H:[ R t ] 3x3 homogeneous transformation matrix, r translation vector
# v: [x,y], v3: [z], th:theta

def v2t(v, v3, th):
    c = math.cos(th)
    s = math.sin(th)
    H = np.matrix([[c, -s, 0, v[0,0]],
                    [s, c, 0, v[1,0]],
                    [0, 0, 1, v3    ],
                    [0, 0, 0, 1     ]], dtype='float64')

    return(H)

# From 6d vector to homogeneous matrix
def v2t_6(v):
    T = np.eye(4)
    T[0:3,0:3] = Rx(v[3]) @ Ry(v[4]) @ Rz(v[5])
    T[0:3,[3]] = v[0:3].reshape((3, 1))
    return(T)



# This function converts a perturbation into a matrix
# computes the homogeneous transform matrix H of the derivative of the pose vector v
# H:[ R t ] 3x3 homogeneous transformation matrix, r translation vector
# v: [x,y], v3: [z], th:theta
def v2t_derivative(v, v3, th):
    c = math.cos(th)
    s = math.sin(th)
    H = np.matrix([[-s, -c, 0, v[0,0]],
                   [c,  -s, 0, v[1,0]],
                   [0,   0, 1, v3    ],
                   [0,   0, 0, 1     ]], dtype='float64')

    return(H)


# Converts a rotation matrix into a unit normalized quaternion
def mat2quat(R):
    qw4 = 2 * math.sqrt(1+R[0,0]+R[1,1]+R[2,2])
    q  = np.array([(R[2,1]-R[1,2])/qw4, (R[0,2]-R[2,0])/qw4, (R[1,0]-R[0,1])/qw4])
    return q

# From homogeneous matrix to 6d vector
def t2v(X):
    x = np.zeros(6)
    x[0:3] = X[0:3,3]
    x[3:6] = mat2quat(X[0:3,0:3])
    return x

# normalizes and angle between -pi and pi
# theta: input angle
# out: output angle
def normalizeAngle(theta):
    out = math.atan2(math.sin(theta),math.cos(theta))
    return(out)

# This function calculates a skew-symmetric matrix  --check the negatives are flipped
def skew(vec):
    S = np.array([  [0,          -vec.item(2),         vec.item(1)],
                    [vec.item(2),      0,              -vec.item(0)],
                    [-vec.item(1),     vec.item(0),              0]])
    return S


# Rotation matrix around x axis
def Rx(rot_x):
    c = math.cos(rot_x)
    s = math.sin(rot_x)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return R

# Rotation matrix around y axis
def Ry(rot_y):
    c = math.cos(rot_y)
    s = math.sin(rot_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return R

# Rotation matrix around z axis
def Rz(rot_z):
    c = math.cos(rot_z)
    s = math.sin(rot_z)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R

def flattenIsometryByColumns(T):
    v = np.zeros([12, 1])
    v[0:9] = np.reshape(T[0:3,0:3].transpose(), [9, 1])
    v[9:12] = T[0:3,[3]]
    return v

############################ DERIVATIVES ######################################

# Derivative of rotation matrix w.r.t rotation around x, in 0
Rx0 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])

# Derivative of rotation matrix w.r.t rotation around y, in 0
Ry0 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])

# Derivative of rotation matrix w.r.t rotation around z, in 0
Rz0 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])



# Applies a perturbation to a set of landmarks and robot poses
# Input:
#   XR: the robot poses (4x4xnum_poses: array of homogeneous matrices)
#   XL: the landmark pose (3xnum_landmarks matrix of landmarks)
#   num_poses: number of poses in XR (added for consistency)
#   num_landmarks: number of landmarks in XL (added for consistency)
#   dx: the perturbation vector of appropriate dimensions
#       the poses come first, then the landmarks
# Output:
#   XR: the robot poses obtained by applying the perturbation
#   XL: the landmarks obtained by applying the perturbation
def boxPlus(XR, XL, num_poses, num_landmarks, dx):
    for pose_index in range(num_poses):
        # pose_matrix_index = pose_index * 6
        pose_matrix_index = poseMatrixIndex(pose_index, num_poses, num_landmarks)
        dxr = dx[pose_matrix_index:pose_matrix_index + pose_dim]
        XR[:, :, pose_index] = v2t_6(dxr) @ XR[:, :, pose_index]

    for landmark_index in range(num_landmarks):
        # landmark_matrix_index = num_poses * 6 + landmark_index * 3
        landmark_matrix_index = landmarkMatrixIndex(landmark_index, num_poses, num_landmarks)
        dxl = dx[landmark_matrix_index:landmark_matrix_index + landmark_dim, :]
        XL[:, landmark_index] += dxl

    return XR, XL


############################ INDICES ######################################
# Retrieves the index in the perturbation vector, that corresponds to
# a certain pose
# Input:
#   pose_index or landmark_index:   the index of the pose for which we want
#                                   to compute the index
#   num_poses:      number of pose variables in the state
#   num_landmarks:  number of pose variables in the state
# Output:
#   v_idx: the index of the sub-vector corrsponding to
#          pose_index, in the array of perturbations  (-1 if error)
def poseMatrixIndex(pose_index, num_poses, num_landmarks):
    if (pose_index>num_poses):
        return -1
    indexPose = pose_index * pose_dim
    return indexPose

def landmarkMatrixIndex(landmark_index, num_poses, num_landmarks):
    if (landmark_index>num_landmarks):
        return -1
    indexLand = num_poses * pose_dim + landmark_index * landmark_dim
    return indexLand
