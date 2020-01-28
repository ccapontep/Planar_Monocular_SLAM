
"""
@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Helpful Tool Functions Setup
"""

from funcTools import *
import math
from numpy.linalg import inv


# error and jacobian of a measured landmark
# input:
#   Xr: the robot pose in world frame (4x4 homogeneous matrix)
#   Xl: the landmark pose (3x1 vector, 3d pose in world frame)
#   z:  measured position of landmark
# output:
#   e: 3x1 is the difference between prediction and measurement
#   Jr: 3x6 derivative w.r.t a the error and a perturbation on the
#       pose
#   Jl: 3x3 derivative w.r.t a the error and a perturbation on the
#       landmark

def landmarkErrorAndJacobian(Xr,Xl,z):
    # inverse transform
    iR = np.transpose(Xr[0:3,0:3])
    it = -iR @ Xr[0:3,3].reshape(3,1)

    #prediction
    loc_rob = iR @ Xl + it
    e = loc_rob - z

    Jr = np.zeros((3,6))
    Jr[0:3,0:3] = -iR
    Jr[0:3,3:6] = iR @ skew(Xl)
    Jl = iR

    return(e,Jr,Jl)



#linearizes the robot-landmark measurements
#   XR: the initial robot poses (4x4xnum_poses: array of homogeneous matrices)
#   XL: the initial landmark estimates (3xnum_landmarks matrix of landmarks)
#   Z:  the measurements (3xnum_measurements)
#   associations: 2xnum_measurements.
#                 associations(:,k)=[p_idx,l_idx]' means the kth measurement
#                 refers to an observation made from pose p_idx, that
#                 observed landmark l_idx
#   num_poses: number of poses in XR (added for consistency)
#   num_landmarks: number of landmarks in XL (added for consistency)
#   kernel_threshod: robust kernel threshold
# output:
#   XR: the robot poses after optimization
#   XL: the landmarks after optimization
#   chi_stats: array 1:num_iterations, containing evolution of chi2
#   num_inliers: array 1:num_iterations, containing evolution of inliers

def linearizeLandmarks(rob_poseH, rob_poseH_gt, land_triang, land_triang_gt, state_size, num_poses, num_landmarks, kernel_threshold_land, landmark_associations):

    num_landmark_measurements = num_poses*num_landmarks
    Zl = np.ones((3,num_landmark_measurements))*-1
    # landmark_associations = np.zeros((2,num_landmark_measurements)).astype(int)

    measurement_num = 0
    for pose_num in range(num_poses):
        Xr = np.linalg.inv(rob_poseH_gt[:,:,pose_num])
        for landmark_num in range(num_landmarks):
            Xl = land_triang_gt[:,landmark_num].reshape(3,1)
            # landmark_associations[:,measurement_num] = [pose_num, landmark_num]
            Zl[:,measurement_num] = (Xr[0:3,0:3] @ Xl + Xr[0:3,3].reshape(3,1)).reshape(3,)
            measurement_num += 1

    # print('landmark_associations \n', landmark_associations)

    H = np.zeros([state_size, state_size])
    b = np.zeros([state_size,1])

    chi_tot = 0
    num_inliers = 0

    if Zl[0,0] != -1:
        for measurement_num in range(Zl.shape[1]):
            pose_index = landmark_associations[0,measurement_num]
            landmark_index = landmark_associations[1,measurement_num]
            z = Zl[:, measurement_num].reshape(3,1)
            Xr = rob_poseH[:, :, pose_index]
            Xl = land_triang[:, landmark_index]
            e, Jr, Jl = landmarkErrorAndJacobian(Xr, Xl, z)
            chi = (e.transpose() @ e)[0,0]
        if chi > kernel_threshold_land:
            e = e @ math.sqrt(kernel_threshold_land/chi)
            chi = kernel_threshold_land
        else:
            num_inliers += 1
        chi_tot += chi

    pose_matrix_index = 6 * pose_index
    landmark_matrix_index = 6 * num_poses + 3 * landmark_index

    H[pose_matrix_index:pose_matrix_index+6,
      pose_matrix_index:pose_matrix_index+6] += np.transpose(Jr) @ Jr

    H[pose_matrix_index:pose_matrix_index+6,
      landmark_matrix_index:landmark_matrix_index+3] += np.transpose(Jr) @ Jl

    H[landmark_matrix_index:landmark_matrix_index+3,
      landmark_matrix_index:landmark_matrix_index+3] += np.transpose(Jl) @ Jl

    H[landmark_matrix_index:landmark_matrix_index+3,
      pose_matrix_index:pose_matrix_index+6] += np.transpose(Jl) @ Jr

    b[pose_matrix_index:pose_matrix_index+6] += np.transpose(Jr) @ e
    b[landmark_matrix_index:landmark_matrix_index+3] += np.transpose(Jl) @ e

    return(H,b, chi_tot, num_inliers)
