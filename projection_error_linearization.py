
"""
@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Transition Function Setup
"""

from funcTools import *
import math

# Error and jacobian of a measured landmark
# Input:
#   rob_poseH: the robot pose in world frame (4x4 homogeneous matrix)
#   land_triang: the landmark pose (3x1 vector, 3d pose in world frame)
#   z:  projection of the landmark on the image plane
# Output:
#   e: 2x1 is the difference between prediction and measurement
#   Jr: 2x6 derivative w.r.t a the error and a perturbation on the
#       pose
#   Jl: 2x3 derivative w.r.t a the error and a perturbation on the
#       landmark
#   is_valid: true if projection ok
def projectionErrorAndJacobian(rob_poseH,land_triang,z):

    # Inverse transform
    rob_cam = rob_poseH @ cam_transform
    rob_inv = rob_cam[0:3,0:3].transpose()
    it = -rob_inv @ rob_cam[0:3,3]

    # Point in the camera
    loc_rob = rob_inv @ land_triang + it
    loc_cam = cam_matrix @ loc_rob
    loc_img = loc_cam[0:2,0] / loc_cam[2,0]

    # Check if point is in the field of view of the camera
    if z[0,0] == -1e7 or loc_rob[2,0]<z_near or loc_rob[2,0]>z_far or loc_img[0,0]<0 or loc_img[0,0]>cam_width or loc_img[1,0]<0 or loc_img[1,0]>cam_height:
        is_valid = False
        return -1, -1, -1, is_valid
    else:
        # Derivative of loc_cam wrt dx (Robot and Landmark)
        Jwr = np.zeros([3,6])
        Jwr[0:3, 0:3] = -rob_inv
        Jwr[0:3, 3:6] = rob_inv @ skew(land_triang)
        Jwl = rob_inv

        Jp = np.matrix([[1/loc_cam[2,0],  0, -loc_cam[0,0]/loc_cam[2,0]**2],
                       [0,  1/loc_cam[2,0], -loc_cam[1,0]/loc_cam[2,0]**2]])

        # Projection error
        proj_error = (loc_img - z).reshape([-1, 1]) #.reshape([-1, 1])

        Jr = Jp @ cam_matrix @ Jwr
        Jl = Jp @ cam_matrix @ Jwl
        is_valid = True

    return proj_error, Jr, Jl, is_valid

# Linearizes the robot-landmark measurements
# Input:
#   rob_poseH: the initial robot poses (4x4xnum_poses: array of homogeneous matrices)
#   land_triang: the initial landmark estimates (3xnum_landmarks matrix of landmarks)
#   Z:  the measurements (2xnum_measurements)
#   associations: 2xnum_measurements.
#                 associations(:,k)=[p_idx,l_idx]' means the kth measurement
#                 refers to an observation made from pose p_idx, that
#                 observed landmark l_idx
#   num_poses: number of poses in rob_poseH (added for consistency)
#   num_landmarks: number of landmarks in land_triang (added for consistency)
#   kernel_threshod: robust kernel threshold
# Output:
#   rob_poseH: the robot poses after optimization
#   land_triang: the landmarks after optimization
#   chi_stats: array 1:num_iterations, containing evolution of chi2
#   num_inliers: array 1:num_iterations, containing evolution of inliers
def linearizeProjections(rob_poseH, land_triang, land_uv, projection_associations, num_poses, num_landmarks, kernel_threshold_proj):
    state_size = pose_dim * num_poses + landmark_dim * num_landmarks

    H = np.zeros([state_size, state_size])
    b = np.zeros([state_size,1])

    chi_tot = 0
    num_inliers = 0

    for measurement_num in range(land_uv.shape[1]):

        pose_index = projection_associations[0, measurement_num]
        landmark_index = projection_associations[1, measurement_num]

        z = land_uv[0:2, measurement_num].reshape(2,1)

        rob_poseH_curr = rob_poseH[:, :, pose_index]
        land_triang_curr = land_triang[:, landmark_index]

        # Calculate Error and Jacobian
        e, Jr, Jl, is_valid = projectionErrorAndJacobian(rob_poseH_curr, land_triang_curr, z)

        if is_valid:
            chi = e.transpose() @ e
            if chi>kernel_threshold_proj:
                e *= math.sqrt(kernel_threshold_proj / chi)
                chi = kernel_threshold_proj
            else:
                num_inliers += 1
            chi_tot += chi

            # Indices
            pose_matrix_index = poseMatrixIndex(pose_index, num_poses, num_landmarks)
            landmark_matrix_index = landmarkMatrixIndex(landmark_index, num_poses, num_landmarks)

            # omega_proj = np.identity(2)
            # use below for continuous SLAM:
            omega_proj = np.identity(2) * 1e-3

            H[pose_matrix_index:pose_matrix_index+pose_dim,
              pose_matrix_index:pose_matrix_index+pose_dim] += Jr.transpose() @ omega_proj @ Jr

            H[pose_matrix_index:pose_matrix_index+pose_dim,
              landmark_matrix_index:landmark_matrix_index+landmark_dim] += Jr.transpose() @ omega_proj @ Jl

            H[landmark_matrix_index:landmark_matrix_index+landmark_dim,
              landmark_matrix_index:landmark_matrix_index+landmark_dim] += Jl.transpose() @ omega_proj @ Jl

            H[landmark_matrix_index:landmark_matrix_index+landmark_dim,
              pose_matrix_index:pose_matrix_index+pose_dim] += Jl.transpose() @ omega_proj @ Jr

            b[pose_matrix_index:pose_matrix_index+pose_dim] += Jr.transpose() @ omega_proj @ e
            b[landmark_matrix_index:landmark_matrix_index+landmark_dim] += Jl.transpose() @ omega_proj @ e


    return H, b, chi_tot, num_inliers
