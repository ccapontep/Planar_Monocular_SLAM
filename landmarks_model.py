
"""
@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Helpful Tool Functions Setup
"""

from funcTools import *

def landmark_model(t, rob_poseH, land_triang, land_uv, meas_lpoint, state_to_id_map, projection_associations, landmark_associations):

    # calculate number of poses and landmarks
    num_poses = rob_poseH.shape[2]
    if land_triang[0,0] != -1:
        num_landmarks = land_triang.shape[1]
    else: num_landmarks = 0

    # count the amount of measurements
    meas_items = len(meas_lpoint) - 1

    # Get landmarks ids that are currently measured
    meas_land = list()
    for i in range(len(meas_lpoint)):
        meas_land.append(meas_lpoint[i][0][1])

    # Get landmarks added to the state
    state_land = list(state_to_id_map[0:num_landmarks,0])

    # Create association for each landmark and add their u,v values
    for i in range(len(state_land)):
        landmark_id = state_to_id_map[i,0]
        landmark_asso = i
        if landmark_id in meas_land:
            k = meas_land.index(landmark_id)

            u = meas_lpoint[k][1][0]
            v = meas_lpoint[k][1][1]

            if projection_associations[0,0] == -1:
                land_uv = np.array([u, v, landmark_asso]).reshape(3,1)
                projection_associations =  np.array([t, landmark_asso, landmark_id]).reshape(3,1)
                landmark_associations = np.array([t, 0]).reshape(2,1)
            else:
                land_uv = np.hstack([land_uv, np.array([u, v, landmark_asso]).reshape(3,1)])
                projection_associations = np.hstack([projection_associations, np.array([t, landmark_asso, landmark_id]).reshape(3,1)])
                landmark_associations = np.hstack([landmark_associations, np.array([t, landmark_asso]).reshape(2,1)])


    return(num_poses, num_landmarks, land_uv, projection_associations,landmark_associations)
