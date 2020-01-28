
"""
@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Add New Landmark Function Setup
"""

# Import libraries
import numpy as np

from data import get_all_data
from uv_to_xy import uv_to_xy


# This function computes gets the new landmark that has not been previously seen
#inputs:
#  rob_poseH: mean,
#  sigma: covariance of the robot-landmark set (x,y, theta, l_1, ..., l_N)
#
#  observations:
#            a structure containing n observations of landmarks
#            for each observation we have
#            - the index of the landmark seen
#            - the location where we have seen the landmark (x,y) w.r.t the robot
#
#  id_to_state_map:
#            mapping that given the id of the measurement, returns its position in the rob_poseH vector
#  state_to_id_map:
#            mapping that given the index of rob_poseH vector, returns the id of the respective landmark
#
#outputs:
#  [rob_poseH, sigma]: the updated mean and covariance
#  [id_to_state_map, state_to_id_map]: the updated mapping vector between landmark position in rob_poseH vector and its id

def newlandmark(t, world_data, rob_poseH, land_triang, land_triang_gt, observations, id_to_state_map, state_to_id_map, rob_update, robot_pose_map, robot_gt_pose_map, land_id, Land_TriangPrev):

    # Get how many landmarks have been seen in current step
    M = len(observations)

    # To check that the id_to_state_map is being set correctly (not
    # repeating the same id number)
    from collections import Counter
    a = Counter(list(id_to_state_map[:,0]))
    # Current number of landmarks in the state
    # if the first time, start at n = 0, else check the id map
    if t == 0:
        n = int(len(land_triang[0])) -1 # number of columns which repr # landmarks
    else:
        n = int(max(list(a.items()))[0])


    # First of all we are interested in REOBSERVED landmark
    for i in range(M):

        # Get info about each observed landmark
        measurement = observations[i]
        meas_land = observations[i][0][2] # current landmark number

        # To check that the id_to_state_map is being set correctly (not
        # repeating the same id number)
        b = Counter(list(id_to_state_map[:,4]))
        # Current number of landmarks in the state
        # if the first time, start at n = 0, else check the id map
        if t == 0:
            q = int(len(land_triang[0])) -1
        else:
            q = int(max(list(b.items()))[0])

        #fetch the value in the state vector corresponding to the actual measurement
        state_pos_landmark_1 = id_to_state_map[meas_land, 0]
        state_pos_landmark_2 = id_to_state_map[meas_land, 4]

        # Temp for correct location of landmark from world_data
        l_loc = world_data[meas_land][1:4]
        landmark_loc = np.array([l_loc[0], l_loc[1], l_loc[2]]).reshape(3,1)

        # Get the measurement col, rho of each landmark (u, v, 1)
        u = measurement[1][0]
        v = measurement[1][1]


### For the landmark seen for the first time ###########

        # If current landmark is observed by the first time
        if state_pos_landmark_1 == -1 and state_pos_landmark_2 == -1:

            # Get new values for landmark id and set them to the mappings of id and state
            n += 1
            id_to_state_map[meas_land, 0] = n

            # If it is the first time seeing this n landmark:
            id_to_state_map[meas_land, 1] = t # add the time step of when it was seen
            id_to_state_map[meas_land, 2] = u # add the corresponding u
            id_to_state_map[meas_land, 3] = v # add the corresponding v


############# DELAYED INITIALIZATION DUE TO BEARING-ONLY ######################
# This section is for the collection of information before a landmark is
# initialized, since one angle is not enough to describe the position of the
# landmark. Therefore we collect as many measurements to calculate the
# landmark location and initialize it in the state vector.



        # Else, if it is the second time observing the same landmark:
        elif state_pos_landmark_2 == -1 and state_pos_landmark_1 != -1:

            id_to_state_map[meas_land, 5] = t # add new time step
            id_to_state_map[meas_land, 6] = u # add new u
            id_to_state_map[meas_land, 7] = v # add new v

            # Convert uv coordinate to xy
            landmark_pos_world, id_to_state_map, alpha, beta, b, rho, gamma, dist2land = uv_to_xy(t, u, v, meas_land, id_to_state_map, rob_poseH, robot_gt_pose_map, robot_pose_map, rob_update)

            # If alpha is less than 5.7 degrees, erase the data of observation.
            # Will have to wait to the next time to reobserve the landmark in
            # order to be initialized.
            # https://core.ac.uk/download/pdf/41760856.pdf

            if alpha >= 0.1 and id_to_state_map[meas_land, 8] < 1:
                id_to_state_map[meas_land, 8] += 1


            if alpha < 0.1:
                # Erase data of this information, so it can get new ones until initiation.
                id_to_state_map[meas_land, 4:8] = -1

            # If alpha is greater than 5 degrees, initiate the landmark.
            elif alpha >= 0.1 and id_to_state_map[meas_land, 8] == 1:
                q += 1
                id_to_state_map[meas_land, 4] = q
                landmark_pos_world, id_to_state_map, alpha, beta, b, rho, gamma, dist2land = uv_to_xy(t, u, v, meas_land, id_to_state_map, rob_poseH, robot_gt_pose_map, robot_pose_map, rob_update)

                # Add the landmark to the state map
                state_to_id_map[q, 0] = meas_land

                # error_pred_gt = landmark_loc - landmark_pos_world[0:3,0]

                landmark_pos_world = landmark_pos_world[0:3,0]
                #retrieve from the index the position of the landmark block in the state
                g = id_to_state_map[meas_land, 4] # get the index in state
                id_state = int(3*g)

                #adding the landmark state to the full state
                if id_state == 0:
                    land_triang = landmark_pos_world
                    land_triang_gt = landmark_loc
                    land_id.append(meas_land)
                    Land_TriangPrev = landmark_pos_world
                else:
                    land_triang = np.hstack([land_triang, landmark_pos_world])
                    land_triang_gt = np.hstack([land_triang_gt, landmark_loc])
                    land_id.append(meas_land)
                    Land_TriangPrev = np.hstack([Land_TriangPrev, landmark_pos_world])


    return (land_triang, land_triang_gt, id_to_state_map, state_to_id_map, land_id, Land_TriangPrev)
