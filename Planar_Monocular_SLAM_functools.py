#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 17:21:44 2019

@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Helpful Tool Functions Setup
"""

# Import libraries
import numpy as np
import math
from mpmath import mp
mp.dps = 8 # precision for rounding the trig functions


###############################################################################

#                           HELPFUL TOOL FUNCTIONS SETUP

###############################################################################


# This function converts a perturbation into a matrix
# computes the homogeneous transform matrix H of the pose vector v
# H:[ R t ] 3x3 homogeneous transformation matrix, r translation vector
# v: [x,y,theta]  2D pose vector

def v2t(v, v3, th):
    c = mp.cos(th)
    s = mp.sin(th)
    H = np.matrix([[c, -s, 0, v[0,0]],
                    [s, c, 0, v[1,0]],
                    [0, 0, 1, v3    ],
                    [0, 0, 0, 1     ]], dtype='float64')
    return(H)
    
# computes the pose 2d pose vector v from an homogeneous transform H
# H:[ R t ] 3x3 homogeneous transformation matrix, r translation vector
# v: [x,y,theta]  2D pose vector
def t2v(H):
    v = np.zeros((3, 1))
    v[0:2] = H[0:2,2]
    v[2,0] = math.atan2(H[1,0], H[0,0])
    
    return(v)
    
# computes the pose 2d pose vector v from an homogeneous transform H
# H:[ R t ] 3x3 homogeneous transformation matrix, r translation vector
# v: [x,y]  2D position vector
def t2v2(H):
    v = np.zeros((2, 1))
    v[0:2] = H[0:2,2]
    
    return(v)

# normalizes and angle between -pi and pi
# theta: input angle
# out: output angle
def normalizeAngle(theta):
    out = math.atan2(math.sin(theta),math.cos(theta))
    return(out)
    
# This funciion is the boxplus for matrices in homogenuous transform
def box_plus(N, m):
    M = v2t(m)
    out = M * N
    return(out)
    
# This funciion is the boxplus for matrices in homogenuous transform
def box_minus(N, M):
    out = t2v(np.transpose(M)*N)
    return(out)
    

###############################################################################

#                          END HELPFUL TOOL FUNCTIONS SETUP

###############################################################################





