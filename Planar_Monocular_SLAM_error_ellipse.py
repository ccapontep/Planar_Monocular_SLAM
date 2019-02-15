#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 17:21:44 2019

@author: Cecilia Aponte
Probabilistic Robotics
Final Project
Planar Monocular SLAM -- Error Ellipse Setup
"""

# Import libraries
import numpy as np
import math



###############################################################################

#                           ERROR ELLIPSE SETUP

###############################################################################

# This function calculates the error to create a 2D ellipse of the error
def error_ellipse(sigma):
   
    # Sort the eigenvalue by highest to lowest value to draw correctly
    #    https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
    #    https://www.soest.hawaii.edu/martel/Courses/GG303/Lab.09.2017.pptx.pdf -- slide 20
    
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        vals_sor, vecs_sor = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals_sor, vecs_sor = vals[order], vecs[:, order] 
        return vals, vecs, vals_sor, vecs_sor
    
    eigenvalues, vecs, vals_sor, vecs_sor = eigsorted(sigma[0:2, 0:2])
#    print('eigenvalues, vecs, vals_sor, vecs_sor', eigenvalues, vecs, vals_sor, vecs_sor)
    #   eigen function returns vectors as: column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    theta = np.degrees(np.arctan2(vecs_sor[1,0], vecs_sor[0,0])) # angle direction of error
    #   To find the error of mu to draw ellipse by: 2 * nstd * np.sqrt(vals)
    #   where nstd is the std deviation, in this case 1 std
    eigen_error = np.array([2*math.sqrt(eigenvalues[0]), 2*math.sqrt(eigenvalues[1]), 
                      theta])
    
    return(eigen_error)



