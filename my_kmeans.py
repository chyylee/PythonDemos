#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:01:46 2022

@author: chlee

My kmeans algorithm. Modified from a tutorial by Dr J Rogel-Salazar.

This script contains several functions:
    centroids = initiate_centroids(k,data)
        'centroids': DataFrame with centroids
    err = rsserr(a,b)
        'err' = root sum of squared error
    assignation, assign_errors = centroid_assignation(data,centroids)
        'assignation': list of assigned centroid
        'assign_errors': list of errors corresponding to the assigned centroid
    centroid, err, centroids =kmeans(data, k=2, tol=1e-4)
        'centroid': pandas Series of assigned centroid for each point
        'err': list of error for each point
        'centroids': DataFrame of final centroids for each clustter
        
"""

import numpy as np
import pandas as pd

def initiate_centroids(k, data):
    '''
    Define initial centroids using random sampling
    k: number of centroids
    data: pandas dataframe with input data
    '''
    centroids = data.sample(k)
    return centroids

def rsserr(a,b):
    '''
    Calculate the root of sum of squared errors. 
    a: numpy array
    b: numpy array
    '''
    return np.square(np.sum((a-b)**2)) 

def centroid_assignation(data, centroids):
    '''
    Given a dataframe `data` and a set of `centroids`, we assign each
    data point in `data` to a centroid. 
    - data: pandas dataframe with observations
    - centroids: pandas dataframe with centroids (returned by initiate_centroids)
    '''
    k = centroids.shape[0]
    n = data.shape[0]
    assignation = []
    assign_errors = []

    # Loop through each data point and find centroid with minimal error (rsserr)
    for obs in range(n):
        # Estimate error
        all_errors = np.array([])
        for centroid in range(k):
            err = rsserr(centroids.iloc[centroid, :], data.iloc[obs,:])
            all_errors = np.append(all_errors, err)

        # Get the nearest centroid and the error
        nearest_centroid =  np.where(all_errors==np.amin(all_errors))[0].tolist()[0]
        nearest_centroid_error = np.amin(all_errors)

        # Add values to corresponding lists
        assignation.append(nearest_centroid)
        assign_errors.append(nearest_centroid_error)

    return assignation, assign_errors

# The full kmeans algorithm
def kmeans(data, k=2, tol=1e-4):
    '''
    K-means implementationd for a 
    `data`:  DataFrame with observations
    `k`: number of clusters, default k=2
    `tol`: error tolerance, default tolerance=1E-4
    '''
    # Let us work in a copy, so we don't mess the original
    working_dset = data.copy()
    # We define some variables to hold the error, the 
    # stopping signal and a counter for the iterations
    err = []
    goahead = True
    j = 0
    
    # Step 2: Initiate clusters by defining centroids 
    centroids = initiate_centroids(k, data)

    while(goahead):
        # Step 3 and 4 - Assign centroids and calculate error
        working_dset['centroid'], j_err = centroid_assignation(working_dset, centroids) 
        err.append(sum(j_err))
        
        # Step 5 - Update centroid position
        centroids = working_dset.groupby('centroid').agg('mean').reset_index(drop = True)

        # Step 6 - Restart the iteration
        if j>0:
            # Is the error less than a tolerance (1E-4)
            if err[j-1]-err[j]<=tol:
                goahead = False
        j+=1

    working_dset['centroid'], j_err = centroid_assignation(working_dset, centroids)
    
    # Create new centroids
    centroids = working_dset.groupby('centroid').agg('mean').reset_index(drop = True)
    return working_dset['centroid'], j_err, centroids