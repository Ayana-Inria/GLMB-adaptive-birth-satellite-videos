#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Camilo Aguilar
# Created Date: 27/07/2002
# version ='1.0'
# ---------------------------------------------------------------------------
""" 
    main.py:
        Main function for object detection in the paper:
"""
import numpy as np
import time
import os

# Parameters 
from parameters import get_dataset_parameters

# Data reader
import src.data_reader.data_reader as data_reader

import src.filter_main as filter_fn


# Metrics
# import src.evaluations.metrics as metrics

# import argparse


def main():
    print("=========================================")
    print(" MAIN FUNCTION ")
    print("=========================================")

    parameters = get_dataset_parameters(AOI_number=2)

    # FILTER
    print("Starting Filter")
    filter_fn.perform_adaptive_birth_glmb_filter(parameters)

    
    print("Saving Filter Trajectories")
    data_reader.save_detection_with_trajecories_smart(parameters, GT_trajectories=False, trajectory_length=8)

    # Trajectory Plotting
    print("Saving Ground Truth Trajectories")
    try:
        data_reader.save_detection_with_trajecories_smart(parameters, GT_trajectories=True, trajectory_length=8)
    except IndexError:
        print("Need all 512 images to plot all the gt trajectories")

    # METRICS
    print("Calculating Metrics")
    data_reader.calculate_metrics(parameters, c=5)

    # Done
    print("Done")


if __name__ == '__main__':
    main()
