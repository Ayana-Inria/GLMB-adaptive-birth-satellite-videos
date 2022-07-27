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


def main(parameters):
    print("=========================================")
    print(" MAIN FUNCTION ")
    print("AOI: {} {}.".format(parameters["AOI_number"], parameters["AOI"]))
    print("=========================================")

    # FILTER
    print("Starting Filter")
    filter_fn.perform_adaptive_birth_glmb_filter(parameters)

    # TRAJECTORIES
    print("Saving Ground Truth Trajectories")
    data_reader.save_detection_with_trajecories_smart(parameters, GT_trajectories=True, trajectory_length=8)
    
    print("Saving Filter Trajectories")
    data_reader.save_detection_with_trajecories_smart(parameters, GT_trajectories=False, trajectory_length=8)


    # METRICS
    print("Calculating Metrics")
    metrics.calculate_metrics_and_plot(parameters, ellapsed_time=0, c=5)

    # Done
    print("Done")


if __name__ == '__main__':
    '''
        filter_type:
            0: GM-PHD
            1: SORT
            2: GLMB
            3: Hybrid GLMB PHD

    '''
    parameters = get_dataset_parameters(AOI_number=1, filter_type=1)
    main(parameters)
