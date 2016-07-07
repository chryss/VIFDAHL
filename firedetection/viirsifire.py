#!/usr/bin/env python
# coding=utf-8
"""
Generate fire product from VIIRS I-band data

Chris Waigl, 2016-04-01
"""

import numpy as np

ANOM4_1 = 193
SATT4 = 367.
MINT4 = 320.
MINT5_DAY = 285.
MINT5_SPECIAL = 312.
CLOUDT5_NIGHT = 265.
NORMDIFFTHRESH1 = 0.05
NORMDIFFTHRESH2 = 0.02
NORMDIFFTHRESH3 = 0.015

def getfireconditions_fromrasters(
    i4tb, i5tb, 
    pixq4, pixq5,
    daytime=True,
    mint4= MINT4,
    mint5_day=MINT5_DAY,
    mint5_special=MINT5_SPECIAL):
    """Returns anomalous, high, low boolean fire condition rasters"""
    
    i45 =  (i4tb - i5tb) / (i4tb + i5tb)
    
    anomalouscondition = np.logical_or(
        np.logical_and(pixq4 == ANOM4_1, i4tb < 360),
        np.logical_and(pixq5 == 0, i4tb == SATT4))
    if daytime:
        hotcondition = (i45 >= NORMDIFFTHRESH1)
        hotcondition = np.logical_and(
            np.logical_and(hotcondition, i4tb > mint4), i5tb > mint5_day)
        hotcondition = np.logical_and(
            hotcondition, ~anomalouscondition)
        warmcondition = np.logical_or(
            i45 >= NORMDIFFTHRESH2, np.logical_and(
                i45 >= NORMDIFFTHRESH3, i5tb >= mint5_special))
        warmcondition = np.logical_and(
            warmcondition, i4tb > mint4)
        warmcondition = np.logical_and(
            ~hotcondition, warmcondition)
        warmcondition = np.logical_and(
            warmcondition, i5tb > mint5_day)
    else:
        cloudcondition = (i5tb < CLOUDT5_NIGHT)
        hotcondition = (i45 >= NORMDIFFTHRESH1)
        hotcondition = np.logical_and(~cloudcondition, hotcondition)
        hotcondition = np.logical_and(
            ~anomalouscondition, hotcondition)
        warmcondition = np.logical_and(~hotcondition, i45 >= NORMDIFFTHRESH3)
        warmcondition = np.logical_and(i4tb > mint4, warmcondition)
        warmcondition = np.logical_and(~cloudcondition, warmcondition)
        
    return anomalouscondition, hotcondition, warmcondition