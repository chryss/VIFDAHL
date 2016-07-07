#!/usr/bin/env python
# coding=utf-8
"""
Generate fire product from Landsat 8 data
as per
Schroeder, W., et al., Active fire detection using Landsat-8/OLI data,
Remote Sensing of Environment (2015),
http://dx.doi.org/10.1016/j.rse.2015.08.032

Also contains a nice water mask.

Chris Waigl, 2016-07-01
"""

from __future__ import print_function, unicode_literals
import numpy as np

def get_l8fire(landsatscene, mask=None, high_only=False, anomalies=True):
    """
    Takes L8 scene, returns 2D Boolean numpy array of same shape
    """
    rho = get_reflectances(landsatscene, mask=mask)
    allfire, unambiguous, anomalous, marginal = get_l8fire_frombands(
        rho['1'], rho['2'], rho['3'], rho['4'],
        rho['5'], rho['6'], rho['7'],
        high_only=high_only, anomalies=anomalies)
    return allfire, unambiguous, anomalous, marginal

def get_l8fire_frombands(rho1, rho2, rho3, rho4,
                          rho5, rho6, rho7,
                          high_only=False, anomalies=True):
    """rho[n] is the reflectance in band n"""
    R75 = rho7/rho5

    firecond1 = get_unambiguousfire(rho5, rho7, R75)
    firecond2 = None
    firecond3 = None

    if anomalies:
        firecond2 = get_anomalousfire(rho1, rho5, rho6, rho7)

    if not high_only:
        firecond3 = get_marginalfire(rho5, rho7, R75)
        # check if marginal candidates are actual fire pixels
        R76 = rho7/rho6
        firecond3 = np.logical_and(~firecond1, firecond3)
        firecond3 = np.logical_and(~firecond2, firecond3)
        firecond3 = np.logical_and(firecond3, R76 > 0)
        if not np.all(firecond2):
            otherfirecond = firecond1
        else:
            otherfirecond = np.logical_or(
                firecond1, firecond2
            )
        firecond3 = get_verified_fires(
            firecond3, otherfirecond, rho1, rho2,
            rho3, rho4, rho5, rho6, rho7)
    allfires = np.logical_or(firecond1, firecond2)
    allfires = np.logical_or(allfires, firecond3)
    return allfires, firecond1, firecond2, firecond3


def get_unambiguousfire(rho5, rho7, R75):
    """Boolean 2D array of unambiguous fire pixels"""
    firecond1 = np.logical_and(R75 > 2.5, rho7 > .5)
    return np.logical_and(firecond1, rho7 - rho5 > .3)

def get_anomalousfire(rho1, rho5, rho6, rho7):
    """Boolean 2D array of anomalous (sensor-saturation) fire pixels"""
    firecond2 = np.logical_and(rho6 > .8, rho1 < .2)
    firecond2 = np.logical_and(firecond2, np.logical_or(rho5 > .4, rho7 < .1))
    return firecond2

def get_marginalfire(rho5, rho7, R75):
    """Boolean 2D array of marginal/questionable fire pixels"""
    return np.logical_and(R75 > 1.8, rho7 - rho5 > .17)

def get_verified_fires(firecond3, otherfirecond,
                       rho1, rho2, rho3,
                       rho4, rho5, rho6, rho7):
    """Boolean 2D array of verified marginal fire pixels"""
    iidxmax, jidxmax = firecond3.shape
    output = np.zeros((iidxmax, jidxmax))

    windows = [get_window(ii, jj, 30, iidxmax, jidxmax)
               for ii, jj in np.argwhere(firecond3)]
    window = np.any(windows, axis=0)
    validwindow = get_valid_pixels(otherfirecond, rho1, rho2, rho3,
                                   rho4, rho5, rho6, rho7,
                                   mask=~window)

    for ii, jj in np.argwhere(firecond3):
        window = get_window(ii, jj, 30, iidxmax, jidxmax)
        newmask = np.logical_or(~window, ~validwindow.data)
        rho7_win = np.ma.masked_array(rho7, mask=newmask)
        R75_win = np.ma.masked_array(rho7/rho5, mask=newmask)
        rho7_bar = np.mean(rho7_win.flatten())
        rho7_std = np.std(rho7_win.flatten())
        R75_bar = np.mean(R75_win.flatten())
        R75_std = np.std(R75_win.flatten())
        rho7_test = rho7_win[ii, jj] - rho7_bar > max(3*rho7_std, 0.08)
        R75_test = R75_win[ii, jj]- R75_bar > max(3*R75_std, 0.8)
        if rho7_test and R75_test:
            output[ii, jj] = 1
    return output == 1

def get_valid_pixels(otherfirecond, rho1, rho2, rho3,
                     rho4, rho5, rho6, rho7, mask=None):
    """returns masked array of 1 for valid, 0 for not"""
    if not np.any(mask):
        mask = np.zeros(otherfirecond.shape)
    rho = {}
    for rho in [rho1, rho2, rho3, rho4, rho5, rho6, rho7]:
        rho = np.ma.masked_array(rho, mask=mask)
    watercond = get_l8watermask_frombands(
        rho1, rho2, rho3,
        rho4, rho5, rho6, rho7)
    greater0cond = rho7 > 0
    finalcond = np.logical_and(greater0cond, ~watercond)
    finalcond = np.logical_and(finalcond, ~otherfirecond)
    return np.ma.masked_array(finalcond, mask=mask)

def get_window(ii, jj, N, iidxmax, jidxmax):
    """Return 2D Boolean array that is True where a window of size N
    around a given point is masked out """
    imin = max(0, ii-N)
    imax = min(iidxmax, ii+N)
    jmin = max(0, jj-N)
    jmax = min(jidxmax, jj+N)
    mask1 = np.zeros((iidxmax, jidxmax))
    mask1[imin:imax+1, jmin:jmax+1] = 1
    return mask1 == 1

def get_l8watermask(landsatscene, mask=None):
    """
    Takes L8 scene, returns 2D Boolean numpy array of same shape
    """
    rho = get_reflectances(landsatscene, mask=mask)
    return get_l8watermask_frombands(rho['1'], rho['2'], rho['3'], rho['4'],
                                     rho['5'], rho['6'], rho['7'])

def get_l8watermask_frombands(
        rho1, rho2, rho3,
        rho4, rho5, rho6, rho7):
    """
    Takes L8 bands, returns 2D Boolean numpy array of same shape
    """
    turbidwater = get_l8turbidwater(rho1, rho2, rho3, rho4, rho5, rho6, rho7)
    deepwater = get_l8deepwater(rho1, rho2, rho3, rho4, rho5, rho6, rho7)
    return np.logical_or(turbidwater, deepwater)

def get_l8commonwater(rho1, rho4, rho5, rho6, rho7):
    """Returns Boolean numpy array common to turbid and deep water schemes"""
    water1cond = np.logical_and(rho4 > rho5, rho5 > rho6)
    water1cond = np.logical_and(water1cond, rho6 > rho7)
    water1cond = np.logical_and(water1cond, rho1 - rho7 < 0.2)
    return water1cond

def get_l8turbidwater(rho1, rho2, rho3, rho4, rho5, rho6, rho7):
    """Returns Boolean numpy array that marks shallow, turbid water"""
    watercond2 = get_l8commonwater(rho1, rho4, rho5, rho6, rho7)
    watercond2 = np.logical_and(watercond2, rho3 > rho2)
    return watercond2

def get_l8deepwater(rho1, rho2, rho3, rho4, rho5, rho6, rho7):
    """Returns Boolean numpy array that marks deep, clear water"""
    watercond3 = get_l8commonwater(rho1, rho4, rho5, rho6, rho7)
    watercondextra = np.logical_and(rho1 > rho2, rho2 > rho3)
    watercondextra = np.logical_and(watercondextra, rho3 > rho4)
    return np.logical_and(watercond3, watercondextra)

def get_reflectances(landsatscene, mask=None):
    """Returns dictionary of 2D reflectance arrays with optional global mask"""
    if not np.any(mask):
        mask = np.zeros(landsatscene.band1.data.shape)
    rho = {}
    for ii in range(1, 8):
        rho[str(ii)] = np.ma.masked_array(landsatscene.__getattr__(
            'band' + str(ii)).reflectance, mask=mask)
    return rho
