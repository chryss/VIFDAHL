#!/usr/bin/env python
# coding=utf-8
"""
Helper functions for processing VIIRS HDF5 files: handling swath and
geolocation data

Chris Waigl, 2016-03-01
"""

import numpy as np
from functools import reduce

def get_zeros(viirscene, imin=0, imax=None, jmin=0, jmax=None):
    """Identify jumps in in-track latitudes of pygaarst.HDF5.VIIRSHDF5 scene"""
    lats = viirscene.lats[imin:imax, jmin:jmax] 
    deltas = lats[:-1, :] - lats[1:, :]
    if viirscene.ascending_node:
        deltas[deltas > 0] = 0
    else:
        deltas[deltas < 0] = 0

    right = np.where(deltas[:, 0] == 0)[0]
    left = np.where(deltas[:, -1] == 0)[0]
    middle = np.where(deltas[:, deltas.shape[1]//2] == 0)[0]
    return reduce(np.union1d, (right, left, middle))

def get_skips_by_col(col, zeros=[], length=None, ascending=True):
    """"""
    if not length:
        length = col.shape[0]
    if len(zeros) == 0:
        return np.zeros(length)
    skipindices = []
    outcol = np.zeros(length)
    if length < np.max(np.max(zeros)):
        print("*** ERROR in function get_skips_by_col: index out of range")
        return
    fac = 1.
    if not ascending:
        fac = -1.
    for idx in zeros:
        step = np.argmax(fac*col[idx:] > fac*col[idx])
        if step == 0:
            skipindices.extend(range(idx+1, length))
        else:
            halfstepL = step//2
            if halfstepL > idx:
                halfstepL == idx
            halfstepR = step - halfstepL
            skipindices.extend(range(idx-halfstepL+1, idx+halfstepR))
    outcol[skipindices] = 1.
    return outcol

def get_skips(viirsscene, imin=0, imax=None, jmin=0, jmax=None):
    return np.apply_along_axis(get_skips_by_col, 
            0, 
            viirsscene.lats[imin:imax, jmin:jmax],
            zeros=get_zeros(viirsscene, imin=imin, imax=imax, jmin=jmin, jmax=jmax), 
            length=imax-imin if imax else viirsscene.lats.shape[0]-imin,
            ascending=viirsscene.ascending_node)

def get_badrows(rst, anomvalue=1.):
    right = np.where(rst[:, 0] == anomvalue)[0]
    left = np.where(rst[:, -1] == anomvalue)[0]
    middle = np.where(rst[:, rst.shape[1]//2] == anomvalue)[0]
    return reduce(np.union1d, (right, left, middle))
    
def is_nightscene(viirsscene, imin=0, imax=None, jmin=0, jmax=None):
    nightscene = True
    zenithangles = viirsscene.geodata['SolarZenithAngle'][imin:imax, jmin:jmax]
    if np.all(zenithangles <= 90.):
        nightscene = False
    elif (np.any(zenithangles > 90.) & np.any(zenithangles <= 90.)):
        print(u"WARNING: mixed scene with zenith angles above and below 90°.")
    return nightscene

def get_corners_wgs_from_latlon(i, j, lons, lats):
    import pyproj
    g = pyproj.Geod(ellps='WGS84')    
    corners = []
    maxi, maxj = lons.shape
    deli1 = 1 if i < maxi-1 else 0
    deli2 = -1 if i > 0 else 0
    delj1 = 1 if j < maxj-1  else 0
    delj2 = -1 if j > 0 else 0
    cornerlist = [(deli1, delj1), (deli2, delj1), (deli2, delj2), (deli1, delj2)]
    for deli, delj in cornerlist:
        fwd_az, bck_az, dist = g.inv(
            lons[i, j], lats[i, j], 
            lons[i+deli, j+delj], lats[i+deli, j+delj])
        newlon, newlat, bck_az = g.fwd(
            lons[i, j], lats[i, j] , fwd_az, dist/2.)
        corners.append((newlon, newlat))
    return corners

def get_corners_wgs(i, j, scene):
    lons = scene.lons
    lats = scene.lats
    return get_corners_wgs_from_latlon(i, j, lons, lats)

def get_corners_simple_from_latlon(i, j, lons, lats):
    corners = []
    cornerlist = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    for deli, delj in cornerlist:
        newlon = (lons[i+deli, j+delj] + lons[i, j])/2
        newlat = (lats[i+deli, j+delj] + lats[i, j])/2
        corners.append((newlon, newlat))
    return corners

def get_corners_simple(i, j, scene):
    lons = scene.lons
    lats = scene.lats
    return get_corners_simple_from_latlon(i, j, lons, lats)
