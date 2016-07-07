#!/usr/bin/env python
"""
Helper functions for processing VIIRS HDF5 files in order to 
extract fire information. And for plotting on a map.

Chris Waigl, 2016-03-01
"""

import os
import glob
import re
import datetime as dt
from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Polygon

from pygaarst import raster

earth='cornsilk'
water='lightskyblue'

BANDFILES = {
    u'dnb': ['SVDNB', u'GDNBO'],
    u'iband': [u'SVI01', u'SVI02', u'SVI03', u'SVI04', u'SVI05', u'GITCO'],
    u'mband': [u'SVM01', u'SVM02', u'SVM03', u'SVM04', u'SVM05', 
               u'SVM06', u'SVM07', u'SVM08', u'SVM09', u'SVM10', 
               u'SVM11', u'SVM12', u'SVM13', u'SVM14', u'SVM15', 
               u'SVM16', u'GMTCO'],
}

gisbasedir = "/Volumes/SCIENCE_mobile_Mac/GENERAL_GIS/"
railroads = os.path.join(gisbasedir, "matsugov.us/rr/railroad_latlon")
primaryroads = os.path.join(gisbasedir, "catalog.data.gov/tl_2013_02_prisecroads/tl_2013_02_prisecroads")
sideroads = os.path.join(gisbasedir, "matsugov.us/rds/rds_latlon")
sockeye_lon, sockeye_lat = (-150.08544, 61.84486)
fairbanks_lon, fairbanks_lat = (-147.723056, 64.843611)

def getgoodrows(coordarraysarray, minval=-180., maxval=180.):
    """Returns first and last row idx for which all elements are between min and max
    
    In VIIRS latitude/longitude arrays, -999.23 is used as a fill value """
    goodrows = np.where(np.all(
        (coordarraysarray > minval) & (coordarraysarray < maxval), axis=1))[0]
    return goodrows[0], goodrows[-1]
    
def isoutsideAK(latsarray, minlat=55., maxlat=72.):
    """Presumes 2D array of latitudes - looks only at first and last valid row"""
    firstrow, lastrow = getgoodrows(latsarray)
    alloutside = True
    

def dedupedlist(mylist):
    return list(OrderedDict.fromkeys(mylist))

def getoverpasses(basedir, scenelist=[], ):
    regex = re.compile(
r"(?P<ftype>[A-Z0-9]{5})_[a-z]+_d(?P<date>\d{8})_t(?P<time>\d{7})_e\d+_b(\d+)_c\d+_\w+.h5")
    if scenelist:
        subdirs = filter(
            os.path.isdir, 
            [os.path.join(basedir, item) for item in scenelist])
    else:
        subdirs = sorted(glob.glob(
            basedir +
            '/20[0-1][0-9]_[0-1][0-9]_[0-3][0-9]_[0-9][0-9][0-9]_[0-2][0-9][0-6][0-9]'))
    overpasses = OrderedDict()
    for subdir in subdirs:
        basename = os.path.split(subdir)[-1]
        overpasses[basename] = {}
        overpasses[basename]['dir'] = os.path.join(subdir, 'sdr')
        datafiles = sorted(
            [item for item in os.listdir(
                overpasses[basename]['dir']) if item.endswith('.h5')])
        if len(datafiles)%25 != 0:
            overpasses[basename]['message'] = "Some data files are missing in {}: {} is not divisible by 25".format(basename, len(datafiles))
        numgran = len(datafiles)//25
        overpasses[basename]['numgranules'] = numgran
        mos = [regex.search(filename) for filename in datafiles]
        overpasses[basename]['datetimes'] = dedupedlist([mo.groupdict()['date'] + '_' + mo.groupdict()['time'] for mo in mos])
        for ftype in [mo.groupdict()['ftype'] for mo in mos ]:
            overpasses[basename][ftype] = [filename for filename in datafiles if filename.startswith(ftype) ]
    return overpasses

def getfilesbygranule(basedir, scenelist=[]):
    regex = re.compile(r"(?P<ftype>[A-Z0-9]{5})_[a-z]+_d(?P<date>\d{8})_t(?P<time>\d{7})_e\d+_b(\d+)_c\d+_\w+.h5")
    if scenelist:
        subdirs = filter(os.path.isdir, [os.path.join(basedir, item) for item in scenelist])
    else:
        subdirs = sorted(glob.glob(basedir + '/20[0-1][0-9]_[0-1][0-9]_[0-3][0-9]_[0-9][0-9][0-9]_[0-2][0-9][0-6][0-9]'))
    overpasses = OrderedDict()
    for subdir in subdirs:
        basename = os.path.split(subdir)[-1]
        overpasses[basename] = {}
        overpasses[basename]['dir'] = os.path.join(subdir, 'sdr')
        datafiles = sorted([item for item in os.listdir(overpasses[basename]['dir']) if item.endswith('.h5')])
        if len(datafiles)%25 != 0:
            overpasses[basename]['message'] = "Some data files are missing in {}: {} is not divisible by 25".format(basename, len(datafiles))
        numgran = len(datafiles)//25
        mos = [regex.search(filename) for filename in datafiles]
        for mo, fname in zip(mos, datafiles):
            granulestr = mo.groupdict()['date'] + '_' + mo.groupdict()['time']
            ftype = mo.groupdict()['ftype']
            try: 
                overpasses[basename][granulestr][ftype] = fname
            except KeyError:
                overpasses[basename][granulestr] = {}
                overpasses[basename][granulestr][ftype] = fname
    return overpasses
    
def checkdir(basedir, subdirlist=[]):
    if not subdirlist:
        dirlist = sorted(glob.glob(os.path.join(basedir, "2015*")))
    else:
        dirlist = filter(os.path.isdir, [os.path.join(basedir, item) for item in subdirlist])
    for dir in dirlist:
        dirname = os.path.split(dir)[1]
        errormsg = None
        try:
            numfiles = len(glob.glob(os.path.join(dir, "sdr", "*.h5")))
        except:
            errormsg = "No data files found"
        if errormsg:
            print("{}: {}".format(dirname, errormsg))
        elif numfiles%25 != 0:
            print("{}: {}".format(dirname, numfiles))

def getedge(viirsdataset, step=50):
    frst, lst = getgoodrows(viirsdataset.lats)
    edgelons = np.concatenate((
        viirsdataset.lons[frst, ::step], 
        viirsdataset.lons[frst:lst-step:step, -1], 
        viirsdataset.lons[lst, ::-step], 
        viirsdataset.lons[lst:frst:-step, 0])) 
    edgelats = np.concatenate((
        viirsdataset.lats[frst, ::step], 
        viirsdataset.lats[frst:lst-step:step, -1], 
        viirsdataset.lats[lst, ::-step], 
        viirsdataset.lats[lst:frst:-step, 0]))
    return edgelons, edgelats
    
def checkviirsganulecomplete(granuledict, dataset='iband'):
    dataset = dataset.lower()
    complete = True
    if dataset not in BANDFILES.keys():
        print("Unknown band type '{}' for viirs granule. Valid values are: {}.".format(
            dataset, ', '.join(BANDFILES.keys())))
        return
    complete = True
    for bandname in BANDFILES[dataset]:
        try:
            if not granuledict[bandname]:
                complete = False
                print("detected missing band {}".format(bandname))
                return complete
        except KeyError:
            complete = False
            print("detected missing key for band {}".format(bandname))
            return complete
    return complete

def getgranulecatalog(basedir, overpassdirlist=None):
    intermediary = getfilesbygranule(basedir, scenelist=overpassdirlist)
    catalog = {}
    for overpass in intermediary:
        for granule in intermediary[overpass]:
            if granule in ['dir', 'message']: continue
            print(granule)
            catalog[granule] = intermediary[overpass][granule]
            catalog[granule][u'dir'] = intermediary[overpass]['dir']
            for datasettype in BANDFILES:
                catalog[granule][datasettype + u'_complete'] = checkviirsganulecomplete(catalog[granule])
            if catalog[granule][u'iband_complete']:
                try:
                    viirs = raster.VIIRSHDF5(os.path.join(
                            catalog[granule][u'dir'], 
                            catalog[granule][u'SVI01']))
                except IOError:
                    print("cannot access data file for I-band in {}".format(granule))
                    catalog[granule][u'iband_complete'] = False
                    continue
                catalog[granule][u'granuleID'] = viirs.meta[u'Data_Product'][u'AggregateBeginningGranuleID']
                catalog[granule][u'orbitnumber'] = viirs.meta[u'Data_Product'][u'AggregateBeginningOrbitNumber']
                try:
                    catalog[granule][u'ascending_node'] = viirs.ascending_node
                    edgelons, edgelats = getedge(viirs)
                except IOError:
                    print("cannot access geodata file for I-band in {}".format(granule))
                    catalog[granule][u'iband_complete'] = False
                    continue
                catalog[granule][u'edgepolygon_I'] = Polygon(zip(edgelons, edgelats)).wkt
                viirs.close()
    return catalog 

def generate_overviewbase():
    mm = Basemap(
        width=2000000, height=1800000, 
        resolution='l', 
        projection='aea', 
        lat_1=60., lat_2=70., lat_0=65, lon_0=-150)
    mm.drawcoastlines()
    mm.drawrivers(color=water, linewidth=1.5)
    mm.drawmeridians(np.arange(-180, 180, 5), labels=[False, False, False, 1])
    mm.drawparallels(np.arange(0, 80, 2), labels=[1, 1, False, False])
    mm.fillcontinents(        
        color=earth,
        lake_color=water)
    mm.drawmapboundary(fill_color=water)
#    mm.readshapefile(
#        "/Volumes/SCIENCE/GIS_Data/catalog.data.gov/tl_2013_02_prisecroads/tl_2013_02_prisecroads", 
#        'roads', 
#        color="slategrey", linewidth=2)
    return mm

def generate_willowbase(zoom_in=False, resolution='i'):
    if zoom_in:
        width, height = 50000, 40000
        lat_0 = 61.8
        lon_0 = -150.1
    else:
        width, height = 100000, 80000
        lat_0 = 61.8
        lon_0 = -149.75
    mm = Basemap(
        width=width, height=height, 
        resolution=resolution, 
        projection='aea', 
        lat_1=55., lat_2=65., lat_0=lat_0, lon_0=lon_0)
    mm.drawcoastlines()
    mm.drawrivers(color=water, linewidth=3, zorder=5)
    mm.drawmeridians(np.arange(-180, 180, 0.25), labels=[False, False, False, 1])
    mm.drawparallels(np.arange(0, 80, 0.25), labels=[True, True, False, False])
    mm.fillcontinents(        
        color=earth,
        lake_color=water)
    mm.drawmapboundary(fill_color=water)
    mm.readshapefile(
        primaryroads, 
        'roads', 
        color="darkslategrey", linewidth=3)
    return mm

def generate_fairbanksbase():
    mm = Basemap(
        width=300000, height=200000, 
        resolution='f', 
        projection='aea', 
        lat_1=60., lat_2=70., lat_0=fairbanks_lat, lon_0=fairbanks_lon)
    mm.drawcoastlines()
    mm.drawrivers(color=water, linewidth=1.5)
    mm.drawmeridians(np.arange(-180, 180, 2), labels=[False, False, False, 1])
    mm.drawparallels(np.arange(0, 80, 1), labels=[False, 1, False, False])
    mm.fillcontinents(        
        color=earth,
        lake_color=water)
    mm.drawmapboundary(fill_color=water)
    mm.readshapefile(
        "/Volumes/SCIENCE/GIS_Data/catalog.data.gov/tl_2013_02_prisecroads/tl_2013_02_prisecroads", 
        'roads', 
        color="slategrey", linewidth=2)
    return mm

def makeoverviewplot(viirsdatasets=[], datasetname=None, labels=[], earth=earth, water=water):
    fig1 = plt.figure(1, figsize=(15, 15))
    ax1 = fig1.add_subplot(111)
    mm = generate_overviewbase()
    if not labels or len(labels) != len(viirsdatasets):
        labels = map(str, range(len(viirsdatasets)))
    if viirsdatasets:
        current_palette = sns.color_palette("Paired", n_colors=len(viirsdatasets))
        for idx, viirsdataset in enumerate(viirsdatasets):
            lons, lats = getedge(viirsdataset)
            x, y = mm(lons, lats)
            ax1.plot(x, y,  
                linewidth=3, color=current_palette[idx], label=labels[idx])
    # add sockeye fire 
    x, y = mm(sockeye_lon, sockeye_lat)
    ax1.scatter(x, y, 50, marker='o', color='r', zorder=3)
    plt.legend()
    ax1.set_title('Alaska VIIRS overpass: {}'.format(datasetname))
    plt.show()

def makeplot(viirsdatasetlist, title=None, band='i4'):
    """Makes a plot of a single-band VIIRS HDF5 dataset object, I4 band"""
    fig1 = plt.figure(1, figsize=(20, 20))
    ax1 = fig1.add_subplot(111)
    # mapbase
    mm = generate_overviewbase()    # data 
    for viirsdataset in viirsdatasetlist:
        mult, add = viirsdataset.I4['BrightnessTemperatureFactors'][:]
        i4tb = viirsdataset.I4['BrightnessTemperature'][:]
        basemasked_i4tb = np.ma.masked_where(i4tb == np.max(i4tb), i4tb)
        basemasked_i4tb_scaled = basemasked_i4tb * mult + add
        xx, yy = mm(viirsdataset.lons, viirsdataset.lats)
        dataplt = mm.pcolormesh(xx, yy, basemasked_i4tb_scaled, edgecolors='None',vmin=280, vmax=370, zorder=10)
    
    if not title:
        datestamp = getdatestamp_AKDT(viirsdataset)
        ax1.set_title('Western Alaska: Brightness temperature from band {}, {}'.format(
            viirsdataset.meta['Data_Product']['N_Collection_Short_Name'], datestamp))
    else:
        ax1.set_title(title)
    mm.drawcoastlines(color="slategrey", zorder=20)
    cbar = mm.colorbar(dataplt, location='bottom', pad="15%")
    cbar.set_label("$T_B$ in $K$")

def getdatestamp_AKDT(viirsdataset, spaces=True):
    timestamp =  (viirsdataset.meta['Data_Product']['AggregateBeginningDate'] +  
                  u'_' + viirsdataset.meta['Data_Product']['AggregateBeginningTime'])
    if spaces:
        datestamp_AK = (dt.datetime.strptime(timestamp, '%Y%m%d_%H%M%S.%fZ') + 
                    dt.timedelta(hours=-8)).strftime('%Y-%m-%d %H:%M:%S AKDT')
    else:
        datestamp_AK = (dt.datetime.strptime(timestamp, '%Y%m%d_%H%M%S.%fZ') + 
                    dt.timedelta(hours=-8)).strftime('%Y%m%d_%H%M%S_AKDT')
    return datestamp_AK

def get_date_UTC(viirsdataset):
    dateutc = viirsdataset.meta['Data_Product']['AggregateBeginningDate']
    return dt.datetime.strptime(dateutc, '%Y%m%d').strftime('%Y-%m-%d')

def get_time_UTC(viirsdataset):
    timeutc = viirsdataset.meta['Data_Product']['AggregateBeginningTime']
    return dt.datetime.strptime(timeutc, '%H%M%S.%fZ').strftime('%H%M')
