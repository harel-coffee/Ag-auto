#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is intended to be run nightly as a cron job.
It downloads the newest data from the following sources:

GEFS: global ensemble forecast system.
CFS: climate forecast system
NMME:
METAR:
REACCH:
"""

#MODULES
import sys, os, subprocess, traceback
import numpy as np
from urllib.request import urlopen
from datetime import date, datetime, timedelta


#################
#GLOBAL VARIABLES

#On the server
#APPDIR = "/srv/shiny-server/cbcctdev"
#WGRIB2_PATH = "/usr/bin/wgrib2"

#For Windows development
#USER = "Nicholas Potter"
#APPDIR = 'C:/Users/' + USER + '/work/code/cbcct/analysis/app'
#WGRIB2_PATH = "C:/Users/" + USER + '/bin/wgrib2'

#For Linux development
USER = "potterzot"
APPDIR = "/home/" + USER + "/reason/work/csanr/climate-change/cbcct/analysis/app"
WGRIB2_PATH = "/usr/bin/wgrib2"

#For MAC/OS development

################
# HELPER METHODS
def download(url, local):
    # check for files existing already and skip download if yes.
    test_local = local.replace(".grb2", ".nc4")
    test_local = test_local.replace(".grb", ".nc4")
    if os.path.isfile(test_local):
        print("File already exists locally, skipping download.")
        return False
    else:
        print("downloading {}".format(local))
        try:
            with open(local, mode='wb') as f:
                response = urlopen(url)
                f.write(response.read())
        except Exception as e:
            os.remove(local)
            print("Error downloading file: " + url + "\n and/or saving file: " + local)
            print(e)
    return True


# MAIN METHODS
def get_all(data_root, lat_range, lon_range, today, debug = False):
    """Fetch all sources: GEFS, CFS, METAR, REACCH, NMME, and NASS."""
    get_and_combine_cfs(data_root, lat_range, lon_range, today, debug = debug)
    get_gefs(data_root, lat_range, lon_range, today, debug = debug)
    get_ghcnd(data_root, lat_range, lon_range, today, debug = debug)
    get_metar(data_root, lat_range, lon_range, today, debug = debug)
    get_and_convert_nass(data_root, debug = debug)
    get_and_combine_nmme(data_root, lat_range, lon_range, today, debug = debug)
    get_and_combine_reacch(data_root, lat_range, lon_range, today, debug = debug)

#################################
# CFS
def get_cfs(data_dir, lat_range, lon_range, today, variables = None, debug = False):
    """
    Fetch the lastest CFS forecast data.

    Parameters
    ----------
    data_dir: directory to save data file in.
    today: today's date.
    variables: list of variable names to retrieve. Default is TMAX, TMIN, and TMP.

    Returns
    -------
    True if download without errors.
    """
    import subprocess
    from distutils import spawn

    #check that wgrib2 is available

    wgrib2 = spawn.find_executable("wgrib2")
    if wgrib2 == None:
      if os.path.isfile(WGRIB2_PATH):
        wgrib2 = WGRIB2_PATH
      else:
        print("Getting CFS data requires the wgrib2 program to be in the path, but it is not. Please install it or edit the path to include it.")
        sys.exit(2)

    #Example URL: "http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.20160209/06/time_grib_01/"
    forecast_times = ['00', '06','12','18']
    forecast_numbers = ['01','02','03','04']

    url_base = "http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs"

    data_date = (today - timedelta(1)).strftime("%Y%m%d")
    ###Iterate through files to download and convert to netcdf
    for t in forecast_times:
      url_file = "cfs." + data_date
      for n in forecast_numbers:
        url_root = "/".join([url_base, url_file, t, "time_grib_" + str(n)])

        for v in variables:
          variable_filename = ".".join([v, n, data_date + str(t)]) + ".daily.grb2"
          url = url_root + "/" + variable_filename
          froot = data_dir + "cfs_" + "_".join([data_date, v, t, n])

          #get the file
          local = froot + ".grb2"
          download(url, local)

          #create a subset file
          if os.path.isfile(local):
              res1 = subprocess.call([wgrib2,
                                      local,
                                      "-small_grib",
                                      "{0}:{1}".format(*lon_range),
                                      "{0}:{1}".format(*lat_range),
                                      local+".small"])
              if res1 != 0:
                print("Error subsetting {0} grib2 file".format(local))
                sys.exit(2)

              #convert to nc4
              ncfile = froot + ".nc4"
              res2 = subprocess.call([wgrib2,
                                     local+".small",
                                     '-netcdf',
                                     ncfile])

              if res2 != 0:
                print("Error converting {0} to netcdf.".format(local))
                sys.exit(2)
              elif not debug:
                os.remove(local)
                os.remove(local+".small")

    return True


def combine_cfs(data_dir, today, variables = ['tmp2m']):
    """
    Combine CFS forecast netcdf files into a single netcdf file with distribution statistics.

    Parameters
    ----------
    data_dir:   The directory containing CFS data files.
    data_date:  String of the file dates.
    variables:  list of variable names to summarize.

    Returns
    -------
    True if no errors.

    """
    from netCDF4 import Dataset as nc
    from collections import OrderedDict

    def read_data(fcst, ens, variables, data_date, data_dir):
        """Load the selected variables from the files into an array."""
        res = {}
        for k, v in enumerate(variables):
            for i, f in enumerate(fcst):
                for j, e in enumerate(ens):
                    in_fn = "_".join(["cfs", data_date, v, f, e]) + ".nc4"
                    with nc(data_dir + in_fn, "r", format="NETCDF4") as in_cdf:
                        if i == 0 and j == 0: #get initial dimensions
                            d = np.empty(shape=(
                                len(fcst),
                                len(ens),
                                len(in_cdf.variables['time'][:]),
                                len(in_cdf.variables['latitude'][:]),
                                len(in_cdf.variables['longitude'][:])
                                ), dtype=float)
                            d[:,:,:,:,:] = np.nan #set all to nan

                        #add the variable data to the data array
                        vals = in_cdf.variables[ var_dict[v] ][ :,:,: ]
                        if vals.shape[0] == d.shape[2]:
                            d[i, j, :,:,:] = vals
                        else:
                            d[i, j, i:,:,:] = vals
            res[v] = d
        return res

    def read_dims(fn):
        """Read ncdf4 file and return a dict of dimension values."""
        dims = OrderedDict()
        with nc(fn, "r", format="NETCDF4") as in_cdf:
            for d in in_cdf.dimensions.keys():
                dims.update({d: in_cdf.variables[d][:]})
        return dims

    def write_ncdf4(cdf, dims, d, model):
        """Write CFS data to a file."""
        with nc(cdf, "w", format="NETCDF4") as out_cdf:
            #Create the file dimensions and metadata
            out_cdf.description = ("CFS forecasts for times " +
                                  ", ".join(model['fcst']) +
                                  " and ensembles " +
                                  ", ".join(model['ens']) +
                                  " " + today.strftime("%Y%m%d") + ".")

            #Add dimensions
            for dim in dims.keys():
                out_cdf.createDimension(dim, len(dims[dim]))
                dvar = out_cdf.createVariable(dim, 'd', (dim, ))
                dvar[:] = dims[dim]

            for k, v in enumerate(d.keys()):
                out_v = out_cdf.createVariable(v, 'd',
                    ('fcst', 'ens', 'time', 'latitude', 'longitude'))
                out_v[:] = d[v][:]

        return True

    var_dict = {"tmp2m": "TMP_2maboveground", 
                "prate": "PRATE_surface"}
    data_date = (today - timedelta(1)).strftime("%Y%m%d")
    forecast_numbers = ['01', '02', '03', '04']
    forecast_times = ['00', '06', '12', '18']

    models = {
        1: {'ens': ['01'],
            'fcst': ['00', '06', '12', '18']
        },
        2: {'ens': ['02', '03', '04'],
            'fcst': ['00']
        },
        3: {'ens': ['02', '03', '04'],
            'fcst': ['06', '12', '18']
        }
    }

    for k,v in models.items():
        dims = OrderedDict()
        dims['fcst'] = [int(x) for x in v['fcst']]
        dims['ens'] = [int(x) for x in v['ens']]
        #get a dict of the dimension names and values
        dims2 = read_dims(data_dir +
                         "_".join(["cfs",
                                   data_date,
                                   variables[0],
                                   v['fcst'][0],
                                   v['ens'][0]
                                  ]) +
                         ".nc4")

        {dims.update({x: dims2[x]}) for x in dims2.keys()}
            
        #read the variable data
        d = read_data(v['fcst'], v['ens'], variables, data_date, data_dir)

        #write the data to ncdf
        cdf = "cfs" + str(k) + "_" + today.strftime("%Y%m%d") + ".nc4"
        write_ncdf4(data_dir + cdf, dims, d, v)

    return True


def get_and_combine_cfs(data_root, lat_range, lon_range, today, variables = None, debug = False):
    """
    Fetch CFS data and combine into a single NETCDF file.

    Parameters
    ----------
    data_dir:  directory to save data file in.
    lat_range: list of two latitudes defining the range of values to return.
    lon_range: list of two longitudes defining the range of values to return.
    today:     date object to base off of. GEFS is generally available within a few hours of present. Here we get it in the early AM to fetch the current day's forecast at 00 hours.
    variables: list of variables to download.
    debug:     boolean on whether to print debug messages and save source files when done.

    Returns
    -------
    True if downloaded and converted without error.
    """
    import glob

    #Set variables
    if variables is None:
        variables = ["tmp2m", "prate"]

    data_dir = data_root + "/cfs/"

    #download, subset, and convert the data
    get_cfs(data_dir, lat_range, lon_range, today, variables, debug)

    #combine to a single NETCDF file
    try:
        combine_cfs(data_dir, today, variables)
        for v in variables:
            files = glob.glob(data_dir + 'cfs_*' + v + '*.nc4')
            if not debug:
                [os.remove(f) for f in files]
    except Exception as e:
        print("Combining CFS files failed.")
        print(traceback.print_exc())
        print(e)
        sys.exit(2)

    return True

######
# GEFS
def get_gefs(data_root, lat_range, lon_range, today, variables = None, debug = False):
    """
    Fetch the latest GEFS forecast data.

    Parameters
    ----------
    data_dir:  directory to save data file in.
    lat_range: list of two latitudes defining the range of values to return.
    lon_range: list of two longitudes defining the range of values to return.
    today: date object to base off of. GEFS is generally available within a few hours of present. Here we get it in the early AM to fetch the current day's forecast at 00 hours.
    debug:     boolean on whether to print debug messages and save source files when done.
    variables: list of variables to download.

    Returns
    -------
    True if download without errors.
    """
    if variables is None:
        variables = ["Temperature_height_above_ground_ens", "Total_precipitation_surface_6_Hour_Accumulation_ens"]

    data_date = today.strftime("%Y%m%d")
    data_dir = data_root + "/gefs/"

    ###Set URL
    #example: http://thredds.ucar.edu/thredds/ncss/grib/NCEP/GEFS/Global_1p0deg_Ensemble/members/GEFS_Global_1p0deg_Ensemble_20151230_0000.grib2
    #?var=Temperature_height_above_ground_ens
    #&var=Total_precipitation_surface_6_Hour_Accumulation_ens
    #&disableProjSubset=on&horizStride=1
    #&time_start=2015-12-30T00%3A00%3A00Z&time_end=2016-01-07T12%3A00%3A00Z
    #&timeStride=1&vertCoord=&addLatLon=true&accept=netcdf4
    url_base = "http://thredds.ucar.edu/thredds/ncss/grib/NCEP/GEFS/Global_1p0deg_Ensemble/members/GEFS_Global_1p0deg_Ensemble_" + data_date + "_0000.grib2"

    #times to fetch. GEFS is 17 days of data at 6 hour intervals
    time_start = today.strftime("%Y-%m-%d")
    time_end = (today + timedelta(17)).strftime("%Y-%m-%d")

    url = url_base + "?var=" + "&var=".join(variables) + \
        "&disableLLSubset=off"+ "&diasbleProjSubset=off" + "&horizStride=1" + \
        "&north=" + str(lat_range[1]) + \
        "&west=" + str(lon_range[0]) + \
        "&east=" + str(lon_range[1]) + \
        "&south=" + str(lat_range[0]) + \
        "&time_start=" + time_start + "T00%3A00%3A00Z" + \
        "&time_end=" + time_end + "T00%3A00%3A00Z" + \
        "&timeStride=1" + "&vertCoord=" + "&addLatLon=true" + \
        "&accept=netcdf4"

    local = data_dir + "gefs_" + data_date + ".nc4"

    #open and download file
    return download(url,local)

#######
# GHCND
def get_ghcnd(data_root, lat_range, lon_range, today, variables = None, debug = False):
    """
    Get GHCN Daily station data for specified lat/lon ranges.

    Parameters
    ----------
    data_dir: directory to save data file in.
    lat_range: list of two latitudes defining the range of values to return.
    lon_range: list of two longitudes defining the range of values to return.
    today: date object to base off of. GEFS is generally available within a few hours of present. Here we get it in the early AM to fetch the current day's forecast at 00 hours.
    debug: boolean on whether to print debug messages and save source files when done.
    variables: list of variables to download.

    Returns
    -------
    True if download without errors.
    """
    if variables is None:
        variables = None

    data_dir = data_root + "/ghcnd/"
    data_date = today - timedelta(1)


    #base url
    url_base = "ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily"
    #get list of stations to start with
    url_station_list = url_base + "/ghcnd-stations.txt"
    local_station_list = "/".join([data_dir, "ghcnd", "ghcnd-stations.txt"])
    download(url_station_list, local_station_list)

    delimiters = [11, 1, 8, 1, 9, 1, 6, 1, 2, 1, 30, 1, 3, 1, 3, 1, 5]
    stations = np.genfromtxt(local_station_list, dtype=None, delimiter=delimiters, unpack = True, usecols=(0,2,4))
    our_stations = [x[0] for x in stations if (x[1] >= lat_range[0]) & (x[1] <= lat_range[1]) & (x[2] >= lon_range[0]) & (x[2] <= lon_range[1])]

    #download the month's station data
    for s in our_stations:
      url = url_base + "/all/" + s + ".dly"
      local = data_dir + s + ".dly"
      download(url, local)

    return True


#######
# METAR
def get_metar(data_root, lat_range, lon_range, today, variables = None, debug = False):
    """
    Fetch Metar Station climate data.

    Parameters
    ----------
    data_dir:  directory to save data to
    lat_range: list of range of latitudes to collect
    lon_range: list of range of longitudes to collect
    debug:     boolean on whether to print debug messages and save source files when done.
    variables: list of variable names to fetch

    Returns
    -------
    True if downloaded without errors
    """
    if variables is None:
        variables = ["air_temperature", "precipitation_amount_hourly", "wind_speed"]

    data_dir = data_root + "/metar/"


    ###Build the URL
    #example: http://thredds.ucar.edu/thredds/ncss/nws/metar/ncdecoded/Metar_Station_Data_fc.cdmr?req=station
    #&var=air_temperature
    #&var=precipitation_amount_hourly
    #&var=wind_speed
    #&north=55.000000&west=-125&east=-109&south=40.000000
    #&time_start=2016-01-05T00%3A00%3A00Z&time_end=2016-01-06T00%3A00%3A00Z&accept=netcdf4
    url_base = "http://thredds.ucar.edu/thredds/ncss/nws/metar/ncdecoded/Metar_Station_Data_fc.cdmr?req=station"

    #date start is 11th of last month, end is 10th of this month
    year = today.strftime("%Y")
    this_month = today.strftime("%m")
    last_month = str(int(this_month) - 1).zfill(2)

    time_end = year + '-' + this_month + "-10"
    if this_month == '01':
      time_start = str(int(year)-1) + '-12-11'
    else:
      time_start = year + "-" + last_month + "-11"

    url = url_base + "&var=" + "&var=".join(variables) + \
        "&north=" + str(lat_range[1]) + \
        "&west=" + str(lon_range[0]) + \
        "&east=" + str(lon_range[1]) + \
        "&south=" + str(lat_range[0]) + \
        "&time_start=" + time_start + "T00%3A00%3A00Z" + \
        "&time_end=" + time_end + "T00%3A00%3A00Z" + \
        "&accept=netcdf4"

    local = data_dir + "metar_" + today.strftime("%Y%m") + ".nc4"

    #download file
    download(url, local)

    return True



##################
## NMME
def combine_nmme(data_dir, data_date, variables = None, models = None):
    """Convert NMME anomalies to netcdf and combine as summary stats.

    Parameters
    ----------
    data_dir:   directory to read/write data to.
    data_date:  String of data file date.
    variables:  list of variable names to combine.
    models:     list of model names to combine.

    Returns
    -------
    True if successful.
    """
    from netCDF4 import Dataset as nc
    from collections import OrderedDict

    num_models = len(models)
    num_fcst_months = 8

    nmme_filename = "nmme_" + data_date + ".nc4"

    # create the variables and set values
    """Write CFS data to a file."""
    with nc(data_dir + nmme_filename, "w", format="NETCDF4") as out_cdf:
        #Create the file dimensions and metadata
        out_cdf.description = ("NMME Anomalies for models " +
                              ", ".join(models) + " for " + data_date + ".")

        #Add variable data for each variable and model
        for i, v in enumerate(variables): #i is just for determining if the first
            for j, m in enumerate(models):
                fn = ".".join([v, data_date, m, "ensmean", "anom", "1x1", "nc4"])
                with nc(data_dir + fn, "r") as in_cdf:
                    if i==0 and j==0: #get the dimensions
                        out_cdf.createDimension("model", len(models))
                        dvar = out_cdf.createVariable("model", 'S12', ("model",))
                        for z in range(0, len(models)): #have to assign by index
                            dvar[z] = models[z]
                        for d in in_cdf.dimensions.keys():
                            dvals = in_cdf.variables[d][:]
                            out_cdf.createDimension(d, len(dvals))
                            dvar = out_cdf.createVariable(d, 'd', (d,))
                            dvar[:] = dvals
                        dims = out_cdf.dimensions.keys()
                    if j==0: #only create the variable on the first model
                        out_v = out_cdf.createVariable(v, 'd', tuple(dims))

                    #values differ by model for each variable
                    vals = in_cdf.variables[ v ][ :,:,: ] #month, lat, lon
                    out_v[j, :, :, :] = vals

    return True


def get_nmme(data_dir, lat_range, lon_range, data_date, variables = None, models = None):
    """
    Fetch NMME anomalies.

    Parameters
    ----------
    data_dir:   directory to save data to
    lat_range:  list of range of latitudes to collect
    lon_range:  list of range of longitudes to collect
    data_date:  String of data file date
    variables:  list of variable names to fetch
    models:     list of model names to fetch

    Returns
    -------
    True if downloaded without errors
    """
    import pygrib
    from netCDF4 import Dataset as nc
    from collections import OrderedDict

    ###Build the URL
    #ftp://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom/ENSMEAN/2016040800/tmp2m.2016040800.ENSMEAN.ensmean.anom.1x1.grb
    url_base = "ftp://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom/ENSMEAN"

    dims = OrderedDict()
    dims.update({"month": [x for x in range(1,9)]})
    dims.update({"latitude": [x for x in range(1,182)]})
    dims.update({"longitude": [x for x in range(1,361)]})

    for m in models:
      for v in variables:
        for a in ["anom", "fcst"]:
            filename = ".".join([v, data_date, m, "ensmean", a, "1x1", "grb"])
            url = "/".join([url_base, data_date, filename])
            local = data_dir + filename
            dl = download(url, local)

            # Create a NetCDF file if not already exist
            ncdf_filename = local.replace(".grb", ".nc4")
            if not os.path.isfile(ncdf_filename):
                with nc(ncdf_filename, "w", format="NETCDF4") as out_cdf:
                    # attributes
                    out_cdf.description = "NMME model " + m + "."

                    # dimensions
                    for k, val in dims.items():
                        out_cdf.createDimension(k, len(val))
                        out_var = out_cdf.createVariable(k, 'd', (k,))
                        out_var[:] = val

                    # values
                    with pygrib.open(local) as grbs:
                        out_v = out_cdf.createVariable(v, 'd', tuple(dims.keys()))
                        for j,g in enumerate(grbs):
                            out_v[j,:,:] = g.values

    return True


def get_and_combine_nmme(data_root, lat_range, lon_range, today, variables = None, models = None, debug = False):
    """
    Fetch NMME Forecast anomalies and combine to a single file.

    Parameters
    ----------
    data_dir:   directory to save data to
    lat_range:  list of range of latitudes to collect
    lon_range:  list of range of longitudes to collect
    today:      a date object
    variables:  list of variable names to fetch
    models:     list of model names to fetch
    debug:      boolean on whether to print debug messages and save source files when done.

    Returns
    -------
    True if no errors
    """
    from glob import glob

    if variables is None:
        variables = ["tmp2m", "prate", "tmpsfc"]

    if models is None:
      models = ["CFSv2", "CMC1", "CMC2", "GFDL", "GFDL_FLOR", "NASA", "NCAR_CCSM4"]

    data_dir = data_root + "/nmme/"
    
    #data is replaced on the 8th of the month, so subtract 8 days from today
    #gives us last month is 8th or earlier, and this month if 9th or later
    data_date = (today - timedelta(days=8)).strftime("%Y%m") + "0800"

    #Fetch the nmme anomalies first.
    try:
      get_nmme(data_dir, lat_range, lon_range, data_date, variables, models)
      if not debug:
        [os.remove(x) for x in glob(data_dir + '*.grb')]
    except Exception as e:
      print("error downloading NMME data.")
      print(e)
      sys.exit(2)

    #Combine and save as netcdf
    try:
      combine_nmme(data_dir, data_date, variables, models)
      if not debug:
        for v in variables:
          [os.remove(x) for x in glob(data_dir + v + '*')]
    except Exception as e:
      print("Combining NMME files failed.")
      print(traceback.print_exc())
      print(e)
      sys.exit(2)

    return True


#################
## NASS
def convert_nass(data_dir, infile, outfile = "nass_qs.csv.gz", sqlfile="nass_qs.sqlite"):
    """
    Convert NASS QuickStats crop data into a HDF5 file.

    Parameters
    ----------
    data_dir directory of data.
    infile name of gzipped file to convert.
    outfile name of intermediary gzipped file to create.
    sqlfile name of sqlfile to create.

    Returns
    -------
    True if processed without (detected) errors.
    """

    import sqlite3
    import pandas as pd
    import gzip
    import csv

    #READ IN HELPERS
    def str_to_float(s):
        """Convert string w/ commas to float."""
        s = s.replace(' ', '')
        if s in ['(D)', '(Z)', 'NA', '(NA)', '(X)', '(S)', '(DU)', '', '(-)']:
            res = np.nan
        else:
            res = float(s.replace(',',''))
        return res

    def standardize_bearing(s):
        """Find whether the value is for bearing, non-bearing, or neither"""
        res = pd.Series(["NOT SPECIFIED" for x in range(0,len(s))])
        for x in ["BEARING", "NON-BEARING", "BEARING & NON-BEARING"]:
            i = s.apply(lambda r: r.find(x))
            res[i != -1] = x

        return res


    def standardize_double_cropped(s):
        """Set double cropped status. Takes a string and returns a string"""
        res = pd.Series(["NOT SPECIFIED" for x in range(0,len(s))])
        for x in ["DOUBLE CROPPED"]:
            i = s.apply(lambda r: r.find(x))
            res[i != -1] = x

        return res

    def standardize_irrigated(s):
        """Set irrigation status. Takes a string and returns a string value"""
        res = pd.Series(["NOT SPECIFIED" for x in range(0,len(s))])
        non_irrigated = s.apply(lambda r: r.find("NON-IRRIGATED"))
        irrigated = s.apply(lambda r: r.find(" IRRIGATED "))
        part_irrigated = s.apply(lambda r: r.find("PART-IRRIGATED"))
        none_of_crop = s.apply(lambda r: r.find("NONE OF CROP"))

        res[((non_irrigated != -1) | ((irrigated != -1) & (none_of_crop != -1)))] = "NON-IRRIGATED"
        res[irrigated != -1] = "IRRIGATED"
        res[part_irrigated != -1] = "PART-IRRIGATED"

        return res


    def standardize_processing(s):
        """Takes the short description and returns a processed state."""
        res = pd.Series(["NOT SPECIFIED" for x in range(0,len(s))])
        for x in ["PROCESSING", "FRESH"]:
            i = s.apply(lambda r: r.find(x))
            res[i != -1] = x

        return res


    def standardize_organic(s):
        """Determine organic status."""

        res = pd.Series(["NOT SPECIFIED" for x in range(0,len(s))])
        i = s.apply(lambda r: r.find("ORGANIC"))
        res[i != -1] = "ORGANIC"

        return res


    def standardize_crop_name(df):
        """Creates a name from the row data."""
        name = pd.Series(df['COMMODITY_DESC'])
        not_all_classes = (df['CLASS_DESC'] != "ALL CLASSES")
        name[not_all_classes] += df.loc[not_all_classes, 'CLASS_DESC']

        p = df['SHORT_DESC'].apply(lambda r: r.find("PROCESSING"))
        f = df['SHORT_DESC'].apply(lambda r: r.find("FRESH"))
        name[p != -1] += ", PROCESSING"
        name[f != -1] += ", FRESH"

        return name

    #READING ARGUMENTS
    data_dir = data_dir + "/"
    converters = {'VALUE': str_to_float}
    col_dtypes = {'SOURCE_DESC':"str",
        'SECTOR_DESC': "str",
        'AGG_LEVEL_DESC': "str",
        'STATE_ALPHA': "str",
        'STATE_NAME': "str",
        'COUNTY_NAME': "str",
        'YEAR': "str",
        'COMMODITY_DESC': "str",
        'CLASS_DESC': "str",
        'DOMAIN_DESC': "str",
        'DOMAINCAT_DESC': "str",
        'SHORT_DESC': "str",
        'STATISTICCAT_DESC': "str",
        'UNIT_DESC': "str",
        'VALUE': np.dtype('float64')
    }
    cols = list(col_dtypes.keys())
    incl_states = ["WA", "ID", "OR"]

    try:
        with gzip.open(data_dir + infile, mode='rt') as fi:
            reader = csv.DictReader(fi, delimiter='\t')

            fo = gzip.open(data_dir + outfile, 'wt')
            writer = csv.DictWriter(fo, extrasaction='ignore', fieldnames=cols)
            writer.writeheader()

            for i, row in enumerate(reader):
                if ((row['AGG_LEVEL_DESC']=="COUNTY") and
                    (row['STATE_ALPHA'] in  incl_states) and
                    (row['DOMAINCAT_DESC']=="NOT SPECIFIED")):

                    #outrow = {x:row[x] for x in cols}
                    #Python 2.6
                    outrow = dict((x,row[x]) for x in cols)
                    outrow['VALUE'] = str_to_float(outrow['VALUE'])
                    writer.writerow(outrow)
            fo.close()

    except Exception as e:
        print("Error in converting: {}".format(e))
        sys.exit(2)

    #create some useful variables for distinguishing
    df = pd.read_csv(data_dir+outfile, compression='gzip')
    df['full_name'] = standardize_crop_name(df)
    df['irrigated'] = standardize_irrigated(df['SHORT_DESC'])
    df['bearing'] = standardize_bearing(df['SHORT_DESC'])
    df['organic'] = standardize_organic(df['SHORT_DESC'])
    df['double_cropped'] = standardize_double_cropped(df['SHORT_DESC'])
    df['processing'] = standardize_processing(df['SHORT_DESC'])

    #save to sql db
    sqlcon = sqlite3.connect(data_dir+sqlfile)
    cursor = sqlcon.cursor()
    cursor.execute('DROP TABLE nass_qs')
    sqlcon.commit()
    df.to_sql("nass_qs", sqlcon)
    sqlcon.close()

    return True


def get_and_convert_nass(data_dir, debug = False):
    """
    Fetch NASS QuickStats data on crops.

    Parameters
    ----------
    data_dir:   directory to save data to.
    debug:      boolean on whether to print debug messages and save source files when done.

    Returns
    -------
    True if downloaded without errors.
    """
    from ftplib import FTP

    ###Build URL and local locations
    server = "ftp.nass.usda.gov"
    ftp_dir = "quickstats"

    #data directory for this data
    data_dir = data_root + "/nass/"
    ###Get the data
    try:
        ftp = FTP(server)
    except:
        print("Couldn't access server {}.".format(server))
        sys.exit(2)

    login_msg = ftp.login()
    if login_msg != "230 Login successful.":
        print("Couldn't log in to server {}.".format(server))
        sys.exit(2)
    else:
        cwd_msg = ftp.cwd(ftp_dir) #Change directory
        ftp_files = ftp.nlst()     #list files
        ftp_file = [x for x in ftp_files if "crop" in x][0] #get crop filename

        #Fetch data
        ret_msg = ftp.retrbinary('RETR ' + ftp_file, open(data_dir + ftp_file, 'wb').write)
        ftp.quit()

        if ret_msg != '226 File send OK.':
            print("Unable to download {0}. {1}".format(ftp_file, ret_msg))
            sys.exit(2)
        else:
            try: #convert to HDF5
                convert_nass(data_dir, ftp_file)
                if not debug:
                    os.remove(data_dir + ftp_file) #remove the file after conversion
            except Exception as e:
                print("Error converting to HDF5: {}".format(e))
                print(traceback.print_exc())
                sys.exit(2)

    return True

########
# REACCH
def get_reacch(data_dir, lat_range, lon_range, data_date, variable_list):
    """
    Fetch REACCH present-year historical data.

    Parameters
    ----------
    data_dir: the reacch data directory.
    lat_range: range of latitudes to fetch.
    lon_range: range of longitudes to fetch.
    data_date: Date of data.
    variable: a dict including the variable name and it's file name on the reacch thredds server.

    Returns
    -------
    True if data downloaded successfully.
    """

    time_start = data_date.strftime("%Y") + "-01-01"
    time_end = data_date.strftime("%Y") + "-12-31"

    ###Build the URL
    #https://www.reacchpna.org/thredds/ncss/agg_met_pr_1979_2016_CONUS.nc?north=49.3960&west=-125&east=-67.0638&south=25.0626&disableProjSubset=on&horizStride=1&time_start=1979-01-01T00%3A00%3A00Z&time_end=2016-05-16T00%3A00%3A00Z&timeStride=1&addLatLon=true&accept=netcdf4

    #example https://www.reacchpna.org/thredds/ncss/grid/
    #agg_met_tmmn_1979_2015_CONUS.nc?
    #var=daily_minimum_temperature&
    #north=45&west=-121&east=-109&south=40&horizStride=1&
    #time_start=2015-01-01T00%3A00%3A00Z&time_end=2015-12-31T00%3A00%3A00Z&timeStride=1&
    #addLatLon=true&accept=netcdf4
    url_base = "https://www.reacchpna.org/thredds/ncss/"

    for vname, vfile in variable_list.items():
        url = "&".join([url_base + vfile + "?var=" + vname,
                      "north=" + str(lat_range[1]),
                      "west=" + str(lon_range[0]),
                      "east=" + str(lon_range[1]),
                      "south=" + str(lat_range[0]),
                      "horizStride=1",
                      "time_start=" + time_start + "T00%3A00%3A00Z",
                      "time_end=" + time_end + "T00%3A00%3A00Z",
                      "timeStride=1",
                      "addLatLon=true",
                      "accept=netcdf4"])

        local = data_dir + vfile

        #download file
        download(url, local)

    return True


def combine_reacch(data_dir, data_date):
    """
    Combine all REACCH data files into a single NETCDF file.

    Parameters
    ----------
    data_dir: directory of REACCH data.

    Returns
    -------
    True if no errors.
    """
    import glob
    from netCDF4 import Dataset

    files = glob.glob(data_dir + "agg_met_*.nc")

    new_data = data_dir + "reacch_" + data_date.strftime("%Y%m%d") + ".nc4"
    try: 
        with Dataset(new_data, "w", format="NETCDF4") as out_cdf:
          #iterate through each file to append
          for i,f in enumerate(files):
              ds = []
              with Dataset(f, "r") as in_cdf:
                  for d_name, the_dim in in_cdf.dimensions.items():
                      ds.append(d_name)
                      if i==0:
                          out_cdf.createDimension(d_name, the_dim.size)
  
                  for v_name, var_in in in_cdf.variables.items():
                      if i==0 or v_name not in ds:
                          nv = out_cdf.createVariable(v_name, var_in.dtype, var_in.dimensions)
                          nv.setncatts({k: var_in.getncattr(k) for k in var_in.ncattrs()})
                          nv[:] = var_in[:]
    except Exception as e:
        os.remove(new_data)
        
    return True


def get_and_combine_reacch(data_root, lat_range, lon_range, today, variables = None, debug = False):
    """
    Fetch and combine REACCH data into a single NETCDF file.

    Parameters
    ----------
    data_root: the root of the data directory.
    lat_range: range of latitudes to fetch.
    lon_range: range of longitudes to fetch.
    today:     today's date.
    variables: list of variable names.
    debug:     boolean on whether to print debug messages and save source files when done.

    Returns
    -------
    True if no errors
    """
    import glob

    if variables is None:
        variables = [
            "daily_minimum_temperature",
            "daily_maximum_temperature",
            "precipitation_amount"
        ]

    var_dict = {"daily_minimum_temperature": "agg_met_tmmn_1979_2016_CONUS.nc",
                "daily_maximum_temperature": "agg_met_tmmx_1979_2016_CONUS.nc",
                "precipitation_amount": "agg_met_pr_1979_2016_CONUS.nc"}

    #reacch data directory
    if sys.platform == "win32":
        data_dir = data_root + "\\reacch\\"
    else:
        data_dir = data_root + "/reacch/"

    #times to fetch.
    data_date = today - timedelta(1)

    #Get the data first:
    variable_list = {v: var_dict[v] for v in variables}
    get_reacch(data_dir, lat_range, lon_range, data_date, variable_list)

    #Combine the data into a single netcdf file.
    try:
        combine_reacch(data_dir, today)
        files = glob.glob(data_dir + 'agg_met_*.nc')
        if not debug:
            [os.remove(f) for f in files]
    except Exception as e:
        print("Unable to combine REACCH data.")
        print(traceback.print_exc())
        sys.exit(2)

    return True

def main(argv):
    """"""
    #MODULES
    import getopt

    #non-mutable vars
    err_msg = 'Usage: get_data [-dhD --help --latmin --latmax --lonmin --lonmax --debug] <data_src> ...'
    sources = ["cfs", "gefs", "ghcnd", "metar", "nass", "nmme", "reacch", "all"]

    #mutable vars
    debug = False

    today = date.today()
    #today = datetime.strptime("2016-08-02", "%Y-%m-%d")

    data_root = APPDIR + '/data'

    lat_range = [41, 55]
    lon_range = [-125, -109]

    year = 2015

    #Check syntax.
    try:
        opts, args = getopt.getopt(argv, 'hDd:', ['help', 'debug', 'data_root',
            'latmin', 'latmax', 'lonmin', 'lonmax', 'year'])
    except getopt.GetoptError:
        print(err_msg)
        sys.exit(2)

    #Loop through options
    for opt, arg in opts:
        if opt in ['-h', '--help']:
            print(err_msg)
            sys.exit()
        elif opt in ['-d', '--data_root']:
            data_root = arg
        elif opt =='--date':
            today = arg
        elif opt=='--latmin':
            lat_range[0] = arg
        elif opt=='--latmax':
            lat_range[1] = arg
        elif opt=='--lonmin':
            lon_range[0] = arg
        elif opt=='--lonmax':
            lon_range[1] = arg
        elif opt=='--year':
            year = arg
        elif opt in ['-D', '--debug']:
            debug = True

    #Must have at least one argument
    if len(args) == 0:
        print("Arguments must be at least one of: {}".format(sources))
        print(err_msg)
        sys.exit(2)

    #Get listed forecasts
    for arg in args:
        if arg=='all':
            get_all(data_root, lat_range, lon_range, today, debug = debug)
        elif arg=='cfs':
            get_and_combine_cfs(data_root, lat_range, lon_range, today, debug = debug)
        elif arg=='gefs':
            get_gefs(data_root, lat_range, lon_range, today, debug = debug)
        elif arg=='ghcnd':
            get_ghcnd(data_root, lat_range, lon_range, today, debug = debug)
        elif arg=='nmme':
            get_and_combine_nmme(data_root, lat_range, lon_range, today, debug = debug)
        elif arg=='nass':
            get_and_convert_nass(data_root, debug = debug)
        elif arg=='metar':
            get_metar(data_root, lat_range, lon_range, today, debug = debug)
        elif arg=='reacch':
            get_and_combine_reacch(data_root, lat_range, lon_range, today, debug = debug)
        else:
            print("Source must be at least one of: " + ", ".join(sources))
            print(err_msg)
            sys.exit(2)


if __name__ == '__main__':
    main(sys.argv[1:])
