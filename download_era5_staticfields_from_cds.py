#!/usr/bin/env python
#
#

import os, sys, itertools
import numpy as np
import cdsapi
server = cdsapi.Client()



params = [
        "land_sea_mask",
        "lake_cover",
        "lake_depth",
        "orography", 
        "geopotential",
        "soil_type",
        "low_vegetation_cover",
        "high_vegetation_cover",
        "type_of_low_vegetation",
        "type_of_high_vegetation",
        "angle_of_sub_gridscale_orography",
        "anisotropy_of_sub_gridscale_orography",
        "slope_of_sub_gridscale_orography",
        "standard_deviation_of_filtered_subgrid_orography",
        "standard_deviation_of_orography",
]


nc_file  = 'era5_static_surface_variables.nc'

if os.path.exists(nc_file):
    print('File exists',nc_file)
    pass

else:
    # Global domain
    opts = {
            'product_type'      : 'reanalysis',  
            'download_format'   : 'unarchived',
            'format'            : 'netcdf',
            'variable'          : params, 
            'year'              : '2025',
            'month'             : '01', 
            'day'               : '01', 
            'time'              : '00:00',
           }
    
    print('Fetching data for', nc_file)
    
    
    try:
        dataset = 'reanalysis-era5-single-levels'
        server.retrieve(dataset, opts, nc_file).download()
    
    except:
        print('Retrieval failed for',nc_file)
    



