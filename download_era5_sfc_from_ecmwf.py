#!/usr/bin/env python
#
#

import os, sys
import numpy as np
import cdsapi
server = cdsapi.Client()




             


code2name = { 

                'sprs':     'surface_pressure', # THIS
                'mtpr':     'mean_total_precipitation_rate',
                'msdlwrf':  'mean_surface_downward_long_wave_radiation_flux', # THIS
                'msnlwrf':  'mean_surface_net_long_wave_radiation_flux',  # THIS
                'msdswrf':  'mean_surface_downward_short_wave_radiation_flux',
                'meva':     'mean_evaporation_rate',
                'mx2t':     'maximum_2m_temperature_since_previous_post_processing',
                'mn2t':     'minimum_2m_temperature_since_previous_post_processing',
                'pmsl':     'mean_sea_level_pressure',
                'te2m':     '2m_temperature',
                'sst':      'sea_surface_temperature',
                'snw':      'snow_depth',
                'sic':      'sea_ice_cover',
                'smo':      ['volumetric_soil_water_layer_1','volumetric_soil_water_layer_2','volumetric_soil_water_layer_3','volumetric_soil_water_layer_4'],
                'ste':      ['soil_temperature_level_1', 'soil_temperature_level_2', 'soil_temperature_level_3', 'soil_temperature_level_4'],
                'prec':     'total_precipitation',
                'lsmask':   'land_sea_mask',
                'u10m':     '10m_u_component_of_wind',
                'v10m':     '10m_v_component_of_wind',
                'u100m':    '100m_u_component_of_wind', 
                'v100m':    '100m_v_component_of_wind',
                'wgst':     'instantaneous_10m_wind_gust', 
                'wgsp':     '10m_wind_gust_since_previous_post_processing',
                'evap':     'evaporation',
                'sswf':     'mean_surface_direct_short_wave_radiation_flux',
                'tsksr':    'total_sky_direct_solar_radiation_at_surface', # THIS
                'ssrd':     'surface_solar_radiation_downwards', # THIS
                'slhf':     'mean_surface_latent_heat_flux',
                'sshf':     'mean_surface_sensible_heat_flux',
                'tclc':     'total_cloud_cover',

                'dw2m':     '2m_dewpoint_temperature', 
                'asgo':     'angle_of_sub_gridscale_orography', 
                'cape':     'convective_available_potential_energy',
                'cinh':     'convective_inhibition', 
                'cprc':     'convective_rain_rate', 
                'hvgc':     'high_vegetation_cover',
                'ptyp':     'precipitation_type',
                'blhg':     'boundary_layer_height',
                'tcow':     'total_column_water',
                
                'kidx':     'k_index', 
                'ttid':     'total_totals_index',
                'laih':     'leaf_area_index_high_vegetation', 
                
                'bld':      'boundary_layer_dissipation',
                'zust':     'friction_velocity',
                'mx2t':     'maximum_2m_temperature_since_previous_post_processing',
                'mn2t':     'minimum_2m_temperature_since_previous_post_processing',
                'sknt':     'skin_temperature',
                'vimd':     'vertically_integrated_moisture_divergence',
                'deg0l':    'zero_degree_level',
                
            }




varname = str(sys.argv[1]) 


years = np.arange(2019,2026).astype(int)


for year in years:
    
    name = code2name[varname]
    basename = '%s_era5__%04d' % (name,year)
    nc_file  = '%s.nc'  % (basename)
    
    if os.path.exists(nc_file):
        print('File exists',nc_file)
        pass
    
    else:
        # Northern Fennoscandia (mostly Finland)
        opts = {
                'product_type'      : 'reanalysis',  
                'download_format'   : 'unarchived',
                'format'            : 'netcdf',
                'area'              : [73, 20, 62, 35],
                'variable'          : code2name[varname], 
                'year'              : '%04d' % (year),
                'month'             : ['%02d' % (i+1) for i in range(12)], 
                'day'               : ['%02d' % (i+1) for i in range(31)], 
                'time'              : ['%02d:00' % (i+0) for i in range(24)],
               }
        
        print('Fetching data for', nc_file)
        
        
        try:
            #dataset = 'reanalysis-era5-single-levels-preliminary-back-extension'
            dataset = 'reanalysis-era5-single-levels'
            server.retrieve(dataset, opts, nc_file).download()
        
        except:
            print('Retrieval failed for',nc_file)
        



