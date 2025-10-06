

# Read modules
import sys, ast, importlib, datetime, itertools, os, random, glob, joblib, pickle
import numpy as np
import pandas as pd
import xarray as xr; #xr.set_options(file_cache_maxsize=1)

import s3fs

import geopandas as gpd

import xgboost as xgb



from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.utils import resample


import matplotlib.pyplot as plt








# Local directories
code_dir='/lustre/home/kamarain/resiclim-microclimate/'
rslt_dir='/lustre/tmp/kamarain/resiclim-microclimate/'
era5_dir='/lustre/tmp/kamarain/ERA5_NFin/' 






# Read own module
sys.path.append(code_dir)
import functions as fcts
fcts=importlib.reload(fcts)


# Metadata
era5_vars, help_vars, lags, regions = fcts.get_metadata()




# Logger coordinates
all_coords = gpd.read_file(f'{code_dir}site_coordinates.gpkg').rename(columns={'X_tm35fin':'X', 'Y_tm35fin':'Y'})

# Näissä rajattu pois lähteet ym!
preselected_coords = pd.read_csv(f'{code_dir}site_coordinates.csv').rename(columns={'X_tm35fin':'X', 'Y_tm35fin':'Y'})

# Rajaa KIL MI pois
kil_drop = (preselected_coords['area']=='KIL') & (preselected_coords['X']<257000) & (preselected_coords['X']>252000)
preselected_coords = preselected_coords.loc[~kil_drop]


coords = pd.concat([preselected_coords, all_coords.loc[(all_coords['area'] == 'RAS') | (all_coords['area'] == 'TII')]])

coords = coords.drop(columns='geometry')

# Pick randomly 50-100 measurement sites per region
crds = []
for rgn in regions:
    sites = coords.loc[coords['area']==rgn]
    sample_size = np.min([100, len(sites)])
    print(rgn, sites.shape, sample_size)
    
    sites = sites.sample(sample_size, random_state=99).sort_index()
    crds.append(sites)
    



logger_crds = pd.concat(crds).reset_index().rename(columns={'Lat':'lat', 'Lon':'lon', 'X':'x', 'Y':'y'})



logger_crds.to_csv(rslt_dir+f'logger_locations_sample_22-09-2025.csv', index=False)





"""


centers = coords.drop(columns=['site']).groupby('area').mean()

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define margins based on your data
margin = 1
lon_min = centers['Lon'].min() - margin
lon_max = centers['Lon'].max() + margin
lat_min = centers['Lat'].min() - margin
lat_max = centers['Lat'].max() + margin

# Create a figure with a PlateCarree projection
fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Add shorelines and country boundaries
ax.coastlines(resolution='10m')
ax.add_feature(cfeature.BORDERS, linestyle='-')

# Plot the points
ax.scatter(centers['Lon'], centers['Lat'], color='blue', transform=ccrs.PlateCarree())

# Annotate each point with its name
for idx, row in centers.iterrows():
    ax.text(row['Lon'] + 0.05, row['Lat'] + 0.05, row.name, transform=ccrs.PlateCarree())

plt.tight_layout(); 
fig.savefig(rslt_dir+f'fig_area_locations.pdf')
fig.savefig(rslt_dir+f'fig_area_locations.png', dpi=200)
#plt.show(); 
plt.clf(); plt.close('all')

"""


# 







# Collect all logger data to one dataframe, mangle, and transform to 1-hourly


logger_data = []

for region in regions:
    print(region)
    #tmst_dir = f'/fmi/projappl/project_2005030/tomst_data/{region}'
    tmst_path = f'/lustre/tmp/kamarain/resiclim-microclimate/tomst_data/tomst_data_cleaned_{region}.csv'
    #tmst_dir = f'/projappl/project_2007415/repos/MICROCLIMATES/output/{region}'
    df_logger = pd.read_csv(tmst_path, parse_dates=['datetime'])
    df_logger.rename(columns={'datetime':'time'}, inplace=True)
    
    # Ensure datetime datatype
    df_logger['time'] = pd.to_datetime(df_logger['time'])
    
    print(df_logger)
    
    # Select valid data only
    df_logger = df_logger.loc[df_logger['error_tomst']==0].drop(columns='error_tomst')
    
    plot = True
    if plot:
        # Plot raw data
        f, axes = plt.subplots(2,1, figsize=(12,9))
        df_mean = df_logger.loc[:, ['time','T1','T2','T3']].groupby('time').mean()
        
        axes[0].plot(df_logger['T3'], c='tab:blue', label='T3')
        axes[0].plot(df_logger['T2'], c='tab:orange', label='T2')
        axes[0].plot(df_logger['T1'], c='tab:red', label='T1')
        axes[1].plot(df_mean['T3'], c='tab:blue', label='T3')
        axes[1].plot(df_mean['T2'], c='tab:orange', label='T2')
        axes[1].plot(df_mean['T1'], c='tab:red', label='T1')
        
        axes[0].set_title(region+', All data'); 
        axes[1].set_title(region+', Mean over logger sites'); 
        
        plt.tight_layout(); plt.legend()
        f.savefig(rslt_dir+f'fig_timeseries_simple_{region}.pdf')
        f.savefig(rslt_dir+f'fig_timeseries_simple_{region}.png', dpi=200)
        #plt.show(); 
        plt.clf(); plt.close('all')
    
    
    #logger_data.append(df_logger)
    print(region, df_logger['time'].min(), df_logger['time'].max())
    
    #df_logger['time'] = df_logger['time'].dt.tz_localize('UTC').dt.tz_convert('Europe/Helsinki')
    
    
    
    # Group dataframe by 'site' or 'tomst_id'
    #grouped = df_logger.groupby('tomst_id')  
    grouped = df_logger.groupby('site')  
    
    # Initialize an empty list to store the resampled dataframes
    hourly_medians_list = []
    
    # Iterate over each group, resample, and calculate mean
    for name, group in grouped:
        #print(name, group)
        # Set 'time' column as the index for the current group
        group.set_index('time', inplace=True)
        group.index = group.index.tz_convert(None)
        
        # Resample the current group to 1-hour frequency and calculate the mean
        hourly_medians = group.resample('1h').median(numeric_only=True)
        hourly_medians['site'] = name # group['site'].iloc[0]
        hourly_medians['tomst_id'] = group['tomst_id'].resample('1h').min() 
        
        # Select certain years
        #hourly_medians = hourly_medians.iloc[np.isin(pd.to_datetime(hourly_medians.index.values).year, range(2019,2026))]
        
        # Reset index to get 'time' back as a column
        hourly_medians = hourly_medians.dropna().reset_index()
        
        # Append the resampled dataframe to the list
        hourly_medians_list.append(hourly_medians)
    
    
    # Concatenate all resampled dataframes into a single dataframe
    df_hourly_medians = pd.concat(hourly_medians_list)
    
    # Merge df_hourly_medians with logger_crds on the 'site' column
    df_hourly_medians = pd.merge(df_hourly_medians, logger_crds[['site', 'lon', 'lat', 'x', 'y']], on='site', how='left')
    
    
    #df_hourly_medians = df_hourly_medians.rename(columns={'Lat':'lat', 'Lon':'lon', 'X':'x', 'Y':'y'})
    
    df_hourly_medians = df_hourly_medians.sort_values(['site','time']).reset_index(drop=True)
    
    
    df_hourly_medians['region'] = region
    
    
    fcts.print_ram_state()
    logger_data.append(df_hourly_medians)



logger_data = pd.concat(logger_data).reset_index(drop=True).dropna()


logger_data.to_csv(rslt_dir+'logger_data_selected_22-09-2025.csv', index=False)






