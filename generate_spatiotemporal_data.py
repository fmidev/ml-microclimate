

# Local directories
code_dir='/lustre/home/kamarain/resiclim-microclimate/'
rslt_dir='/lustre/tmp/kamarain/resiclim-microclimate/'
era5_dir='/lustre/tmp/kamarain/ERA5_NFin/' 



# Read modules
import sys, ast, importlib, datetime, itertools, os, random, glob, joblib
import numpy as np
import pandas as pd
import xarray as xr; #xr.set_options(file_cache_maxsize=1)

from datetime import timedelta

import xgboost as xgb



from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.utils import resample



import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from scipy.stats import pearsonr

import seaborn as sns



# Read own module
sys.path.append(code_dir)
import functions as fcts
fcts=importlib.reload(fcts)




version = '15-10-2025'



region = 'ULV'
year = 2019
month = 11


# Command line arguments
region = str(sys.argv[1])
year   = int(sys.argv[2])
month  = int(sys.argv[3])


print('Generating data for', region, year, month, flush=True)





# Metadata
era5_vars, help_vars, accumulations, lags, regions = fcts.get_metadata()



# Select model for the specific validation year
model_file = glob.glob(rslt_dir+f'model_{region}_{year}.pkl')

# Just some model if validation year was not available
if len(model_file)==0:
    model_files = glob.glob(rslt_dir+f'model_{region}_*.pkl')
    np.random.seed(99)
    model_file = np.random.choice(model_files, 1)


print(model_file)
fitted_ensemble = joblib.load(model_file[0])
fcts.print_ram_state(f'after reading the model file for predicting {region} {year} {month}')




# Define the extent of boundaries of regions based on the available logger data extent with 500m margins
logger_data = pd.read_csv(rslt_dir+f'logger_data_selected_{version}.csv', index_col=False, parse_dates=['time'])
region_idx = logger_data['region'] == region

x_min = logger_data.loc[region_idx, 'x'].min() - 500
x_max = logger_data.loc[region_idx, 'x'].max() + 500
y_min = logger_data.loc[region_idx, 'y'].min() - 500
y_max = logger_data.loc[region_idx, 'y'].max() + 500



# Read detailed static spatial features 
ds_dem = fcts.read_dem('/lustre/tmp/kamarain/resiclim-microclimate', region)#.drop('spatial_ref')
fcts.print_ram_state()

# Select the region
ds_dem = ds_dem.sortby(['x','y'])
ds_dem = ds_dem.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))


# Create a boolean mask where every point has no NaN values across all variables
mask = ds_dem.to_array().notnull().all(dim="variable")

# Apply the mask to the dataset and drop the coordinates where the mask is False
ds_dem = ds_dem.where(mask, drop=True)


# Select the coarsening factor N so that the data won't become too large to fit into memory
n_points_max = 90000;  # 100000; 
N = 0; n_points = 1e9
while n_points > n_points_max:
    N += 1
    ds_test = ds_dem.isel(y=slice(0, None, N), x=slice(0, None, N)).stack(points=('x','y')) 
    n_points = ds_test.points.shape[0]
    print(N, n_points, n_points_max)


# Coarsen the spatial resolution of the DEM data
ds_dem = ds_dem.isel(y=slice(0, None, N), x=slice(0, None, N))
ds_dem = ds_dem.stack(points=('x','y')) 
fcts.print_ram_state(f'after processing DEM data for {region} {year} {month}')


# Create the mesh for the target grid with n_points points
points = ds_dem[['x','y']].to_dataframe().reset_index(drop=True)[['x','y']]
points[['lon','lat']] = np.nan
points['lon'], points['lat'] = fcts.etrs_tm35fin_to_wgs84(points['x'], points['y'])

interp_points = xr.Dataset({"lat": ("points", points["lat"].values), 
                            "lon": ("points", points["lon"].values),
                            "y": ("points", points["y"].values), 
                            "x": ("points", points["x"].values),})




# Read ERA5 dynamic and static features for Fennoscandia
t_range = fcts.generate_hourly_timerange(year, month)
ds_era5_all = fcts.read_era5(era5_dir, [62, 71], [19, 32], lags, t_range)
fcts.print_ram_state(f'after reading ERA5 over Fennoscandia for {region} {year} {month}')


# Interpolate all variables of the ERA5 data to target domain 
ds_era5 = ds_era5_all.load().interp(
    lat=interp_points['lat'],
    lon=interp_points['lon'],
    method='linear')
fcts.print_ram_state(f'after interpolation of ERA5 for {region} {year} {month}')


# Create combination dataset by merging DEM and ERA5
ds_dem_withtime = ds_dem.expand_dims(time=ds_era5.time)
fcts.print_ram_state(f'after expanding time dimension for DEM at {region} {year} {month}')

ds_combined = xr.merge([ds_era5, ds_dem_withtime])
fcts.print_ram_state(f'after merging DEM and ERA5 datasets for {region} {year} {month}')

ds_combined = ds_combined.reset_index('points')
fcts.print_ram_state(f'after resetting index for {region} {year} {month}')

ds_combined = ds_combined.reset_coords(['lat','lon','y','x'])
fcts.print_ram_state(f'after resetting coords for {region} {year} {month}')


# Convert interpolated xarray Dataset to DataFrame
df_combined = ds_combined.to_dataframe()
df_index_original = df_combined.index
fcts.print_ram_state(f'after converting xarray dataset to dataframe for {region} {year} {month}') 


df_combined = df_combined.reset_index()
fcts.print_ram_state(f'after resetting index for {region} {year} {month}')


# Collect PISR data to one temporal column

# Ensure time is datetime
if not pd.api.types.is_datetime64_any_dtype(df_combined['time']):
    df_combined['time'] = pd.to_datetime(df_combined['time'])

# Create the target column
df_combined['St_pisr'] = np.nan #pd.NA
df_combined['St_pisr'] = df_combined['St_pisr'].astype(float)

# Fill per month (no big arrays, memory-safe)
for m in range(1, 13):
    col = f'St_pisr_{m}'
    if col in df_combined.columns:
        mask = df_combined['time'].dt.month.eq(m)
        df_combined.loc[mask, 'St_pisr'] = df_combined.loc[mask, col]

# Drop the original monthly columns that exist
pisr_cols = [c for c in df_combined.columns if c.startswith('St_pisr_')]
df_combined = df_combined.drop(columns=pisr_cols)

fcts.print_ram_state(f'after collecting PISR data for {region} {year} {month}')


# Define temporal cycle predictors
df_cycl = fcts.cyclical_predictors(pd.DataFrame(index=df_combined.time), ann=True, diu=True, chn=True).reset_index().drop(columns='time')

# Merge all data together
all_data = pd.concat([df_combined, df_cycl], axis=1)
fcts.print_ram_state(f'after combining all data for {region} {year} {month}')

# Define X matrix
x_cols = fitted_ensemble.feature_names_in_
X = all_data[x_cols]
fcts.print_ram_state(f'after selecting X for {region} {year} {month}')





# Predict
prediction = fitted_ensemble.predict(X)
fcts.print_ram_state(f'after generating predictions for {region} {year} {month}')

# Create a prediction dataframe and fill it
df_Y = pd.DataFrame(index=df_index_original, columns=['T1_predicted_offset','T2_predicted_offset','T3_predicted_offset'], data=prediction)

# Back transformation to Xarray dataset
ds_Y = df_Y.to_xarray()
fcts.print_ram_state(f'after transforming predictions from dataframe to dataset in {region} {year} {month}')


# Back to regular temperature values from offsets
ds_Y['skt_ERA5'] = ds_era5['E5dyna_skt_+000'] - 273.15
ds_Y['T1_predicted'] = ds_Y['T1_predicted_offset'] + ds_Y['skt_ERA5']
ds_Y['T2_predicted'] = ds_Y['T2_predicted_offset'] + ds_Y['skt_ERA5']
ds_Y['T3_predicted'] = ds_Y['T3_predicted_offset'] + ds_Y['skt_ERA5']

ds_Y['points'] = ds_dem.points; ds_Y = ds_Y.set_index(points=['x','y']) 

# Back to 3d from 2d 
ds_Y = ds_Y.drop(['lat','lon']).unstack('points').transpose('time','y','x')
fcts.print_ram_state(f'after transforming dataset from 2D to 3D in {region} {year} {month}')

# Clip data exactly to one month
ds_idx = pd.to_datetime(ds_Y.time.values).month==month
ds_Y = ds_Y.sel(time=ds_idx)

# Save data
ds_Y.to_netcdf(rslt_dir+f'generated_data/data_{region}_{year}_{str(month).zfill(2)}.nc')
fcts.print_ram_state(f'dataset saved to disk for {region} {year} {month}')



