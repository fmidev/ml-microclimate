

# Read modules
import sys, ast, importlib, datetime, itertools, os, random, glob, joblib
import numpy as np
import pandas as pd
import xarray as xr; xr.set_options(file_cache_maxsize=1)

from datetime import timedelta

import xgboost as xgb



from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.utils import resample




import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from scipy.stats import pearsonr

import seaborn as sns




# Local directories
code_dir='/lustre/home/kamarain/resiclim-microclimate/'
rslt_dir='/lustre/tmp/kamarain/resiclim-microclimate/'
era5_dir='/lustre/tmp/kamarain/ERA5_NFin/' 


# Remote directory path
inpt_dir = 'resiclim-microclimate/varrio_5_month/'




region = 'PAL'
year = 2024
month = 6



region = str(sys.argv[1])
year   = int(sys.argv[2])
month  = int(sys.argv[3])





# Read own module
sys.path.append(code_dir)
import functions as fcts
fcts=importlib.reload(fcts)


# Metadata
era5_vars, help_vars, lags, regions = fcts.get_metadata()


"""
# Set up the S3 file system
access = os.getenv('S3_RESIC_FMI_ACCESS') 
secret = os.getenv('S3_RESIC_FMI_SECRET') 

fs = s3fs.S3FileSystem(anon=False, key=access, secret=secret,
    client_kwargs={'endpoint_url': 'https://a3s.fi'})

"""




#
logger_data = pd.read_csv(rslt_dir+'logger_data_selected.csv', index_col=False, parse_dates=['time'])

region_idx = logger_data['region'] == region
x_min = logger_data.loc[region_idx, 'x'].min() - 500
x_max = logger_data.loc[region_idx, 'x'].max() + 500
y_min = logger_data.loc[region_idx, 'y'].min() - 500
y_max = logger_data.loc[region_idx, 'y'].max() + 500








#for region, year in itertools.product(regions, [2019,2020,2021,2022,2023,2024]):
#for region in regions:
print(region, year, month, flush=True)

#ds_dem = fcts.read_dem(fs, region).drop('spatial_ref'); 
ds_dem = fcts.read_dem('/lustre/tmp/kamarain/resiclim-microclimate', region).drop('spatial_ref')
fcts.print_ram_state()

# Select areas covered by the loggers plus some extra for the edges
ds_dem = ds_dem.sortby(['x','y'])
ds_dem = ds_dem.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))


# Create a boolean mask where every point has no NaN values across all variables
mask = ds_dem.to_array().notnull().all(dim="variable")

# Apply the mask to the dataset and drop the coordinates where the mask is False
ds_dem = ds_dem.where(mask, drop=True)

# Coarsen the spatial resolution. Select the coarsing factor so that the data won't become too large
n_points_max = 52000
n_points_max = 130000
dxdy = 0
n_points = 1e9
while n_points > n_points_max:
    dxdy += 1
    print(dxdy)
    ds_test = ds_dem.isel(y=slice(0, None, dxdy), x=slice(0, None, dxdy)).stack(points=('x','y')) 
    n_points = ds_test.points.shape[0]
    


ds_dem = ds_dem.isel(y=slice(0, None, dxdy), x=slice(0, None, dxdy))
fcts.print_ram_state(f'DEM data for {region} was read')

ds_dem = ds_dem.stack(points=('x','y')) 

points = ds_dem[['x','y']].to_dataframe().reset_index(drop=True)[['x','y']]

points[['lon','lat']] = np.nan
points['lon'], points['lat'] = fcts.etrs_tm35fin_to_wgs84(points['x'], points['y']); #fcts.print_ram_state()

interp_points = xr.Dataset({"lat": ("points", points["lat"].values), 
                            "lon": ("points", points["lon"].values),
                            "y": ("points", points["y"].values), 
                            "x": ("points", points["x"].values),})

#ds_dem = ds_dem.interp(y=interp_points['y'],x=interp_points['x'],method='nearest')
#df_dem = ds_dem.to_dataframe().reset_index(); fcts.print_ram_state()

#fcts.print_ram_state()

# Select model for the specific validation year
model_file = glob.glob(rslt_dir+f'model_{region}_{year}.pkl')

# Just some model if validation year was not available
if len(model_file)==0:
    model_files = glob.glob(rslt_dir+f'model_{region}_*.pkl')
    np.random.seed(99)
    model_file = np.random.choice(model_files, 1)


#fitted_ensemble = xgb.XGBRegressor()
#fitted_ensemble.load_model(model_file[0])
fitted_ensemble = joblib.load(model_file[0])

#for year,month in itertools.product([2019,2020,2021,2022,2023,], np.arange(1,13)):


t_range = fcts.generate_hourly_timerange(year, month)

ds_era5_all = fcts.read_era5(era5_dir, era5_vars, help_vars, 
                             [62, 71], [19, 32], lags, t_range).drop('number')

fcts.print_ram_state(f'ERA5 was read for {region}')


# Interpolate all variables in era5_data 
ds_era5 = ds_era5_all.load().interp(
    lat=interp_points['lat'],
    lon=interp_points['lon'],
    method='linear'); fcts.print_ram_state(f'interpolation for {region}')

ds_dem_expanded = ds_dem.expand_dims(time=ds_era5.time); fcts.print_ram_state(f'expand dimensions for {region}')
ds_combined = xr.merge([ds_era5, ds_dem_expanded]); fcts.print_ram_state(f'merge datasets for {region}')
ds_combined = ds_combined.reset_index('points'); fcts.print_ram_state(f'reset index for {region}')
ds_combined = ds_combined.reset_coords(['lat','lon','y','x']); fcts.print_ram_state(f'reset coords for {region}')

# Convert interpolated xarray Dataset to DataFrame
df_combined = ds_combined.to_dataframe(); fcts.print_ram_state(f'convert to dataframe for {region}')#.drop(columns=['points','number','time','lat','lon'])
df_orig_index = df_combined.index
#df_combined = df_combined.drop(columns=['x','y']).reset_index()
#df_combined = df_combined.reset_index().sort_values(['points','time']).reset_index(drop=True); fcts.print_ram_state('sort dataframe')
df_combined = df_combined.reset_index(); fcts.print_ram_state(f'reset index for {region}')

# Define temporal cycle predictors
df_cycl = fcts.cyclical_predictors(pd.DataFrame(index=df_combined.time), ann=True, diu=True, chn=True).reset_index().drop(columns='time')

# Names of different variables
sptial_variables = ['lat','lon','y','x','points',]
reanal_variables = list(ds_era5.data_vars)
static_variables = list(ds_dem.data_vars)
cyclic_variables = list(df_cycl.columns)

# Merge all data together
all_data = pd.concat([df_combined, df_cycl], axis=1); fcts.print_ram_state(f'everything combined for {region}')


X = all_data[['time'] + reanal_variables + static_variables + cyclic_variables]; fcts.print_ram_state(f'X selected for {region}')


prediction = fitted_ensemble.predict(X.drop(columns='time')); fcts.print_ram_state(f'predictions generated for {region}')

df_Y = pd.DataFrame(index=df_orig_index, columns=['T1_predicted_offset','T2_predicted_offset','T3_predicted_offset'], data=prediction)
ds_Y = df_Y.to_xarray(); 
fcts.print_ram_state(f'predictions transformed to dataset in {region}')

ds_Y['T2m_ERA5'] = ds_era5['E5_t2m_+000'] - 273.15
ds_Y['skt_ERA5'] = ds_era5['E5_skt_+000'] - 273.15
ds_Y['T1_predicted'] = ds_Y['T1_predicted_offset'] + ds_Y['T2m_ERA5']
ds_Y['T2_predicted'] = ds_Y['T2_predicted_offset'] + ds_Y['T2m_ERA5']
ds_Y['T3_predicted'] = ds_Y['T3_predicted_offset'] + ds_Y['T2m_ERA5']

ds_Y['points'] = ds_dem.points; ds_Y = ds_Y.set_index(points=['x','y']) #ds_Y = ds_Y.set_index(points=['x', 'y'])

ds_Y = ds_Y.drop(['lat','lon']).unstack('points').transpose('time','y','x')
fcts.print_ram_state(f'dataset transformed from 2D to 3D in {region}')

ds_idx = pd.to_datetime(ds_Y.time.values).month==month
ds_Y = ds_Y.sel(time=ds_idx)

ds_Y.to_netcdf(rslt_dir+f'generated_data/data_{region}_{year}_{str(month).zfill(2)}.nc')
fcts.print_ram_state(f'dataset saved to disk for {region}')

"""
logger_time = pd.to_datetime(logger_data['time'].values)
idx_logger = (logger_data['region']==region) & (logger_time.month==month) & (logger_time.year==year)

logr_T = logger_data.loc[idx_logger,['time','T1','T2','T3']].groupby('time').mean()
xgbs_T = ds_Y.to_dataframe().reset_index().groupby('time').mean()
#ds_Y['T1'].mean(['x','y']).plot();ds_Y['T2'].mean(['x','y']).plot();ds_Y['T3'].mean(['x','y']).plot(); plt.show()

plt.plot(logr_T['T3'], c='k', label='OBS')
plt.plot(xgbs_T['T2m_ERA5'], c='gray',label='ERA5')
plt.plot(xgbs_T['T3_offset'],label='Offset')
plt.plot(xgbs_T['T3'],label='XGBoost'); plt.legend(); plt.show()
"""


