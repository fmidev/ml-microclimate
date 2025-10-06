

# Read modules
import sys, importlib
import numpy as np
import pandas as pd
import xarray as xr

import geopandas as gpd

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
smp_coords = pd.read_csv(rslt_dir+f'logger_locations_sample_22-09-2025.csv', index_col=False)
all_coords = gpd.read_file(f'{code_dir}site_coordinates_all.gpkg').rename(columns={'X_tm35fin':'X', 'Y_tm35fin':'Y'})

# Logger data
logger_data = pd.read_csv(rslt_dir+'logger_data_selected_22-09-2025.csv', index_col=False, parse_dates=['time'])

# Extract the coordinate data of the sites in different regions
site_points = logger_data.groupby('site').mean(numeric_only=True)[['x','y','lon','lat']]

# For interpolation, create a helper xarray dataset with the same dimensions as in logger_data
interp_points = xr.Dataset({"time": ("points", logger_data["time"].values), 
                            "site": ("points", logger_data["site"].values), 
                            "lat": ("points", logger_data["lat"].values), 
                            "lon": ("points", logger_data["lon"].values),
                            "y": ("points", logger_data["y"].values), 
                            "x": ("points", logger_data["x"].values),})




from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=12)

# Read preprocessed DEM data from netcdf files
plot = True

if plot: f, axes = plt.subplots(2,3, figsize=(12,7),)

dem_data = []
for i,region in enumerate(regions):
    print('\nDEM for',region)
    #ds_dem = fcts.read_dem(fs, region)
    ds_dem = fcts.read_dem('/lustre/tmp/kamarain/resiclim-microclimate', region).drop('spatial_ref')
    
    region_idx = logger_data['region'] == region
    x_min = logger_data.loc[region_idx, 'x'].min() - 500
    x_max = logger_data.loc[region_idx, 'x'].max() + 500
    y_min = logger_data.loc[region_idx, 'y'].min() - 500
    y_max = logger_data.loc[region_idx, 'y'].max() + 500
    
    ds_dem = ds_dem.sortby(['x','y'])
    ds_dem = ds_dem.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))
    
    dem_data.append(ds_dem)
    fcts.print_ram_state()
    
    if plot:
        ax = axes.ravel()[i]
        all_loggers = all_coords.loc[all_coords['area']==region]
        
        
        levels = np.arange(np.round(ds_dem['St_dem10m'].min(), -1), 
                           np.round(ds_dem['St_dem10m'].max(), -1), 10)
        
        m1 = ds_dem['St_dem10m'].plot.contourf(ax=ax, cmap='gray',robust=True, 
                                           alpha=1, levels=levels, zorder=-1,
                                           cbar_kwargs={'label': 'Elevation [m]'})
        m1.set_rasterized(True)
        
        #m2 = ds_dem['St_dem10m'].plot.contour(ax=ax, levels=levels, robust=True, 
        #                                      linewidths=0.5, colors='k', zorder=0)#cbar_kwargs={'label': 'Elevation [m]'})
        
        
        #ds['150cm_temp'].mean(['time']).plot(cmap='nipy_spectral',robust=True, alpha=0.8)
        #mask_ds.plot(alpha=0.5, add_colorbar=False)
        ax.scatter(x=all_loggers['X'].values, y=all_loggers['Y'].values, s=50,
                    c='blue',label='All measurement sites',edgecolors='k',alpha=1)
        
        ax.scatter(x=site_points['x'].values, y=site_points['y'].values, s=10,
                    c='red',label='Sampled measurement sites',edgecolors='red',alpha=1)
        
        if i==0: ax.legend(loc='upper right', fontsize='small')
        #ax.set_colorbar(m, loc='ll', label='Elevation [m]')
        
        ax.set_title(region)
        
        ax.set_xlim([x_min, x_max]); ax.set_ylim([y_min, y_max])
        ax.set_xticks([],[]); ax.set_yticks([],[])
        ax.set_xlabel(''); ax.set_ylabel('')
        
        # Scale bar
        scalebar = AnchoredSizeBar(ax.transData,
                           1000, '1 km', 'lower left', 
                           pad=0.3,
                           color='Orange',
                           frameon=False,
                           size_vertical=1,
                           fontproperties=fontprops)
        
        ax.add_artist(scalebar)



if plot:
    plt.tight_layout(); 
    f.savefig(rslt_dir+f'fig_site_points_all_and_sampled.pdf')
    f.savefig(rslt_dir+f'fig_site_points_all_and_sampled.png', dpi=200)
    #plt.show(); 
    plt.clf(); plt.close('all')




dem_ds = xr.merge(dem_data)

# Interpolate DEM data to measurement sites
dem_ds_interp = dem_ds.interp(x=interp_points['x'], 
                              y=interp_points['y'], method='linear')

dem_ds_interp = dem_ds_interp.assign_coords({'time': interp_points['time'], 
                                             'site': interp_points['site']})

# Convert interpolated xarray Dataset to DataFrame
#dem_data_df = dem_ds_interp.to_dataframe().reset_index().drop(columns=['x','y','points'])
dem_data_df = dem_ds_interp.to_dataframe().reset_index().drop(columns=['points'])

dem_data_df.to_csv(rslt_dir+'dem_data_selected_22-09-2025.csv', index=False)


