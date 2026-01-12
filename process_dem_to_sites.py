

# Read modules
import sys, importlib
import numpy as np
import pandas as pd
import xarray as xr

import geopandas as gpd










# Local directories
code_dir='/lustre/home/kamarain/resiclim-microclimate/'
rslt_dir='/lustre/tmp/kamarain/resiclim-microclimate/'
era5_dir='/lustre/tmp/kamarain/ERA5_NFin/' 




version = '15-10-2025'

plot = True




# Read own module
sys.path.append(code_dir)
import functions as fcts
fcts=importlib.reload(fcts)



# Metadata
era5_vars, help_vars, accumulations, lags, regions = fcts.get_metadata()


# Logger coordinates
smp_coords = pd.read_csv(rslt_dir+f'logger_locations_sample_{version}.csv', index_col=False)
all_coords = gpd.read_file(f'{code_dir}site_coordinates_all.gpkg').rename(columns={'X_tm35fin':'X', 'Y_tm35fin':'Y'})

# Logger data
logger_data = pd.read_csv(rslt_dir+f'logger_data_selected_{version}.csv', index_col=False, parse_dates=['time'])

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


if plot: 
    import matplotlib; matplotlib.use('agg')
    import matplotlib.pyplot as plt
    f, axes = plt.subplots(2,3, figsize=(12,7),)

dem_data = []
for i,region in enumerate(regions):
    print('\nDEM for',region)
    #ds_dem = fcts.read_dem(fs, region)
    ds_dem = fcts.read_dem('/lustre/tmp/kamarain/resiclim-microclimate', region)#.drop('spatial_ref')
    
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


# Collect PISR data to one temporal column

# Ensure time is datetime
if not pd.api.types.is_datetime64_any_dtype(dem_data_df['time']):
    dem_data_df['time'] = pd.to_datetime(dem_data_df['time'])

# Create the target column
dem_data_df['St_pisr'] = pd.NA

# Fill per month (no big arrays, memory-safe)
for m in range(1, 13):
    col = f'St_pisr_{m}'
    if col in dem_data_df.columns:
        mask = dem_data_df['time'].dt.month.eq(m)
        dem_data_df.loc[mask, 'St_pisr'] = dem_data_df.loc[mask, col]

# Drop the original monthly columns that exist
pisr_cols = [c for c in dem_data_df.columns if c.startswith('St_pisr_')]
dem_data_df = dem_data_df.drop(columns=pisr_cols)

"""
dem_data_df[['time','St_pisr']].groupby('time').mean().plot(); plt.savefig('fig_pisr_time.png')
dem_data_df[['site','St_pisr']].groupby('site').mean().plot(); plt.savefig('fig_pisr_site.png')
"""


dem_data_df.to_csv(rslt_dir+f'dem_data_selected_{version}.csv', index=False)






if plot:
    
    # Correlation matrix
    df = dem_data_df.drop(columns=['x','y','time','site'])#.dropna()
    sorted_cols = np.sort(df.columns)
    corr = df[sorted_cols].corr().round(2)
    
    
    import seaborn as sns
    fig = plt.figure(figsize=(25,20)) #(35, 30))
    
    sns.heatmap(corr, annot=True, cmap='coolwarm',
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.title('Correlation Matrix', fontsize=16); plt.tight_layout()
    fig.savefig(rslt_dir+f'fig_staticfeatures_correlationmatrix_{version}.png')
    fig.savefig(rslt_dir+f'fig_staticfeatures_correlationmatrix_{version}.pdf')
    plt.clf(); plt.close('all')
    
    
    
    import matplotlib.patheffects as pe
    
    for i,region in enumerate(regions):
        ds_dem = dem_data[i]
        
        # Combine all variables into a DataArray where True = not NaN
        valid_mask = xr.concat([~ds_dem[var].isnull() for var in ds_dem.data_vars], dim="vars")
        
        # Reduce across the new 'vars' dimension — keep only where all variables are valid
        all_valid = valid_mask.all(dim="vars")
        
        # Create final mask: 1 where valid, NaN where at least one is NaN
        nan_mask = xr.where(all_valid, 1, np.nan)
        ds_mask = nan_mask.interp_like(ds_dem)
        
        ds_dem_masked = ds_dem*ds_mask
        ds_dem_masked = ds_dem_masked.dropna('y',how='all').dropna('x',how='all')
        
        # Choose grid shape: (rows, cols)
        rows, cols = 5, 6

        # Only variables with 2D (y, x)
        vars2d = [v for v in ds_dem_masked.data_vars if (set(ds_dem_masked[v].dims) == {"y","x"} and not "pisr" in v)]

        total = rows * cols
        if len(vars2d) > total: print(f"Note: showing first {total} of {len(vars2d)} variables.")
        
        vars2d = sorted(vars2d[:total])

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8), squeeze=False)

        i = 0
        for r in range(rows):
            for c in range(cols):
                ax = axes[r, c]
                ax.set_axis_off()
                if i < len(vars2d):
                    v = vars2d[i]
                    da = ds_dem_masked[v]
                    da.plot(ax=ax, cmap="jet", robust=True, add_colorbar=False)

                    # Build short, bold label
                    label = v.replace("St_", "").upper()

                    # Text box in axes coords (top-left), with white background
                    ax.text(
                        0.01, 0.99, label,
                        transform=ax.transAxes,
                        va="top", ha="left",
                        fontsize=8, fontweight="bold",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=1, boxstyle="round,pad=0.2"),
                        path_effects=[pe.withStroke(linewidth=1.0, foreground="black", alpha=0.25)]
                    )
                    i += 1

        # Tight layout, minimal gaps
        plt.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995, wspace=0.005, hspace=0.005)

        # Save/show
        # fig.savefig(rslt_dir+f'fig_static_features_{region}.pdf')
        fig.savefig(rslt_dir+f'fig_static_features_{region}_{version}.png', dpi=200)
        plt.show(); plt.clf(); plt.close('all')

