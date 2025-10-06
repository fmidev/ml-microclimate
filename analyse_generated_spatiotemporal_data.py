

# Read modules
import sys, ast, importlib, datetime, itertools, os, random, glob, joblib
import numpy as np
import pandas as pd
import xarray as xr; xr.set_options(file_cache_maxsize=1)

import s3fs

from datetime import timedelta

import xgboost as xgb



from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.utils import resample



import matplotlib; matplotlib.use('agg')
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


logger_data = pd.read_csv(rslt_dir+'logger_data_selected.csv', index_col=False, parse_dates=['time'])

#logger_region = logger_data.loc[logger_data['region']==region]


dem_data = {}
dem_mask = {}
for region in regions:
    print('Reading DEM for',region)
    ds_dem = fcts.read_dem('/lustre/tmp/kamarain/resiclim-microclimate', region).drop('spatial_ref')
    
    # Combine all variables into a DataArray where True = not NaN
    valid_mask = xr.concat([~ds_dem[var].isnull() for var in ds_dem.data_vars], dim="vars")
    
    # Reduce across the new 'vars' dimension — keep only where all variables are valid
    all_valid = valid_mask.all(dim="vars")
    
    # Create final mask: 1 where valid, NaN where at least one is NaN
    nan_mask = xr.where(all_valid, 1, np.nan)
    
    
    dem_data[region] = ds_dem
    dem_mask[region] = nan_mask








f1, axes1 = plt.subplots(2,3, figsize=(16,12))#, sharex=True, sharey=True)
for ax1,region in zip(axes1.ravel(), regions):
    print(region)
    files = np.sort(glob.glob(rslt_dir+f'generated_data/data_{region}_*.nc'))
    ds = xr.open_mfdataset(files)
    
    
    ds_dem = dem_data[region] #fcts.read_dem(fs, region).drop('spatial_ref'); 
    nan_mask = dem_mask[region] 
    ds_mask = nan_mask.interp_like(ds)
    
    
    ds = ds*ds_mask
    
    for vrb in ('T1','T2','T3'):
        ds[f"{vrb}_spatial_anomaly"] = ds[f"{vrb}_predicted"] - ds[f"{vrb}_predicted"].median(dim=["y", "x"])
        
        #climatology = ds.groupby('time.month').median('time')
        #ds[f"{vrb}_tempral_anomaly"] = ds[f"{vrb}_predicted"].groupby('time.month') - climatology[f"{vrb}_predicted"]
    
    xlim = [float(ds.x.min().values), float(ds.x.max().values)]
    ylim = [float(ds.y.min().values), float(ds.y.max().values)]
    
    height_contours = ds_dem['dem10m']
    height_contours[:] = fcts.Gauss_filter(height_contours, (5,5))
    height_contours = height_contours.sel(x=slice(xlim[0], xlim[1]), y=slice(ylim[1], ylim[0]))
    height_contours[:] = nan_mask*height_contours    
    
    out = ds['T3_spatial_anomaly'].median(['time'])
    out[:] = fcts.Gauss_filter(out.values, (0.7,0.7))
    out[:] = ds_mask*out
    #out.plot.contourf(ax=ax1,cmap='RdYlBu_r',antialiased=True,center=0,vmin=-1.5,vmax=1.5,levels=21,extend='both'); #plt.show()
    out.plot.contourf(ax=ax1,cmap='RdYlBu_r',robust=True,center=0,levels=21,extend='both'); #plt.show()
    
    hc1 = height_contours.plot.contour(ax=ax1, colors='k', linewidths=0.5, levels=10)
    ax1.clabel(hc1, fontsize=7)
    
    ax1.set_xlim(xlim); ax1.set_ylim(ylim)
    ax1.set_xlabel(''); ax1.set_ylabel('')
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title(f'T3 spatial anomaly in {region}') 
    
    f2, axes2 = plt.subplots(1,3, figsize=(16,10))#, sharex=True, sharey=True)
    #for axs2,vrb in zip(axes2.T, ('T1','T2','T3')):
    for ax2,vrb in zip(axes2.ravel(), ('T1','T2','T3')):
    #for vrb in ('T1','T2','T3'):
        print(region,vrb)
        
        out_spat = ds[f'{vrb}_spatial_anomaly'].median(['time'])
        out_spat[:] = fcts.Gauss_filter(out_spat.values, (0.7,0.7))
        out_spat[:] = ds_mask*out_spat
        
        #out_tmpr = ds[f'{vrb}_tempral_anomaly'].median(['time'])
        #out_tmpr[:] = fcts.Gauss_filter(out_tmpr.values, (0.7,0.7))
        #out_tmpr[:] = ds_mask*out_tmpr
        
        #out.plot.contourf(ax=ax2,cmap='RdYlBu_r',antialiased=True,center=0,vmin=-1.5,vmax=1.5,levels=21,extend='both'); #plt.show()
        out_spat.plot.contourf(ax=ax2[0],cmap='RdYlBu_r',center=0,levels=11,extend='both'); #plt.show()
        #out_tmpr.plot.contourf(ax=axs2[1],cmap='RdYlBu_r',center=0,levels=11,extend='both'); #plt.show()
        
        #for ax2 in axs2:
        hc2 = height_contours.plot.contour(ax=ax2, colors='k', linewidths=0.5, levels=10, robust=True,extend='both')
        ax2.clabel(hc2, fontsize=7)
        ax2.set_xlim(xlim); ax2.set_ylim(ylim)
        ax2.set_xlabel(''); ax2.set_ylabel('')
        ax2.set_xticks([]); ax2.set_yticks([])
        
        ax2[0].set_title(f'{vrb} spatial anomaly in {region}')
        #axs2[1].set_title(f'{vrb} temporal anomaly in {region}')
        
        
    f2.tight_layout()
    f2.savefig(rslt_dir+'fig_anomaly_'+region+'_'+str(datetime.datetime.now().date())+'.pdf')
    f2.savefig(rslt_dir+'fig_anomaly_'+region+'_'+str(datetime.datetime.now().date())+'.png', dpi=200)
    #f2.show(); 
    f2.clf(); #f2.close('all')
    
    """
    f2, axes2 = plt.subplots(4,3, figsize=(16,16))#, sharex=True, sharey=True)
    #for ax2,ssn,vrb in zip(axes2.ravel(), ['DJF','MAM','JJA','SON'], ('T1','T2','T3')):
    k=0
    for ssn,vrb in itertools.product(['DJF','MAM','JJA','SON'], ('T1','T2','T3')):
        ax2 = axes2.ravel()[k]
        print(region,ssn,vrb,k)
        t_idx = ds['time.season'] == ssn
        out = ds[f'{vrb}_anomaly'].sel(time=t_idx).median(['time'])
        out[:] = fcts.Gauss_filter(out.values, (0.7,0.7))
        out[:] = ds_mask*out
        
        out.plot.contourf(ax=ax2,cmap='RdYlBu_r',antialiased=True,center=0,vmin=-2.0,vmax=2.0,levels=21,extend='both'); #plt.show()
        #out.plot.contourf(ax=ax2,cmap='RdYlBu_r',antialiased=True,center=0,levels=21,extend='both'); #plt.show()
        
        hc2 = height_contours.plot.contour(ax=ax2, colors='k', linewidths=0.5, levels=10)
        ax2.clabel(hc2, fontsize=7)
        
        ax2.set_xlim(xlim); ax2.set_ylim(ylim)
        ax2.set_xlabel(''); ax2.set_ylabel('')
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_title(f'{vrb} {ssn} anomaly in {region}')
        k +=1
    
    f2.tight_layout()
    f2.savefig(rslt_dir+'fig_anomaly_seasonal_'+region+'_'+str(datetime.datetime.now().date())+'.pdf')
    f2.savefig(rslt_dir+'fig_anomaly_seasonal_'+region+'_'+str(datetime.datetime.now().date())+'.png', dpi=200)
    #f2.show(); 
    f2.clf(); #f2.close('all')
    """


f1.tight_layout()
f1.savefig(rslt_dir+'fig_anomaly_regions_'+str(datetime.datetime.now().date())+'.pdf')
f1.savefig(rslt_dir+'fig_anomaly_regions_'+str(datetime.datetime.now().date())+'.png', dpi=200)
f1.show(); f1.clf(); plt.close('all')



ds['T3'].mean(['x','y']).plot(); ds['T2m_ERA5'].mean(['x','y']).plot(); #plt.show()

ds['T3'].sel(x=ds.x.max(), y=ds.y.max()).plot(); ds['T2m_ERA5'].sel(x=ds.x.max(), y=ds.y.max()).plot(); #plt.show()




ds['smoothed_T3'] = xr.full_like(ds['T3'], np.nan); ds['smoothed_T3'][:] = fcts.Gauss_filter(ds['T3'].values, (1,1,1))
smoothed = fcts.Gauss_filter(ds['T3'].mean(['time']))
ds['smoothed_T3'].mean(['time']).plot(cmap='jet',robust=True); #plt.show()

ds_tmean = ds.mean(['time']).compute()
ds_smean = ds.mean(['x','y']).compute()



ds['T2m_ERA5'].mean(['time']).plot(cmap='jet'); #plt.show()



"""
plt.scatter(x=site_points['x'].values, y=site_points['y'].values, 
            c='red',label='Logger locations',edgecolors='k',alpha=1)

plt.legend(loc='upper right')

plt.tight_layout(); 
f.savefig(rslt_dir+f'fig_site_points_loggers_selected_{region}.pdf')
f.savefig(rslt_dir+f'fig_site_points_loggers_selected_{region}.png', dpi=200)
#plt.show(); 
plt.clf(); plt.close('all')
"""



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


