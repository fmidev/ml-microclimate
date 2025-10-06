

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




region      = 'VAR'
val_year    = 2025

fit         = False
optimize    = False
#additional_sampling = True







# Read command line arguments
#vrbs = ast.literal_eval(sys.argv[1])
#fit = ast.literal_eval(sys.argv[2])

region      = str(sys.argv[1])
val_year    = int(sys.argv[2])

optimize    = ast.literal_eval(sys.argv[3])
fit         = ast.literal_eval(sys.argv[4])




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
coords = pd.read_csv(rslt_dir+f'logger_locations_sample_22-09-2025.csv', index_col=False)



# Logger data
logger_data = pd.read_csv(rslt_dir+'logger_data_selected_22-09-2025.csv', index_col=False, parse_dates=['time'])
#logger_data = logger_data.loc[logger_data['error_tomst']==0]

# Extract the coordinate data of the sites in different regions
site_points = logger_data.groupby('site').mean(numeric_only=True)[['x','y','lon','lat']]

# For interpolation, create a helper xarray dataset with the same dimensions as in logger_data
interp_points = xr.Dataset({"time": ("points", logger_data["time"].values), 
                            "lat": ("points", logger_data["lat"].values), 
                            "lon": ("points", logger_data["lon"].values),
                            "y": ("points", logger_data["y"].values), 
                            "x": ("points", logger_data["x"].values),})





"""

# Read preprocessed DEM data from netcdf files
dem_data = []
plot_examples = True
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
    
    if plot_examples:
        lgr = coords.loc[coords['area']==region]
        
        f, ax = plt.subplots(1,1, figsize=(9,7))
        
        #dem['top_posn_idx_w3'].plot(cmap='RdGy',robust=True, alpha=0.8)
        #ds_dem['top_posn_idx_w3'].plot(cmap='RdGy',robust=True, alpha=0.8)
        m = ds_dem['dem10m'].plot.contourf(ax=ax, cmap='copper',robust=True, 
                                           alpha=1, levels=30, 
                                           cbar_kwargs={'label': 'Elevation [m]'})
        
        m.set_rasterized(True)
        
        #ds['150cm_temp'].mean(['time']).plot(cmap='nipy_spectral',robust=True, alpha=0.8)
        #mask_ds.plot(alpha=0.5, add_colorbar=False)
        ax.scatter(x=lgr['X'].values, y=lgr['Y'].values, s=50,
                    c='blue',label='All logger locations',edgecolors='k',alpha=1)
        
        ax.scatter(x=site_points['x'].values, y=site_points['y'].values, s=10,
                    c='red',label='Sampled logger locations',edgecolors='red',alpha=1)
        
        ax.legend(loc='upper right')
        #ax.set_colorbar(m, loc='ll', label='Elevation [m]')
        
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        
        plt.tight_layout(); 
        f.savefig(rslt_dir+f'fig_site_points_loggers_selected_{region}.pdf')
        f.savefig(rslt_dir+f'fig_site_points_loggers_selected_{region}.png', dpi=200)
        #plt.show(); 
        plt.clf(); plt.close('all')




dem_ds = xr.merge(dem_data)

# Interpolate DEM data to logger locations
dem_ds_interp = dem_ds.interp(x=interp_points['x'], y=interp_points['y'], method='linear')

# Convert interpolated xarray Dataset to DataFrame
dem_data_df = dem_ds_interp.to_dataframe().reset_index().drop(columns=['x','y','points'])

dem_data_df.to_csv(rslt_dir+'dem_data_selected.csv', index=False)



"""

#dem_data_df = pd.read_csv(rslt_dir+'dem_data_all.csv', index_col=False)

dem_data_df = pd.read_csv(rslt_dir+'dem_data_selected_22-09-2025.csv', index_col=False)




fcts.print_ram_state()



# Read ERA5 predictor data for the Northern Finland
lat_range, lon_range = [62, 72], [19, 32]
era5_data = fcts.read_era5(era5_dir, era5_vars, help_vars, lat_range, lon_range, 
                           lags, pd.date_range('2019-01-01','2025-08-31',freq='1h'))

era5_static = xr.open_dataset(era5_dir+'era5_static_surface_variables.nc')
era5_static = fcts.adjust_lats_lons(era5_static).sel(lat=slice(lat_range[0], lat_range[1]), lon=slice(lon_range[0], lon_range[1]))
era5_static = era5_static.drop(['valid_time','number','expver']).squeeze()
for v in era5_static.data_vars:
    era5_static = era5_static.rename({v: 'E5_'+v+'_static'})


era5_data =  xr.merge([era5_data, era5_static])
#    ds_era5_static = xr.open_dataset(f'{file_path}/era5_static_surface_variables.nc').load() 

fcts.print_ram_state()





#site_points_era5 = fcts.nearest_gridpoints(era5_data, site_points)







# Interpolate all variables in era5_data at the site locations
era5_data_interpolated = era5_data.interp(
    time=interp_points['time'],
    lat=interp_points['lat'],
    lon=interp_points['lon'],
    method='linear')


# Convert interpolated xarray Dataset to DataFrame
era5_data_interpolated_df = era5_data_interpolated.to_dataframe().reset_index().drop(columns=['points','number','time','lat','lon'])


# Define temporal cycle predictors
cyclical_df = fcts.cyclical_predictors(pd.DataFrame(index=logger_data.time), ann=True, diu=True, chn=True).reset_index().drop(columns='time')




# Merge all data together
all_data = pd.concat([  logger_data.reset_index(drop=True), 
                        era5_data_interpolated_df.reset_index(drop=True), 
                        dem_data_df.drop(columns=['x','y','time','site']).reset_index(drop=True),
                        cyclical_df.reset_index(drop=True)], axis=1)





# Names of different variables
target_variables = ['T1', 'T2', 'T3']
sptial_variables = ['lat','lon','y','x','region','site'] # ['lat','lon','y','x','region','site']
reanal_variables = list(era5_data_interpolated_df.columns)
static_variables = list(dem_data_df.drop(columns=['x','y','time','site']).columns)
cyclic_variables = list(cyclical_df.columns)
era5_variables = ['E5_t2m_degC', 'E5_skt_degC']

all_data['E5_t2m_degC'] = all_data['E5_t2m_+000'] - 273.15
all_data['E5_skt_degC'] = all_data['E5_skt_+000'] - 273.15



# Define large-scale ERA5 variables, and calculate offset variables based on them 
offset_variables = []
for v in target_variables:
    offset_variable = v+'_offset'
    all_data[offset_variable] = all_data[v] - all_data['E5_skt_degC'] #(all_data['E5_t2m_+000'] - 273.15)
    offset_variables.append(offset_variable)






# Define X and Y data for fitting
X = all_data[['time'] + sptial_variables + reanal_variables + static_variables + cyclic_variables]
Y = all_data[['time'] + sptial_variables + target_variables + offset_variables + era5_variables]


fcts.print_ram_state()







# Create target variable names and columns to the target dataframe
xgboost_variables = []
for v in offset_variables:
    Y.loc[:,'xgboost_'+v] = np.nan
    xgboost_variables.append('xgboost_'+v)

lasso_variables = []
for v in offset_variables:
    Y.loc[:,'lasso_'+v] = np.nan
    lasso_variables.append('lasso_'+v)


print(X)
print(Y)













#all_yrs = np.arange(2022,2024).astype(int)
all_yrs = np.arange(2019,2026).astype(int)



trn_years = all_yrs.copy(); trn_years = np.delete(trn_years, trn_years==val_year)
val_years = [val_year]

#val_year = val_years[0]
#trn_regions = regions[trn_idx_spt]
#val_regions = regions[val_idx_spt]


trn_regions = regions.copy(); trn_regions.remove(region)
val_regions = [region]

trn_idx = np.isin(pd.to_datetime(X['time'].values).year, trn_years) & np.isin(X['region'].values, trn_regions) & ~np.isnan(Y[offset_variables]).any(axis=1)
val_idx = np.isin(pd.to_datetime(X['time'].values).year, val_years) & np.isin(X['region'].values, val_regions) & ~np.isnan(Y[offset_variables]).any(axis=1)    
#trn_idx = np.isin(X['region'].values, trn_regions) & ~np.isnan(Y[offset_variables]).any(axis=1)
#val_idx = np.isin(X['region'].values, val_regions) & ~np.isnan(Y[offset_variables]).any(axis=1)

if val_idx.sum() == 0: 
    print('No data to validate, skipping',val_years,region)
    sys.exit('Exit')
    #print('No data to validate, skipping',val_years,region)
    #continue
    
#print('Train:', trn_years, trn_regions, trn_idx.sum())
#print('Valid:', val_years, val_regions, val_idx.sum())


drop_cols = ['region','site','x','y']
X_trn = X.drop(columns=drop_cols).loc[trn_idx]; Y_trn = Y[offset_variables].loc[trn_idx]
X_val = X.drop(columns=drop_cols).loc[val_idx]; Y_val = Y[offset_variables].loc[val_idx]



print('Train:', trn_regions, trn_idx.sum(), X_trn.shape, trn_years)
print('Valid:', val_regions, val_idx.sum(), X_val.shape, val_years)
print('\n')



fcts=importlib.reload(fcts)
t = fcts.stopwatch('start')

# Optimize hyperparameters and save them
if optimize:
    
    sample = Y_trn.sample(1000000).index
    X_trn_tuning, Y_trn_tuning = X_trn.loc[sample], Y_trn.loc[sample]
    
    # Note the hyperparameters are NOT tuned against the validation year ...
    best_params, results_df = fcts.tune_hyperparams(X_trn_tuning, Y_trn_tuning, num_trials=200)
    
    # ... but the validation year value is used to identify the current tuning
    results_df.to_csv(rslt_dir+f'optuna_results_{region}_{val_year}.csv', index=False)
    with open(rslt_dir+f'best_params_{region}_{val_year}.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    
    print(f'Optimization of {target_variables} for {region} {val_year} took {fcts.stopwatch("stop", t)} – now quitting')
    sys.exit()
    

# Use preoptimized or default hyperparams
if not optimize:
    param_file = glob.glob(rslt_dir+f'best_params_{region}_{val_year}.pkl')
    if len(param_file) > 0:
        # Use the pre-optimized parameters
        with open(param_file[0], 'rb') as f:
            best_params = pickle.load(f)
        
        base_estim = fcts.define_xgb(best_params)
    else:
        # Use the default, non-optimized parameters
        base_estim = fcts.xgb_default_estim()
        #base_estim.set_params(max_depth=7)

    import multiprocessing
    n_jobs = multiprocessing.cpu_count()
    base_estim.set_params(base_score=float(np.median(Y_trn)))
    base_estim.set_params(max_bin=256)
    base_estim.set_params(n_jobs=n_jobs)


if fit:
    
    # No early stopping 
    fitted_ensemble = fcts.fit_ensemble(
        X_trn.drop(columns='time'), Y_trn,
        X_val.drop(columns='time'), Y_val,
        base_estim, verbose=10,)
    
    model_file = rslt_dir+f'model_{region}_{val_year}.pkl'
    joblib.dump(fitted_ensemble, model_file)
    
    # LASSO reference model
    params_lasso = fcts.params_lasso()
    bagging_ens = fcts.bagging_model(X_trn.drop(columns='time'), Y_trn, params_lasso)
    
    model_file = rslt_dir+f'lasso_{region}_{val_year}.pkl'
    joblib.dump(bagging_ens, model_file)
    
    print(f'Fitting {target_variables} for {region} {val_year} took {fcts.stopwatch("stop", t)} – now quitting')
    sys.exit()
    
if not fit:
    # Load and use the previously trained models
    fitted_ensemble = joblib.load(rslt_dir+f'model_{region}_{val_year}.pkl')
    bagging_ens = joblib.load(rslt_dir+f'lasso_{region}_{val_year}.pkl')




# Save the forecast
Y.loc[val_idx,xgboost_variables] = fitted_ensemble.predict(X_val.drop(columns='time'))
Y.loc[val_idx,lasso_variables]   = bagging_ens.predict(X_val.drop(columns='time').ffill().bfill().fillna(0))




Y_val = Y.loc[val_idx]




# Back to original variables from offsets
for vt,vo in zip(target_variables,offset_variables):
    #y['predicted_'+vt] = y['predicted_'+vo] + (all_data.loc[val_idx,'E5_t2m_degC'] - 273.15)
    #y['predicted_'+vt] = y['predicted_'+vo] + all_data.loc[val_idx,'E5_skt_degC']
    Y_val['xgboost_'+vt] = Y_val['xgboost_'+vo] + all_data.loc[val_idx,'E5_skt_degC']
    Y_val['lasso_'+vt] = Y_val['lasso_'+vo] + all_data.loc[val_idx,'E5_skt_degC']


# Save data
Y_val.to_csv(rslt_dir+f'Y_{region}_{val_year}.csv', index=False)





print(region, val_year, 'XGB corr', fcts.calc_corr(Y_val['T3'], Y_val['xgboost_T3']))
print(region, val_year, 'XGB rmse', fcts.calc_rmse(Y_val['T3'], Y_val['xgboost_T3']), '\n')

print(region, val_year, 'LAS corr', fcts.calc_corr(Y_val['T3'], Y_val['lasso_T3']))
print(region, val_year, 'LAS rmse', fcts.calc_rmse(Y_val['T3'], Y_val['lasso_T3']), '\n')


#plt.plot(y[['T3','predicted_T3']]); plt.title(region); plt.show()
'''
f, axes = plt.subplots(1,1, figsize=(9,3))

axes.plot(y['T3']); axes.plot(y['predicted_T3'])
axes.set_title(region); 

plt.tight_layout(); 
f.savefig(rslt_dir+f'fig_timeseries_{region}.pdf')
f.savefig(rslt_dir+f'fig_timeseries_{region}.png', dpi=200)
#plt.show(); 
plt.clf(); plt.close('all')

# 11.04.2025
KIL 0.9141031648404148
KIL 3.0134917480033154 

PAL 0.9615762762763685
PAL 2.2146577582578444 

ULV 0.9631958707480079
ULV 2.2603513954855545 

VAR 0.949205169275256
VAR 2.2781565052021215 

TII 0.9473024258847191
TII 2.621772994669837 

RAS 0.9140269240093535
RAS 2.9154998095076623 


# xx.03.2025
KIL 0.9002052578820039
KIL 3.6158527541008665

PAL 0.9597825348749999
PAL 2.522305837848495

ULV 0.9493468061959379
ULV 2.982234393699733

VAR 0.9439998719389615
VAR 2.452484622064946

'''





# SHAP 


# Group the data embedded in the individual predictors 
global_scores   = pd.DataFrame(columns=['Predictor variable', 'Mean SHAP'])
cyclical_scores = pd.DataFrame(columns=['Predictor variable', 'Mean SHAP'])
static_scores   = pd.DataFrame(columns=['Predictor variable', 'Mean SHAP'])
era5_scores     = pd.DataFrame(columns=['Predictor variable', 'Lag', 'Mean SHAP'])



mdl = fitted_ensemble


#X_shap = X_val.drop(columns='time')
X_shap = X_trn.drop(columns='time').sample(500000, random_state=99)

features, importance, shap_values, explainer = fcts.extract_shap_values(X_shap, mdl, with_xgb=False)

df_importance = pd.DataFrame(index=features, data=importance, columns=['Mean SHAP'])

shap_values.to_csv(rslt_dir+f'shap_{region}_{val_year}.csv', index=False)

df_importance.to_csv(rslt_dir+f'importance_{region}_{val_year}.csv', index=False)



"""

gc_row=0; cc_row=0; st_row=0; gl_row=0
#for j,trgt in enumerate(vrbs):
for ft, sh in zip(features, importance):
    
    global_scores.loc[gl_row] = (ft, sh); gl_row += 1 
    
    if ft.split('_')[0]=='E5' and len(ft.split('_'))==3 and not 'static' in ft:
        _, prd, lag = ft.split('_')
        
        lag = int(lag)
        
        era5_scores.loc[gc_row] = ('E5_'+prd, lag, sh); gc_row += 1 
    
    if ft.split('_')[0]=='E5' and len(ft.split('_'))==4 and not 'static' in ft:
        _, prd, _, lag = ft.split('_')
        
        lag = int(lag)
        
        era5_scores.loc[gc_row] = ('E5_'+prd, lag, sh); gc_row += 1 
    
    if ft.split('_')[0]=='Cycle':
        prd = ft
        
        cyclical_scores.loc[cc_row] = (prd, sh); cc_row += 1
    
    if ft.split('_')[0]=='St' or ft=='lat' or ft=='lon' or 'static' in ft:
        prd = ft
        
        static_scores.loc[st_row] = (prd, sh); st_row += 1



#for trgt in vrbs:
    
mean_scores_glob   = global_scores.groupby('Predictor variable').mean().sort_values('Mean SHAP')['Mean SHAP']
mean_scores_cycl = cyclical_scores.groupby('Predictor variable').mean().sort_values('Mean SHAP')['Mean SHAP']
mean_scores_stat   = static_scores.groupby('Predictor variable').mean().sort_values('Mean SHAP')['Mean SHAP']
mean_scores_era5   = era5_scores.groupby('Predictor variable').mean().sort_values('Mean SHAP')['Mean SHAP']
mean_scores_lagd   = era5_scores.drop(columns=['Predictor variable']).groupby('Lag').mean()['Mean SHAP']

mean_scores = {'glob': mean_scores_glob,
               'cycl': mean_scores_cycl,
               'stat': mean_scores_stat,
               'era5': mean_scores_era5,
               'lagd': mean_scores_lagd,
               'comb': pd.concat([mean_scores_cycl, mean_scores_stat, mean_scores_era5]).sort_values()}


today = datetime.datetime.now().date()


for key in mean_scores:
    print('Plotting SHAP for',key)
    
    score = mean_scores[key]
    
    f, ax = plt.subplots(1,1,figsize=(6, np.sqrt(len(score)*5)))
    score.plot.barh(ax=ax, legend=False)
    ax.set_xlabel('mean(|SHAP|)')
    ax.set_title('Mean SHAP over fittings')
    ax.set_xscale('log')
    plt.tight_layout()
    f.savefig(rslt_dir+f'fig_SHAP_TOMS_{region}_{val_year}_{key}_{today}.pdf')
    f.savefig(rslt_dir+f'fig_SHAP_TOMS_{region}_{val_year}_{key}_{today}.png', dpi=200)
    #plt.show()
    plt.clf(); plt.close('all')


"""





import math
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay


#X_no_time_val = X_val.drop(columns='time')
#X_no_time_trn = X_trn.drop(columns='time')

target_idx = 2  # <- choose which output to plot


features_E5 = [v for v in X_shap.columns if ('E5' in v and '+000' in v)]
features_ST = [v for v in X_shap.columns if ('St_' in v or '_static' in v or v=='lat' or v=='lon')]
features_CY = [v for v in X_shap.columns if ('Cycle' in v)]# and not 'years' in v)]

names = ['ERA5', 'STATIC', 'CYCLES']

for name, features in zip(names, [features_E5, features_ST, features_CY]):

    ncols = math.ceil(np.sqrt(len(features)))
    nrows = math.ceil(len(features) / ncols)
    
    print(ncols,nrows,name,features)
    
    f, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)

    for i, v in enumerate(features):
        print(v)
        ax_i = axes.flat[i]
        PartialDependenceDisplay.from_estimator(
            mdl,
            X_shap,
            [v],
            kind='average',
            grid_resolution=50,
            target=target_idx,     # <-- key line
            #subsample=500000,
            ax=ax_i
        )
        ax_i.set_title(v)

    for j in range(len(features), nrows*ncols):
        axes.flat[j].set_visible(False)

    plt.tight_layout()
    f.savefig(rslt_dir + f'fig_PDP_{name}_{region}_{val_year}.png', dpi=200)
    plt.clf(); plt.close('all')



"""

features = [v for v in X_no_time_val.columns if ('St_' in v or '_static' in v or v=='lat' or v=='lon')]

ncols = 8
nrows = math.ceil(len(features) / ncols)
f, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)

for i, v in enumerate(features):
    print(v)
    ax_i = axes.flat[i]
    PartialDependenceDisplay.from_estimator(
        mdl,
        X_no_time_val,
        [v],
        kind='average',
        grid_resolution=50,
        target=target_idx,     # <-- key line
        ax=ax_i
    )
    ax_i.set_title(v)

for j in range(len(features), nrows*ncols):
    axes.flat[j].set_visible(False)

plt.tight_layout()
f.savefig(rslt_dir + f'fig_PDP_STATIC_{region}_{val_year}.png', dpi=200)
plt.clf(); plt.close('all')




features = [v for v in X_no_time_val.columns if ('Cycle' in v)]# and not 'years' in v)]

ncols = 3
nrows = math.ceil(len(features) / ncols)
f, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)

for i, v in enumerate(features):
    print(v)
    ax_i = axes.flat[i]
    PartialDependenceDisplay.from_estimator(
        mdl,
        X_no_time_trn,
        [v],
        kind='average',
        grid_resolution=50,
        target=target_idx,     # <-- key line
        ax=ax_i
    )
    ax_i.set_title(v)

for j in range(len(features), nrows*ncols):
    axes.flat[j].set_visible(False)

plt.tight_layout()
f.savefig(rslt_dir + f'fig_PDP_CYCLIC_{region}_{val_year}.png', dpi=200)
plt.clf(); plt.close('all')









f, axes = plt.subplots(2,3,figsize=(6,6))

axes = axes.flatten()

ax_idx = 0
for v in X_sample.drop(columns=['time','gcl',]).columns:
    print(v)
    if 'Cy' in v:
        ax = axes[ax_idx]
        
        f_ax, p = shap.partial_dependence_plot(
                v,
                mdl.predict,
                X_sample.drop(columns=['time','gcl',]),
                ice=False,
                model_expected_value=True,
                feature_expected_value=True,
                #ax=axes[ax_idx], 
                show=False)
        
        ax.plot(p)
        
        print(v, ax_idx, ax, f_ax, p)
        ax_idx += 1


plt.tight_layout()
f.savefig(rslt_dir+'fig_SHAP_partialdependence_TOMS_CYCLES_'+trgt+'_'+str(datetime.datetime.now().date())+'.png', dpi=200)

plt.clf(); plt.close('all')

"""




