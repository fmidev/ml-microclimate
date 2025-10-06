

# Read modules
import sys, ast, importlib, datetime, itertools, os, random, glob, joblib
import numpy as np
import pandas as pd
import xarray as xr; xr.set_options(file_cache_maxsize=1)

import s3fs

from datetime import timedelta

import xgboost as xgb



# set spill directory
os.environ['DASK_TEMPORARY_DIRECTORY'] = '/lustre/tmp/kamarain/dask-spill' 

# Launch a Dask client
#from dask.distributed import Client
#client = Client(n_workers=12, threads_per_worker=1, memory_limit='32GB')
#print(client)




from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.utils import resample

from sklearn.metrics import mean_squared_error, mean_absolute_error



#import matplotlib; matplotlib.use('agg')
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


regions = ['RAS', 'TII', 'KIL', 'PAL', 'ULV', 'VAR']

temp_levels = {'T1': '-6', 'T2': '0', 'T3': '15'}


today = str(datetime.datetime.now().date())



# Read own module
sys.path.append(code_dir)
import functions as fcts
fcts=importlib.reload(fcts)




# Set up the S3 file system
access = os.getenv('S3_RESIC_FMI_ACCESS') 
secret = os.getenv('S3_RESIC_FMI_SECRET') 

fs = s3fs.S3FileSystem(anon=False, key=access, secret=secret,
    client_kwargs={'endpoint_url': 'https://a3s.fi'})











#logger_data = pd.read_csv(rslt_dir+'logger_data_selected.csv', index_col=False, parse_dates=['time'])






# ----------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------
'''
def compute_corr_rmse(obs, pred):
    """Compute Pearson correlation and RMSE between obs and pred."""
    corr = fcts.calc_corr(obs, pred)
    rmse = fcts.calc_rmse(obs, pred)
    return corr, rmse

def compute_mean_std(arr):
    """Compute mean and std of an array."""
    return np.mean(arr), np.std(arr)
'''

def corr_rmse_text(obs, input_dict): #era, xgb, rfr):
    """Return a formatted string with correlation and RMSE for ERA5 & XGB."""
    out = []
    for key in input_dict.keys():
        cor = fcts.calc_corr(obs, input_dict[key])
        rms = fcts.calc_rmse(obs, input_dict[key])
        
        out.append(f"{key}: CORR={cor:.2f}, RMSE={rms:.2f}\n")
    
    return ''.join(out)

"""
    c_era, r_era = compute_corr_rmse(obs, era)
    c_xgb, r_xgb = compute_corr_rmse(obs, xgb)
    c_rfr, r_rfr = compute_corr_rmse(obs, rfr)
    return (f"ERA5: Corr={c_era:.2f}, RMSE={r_era:.2f}\n"
            f"MCLF: Corr={c_rfr:.2f}, RMSE={r_rfr:.2f}\n"
            f"XGB: Corr={c_xgb:.2f}, RMSE={r_xgb:.2f}")
"""

def mean_std_text(input_dict): #err_era, err_xgb, err_rfr):
    """Return a formatted string with mean and std for ERA5 & XGB errors."""
    
    out = []
    for key in input_dict.keys():
        mea = np.nanmean(input_dict[key])
        std = np.nanstd(input_dict[key])
        
        out.append(f"{key}: MEAN={mea:.2f}, STD={std:.2f}\n")
    
    return ''.join(out)
    
"""    
    m_era, s_era = compute_mean_std(err_era)
    m_xgb, s_xgb = compute_mean_std(err_xgb)
    m_rfr, s_rfr = compute_mean_std(err_rfr)
    return (f"ERA5 error: mean={m_era:.2f}, std={s_era:.2f}\n"
            f"MCLF error: mean={m_rfr:.2f}, std={s_rfr:.2f}\n"
            f"XGB error: mean={m_xgb:.2f}, std={s_xgb:.2f}")
"""









read_computed_results = False

if read_computed_results:
    
    refr_pths = {'KIL': '/lustre/tmp/kamarain/resiclim-microclimf/KIL_AIL/*',
                 'PAL': '/lustre/tmp/kamarain/resiclim-microclimf/PAL/*',
                 'VAR': '/lustre/tmp/kamarain/resiclim-microclimf/VAR/*',}
    
    Y = []
    for region in regions:
        print('Reading',region)
        # Read data
        y = []
        for year in ['2019','2020','2021','2022','2023','2024','2025']:
            try:
                df = pd.read_csv(rslt_dir+f'Y_{region}_{year}.csv', parse_dates=['time'])
                y.append(df)
            except:
                print('No data for',region,year)
        
        y = pd.concat(y) 
        
        
        #if region in ['KIL', 'PAL', 'VAR']:
        if region in ['KIL', 'VAR']:
            #ds_microclimf = xr.open_mfdataset(refr_pths[region], chunks={}, combine='by_coords').drop('crs')
            ds_microclimf = xr.open_mfdataset(refr_pths[region], chunks=None).drop('crs').sortby('time')
            
            
            # Step 2: Extract unique site coordinates
            site_coords = y[['site', 'x', 'y']].drop_duplicates().set_index('site')

            # Step 3: Create a site-aware Dataset for spatial interpolation
            # Here, site is a coordinate with string labels
            site_coords_xr = xr.Dataset({
                'x': ('site', site_coords['x']),
                'y': ('site', site_coords['y']),})
            
            site_coords_xr = site_coords_xr.assign_coords(site=site_coords.index)

            # Step 4: Interpolate only in space (to get time x site grid)
            ds_site_temp = ds_microclimf.interp(
                x=site_coords_xr['x'],
                y=site_coords_xr['y'],
                method='linear')['15cm_temp'] / 100.
            

            # Step 5: Convert the result to a tidy DataFrame
            df_site_temp = ds_site_temp.to_dataframe().reset_index()  # columns: time, site, 15cm_temp
            
            # Step 6: Merge back to `y` based on time and site
            y = y.merge(df_site_temp[['time','site','15cm_temp']], on=['site', 'time'], how='left')

            # Step 7: Rename to final output column
            y = y.rename(columns={'15cm_temp': 'microclimf_T3'})
            
            # Destroy results from outside the Microclimf period to increase comparability
            y = y.dropna()
        
        fcts.print_ram_state()
        Y.append(y)
        
    Y = pd.concat(Y)
    #Y.sample(100000).sort_values(by=['region','site','time']).to_csv('Y_sample.csv')

    Y.sort_values(by=['region','site','time']).to_csv(f'{rslt_dir}Y.csv', index=False)










if not read_computed_results:
    Y = pd.read_csv(f'{rslt_dir}Y.csv', index_col=False)




# Data for text
y = Y.dropna(axis=0)
for name, mth_grouping in zip(['ALL','SUMMER'], [range(1,13), range(5,10)]):
    # All data
    idx = np.isin(pd.to_datetime(y.time.values).month, mth_grouping)
    round_decimals = 1
    print(f'{name} mths XGBo rmse:', fcts.calc_rmse(y.loc[idx,'T3'], y.loc[idx,'xgboost_T3']).round(round_decimals))
    print(f'{name} mths LSSO rmse:', fcts.calc_rmse(y.loc[idx,'T3'], y.loc[idx,'lasso_T3']).round(round_decimals))
    print(f'{name} mths ERA5 rmse:', fcts.calc_rmse(y.loc[idx,'T3'], y.loc[idx,'E5_skt_degC']).round(round_decimals))
    if 'microclimf_T3' in y: 
        print(f'{name} mths MCLF rmse:', fcts.calc_rmse(y.loc[idx,'T3'], y.loc[idx,'microclimf_T3']).round(round_decimals))

    round_decimals = 2
    print(f'{name} mths XGBo corr:', fcts.calc_corr(y.loc[idx,'T3'], y.loc[idx,'xgboost_T3']).round(round_decimals))
    print(f'{name} mths LSSO corr:', fcts.calc_corr(y.loc[idx,'T3'], y.loc[idx,'lasso_T3']).round(round_decimals))
    print(f'{name} mths ERA5 corr:', fcts.calc_corr(y.loc[idx,'T3'], y.loc[idx,'E5_skt_degC']).round(round_decimals))
    if 'microclimf_T3' in y: 
        print(f'{name} mths MCLF corr:', fcts.calc_corr(y.loc[idx,'T3'], y.loc[idx,'microclimf_T3']).round(round_decimals))

    print('\n')



# Data for tables
for name, mth_grouping in zip(['ALL','SUMMER'], [range(1,13), range(5,10)]):
    for region in regions:
        
        y = Y.loc[Y.region==region].dropna(axis=1, how='all').dropna(axis=0)

        idx = np.isin(pd.to_datetime(y.time.values).month, mth_grouping)
        round_decimals = 1
        print(f'{region} {name} mths XGBo rmse:', fcts.calc_rmse(y.loc[idx,'T3'], y.loc[idx,'xgboost_T3']).round(round_decimals))
        print(f'{region} {name} mths LSSO rmse:', fcts.calc_rmse(y.loc[idx,'T3'], y.loc[idx,'lasso_T3']).round(round_decimals))
        print(f'{region} {name} mths ERA5 rmse:', fcts.calc_rmse(y.loc[idx,'T3'], y.loc[idx,'E5_skt_degC']).round(round_decimals))
        if 'microclimf_T3' in y: 
            print(f'{region} {name} mths MCLF rmse:', fcts.calc_rmse(y.loc[idx,'T3'], y.loc[idx,'microclimf_T3']).round(round_decimals))

        round_decimals = 2
        print(f'{region} {name} mths XGBo corr:', fcts.calc_corr(y.loc[idx,'T3'], y.loc[idx,'xgboost_T3']).round(round_decimals))
        print(f'{region} {name} mths LSSO corr:', fcts.calc_corr(y.loc[idx,'T3'], y.loc[idx,'lasso_T3']).round(round_decimals))
        print(f'{region} {name} mths ERA5 corr:', fcts.calc_corr(y.loc[idx,'T3'], y.loc[idx,'E5_skt_degC']).round(round_decimals))
        if 'microclimf_T3' in y: 
            print(f'{region} {name} mths MCLF corr:', fcts.calc_corr(y.loc[idx,'T3'], y.loc[idx,'microclimf_T3']).round(round_decimals))

        print('\n')






#df_eval = Y.rename(columns={"T3": "obs","xgboost_T3": "pred"}).set_index(pd.to_datetime(Y["time"]).dt.tz_localize("UTC"))

#df_eval = df_eval[["obs", "pred", "site", "lat", "lon", "region", "x", "y"]]

import spatiotemporal_skill_and_bioclimatic_indicators as spt

#from arctic_temp_skill import ArcticTempSkill

#ats = spt.ArcticTempSkill(Y.dropna(), obs_col="T3", model_cols=["xgboost_T3","lasso_T3","microclimf_T3",'E5_skt_degC'])
ats = spt.ArcticTempSkill(Y, obs_col="T3", model_cols=["xgboost_T3","lasso_T3","microclimf_T3",'E5_skt_degC'])

"""
df = ats.df.copy()
for col in ["xgboost_T3","lasso_T3","microclimf_T3",'E5_skt_degC']:
    r = df[col] - df["T3"]
    r_demean = r - r.groupby(df["site"]).transform("mean")
    df[col + "_demean_pred"] = df["T3"] + r_demean  # same units; semivariogram will use residual = model - obs
ats_d = spt.ArcticTempSkill(df, obs_col="T3", model_cols=[c+"_demean_pred" for c in ["xgboost_T3","lasso_T3","microclimf_T3",'E5_skt_degC']])
# recompute SS(h) with these *_demean_pred columns
ats = ats_d
"""



# 1) Overall and by-region core skill
overall = ats.summarize_regression()
by_region = ats.summarize_regression(groupby="region")
by_month = ats.summarize_regression(groupby=["month","site"])
by_allgroups = ats.summarize_regression(groupby=["month","region","site"])


sns.boxplot(by_allgroups, x='region', y='R', hue='model'); plt.show()
sns.boxplot(by_allgroups, x='month', y='R', hue='model'); plt.show()
#sns.boxplot(by_allgroups, x='site', y='R', hue='model'); plt.show()


fcts.plot_ridgeline(by_allgroups, metric="RMSE", group_col="model", max_groups=4, title="RMSE by model")


# 2) Frost-event skill at 0 °C (or e.g., -2 °C for severe frost)
frost_p0 = ats.frost_event_skill(threshold=0.0, groupby=["region","site"])
frost_m2 = ats.frost_event_skill(threshold=-2.0, groupby="region")

# 3) Bioclim indicators for May–Sep, base 0 °C
#bioclim = ats.bioclim_summary(months=(5,6,7,8,9), base_temp=0.0, frost_thresh=0.0)
bioclim = ats.bioclim_summary(months=tuple(range(1,13)), base_temp=0.0, frost_thresh=0.0)



indicators = ['FrostHours','GrowingDegreeHours','FreezeThawCycles',
              'MeanTemp','DiurnalAmplitude','SummerWarmthIndex','WinterColdnessIndex',
              'NearZeroEpisodeLength','NearZeroEpisodeCount']

for indicator in indicators:
    for region in regions:
        region_idx = bioclim['region'] == region
        data = bioclim.loc[region_idx,[ #f'obs_{indicator}', 
                                        f'xgboost_T3_{indicator}_bias', 
                                        f'lasso_T3_{indicator}_bias',
                                        f'microclimf_T3_{indicator}_bias',
                                        f'E5_skt_degC_{indicator}_bias']]#
        
        data.columns = ['XGBoost','LASSO','Microclimf','ERA5 SKT']
        #data.columns = ['Observed','XGBoost','LASSO','Microclimf','ERA5 SKT']
        
        #data.plot(); plt.title(f'{region}, {indicator}'); plt.show()
        
        sns.boxplot(data.dropna()); plt.title(f'{indicator}, {region}'); plt.show()


# 4) Microclimate-scale spatial skill (semivariogram up to 10 km), per region
vg_xgbs = ats.semivariogram(model="xgboost_T3", max_range_m=10_000, n_bins=50, n_hours=10000,
                       n_pairs_per_hour=5000, seed=42, by_region=False)

vg_lass = ats.semivariogram(model="lasso_T3", max_range_m=10_000, n_bins=50, n_hours=10000,
                       n_pairs_per_hour=5000, seed=42, by_region=False)

vg_mclf = ats.semivariogram(model="microclimf_T3", max_range_m=10_000, n_bins=50, n_hours=10000,
                       n_pairs_per_hour=5000, seed=42, by_region=False)
                       
vg_era5 = ats.semivariogram(model="E5_skt_degC", max_range_m=10_000, n_bins=50, n_hours=10000,
                       n_pairs_per_hour=5000, seed=42, by_region=False)



# Observational field variogram, matched to each model’s availability (important!)
vg_obsd = ats.field_variogram("T3", max_range_m=10_000, n_bins=50, n_hours=10000,
                                      n_pairs_per_hour=5000, seed=42, by_region=False)

def attach_obs(vg_err, vg_obsd):
    return vg_err.merge(
        vg_obsd.rename(columns={"semivariance":"gamma_obs","count":"obs_count"})[
            #["region","h_center_m","gamma_obs","obs_count"]],
            ["h_center_m","gamma_obs","obs_count"]],
        #on=["region","h_center_m"], how="left"
        on=["h_center_m"], how="left"
    ).rename(columns={"semivariance":"gamma_err","count":"err_count"})



cmp_xgbs = attach_obs(vg_xgbs, vg_obsd)
cmp_lass = attach_obs(vg_lass, vg_obsd)
cmp_mclf = attach_obs(vg_mclf, vg_obsd)
cmp_era5 = attach_obs(vg_era5, vg_obsd)


# Scale-dependent metrics
for df in (cmp_xgbs, cmp_lass, cmp_mclf, cmp_era5):
    df["ratio"] = df["gamma_err"] / df["gamma_obs"]          # <1 is good
    df.loc[(df["gamma_obs"]<=0) | ~np.isfinite(df["ratio"]), "ratio"] = np.nan
    df["SS"] = 1.0 - df["ratio"]                             # skill score, >0 is good
    df["pair_RMSE"] = np.sqrt(2.0 * df["gamma_err"])         # interpretable (°C) at each lag




for region in regions: #["KIL","VAR"]:
    fig, ax = plt.subplots()
    ax.axhline(0, lw=1)
    for name, df in [("microclimf", cmp_mclf), ("xgboost", cmp_xgbs), ("lasso", cmp_lass), ("ERA5 skt", cmp_era5)]:
        if 'region' in df.columns:
            sub = df[(df.region==region) & (df.err_count>=50) & (df.obs_count>=50)]  # guard low-sample bins
        else:
            sub = df[(df.err_count>=50) & (df.obs_count>=50)] 
            
        ax.plot(sub["h_center_m"]/1000.0, sub["SS"], label=name, lw=4)
        #ax.plot(sub["h_center_m"]/1000.0, sub["pair_RMSE"], label=name)
    
    if 'region' in df.columns:
        ax.set_title(f"{region} – scale-dependent skill SS(h)")
    else:
        ax.set_title(f"Scale-dependent skill SS(h)")
    #ax.set_title(f"{region} – scale-dependent rmse(h)")
    
    ax.set_xlabel("Lag h (km)")
    
    ax.set_ylabel("SS = 1 − γ_err/γ_obs")
    #ax.set_ylabel("RMSE")
    
    
    ax.legend()
    plt.show()
    


plt.plot(vg_xgbs['h_center_m'], vg_xgbs['semivariance'],c='b')

#plt.plot(vg_xgbs.loc[vg_xgbs.region==region,'h_center_m'], vg_xgbs.loc[vg_xgbs.region==region,'semivariance'])

plt.plot(vg_lass['h_center_m'], vg_lass['semivariance'],c='r')
plt.plot(vg_mclf['h_center_m'], vg_mclf['semivariance'],c='m')
plt.plot(vg_era5['h_center_m'], vg_era5['semivariance'],c='c')

plt.plot(vg_obsd['h_center_m'], vg_obsd['semivariance'],c='k',ls='--')

plt.show()




for region in regions: #["KIL","VAR"]:
    plt.plot(vg_xgbs.loc[vg_xgbs.region==region,'h_center_m'], vg_xgbs.loc[vg_xgbs.region==region,'semivariance'],c='b')

    #plt.plot(vg_xgbs.loc[vg_xgbs.region==region,'h_center_m'], vg_xgbs.loc[vg_xgbs.region==region,'semivariance'])
    
    plt.plot(vg_lass.loc[vg_lass.region==region,'h_center_m'], vg_lass.loc[vg_lass.region==region,'semivariance'],c='r')
    plt.plot(vg_mclf.loc[vg_mclf.region==region,'h_center_m'], vg_mclf.loc[vg_mclf.region==region,'semivariance'],c='m')
    plt.plot(vg_era5.loc[vg_era5.region==region,'h_center_m'], vg_era5.loc[vg_era5.region==region,'semivariance'],c='c')
    
    plt.plot(vg_obs_for_xgbs.loc[vg_obs_for_xgbs.region==region,'h_center_m'], vg_obs_for_xgbs.loc[vg_obs_for_xgbs.region==region,'semivariance'],c='k',ls='--')
    
    plt.show()



"""
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


cluster_results = []
for region in regions:
    df_region = Y[Y['region'] == region].copy()
    features = ['T3', 'predicted_T3', 'E5_skt_degC']
    df_features = df_region[features].dropna()

    if len(df_features) < 100:
        continue

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_result = df_features.copy()
    df_result['cluster'] = clusters
    df_result['region'] = region
    cluster_results.append(df_result)

# Yhdistetään tulokset
#df_clusters = pd.concat(cluster_results, ignore_index=True)

# Piirretään scatterplot
#for region in regions:
    print(region)
    plt.figure(figsize=(10, 6))
    #data = df_result # df_clusters.loc[df_clusters.region == region]
    #sns.scatterplot(data=df_clusters, x="T3", y="predicted_T3", hue="cluster", style="region", alpha=0.5)
    sns.scatterplot(data=df_result, x="T3", y="predicted_T3", hue="cluster", alpha=0.5)
    plt.title("Klusterointi: T3 vs. Predicted_T3")
    plt.xlabel("Havaittu lämpötila (T3)")
    plt.ylabel("Mallinnettu lämpötila (predicted_T3)")
    plt.legend(title="Klusteri")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(rslt_dir + f'fig_clustering_{region}.png', dpi=200)
    plt.show()
"""



# Optimization result
for region in regions:
    y = Y.loc[Y.region == region]

    # Optuna results
    df_result = pd.read_csv(rslt_dir+f'optuna_results_{region}_2022.csv', index_col=False)
    

    # List only the parameters that vary across trials:
    params = [
        "params_colsample_bytree",
        "params_learning_rate",
        "params_max_depth",
        "params_min_child_weight",
        "params_reg_alpha",
        "params_subsample",
    ]

    fig, axes = plt.subplots(
        nrows=len(params) + 1,  # One extra subplot for the RMSE values
        ncols=1,
        figsize=(10, 14),
        sharex=True
    )

    ##############################################################################
    # 1) RMSE vs. Trial (log scale)
    ##############################################################################
    ax_val = axes[0]
    ax_val.plot(
        df_result["number"],
        df_result["value"], 
        marker='o', 
        color='tab:Red',
        label="RMSE"
    )
    ax_val.set_title(f"Optuna trials in {region}")
    ax_val.set_yscale("log")  # Log scale for the RMSE axis
    ax_val.set_ylabel("log(RMSE)")
    ax_val.grid(True)
    ax_val.legend()

    ##############################################################################
    # 2) Parameter Subplots, color‐coded by RMSE (log scale in the color)
    ##############################################################################
    norm = LogNorm(vmin=df_result["value"].min(), vmax=df_result["value"].max())

    for i, param in enumerate(params, start=1):
        ax = axes[i]
        # Optional: draw a thin line connecting the points
        ax.plot(
            df_result["number"], 
            df_result[param], 
            lw=0.5,
            color='gray'
        )
        sc = ax.scatter(
            df_result["number"], 
            df_result[param], 
            c=df_result["value"], 
            cmap="plasma_r", 
            edgecolor='k', 
            s=60,
            norm=norm
        )
        ax.set_ylabel(param)
        ax.grid(True)

    axes[-1].set_xlabel("Trial Number")

    ##############################################################################
    # 3) Place colorbar outside the plotting area
    ##############################################################################
    # First, tighten the layout but leave some space on the right for the colorbar.
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # This shrinks the subplots to 85% of the figure width

    # Create a new Axes on the right for the colorbar. 
    # [x-position, y-position, width, height] in figure fraction.
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])

    # Draw the colorbar in this new Axes.
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("log(RMSE)")

    # Save figure and show
    fig.savefig(rslt_dir + f'fig_optimization_{region}_{today}.png', dpi=200)
    #plt.show()
    

    """
    data_cols = ['T1', 'T2', 'T3', 'T1_offset', 'T2_offset', 'T3_offset', 'fcs_T1_offset', 'fcs_T2_offset', 'fcs_T3_offset', 'fcs_T1', 'fcs_T2', 'fcs_T3']


    #y = Y.loc[Y['region']==region]
    #y_2022 = y.loc[pd.to_datetime(y.time.values).year==2022, ['time']+data_cols].groupby('time').mean()
    y_mean = y.loc[:, ['time']+data_cols].groupby('time').mean()
    
    print(region, fcts.calc_corr(y['T3'], y['fcs_T3']))
    print(region, fcts.calc_rmse(y['T3'], y['fcs_T3']), '\n')
    
    #plt.plot(y[['T3','fcs_T3']]); plt.title(region); plt.show()
    
    f, axes = plt.subplots(2,1, figsize=(12,9))
    
    axes[0].plot(y['T3']); axes[0].plot(y['fcs_T3'])
    axes[1].plot(y_mean['T3'])
    axes[1].plot(y_mean['fcs_T3'])
    
    axes[0].set_title(region+', T15cm, all data'); 
    axes[1].set_title(region+', T15cm, mean over logger sites'); 
    
    plt.tight_layout(); 
    f.savefig(rslt_dir+f'fig_timeseries_{region}.pdf')
    f.savefig(rslt_dir+f'fig_timeseries_{region}.png', dpi=200)
    plt.show(); 
    plt.clf(); plt.close('all')
    
    err_era_all = y['E5_t2m_+000_degC'] - y['T3']
    err_xgb_all = y['fcs_T3'] - y['T3']
    
    y_monthly_err_era = err_era_all.groupby([pd.to_datetime(y.time.values).month, y.site]).mean().unstack(level=0)
    y_monthly_err_xgb = err_xgb_all.groupby([pd.to_datetime(y.time.values).month, y.site]).mean().unstack(level=0)
    plt.imshow(y_monthly_err_xgb.values, aspect='auto', cmap='seismic'); plt.show()
    
    plt.plot(y_monthly_err_era.T, c='b'); #plt.show()
    plt.plot(y_monthly_err_xgb.T, c='r'); plt.show()
    """



# Error analysis
for region in regions:
    y = Y.loc[Y.region == region]
    
    # ----------------------------------------------------------------
    # Prepare aggregated data
    # ----------------------------------------------------------------
    y_tmean = (
        y[['time','T1','T2','T3','T1_offset','T2_offset','T3_offset',
           'E5_t2m_degC','E5_skt_degC',
           'xgboost_T1_offset','xgboost_T2_offset','xgboost_T3_offset',
           'xgboost_T1','xgboost_T2','xgboost_T3',
           'lasso_T1_offset','lasso_T2_offset','lasso_T3_offset',
           'lasso_T1','lasso_T2','lasso_T3',
           'microclimf_T3',]]
        .groupby('time')
        .mean()
    )

    y_smean = (
        y[['lat','lon','T1','T2','T3','T1_offset','T2_offset','T3_offset',
           'E5_t2m_degC','E5_skt_degC',
           'xgboost_T1_offset','xgboost_T2_offset','xgboost_T3_offset',
           'xgboost_T1','xgboost_T2','xgboost_T3',
           'lasso_T1_offset','lasso_T2_offset','lasso_T3_offset',
           'lasso_T1','lasso_T2','lasso_T3',
           'microclimf_T3',]]
        .groupby(['lat','lon'])
        .mean()
        .reset_index()
    )

    # ----------------------------------------------------------------
    # Compute error arrays
    # ----------------------------------------------------------------
    err_era_all = y['E5_skt_degC'] - y['T3']
    err_xgb_all = y['xgboost_T3'] - y['T3']
    err_lss_all = y['xgboost_T3'] - y['T3']
    err_rfr_all = y['microclimf_T3'] - y['T3']

    err_era_time = y_tmean['E5_skt_degC'] - y_tmean['T3']
    err_xgb_time = y_tmean['xgboost_T3'] - y_tmean['T3']
    err_lss_time = y_tmean['lasso_T3'] - y_tmean['T3']
    err_rfr_time = y_tmean['microclimf_T3'] - y_tmean['T3']

    err_era_site = y_smean['E5_skt_degC'] - y_smean['T3']
    err_xgb_site = y_smean['xgboost_T3'] - y_smean['T3']
    err_lss_site = y_smean['lasso_T3'] - y_smean['T3']
    err_rfr_site = y_smean['microclimf_T3'] - y_smean['T3']

    # ----------------------------------------------------------------
    # Define plotting configuration in a single structure
    # ----------------------------------------------------------------
    plot_data = [
        {
            "obs":     y["T3"].values.ravel(),
            "era":     y["E5_skt_degC"].values.ravel(),
            "xgb":     y["xgboost_T3"].values.ravel(),
            "lss":     y["lasso_T3"].values.ravel(),
            "rfr":     y["microclimf_T3"].values.ravel(),
            "err_era": err_era_all,
            "err_xgb": err_xgb_all,
            "err_lss": err_lss_all,
            "err_rfr": err_rfr_all,
            "title":   region + ", T15cm, all data",
            "xlabel":  "Data point",
            "bins":    30
        },
        {
            "obs":     y_tmean["T3"],
            "era":     y_tmean["E5_skt_degC"],
            "xgb":     y_tmean["xgboost_T3"],
            "lss":     y_tmean["lasso_T3"],
            "rfr":     y_tmean["microclimf_T3"],
            "err_era": err_era_time,
            "err_xgb": err_xgb_time,
            "err_lss": err_lss_time,
            "err_rfr": err_rfr_time,
            "title":   region + ", T15cm, mean over logger sites",
            "xlabel":  "Time",
            "bins":    30
        },
        {
            "obs":     y_smean["T3"],
            "era":     y_smean["E5_skt_degC"],
            "xgb":     y_smean["xgboost_T3"],
            "lss":     y_smean["lasso_T3"],
            "rfr":     y_smean["microclimf_T3"],
            "err_era": err_era_site,
            "err_xgb": err_xgb_site,
            "err_lss": err_lss_site,
            "err_rfr": err_rfr_site,
            "title":   region + ", T15cm, mean over time steps",
            "xlabel":  "Logger index",
            "bins":    10
        },
    ]

    # ----------------------------------------------------------------
    # Create the figure and axes
    # ----------------------------------------------------------------
    f, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(14, 9),
        gridspec_kw={'width_ratios': [3, 1]}
    )

    # A bbox style similar to the default legend style:
    legend_bbox = dict(boxstyle='round', facecolor='white', edgecolor='0.8', alpha=0.8)

    # ----------------------------------------------------------------
    # Loop over each row and plot
    # ----------------------------------------------------------------
    for (ax_ts, ax_err), data in zip(axes, plot_data):
        # 1) Time series (left column)
        ax_ts.plot(data["obs"], label='Observed', color='k', lw=2)
        ax_ts.plot(data["lss"], label='LASSO', color='m', lw=2, alpha=0.8)
        ax_ts.plot(data["era"], label='ERA5', color='tab:Red', alpha=0.8)
        
        if np.isnan(data['rfr']).sum() < len(data['rfr']): 
            ax_ts.plot(data["rfr"], label='MCLF', color='tab:Cyan', alpha=0.8)
        
        ax_ts.plot(data["xgb"], label='XGBoost', color='tab:Orange', alpha=0.8)
        ax_ts.set_title(data["title"])
        ax_ts.set_xlabel(data["xlabel"])
        ax_ts.legend(loc='lower right')

        # Annotate correlation & RMSE
        text_left = corr_rmse_text(data["obs"], {'ERA':data["era"], 'LSS':data["lss"], 'XGB':data["xgb"], 'MCF':data["rfr"]})
        ax_ts.text(
            0.01, 0.95, text_left,
            transform=ax_ts.transAxes,
            va='top', ha='left', fontsize=10,
            bbox=legend_bbox
        )

        # 2) Error distribution (right column)
        ax_err.hist(data["err_era"], bins=data["bins"], alpha=0.5,label='ERA5 error', color='tab:Red')
        ax_err.hist(data["err_lss"], bins=data["bins"], alpha=0.5,label='LASSO error', color='m')
        
        if len(data["err_rfr"].dropna() > 0): 
            ax_err.hist(data["err_rfr"], bins=data["bins"], alpha=0.5,label='MCLF error', color='tab:Cyan')
        
        ax_err.hist(data["err_xgb"], bins=data["bins"], alpha=0.5,label='XGBoost error', color='tab:Orange')
        
        ax_err.set_title("Error distribution")
        ax_err.legend(loc='lower right')

        # Annotate mean & std
        text_right = mean_std_text({'ERA err':data["err_era"], 'XGB err':data["err_xgb"], 'LSS err':data["err_lss"], 'MCF err':data["err_rfr"]})
        ax_err.text(
            0.05, 0.95, text_right,
            transform=ax_err.transAxes,
            va='top', ha='left', fontsize=10,
            bbox=legend_bbox
        )

    # ----------------------------------------------------------------
    # Final layout and save
    # ----------------------------------------------------------------
    plt.tight_layout()
    f.savefig(rslt_dir + f'fig_timeseries_{region}_{today}.pdf')
    f.savefig(rslt_dir + f'fig_timeseries_{region}_{today}.png', dpi=200)

    plt.clf()
    plt.close('all')






for region in regions:
    y = Y.loc[Y.region == region]
    
    # -------------------------------
    # Prepare monthly data for boxplots
    # -------------------------------
    # Ensure that 'time' is in datetime format and add a 'month' column.
    y['time'] = pd.to_datetime(y['time'])
    y['month'] = y['time'].dt.month

    # Create error columns if not already present.
    y['err_era'] = y['E5_skt_degC'] - y['T3']
    y['err_xgb'] = y['predicted_T3'] - y['T3']

    # Compute the mean error per month per site (for boxplots).
    monthly_err_df = y.groupby(['month', 'site']).agg({
        'err_era': 'mean',
        'err_xgb': 'mean'
    }).reset_index()

    # Melt the DataFrame into long format for seaborn.
    monthly_err_long = pd.melt(monthly_err_df, id_vars=['month', 'site'],
                               value_vars=['err_era', 'err_xgb'],
                               var_name='Method', value_name='Error')

    # Replace method names for nicer labels.
    monthly_err_long['Method'] = monthly_err_long['Method'].replace({
        'err_era': 'ERA5',
        'err_xgb': 'XGBoost'
    })

    # Plot monthly error boxplots.
    plt.figure(figsize=(10,6))
    sns.set_style('whitegrid')
    plt.axhline(0, color='gray', lw=1)
    sns.boxplot(x='month', y='Error', hue='Method', data=monthly_err_long, width=0.3)
    plt.title(f"Monthly Error Distribution by Site in {region}")
    plt.xlabel("Month")
    plt.ylabel("Mean Error (°C)")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(rslt_dir + f"monthly_error_boxplot_{region}_{today}.png", dpi=200)
    plt.savefig(rslt_dir + f"monthly_error_boxplot_{region}_{today}.pdf")
    #plt.show()






for region in regions:
    y = Y.loc[Y.region == region]
    
    y['time'] = pd.to_datetime(y['time'])
    y['month'] = y['time'].dt.month
    
    # -------------------------------
    # Compute monthly performance metrics
    # -------------------------------
    # For performance metrics, we aggregate the data by month and site using the mean of T3, ERA5, and XGBoost.
    monthly_df = y.groupby(['month', 'site']).agg({
        'T3': 'mean',
        'E5_skt_degC': 'mean',
        'predicted_T3': 'mean'
    }).reset_index()

    # Loop over months and compute for each method:
    #  - Mean Error = mean(prediction - observation)
    #  - MAE = mean(|prediction - observation|)
    #  - RMSE = sqrt(mean((prediction - observation)^2))
    #  - Correlation = Pearson correlation between observation and prediction
    monthly_metrics = []
    for m in sorted(monthly_df['month'].unique()):
        dfm = monthly_df[monthly_df['month'] == m]
        
        # ERA5 metrics:
        diff_era = dfm['E5_skt_degC'] - dfm['T3']
        mean_error_era = diff_era.mean()
        mae_era = np.abs(diff_era).mean()
        rmse_era = np.sqrt(np.mean(diff_era**2))
        corr_era = pearsonr(dfm['T3'], dfm['E5_skt_degC'])[0]
        
        # XGBoost metrics:
        diff_xgb = dfm['predicted_T3'] - dfm['T3']
        mean_error_xgb = diff_xgb.mean()
        mae_xgb = np.abs(diff_xgb).mean()
        rmse_xgb = np.sqrt(np.mean(diff_xgb**2))
        corr_xgb = pearsonr(dfm['T3'], dfm['predicted_T3'])[0]
        
        monthly_metrics.append({
            'month': m,
            'ERA5_mean_error': mean_error_era,
            'XGB_mean_error': mean_error_xgb,
            'ERA5_MAE': mae_era,
            'XGB_MAE': mae_xgb,
            'ERA5_RMSE': rmse_era,
            'XGB_RMSE': rmse_xgb,
            'ERA5_corr': corr_era,
            'XGB_corr': corr_xgb,
        })

    monthly_metrics_df = pd.DataFrame(monthly_metrics).sort_values('month')

    # -------------------------------
    # Plot metrics in a 2x2 panel figure
    # -------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Mean Error
    axes[0,0].plot(monthly_metrics_df['month'], monthly_metrics_df['ERA5_mean_error'], marker='o', label='ERA5')
    axes[0,0].plot(monthly_metrics_df['month'], monthly_metrics_df['XGB_mean_error'], marker='o', label='XGBoost')
    axes[0,0].set_title('Mean Error')
    axes[0,0].set_xlabel('Month')
    axes[0,0].set_ylabel('Mean Error (°C)')
    axes[0,0].legend()

    # Panel 2: MAE
    axes[0,1].plot(monthly_metrics_df['month'], monthly_metrics_df['ERA5_MAE'], marker='o', label='ERA5')
    axes[0,1].plot(monthly_metrics_df['month'], monthly_metrics_df['XGB_MAE'], marker='o', label='XGBoost')
    axes[0,1].set_title('Mean Absolute Error (MAE)')
    axes[0,1].set_xlabel('Month')
    axes[0,1].set_ylabel('MAE (°C)')
    axes[0,1].legend()

    # Panel 3: RMSE
    axes[1,0].plot(monthly_metrics_df['month'], monthly_metrics_df['ERA5_RMSE'], marker='o', label='ERA5')
    axes[1,0].plot(monthly_metrics_df['month'], monthly_metrics_df['XGB_RMSE'], marker='o', label='XGBoost')
    axes[1,0].set_title('Root Mean Squared Error (RMSE)')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel('RMSE (°C)')
    axes[1,0].legend()

    # Panel 4: Correlation
    axes[1,1].plot(monthly_metrics_df['month'], monthly_metrics_df['ERA5_corr'], marker='o', label='ERA5')
    axes[1,1].plot(monthly_metrics_df['month'], monthly_metrics_df['XGB_corr'], marker='o', label='XGBoost')
    axes[1,1].set_title('Correlation')
    axes[1,1].set_xlabel('Month')
    axes[1,1].set_ylabel('Pearson Correlation')
    axes[1,1].legend()

    plt.suptitle(f"Monthly Error Metrics in {region}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(rslt_dir + f"monthly_error_metrics_{region}_{today}.png", dpi=200)
    plt.savefig(rslt_dir + f"monthly_error_metrics_{region}_{today}.pdf")
    #plt.show()
















"""
# 24.09.2025
RAS XGBo corr: 0.9228157384998518
RAS ERA5 corr: 0.8788566915636243
RAS XGBo rmse: 2.9282697905675303
RAS ERA5 rmse: 5.951307426379148

TII XGBo corr: 0.9746802488699033
TII ERA5 corr: 0.8946062455754656
TII XGBo rmse: 1.7998307998052037
TII ERA5 rmse: 4.465798115119979

KIL XGBo corr: 0.9270701845405752
KIL ERA5 corr: 0.8678895358084169
KIL XGBo rmse: 2.888145059838771
KIL ERA5 rmse: 6.540303441488462

PAL XGBo corr: 0.9669858664206605
PAL ERA5 corr: 0.8674186157841699
PAL XGBo rmse: 2.078088676758043
PAL ERA5 rmse: 6.888848835827947

ULV XGBo corr: 0.9768402181121103
ULV ERA5 corr: 0.8870622369668144
ULV XGBo rmse: 1.8272941688333728
ULV ERA5 rmse: 5.353241049912897

VAR XGBo corr: 0.9624155339900625
VAR ERA5 corr: 0.8714808878090504
VAR XGBo rmse: 2.070434885694126
VAR ERA5 rmse: 5.826121592225976



# 12.09.2025
RAS XGBo corr: 0.9261714547908341
RAS ERA5 corr: 0.8835361857810579
RAS XGBo rmse: 2.845512142125432
RAS ERA5 rmse: 5.875870519228269

TII XGBo corr: 0.9748298242061234
TII ERA5 corr: 0.8948598748107797
TII XGBo rmse: 1.7902518747217158
TII ERA5 rmse: 4.460921480934865

KIL XGBo corr: 0.9238053454011148
KIL ERA5 corr: 0.8711070655280978
KIL XGBo rmse: 3.001232226218933
KIL ERA5 rmse: 6.47520175838607

PAL XGBo corr: 0.96825976097776
PAL ERA5 corr: 0.8704700680174976
PAL XGBo rmse: 2.087435434294223
PAL ERA5 rmse: 6.795326780182577

ULV XGBo corr: 0.9768312176107172
ULV LSSO corr: 0.9075956739946461
ULV XGBo rmse: 1.8205097296573582
ULV ERA5 rmse: 5.296369892458496

VAR XGBo corr: 0.9612155761101826
VAR ERA5 corr: 0.872672243242422
VAR XGBo rmse: 2.112491966567577
VAR ERA5 rmse: 5.809419125195479





# 19.05.2025
RAS XGBo corr: 0.9009318588708546
RAS ERA5 corr: 0.851466619186287
RAS XGBo rmse: 3.1692155394069137
RAS ERA5 rmse: 4.989810729586906 

TII XGBo corr: 0.9463873255265496
TII ERA5 corr: 0.8874811593148749
TII XGBo rmse: 2.669349068481066
TII ERA5 rmse: 4.55954740542191 

KIL XGBo corr: 0.9019531365945943
KIL ERA5 corr: 0.8467827353118039
KIL XGBo rmse: 3.135146403546828
KIL ERA5 rmse: 5.0534761956881304 

PAL XGBo corr: 0.9524198382540555
PAL ERA5 corr: 0.8851053860111224
PAL XGBo rmse: 2.5587098449584254
PAL ERA5 rmse: 4.523686593435701 

ULV XGBo corr: 0.9556693725320904
ULV ERA5 corr: 0.9034216339034015
ULV XGBo rmse: 2.48677890557652
ULV ERA5 rmse: 4.196011015655426 

VAR XGBo corr: 0.9467684641987114
VAR ERA5 corr: 0.876450121521228
VAR XGBo rmse: 2.333610444153559
VAR ERA5 rmse: 4.3500017655061844 



# 09.05.2025
RAS XGBo corr: 0.9113461620798025
RAS ERA5 corr: 0.851466619186287
RAS XGBo rmse: 2.996365907666934
RAS ERA5 rmse: 4.989810729586906 

TII XGBo corr: 0.9502420325121979
TII ERA5 corr: 0.8874811593148749
TII XGBo rmse: 2.5729194067370083
TII ERA5 rmse: 4.55954740542191 

KIL XGBo corr: 0.9089137706761163
KIL ERA5 corr: 0.8529011424974617
KIL XGBo rmse: 3.049279786267227
KIL ERA5 rmse: 4.969318226512857 

PAL XGBo corr: 0.9614812225321027
PAL ERA5 corr: 0.8838091423835482
PAL XGBo rmse: 2.241405599643188
PAL ERA5 rmse: 4.540920007655459 

ULV XGBo corr: 0.9619697920763837
ULV ERA5 corr: 0.9035613240607178
ULV XGBo rmse: 2.279769647781943
ULV ERA5 rmse: 4.175775717652146 

VAR XGBo corr: 0.9489914372724455
VAR ERA5 corr: 0.876450121521228
VAR XGBo rmse: 2.2883063118674754
VAR ERA5 rmse: 4.3500017655061844 



# 16.04.2025
RAS XGBo corr: 0.912342610127058
RAS ERA5 corr: 0.8395206715084514
RAS XGBo rmse: 2.994974401434932
RAS ERA5 rmse: 5.264310352800041 

TII XGBo corr: 0.9489701635930211
TII ERA5 corr: 0.8754414935113296
TII XGBo rmse: 2.5877929971859044
TII ERA5 rmse: 4.767036006716233 

KIL XGBo corr: 0.9079064831805341
KIL ERA5 corr: 0.8415370999473681
KIL XGBo rmse: 3.0545341959282686
KIL ERA5 rmse: 5.252908401237409 

PAL XGBo corr: 0.9615252962024143
PAL ERA5 corr: 0.8741265786515284
PAL XGBo rmse: 2.225686945196311
PAL ERA5 rmse: 4.837047117761403 

ULV XGBo corr: 0.9610565112852085
ULV ERA5 corr: 0.895921424620771
ULV XGBo rmse: 2.3076681779438646
ULV ERA5 rmse: 4.414345507170204 

VAR XGBo corr: 0.9473376505384624
VAR ERA5 corr: 0.8647356880628109
VAR XGBo rmse: 2.327624465562048
VAR ERA5 rmse: 4.635915064734512 


# 11.04.2025
RAS XGBo corr: 0.9140269240093535
RAS ERA5 corr: 0.8395206715084514
RAS XGBo rmse: 2.9154998095076623
RAS ERA5 rmse: 5.264310352800041 

TII XGBo corr: 0.9473024258847191
TII ERA5 corr: 0.8754414935113296
TII XGBo rmse: 2.621772994669837
TII ERA5 rmse: 4.767036006716233 

KIL XGBo corr: 0.9141031648404148
KIL ERA5 corr: 0.8415370999473681
KIL XGBo rmse: 3.0134917480033154
KIL ERA5 rmse: 5.252908401237409 

PAL XGBo corr: 0.9615762762763685
PAL ERA5 corr: 0.8741265786515284
PAL XGBo rmse: 2.2146577582578444
PAL ERA5 rmse: 4.837047117761403 

ULV XGBo corr: 0.9631958707480079
ULV ERA5 corr: 0.895921424620771
ULV XGBo rmse: 2.2603513954855545
ULV ERA5 rmse: 4.414345507170204 

VAR XGBo corr: 0.949205169275256
VAR ERA5 corr: 0.8647356880628109
VAR XGBo rmse: 2.2781565052021215
VAR ERA5 rmse: 4.635915064734512 


# 13.03.2025
RAS XGBo corr: 0.9140741190015246
RAS ERA5 corr: 0.8395206715084514
RAS XGBo rmse: 2.9053157164038503
RAS ERA5 rmse: 5.264310352800041 

TII XGBo corr: 0.9404940497050043
TII ERA5 corr: 0.8754414935113296
TII XGBo rmse: 2.763039369577772
TII ERA5 rmse: 4.767036006716233 

KIL XGBo corr: 0.9139606119770798
KIL ERA5 corr: 0.8415370999473681
KIL XGBo rmse: 2.993874870584996
KIL ERA5 rmse: 5.252908401237409 

PAL XGBo corr: 0.9619470118729248
PAL ERA5 corr: 0.8741265786515284
PAL XGBo rmse: 2.2999582830930048
PAL ERA5 rmse: 4.837047117761403 

ULV XGBo corr: 0.9619506706972031
ULV ERA5 corr: 0.895921424620771
ULV XGBo rmse: 2.2963938855614607
ULV ERA5 rmse: 4.414345507170204 

VAR XGBo corr: 0.9479523588436166
VAR ERA5 corr: 0.8647356880628109
VAR XGBo rmse: 2.3175441933338856
VAR ERA5 rmse: 4.635915064734512 

"""







"""

# Read microclimf simulations for reference
ds_microclimf = fcts.read_microclimf(fs, inpt_dir, vrbs, all_yrs, site_points.values)
fcts.print_ram_state()

for v in vrbs:
    ds_microclimf[v] = ds_microclimf[v]/100.



# Transform microclimf to dataframe
df_microclimf = ds_microclimf.load().to_dataframe().reset_index().sort_values(['gcl','time'])#.astype(float)
for v in vrbs:
    df_microclimf = df_microclimf.rename(columns={v: 'microclimf_'+v})



fcts.print_ram_state()


# Combine microclimf and machine learning results and observations
data = pd.merge(Y, df_microclimf, on=['gcl','time'])



"""







# Calculate daily means
y = Y.copy(deep=True)

# Ensure time column is in datetime format
y['time'] = pd.to_datetime(y['time'])

# Create a date column (without hour/minute)
y['date'] = y['time'].dt.date

# Group by region, site, and date, then calculate daily mean
y = y.groupby(['region', 'site', 'date']).mean(numeric_only=True).reset_index()


# Scatter plots / density histograms
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, SymLogNorm
import seaborn as sns


areas = ['all'] + regions

vrbs = ['T1','T2','T3']


for rgn in areas:
    
    for vrb in vrbs:
        to_plot = {}
        if rgn == 'all':
            to_plot['XGBoost '+vrb+' all'] = y[[vrb,'xgboost_'+vrb]]
            to_plot['Lasso '+vrb+' all'] = y[[vrb,'lasso_'+vrb]]
            
            try: to_plot['Microclimf '+vrb+' all']    = y[[vrb,'microclimf_'+vrb]]
            except: to_plot['Microclimf '+vrb+' all']    = None
            
            to_plot['ERA5 '+vrb+' all']    = y[[vrb,'E5_skt_degC']]
            to_plot['Offset of XGBoost '+vrb+' all'] = y[[vrb+'_offset', 'xgboost_'+vrb+'_offset']]
            to_plot['Offset of Lasso '+vrb+' all'] = y[[vrb+'_offset', 'lasso_'+vrb+'_offset']]
        else:
            to_plot['XGBoost '+vrb+' '+rgn] = y.loc[y['region']==rgn,[vrb,'xgboost_'+vrb]]
            to_plot['Lasso '+vrb+' '+rgn] = y.loc[y['region']==rgn,[vrb,'lasso_'+vrb]]
            
            try: to_plot['Microclimf '+vrb+' '+rgn]    = y.loc[y['region']==rgn,[vrb,'microclimf_'+vrb]]
            except: to_plot['Microclimf '+vrb+' '+rgn] = None
            
            to_plot['ERA5 '+vrb+' '+rgn]    = y.loc[y['region']==rgn,[vrb,'E5_skt_degC']]
            to_plot['Offset of XGBoost '+vrb+' '+rgn] = y.loc[y['region']==rgn,[vrb+'_offset', 'xgboost_'+vrb+'_offset']]
            to_plot['Offset of Lasso '+vrb+' '+rgn] = y.loc[y['region']==rgn,[vrb+'_offset', 'lasso_'+vrb+'_offset']]
        
        sns.set_theme(style="ticks")
        #f, axes = plt.subplots(2,int(len(to_plot)/2), figsize=(6*len(to_plot),6))#, sharex=True, sharey=True)
        #f, axes = plt.subplots(4,2, figsize=(8,16))#, sharex=True, sharey=True)
        f, axes = plt.subplots(3,2, figsize=(8,12))#, sharex=True, sharey=True)
        #if len(to_plot) == 1: 
        #    axes = [axes]
        
        for ax, v in zip(axes.ravel(), to_plot.keys()):
            print(v)
            pl = to_plot[v]
            if pl is None: continue
            
            x_ = pl[pl.columns[0]]
            y_ = pl[pl.columns[1]]
            
            
            corr = fcts.calc_corr(x_, y_).round(4)
            rmse = fcts.calc_rmse(x_, y_).round(4)
            r2ss = fcts.calc_r2ss(x_, y_).round(4)
            
            vmin, vmax = np.nanmin(np.ravel(pl)) - 0.5, np.nanmax(np.ravel(pl)) + 0.5
            bin_edges = np.linspace(vmin, vmax, 50)
            ax.hist2d(x_, y_, bins=bin_edges, cmap='YlGnBu', norm=LogNorm(1))
            
            print(vmin,vmax,corr,rmse,r2ss,)
            print(x_,y_)
            
            # Add 1:1 line and axes
            ax.plot([vmin, vmax], [vmin, vmax], color='b', linestyle="--")
            ax.axhline(0, color='k', linestyle="--")
            ax.axvline(0, color='k', linestyle="--")
            
            # Add red dashed linear regression line
            m, b = np.polyfit(x_, y_, 1)
            ax.plot([vmin, vmax], [m*vmin + b, m*vmax + b], 'r--', label='Linear fit')
            
            ax.set_xlim([vmin, vmax])
            ax.set_ylim([vmin, vmax])
            ax.set_xlabel('Observed data')
            ax.set_ylabel('Modeled data')
            ax.set_title(v)
            
            ax.text(0.42, 0.05, 'R$^2$-score: ' + str(r2ss) +
                    '\nCorrelation: ' + str(corr) +
                    '\nRMSE: ' + str(rmse), transform=ax.transAxes,
                    bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=.5'))
        
        plt.tight_layout(); 
        f.savefig(rslt_dir+f'fig_scatter_DAILYMEAN_TOMS_{vrb}_{rgn}_{today}.pdf')
        f.savefig(rslt_dir+f'fig_scatter_DAILYMEAN_TOMS_{vrb}_{rgn}_{today}.png', dpi=200)
        #plt.show(); 
        plt.clf(); plt.close('all')














# DAILY metrics




metrics    = {  'RMS': fcts.calc_rmse,
                'MAE': fcts.calc_mae,
                'R2S': fcts.calc_r2ss,
                'COR': fcts.calc_corr,
                }

approaches = {  'E5_skt_degC': 'ERA5 raw', 
                'xgboost_T3': 'XGBoost',
                'lasso_T3': 'LASSO',
                'microclimf_T3': 'Microclimf',
                }




sns.set_theme(style="darkgrid")
f, axes = plt.subplots(6,4, figsize=(14,14))#, sharex=True, sharey=True)


#for ax, metric_name in zip(axes.ravel(), metrics.keys()):

regions = ['VAR', 'RAS', 'TII', 'KIL', 'PAL', 'ULV']
k=0
for i,region in enumerate(regions):
    
    y = Y.loc[Y.region==region]
    
    # Ensure time column is in datetime format
    y['time'] = pd.to_datetime(y['time'])

    # Create a date column (without hour/minute)
    y['date'] = y['time'].dt.date

    for j,metric_name in enumerate(metrics.keys()):
        
        ax = axes.ravel()[k]
        print(region, metric_name)
        for approach_name in approaches.keys():
            name = approaches[approach_name]
            metric = metrics[metric_name]
            
            daily_metrics = y.groupby('date').apply(lambda df: metric(df['T3'], df[approach_name]))
            daily_metrics = daily_metrics.where(daily_metrics > -2)
            print(daily_metrics)
            
            sns.kdeplot(data=daily_metrics.astype(float), label=name, ax=ax)
        
        #if metric_name=='RMS': ax.legend()
        if k==0: ax.legend()
        
        ax.set_title(f'{metric_name} {region}')
        
        k+=1


plt.tight_layout(); 
f.savefig(rslt_dir+f'fig_kdeplot_DAILY_TOMS_{today}.pdf')
f.savefig(rslt_dir+f'fig_kdeplot_DAILY_TOMS_{today}.png', dpi=200)
#plt.show(); 
plt.show(); plt.clf(); plt.close('all')




sns.set_theme(style="whitegrid")
f, axes = plt.subplots(1,4, figsize=(14,4))#, sharex=True, sharey=True)

y = Y.copy().reset_index(drop=True)

# Ensure time column is in datetime format
y['time'] = pd.to_datetime(y['time'])

# Create a date column (without hour/minute)
y['date'] = y['time'].dt.date


grouped = y.groupby(['region','date'])
for j,metric_name in enumerate(metrics.keys()):
    
    ax = axes.ravel()[j]
    for approach_name in approaches.keys():
        #print(metric_name, approach_name)
        name = approaches[approach_name]
        metric = metrics[metric_name]
        
        daily_metrics = grouped.apply(lambda df: metric(df['T3'], df[approach_name]))
        daily_metrics = daily_metrics.where((daily_metrics > -2) & (daily_metrics < 15))
        print(metric_name, approach_name, np.min(daily_metrics), np.mean(daily_metrics), np.max(daily_metrics))
        
        sns.kdeplot(data=daily_metrics.astype(float), label=name, cut=0, ax=ax)
    
    ax.legend()
    ax.set_title(f'{metric_name}')
    


plt.tight_layout(); 
f.savefig(rslt_dir+f'fig_kdeplot_DAILY_TOMS_regionmean_{today}.pdf')
f.savefig(rslt_dir+f'fig_kdeplot_DAILY_TOMS_regionmean_{today}.png', dpi=200)
#plt.show(); 
plt.show(); plt.clf(); plt.close('all')









# HOURLY metrics

metrics    = {  'RMS': fcts.calc_rmse,
                #'MAE': fcts.calc_mae,
                #'R2S': fcts.calc_r2ss,
                'Correlation': fcts.calc_corr,
                }

approaches = {  'xgboost_T3': 'XGBoost',
                'lasso_T3': 'LASSO',
                'microclimf_T3': 'Microclimf',
                'E5_skt_degC': 'ERA5 skt', 
                }

colors  =    {  'xgboost_T3': 'tab:orange',
                'lasso_T3': 'tab:green',
                'microclimf_T3': 'tab:cyan',
                'E5_skt_degC': 'tab:red', 
                }








sns.set_theme(style="darkgrid")
f, axes = plt.subplots(6,3, figsize=(14,10))#, sharex=True, sharey=True)


#for ax, metric_name in zip(axes.ravel(), metrics.keys()):


k=0
for i,region in enumerate(regions):
    
    y = Y.loc[Y.region==region]
    
    # Ensure time column is in datetime format
    y['time'] = pd.to_datetime(y['time'])

    for j,metric_name in enumerate(metrics.keys()):
        
        ax = axes.ravel()[k]
        print(region, metric_name)
        for approach_name in approaches.keys():
            name = approaches[approach_name]
            metric = metrics[metric_name]
            
            hourly_metrics = y.groupby('time').apply(lambda df: metric(df['T3'], df[approach_name]))
            hourly_metrics = hourly_metrics.where(hourly_metrics > -2)
            print(region, metric_name, approach_name)
            print(hourly_metrics)
            
            sns.kdeplot(data=hourly_metrics.astype(float), label=name, ax=ax, cut=0, bw_adjust=0.75)
        
        #if metric_name=='RMS': ax.legend()
        if k==0: ax.legend()
        
        ax.set_title(f'{metric_name} {region}')
        
        k+=1


plt.tight_layout(); 
f.savefig(rslt_dir+f'fig_kdeplot_HOURLY_TOMS_{today}.pdf')
f.savefig(rslt_dir+f'fig_kdeplot_HOURLY_TOMS_{today}.png', dpi=200)
#plt.show(); 
plt.show(); plt.clf(); plt.close('all')





# THIS analysis
sns.set_theme(style="whitegrid")
f, axes = plt.subplots(2,2, figsize=(8,8))#, sharex=True, sharey=True)

k = 0
for grp_name, mth_grouping in zip(['Year-round','Summer'], [range(1,13), range(5,10)]):
     
    idx = ((Y.region=='KIL') | (Y.region == 'VAR')) & np.isin(pd.to_datetime(Y.time.values).month, mth_grouping)
    y = Y.loc[idx].dropna()

    # Ensure time column is in datetime format
    y['time'] = pd.to_datetime(y['time'])

    for j,metric_name in enumerate(metrics.keys()):
        
        ax = axes.ravel()[k]
        
        for approach_name in approaches.keys():
            name = approaches[approach_name]
            metric = metrics[metric_name]
            print('\n', grp_name, metric_name, approach_name)
            
            hourly_metrics = y.groupby(['region','time']).apply(lambda df: metric(df['T3'], df[approach_name]))
            hourly_metrics = hourly_metrics.reset_index().groupby('time').mean(numeric_only=True)
            hourly_metrics = hourly_metrics.where((hourly_metrics > -2) & (hourly_metrics < 12))
            
            print(hourly_metrics)
            
            sns.kdeplot(data=hourly_metrics[0].astype(float), color=colors[approach_name], 
                        label=name, ax=ax, cut=0, bw_adjust=0.75)
        
        if k==0: ax.legend()
        
        #ax.set_xlabel(f'{metric_name}')
        ax.set_xlabel('')
        ax.set_title(f'{grp_name} {metric_name}')
        
        k+=1

plt.suptitle('Density of hourly regression metrics, mean over KIL and VAR regions')

plt.tight_layout(); 
f.savefig(rslt_dir+f'fig_kdeplot_HOURLY_KILVAR_TOMS_{today}.pdf')
f.savefig(rslt_dir+f'fig_kdeplot_HOURLY_KILVAR_TOMS_{today}.png', dpi=200)
#plt.show(); 
plt.show(); plt.clf(); plt.close('all')













# REGIONAL and CLIMATOLOGICAL metrics

# THIS analysis
for obs_key in ['T1','T2','T3']:

    temp_level = temp_levels[obs_key]

    val_cols = [obs_key, f"xgboost_{obs_key}", f"lasso_{obs_key}", "microclimf_T3", "E5_skt_degC"]
    regional_temp_range = fcts.regional_daily_temp_range_climatology(Y,
        #value_cols=["T3", "xgboost_T3", "lasso_T3", "microclimf_T3", "E5_skt_degC"],
        value_cols=val_cols,
        time_col="time",drop_feb29=True)


    labels = {}
    for mdl,lbl in zip(val_cols, ['Observed', 'XGBoost', 'Lasso', 'Microclimf', 'ERA5 skt']): labels[mdl] = lbl

    colors = {}
    for mdl,clr in zip(val_cols, ['k', 'tab:orange', 'tab:green', 'tab:cyan', 'tab:red']): colors[mdl] = clr


    f, axes = plt.subplots(2,3, figsize=(10,8), sharex=True, sharey=True)

    for j,(ax,region) in enumerate(zip(axes.ravel(), regions)):
        for i,model in enumerate(val_cols[1:]):
            
            obs_raw = regional_temp_range.loc[region,f'{obs_key}_spatial_range_daily_mean'].copy()
            mod_raw = regional_temp_range.loc[region,f'{model}_spatial_range_daily_mean'].copy()
            
            obs_Gau = fcts.Gauss_filter(obs_raw.copy(), sigma=(15))
            mod_Gau = fcts.Gauss_filter(mod_raw.copy(), sigma=(15))
            
            # Fill_between
            if i==0:
                obs_raw.plot(ax=ax, color='k', lw=0.5, label='')
                obs_Gau.plot(ax=ax, color='k', lw=2, label=labels[obs_key])
            
            mod_raw.plot(ax=ax, color=colors[model], lw=0.5, label='')
            mod_Gau.plot(ax=ax, color=colors[model], lw=2, label=labels[model])#model.replace('_',' '))
            
            if j==0: ax.legend()
        
        ax.set_title(f'{region}')
        ax.set_ylabel('°C')
        ax.set_xlabel('day-of-year')

    plt.suptitle(f'Maximum spatial {temp_level}cm temperature difference across sites, daily mean over years')

    plt.tight_layout(); 
    f.savefig(rslt_dir+f'fig_temprangeclimatology_{obs_key}_TOMS_{today}.pdf')
    f.savefig(rslt_dir+f'fig_temprangeclimatology_{obs_key}_TOMS_{today}.png', dpi=200)
    #plt.show(); 
    plt.show(); plt.clf(); plt.close('all')




regional_meanminmax = fcts.regional_daily_meanminmax_climatology(Y,
    value_cols=["T3", "xgboost_T3", "lasso_T3", "microclimf_T3", "E5_skt_degC"],
    time_col="time",drop_feb29=True)



#f, axes = plt.subplots(2,3, figsize=(10,8))
f, axes = plt.subplots(1,2, figsize=(8,5))
colors = {"xgboost_T3":'tab:orange', "lasso_T3":'tab:green', "microclimf_T3":'tab:cyan', "E5_skt_degC":'tab:red'}
for i,model in enumerate(["xgboost_T3", "lasso_T3", "microclimf_T3",]): # "E5_skt_degC"]):
#for i,model in enumerate(["xgboost_T3", "microclimf_T3"]):
    
    #for ax,region in zip(axes.ravel(), regions):
    for ax,region in zip(axes.ravel(), ['KIL','VAR']):
        
        obs_raw = regional_meanminmax.loc[region,['T3_min','T3_mean','T3_max']]
        mod_raw = regional_meanminmax.loc[region,[f'{model}_min',f'{model}_mean',f'{model}_max']]
        
        obs_Gau = fcts.Gauss_filter(regional_meanminmax.loc[region,['T3_min','T3_mean','T3_max']], sigma=(15,0))
        mod_Gau = fcts.Gauss_filter(regional_meanminmax.loc[region,[f'{model}_min',f'{model}_mean',f'{model}_max']], sigma=(15,0))
        
        # Fill_between
        if i==0:
            ax.fill_between(x=obs_raw.index, y1=obs_raw['T3_min'],y2=obs_raw['T3_max'], color='gray', lw=0., alpha=0.5)
            #obs_Gau['T3_mean'].plot(ax=ax, color='k', lw=3, label='Observed')
        

        for mdl, lw in zip([f'{model}_min',f'{model}_max'], [0.5,0.5]):
            mod_raw[mdl].plot(ax=ax, color=colors[model], lw=lw, label='')
        
        #mod_Gau[f'{model}_mean'].plot(ax=ax, color=colors[model], lw=2, label=model)
        
        ax.set_title(f'{region}')
        ax.legend()


plt.tight_layout(); 
f.savefig(rslt_dir+f'fig_minmaxmeanclimatology_TOMS_{today}.pdf')
f.savefig(rslt_dir+f'fig_minmaxmeanclimatology_TOMS_{today}.png', dpi=200)
#plt.show(); 
plt.show(); plt.clf(); plt.close('all')





