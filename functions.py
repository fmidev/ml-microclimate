#!/usr/bin/env python



import itertools
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb


import io, os, psutil
import s3fs, fsspec










# --- Basic S3 routines ---

def ls_s3(fs, remote_path):
    
    return fs.ls(remote_path)
    

def isfile_s3(fs, remote_path):
    
    return fs.isfile(remote_path)





# --- Reading from S3 ---


def read_mf_s3(fs, remote_paths, **kwargs):
    
    
    # Open a set of remote netCDF files, leave them open
    opened_files = []
    for file_path in remote_paths:
        opened_files.append(fs.open(file_path))
    
    # Read lazily
    ds = xr.open_mfdataset(opened_files, **kwargs)

    return ds



def read_nc_s3(fs, remote_path, **kwargs):

    # Read one file into memory, leave open
    ds = xr.open_dataset(fs.open(remote_path, 'rb') , **kwargs)

    return ds


def read_zr_s3(list_of_remote_zarr_files, **kwargs):
    
    # Might not work
    ds = xr.open_mfdataset(list_of_remote_zarr_files, engine='zarr', 
                backend_kwargs=dict(storage_options={'anon': True}))
    
    return ds



# --- Uploading existing data from disk into S3 ---

def upload_data_s3(fs, remote_path, local_path):
    
    fs.put(local_path, remote_path)
    fs.chmod(remote_path, acl='public-read')
    
    pass





# --- Saving special data directly from memory into S3 --- 


def csv_save_s3(fs, df, remote_path, **kwargs):
    
    with fs.open(remote_path,'w') as f: 
        df.to_csv(f, **kwargs)
    
    fs.chmod(remote_path, acl='public-read')
    
    pass



def fig_save_s3(s3, fig, bucket, remote_path, **kwargs):
    
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png', **kwargs)
    
    s3.Object(bucket,remote_path).put(Body=img_data.getvalue(), ContentType='image/png')
    s3.ObjectAcl(bucket,remote_path).put(ACL='public-read')
    
    pass








# --- Memory usage tracking ---

def print_ram_state(comment=''):
    # https://stackoverflow.com/a/21632554
    
    import os, psutil
    
    process = psutil.Process(os.getpid())
    ram_state = process.memory_info().rss
    
    print(f"RAM: {mb(ram_state) :.2f} MB {comment}", flush=True)

def mb(nbytes):
    return nbytes / (1024 * 1024)









# --- Data analysis and evaluation ---

def find_remove_trends(da):
    
    da = da.copy(deep=True)
    
    p = da.polyfit(dim='time', deg=1, skipna=True)
    fit = xr.polyval(da['time'], p.polyfit_coefficients)
    da -= fit
    
    return da, fit, p




def extract_shap_values(X_test, model, with_xgb=True):
    
    if with_xgb:
        import xgboost as xgb
        
        dmX = xgb.DMatrix(X_test, label=None, nthread=-1)
        booster = model.get_booster()
        
        shap_values = booster.predict(dmX, pred_contribs=True)
        explainer = np.nan
        
    else:
        import shap
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test, approximate=True, check_additivity=False)
        shap_values = shap_values
    
    
    if len(shap_values.shape) == 3:
        shap_values = np.mean(shap_values, axis=2)
    
    
    print(shap_values.shape)
    
    if type(shap_values)==np.ndarray:
        data = pd.DataFrame(np.abs(shap_values).mean(axis=0), index=list(X_test.columns))
        data = data.sort_values(by=0,ascending=False)
        
        sorted_names, sorted_importance = list(data.index), data.values.squeeze()
        
        shap_values = pd.DataFrame(columns=X_test.columns, index=X_test.index, data=shap_values)
        
        #return [sorted_names], [sorted_importance], [shap_values], explainer
    
    if type(shap_values)==list:
        sorted_names = []; sorted_importance = []
        for i, item in enumerate(shap_values):
            data = pd.DataFrame(np.abs(item).mean(axis=0), index=list(X_test.columns))
            data = data.sort_values(by=0,ascending=False)
            
            std_names, std_importance = list(data.index), data.values.squeeze()
            sorted_names.append(std_names); sorted_importance.append(std_importance)
            
            shap_values[i] = pd.DataFrame(columns=X_test.columns, index=X_test.index, data=item)
             
    
    return sorted_names, sorted_importance, shap_values, explainer





def add_day_of_year(df, time_col="time", tz="UTC", drop_feb29=True):
    out = df.copy()
    dt = pd.to_datetime(out[time_col], errors="coerce")
    if tz is not None and (dt.dt.tz is None):
        dt = dt.dt.tz_localize(tz)

    if drop_feb29:
        mask_feb29 = (dt.dt.month == 2) & (dt.dt.day == 29)
        out = out.loc[~mask_feb29].copy()
        dt = pd.to_datetime(out[time_col], errors="coerce")
        if tz is not None and (dt.dt.tz is None):
            dt = dt.dt.tz_localize(tz)
        # Shift DOY after Feb 28 in leap years so DOY ∈ 1..365
        doy = dt.dt.dayofyear - ((dt.dt.is_leap_year) & (dt.dt.dayofyear > 59)).astype(int)
    else:
        doy = dt.dt.dayofyear  # 1..366

    out["doy"] = doy.astype("int16")
    out["date"] = dt.dt.normalize().dt.date
    return out


def regional_daily_meanminmax_climatology(df, value_cols, time_col="time", drop_feb29=True):
    """
    Regional daily climatology (min/mean/max) for each day-of-year (DOY).
    First aggregates to daily by date, then averages by DOY across years.
    This avoids bias from days with more hourly records.
    """
    use = [c for c in value_cols if c in df.columns]
    if not use:
        raise ValueError("No value_cols found.")

    tmp = add_day_of_year(df, time_col=time_col, drop_feb29=drop_feb29)

    # Step 1: daily per region (calendar day)
    daily = tmp.groupby(["region", "date"])[use].agg(["min", "mean", "max"])
    daily.columns = [f"{v}_{stat}" for (v, stat) in daily.columns]
    # Reattach DOY (unique per date)
    daily = daily.reset_index()
    daily["doy"] = pd.to_datetime(daily["date"]).dt.dayofyear
    if drop_feb29:
        # compress DOY to 1..365 (dates already have no Feb 29, but in leap years doys > 59 need shift)
        y = pd.to_datetime(daily["date"])
        daily["doy"] = daily["doy"] - ((y.dt.is_leap_year) & (daily["doy"] > 59)).astype(int)

    # Step 2: climatology across years → mean over all years for each DOY
    clim = daily.groupby(["region", "doy"]).mean(numeric_only=True).sort_index()
    return clim




def regional_daily_temp_range_climatology(
    df,
    value_cols,
    time_col="time",
    group_col="region",
    drop_feb29=True,
    within_day_agg="mean",   # "mean" or "max": combine 24 hourly ranges into a daily value
):
    """
    Daily climatology of within-region spatial temperature range.

    Steps:
      (1) For each (region, hour): spatial range = max(site) − min(site),
          after dropping rows with NaN in the value column.
      (2) For each (region, date): aggregate hourly ranges over the day (mean or max).
      (3) For each (region, DOY): average daily values across years (climatology).
          Optionally drop Feb 29 and compress to 365 days.

    Returns
    -------
    DataFrame with MultiIndex (region, doy) and columns like:
      '<var>_spatial_range_daily_mean' or '<var>_spatial_range_daily_max'
    """
    use = [c for c in value_cols if c in df.columns]
    if not use:
        raise ValueError("No value_cols found in df.")

    out = df.copy()

    # Parse time as tz-aware UTC
    dt = pd.to_datetime(out[time_col], errors="coerce", utc=True)

    # Optionally drop Feb 29
    if drop_feb29:
        mask_feb29 = (dt.dt.month == 2) & (dt.dt.day == 29)
        out = out.loc[~mask_feb29].copy()
        dt = pd.to_datetime(out[time_col], errors="coerce", utc=True)

    # Hour and date keys
    out["_hour"] = dt.dt.floor("H")
    out["_date"] = dt.dt.normalize().dt.date

    # -------- (1) Hourly spatial range per region --------
    # Drop NaNs before grouping (per value column separately)
    hourly_range_list = []
    for var in use:
        tmp = out[[group_col, "_hour", var]].dropna()
        gb = tmp.groupby([group_col, "_hour"], sort=False)[var]
        rng = gb.max() - gb.min()
        rng.name = f"{var}_spatial_range"
        hourly_range_list.append(rng)

    hourly_range = pd.concat(hourly_range_list, axis=1)#.dropna()

    # -------- (2) Aggregate hourly ranges to daily per region --------
    daily = hourly_range.reset_index()
    daily["_date"] = pd.to_datetime(daily["_hour"]).dt.normalize().dt.date

    if within_day_agg not in {"mean", "max"}:
        raise ValueError("within_day_agg must be 'mean' or 'max'")

    daily = daily.groupby([group_col, "_date"]).agg(within_day_agg)
    daily.columns = [f"{c}_daily_{within_day_agg}" for c in daily.columns]
    
    #daily = daily.dropna()
    # -------- (3) DOY climatology across years --------
    dts = pd.to_datetime(daily.reset_index()["_date"])
    doy = dts.dt.dayofyear
    if drop_feb29:
        # compress to 1..365 (already removed Feb 29; fix DOY numbering in leap years)
        doy = doy - ((dts.dt.is_leap_year) & (doy > 59)).astype(int)

    daily_reset = daily.reset_index()
    daily_reset["doy"] = doy.astype("int16")

    clim = (
        daily_reset.groupby([group_col, "doy"])
                   .mean(numeric_only=True)
                   .sort_index()
    )
    return clim


'''
def add_hour_of_year(df, time_col="time", tz="UTC", drop_feb29=False):
    """
    Adds a 'hoy' column (0-based hour-of-year).
    If drop_feb29=True, removes Feb 29 hours and compresses HOY to 0..8759.
    """
    out = df.copy()

    # Ensure tz-aware datetimes
    dt = pd.to_datetime(out[time_col], errors="coerce")
    if tz is not None and (dt.dt.tz is None):
        dt = dt.dt.tz_localize(tz)

    # Components
    doy  = dt.dt.dayofyear
    hour = dt.dt.hour

    if not drop_feb29:
        # Raw HOY (leap years will have 8784 hours)
        out["hoy"] = (doy - 1) * 24 + hour
        return out

    # Climatological HOY: drop Feb 29 hours and compress
    is_leap = dt.dt.is_leap_year
    is_feb29 = (dt.dt.month == 2) & (dt.dt.day == 29)
    # Remove Feb 29 rows
    out = out.loc[~is_feb29].copy()

    # Recompute dt/doy/hour for the filtered rows
    dt2   = pd.to_datetime(out[time_col], errors="coerce")
    if tz is not None and (dt2.dt.tz is None):
        dt2 = dt2.dt.tz_localize(tz)

    doy2  = dt2.dt.dayofyear
    hour2 = dt2.dt.hour
    # Subtract 1 day for all days after Feb 28 in leap years to compress to 365-day HOY
    shift = ((dt2.dt.is_leap_year) & (doy2 > 59)).astype(int)
    doy2c = doy2 - shift
    out["hoy"] = (doy2c - 1) * 24 + hour2  # 0..8759
    return out


def regional_hourly_summary(df, value_cols, time_col="time", drop_feb29=True):
    """
    Returns regional minima, mean, and maxima for each hour_of_year (HOY).
    value_cols: list of columns to aggregate (e.g., ["T3","xgboost_T3"])
    """
    # Make sure columns exist
    value_cols = [c for c in value_cols if c in df.columns]
    if not value_cols:
        raise ValueError("No value columns found in df.")

    tmp = add_hour_of_year(df, time_col=time_col, drop_feb29=drop_feb29)

    # Group by region and HOY; get min/mean/max (skipna=True by default)
    agg = tmp.groupby(["region", "hoy"])[value_cols].agg(["min", "mean", "max"])

    # Optional: make columns a single level like 'T3_mean'
    agg.columns = [f"{v}_{stat}" for (v, stat) in agg.columns]
    return agg.sort_index()
'''




def calc_r2ss(obs,mod,multioutput='raw_values'): 
    from sklearn.metrics import r2_score
    
    mask_o = np.isnan(obs) | np.isinf(obs)
    mask_m = np.isnan(mod) | np.isinf(mod)
    mask = mask_o + mask_m
    
    if len(obs[~mask]) == 0 or len(mod[~mask]) == 0: 
        result = np.nan
    else:
        result = r2_score(obs[~mask], mod[~mask], multioutput=multioutput)
    
    return np.squeeze(result).astype(float)


def calc_rmse(a,b,axis=0): 
    return np.sqrt(np.nanmean((a-b)**2, axis=axis))

def calc_mse(a,b,axis=0):  
    return np.nanmean((a-b)**2, axis=axis)

def calc_mae(a,b,axis=0):  
    return np.nanmean(np.abs(a-b), axis=axis)

def calc_sprc(a, b, return_pvalue=False):
    from scipy import stats
    if return_pvalue:     return stats.spearmanr(a, b, nan_policy='omit')
    if not return_pvalue: return stats.spearmanr(a, b, nan_policy='omit')[0]

def calc_pseudo_huber(a, b, delta=1.0, axis=0):
    """
    Compute mean pseudo-Huber error between arrays a and b.

    Parameters
    ----------
    a, b : array_like
        Input arrays.
    delta : float, optional (default=1.0)
        Scale parameter controlling the quadratic-to-linear transition.
    axis : int, optional (default=0)
        Axis along which to compute the mean.

    Returns
    -------
    float or ndarray
        Mean pseudo-Huber error.
    """
    diff = a - b
    return np.nanmean(delta**2 * (np.sqrt(1 + (diff/delta)**2) - 1), axis=axis)


def calc_nse(obs,mod):
    return 1-(np.nansum((obs-mod)**2)/np.nansum((obs-np.nanmean(mod))**2))


def calc_corr(a, b, axis=0):
    mask_a = np.isnan(a) | np.isinf(a)
    mask_b = np.isnan(b) | np.isinf(b)
    mask = mask_a + mask_b
    _a = a.copy()
    _b = b.copy()
    try:
        _a[mask] = np.nan
        _b[mask] = np.nan
    except:
        pass
    _a = _a - np.nanmean(_a, axis=axis, keepdims=True)
    _b = _b - np.nanmean(_b, axis=axis, keepdims=True)
    std_a = np.sqrt(np.nanmean(_a**2, axis=axis)) 
    std_b = np.sqrt(np.nanmean(_b**2, axis=axis)) 
    return np.nanmean(_a * _b, axis=axis)/(std_a*std_b)


def calc_corr_xr(a, b, axis=0):
    _a = a - np.mean(a, axis=axis)
    _b = b - np.mean(b, axis=axis)
    std_a = np.sqrt(np.mean(np.power(_a,2), axis=axis)) 
    std_b = np.sqrt(np.mean(np.power(_b,2), axis=axis)) 
    return np.mean(np.multiply(_a, _b), axis=axis)/np.multiply(std_a, std_b)

def vcorrcoef(X,y):
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    r_num = np.sum((X-Xm)*(y-ym),axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
    r = r_num/r_den
    return r



# --- Manipulators ---

def regrid_dataset(ds, lat_new, lon_new, method='nearest'):
    # Interpolate using xarray's interp method
    return ds.interp(lat=lat_new, lon=lon_new, method=method)


def Gauss_filter(data, sigma=(0,1,1), mode='wrap'):
    """ Smooth data (spatially in 3D as default) using Gaussian filter """   
    import scipy.ndimage.filters as flt
    
    try: data_vals=data.values
    except: data_vals=data
    
    U=data_vals.copy()
    V=data_vals.copy()
    V[np.isnan(U)]=0
    VV=flt.gaussian_filter(V,sigma=sigma, mode=mode)

    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=flt.gaussian_filter(W,sigma=sigma, mode=mode)

    Z=VV/WW
    Z[np.isnan(U)] = np.nan
    
    data[:] = Z
    
    return data #Z



def apply_PCA(data, ncomp, pca_model_in=False, fillna=True, svd_solver='randomized'): 
    """ Decomposition of data with principal component analysis. 
        Assumes data to be of shape (time, gridcell). """
    
    from sklearn.impute import SimpleImputer
    import sklearn.decomposition as dc
    
    svl = np.nan
    
    if data.shape[1] < ncomp:
        ncomp = data.shape[1]
        print('Reducing number of components to',ncomp)
    
    if fillna: 
        data = SimpleImputer().fit_transform(data)
    
    if pca_model_in:
        pca = pca_model_in
        cps = pca.transform(data)
    
    
    if not pca_model_in:
        # Perform the PCA
        pca = dc.PCA(n_components=ncomp, whiten=False, random_state=99, svd_solver=svd_solver)# svd_solver='full')
        cps = pca.fit_transform(data)
        svl = pca.singular_values_ 
    
    return cps,pca,svl




# --- Physical formulas ---

def calc_rh1(T, Td):
    '''
    https://archive.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html
    
           T = temperature in deg C;
           Td = dew point in deg C;
            
           es = saturation vapor pressure in mb;
           e = vapor pressure in mb;
           RH = Relative Humidity in percent 
    '''
    es = 6.112*np.exp((17.67*T)/(T + 243.5))
    e = 6.112*np.exp((17.67*Td)/(Td + 243.5))
    RH = 100.0 * (e/es)
    
    return RH


def calc_rh2(T, Td):
    '''
    https://en.wikipedia.org/wiki/Dew_point
    '''
    RH = 100.0 - 5*(T - Td)
    
    return RH



def calc_q1(p, Td):
    '''
    https://archive.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html
    
           Td = dew point in deg C;
           p = surface pressure in mb;
           
           e = vapor pressure in mb;
           q = specific humidity in kg/kg. 
    '''
    e = 6.112*np.exp((17.67*Td)/(Td + 243.5))
    q = (0.622 * e)/(p - (0.378 * e))
    
    return q



def calc_q2(p, Td):
    '''
    https://anonchatgpt.com/
    '''
    e = 6.112*np.exp((17.67*Td)/(Td + 243.5))
    w = (622*e)/(p-e)
    q = 0.622*(e/(p-e))*(w/100)
    
    return q


def solar_zenith_angle(latitude, longitude, date, hour):
    '''
    https://anonchatgpt.com/
    '''
    import math, datetime
    from pandas import to_datetime
    
    # Convert the date string to pandas datetime object
    date = to_datetime(date)
    
    # Convert latitude and longitude to radians
    lat_rad = math.radians(latitude)
    long_rad = math.radians(longitude)
    
    # Calculate day of the year (1-365)
    d = (date - datetime.datetime(date.year, 1, 1)).days + 1
    
    # Calculate solar declination angle
    delta = -23.45 * math.cos((2 * math.pi / 365) * (d + 10))
    
    # Calculate hour angle in radians
    h = (15 * (hour - 12)) * math.pi / 180
    
    # Calculate solar zenith angle
    sin_term = math.sin(lat_rad) * math.sin(math.radians(delta))
    cos_term = math.cos(lat_rad) * math.cos(math.radians(delta)) * math.cos(h)
    zenith_angle = math.degrees(math.acos(sin_term + cos_term))
    
    return zenith_angle




def calculate_solar_zenith_angle(latitude_list, longitude_list, date_list):
    """
    Calculates the solar zenith angle for a given latitude, longitude, and date using the pysolar library.
    
    Parameters:
        latitude (float): Latitude in decimal degrees.
        longitude (float): Longitude in decimal degrees.
        date (datetime.date): Date for which to calculate the solar zenith angle.
    
    Returns:
        float: Solar zenith angle in degrees.
    """
    from pysolar import solar
    
    altitude = solar.get_altitude_fast(latitude_list, longitude_list, date_list)
    zenith_angle = 90.0 - altitude
    return zenith_angle





# --- DEM analysis ---


from pyproj import Transformer

# Define transformer from ETRS-TM35FIN to WGS84, initialize transformer once
_transformer_etrs_to_wgs84 = Transformer.from_crs("EPSG:3067", "EPSG:4326", always_xy=True)

def etrs_tm35fin_to_wgs84(etrs_x, etrs_y):
    # Convert inputs to NumPy arrays if they aren't already
    etrs_x = np.asarray(etrs_x)
    etrs_y = np.asarray(etrs_y)

    # Perform vectorized transformation
    lon, lat = _transformer_etrs_to_wgs84.transform(etrs_x, etrs_y)

    return lon, lat



from scipy import signal
from scipy.ndimage import filters


def haversine(coords1, coords2):
    import numpy as np
    
    # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
    lon1, lat1 = coords1
    lon2, lat2 = coords2
    
    R = 6371000  # radius of Earth in meters
    phi_1 = np.radians(lat1)
    phi_2 = np.radians(lat2)

    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2.0) ** 2
    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    meters = R * c  # output distance in meters
    km = meters / 1000.0  # output distance in kilometers

    meters = round(meters, 3)
    km = round(km, 3)


    print("Distance: "+str(meters)+" m")
    return meters


def grid_spacing(lons_1d, lats_1d):
    
    # Define grid spacing in degrees
    grid_spacing_x = np.mean(np.diff(lons_1d))
    grid_spacing_y = np.mean(np.diff(lats_1d))
    
    # Earth's radius in kilometers
    R = 6371
    
    # Distance between neighboring grid points in radians and in a 2-dimensional grid
    dlon = np.zeros(lons_1d.shape); dlon[:] = np.radians(grid_spacing_x)
    dlat = np.zeros(lats_1d.shape); dlat[:] = np.radians(grid_spacing_y)
    dlon, dlat = np.meshgrid(dlon, dlat)
    
    # Grid point values in radians and in a 2-dimensional grid
    lons, lats = np.meshgrid(lons_1d, lats_1d)
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    
    # Distances in x and y direction in meters
    dx = R * np.cos(lats_rad) * dlon * 1000
    dy = R * dlat * 1000
    
    return dx, dy



def dir_as_sincos(theta):
    dsin = np.sin(theta*(np.pi/180.))
    dcos = np.cos(theta*(np.pi/180.))
    return dsin, dcos


def calc_dem_features(dem_data, x, y, window_size):
    
    window_size_y = np.array(window_size[0])
    window_size_x = np.array(window_size[1])
    
    # Create NaN mask, fill nans with zeros
    nans = np.full(dem_data.shape, np.nan); nans[~np.isnan(dem_data)] = 1
    #dem_data_zfill = np.nan_to_num(dem_data)
    
    # Calculate distances between grid cells in meters
    dx, dy = np.diff(x).mean(), np.diff(y).mean()
    
    # Calculate slopes in radians in x and y direction
    slope_x = np.arctan(np.gradient(dem_data, axis=1) / dx) 
    slope_y = np.arctan(np.gradient(dem_data, axis=0) / dy) 
    
    # Calculate slopes in radians at different spatial scales: scale defined by the window parameter
    #kernel = np.ones((window_size_x, window_size_y)) * np.blackman(window_size_y) * np.blackman(window_size_x)[:,np.newaxis]
    kernel = np.outer(np.blackman(window_size_y), np.blackman(window_size_x))
    kernel /= kernel.sum()  # Normalize kernel
    slope_x = signal.convolve(slope_x, kernel, mode='same',method='direct')/np.sum(kernel) 
    slope_y = signal.convolve(slope_y, kernel, mode='same',method='direct')/np.sum(kernel) 
    
    # Calculate the non-directional slope in degrees
    slope = np.rad2deg(np.arctan(np.sqrt(slope_x**2 + slope_y**2)))
    
    # Calculate aspect in degrees
    aspect = np.rad2deg(np.arctan2(slope_x, slope_y))
    aspect = (aspect + 360) % 360
    
    # Calculate sine and cosine of the aspect
    aspect_sin = np.sin(np.deg2rad(aspect))
    aspect_cos = np.cos(np.deg2rad(aspect))
    
    # Calculate slopes in degrees
    slope_x = np.rad2deg(slope_x) 
    slope_y = np.rad2deg(slope_y)
    
    
    # Calculate Topographic Position Index
    avg_elevation = Gauss_filter(np.copy(dem_data), sigma=(window_size_x,window_size_y))
    tpi = dem_data - avg_elevation
    
    
    ws = str(int(np.mean(window_size)))
    return {
        f'slope_degrees_w{ws}': slope * nans,
        f'slope_x_degrees_w{ws}': np.rad2deg(slope_x) * nans,
        f'slope_y_degrees_w{ws}': np.rad2deg(slope_y) * nans,
        f'aspect_degrees_w{ws}': aspect * nans,
        f'aspect_cos_w{ws}': aspect_cos * nans,
        f'aspect_sin_w{ws}': aspect_sin * nans,
        f'top_posn_idx_w{ws}': tpi * nans,}



"""

def calc_dem_features(dem_data, x, y, window_size):
    window_size_y, window_size_x = window_size
    
    # Create NaN mask
    nans = np.full(dem_data.shape, np.nan)
    nans[~np.isnan(dem_data)] = 1
    
    # Calculate cell size in meters
    dx, dy = np.diff(x).mean(), np.diff(y).mean()
    
    # Calculate slopes in x and y directions
    slope_x = np.gradient(dem_data, axis=1) / dx
    slope_y = np.gradient(dem_data, axis=0) / dy
    
    # Create a Blackman kernel for smoothing
    kernel = np.outer(np.blackman(window_size_y), np.blackman(window_size_x))
    kernel /= kernel.sum()
    
    # Smooth slopes using convolution
    slope_x = signal.convolve(slope_x, kernel, mode='same', method='direct')
    slope_y = signal.convolve(slope_y, kernel, mode='same', method='direct')
    
    # Calculate slope magnitude and aspect
    slope = np.rad2deg(np.arctan(np.sqrt(slope_x**2 + slope_y**2)))
    aspect = (np.rad2deg(np.arctan2(slope_y, -slope_x)) + 360) % 360
    aspect_sin = np.sin(np.deg2rad(aspect))
    aspect_cos = np.cos(np.deg2rad(aspect))
    
    # Calculate TPI
    avg_elevation = Gauss_filter(np.copy(dem_data), sigma=(window_size_x, window_size_y))
    tpi = dem_data - avg_elevation
    
    ws = str(int(np.mean(window_size)))
    return {
        f'slope_degrees_w{ws}': slope * nans,
        f'slope_x_degrees_w{ws}': np.rad2deg(slope_x) * nans,
        f'slope_y_degrees_w{ws}': np.rad2deg(slope_y) * nans,
        f'aspect_degrees_w{ws}': aspect * nans,
        f'aspect_cos_w{ws}': aspect_cos * nans,
        f'aspect_sin_w{ws}': aspect_sin * nans,
        f'top_posn_idx_w{ws}': tpi * nans,
    }
"""










# --- Reading and data extraction ---


from datetime import timedelta
import calendar
def generate_hourly_timerange(year, month):
    """
    Generate an hourly time range for a given year and month, extending the range
    by 4 hours before the start and after the end of the month.

    Parameters:
        year (int): The year (e.g., 2025)
        month (int): The month (e.g., 4 for April)
    
    Returns:
        pandas.DatetimeIndex: Hourly time range with extra hours on both sides.
    """
    # Determine the correct number of days in the specified month
    last_day = calendar.monthrange(year, month)[1]
    
    # Construct the beginning datetime (first day at midnight) and subtract 4 hours
    bgn = pd.to_datetime(f'{year}-{month:02d}-01T00:00') - timedelta(hours=4)
    
    # Construct the ending datetime (last day at 23:00) and add 4 hours
    end = pd.to_datetime(f'{year}-{month:02d}-{last_day}T23:00') + timedelta(hours=4)
    
    # Create an hourly time range between the two datetimes
    time_range = pd.date_range(bgn, end, freq='1h')
    return time_range



def get_metadata():

    # Predictor variables
    era5_vars = [   '2m_temperature',
                    '2m_dewpoint_temperature',
                    'mean_surface_downward_long_wave_radiation_flux',
                    'mean_surface_net_long_wave_radiation_flux',
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    'total_sky_direct_solar_radiation_at_surface',
                    'surface_solar_radiation_downwards',
                    'skin_temperature',
                    'snow_depth',
                    'mean_evaporation_rate',] 

    help_vars = [   'd2m',
                    'ssrd',
                    'fdir']


    # Lags (time steps) to be used for lagging the predictor data
    lags = [-3,0,3,12,24,48]
    lags = [-3,0,3,6]


    regions = ['RAS', 'TII', 'KIL', 'PAL', 'ULV', 'VAR']

    return era5_vars, help_vars, lags, regions


def read_all_dem_data(fs, regions, plot_examples=False, site_points=[], rslt_dir=''):
    # Read preprocessed DEM data from netcdf files
    dem_data = []
    
    for i,region in enumerate(regions):
        print('\nDEM for',region)
        dem = read_dem(fs, region)
        dem_data.append(dem)
        print_ram_state(region)
        
        
        if plot_examples:
            import matplotlib.pyplot as plt
            
            f, axes = plt.subplots(1,1, figsize=(9,7))
            
            #dem['top_posn_idx_w3'].plot(cmap='RdGy',robust=True, alpha=0.8)
            dem_data[i]['top_posn_idx_w3'].plot(cmap='RdGy',robust=True, alpha=0.8)
            
            #ds['150cm_temp'].mean(['time']).plot(cmap='nipy_spectral',robust=True, alpha=0.8)
            #mask_ds.plot(alpha=0.5, add_colorbar=False)
            
            plt.scatter(x=site_points['x'].values, y=site_points['y'].values, 
                        c='red',label='Logger locations',edgecolors='k',alpha=1)
            
            plt.legend(loc='upper right')
            
            plt.tight_layout(); 
            f.savefig(rslt_dir+f'fig_site_points_loggers_selected_{region}.pdf')
            f.savefig(rslt_dir+f'fig_site_points_loggers_selected_{region}.png', dpi=200)
            #plt.show(); 
            plt.clf(); plt.close('all')
        

    dem_ds = xr.merge(dem_data)
    return dem_ds



def read_era5(era5_dir, era5_vars, help_vars, lats, lons, lags, other_time):
    
    
    era5_data = []
    for v in era5_vars:
        
        try:
            era5_ds = adjust_lats_lons(xr.open_mfdataset(era5_dir+v+'*.nc')).sel(
                            lat=slice(lats[0], lats[1]),
                            lon=slice(lons[0], lons[1]),).copy(deep=True)
            
            era5_ds = era5_ds.rename({'valid_time':'time'})
            
            timesteps = np.intersect1d(other_time, era5_ds.time)
            era5_ds = era5_ds.sel(time=timesteps)
            
            data_var = list(era5_ds.data_vars)[0]
            if 'expver' in list(era5_ds.coords):
                #era5_ds = era5_ds.sel(expver=1)
                era5_ds = era5_ds.drop('expver')
            
            #spatial_mean = era5_ds[data_var].mean(['lat','lon'])
            #era5_ds[data_var][:] = np.nan 
            #era5_ds[data_var][:] = era5_ds[data_var].fillna(spatial_mean)
            
            era5_data.append(era5_ds)
            print('ERA5',v,data_var,'was read', flush=True)
        except:
            print('ERA5',v,'FAILED', flush=True)
    
    print_ram_state()
    
    grid_data = xr.merge(era5_data)
    
    
    if 'expver' in grid_data.coords:
        grid_data = grid_data.drop('expver')
    
    
    # Derive additional predictors based on
    # https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13877
    
    # Spesific humidity
    #grid_data['q'] = xr.full_like(grid_data['t2m'], np.nan)
    #grid_data['q'][:] = calc_q1(grid_data['sp']/100., grid_data['d2m']-273.15)
    
    
    # Relative humidity
    grid_data['rh'] = xr.full_like(grid_data['t2m'], np.nan)
    grid_data['rh'][:] = calc_rh1(grid_data['t2m']-273.15, grid_data['d2m']-273.15)
    
    
    # Emissivity
    #grid_data['emis'] = xr.full_like(grid_data['t2m'], np.nan)
    #grid_data['emis'][:] = grid_data['msdwlwrf'] / (grid_data['msnlwrf'] + grid_data['msdwlwrf'])
    
    # Upward longwave radiation
    #grid_data['uwlr'] = xr.full_like(grid_data['t2m'], np.nan)
    #grid_data['uwlr'][:] = grid_data['msnlwrf'] - grid_data['msdwlwrf']
    
    # Diffuse radiation
    grid_data['difr'] = xr.full_like(grid_data['t2m'], np.nan)
    grid_data['difr'][:] = grid_data['ssrd'] - grid_data['fdir'] 
    
    # T-Td difference
    #grid_data['dt2m'] = xr.full_like(grid_data['t2m'], np.nan)
    #grid_data['dt2m'] = grid_data['t2m']-grid_data['d2m']
    
    #for v in help_vars + ['d2m', 'msdwlwrf', 'msnlwrf', 'uwlr', 'ssrd', 'fdir']:
    #for v in help_vars + ['msdwlwrf', 'msnlwrf', 'uwlr', 'ssrd', 'fdir', 'tcc', 'sp']:
    for v in help_vars + ['avg_sdlwrf', 'avg_snlwrf', 'ssrd', 'fdir', 'tcc', 'sp']:
        if v in grid_data.data_vars:
            grid_data = grid_data.drop(v)
    
    
    # Lag data 
    drop_vrbs = list(grid_data.data_vars)
    for v in drop_vrbs:
        for lag in lags:
            sign = '+'
            if np.sign(lag)==1: sign = '-'
            v_lag = f'E5_{v}_{sign}{str(np.abs(lag)).zfill(3)}'
            
            grid_data[v_lag] = grid_data[v].shift(time=lag)

    
    grid_data = grid_data.drop(drop_vrbs)
    
    
    return grid_data



#def read_microclimf(fs, dirs, vrbs, years, interp_object=[]):#points=[]):
def read_microclimf(fs, dirs, interp_object=[]):#points=[]):
    
    # List and read subset of simulation files
    list_datasets = []
    
    for inpt_dir in dirs:
        #for year in years:
        #for vrb in vrbs:
        #files = fs.glob(inpt_dir+'20*/'+vrb+'_*_20*.nc') 
        #files = fs.glob(f'{inpt_dir}{year}/{vrb}_*_{year}*.nc')
        files = fs.glob(f'{inpt_dir}/*.nc')
        print('Reading',files)
        if len(files) > 0:
            for file in files[0:3]:
                print('File:',file)
                #ds = read_mf_s3(fs, files, parallel=True).transpose('time','lat','lon')
                ds = read_nc_s3(fs, file).transpose('time','y','x')
                #if len(points)==0:
                #    #ds = read_nc_s3(fs, file).transpose('time','lat','lon')
                #    ds = read_nc_s3(fs, file).transpose('time','y','x')
                if len(interp_object)>0:
                    """
                    ds = read_nc_s3(fs, file).transpose('time','lat','lon').sel(
                        lat=xr.DataArray(np.array(points)[:,0], dims='gcl'), \
                        lon=xr.DataArray(np.array(points)[:,1], dims='gcl'), \
                        method='nearest').load()
                    """
                    ds = ds.interp(x=interp_object['x'], y=interp_object['y'], method='linear')
                    
                list_datasets.append(ds)
                print_ram_state()
                
    #t150_files = fs.glob(inpt_dir+'20*/150cm_temp_*_20*.nc') 
    #r150_files = fs.glob(inpt_dir+'20*/150cm_relhum_*_20*.nc') 
    #t15__files = fs.glob(inpt_dir+'20*/15cm_temp_*_20*.nc') 
    #r15__files = fs.glob(inpt_dir+'20*/15cm_relhum_*_20*.nc') 
    
    # Read data with chunks
    #dict_datasets = {}
    #for vrb_list in dict_vrbs.keys():
    #    
    #    ds = read_mf_s3(fs, t150_files).transpose('time','lat','lon')
    
    #ds_r150 = read_mf_s3(fs, r150_files).transpose('time','lat','lon')
    #ds__t15 = read_mf_s3(fs, t15__files).transpose('time','lat','lon')
    #ds__r15 = read_mf_s3(fs, r15__files).transpose('time','lat','lon')
    
    
    #ds = xr.merge([ds_t150, ds_r150, ds__t15, ds__r15])
    ds_all_vrbs = xr.concat(list_datasets, dim='time')/100.
    
    return ds_all_vrbs



import terrain_feature_engineering as tfe
def read_dem(file_path, region):
    
    #dem = read_mf_s3(fs, fs.glob(f'resiclim-microclimate/dem_features_{region}.nc') ).load() 
    dem = xr.open_dataset(f'{file_path}/dem_features_{region}.nc').load() 
    
    vrbs = list(dem.data_vars)
    vrb0 = 'dem10m'
    
    
    include_list = ['pisr_1','pisr_2','pisr_3','pisr_4','pisr_5','pisr_6','pisr_7','pisr_8',
                    'pisr_9','pisr_10','pisr_11','pisr_12',
                    'windexp500','svf','mbi','norm_height','mpi2000','mid_slope_position',
                    'swi_suction16', 'swi_suction256',
                    'dem10m','diurnalaniheat',]
    
    for vrb in vrbs:
        if vrb not in include_list and vrb in vrbs:
            dem = dem.drop(vrb)
    
    features_50 = calc_dem_features(dem[vrb0].values, dem.y.values, dem.x.values, [50,50])
    features_3  = calc_dem_features(dem[vrb0].values, dem.y.values, dem.x.values, [3,3])
    
    features = features_3 | features_50
    
    
    
    for key, val in zip(features.keys(), features.values()):
        
        dem[key] = xr.full_like(dem[vrb0], np.nan)
        dem[key].values = val
        dem[key].attrs = {}
    
    features_other = tfe.build_static_predictors(dem)
    
    dem = xr.merge([dem,features_other])
    
    for v in dem.data_vars:
        dem = dem.rename({v:'St_'+v})
        print(v, 'St_'+v)  
    
    
    return dem




def define_mask(dem):
    
    xx, yy = np.meshgrid(dem.lon.values, dem.lat.values)
    
    b, a = np.polyfit([29.62, 29.69], [67.779, 67.754], 1)
    mask = (yy < xx*b + a)
    
    for v in list(dem.data_vars):
        v_mask = ~np.isnan(dem[v].values)
        mask = mask*v_mask
    
    
    
    mask_ds = xr.full_like(dem['dtm'], 0).rename({'dtm', 'mask'}); 
    mask_ds[:] = mask; mask_ds = mask_ds.where(mask, other=np.nan)
    
    return mask, mask_ds





'''
def derive_X_data(df_dict, dem, points,)# lags, all_yrs):
    
    t = stopwatch('start')
    
    G = define_lags(df_dict)
    print('Gridded predictor data defined')
    
    oo = dem.sel(   lat=xr.DataArray(np.array(points)[:,0], dims='gcl'), \
                    lon=xr.DataArray(np.array(points)[:,1], dims='gcl'), \
                    method='nearest').to_dataframe().reset_index().sort_values(['gcl']).astype(float)
    
    for v in oo.columns:
        if v != 'gcl':
            oo = oo.rename(columns={v: 'Dm_'+v})
    
    
    O = pd.merge(left=G[['gcl','time','lon','lat']],right=oo,on='gcl')
    print('DEM predictor data defined')
    
    
    C = cyclical_predictors(pd.DataFrame(index=G.time), ann=True, diu=True).reset_index()
    C.index = G.index
    print('Cyclical predictor data defined')
    
    
    X = pd.concat([G, O, C],axis=1) #.rename_axis('time').reset_index()
    X = X.loc[:,~X.columns.duplicated()]
    
    print('Creation of X matrix took ' + stopwatch('stop', t))
    return X.reset_index(drop=True)


'''

#def derive_X_data(df_dict, dem, points):
def derive_X_data(df, dem_df, points):
    import datetime
    
    t = stopwatch('start')
    
    #G = define_lags(df_dict).reset_index(drop=True)
    G = df
    print('Gridded predictor data defined')
    
    #oo = dem.sel(   lat=xr.DataArray(np.array(points)[:,0], dims='gcl'), \
    #                lon=xr.DataArray(np.array(points)[:,1], dims='gcl'), \
    #                method='nearest').to_dataframe().reset_index().sort_values(['gcl']).astype(float)
    oo = dem_df
    
    oo['gcl'] = oo['gcl'].astype(int)
    
    for v in oo.columns:
        if v != 'gcl': 
            oo = oo.rename(columns={v: 'Dm_'+v})
    
    
    O = pd.merge(left=G[['gcl','time','lon','lat']],right=oo,on=['gcl',])
    print('DEM predictor data defined')
    
    
    C = cyclical_predictors(pd.DataFrame(index=G.time), ann=True, diu=True, chn=True).reset_index()
    C.index = G.index
    print('Cyclical predictor data defined')
    
    
    #Z = G[['gcl','time','lon','lat']].copy(deep=True); Z.index = G.index
    #Z['Cy_zenith_angle'] = calculate_solar_zenith_angle(Z.lat.values, Z.lon.values, Z.time.values)  
    #print('Solar zenith angle predictor data defined')
    
    
    X = pd.concat([G, O, C,],axis=1) 
    X = X.loc[:,~X.columns.duplicated()]
    
    print('Creation of X matrix took ' + stopwatch('stop', t))
    return X.reset_index(drop=True)




def define_lags(df_dict):
    
    lags = list(df_dict.keys())
    df   = list(df_dict.values())[0]
    
    drops = []
    for c in ['gcl', 'time', 'lon', 'lat']:
        if c in list(df.columns): drops.append(c)
    
    # Create columns, fill in with data
    new_columns = []    
    for lag in lags:
        for col in list(df.drop(columns=drops).columns):
            sign = '+'
            if np.sign(lag)==1: sign = '-'
            
            lg = "{:03d}".format(np.abs(lag))
            
            output_vrb = 'E5_'+col+'_'+sign+lg
            new_columns.append(output_vrb)
                
    
    index = df.index
    columns = drops+new_columns
    
    data1 = df[drops]
    data2 = pd.concat(list(df_dict.values()), axis=1).drop(columns=drops)
    data = pd.concat([data1,data2.astype(float)],axis=1)#.values
    
    #renaming_dict = dict(map(lambda i,j : (i,j) , list(data.columns),columns)) # dict(zip(list(data.columns), columns))
    #data_out = pd.DataFrame(index=index, data=data, columns=columns)#.convert_dtypes()
    
    #data_out[new_columns] = data_out[new_columns].astype(float)
    data.columns = columns
    
    return data



'''
def define_lags(grid_df, lags, dropna=True):
    
    drops = []
    for c in ['gcl', 'time', 'lon', 'lat', 'expver']:
        if c in list(grid_df.columns): drops.append(c)
    
    metadata = ['gcl', 'time', 'lon', 'lat']
    
    # Create columns, fill in with data
    new_columns = []
    new_data = []
    for col in list(grid_df.drop(columns=drops).columns):
        for lag in lags:
            sign = '+'
            if np.sign(lag)==1: sign = '-'
            
            lg = "{:03d}".format(np.abs(lag))
            
            output_vrb = 'E5_'+col+'_'+sign+lg
            new_columns.append(output_vrb)
            new_data.append(grid_df[col].shift(lag).values)
                
    
    new_data = pd.DataFrame(index=grid_df.index, columns=new_columns, data=np.array(new_data).T)
    data = pd.concat([grid_df[metadata],new_data.astype(float)],axis=1)
    
    if dropna:
        data = data.dropna(axis='index', how='any')
    
    return data
'''




def nearest_gridpoints(ds, points):
    import xarray as xr
    
    grid = xr.Dataset({"lat": (["lat"], ds.lat.values),"lon": (["lon"], ds.lon.values),})
    
    points_grid = grid.sel(lat=xr.DataArray(np.array(points)[:,0], dims='gcl'), \
                           lon=xr.DataArray(np.array(points)[:,1], dims='gcl'), method='nearest')
    
    nearest_points = [tuple(row) for row in np.array([points_grid.gcl.lat.values, 
                                                      points_grid.gcl.lon.values]).T]
    
    return nearest_points


def read_mf_s3(fs, remote_paths, **kwargs):
    
    
    # Open a set of remote netCDF files, leave them open
    opened_files = []
    for file_path in remote_paths:
        opened_files.append(fs.open(file_path))
    
    # Read lazily
    ds = xr.open_mfdataset(opened_files, **kwargs)

    return ds



def bool_index_to_int_index(bool_index):
    return np.where(bool_index)[0]




def dir_as_sincos(theta):
    dsin = np.sin(theta*(np.pi/180.))
    dcos = np.cos(theta*(np.pi/180.))
    return dsin, dcos



def adjust_lats_lons(ds):
    coord_names =   [['longitude', 'latitude'],
                    ['X', 'Y'],]
    
    for nmes in coord_names:
        try:
            ds = ds.rename({nmes[0]: 'lon', nmes[1]: 'lat'}) 
        except: 
            pass  
    
    
    if(ds.lon.values.max() > 180):
        print('Transforming longitudes from [0,360] to [-180,180]', flush=True)
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        
    return ds.sortby(['lon','lat'])
    







def cyclical_predictors(X, ann=False, mon=False, wee=False, diu=False, chn=False, hol=False):
    
    import pandas as pd
    from datetime import date
    
    
    A = pd.DataFrame(index=X.index)
    
    if ann:
        # Annual cycles 
        ann_sin = np.sin(2 * np.pi * X.index.dayofyear/366.)
        ann_cos = np.cos(2 * np.pi * X.index.dayofyear/366.)
        A['Cycle_ann_sin'] = ann_sin
        A['Cycle_ann_cos'] = ann_cos
    
    if mon:
        # Monthly cycles 
        mon_sin = np.sin(2 * np.pi * X.index.days_in_month/31.)
        mon_cos = np.cos(2 * np.pi * X.index.days_in_month/31.)
        A['Cycle_mon_sin'] = mon_sin
        A['Cycle_mon_cos'] = mon_cos
    
    if wee:
        # Weekly cycles 
        wee_sin = np.sin(2 * np.pi * X.index.weekday/7.)
        wee_cos = np.cos(2 * np.pi * X.index.weekday/7.)
        A['Cycle_wee_sin'] = wee_sin
        A['Cycle_wee_cos'] = wee_cos  
    
    if diu:
        # Diurnal cycles 
        diu_sin = np.sin(2 * np.pi * X.index.hour/24.)
        diu_cos = np.cos(2 * np.pi * X.index.hour/24.)
        A['Cycle_diu_sin'] = diu_sin
        A['Cycle_diu_cos'] = diu_cos
    
    if chn:
        # Long term changes along years
        A['Cycle_years'] = X.index.year 
    
    return A



# --- Fitting ---


def params_lasso():
    params = dict(
        n_jobs=-1,
        n_estimators=50,
        alpha=0.01,
        p_smpl=0.6, 
        p_feat=0.6,)
    return params







from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LassoLarsCV, QuantileRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV, MultiTaskLassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings('ignore', category=ConvergenceWarning)

def bagging_model(X, Y, params,):
    # Clean data
    X = X.ffill().bfill().fillna(0)
    Y = Y.ffill().bfill().fillna(0)
    
    cv = KFold(3, shuffle=False)
    steps = [('normalizer', QuantileTransformer(output_distribution='normal')), 
             ('lasso', LassoLarsCV(cv=cv, eps=0.1, max_iter=1000))]
    
    base_estim = Pipeline(steps)

    # Bagging wrapper — use n_jobs=1 here to avoid oversubscription
    bagging = BaggingRegressor(
        estimator=base_estim,
        n_estimators=params['n_estimators'],
        max_samples=params['p_smpl'], 
        max_features=params['p_feat'],
        bootstrap=False,
        bootstrap_features=False,
        oob_score=False,
        n_jobs=1,  # single-threaded inside BaggingRegressor
        random_state=99,
        verbose=False)

    # Multi-output wrapper 
    model = MultiOutputRegressor(bagging, n_jobs=50)

    # Fit model
    model.fit(X, Y)

    return model




def bagging_model_fast(X, Y, params):
    # Clean (keep float64 to avoid Gram issues)
    X = np.asarray(X.ffill().bfill().fillna(0.0), dtype=np.float64)
    Y = np.asarray(Y.ffill().bfill().fillna(0.0), dtype=np.float64)

    cv = KFold(3, shuffle=False)

    bag = BaggingRegressor(
        estimator=LassoCV(cv=cv, max_iter=5000, precompute=False, n_jobs=1, random_state=0),
        n_estimators=params['n_estimators'],
        max_samples=params['p_smpl'],
        max_features=params['p_feat'],
        bootstrap=False,
        bootstrap_features=False,
        oob_score=False,
        n_jobs=1,
        random_state=99,
        verbose=False
    )

    # Fit QuantileTransformer ONCE, then bag across targets (modest parallelism)
    model = Pipeline([
        ('normalizer', QuantileTransformer(output_distribution='normal',
                                           n_quantiles=min(1024, len(X)),
                                           subsample=min(100_000, len(X)),
                                           random_state=0)),
        ('bag_multi', MultiOutputRegressor(bag, n_jobs=min(8, Y.shape[1]))),
    ])

    model.fit(X, Y)
    return model





import xgboost as xgb
def xgb_default_estim():
    
    # Define the model
    estim = xgb.XGBRegressor(
        enable_categorical=False,
        
        objective='reg:squarederror',
        eval_metric='rmse',
        #objective='reg:pseudohubererror',
        #eval_metric='mphe',
        #huber_slope=1,
        
        tree_method='hist',
        max_bin=256,
        booster='gbtree',
        n_estimators=500,  
        #early_stopping_rounds=100,
        learning_rate=0.03, 
        max_depth=8,
        reg_alpha=0.01,
        reg_lambda=0.01,
        gamma=0.1,
        n_jobs=8, 
        subsample=0.65, 
        colsample_bytree=0.65,
        colsample_bynode=1, 
        min_child_weight=2,
        random_state=99)
    
    return estim


def define_xgb(params):
    return xgb.XGBRegressor(**params)


def fit_ensemble(X_trn, Y_trn, X_val, Y_val, base_estim, verbose=False):
    
    fitting = base_estim.fit(
        X_trn, Y_trn,
        eval_set=[  (X_trn, Y_trn), 
                    (X_val, Y_val)],
        verbose=verbose)
    
    return fitting





import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
import gc
#from optuna.integration import XGBoostPruningCallback
import optuna 

def optuna_objective_xgb(trial,X,Y):
    
    # Precompute ISO week once
    data_weeks = pd.to_datetime(X.time.values).isocalendar().week.values
    weeks = np.unique(data_weeks)
    #weeks = np.arange(1,53)
    
    param = {
        'n_jobs': 6,
        'verbosity': 1,
        'booster': 'gbtree',
        
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        #'objective': 'reg:pseudohubererror',
        #'eval_metric': 'mphe',
        #'base_score': float(np.median(Y)),
        
        'tree_method': 'hist',
        'max_bin': 256,
        'n_estimators': 400,
        'early_stopping_rounds': 100,
        
        #'huber_slope':      trial.suggest_float('huber_slope',      0.05,   20.0, log=True),
        'reg_alpha':        trial.suggest_float('reg_alpha',        1e-8,   10.0),# log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda',       1e-8,   10.0),# log=True),
        'gamma':            trial.suggest_float('gamma',            1e-8,   10.0),# log=True),
        'learning_rate':    trial.suggest_float('learning_rate',    1e-3,   2.0,),# log=True),
        'subsample':        trial.suggest_float('subsample',        0.5,    1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1,    1.0),
        'max_depth':        trial.suggest_int('max_depth',          4,      16, step=1),
        'min_child_weight': trial.suggest_int('min_child_weight',   1,      60, step=2),}
    
    

    kf = KFold(3, shuffle=True, random_state=33) 
        
    scores = []#; r2_scores = []
    for trn_idx, val_idx in kf.split(weeks):
        
        #data_weeks = pd.to_datetime(X.time.values).isocalendar().week.values
        
        #X_trn = X.loc[np.isin(data_weeks, weeks[trn_idx])] 
        #Y_trn = Y.loc[np.isin(data_weeks, weeks[trn_idx])] 
        #X_val = X.loc[np.isin(data_weeks, weeks[val_idx])] 
        #Y_val = Y.loc[np.isin(data_weeks, weeks[val_idx])] 
        
        trn_weeks = weeks[trn_idx]
        val_weeks = weeks[val_idx]

        trn_mask = np.isin(data_weeks, trn_weeks)
        val_mask = np.isin(data_weeks, val_weeks)

        X_trn = X.loc[trn_mask].drop(columns="time")
        Y_trn = Y.loc[trn_mask]
        X_val = X.loc[val_mask].drop(columns="time")
        Y_val = Y.loc[val_mask]
        
        model = define_xgb(param)
        #model = fit_ensemble(
        #    X_trn.drop(columns='time'), Y_trn, X_val.drop(columns='time'), Y_val, 
        #    estimator, verbose=50)
        
        model.fit(
            X_trn, Y_trn,
            eval_set=[  (X_trn, Y_trn), 
                        (X_val, Y_val)],
            verbose=50,
            #callbacks=[OptunaPruningCallback(trial, "validation_0-mphe", interval=10)]
        )
        
        #pred_labels = model.predict(X_val)
        
        #score = calc_rmse(Y_val.values, pred_labels)
        #score = calc_pseudo_huber(Y_val.values, pred_labels)
        
        #print(score)#, r2_score)
        #scores.append(score)
        print(model.best_score)
        scores.append(model.best_score)
        
        del model
        gc.collect()
        
        
    
    return float(np.nanmean(scores))





import optuna
def tune_hyperparams(X_train, Y_train, num_trials=100, params_in=False):
    """
    Hyperparameter tuning using Optuna.
    """
    
    
    
    # Set up Optuna sampler with a fraction of trials dedicated to exploration
    num_startup_trials = int(num_trials / 3)
    sampler = optuna.samplers.TPESampler(n_startup_trials=num_startup_trials)
    
    
    # Create an Optuna study 
    #storage = "sqlite:///optuna_study.db"
    study = optuna.create_study(direction='minimize', sampler=sampler,)# storage=storage, load_if_exists=True)
    
    
    # Initialize parameters for the first trial
    if not params_in:
        params = xgb_default_estim().get_params()
    else:
        params = params_in.copy()
    
    # Enqueue the first trial
    study.enqueue_trial(params)
    
    # Optimize hyperparameters 
    study.optimize(lambda trial: optuna_objective_xgb(trial, X_train, Y_train), n_trials=num_trials, n_jobs=40)
    
    # Update the initial parameters with the best hyperparameters found
    best_params = study.best_params
    params.update(best_params)
    
    print('Best trial:', best_params, flush=True)
    
    df_result = study.trials_dataframe()
    #df.to_csv("optuna_results.csv", index=False)
    
    return params, df_result










def stopwatch(start_stop, t=-99):
    import time
    
    if start_stop=='start':
        t = time.time()
        return t
    
    if start_stop=='stop':
        elapsed = time.time() - t
        return time.strftime("%H hours, %M minutes, %S seconds",time.gmtime(elapsed))




def sample_points(ds, indices, min_distance_threshold, num_samples, other_coords=[]):
    
    from scipy.spatial.distance import cdist
    
    latitude_samples = []
    longitude_samples = []
    #distances = []
    
    
    i=0
    np.random.seed(5)
    while len(latitude_samples) < num_samples:
        
        # Randomly select a candidate latitude and longitude
        random_index = np.random.randint(0, len(indices[0]))
        
        candidate_latitude = ds.lat.values[indices[0][random_index]]
        candidate_longitude = ds.lon.values[indices[1][random_index]]
        candidate_coords = np.array([(candidate_latitude, candidate_longitude)])
        
        if len(latitude_samples) == 0:
            
            # If no sampled points exist yet, add the candidate point to the samples
            latitude_samples.append(candidate_latitude)
            longitude_samples.append(candidate_longitude)
            #existing_coords = candidate_coords
            
        
        # Check the distance between the candidate point and existing sampled points
        existing_coords = np.array(list(zip(latitude_samples, longitude_samples)))
        if len(other_coords) > 0:
            existing_coords = np.append(existing_coords, other_coords, axis=0)
        
        distances = cdist(candidate_coords, existing_coords)
        min_distance = distances.min(axis=1)
        
        if min_distance >= min_distance_threshold:
            # If the candidate point satisfies the distance constraint, add it to the samples
            latitude_samples.append(candidate_latitude)
            longitude_samples.append(candidate_longitude)
        
        i+=1
    
    print(num_samples,'points sampled with',i,'iterations')
    return latitude_samples, longitude_samples


def dist_points(point1, point2):
    
    radius = 6371 # Earth radius
    
    lat1 = point1[0]*np.pi/180.
    lat2 = point2[0]*np.pi/180.
    
    lon1 = point1[1]*np.pi/180.
    lon2 = point2[1]*np.pi/180.
    
    deltaLat = lat2-lat1
    deltaLon = lon2-lon1
    
    a = np.sin((deltaLat)/2.)**2 + np.cos(lat1)*np.cos(lat2) * np.sin(deltaLon/2.)**2
    
    c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    
    x = deltaLon*np.cos((lat1+lat2)/2.)
    y = deltaLat
    
    d1 = radius*c                   # Haversine distance
    d2 = radius*np.sqrt(x*x + y*y)  # Pythagoran distance
    
    return d1,d2



# --- Plotting ---

# Ridgeline plots for the metrics dataframe (by_allgroups) using matplotlib only
# No seaborn, no subplots, no explicit colors.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helper: smooth a histogram with a gaussian kernel (no scipy needed) ----------
def _gaussian_kernel(size=81, sigma=8.0):
    x = np.linspace(-int(size//2), int(size//2), int(size))
    k = np.exp(-0.5*(x/sigma)**2)
    k /= k.sum()
    return k

def _kde_like(y, bins=200, smooth_sigma=8.0, xlim=None):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return None, None

    if xlim is None:
        lo, hi = np.nanmin(y), np.nanmax(y)
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
    else:
        lo, hi = xlim

    hist, edges = np.histogram(y, bins=bins, range=(lo, hi), density=True)

    # build kernel (can be longer than hist)
    k = _gaussian_kernel(size=min(401, 2*bins+1), sigma=smooth_sigma)
    dens = np.convolve(hist, k, mode="same")

    # if kernel > hist, 'same' gives len=max(M,N); trim to 'bins' length
    if len(dens) != len(hist):
        start = (len(dens) - len(hist)) // 2
        dens = dens[start:start + len(hist)]

    x = 0.5*(edges[:-1] + edges[1:])
    return x, dens


# ---------- main plotting function ----------
def plot_ridgeline(
    df,
    metric="RMSE",
    group_col="model",
    *,
    filter_dict=None,
    max_groups=10,
    bins=200,
    smooth_sigma=10.0,
    height_per_group=0.6,
    overlap=0.35,
    sort_by="median",   # "median", "mean", or "name"
    figsize=(10, 8),
    title=None,
):
    """
    Create a ridgeline plot of `metric` distributions across categories in `group_col`.
    - df: your DataFrame (e.g., by_allgroups)
    - metric: one of ["RMSE","MAE","Bias","R","R2"]
    - group_col: e.g., "model", "month", "region", or "site"
    - filter_dict: dict like {"model":"xgboost_T3", "region":"ULV"} to pre-filter
    - max_groups: limit the number of groups shown (auto-chosen by data volume)
    - sort_by: order groups by "median", "mean", or alphabetic "name"
    """
    if filter_dict:
        q = np.ones(len(df), dtype=bool)
        for k, v in filter_dict.items():
            if isinstance(v, (list, tuple, set)):
                q &= df[k].isin(list(v))
            else:
                q &= (df[k] == v)
        data = df.loc[q].copy()
    else:
        data = df.copy()

    # Keep only finite metric values
    data = data[np.isfinite(data[metric])]

    if data.empty:
        raise ValueError("No finite data to plot after filtering.")

    # Determine group order
    gb = data.groupby(group_col)[metric]
    stats = gb.agg(["count", "median", "mean"]).reset_index()
    # prefer groups with more data
    stats = stats.sort_values("count", ascending=False)
    stats = stats.head(max_groups)

    if sort_by == "median":
        stats = stats.sort_values("median", ascending=True)
    elif sort_by == "mean":
        stats = stats.sort_values("mean", ascending=True)
    elif sort_by == "name":
        stats = stats.sort_values(group_col, ascending=True)

    groups = stats[group_col].tolist()
    # Shared x-limits across all groups
    xmin = data.loc[data[group_col].isin(groups), metric].min()
    xmax = data.loc[data[group_col].isin(groups), metric].max()
    if xmin == xmax:
        xmin, xmax = xmin - 1.0, xmax + 1.0

    # Layout math (single axes)
    n = len(groups)
    vspace = height_per_group * (1 - overlap)
    total_height = height_per_group + (n - 1) * vspace
    fig, ax = plt.subplots(figsize=figsize)
    baseline = 0.0

    # Plot each group
    for i, gname in enumerate(groups):
        y = data.loc[data[group_col] == gname, metric].values
        x, dens = _kde_like(y, bins=bins, smooth_sigma=smooth_sigma, xlim=(xmin, xmax))
        if x is None:
            continue

        # Normalize each density to max=1 for consistent vertical scaling
        if np.nanmax(dens) > 0:
            dens = dens / np.nanmax(dens)

        yoff = baseline + i * vspace

        # Filled curve
        ax.fill_between(x, yoff, yoff + dens, alpha=0.8, linewidth=1.5)

        # White outline for separation
        ax.plot(x, yoff + dens, linewidth=1.5)

        # Baseline
        ax.hlines(yoff, xmin, xmax, linewidth=1.5)

        # Label on the left margin
        ax.text(xmin, yoff + 0.2, str(gname), ha="left", va="center", fontweight="bold")

    # Axes styling
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.2, baseline + (n - 1) * vspace + height_per_group + 0.2)
    ax.set_yticks([])
    ax.set_xlabel(metric)
    ax.set_title(title or f"{metric} distribution by {group_col} (n={n} groups)")

    # Clean spines
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()

# -------------------- Examples --------------------
# 1) Across models (all months/regions/sites):
# plot_ridgeline(by_allgroups, metric="RMSE", group_col="model", max_groups=4, title="RMSE by model")

# 2) Within one region, show months for a given model:
# plot_ridgeline(by_allgroups,
#                metric="RMSE",
#                group_col="month",
#                filter_dict={"region": "ULV", "model": "xgboost_T3"},
#                max_groups=12,
#                sort_by="name",
#                title="RMSE by month • ULV • xgboost_T3")

# 3) Compare regions for a model and month:
# plot_ridgeline(by_allgroups,
#                metric="MAE",
#                group_col="region",
#                filter_dict={"model": "lasso_T3", "month": 7},
#                title="MAE by region • lasso_T3 • July")

# 4) If you really want sites, cap the number shown and maybe prefilter:
# plot_ridgeline(by_allgroups.query("region == 'KIL' and model == 'xgboost_T3'"),
#                metric="R2",
#                group_col="site",
#                max_groups=20,
#                sort_by="median",
#                title="R² by site • KIL • xgboost_T3")




