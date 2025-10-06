
"""
terrain_feature_engineering.py
--------------------------------
Utilities to derive static and dynamic geospatial predictors to model hourly
near-surface air temperature in complex terrain.

Requirements:
    - xarray
    - numpy
    - scipy (optional, for ndimage uniform_filter if available)

All functions are designed to be robust to missing variables in your Dataset.
They will check for presence and log (print) what they can/cannot compute.
Grids are assumed to be in projected meters with (y, x) dims. Angles are degrees.

Author: ChatGPT
Date: 2025-09-02
"""

from __future__ import annotations

import numpy as np
import xarray as xr

try:
    from scipy.ndimage import uniform_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -------------------------------
# Helpers
# -------------------------------

def _zscore(da: xr.DataArray) -> xr.DataArray:
    """Standardize a DataArray over spatial dims (y,x)."""
    mean = da.mean(dim=("y", "x"), skipna=True)
    std = da.std(dim=("y", "x"), skipna=True)
    return (da - mean) / (std + 1e-12)


def _norm01(da: xr.DataArray) -> xr.DataArray:
    """Min-max to [0,1] over spatial dims (y,x)."""
    dmin = da.min(dim=("y","x"), skipna=True)
    dmax = da.max(dim=("y","x"), skipna=True)
    return (da - dmin) / ( (dmax - dmin) + 1e-12 )


def _rolling_mean(da: xr.DataArray, win: int) -> xr.DataArray:
    """NaN-aware boxcar mean using either scipy (fast) or xarray rolling."""
    if _HAS_SCIPY:
        # convert to numpy, apply uniform filter ignoring NaN via mask trick
        arr = da.values
        mask = np.isfinite(arr).astype(float)
        arr_filled = np.where(np.isfinite(arr), arr, 0.0)

        size = (win, win)
        num = uniform_filter(arr_filled, size=size, mode="nearest")
        den = uniform_filter(mask, size=size, mode="nearest")
        out = np.where(den > 0, num / den, np.nan)
        return xr.DataArray(out, coords=da.coords, dims=da.dims, name=f"{da.name}_rm{win}")
    else:
        return da.rolling(y=win, x=win, center=True).mean()


def _deg2rad(da: xr.DataArray | float) -> xr.DataArray | float:
    return np.deg2rad(da)


def _wrap_dir_deg(angle):
    """Wrap direction to [0,360)."""
    return (angle % 360 + 360) % 360


# -------------------------------
# Static terrain derivatives
# -------------------------------

def terrain_ruggedness_index(dem: xr.DataArray, win: int = 5) -> xr.DataArray:
    """
    Very simple TRI (mean absolute elevation difference in a window).
    """
    local_mean = _rolling_mean(dem, win)
    tri = np.abs(dem - local_mean)
    tri = _rolling_mean(tri, win)
    tri.name = f"tri_w{win}"
    tri.attrs["long_name"] = f"Terrain Ruggedness Index (win={win})"
    return tri


def vector_ruggedness_measure(slope_deg: xr.DataArray, aspect_deg: xr.DataArray, win: int = 5) -> xr.DataArray:
    """
    VRM following Sappington et al. (2007). Requires slope (deg) and aspect (deg).
    """
    s = _deg2rad(slope_deg)
    a = _deg2rad(aspect_deg)
    # unit normal components
    sx = np.sin(s) * np.cos(a)
    sy = np.sin(s) * np.sin(a)
    sz = np.cos(s)

    # rolling mean of vectors
    mx = _rolling_mean(xr.DataArray(sx, coords=slope_deg.coords, dims=slope_deg.dims), win)
    my = _rolling_mean(xr.DataArray(sy, coords=slope_deg.coords, dims=slope_deg.dims), win)
    mz = _rolling_mean(xr.DataArray(sz, coords=slope_deg.coords, dims=slope_deg.dims), win)

    R = np.sqrt(mx**2 + my**2 + mz**2)
    vrm = 1 - R
    vrm.name = f"vrm_w{win}"
    vrm.attrs["long_name"] = f"Vector Ruggedness Measure (win={win})"
    return vrm


def multiscale_contrasts(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute differences between small and large window morphometrics.
    Expect slope/aspect/top_posn_idx at w3 and w50 scales if available.
    """
    out = xr.Dataset()

    def add_if_present(name_small, name_large, out_name):
        if (name_small in ds) and (name_large in ds):
            out[out_name] = ds[name_small] - ds[name_large]

    add_if_present("slope_degrees_w3", "slope_degrees_w50", "slope_diff_w3_w50")

    # Aspect difference via vector representation to avoid wrap-around issues
    if ("aspect_cos_w3" in ds) and ("aspect_cos_w50" in ds) and ("aspect_sin_w3" in ds) and ("aspect_sin_w50" in ds):
        v3 = xr.apply_ufunc(np.hypot, ds["aspect_cos_w3"] - ds["aspect_cos_w50"],
                            ds["aspect_sin_w3"] - ds["aspect_sin_w50"])
        v3.name = "aspect_vecdiff_w3_w50"
        out[v3.name] = v3

    add_if_present("top_posn_idx_w3", "top_posn_idx_w50", "tpi_diff_w3_w50")
    return out


def cold_air_pooling_index(ds: xr.Dataset) -> xr.DataArray:
    """
    Heuristic Cold-Air Pooling Index (CAPI) combining:
      - low normalized height (norm_height)
      - negative MBI (valley tendency)
      - low SVF (enclosure)
    Returns a 0..1 index (higher = stronger CAP potential).
    """
    parts = []

    if "norm_height" in ds:
        # lower norm_height => stronger CAP, so invert
        parts.append(_zscore(-ds["norm_height"]).rename("capi_norm_height"))
    if "mbi" in ds:
        parts.append(_zscore(-ds["mbi"]).rename("capi_mbi"))
    if "svf" in ds:
        parts.append(_zscore(1 - _norm01(ds["svf"])).rename("capi_svf"))  # lower svf -> higher

    if not parts:
        raise ValueError("Need at least one of ['norm_height','mbi','svf'] to compute CAPI.")

    capi = sum(parts) / len(parts)
    capi = _norm01(capi)
    capi.name = "capi"
    capi.attrs["long_name"] = "Cold-Air Pooling Index (0..1)"
    return capi


def openness_proxy(ds: xr.Dataset, win: int = 25) -> xr.DataArray:
    """
    Proxy for positive openness (exposure) using SVF and local relief.
    True openness needs horizon angles; this uses:
      openness ≈ 0.5*SVF_norm + 0.5*(|DEM - mean DEM|)norm over a window.
    """
    if "dem10m" not in ds:
        raise ValueError("dem10m is required for openness_proxy")

    relief = np.abs(ds["dem10m"] - _rolling_mean(ds["dem10m"], win))
    relief = _norm01(relief)

    svf_norm = _norm01(ds["svf"]) if "svf" in ds else 0.5
    openness = 0.5 * svf_norm + 0.5 * relief
    openness.name = f"openness_proxy_w{win}"
    openness.attrs["long_name"] = f"Openness Proxy (win={win})"
    return openness


# -------------------------------
# Dynamic (hourly) couplings to atmos forcing
# -------------------------------

def month_from_time(time: xr.DataArray) -> xr.DataArray:
    """Return month index (1..12) from time coordinate."""
    return xr.DataArray(time.dt.month, coords=time.coords, dims=time.dims)


def effective_solar_factor_from_pisr(ds: xr.Dataset, time: xr.DataArray) -> xr.DataArray:
    """
    Derive an hourly 'effective radiation factor' using monthly PISR fields (pisr_1..pisr_12).
    This scales external hourly SW downwelling by terrain-dependent monthly potential.
    Output is a factor in [0,1] per (y,x) for each timestamp in 'time'.
    """
    # Stack PISR into a month dimension
    pisr_list = []
    for m in range(1, 13):
        key = f"pisr_{m}"
        if key in ds:
            pisr_list.append(ds[key].expand_dims({"month":[m]}))
    if not pisr_list:
        raise ValueError("No pisr_1..pisr_12 fields found in dataset.")
    pisr = xr.concat(pisr_list, dim="month")

    # Normalize by monthly max to get [0,1] spatial factor
    pisr_norm = pisr / (pisr.max(dim=("y","x"), skipna=True) + 1e-12)

    # Map each time to its month, then select
    mo = month_from_time(time)
    fac = pisr_norm.sel(month=mo)
    fac = fac.rename("effective_solar_factor")
    fac.attrs["long_name"] = "Monthly terrain-modulated solar factor (0..1)"
    return fac


def wind_components_along_slope(wind_speed: xr.DataArray,
                                wind_dir_deg_met: xr.DataArray,
                                aspect_deg: xr.DataArray) -> xr.Dataset:
    """
    Project 10-m wind (meteorological direction: where wind comes FROM, deg clockwise from North)
    onto along-slope and cross-slope components. Positive along-slope means blowing upslope.
    """
    # Convert met direction (FROM) to flow-to azimuth
    flow_to = _wrap_dir_deg(wind_dir_deg_met + 180.0)

    # Difference between flow direction and slope aspect (downslope direction is aspect+180)
    # We want upslope alignment, so compare flow_to with (aspect + 180)
    upslope_dir = _wrap_dir_deg(aspect_deg + 180.0)
    theta = np.deg2rad(_wrap_dir_deg(flow_to - upslope_dir))

    along = wind_speed * np.cos(theta)
    cross = wind_speed * np.sin(theta)

    ds_out = xr.Dataset()
    ds_out["wind_along_slope"] = along
    ds_out["wind_cross_slope"] = cross
    ds_out["wind_along_slope"].attrs["long_name"] = "Wind component along slope (+upslope)"
    ds_out["wind_cross_slope"].attrs["long_name"] = "Wind component cross slope (+left of upslope)"
    return ds_out


def wind_sheltering_factor(windexp: xr.DataArray) -> xr.DataArray:
    """
    Normalize wind exposure index to 0..1 (1 = highly exposed, 0 = sheltered).
    If index is 'exposure', we just min-max. If it's 'protection', invert first.
    """
    # Heuristics: if values are mostly positive, assume exposure; if mostly negative, assume protection.
    finite = np.isfinite(windexp)
    pos_frac = (windexp.where(finite) > 0).mean().item()
    if pos_frac < 0.5:
        windexp = -windexp
    return _norm01(windexp).rename("wind_shelter_exposure")


def effective_wind_speed(wind_speed: xr.DataArray, windexp: xr.DataArray, k: float = 0.6) -> xr.DataArray:
    """
    Blend free-air wind with topographic exposure: ws_eff = ws * ( (1-k) + k * exposure )
    where exposure in [0,1]. k sets how strongly terrain modulates wind felt at site.
    """
    exposure = wind_sheltering_factor(windexp)
    ws_eff = wind_speed * ((1 - k) + k * exposure)
    ws_eff.name = "wind_speed_effective"
    ws_eff.attrs["long_name"] = f"Effective wind speed blended with exposure (k={k})"
    return ws_eff


def cold_bias_predictor(ds_static: xr.Dataset, elev_name: str = "dem10m") -> xr.DataArray:
    """
    Combine CAP index and elevation to create a night-time cold bias predictor.
    """
    capi = cold_air_pooling_index(ds_static)
    if elev_name not in ds_static:
        raise ValueError(f"{elev_name} not found in Dataset")
    bias = _zscore(capi) + 0.5 * _zscore(ds_static[elev_name])
    bias = _norm01(bias)
    bias.name = "cold_bias_potential"
    bias.attrs["long_name"] = "Potential for nocturnal cold bias (0..1)"
    return bias


# -------------------------------
# High-level orchestrator
# -------------------------------

def build_static_predictors(ds: xr.Dataset) -> xr.Dataset:
    """
    Build a static predictor stack from available variables.
    Returns a Dataset with derived variables added (does not modify input).
    """
    out = xr.Dataset()

    # Basic morphometry
    if "dem10m" in ds:
        out["tri_w3"] = terrain_ruggedness_index(ds["dem10m"], win=3)
        out["tri_w50"] = terrain_ruggedness_index(ds["dem10m"], win=50)

    # VRM at two scales (prefer w3/w50 if present for slope/aspect)
    if ("slope_degrees_w3" in ds) and ("aspect_degrees_w3" in ds):
        out["vrm_w3"] = vector_ruggedness_measure(ds["slope_degrees_w3"], ds["aspect_degrees_w3"], win=3)
    if ("slope_degrees_w50" in ds) and ("aspect_degrees_w50" in ds):
        out["vrm_w50"] = vector_ruggedness_measure(ds["slope_degrees_w50"], ds["aspect_degrees_w50"], win=50)

    # Multiscale contrasts
    out = xr.merge([out, multiscale_contrasts(ds)])

    # CAP index
    try:
        out["capi"] = cold_air_pooling_index(ds)
    except Exception as e:
        print(f"[warn] Could not compute CAPI: {e}")

    # Openness proxy
    try:
        out["openness_proxy_w3"] = openness_proxy(ds, win=3)
        out["openness_proxy_w50"] = openness_proxy(ds, win=50)
    except Exception as e:
        print(f"[warn] Could not compute openness_proxy: {e}")

    # Nighttime cold bias potential
    try:
        out["cold_bias_potential"] = cold_bias_predictor(ds)
    except Exception as e:
        print(f"[warn] Could not compute cold_bias_potential: {e}")

    return out


def build_hourly_predictors(ds_static: xr.Dataset,
                            time: xr.DataArray,
                            swdown: xr.DataArray | None = None,
                            wind_speed: xr.DataArray | None = None,
                            wind_dir_deg_met: xr.DataArray | None = None,
                            aspect_for_wind: xr.DataArray | None = None,
                            windexp: xr.DataArray | None = None) -> xr.Dataset:
    """
    Build a dynamic (hourly) predictor stack that couples large-scale forcings to terrain.
    All inputs should be aligned to the same (time, y, x) grid.

    Args:
        ds_static: Dataset with static fields (pisr_1..12 for solar, aspect, windexp, etc.)
        time: time coordinate (1D or 3D aligned with other inputs)
        swdown: hourly downward shortwave radiation (W m-2), optional (used with solar factor)
        wind_speed: hourly 10-m wind speed (m s-1)
        wind_dir_deg_met: hourly 10-m wind direction (meteorological degrees FROM)
        aspect_for_wind: static aspect to project wind along slopes (e.g., aspect_degrees_w50)
        windexp: static wind exposure index (e.g., windexp500)

    Returns:
        xr.Dataset of hourly features with dims matching the inputs.
    """
    features = xr.Dataset()

    # Solar factor for scaling SWdown
    try:
        solar_fac = effective_solar_factor_from_pisr(ds_static, time)
        features["solar_factor"] = solar_fac
        if swdown is not None:
            features["swdown_effective"] = swdown * solar_fac
            features["swdown_effective"].attrs["long_name"] = "Downwelling SW scaled by terrain solar factor"
    except Exception as e:
        print(f"[warn] Solar factor not available: {e}")

    # Wind-related
    try:
        if (wind_speed is not None) and (windexp is not None):
            features["wind_speed_effective"] = effective_wind_speed(wind_speed, windexp)
    except Exception as e:
        print(f"[warn] Effective wind speed not available: {e}")

    try:
        if (wind_speed is not None) and (wind_dir_deg_met is not None) and (aspect_for_wind is not None):
            wc = wind_components_along_slope(wind_speed, wind_dir_deg_met, aspect_for_wind)
            features = xr.merge([features, wc])
    except Exception as e:
        print(f"[warn] Wind components along slope not available: {e}")

    return features


# Example usage is the same as before.
