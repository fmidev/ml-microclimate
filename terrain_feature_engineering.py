
"""
terrain_feature_engineering.py
--------------------------------
Utilities to derive static and dynamic geospatial predictors to model hourly
near-surface air temperature in complex terrain.

New in this version:
- Blackman-smoothed morphometry (slope/aspect/aspect_sin/cos, slope_x/y, TPI).
- CAPI no longer uses MBI; it uses norm_height, SVF, and tpi_* if present.
- Added moisture_cooling_potential() using mpi2000 or SWI suction layers.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

try:
    from scipy.ndimage import uniform_filter, gaussian_filter
    from scipy import signal
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -------------------------------
# Helpers
# -------------------------------

def _zscore(da: xr.DataArray) -> xr.DataArray:
    mean = da.mean(dim=("y", "x"), skipna=True)
    std = da.std(dim=("y", "x"), skipna=True)
    return (da - mean) / (std + 1e-12)


def _norm01(da: xr.DataArray) -> xr.DataArray:
    dmin = da.min(dim=("y","x"), skipna=True)
    dmax = da.max(dim=("y","x"), skipna=True)
    return (da - dmin) / ((dmax - dmin) + 1e-12)


def _rolling_mean(da: xr.DataArray, win: int) -> xr.DataArray:
    if _HAS_SCIPY:
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

def _rolling_min(da: xr.DataArray, win: int) -> xr.DataArray:
    """
    NaN-aware sliding minimum using SciPy if available, else xarray rolling.
    """
    arr = da.values.astype(float)
    mask = np.isfinite(arr).astype(float)

    try:
        from scipy.ndimage import minimum_filter
    except ImportError:
        # fallback to xarray rolling (slow but safe)
        return da.rolling(y=win, x=win, center=True).min()

    # replace NaN with +inf so min() ignores gaps
    arr_filled = np.where(np.isfinite(arr), arr, np.inf)
    res = minimum_filter(arr_filled, size=(win, win), mode="nearest")
    res = np.where(mask > 0, res, np.nan)
    return xr.DataArray(res, coords=da.coords, dims=da.dims)


def _rolling_max(da: xr.DataArray, win: int) -> xr.DataArray:
    """
    NaN-aware sliding maximum using SciPy if available, else xarray rolling.
    """
    arr = da.values.astype(float)
    mask = np.isfinite(arr).astype(float)

    try:
        from scipy.ndimage import maximum_filter
    except ImportError:
        return da.rolling(y=win, x=win, center=True).max()

    # replace NaN with -inf so max() ignores gaps
    arr_filled = np.where(np.isfinite(arr), arr, -np.inf)
    res = maximum_filter(arr_filled, size=(win, win), mode="nearest")
    res = np.where(mask > 0, res, np.nan)
    return xr.DataArray(res, coords=da.coords, dims=da.dims)

def _deg2rad(da: xr.DataArray | float) -> xr.DataArray | float:
    return np.deg2rad(da)


def _wrap_dir_deg(angle):
    return (angle % 360 + 360) % 360


def _spacing_from_coords(coord: xr.DataArray) -> float:
    values = coord.values
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Coordinate must be 1D with at least 2 elements to compute spacing")
    diffs = np.diff(values)
    return float(np.nanmean(diffs))


# -------------------------------
# Static terrain derivatives
# -------------------------------


def _blackman_kernel(wy: int, wx: int):
    """Create a separable 2D Blackman kernel normalized to sum=1."""
    ky = np.blackman(max(int(wy), 1))
    kx = np.blackman(max(int(wx), 1))
    ker = np.outer(ky, kx)
    s = np.nansum(ker)
    if not np.isfinite(s) or s == 0:
        s = 1.0
    return ker / s


def slope_components_blackman(
    dem: xr.DataArray,
    window_size: tuple[int, int] = (11, 11),
    xcoord: xr.DataArray | None = None,
    ycoord: xr.DataArray | None = None,
    degrees: bool = True,
) -> xr.Dataset:
    """
    Compute smoothed slope components (x/y) using a Blackman kernel.
    Returns radians by default or degrees if degrees=True.
    """
    if dem.dims != ("y","x"):
        raise ValueError("DEM must have dims ('y','x')")
    wy, wx = int(window_size[0]), int(window_size[1])
    if xcoord is None: xcoord = dem.coords.get("x", None)
    if ycoord is None: ycoord = dem.coords.get("y", None)
    if (xcoord is None) or (ycoord is None):
        raise ValueError("Need 1D x/y coordinates in meters")

    dx = _spacing_from_coords(xcoord)
    dy = _spacing_from_coords(ycoord)

    arr = dem.values.astype(float)
    nmask = np.isfinite(arr).astype(float)

    # raw gradients (dz/dx, dz/dy)
    gz_y, gz_x = np.gradient(arr, dy, dx)
    # convert to slope angles (radians)
    slope_x = np.arctan2(gz_x, 1.0)
    slope_y = np.arctan2(gz_y, 1.0)

    try:
        from scipy import signal as _sig
        ker = _blackman_kernel(wy, wx)
        def conv(field):
            num = _sig.convolve2d(np.nan_to_num(field), ker, mode="same", boundary="symm")
            den = _sig.convolve2d(nmask, ker, mode="same", boundary="symm")
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where(den > 0, num/den, np.nan)
            return out
        slope_x = conv(slope_x)
        slope_y = conv(slope_y)
    except Exception:
        # fallback: no smoothing if scipy not available
        pass

    if degrees:
        slope_x = np.degrees(slope_x)
        slope_y = np.degrees(slope_y)

    def maskit(a):
        return np.where(np.isfinite(arr), a, np.nan)

    return xr.Dataset(
        {
            "slope_x": (("y","x"), maskit(slope_x)),
            "slope_y": (("y","x"), maskit(slope_y)),
        },
        coords=dem.coords
    )


def slope_magnitude_from_components(slope_x, slope_y, degrees=True) -> xr.DataArray:
    """
    Non-directional slope magnitude from components (in radians if degrees=False).
    If inputs are degrees, we'll internally convert to radians first.
    """
    sx = xr.DataArray(slope_x)
    sy = xr.DataArray(slope_y)
    if degrees:
        sxr = np.radians(sx)
        syr = np.radians(sy)
    else:
        sxr, syr = sx, sy
    slope = np.degrees(np.arctan(np.sqrt(sxr**2 + syr**2))) if degrees else np.arctan(np.sqrt(sxr**2 + syr**2))
    slope.name = "slope"
    return slope


def aspect_from_components(slope_x, slope_y, degrees=True) -> xr.DataArray:
    """
    Aspect computed as arctan2(slope_x, slope_y) following the user's formulation.
    Returns degrees in [0, 360) if degrees=True, else radians in [0, 2π).
    """
    sx = xr.DataArray(slope_x)
    sy = xr.DataArray(slope_y)
    ang = np.arctan2(sx, sy)
    if degrees:
        asp = (np.degrees(ang) + 360.0) % 360.0
    else:
        asp = (ang + 2*np.pi) % (2*np.pi)
    asp = xr.DataArray(asp, coords=sx.coords, dims=sx.dims, name="aspect")
    return asp


def aspect_trig(aspect_deg: xr.DataArray) -> xr.Dataset:
    """Return sine and cosine of aspect (degrees input)."""
    a = xr.DataArray(aspect_deg)
    s = np.sin(np.deg2rad(a))
    c = np.cos(np.deg2rad(a))
    return xr.Dataset({"aspect_sin": s, "aspect_cos": c})


def smooth_dem_gaussian(dem: xr.DataArray, radius_px: int = 50, sigma_frac: float = 1/3) -> xr.DataArray:
    """
    NaN-aware Gaussian smoothing of DEM with a radius specified in pixels.
    Uses sigma = radius_px * sigma_frac (default puts ~99.7% mass within ~3*sigma ≈ radius).
    """
    arr = dem.values.astype(float)
    mask = np.isfinite(arr).astype(float)

    try:
        from scipy.ndimage import gaussian_filter as _gf
    except Exception:
        # Fallback: simple rolling mean with window ~ 2*radius+1 (boxy but works)
        win = max(2*radius_px + 1, 3)
        mean_elev = dem.rolling(y=win, x=win, center=True).mean().values
        return xr.DataArray(mean_elev, coords=dem.coords, dims=dem.dims, name=f"{dem.name}_smooth_w{radius_px}")

    sigma = max(radius_px * sigma_frac, 1.0)
    num = _gf(np.nan_to_num(arr), sigma=(sigma, sigma), mode="nearest")
    den = _gf(mask,               sigma=(sigma, sigma), mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_elev = np.where(den > 0, num / den, np.nan)

    return xr.DataArray(mean_elev, coords=dem.coords, dims=dem.dims, name=f"{dem.name}_smooth_w{radius_px}")


def tpi_gaussian(
    dem: xr.DataArray,
    window_size: tuple[int, int] = (50, 50),
    sigma_frac: float = 1/3,
) -> xr.DataArray:
    """
    Topographic Position Index using Gaussian-smoothed mean elevation.
    sigma is set to window_size*sigma_frac along each axis (in pixels).
    """
    wy, wx = int(window_size[0]), int(window_size[1])
    arr = dem.values.astype(float)
    mask = np.isfinite(arr).astype(float)

    try:
        from scipy.ndimage import gaussian_filter as _gf
        sy = max(wy*sigma_frac, 1.0)
        sx = max(wx*sigma_frac, 1.0)
        num = _gf(np.nan_to_num(arr), sigma=(sy, sx), mode="nearest")
        den = _gf(mask, sigma=(sy, sx), mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_elev = np.where(den > 0, num/den, np.nan)
    except Exception:
        # fallback to rolling mean using the larger of the two as window
        mean_elev = _rolling_mean(dem, max(wy, wx)).values

    tpi = arr - mean_elev
    da = xr.DataArray(tpi, coords=dem.coords, dims=dem.dims, name="tpi")
    return da


def curvature_simple(dem: xr.DataArray, xcoord=None, ycoord=None) -> xr.Dataset:
    """
    Simple 2nd-derivative based curvatures: profile and planform (local).
    Useful predictors for drainage and solar concentration effects.
    """
    if xcoord is None: xcoord = dem.coords.get("x", None)
    if ycoord is None: ycoord = dem.coords.get("y", None)
    if (xcoord is None) or (ycoord is None):
        raise ValueError("Need 1D x/y coordinates in meters")
    dx = _spacing_from_coords(xcoord); dy = _spacing_from_coords(ycoord)
    z = dem.values.astype(float)

    # First derivatives
    zy, zx = np.gradient(z, dy, dx)
    # Second derivatives
    zyy, zyx = np.gradient(zy, dy, dx)
    zxy, zxx = np.gradient(zx, dy, dx)

    # Profile curvature ~ curvature in direction of slope (approx using second derivative along y)
    prof = zyy
    # Planform curvature ~ curvature normal to slope (approx using second derivative along x)
    plan = zxx

    return xr.Dataset({
        "curv_profile": (dem.dims, prof),
        "curv_plan": (dem.dims, plan),
    }, coords=dem.coords)


def heat_load_index(aspect_deg: xr.DataArray, slope_deg: xr.DataArray, latitude_deg: float) -> xr.DataArray:
    """
    McCune (2007) Heat Load Index (semi-empirical). Requires site latitude (deg).
    Produces a relative measure of potential heat load (0..1 approx).
    """
    a = np.deg2rad(aspect_deg)
    s = np.deg2rad(slope_deg)
    lat = np.deg2rad(latitude_deg)
    hli = 0.5 * (1 + np.cos(lat - s) * np.cos(a - np.pi))
    return xr.DataArray(hli, coords=slope_deg.coords, dims=slope_deg.dims, name="heat_load_index")




def terrain_ruggedness_index(dem: xr.DataArray, win: int = 5) -> xr.DataArray:
    local_mean = _rolling_mean(dem, win)
    tri = np.abs(dem - local_mean)
    tri = _rolling_mean(tri, win)
    tri.name = f"tri_w{win}"
    tri.attrs["long_name"] = f"Terrain Ruggedness Index (win={win})"
    return tri


def vector_ruggedness_measure(slope_deg: xr.DataArray, aspect_deg: xr.DataArray, win: int = 5) -> xr.DataArray:
    s = _deg2rad(slope_deg)
    a = _deg2rad(aspect_deg)
    sx = np.sin(s) * np.cos(a)
    sy = np.sin(s) * np.sin(a)
    sz = np.cos(s)

    mx = _rolling_mean(xr.DataArray(sx, coords=slope_deg.coords, dims=slope_deg.dims), win)
    my = _rolling_mean(xr.DataArray(sy, coords=slope_deg.coords, dims=slope_deg.dims), win)
    mz = _rolling_mean(xr.DataArray(sz, coords=slope_deg.coords, dims=slope_deg.dims), win)

    R = np.sqrt(mx**2 + my**2 + mz**2)
    vrm = 1 - R
    vrm.name = f"vrm_w{win}"
    vrm.attrs["long_name"] = f"Vector Ruggedness Measure (win={win})"
    return vrm


def multiscale_contrasts(ds: xr.Dataset) -> xr.Dataset:
    out = xr.Dataset()

    def add_if_present(name_small, name_large, out_name):
        if (name_small in ds) and (name_large in ds):
            out[out_name] = ds[name_small] - ds[name_large]

    add_if_present("slope_degrees_w3", "slope_degrees_w50", "slope_diff_w3_w50")

    if ("aspect_cos_w3" in ds) and ("aspect_cos_w50" in ds) and ("aspect_sin_w3" in ds) and ("aspect_sin_w50" in ds):
        v3 = xr.apply_ufunc(np.hypot, ds["aspect_cos_w3"] - ds["aspect_cos_w50"],
                            ds["aspect_sin_w3"] - ds["aspect_sin_w50"])
        v3.name = "aspect_vecdiff_w3_w50"
        out[v3.name] = v3

    add_if_present("tpi_w3", "tpi_w50", "tpi_diff_w3_w50")
    return out

'''
def cold_air_pooling_index(ds: xr.Dataset) -> xr.DataArray:
    """
    Heuristic Cold-Air Pooling Index (CAPI) combining:
      - low normalized height (norm_height)
      - negative topographic position index (valley tendency) if available (tpi_*)
      - low SVF (enclosure)
    Returns a 0..1 index (higher = stronger CAP potential).
    NOTE: MBI = Mean Biomass Index and is NOT used here.
    """
    parts = []

    if "norm_height" in ds:
        parts.append(_zscore(-ds["norm_height"]).rename("capi_norm_height"))
    tpi_key = None
    for k in ["tpi_w50", "tpi_w3", "top_posn_idx"]:
        if k in ds:
            tpi_key = k
            break
    if tpi_key is not None:
        parts.append(_zscore(-ds[tpi_key]).rename("capi_tpi"))
    if "svf" in ds:
        parts.append(_zscore(1 - _norm01(ds["svf"])).rename("capi_svf"))

    if not parts:
        raise ValueError("Need at least one of ['norm_height','tpi_*','svf'] to compute CAPI.")

    capi = sum(parts) / len(parts)
    capi = _norm01(capi)
    capi.name = "capi"
    capi.attrs["long_name"] = "Cold-Air Pooling Index (0..1)"
    return capi
'''

def cold_air_pooling_index_scale(
    dem: xr.DataArray,
    svf: xr.DataArray | None = None,
    win: int = 50,
    sigma_frac: float = 1/3,
) -> xr.DataArray:
    """
    Scale-aware Cold-Air Pooling Index (CAPI), computed for a chosen window size.

    Parameters
    ----------
    dem : xr.DataArray
        Digital elevation model (y, x) in meters.
    svf : xr.DataArray, optional
        Sky View Factor (0..1). If None, SVF term is omitted.
    win : int
        Window size in pixels (e.g., 3, 25, 50).
    sigma_frac : float
        Fraction to convert window size -> Gaussian sigma for TPI smoothing.

    Returns
    -------
    capi : xr.DataArray
        Cold-Air Pooling Index (0..1), name = f"capi_w{win}"
    """
    # --- compute TPI at this scale ---
    tpi = tpi_gaussian(dem, window_size=(win, win), sigma_frac=sigma_frac)

    # --- compute normalized height (relative elevation within window) ---
    #    norm_height = (DEM - local_min) / (local_max - local_min)
    local_min = _rolling_min(dem, win)
    local_max = _rolling_max(dem, win)
    nh = (dem - local_min) / (local_max - local_min)
    nh = nh.clip(0, 1)

    parts = []

    # Low normalized height → strong CAP potential → use negative sign
    parts.append(_zscore(-nh).rename("capi_norm_height"))

    # Negative TPI → valley → CAP potential
    parts.append(_zscore(-tpi).rename("capi_tpi"))

    # SVF optional: low SVF = enclosed valley → CAP potential
    if svf is not None:
        parts.append(_zscore(1 - _norm01(svf)).rename("capi_svf"))

    # Combine equally
    capi = sum(parts) / len(parts)
    capi = _norm01(capi)
    capi.name = f"capi_w{win}"
    capi.attrs["long_name"] = f"Cold-Air Pooling Index (win={win}, 0..1)"
    return capi




def openness_proxy(ds: xr.Dataset, win: int = 25) -> xr.DataArray:
    if "dem10m" not in ds:
        raise ValueError("dem10m is required for openness_proxy")

    relief = np.abs(ds["dem10m"] - _rolling_mean(ds["dem10m"], win))
    relief = _norm01(relief)

    svf_norm = _norm01(ds["svf"]) if "svf" in ds else 0.5
    openness = 0.5 * svf_norm + 0.5 * relief
    openness.name = f"openness_proxy_w{win}"
    openness.attrs["long_name"] = f"Openness Proxy (win={win})"
    return openness

"""
def morphometry_blackman(
    dem: xr.DataArray,
    window_size: tuple[int, int] = (11, 11),
    xcoord: xr.DataArray | None = None,
    ycoord: xr.DataArray | None = None,
    out_ws_name: str | None = None,
) -> xr.Dataset:
    if dem.dims != ("y","x"):
        raise ValueError("DEM must have dims ('y','x')")

    wy, wx = int(window_size[0]), int(window_size[1])
    ws = out_ws_name or str(int(round((wy + wx)/2)))

    if xcoord is None:
        xcoord = dem.coords.get("x", None)
    if ycoord is None:
        ycoord = dem.coords.get("y", None)
    if (xcoord is None) or (ycoord is None):
        raise ValueError("x and y coordinates are required (as 1D arrays in meters)")

    dx = _spacing_from_coords(xcoord)
    dy = _spacing_from_coords(ycoord)

    arr = dem.values.astype(float)
    nmask = np.isfinite(arr).astype(float)

    gz_y, gz_x = np.gradient(arr, dy, dx)
    slope_x_rad = np.arctan2(gz_x, 1.0)
    slope_y_rad = np.arctan2(gz_y, 1.0)

    if _HAS_SCIPY:
        ky = np.blackman(wy); kx = np.blackman(wx)
        kernel = np.outer(ky, kx)
        kernel /= kernel.sum() if np.isfinite(kernel.sum()) and kernel.sum() != 0 else 1.0

        def conv2(field):
            num = signal.convolve2d(np.nan_to_num(field), kernel, mode="same", boundary="symm")
            den = signal.convolve2d(nmask, kernel, mode="same", boundary="symm")
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where(den > 0, num / den, np.nan)
            return out

        slope_x_rad = conv2(slope_x_rad)
        slope_y_rad = conv2(slope_y_rad)

        sigma_y = max(wy / 3.0, 1.0); sigma_x = max(wx / 3.0, 1.0)
        dem_smooth = gaussian_filter(np.nan_to_num(arr), sigma=(sigma_y, sigma_x), mode="nearest")
        valid_smooth = gaussian_filter(nmask, sigma=(sigma_y, sigma_x), mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            dem_smooth = np.where(valid_smooth > 0, dem_smooth / valid_smooth, np.nan)
        tpi = arr - dem_smooth
    else:
        dem_smooth = _rolling_mean(dem, max(wy, wx)).values
        tpi = arr - dem_smooth

    slope_deg = np.rad2deg(np.arctan(np.sqrt(slope_x_rad**2 + slope_y_rad**2)))
    aspect_deg = (np.degrees(np.arctan2(slope_x_rad, slope_y_rad)) + 360.0) % 360.0
    aspect_sin = np.sin(np.deg2rad(aspect_deg))
    aspect_cos = np.cos(np.deg2rad(aspect_deg))

    slope_x_deg = np.degrees(slope_x_rad)
    slope_y_deg = np.degrees(slope_y_rad)

    def with_mask(a):
        return np.where(np.isfinite(arr), a, np.nan)

    ds_out = xr.Dataset(
        {
            f"slope_degrees_w{ws}": (("y","x"), with_mask(slope_deg)),
            f"slope_x_degrees_w{ws}": (("y","x"), with_mask(slope_x_deg)),
            f"slope_y_degrees_w{ws}": (("y","x"), with_mask(slope_y_deg)),
            f"aspect_degrees_w{ws}": (("y","x"), with_mask(aspect_deg)),
            f"aspect_cos_w{ws}": (("y","x"), with_mask(aspect_cos)),
            f"aspect_sin_w{ws}": (("y","x"), with_mask(aspect_sin)),
            f"tpi_w{ws}": (("y","x"), with_mask(tpi)),
        },
        coords=dem.coords,
    )
    return ds_out
"""

# -------------------------------
# Dynamic couplings
# -------------------------------

def month_from_time(time: xr.DataArray) -> xr.DataArray:
    return xr.DataArray(time.dt.month, coords=time.coords, dims=time.dims)


def effective_solar_factor_from_pisr(ds: xr.Dataset, time: xr.DataArray) -> xr.DataArray:
    pisr_list = []
    for m in range(1, 13):
        key = f"pisr_{m}"
        if key in ds:
            pisr_list.append(ds[key].expand_dims({"month":[m]}))
    if not pisr_list:
        raise ValueError("No pisr_1..pisr_12 fields found in dataset.")
    pisr = xr.concat(pisr_list, dim="month")
    pisr_norm = pisr / (pisr.max(dim=("y","x"), skipna=True) + 1e-12)
    mo = month_from_time(time)
    fac = pisr_norm.sel(month=mo).rename("effective_solar_factor")
    fac.attrs["long_name"] = "Monthly terrain-modulated solar factor (0..1)"
    return fac


def wind_components_along_slope(wind_speed: xr.DataArray,
                                wind_dir_deg_met: xr.DataArray,
                                aspect_deg: xr.DataArray) -> xr.Dataset:
    flow_to = _wrap_dir_deg(wind_dir_deg_met + 180.0)
    upslope_dir = _wrap_dir_deg(aspect_deg + 180.0)
    theta = np.deg2rad(_wrap_dir_deg(flow_to - upslope_dir))
    along = wind_speed * np.cos(theta)
    cross = wind_speed * np.sin(theta)
    ds_out = xr.Dataset()
    ds_out["wind_along_slope"] = along
    ds_out["wind_cross_slope"] = cross
    return ds_out


def wind_sheltering_factor(windexp: xr.DataArray) -> xr.DataArray:
    finite = np.isfinite(windexp)
    pos_frac = (windexp.where(finite) > 0).mean().item()
    if pos_frac < 0.5:
        windexp = -windexp
    return _norm01(windexp).rename("wind_shelter_exposure")


def effective_wind_speed(wind_speed: xr.DataArray, windexp: xr.DataArray, k: float = 0.6) -> xr.DataArray:
    exposure = wind_sheltering_factor(windexp)
    ws_eff = wind_speed * ((1 - k) + k * exposure)
    ws_eff.name = "wind_speed_effective"
    return ws_eff


def moisture_cooling_potential_scale(
    dem: xr.DataArray,
    svf: xr.DataArray | None = None,
    swi: xr.DataArray | None = None,
    mpi: xr.DataArray | None = None,
    win: int = 50,
    sigma_frac: float = 1/3,
    w_terrain: float = 0.5,
    w_observed: float = 0.5,
) -> xr.DataArray:
    """
    Scale-aware moisture-driven evaporative cooling potential (0..1).

    - Terrain component (scale-aware): lower local normalized height & negative TPI,
      optionally low SVF => higher moisture potential.
    - Observed component (scale-aware): mpi/swi locally normalized within the same window.
      This makes the result responsive to `win`.

    If only one component is available, that component is returned.
    """

    # ---- Terrain-derived moisture tendency (scale-aware)
    # TPI at this scale
    tpi = tpi_gaussian(dem, window_size=(win, win), sigma_frac=sigma_frac)

    # Local normalized height within window
    nh_min = _rolling_min(dem, win)
    nh_max = _rolling_max(dem, win)
    nh = (dem - nh_min) / (nh_max - nh_min + 1e-12)
    nh = nh.clip(0, 1)

    parts_terrain = [_zscore(-nh), _zscore(-tpi)]
    if svf is not None:
        parts_terrain.append(_zscore(1 - _norm01(svf)))
    terrain_comp = _norm01(sum(parts_terrain) / len(parts_terrain))

    # ---- Observed moisture (scale-aware local normalization)
    observed_comp = None
    obs = mpi if mpi is not None else swi
    if obs is not None:
        # local min/max on observed field at same scale
        o_min = _rolling_min(obs, win)
        o_max = _rolling_max(obs, win)
        o_loc = (obs - o_min) / (o_max - o_min + 1e-12)
        observed_comp = _norm01(o_loc)

    # ---- Combine
    if observed_comp is None:
        m = terrain_comp
    else:
        # Normalize weights if both present
        wt = float(w_terrain)
        wo = float(w_observed)
        if wt + wo <= 0:
            wt = wo = 0.5
        m = (wt * terrain_comp + wo * observed_comp) / (wt + wo)

    m = _norm01(m)
    m.name = f"mcp_w{win}"
    m.attrs["long_name"] = f"Moisture-driven evaporative cooling potential (win={win}, 0..1)"
    return m


def cold_bias_predictor_scale(
    dem: xr.DataArray,
    svf: xr.DataArray | None = None,
    win: int = 50,
    sigma_frac: float = 1/3,
) -> xr.DataArray:
    """
    Scale-aware cold bias potential index.
    Combines:
      - scale-specific Cold-Air Pooling Index (CAPI)
      - elevation (raw DEM)

    Parameters
    ----------
    dem : xr.DataArray
        Elevation (DEM) array (y, x) in meters.
    svf : xr.DataArray, optional
        Sky View Factor (0..1). If None, SVF term is omitted.
    win : int
        Window size in pixels for scale-aware CAPI.
    sigma_frac : float
        Fraction used for TPI smoothing inside CAPI.

    Returns
    -------
    cold_bias : xr.DataArray
        Cold-bias potential (0..1), name = f"cpb_w{win}"
    """
    capi = cold_air_pooling_index_scale(dem, svf=svf, win=win, sigma_frac=sigma_frac)
    cold_bias = _zscore(capi) + 0.5 * _zscore(dem)
    cold_bias = _norm01(cold_bias)
    cold_bias.name = f"cpb_w{win}"
    cold_bias.attrs["long_name"] = f"Cold bias potential (win={win}, 0..1)"
    return cold_bias


# -------------------------------
# Orchestrators
# -------------------------------
"""
def _ensure_morphometry_from_dem(ds: xr.Dataset) -> xr.Dataset:
    if "dem10m" not in ds:
        return xr.Dataset()
    need_small = any(k not in ds for k in ["slope_degrees_w3","aspect_degrees_w3","tpi_w3"])
    need_large = any(k not in ds for k in ["slope_degrees_w50","aspect_degrees_w50","tpi_w50"])
    out = xr.Dataset()
    if need_small:
        out = xr.merge([out, morphometry_blackman(ds["dem10m"], window_size=(7,7), out_ws_name="3")])
    if need_large:
        out = xr.merge([out, morphometry_blackman(ds["dem10m"], window_size=(51,51), out_ws_name="50")])
    return out
"""

def build_static_predictors(ds: xr.Dataset, latitude_deg: float | None = None) -> xr.Dataset:
    """
    Build static terrain predictors at two pixel scales: 3 px (w3) and 50 px (w50).
    Requires ds['dem10m']. Optionally uses ds['svf'], ds['mpi2000'], ds['swi_suction256'], ds['swi_suction16'].
    If latitude_deg is provided, computes heat_load_index for both scales.
    """
    if "dem10m" not in ds:
        raise ValueError("ds must contain 'dem10m'")
    
    dem = ds["dem10m"]
    svf = ds.get("svf", None)
    mpi = ds.get("mpi2000", None)
    
    swi = ds.get("swi_suction256", None)
    if swi is None:
        swi = ds.get("swi_suction16", None)
    
    out = xr.Dataset()
    out["dem10m"] = dem  # base elevation for reference
    
    # Two spatial scales (in pixels)
    #scales = [("w3", 3), ("w50", 50)]
    scales = [("w3", 3), ("w15", 15)]
    
    for tag, n in scales:
        # --- DEM smoothing & residuals (raw - smooth) ---
        dem_s = smooth_dem_gaussian(dem, radius_px=n, sigma_frac=1/3)
        #out[f"dem10m_smooth_{tag}"]   = dem_s
        
        # --- TRI (surface roughness) ---
        out[f"tri_{tag}"] = terrain_ruggedness_index(dem, win=n)
        
        # --- Slope & aspect (Blackman-smoothed gradients) ---
        comps   = slope_components_blackman(dem, window_size=(n, n))
        slope   = slope_magnitude_from_components(comps["slope_x"], comps["slope_y"])
        aspect  = aspect_from_components(comps["slope_x"], comps["slope_y"])
        atrig   = aspect_trig(aspect)
        
        #out[f"slope_degrees_{tag}"]   = slope
        out[f"slope_x_degrees_{tag}"] = comps["slope_x"]
        out[f"slope_y_degrees_{tag}"] = comps["slope_y"]
        #out[f"aspect_degrees_{tag}"]  = aspect
        #out[f"aspect_sin_{tag}"]      = atrig["aspect_sin"]
        #out[f"aspect_cos_{tag}"]      = atrig["aspect_cos"]
        
        # --- TPI (Gaussian, NaN-aware) ---
        out[f"tpi_{tag}"] = tpi_gaussian(dem, window_size=(n, n))
        
        # --- VRM (needs slope & aspect in degrees) ---
        #out[f"vrm_{tag}"] = vector_ruggedness_measure(slope, aspect, win=n)
        
        # --- Scale-aware CAP-related indices ---
        out[f"capi_{tag}"] = cold_air_pooling_index_scale(dem, svf=svf, win=n)
        #out[f"cpb_{tag}"]  = cold_bias_predictor_scale(dem, svf=svf, win=n)
        
        # --- Scale-aware moisture cooling potential ---
        out[f"mcp_{tag}"] = moisture_cooling_potential_scale(
            dem, svf=svf, swi=swi, mpi=mpi, win=n
        )
        
        # --- Curvatures on smoothed DEM (more stable) ---
        curv = curvature_simple(dem_s)
        out[f"curv_profile_{tag}"] = curv["curv_profile"]
        out[f"curv_plan_{tag}"]    = curv["curv_plan"]
        
        # --- Openness proxy ---
        #out[f"openness_proxy_{tag}"] = openness_proxy(ds, win=n)
        
        # --- Heat Load Index (optional, needs latitude) ---
        #if latitude_deg is not None:
        #    out[f"heat_load_index_{tag}"] = heat_load_index(aspect, slope, latitude_deg=latitude_deg)
    
    # --- Cross-scale contrasts (computed once) ---
    #out["slope_diff_w3_w50"] = out["slope_degrees_w3"] - out["slope_degrees_w50"]
    #out["aspect_vecdiff_w3_w50"] = xr.apply_ufunc(
    #    np.hypot,
    #    out["aspect_cos_w3"] - out["aspect_cos_w50"],
    #    out["aspect_sin_w3"] - out["aspect_sin_w50"],
    #)
    #out["tpi_diff_w3_w50"] = out["tpi_w3"] - out["tpi_w50"]
    
    return out



"""
def build_hourly_predictors(ds_static: xr.Dataset,
                            time: xr.DataArray,
                            swdown: xr.DataArray | None = None,
                            wind_speed: xr.DataArray | None = None,
                            wind_dir_deg_met: xr.DataArray | None = None,
                            aspect_for_wind: xr.DataArray | None = None,
                            windexp: xr.DataArray | None = None) -> xr.Dataset:
    features = xr.Dataset()
    try:
        solar_fac = effective_solar_factor_from_pisr(ds_static, time)
        features["solar_factor"] = solar_fac
        if swdown is not None:
            features["swdown_effective"] = swdown * solar_fac
    except Exception as e:
        print(f"[warn] Solar factor not available: {e}")

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

"""
