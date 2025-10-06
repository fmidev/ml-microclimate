# arctic_temp_skill.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict
import numpy as np
import pandas as pd


ArrayLike = Union[pd.Series, np.ndarray, List[float]]


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d not in (0, 0.0, np.nan) and not np.isnan(d) else np.nan


def _align_dropna(y: ArrayLike, yhat: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    y = pd.Series(y).astype(float)
    yhat = pd.Series(yhat).astype(float)
    m = ~(y.isna() | yhat.isna())
    return y[m].to_numpy(), yhat[m].to_numpy()


def regression_metrics(y: ArrayLike, yhat: ArrayLike) -> Dict[str, float]:
    """
    Core deterministic regression skill: RMSE, MAE, bias, Pearson r, R^2.
    """
    yt, yp = _align_dropna(y, yhat)
    if yt.size == 0:
        return dict(RMSE=np.nan, MAE=np.nan, Bias=np.nan, R=np.nan, R2=np.nan)
    err = yp - yt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    # Pearson correlation (guard for constant series)
    if np.std(yt) == 0 or np.std(yp) == 0:
        r = np.nan
    else:
        r = float(np.corrcoef(yt, yp)[0, 1])
    r2 = float(r ** 2) if pd.notna(r) else np.nan
    return dict(RMSE=rmse, MAE=mae, Bias=bias, R=r, R2=r2)


def event_metrics(y: ArrayLike, yhat: ArrayLike, threshold: float = 0.0) -> Dict[str, float]:
    """
    Deterministic frost-event skill (<= threshold): Precision, Recall, F1, Accuracy, TNR.
    """
    yt, yp = _align_dropna(y, yhat)
    if yt.size == 0:
        return dict(Precision=np.nan, Recall=np.nan, F1=np.nan, Accuracy=np.nan, TNR=np.nan)
    obs = yt <= threshold
    pred = yp <= threshold
    tp = int(np.sum(pred & obs))
    tn = int(np.sum(~pred & ~obs))
    fp = int(np.sum(pred & ~obs))
    fn = int(np.sum(~pred & obs))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if all(pd.notna([precision, recall])) else np.nan
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    tnr = _safe_div(tn, tn + fp)
    return dict(Precision=precision, Recall=recall, F1=f1, Accuracy=acc, TNR=tnr)


def _freeze_thaw_cycles(series: pd.Series, threshold: float = 0.0) -> int:
    """Count sign-crossings across threshold."""
    s = series.dropna().astype(float).to_numpy()
    if s.size < 2:
        return np.nan
    below = s <= threshold
    # transitions where state changes between consecutive samples
    return int(np.sum(below[1:] != below[:-1]))


def _degree_hours(series: pd.Series, base: float = 0.0) -> float:
    """Sum of positive (T-base) across hours (growing degree hours)."""
    s = series.dropna().astype(float).to_numpy()
    return float(np.sum(np.clip(s - base, a_min=0, a_max=None)))

def _episode_lengths_within(temp: pd.Series, low: float, high: float) -> List[int]:
    """Return contiguous episode lengths (hours) with low <= T <= high."""
    mask = (temp >= low) & (temp <= high)
    if mask.empty:
        return []
    # group by runs
    episodes = []
    run = 0
    for val in mask.values:
        if val:
            run += 1
        elif run > 0:
            episodes.append(run)
            run = 0
    if run > 0:
        episodes.append(run)
    return episodes



def _frost_hours(series: pd.Series, threshold: float = 0.0) -> int:
    s = series.dropna().astype(float).to_numpy()
    return int(np.sum(s <= threshold))

def _diurnal_amp(s: pd.Series) -> pd.Series:
    """
    Return a daily Series of diurnal amplitudes (max-min per day).
    This is what the module expects (it later compares obs vs pred per day).
    """
    daily = s.resample("D")
    return daily.max() - daily.min()

def _diurnal_amp_mean(s: pd.Series) -> float:
    """
    Optional helper if you want one scalar: the mean daily amplitude.
    """
    a = _diurnal_amp(s)
    return float(a.mean())

def _summer_warmth_index(s: pd.Series) -> float:
    """
    Sum of monthly mean temperatures where T_month >= 0°C
    over the time span of `s`. Works across years.
    """
    # Monthly means at month starts (timezone preserved)
    mmean = s.resample("MS").mean()
    swi = mmean.clip(lower=0).sum()
    return float(swi)

def _winter_coldness_index(s: pd.Series) -> float:
    """
    Sum of monthly mean temperatures where T_month <= 0°C
    """
    mmean = s.resample("MS").mean()
    wci = mmean.clip(upper=0).sum()
    return float(wci)*(-1.)

def _mean_temp(s: pd.Series) -> float:
    return float(s.mean())
    
def _ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df[time_col] = pd.to_datetime(df[time_col])
    return df


@dataclass
class ArcticTempSkill:
    """
    Skill analysis for modeled hourly 15 cm air temperatures in northern Fennoscandia.

    Expected columns in df:
      - 'time', 'region', 'site', 'lat', 'lon', 'x', 'y'
      - observed: obs_col (default 'T3')
      - models: list of model columns (e.g., ['xgboost_T3', 'lasso_T3', 'T3_microclimf'])
    """
    df: pd.DataFrame
    obs_col: str = "T3"
    model_cols: Optional[Sequence[str]] = None
    time_col: str = "time"

    def __post_init__(self):
        # Required columns
        req = {self.time_col, "region", "site", "lat", "lon"}
        missing = [c for c in req if c not in self.df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        if self.obs_col not in self.df.columns:
            raise ValueError(f"Observed column '{self.obs_col}' not found.")

        # Default models if not provided
        if not self.model_cols:
            candidates = ["xgboost_T3", "lasso_T3", "T3_microclimf"]
            self.model_cols = [c for c in candidates if c in self.df.columns]
        if not self.model_cols:
            raise ValueError("No model columns provided or found.")

        # Ensure datetime and tz-aware
        self.df = _ensure_datetime(self.df, self.time_col).copy()
        if pd.Series(self.df[self.time_col]).dt.tz is None:
            self.df[self.time_col] = pd.to_datetime(self.df[self.time_col]).dt.tz_localize("UTC")

        # Canonical: DatetimeIndex
        self.df = self.df.set_index(self.time_col).sort_index()

        # Keep a time column as well (mirror of index) for any code that expects a column
        self.df[self.time_col] = self.df.index

        # Add derived time columns so groupby="month"/"year"/etc works everywhere
        self._add_time_columns()



    def _add_time_columns(self, hydro_start_month: int = 9) -> None:
        """
        Materialize common time fields from the DatetimeIndex.

        Contract:
          - The DataFrame's index IS the canonical time axis and must be a tz-aware DatetimeIndex.
          - We also keep a mirror time column `self.time_col` for any code or grouping that expects a column.
        """
        idx = self.df.index

        # Safety checks
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a pandas.DatetimeIndex before calling _add_time_columns().")
        if idx.tz is None:
            # Localize (do not convert) to UTC if tz is missing
            self.df.index = self.df.index.tz_localize("UTC")
            idx = self.df.index

        if not (1 <= int(hydro_start_month) <= 12):
            raise ValueError(f"hydro_start_month must be in 1..12, got {hydro_start_month}.")

        # Mirror the index back to the original time column name (kept in sync)
        # Use a new DatetimeIndex to avoid accidental view issues; preserves tz.
        self.df[self.time_col] = pd.DatetimeIndex(idx)

        # Vectorized calendar fields (assigning ndarrays is positionally aligned—fast & safe)
        self.df["year"]  = idx.year.astype("int32")
        self.df["month"] = idx.month.astype("int16")
        self.df["doy"]   = idx.dayofyear.astype("int16")
        self.df["hour"]  = idx.hour.astype("int8")

        # A date-only column (Python date objects); useful for daily groupbys/joins
        # normalize() -> midnight same tz, then .date gives naive date objects.
        self.df["date"]  = idx.normalize().date

        # Hydrological year label: e.g., with hydro_start_month=9, Sep–Dec are labeled next calendar year.
        self.df["hydro_year"] = (idx.year + (idx.month >= hydro_start_month).astype("int32")).astype("int32")



    # ---------- 1) Core regression skill ----------
    def summarize_regression(
        self,
        groupby: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Compute RMSE, MAE, Bias, R, R2 for each model; optionally by group(s) (e.g., 'region', ['region','site']).
        """
        if groupby is None:
            groupby = []
        if isinstance(groupby, str):
            groupby = [groupby]

        rows = []
        if groupby:
            for keys, g in self.df.groupby(groupby, sort=False, dropna=False):
                for m in self.model_cols:
                    mets = regression_metrics(g[self.obs_col], g[m])
                    rows.append(dict(model=m, **{k: keys[i] for i, k in enumerate(groupby)}, **mets))
        else:
            for m in self.model_cols:
                mets = regression_metrics(self.df[self.obs_col], self.df[m])
                rows.append(dict(model=m, **mets))
        return pd.DataFrame(rows)

    # ---------- 2) Frost-event (<= threshold) skill ----------
    def frost_event_skill(
        self,
        threshold: float = 0.0,
        groupby: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Classification-style skill for frost events (<= threshold): Precision, Recall, F1, Accuracy, TNR.
        """
        if groupby is None:
            groupby = []
        if isinstance(groupby, str):
            groupby = [groupby]

        rows = []
        if groupby:
            for keys, g in self.df.groupby(groupby, sort=False, dropna=False):
                for m in self.model_cols:
                    mets = event_metrics(g[self.obs_col], g[m], threshold=threshold)
                    rows.append(dict(model=m, **{k: keys[i] for i, k in enumerate(groupby)}, **mets))
        else:
            for m in self.model_cols:
                mets = event_metrics(self.df[self.obs_col], self.df[m], threshold=threshold)
                rows.append(dict(model=m, **mets))
        return pd.DataFrame(rows)

    # ---------- 3) Bioclimatic indicators ----------
    def bioclim_summary(
        self,
        months: Tuple[int, ...] = tuple(range(1,13)), #(5, 6, 7, 8, 9),  # growing season approx. May–Sep
        base_temp: float = 0.0,                     # GrowingDegreeHours base
        frost_thresh: float = 0.0,                  # frost threshold
        groupby: Sequence[str] = ("region", "site", "year"),
    ) -> pd.DataFrame:
        """
        Summarize high-value indicators per (region, site, year) within selected months:
          - FrostHours (<= frost_thresh)
          - GrowingDegreeHours (Growing Degree Hours, base=base_temp)
          - FreezeThawCycles (crossings across frost_thresh)
        Returns observed, each model, and model-minus-observed deltas.
        """
        df = self.df.copy()
        df["month"] = pd.to_datetime(df.index).month
        df["year"] = pd.to_datetime(df.index).year
        df = df[df["month"].isin(months)].copy()

        def summarize_block(block: pd.DataFrame) -> Dict[str, float]:
            out = {
                "obs_FrostHours": _frost_hours(block[self.obs_col], threshold=frost_thresh),
                "obs_GrowingDegreeHours": _degree_hours(block[self.obs_col], base=base_temp),
                "obs_FreezeThawCycles": _freeze_thaw_cycles(block[self.obs_col], threshold=frost_thresh),
                "obs_MeanTemp": _mean_temp(block[self.obs_col]),
                "obs_DiurnalAmplitude": _diurnal_amp_mean(block[self.obs_col]),
                "obs_SummerWarmthIndex": _summer_warmth_index(block[self.obs_col]),
                "obs_WinterColdnessIndex": _winter_coldness_index(block[self.obs_col]),
                "obs_NearZeroEpisodeLength": np.nanmean(_episode_lengths_within(block[self.obs_col], low=-2., high=+2.)), 
                "obs_NearZeroEpisodeCount": len(_episode_lengths_within(block[self.obs_col], low=-2., high=+2.)),
            }
            
            if out['obs_NearZeroEpisodeCount'] < 2:
                out['obs_NearZeroEpisodeCount'] = np.nan
            
            for m in self.model_cols:
                out[f"{m}_FrostHours"] = _frost_hours(block[m], threshold=frost_thresh)
                out[f"{m}_GrowingDegreeHours"] = _degree_hours(block[m], base=base_temp)
                out[f"{m}_FreezeThawCycles"] = _freeze_thaw_cycles(block[m], threshold=frost_thresh)
                out[f"{m}_MeanTemp"] = _mean_temp(block[m])
                out[f"{m}_DiurnalAmplitude"] = _diurnal_amp_mean(block[m])
                out[f"{m}_SummerWarmthIndex"] = _summer_warmth_index(block[m])
                out[f"{m}_WinterColdnessIndex"] = _winter_coldness_index(block[m])
                out[f"{m}_NearZeroEpisodeLength"] = np.nanmean(_episode_lengths_within(block[m], low=-2., high=+2.))
                out[f"{m}_NearZeroEpisodeCount"] = len(_episode_lengths_within(block[m], low=-2., high=+2.))
                if out[f"{m}_NearZeroEpisodeCount"] < 2:
                    out[f"{m}_NearZeroEpisodeCount"] = np.nan
                
                # biases
                out[f"{m}_FrostHours_bias"] = out[f"{m}_FrostHours"] - out["obs_FrostHours"]
                out[f"{m}_GrowingDegreeHours_bias"] = out[f"{m}_GrowingDegreeHours"] - out["obs_GrowingDegreeHours"]
                out[f"{m}_FreezeThawCycles_bias"] = out[f"{m}_FreezeThawCycles"] - out["obs_FreezeThawCycles"]
                out[f"{m}_MeanTemp_bias"] = out[f"{m}_MeanTemp"] - out["obs_MeanTemp"]
                out[f"{m}_DiurnalAmplitude_bias"] = out[f"{m}_DiurnalAmplitude"] - out["obs_DiurnalAmplitude"]
                out[f"{m}_SummerWarmthIndex_bias"] = out[f"{m}_SummerWarmthIndex"] - out["obs_SummerWarmthIndex"]
                out[f"{m}_WinterColdnessIndex_bias"] = out[f"{m}_WinterColdnessIndex"] - out["obs_WinterColdnessIndex"]
                out[f"{m}_NearZeroEpisodeLength_bias"] = out[f"{m}_NearZeroEpisodeLength"] - out["obs_NearZeroEpisodeLength"]
                out[f"{m}_NearZeroEpisodeCount_bias"] = out[f"{m}_NearZeroEpisodeCount"] - out["obs_NearZeroEpisodeCount"]
            return out

        if not groupby:
            res = summarize_block(df)
            return pd.DataFrame([res])
        grouped = df.groupby(list(groupby), sort=False, dropna=False)
        rows = []
        for keys, g in grouped:
            res = summarize_block(g)
            if not isinstance(keys, tuple):
                keys = (keys,)
            rows.append({k: keys[i] for i, k in enumerate(groupby)} | res)
        return pd.DataFrame(rows)

    # ---------- 4) Spatial skill at microclimate scales (semivariogram) ----------
    def semivariogram(
        self,
        model: str,
        max_range_m: int = 10_000,
        n_bins: int = 20,
        n_hours: int = 100,
        n_pairs_per_hour: int = 5_000,
        seed: int = 0,
        by_region: bool = False,
        prefer_xy: bool = True,
    ) -> pd.DataFrame:
        """
        Empirical semivariogram of residuals (0.5 * (e_i - e_j)^2) vs distance, built from random hourly snapshots.
        This isolates microclimate-scale spatial structure of model errors up to `max_range_m`.

        Parameters
        ----------
        model : str
            Model column to analyze (must exist).
        prefer_xy : bool
            Use projected meters in ('x','y') if available; else fall back to haversine from lat/lon.

        Returns
        -------
        DataFrame with columns: ['h_center_m','semivariance','count', 'region'(opt), 'model'].
        """
        if model not in self.model_cols:
            raise ValueError(f"Model '{model}' not in model_cols.")

        rng = np.random.default_rng(seed)

        df = self.df.dropna(subset=[self.obs_col, model]).copy()
        # coords in meters
        have_xy = prefer_xy and {"x", "y"}.issubset(df.columns)
        if not have_xy:
            # fallback to meters via haversine
            lat = np.deg2rad(df["lat"].to_numpy())
            lon = np.deg2rad(df["lon"].to_numpy())
            R = 6_371_000.0

            def hv_dist(ii, jj):
                dlat = lat[ii] - lat[jj]
                dlon = lon[ii] - lon[jj]
                a = np.sin(dlat / 2) ** 2 + np.cos(lat[ii]) * np.cos(lat[jj]) * np.sin(dlon / 2) ** 2
                return 2 * R * np.arcsin(np.sqrt(a))
        else:
            x = df["x"].to_numpy()
            y = df["y"].to_numpy()

            def xy_dist(ii, jj):
                dx = x[ii] - x[jj]
                dy = y[ii] - y[jj]
                return np.sqrt(dx * dx + dy * dy)

        dist_func = xy_dist if have_xy else hv_dist

        df["resid"] = df[model] - df[self.obs_col]
        # Random subset of hours
        hours = df[self.time_col].dt.floor("H")
        unique_hours = hours.dropna().unique()
        if unique_hours.size == 0:
            return pd.DataFrame(columns=["h_center_m", "semivariance", "count", "model"])
        sample_hours = rng.choice(unique_hours, size=min(n_hours, unique_hours.size), replace=False)

        bins = np.linspace(0.0, float(max_range_m), n_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])

        def accumulate(g: pd.DataFrame, label: Optional[str] = None) -> pd.DataFrame:
            counts = np.zeros(n_bins, dtype=int)
            sums = np.zeros(n_bins, dtype=float)

            # Build index per hour, sample pairs within hour
            g = g.set_index(g[self.time_col].dt.floor("H"))

            for hr in sample_hours:
                if hr not in g.index:
                    continue
                sub = g.loc[hr]
                if isinstance(sub, pd.DataFrame):
                    resid = sub["resid"].to_numpy()
                    N = resid.size
                    if N < 2:
                        continue
                    idx = np.arange(N)
                    # sample pairs (i, j) ~ uniform, ensure i != j
                    k = min(n_pairs_per_hour, N * (N - 1) // 2)  # cap by unique pairs
                    i = rng.integers(0, N, size=k)
                    j = rng.integers(0, N - 1, size=k)
                    j = np.where(j >= i, j + 1, j)  # avoid i == j
                    # if we exceeded unique pairs, replacement is fine; this keeps it fast
                    # map to original row positions inside this hour
                    # distances
                    if have_xy:
                        sub_x = sub["x"].to_numpy()
                        sub_y = sub["y"].to_numpy()
                        d = np.sqrt((sub_x[i] - sub_x[j]) ** 2 + (sub_y[i] - sub_y[j]) ** 2)
                    else:
                        sub_lat = np.deg2rad(sub["lat"].to_numpy())
                        sub_lon = np.deg2rad(sub["lon"].to_numpy())
                        dlat = sub_lat[i] - sub_lat[j]
                        dlon = sub_lon[i] - sub_lon[j]
                        a = np.sin(dlat / 2) ** 2 + np.cos(sub_lat[i]) * np.cos(sub_lat[j]) * np.sin(dlon / 2) ** 2
                        d = 2 * 6_371_000.0 * np.arcsin(np.sqrt(a))

                    mask = d <= max_range_m
                    if not np.any(mask):
                        continue
                    d = d[mask]
                    gamma = 0.5 * (resid[i[mask]] - resid[j[mask]]) ** 2
                    # bin
                    which = np.digitize(d, bins) - 1
                    ok = (which >= 0) & (which < n_bins)
                    which = which[ok]
                    gamma = gamma[ok]
                    # accumulate
                    for b, gval in zip(which, gamma):
                        counts[b] += 1
                        sums[b] += float(gval)

            out = pd.DataFrame({
                "h_center_m": centers,
                "semivariance": np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0),
                "count": counts,
            })
            out["model"] = model
            if label is not None:
                out["region"] = label
            return out

        if by_region:
            parts = []
            for region, g in df.groupby("region", sort=False, dropna=False):
                parts.append(accumulate(g, label=str(region)))
            return pd.concat(parts, ignore_index=True) if parts else accumulate(df)
        else:
            return accumulate(df)

    def field_variogram(
        self,
        field_col: str,
        max_range_m: int = 10_000,
        n_bins: int = 20,
        n_hours: int = 100,
        n_pairs_per_hour: int = 5_000,
        seed: int = 0,
        by_region: bool = False,
        prefer_xy: bool = True,
        co_availability_with: Optional[str] = None,  # e.g. 'xgboost_T3' to match hours/sites to a model
    ) -> pd.DataFrame:
        """
        Empirical semivariogram of a *field* (e.g., observations 'T3').

        If co_availability_with is given, the dataframe is filtered to rows where both
        field_col and that column are non-NaN, so its hours/pairs match the model’s.
        Returns columns: ['h_center_m','semivariance','count','field','region'(opt)].
        """
        cols = [field_col]
        if co_availability_with is not None:
            cols.append(co_availability_with)
        df = self.df.dropna(subset=cols).copy()

        rng = np.random.default_rng(seed)
        have_xy = prefer_xy and {"x","y"}.issubset(df.columns)
        if not have_xy:
            lat = np.deg2rad(df["lat"].to_numpy())
            lon = np.deg2rad(df["lon"].to_numpy())
            R = 6_371_000.0
            def hv_dist(ii, jj):
                dlat = lat[ii]-lat[jj]; dlon = lon[ii]-lon[jj]
                a = np.sin(dlat/2)**2 + np.cos(lat[ii])*np.cos(lat[jj])*np.sin(dlon/2)**2
                return 2*R*np.arcsin(np.sqrt(a))
        else:
            x = df["x"].to_numpy(); y = df["y"].to_numpy()
            def xy_dist(ii, jj):
                dx = x[ii]-x[jj]; dy = y[ii]-y[jj]
                return np.sqrt(dx*dx + dy*dy)
        dist_func = xy_dist if have_xy else hv_dist

        hours = df[self.time_col].dt.floor("H")
        uniq = hours.dropna().unique()
        if uniq.size == 0:
            return pd.DataFrame(columns=["h_center_m","semivariance","count","field"])
        sample_hours = rng.choice(uniq, size=min(n_hours, uniq.size), replace=False)

        bins = np.linspace(0.0, float(max_range_m), n_bins+1)
        centers = 0.5*(bins[:-1]+bins[1:])

        def accumulate(g: pd.DataFrame, label: Optional[str]=None) -> pd.DataFrame:
            counts = np.zeros(n_bins, dtype=int)
            sums = np.zeros(n_bins, dtype=float)
            g = g.set_index(g[self.time_col].dt.floor("H"))
            for hr in sample_hours:
                if hr not in g.index: continue
                sub = g.loc[hr]
                if not isinstance(sub, pd.DataFrame): continue
                vals = sub[field_col].to_numpy()
                N = vals.size
                if N < 2: continue
                i = rng.integers(0, N, size=min(n_pairs_per_hour, max(1, N*(N-1)//2)))
                j = rng.integers(0, N-1, size=i.size)
                j = np.where(j >= i, j+1, j)

                if have_xy:
                    sx = sub["x"].to_numpy(); sy = sub["y"].to_numpy()
                    d = np.sqrt((sx[i]-sx[j])**2 + (sy[i]-sy[j])**2)
                else:
                    slat = np.deg2rad(sub["lat"].to_numpy()); slon = np.deg2rad(sub["lon"].to_numpy())
                    dlat = slat[i]-slat[j]; dlon = slon[i]-slon[j]
                    a = np.sin(dlat/2)**2 + np.cos(slat[i])*np.cos(slat[j])*np.sin(dlon/2)**2
                    d = 2*6_371_000.0*np.arcsin(np.sqrt(a))

                m = d <= max_range_m
                if not np.any(m): continue
                d = d[m]
                gamma = 0.5*(vals[i[m]] - vals[j[m]])**2
                which = np.digitize(d, bins) - 1
                ok = (which >= 0) & (which < n_bins)
                which = which[ok]; gamma = gamma[ok]
                for b, gv in zip(which, gamma):
                    counts[b] += 1; sums[b] += float(gv)

            out = pd.DataFrame({
                "h_center_m": centers,
                "semivariance": np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts>0),
                "count": counts,
                "field": field_col,
            })
            if label is not None:
                out["region"] = label
            return out

        if by_region:
            parts = [accumulate(g, label=str(r)) for r, g in df.groupby("region", sort=False, dropna=False)]
            return pd.concat(parts, ignore_index=True) if parts else accumulate(df)
        else:
            return accumulate(df)


    # ---------- Convenience: tidy long-format comparison ----------
    def compare_models_regression(
        self, groupby: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """Tidy long-format table of regression metrics for easy plotting."""
        wide = self.summarize_regression(groupby=groupby)
        return wide.melt(id_vars=[c for c in wide.columns if c not in {"RMSE","MAE","Bias","R","R2"}],
                         value_vars=["RMSE","MAE","Bias","R","R2"],
                         var_name="metric", value_name="value")

