
"""
export GDAL_NUM_THREADS=ALL_CPUS
export OMP_NUM_THREADS=1
"""

import glob, sys
import numpy as np
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling 



# Read own module
sys.path.append(code_dir)
import functions as fcts
fcts=importlib.reload(fcts)



# Metadata
era5_vars, help_vars, lags, regions = fcts.get_metadata()



quantities = [
    'pisr_1','pisr_2','pisr_3','pisr_4','pisr_5','pisr_6','pisr_7','pisr_8',
    'pisr_9','pisr_10','pisr_11','pisr_12',
    'windexp500','svf','mbi','norm_height','mpi2000','mid_slope_position',
    'swi_suction16','swi_suction256','dem10m','diurnalaniheat',
]


YCHUNK = 2048
XCHUNK = 2048


OPEN_CHUNKS  = {"x": XCHUNK, "y": YCHUNK}   # tune if needed
WRITE_CHUNKS = (YCHUNK, XCHUNK)             # on-disk chunking
FORCE_FLOAT32 = False                       # set True if you want float32 output

for region in regions:
    print(f"\n{region}:")
    all_tiffs = sorted(glob.glob(f"/lustre/tmp/kamarain/resiclim-topo/{region}/*.tif"))
    if not all_tiffs:
        print("No TIFFs found.")
        continue

    # --- Find dem10m for the template grid (required) ---
    dem_candidates = [f for f in all_tiffs if f.split("/")[-1].split(".tif")[0] == "dem10m"]
    if not dem_candidates:
        raise RuntimeError(f"[{region}] dem10m not found; cannot enforce 10 m grid.")
    tmpl_path = dem_candidates[0]
    print("Template (10 m):", tmpl_path)

    # Open template lazily
    tmpl_da = rxr.open_rasterio(tmpl_path, chunks=OPEN_CHUNKS, masked=True).squeeze(drop=True)
    template = tmpl_da.to_dataset(name="__template__")
    #spatial_ref_ = tmpl_da.rio.write_crs(tmpl_da.rio.crs, inplace=False).rio.spatial_ref
    tmpl_da = rxr.open_rasterio(tmpl_path, chunks=OPEN_CHUNKS, masked=True).squeeze(drop=True)
    tmpl_crs = tmpl_da.rio.crs
    tmpl_transform = tmpl_da.rio.transform()
    
    # Only process requested quantities (but ALWAYS use the 10 m template above)
    tiff_files = [f for f in all_tiffs if f.split("/")[-1].split(".tif")[0] in quantities]
    print("Files to process:", len(tiff_files))

    vars_ds = []
    for tiff_file in tiff_files:
        name = tiff_file.split("/")[-1].split(".tif")[0]
        print("  ->", name)

        da = rxr.open_rasterio(tiff_file, chunks=OPEN_CHUNKS, masked=True).squeeze(drop=True)

        # Fast path if it already matches dem10m grid
        same_grid = (
            da.rio.crs == tmpl_da.rio.crs
            and da.sizes.get("x") == tmpl_da.sizes.get("x")
            and da.sizes.get("y") == tmpl_da.sizes.get("y")
            and np.isclose(da.x.data[[0, -1]], tmpl_da.x.data[[0, -1]]).all()
            and np.isclose(da.y.data[[0, -1]], tmpl_da.y.data[[0, -1]]).all()
        )
        if same_grid:
            aligned = da
        else:
            # GDAL-backed warp to the 10 m template grid
            #aligned = da.rio.reproject_match(tmpl_da, resampling="nearest")
            aligned = da.rio.reproject_match(
                tmpl_da,
                resampling=Resampling.nearest,   # or Resampling.bilinear for continuous fields
                num_threads=20
            )
        if FORCE_FLOAT32:
            aligned = aligned.astype("float32")

        aligned = aligned.chunk({"y": WRITE_CHUNKS[0], "x": WRITE_CHUNKS[1]})
        vars_ds.append(aligned.to_dataset(name=name))

    print("Merging…")
    
    final_ds = xr.merge(vars_ds, compat="no_conflicts").chunk({"y": WRITE_CHUNKS[0], "x": WRITE_CHUNKS[1]})
    
    # Write geo metadata onto the merged Dataset (this is the portable way now)
    final_ds = final_ds.rio.write_crs(tmpl_crs, inplace=False)
    final_ds = final_ds.rio.write_transform(tmpl_transform, inplace=False)
    # (optional but nice: also write the grid-mapping/coord system var)
    final_ds = final_ds.rio.write_coordinate_system(inplace=False)
    
    # Make sure in-memory (dask) chunks are never larger than the data
    final_ds = final_ds.chunk({
        "y": min(YCHUNK, final_ds.sizes.get("y", YCHUNK)),
        "x": min(XCHUNK, final_ds.sizes.get("x", XCHUNK)),
    })
    
    # encoding
    #encoding = {v: {"zlib": True, "complevel": 4, "chunksizes": WRITE_CHUNKS} for v in final_ds.data_vars}
    encoding = {}
    for v, da in final_ds.data_vars.items():
        ysize = da.sizes.get("y", 1)
        xsize = da.sizes.get("x", 1)
        cy = min(YCHUNK, ysize)
        cx = min(XCHUNK, xsize)
        encoding[v] = {
            "zlib": True,
            "complevel": 4,
            "chunksizes": (cy, cx),  # (y, x)
        }
    
    out = f"/lustre/tmp/kamarain/resiclim-microclimate/dem_features_{region}.nc"
    print("Saving to", out)
    final_ds.to_netcdf(out, engine="h5netcdf", encoding=encoding)
    
    print(region, "ready!")


sys.exit()



"""

# Read modules
import glob
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr #; xr.set_options(file_cache_maxsize=1)

#import s3fs

import matplotlib.pyplot as plt

#import functions as fcts


regions = ('TII', 'RAS', 'KIL', 'VAR', 'ULV', 'PAL')
regions = ('RAS', 'KIL', 'VAR', 'ULV', 'PAL')

'''
file_listing = pd.read_csv('/users/kamarain/resiclim-microclimate/pekan_maisemamuuttujat.csv', index_col=False)['File Name']
quantities = []
for fname in file_listing:
    quantities.append(fname.split('.')[0])



quantities = ['pisr_1','pisr_2','pisr_3','pisr_4','pisr_5','pisr_6','pisr_7','pisr_8','pisr_9','pisr_10','pisr_11','pisr_12',
              #'tpi_10','tpi_50','tpi_100','tpi_500','tpi_1000',
              'aspect','slope',
              #'VMI_tilavuus','VMI_latvuspeitto','VMI_keskipituus',
              'windexp500','svf','mbi','norm_height','mpi2000','mid_slope_position',
              'swi_suction16', 'swi_suction256',
              'dem10m','dem2m',
              'lidR_chm','diurnalaniheat',]

'''

quantities = [  'pisr_1','pisr_2','pisr_3','pisr_4','pisr_5','pisr_6','pisr_7','pisr_8',
                'pisr_9','pisr_10','pisr_11','pisr_12',
                'windexp500','svf','mbi','norm_height','mpi2000','mid_slope_position',
                'swi_suction16', 'swi_suction256',
                'dem10m','diurnalaniheat',]


for region in regions:
    tiff_files = np.sort(glob.glob(f'/lustre/tmp/kamarain/resiclim-topo/{region}/*.tif'))
    print(region, len(tiff_files), ':\n')
    
    
    results = []; #found_quantities = []
    for tiff_file in tiff_files:
        
        
        name =  tiff_file.split('/')[-1].split('.tif')[0] #riox_image.long_name
        
        if name not in quantities: continue
        
        print(region, tiff_file)
        
        # Load the TIFF file using rioxarray
        riox_image = rxr.open_rasterio(tiff_file, masked=True)

        # Ensure the data is in float format to accommodate NaN values
        riox_image = riox_image.astype(np.float32)
        
        riox_image = riox_image.squeeze().drop('band').to_dataset(name=name).chunk({"x": 100, "y": 100})
        
        results.append(riox_image)
        
        # Save 10m coordinates for later use 
        if np.median(np.diff(riox_image['y'])) == -10.0:
            print('10m coordinates found for',region)
            x_ = riox_image['x']; y_ = riox_image['y']
            spatial_ref_ = riox_image['spatial_ref']
        
        '''
        if riox_image[name].min() < 0:  riox_image[name].plot(figsize=(10,10), cmap='seismic', center=0, robust=True); plt.show()
        if riox_image[name].min() >= 0: riox_image[name].plot(figsize=(10,10), cmap='terrain', robust=True); plt.show()
        
        
        # Extract the (x, y) coordinates
        x_coords = riox_image.coords['x'].values
        y_coords = riox_image.coords['y'].values

        # Create meshgrid of the coordinates
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)

        # Flatten the meshgrid for transformation
        x_flat = x_mesh.flatten()
        y_flat = y_mesh.flatten()

        # Transform the coordinates
        lon_flat, lat_flat = fcts.etrs_tm35fin_to_wgs84(x_flat, y_flat)

        # Reshape the transformed coordinates back to the original shape
        lon_grid = lon_flat.reshape(y_mesh.shape)
        lat_grid = lat_flat.reshape(y_mesh.shape)
        '''
    
    print('Now regridding',region)
    final_results = []
    for i, ds in enumerate(results): 
        print(list(ds.data_vars)[0], np.median(np.diff(ds['y'])), np.median(np.diff(ds['y'])))
        
        ds = ds.interp(x=x_, y=y_, method='nearest')#.load()
        final_results.append(ds)
        
    print('Now merging',region)
    final_ds = xr.merge(final_results)#[quantities]
    final_ds['spatial_ref'] = spatial_ref_
    
    print('Now saving',region)
    final_ds.to_netcdf(f'/lustre/tmp/kamarain/resiclim-microclimate/dem_features_{region}.nc')
    print(region,'ready!\n')

"""


