import numpy as np
from shapely.geometry import box
import os
import xarray as xr
import geopandas as gpd
import pandas as pd
import rasterio
from geocube.api.core import make_geocube
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Keep this or the code breaks because the HDF files aren't closed properly.

mountain_names =  ['pyrenees']#['peru','argentina', 'turkey', 'xian','colorado','alburz_south', 'massif_central', 'pyrenees']

# load the glim database 
file_path = '/exports/csce/datastore/geos/groups/LSDTopoData/lithology/GLIM/'
dem_path = '/exports/csce/datastore/geos/users/s1440040/LSDTopoTools/data/ExampleTopoDatasets/ChiAnalysisData/dems_to_process/'

glimology_df = gpd.read_file(file_path + "LiMW_GIS 2015.gdb")
for mountain in mountain_names:
    print(f'I am cropping the GLiM database for: {mountain}')
    dem_mask = rasterio.open(dem_path + f'{mountain}/input_data/{mountain}_dem.bil')
    dem_mask_xr = xr.open_dataset(dem_path + f'{mountain}/input_data/{mountain}_dem.bil', engine='rasterio')

    # reproject the dem into the glim database crs
    mountain_in_glim_crs = dem_mask_xr.rio.reproject(glimology_df.crs)

    # clip the glim data by the dem extent
    geom=box(*mountain_in_glim_crs.rio.bounds())
    df = gpd.GeoDataFrame({"id":1,"geometry":[geom]})
    df.set_crs(glimology_df.crs, inplace=True)
    glim_clipped = gpd.clip(glimology_df, df.geometry, keep_geom_type=False)

    mask_crs_int = int(dem_mask.crs.to_string()[5:])
    # convert the categorical values for lithology into numbers
    # otherwise we cannot use make_geocube. It only allows for numerical values
    xx_int_glim_unique = np.unique(glim_clipped.xx, return_inverse=True)
    
    #need to save the lithological layer names for the legend 
    xx_int_series = pd.Series(xx_int_glim_unique[0])
    xx_int_series.to_csv(f'{mountain}_litho_codes.csv', index=True, header=False)

    dict_glim = {'su' : 0, 'ss' : 1, 'sm' : 2, 'py' : 3,
                 'sc' : 4, 'ev' : 5, 'mt' : 6, 'pa' : 7,
                 'pi' : 8, 'pb' : 9, 'va' : 10, 'vi' : 11,
                 'vb' : 12, 'ig' : 13, 'wb' : 14, 'nd' : 15,}
    xx_int_glim = glim_clipped.replace({"xx": dict_glim})
    glim_clipped['xx_int'] = xx_int_glim['xx']


    # rasterize the column with the lithological units
    out_grid_glim = make_geocube(vector_data=glim_clipped, measurements=["xx_int"], resolution=(-16, 16)) #for most crs negative comes first in resolution
    breakpoint()
    #save the column with the lithologies to a raster 
    out_grid_glim["xx_int"].rio.to_raster(f"{mountain}_glim.tif")