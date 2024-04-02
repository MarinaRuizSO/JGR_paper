import numpy as np
import rioxarray as rxr
import rasterio
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
def plot_glim_data(mountain, ax, base_path):
    glim_cropped_to_dem = rxr.open_rasterio(f'{mountain}_glim_crs_dem.tif', masked=True, decode_coords='all').squeeze()
    dem_mask = rasterio.open(base_path + f'{mountain}/input_data/{mountain}_dem.bil')

    glim_processed = glim_cropped_to_dem.fillna(99)

    # Plot data using nicer colors

    raster_no_nan = np.nan_to_num(np.squeeze(glim_processed), nan=99)
    unique_values = np.unique(raster_no_nan)

    # read colors from csv file
    litho_csv = pd.read_csv('./glim_lithology_codes.csv')
    
    # select only the legend labels that are in the raster
    unique_litho_csv = litho_csv[litho_csv['Number'].isin(unique_values)]

    # always add the nan values row
    colors = unique_litho_csv.Color.tolist()
    labels = unique_litho_csv.Lithology.tolist()
    # Create a list of labels to use for your legend


    class_bins = (np.array(unique_litho_csv.index.tolist())-0.5).tolist()+[100]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(class_bins,
                        len(colors))
    return norm, cmap, colors, labels, dem_mask, glim_processed
