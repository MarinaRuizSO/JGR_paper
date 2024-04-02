from osgeo import gdal
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
import os
import cmcrameri.cm as cmc
import earthpy.plot as ep
import figure_specs_paper

import xarray as xr
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Keep this or the code breaks because the HDF files aren't closed properly.


dem_path = '/exports/csce/datastore/geos/users/s1440040/LSDTopoTools/data/ExampleTopoDatasets/ChiAnalysisData/dems_to_process/'


# Importing what I need, they are all installable from conda in case you miss one of them
import numpy as np
import numba as nb
import rasterio
import cartopy
import cartopy.mpl.geoaxes
import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import matplotlib.pyplot as plt
import helplotlib as hpl
import lsdtopytools as lsd
# Ignore that last
# %load_ext xsimlab.ipython
import cmcrameri.cm as cmc
import os
import pickle as pkl
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Keep this or the code breaks because the HDF files aren't closed properly.
import cartopy.crs as ccrs
import rioxarray as rxr

from functions_automate_chi_extraction import count_outlets_in_mountain, read_outlet_csv_file, get_dem_crs, coordinate_transform
import pyproj
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import rcParams

# plt.rc('text', usetex=True)
# plt.rcParams['text.latex.preamble'] = [r'\boldmath']
from mpl_toolkits.axes_grid1 import make_axes_locatable
#plt.rcParamaxes.labelweight
plt.rcParams['axes.labelweight'] = 'normal'
# mpl.rcParams["axes.labelsize"] = 24
# mpl.rcParams["font.size"] = 18
#plt.rcParams['fig.labelweight'] = 'bold'
#plt.rc('legend',fontsize=5)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Liberation Sans']
rcParams['font.size'] = 8



     
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

def transform_coords_to_4326(dem_crs, x, y):
    proj = pyproj.Transformer.from_crs(dem_crs, 4326, always_xy=True)
    x1, y1 = (x, y)
    x2, y2 = proj.transform(x1, y1)
    print(f'lon, lat: {x2}, {y2}')
    return x2, y2


def add_basin_outlines(mydem, fig, ax, size_outline = 1, zorder = 2, color = "k", alpha=0.5):
	"""
		Add catchment outlines to any axis given with its associated LSDDEM object ofc.
		No need to mention that catchment needs to be extracted within the LSDDEM object beforehand.
		Arguments:
			fig: the matplotlib figure
			ax: the matplotlib ax
			size_outline: size (in matplotlib units) of the outline
			zorder: the matplotlib zorder to use
			color (any color code compatible with matplotlib): the color of the outline. Default is black
		Returns:
			Nothing, It directly acts on the figure passed in argument
		Authors:
			B.G.
		Date:
			12/2018 (last update on the 23/02/2018)
	"""
	# getting dict of perimeters
	outlines = mydem.cppdem.get_catchment_perimeter()
	# keys are x,y,elevation,basin_key
	for key,val in outlines.items():

		ax.scatter(val["x"], val["y"], c = color, alpha = alpha, s = size_outline, zorder = zorder, lw =0)


def plot_glim(map_ax, map_fig, subplot_letter, mountain_count):
    print(f'I am processing {mountain_range_names[mountain_count]}')

    lon_transform_list = []
    lat_transform_list = []
    filelist = count_outlets_in_mountain(DEM_paths[mountain_count])
    count_file_list = 0
    crs = ccrs.UTM(crs_list_cartopy[mountain_count])
    #map_fig = plt.figure(figsize=(7,6))
    #map_fig = figure_specs_paper.CreateFigure(FigSizeFormat="JGR", AspectRatio=16./9.)
    #map_ax = plt.subplot(projection=crs)
    #map_fig, map_ax = plt.subplots(1,1,figsize=(7,6), projection=crs)
    cmap = cmc.hawaii#plt.cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    color_map = cmc.lajolla

    dem_crs = get_dem_crs(DEM_paths[mountain_count],file_names[mountain_count])
    dataset = rasterio.open(DEM_paths[mountain_count]+file_names[mountain_count])
    dem_extent = dataset.bounds

    lon_left, lat_bottom = transform_coords_to_4326(dem_crs, dem_extent[0], dem_extent[1])
    # lon_right, lat_top = transform_coords_to_4326(dem_crs, dem_extent[2], dem_extent[3])
    # output_raster_name = f'{mountain_range_names[mountain_count]}_glim_crs_dem.tif'
    # output_raster_new = rasterio.open(output_raster_name)
    

    for file in filelist[:]:
        lat_outlet, lon_outlet = read_outlet_csv_file(file)
        dem_crs = get_dem_crs(DEM_paths[mountain_count],file_names[mountain_count])
        basin_name = file.split('/')[-1].split('__')[0]
        lon_transform, lat_transform = coordinate_transform(lat_outlet, lon_outlet, dem_crs)
        lon_transform_list.append(lon_transform)
        lat_transform_list.append(lat_transform)
    
        dem_basins_with_rain = lsd.LSDDEM(path = DEM_paths[mountain_count] ,file_name = file_names[mountain_count])
        dem_basins_with_rain.PreProcessing(filling = True, carving = True, minimum_slope_for_filling = 0.0001)  


        dem_basins_with_rain.ExtractRiverNetwork(method = "area_threshold", area_threshold_min = 200)


        XY_basins_with_rain = dem_basins_with_rain.DefineCatchment(method="from_XY", test_edges = False, min_area = 0,max_area = 0, X_coords = lon_transform_list, Y_coords = lat_transform_list, 
            coord_search_radius_nodes = 30, coord_threshold_stream_order = 3)
        
        # load the dem information from the csv file

        df_basins_with_rain_new = pd.read_csv(DEM_paths[mountain_count]+f'{basin_name}_no_rainfall__chi_map_theta_0_45.csv')
        df_basins_with_rain = df_basins_with_rain_new[df_basins_with_rain_new.m_chi>=0]
        

        #df_basins_with_rain = df_basins_with_rain[df_basins_with_rain.drainage_area>=0.01]
        df_basins_with_rain.drainage_area/=np.max(df_basins_with_rain.drainage_area)
        df_basins_with_rain = df_basins_with_rain[df_basins_with_rain.drainage_area>=0.01]
        
        size_array = lsd.size_my_points(np.log10(df_basins_with_rain.drainage_area), 1,2)
        map_ax.scatter(df_basins_with_rain.x, df_basins_with_rain.y, lw=0, c= "black",  zorder = 5, s=1)
        add_basin_outlines(dem_basins_with_rain, map_fig, map_ax, size_outline = 1, zorder = 4, color = "purple")

        my_dem_raster = lsd.raster_loader.load_raster(DEM_paths[mountain_count]+file_names[mountain_count])

        divider = make_axes_locatable(map_ax)
        count_file_list+=1

    

    print('I am about to go into the glim function')
    norm, cmap, colors, labels, dem_mask, glim_processed = plot_glim_data(mountain_range_names[mountain_count],map_ax, file_path )
    print('I am going to plot the glim data')
    im = glim_processed.plot.imshow(cmap=cmap,
                                    norm=norm,
                                    # Turn off colorbar
                                    add_colorbar=False, ax = map_ax, alpha=0.75)
    

    print('plotted! Now on to adding some extra shit')
    # Add legend using earthpy
    figure_specs_paper.draw_legend(im,titles=labels, size_font=10)
    # im.axes.legend(
    #     prop={"size": 13},
    # )
    #ax.legend(prop=dict(size=18))
    #plt.setp(fontsize='xx-small')
    #plt.legend(fontsize=10)
    map_ax.set_xlim(left=dem_mask.bounds[0], right=dem_mask.bounds[2])
    map_ax.get_xaxis().set_visible(True)
    map_ax.get_yaxis().set_visible(True)
    print('hi I will show plot maybe?')
    latlon_df = pd.read_csv(file_path+'map_locations_lookup_table.csv')
    lonW = int(latlon_df[mountain_range_names[mountain_count]][0])
    lonE = int(latlon_df[mountain_range_names[mountain_count]][1])
    latS = int(latlon_df[mountain_range_names[mountain_count]][2])
    latN = int(latlon_df[mountain_range_names[mountain_count]][3])
    #-13.6927, -69.9541
    axins = inset_axes(map_ax, width="30%", height="30%", loc=f'{latlon_df[mountain_range_names[mountain_count]][4]}', 
                    axes_class=cartopy.mpl.geoaxes.GeoAxes,
                    axes_kwargs=dict(map_projection=ccrs.PlateCarree()))
    axins.add_feature(cartopy.feature.COASTLINE)
    axins.add_feature(cartopy.feature.RIVERS)
    axins.add_feature(cartopy.feature.LAKES)
    axins.set_extent([lonW, lonE, latS, latN])
    axins.stock_img()
    axins.scatter([lon_left], [lat_bottom],
        color='purple', linewidth=1, marker='*', zorder= 10, s = 40
        
        )
    
    # map_ax.text(-0.7, 0.9, subplot_letter, transform=map_ax.transAxes, 
    #         weight='normal')
    # plt.tight_layout()
    # plt.show()
    #src = rasterio.open(file_path+'/argentina/input_data/argentina_dem_hs.bil') 
    topography = rxr.open_rasterio(file_path+f'/{mountain_range_names[mountain_count]}/input_data/{mountain_range_names[mountain_count]}_dem_hs.bil', masked=True, decode_coords='all').squeeze()
    #topography=rioxarray.open_rasterio(file_path+'/argentina/input_data/argentina_dem_hs.bil')
    
    topography.plot.imshow(add_colorbar=False, ax = map_ax, alpha=0.25, cmap = cmc.grayC_r)
    map_ax.set_title('')
    # convert the x and y ticks to km 

    map_ax.set_xticklabels((map_ax.get_xticks()/1000).astype(int))
    map_ax.set_yticklabels((map_ax.get_yticks()/1000).astype(int))
    map_ax.set_xlabel('Easting (km)')
    map_ax.set_ylabel('Northing (km)')
    map_ax.text(-0.3, 0.9, subplot_letter, transform=map_ax.transAxes, 
            weight='normal')
    plt.savefig(file_path+f'{mountain_range_names[mountain_count]}_lithoGLim_all_basins.jpg', dpi=500, bbox_inches='tight')

chi_with_rain = False


file_path = '/exports/csce/datastore/geos/users/s1440040/LSDTopoTools/data/ExampleTopoDatasets/ChiAnalysisData/dems_to_process/'
DEM_paths = [file_path+'pyrenees/input_data/']#, file_path+'argentina/input_data/', file_path+'turkey/input_data/',
#              file_path+'xian/input_data/', file_path+'colorado/input_data/', file_path+'alburz_south/input_data/', 
#              file_path+'pyrenees/input_data/',file_path+'massif_central/input_data/']# complete the list later - first try with just two cases
# mountain_range_names = ['peru','argentina', 'turkey', 'xian','colorado','alburz_south', 'pyrenees', 'massif_central']
# DEM_paths = [file_path+'dems_to_process/peru/input_data/', file_path+'dems_to_process/argentina/input_data/', 
#              file_path+'dems_to_process/turkey/input_data/',file_path+'dems_to_process/xian/input_data/', 
#              file_path+'dems_to_process/colorado/input_data/', file_path+'dems_to_process/alburz_south/input_data/',
#              file_path+'dems_to_process/pyrenees/input_data/', file_path+'dems_to_process/massif_central/input_data/']# complete the list later - first try with just two cases
mountain_range_names =  ['pyrenees']#,'argentina', 'turkey', 'xian','colorado','alburz_south', 'pyrenees', 'massif_central']
file_names = ['pyrenees_dem.bil']#, 'argentina_dem.bil', 'turkey_dem.bil',
            #    'xian_dem.bil', 'colorado_dem.bil', 'alburz_dem.bil',
            #     'pyrenees_dem.bil','massif_central_dem.bil']

crs_list_cartopy = ['31N']#, '20S', '37N', '48N',  '13N','39N','31N', '31N']
area_threshold = 1e7


count = 0

for mountain in mountain_range_names:
    map_fig = figure_specs_paper.CreateFigure(FigSizeFormat="JGR", AspectRatio=16./9., size_font=10)
    crs = ccrs.UTM(crs_list_cartopy[count])
    map_ax = plt.subplot(projection=crs)
    plot_glim(map_ax, map_fig, 'A', count)
    count+=1

