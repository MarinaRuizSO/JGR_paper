# Importing what I need, they are all installable from conda in case you miss one of them
import numpy as np
import numba as nb
import rasterio
import glob
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

from functions_automate_chi_extraction import count_outlets_in_mountain, read_outlet_csv_file, get_dem_crs, coordinate_transform
from function_glim_data import plot_glim_data
import rioxarray as rxr

import pyproj
import matplotlib as mpl
import matplotlib.colors as mcolors
# plt.rc('text', usetex=True)
# plt.rcParams['text.latex.preamble'] = [r'\boldmath']
from mpl_toolkits.axes_grid1 import make_axes_locatable
#plt.rcParamaxes.labelweight
plt.rcParams['axes.labelweight'] = 'normal'
# mpl.rcParams["axes.labelsize"] = 24
# mpl.rcParams["font.size"] = 18
#plt.rcParams['fig.labelweight'] = 'bold'



chi_with_rain = False


# file_path = '/exports/csce/datastore/geos/users/s1440040/LSDTopoTools/data/ExampleTopoDatasets/ChiAnalysisData/dems_to_process/'
# DEM_paths = [file_path+'argentina/input_data/']#[file_path+'pyrenees/input_data/',file_path+'massif_central/input_data/',file_path+'pyrenees/input_data/']#[file_path+'peru/input_data/', file_path+'argentina/input_data/', file_path+'turkey/input_data/',file_path+'xian/input_data/', file_path+'colorado/input_data/', file_path+'alburz_south/input_data/', file_path+'massif_central/input_data/',file_path+'pyrenees/input_data/']# complete the list later - first try with just two cases
# mountain_range_names = ['argentina']#['massif_central', 'pyrenees']#['peru','argentina', 'turkey', 'xian','colorado','alburz_south', 'massif_central', 'pyrenees']
# file_names = ['argentina_dem.bil']#['massif_central_dem.bil', 'pyrenees_dem.bil']#['peru_dem.bil', 'argentina_dem.bil', 'turkey_dem.bil',
#               #'xian_dem.bil', 'colorado_dem.bil', 'alburz_south_dem.bil',
#               #'massif_central_dem.bil', 'pyrenees_dem.bil']
# title_name = ['Andes, Northern Argentina']#['Massif Central, France', 'Pyrénées, Spain-France']#['Andes, Southern Perú', 'Andes, Northern Argentina',
#                #'Kaçkar Mts, Turkey', 'North Qinling Mts, China',
#                 #'Southern Rockies, USA','Alburz Mts, Iran', 
#                 #'Massif Central, France', 'Pyrénées, Spain-France']

# crs_list_cartopy = ['20S']#['19S', '20S', '37N', '48N',  '13N','39N','31N', '31N']
# area_threshold = 1e7
# file_name_number = 0


# read the csv and convert the coordinates to the crs of the dem


def transform_coords_to_4326(dem_crs, x, y):
    proj = pyproj.Transformer.from_crs(dem_crs, 4326, always_xy=True)
    x1, y1 = (x, y)
    x2, y2 = proj.transform(x1, y1)
    print(f'lon, lat: {x2}, {y2}')
    return x2, y2


def add_basin_outlines(mydem, fig, ax, size_outline = 1, zorder = 2, color = "k"):
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

		ax.scatter(val["x"], val["y"], c = color, s = size_outline, zorder = zorder, lw =0)


def calculate_delta_ksn(df_1, df_2):
      delta_ksn = abs(df_1.m_chi - df_2.m_chi)
      return delta_ksn
      
def run_full_ksn_delta_case_ia(map_ax, map_fig, label_count, case_labels,cbar_labels, subplot_letter, mountain_count, DEM_paths, file_names,mountain_range_names, base_path):


    #ksn_bounds = pd.read_csv(DEM_paths[mountain_count]+mountain_range_names[i]+'_ksn_bounds.csv')

    lon_transform_list = []
    lat_transform_list = []
    filelist = count_outlets_in_mountain(DEM_paths[mountain_count])
    count_file_list = 0
    #map_fig, map_ax = plt.subplots(1,1,figsize=(7,6))
    cmap = cmc.hawaii#plt.cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    #cmaplist[0] = (.5, .5, .5, 1.0)

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    color_map = cmc.lajolla

    dem_crs = get_dem_crs(DEM_paths[mountain_count],file_names[mountain_count])
    dataset = rasterio.open(DEM_paths[mountain_count]+file_names[mountain_count])
    dem_extent = dataset.bounds

    lon_left, lat_bottom = transform_coords_to_4326(dem_crs, dem_extent[0], dem_extent[1])
    #lon_right, lat_top = transform_coords_to_4326(dem_crs, dem_extent[2], dem_extent[3])

    for file in filelist[:]:
        lat_outlet, lon_outlet = read_outlet_csv_file(file)
        #breakpoint()
        dem_crs = get_dem_crs(DEM_paths[mountain_count],file_names[mountain_count])
        basin_name = file.split('/')[-1].split('__')[0]
        lon_transform, lat_transform = coordinate_transform(lat_outlet, lon_outlet, dem_crs)
        lon_transform_list.append(lon_transform)
        lat_transform_list.append(lat_transform)
    
        dem_basins_with_rain = lsd.LSDDEM(path = DEM_paths[mountain_count] ,file_name = file_names[mountain_count])
        dem_basins_with_rain.PreProcessing(filling = True, carving = True, minimum_slope_for_filling = 0.0001)  


        dem_basins_with_rain.ExtractRiverNetwork(method = "area_threshold", area_threshold_min = 200)


        # XY_basins_with_rain = dem_basins_with_rain.DefineCatchment(method="min_area", test_edges = False, min_area = area_threshold,max_area = 1e20, X_coords = [], Y_coords = [], 
        # 	 coord_search_radius_nodes = 30, coord_threshold_stream_order = 3)
        XY_basins_with_rain = dem_basins_with_rain.DefineCatchment(method="from_XY", test_edges = False, min_area = 0,max_area = 0, X_coords = lon_transform_list, Y_coords = lat_transform_list, 
            coord_search_radius_nodes = 30, coord_threshold_stream_order = 3)
        
        # load the dem information from the csv file

        df_basins_with_rain_theta_45 = pd.read_csv(DEM_paths[mountain_count]+f'{basin_name}_no_rainfall__chi_map_theta_0_45.csv')
        df_no_nan_theta_45 = df_basins_with_rain_theta_45[df_basins_with_rain_theta_45.m_chi >= 0]
    
        for f in glob.glob(DEM_paths[mountain_count]+f'{basin_name}_no_rainfall__chi_map_theta_best_*.csv'):
            df_basins_with_rain_theta_best = pd.read_csv(f)
        df_no_nan_theta_best = df_basins_with_rain_theta_best[df_basins_with_rain_theta_best.m_chi >= 0]

        ### theta 0.45 ###
        # processing follows the same method as for ksn distortion calculations
        # 1 - normalise the drainage area and remove values that are below 0.1 
        df_no_nan_theta_45.drainage_area/=np.max(df_no_nan_theta_45.drainage_area)
        df_basins_with_rain_theta_45_da_0_1 = df_no_nan_theta_45[df_no_nan_theta_45.drainage_area>=0.1]
        # 2 - normalise the m_chi (ksn) values by the highest in the basin
        #breakpoint()
        df_basins_with_rain_theta_45_da_0_1.m_chi/=np.max(df_basins_with_rain_theta_45_da_0_1.m_chi)
        df_basins_with_rain_theta_45_da_0_1['m_chi_norm'] = df_basins_with_rain_theta_45_da_0_1.m_chi/df_basins_with_rain_theta_45_da_0_1.m_chi.max()

        ### theta best ###
        # processing follows the same method as for ksn distortion calculations
        # 1 - normalise the drainage area and remove values that are below 0.1 
        df_no_nan_theta_best.drainage_area/=np.max(df_no_nan_theta_best.drainage_area)
        df_basins_with_rain_theta_best_da_0_1 = df_no_nan_theta_best[df_no_nan_theta_best.drainage_area>=0.1]
        # 2 - normalise the m_chi (ksn) values by the highest in the basin
        #breakpoint()
        df_basins_with_rain_theta_best_da_0_1.m_chi/=np.max(df_basins_with_rain_theta_best_da_0_1.m_chi)
        df_basins_with_rain_theta_best_da_0_1['m_chi_norm'] = df_basins_with_rain_theta_best_da_0_1.m_chi/df_basins_with_rain_theta_best_da_0_1.m_chi.max()

        df_basins_with_rain_delta_ksn = np.abs(df_basins_with_rain_theta_45_da_0_1.m_chi_norm - df_basins_with_rain_theta_best_da_0_1.m_chi_norm)
        df_basins_with_rain_delta_ksn_no_nan = df_basins_with_rain_delta_ksn[df_basins_with_rain_delta_ksn>=0]
        
        #delta_ksn_joined = pd.merge(df_basins_with_rain_theta_best_da_0_1, df_basins_with_rain_delta_ksn_no_nan, left_index=True, right_index=True)
        delta_ksn_joined = pd.merge(df_basins_with_rain_theta_best_da_0_1, df_basins_with_rain_delta_ksn_no_nan, left_index=True, right_index=True)
        delta_ksn_joined.rename(columns={"m_chi_norm_y": "delta_m_chi_norm"}, inplace=True)

        count_file_list+=1
        size = lsd.quickplot_utilities.size_my_points(delta_ksn_joined.drainage_area.values,1, 5)
        scat = map_ax.scatter(delta_ksn_joined.x,delta_ksn_joined.y, c=delta_ksn_joined.delta_m_chi_norm, cmap = color_map, s = size, lw = 0, zorder = 2, vmax=0.5)#, cmap = cmap)


    
    print('I am about to go into the glim function')
    norm, cmap, colors, labels, dem_mask, glim_processed = plot_glim_data(mountain_range_names[mountain_count],map_ax, base_path )
    print('I am going to plot the glim data')
    im = glim_processed.plot.imshow(cmap=cmap,
                                    norm=norm,
                                    # Turn off colorbar
                                    add_colorbar=False, ax = map_ax, alpha=0.75)
    # define the bins and normalize
    #bounds = np.linspace(0, total_basins_thresh, total_basins_thresh+1)
    #norm = mpl.colors.BoundaryNorm(10, cmap.N)
    # define the bins and normalize
    #bounds = np.linspace(0, total_basins_thresh, total_basins_thresh+1)
    #norm = mpl.colors.BoundaryNorm(10, cmap.N)

    # make the scatter
    # lsd.quickplot.get_basemap(
    # dem_basins_with_rain, # which dem object to plot
    # figsize = (4,4), # figsize
    # cmap = cmc.batlowW, # colormap for elevation
    # cmin = 0, cmax = None, # color limit for elevation (None for defaulting to min max)
    # hillshade = True, # Hillshade ?
    # alpha_hillshade = 0.6,normalise_HS = True, # options
    # hillshade_cmin = 0, hillshade_cmax = 1, # HS cmap boundaries
    # colorbar = False, #Plotting cbar?
    # fig = map_fig, ax = map_ax, colorbar_label = 'Elevation (m)', 
    # colorbar_ax = None, fontsize_ticks = 10) # advanced usage
    topography = rxr.open_rasterio(base_path+f'/{mountain_range_names[mountain_count]}/input_data/{mountain_range_names[mountain_count]}_dem_hs.bil', masked=True, decode_coords='all').squeeze()
    #topography=rioxarray.open_rasterio(file_path+'/argentina/input_data/argentina_dem_hs.bil')
    
    topography.plot.imshow(add_colorbar=False, ax = map_ax, alpha=0.25, cmap = cmc.grayC_r)
    map_ax.set_title('')
    map_ax.set_xlim(left=dataset.bounds[0], right=dataset.bounds[2])
    map_ax.get_xaxis().set_visible(True)
    map_ax.get_yaxis().set_visible(True)
    map_ax.set_xticklabels((map_ax.get_xticks()/1000).astype(int))
    map_ax.set_yticklabels((map_ax.get_yticks()/1000).astype(int))
    map_ax.set_xlabel('Easting (km)')
    map_ax.set_ylabel('Northing (km)')
    map_fig.set_facecolor("w")

    add_basin_outlines(dem_basins_with_rain, map_fig, map_ax, size_outline = 1, zorder = 5, color = "k")
    

    #plt.margins(x=-0.02, y=-0.005)

    #cax = map_ax.inset_axes([1.85, 0, 0.1,  1])
    cbar = plt.colorbar(scat, ax=map_ax,fraction=0.046, pad=0.04)
    cbar.set_label(label=cbar_labels[label_count])#r'$log_{10}(\Delta k_{sn})$', size=24, weight='normal')

    
    #map_ax.set_xbound(lower=350000, upper=490000)
    #map_ax.set_xlim(265000, 322000)


    # lonW = -124
    # lonE = -71
    # latS = 22
    # latN = 49
    # #-13.6927, -69.9541
    # axins = inset_axes(map_ax, width="30%", height="30%", loc="upper right", 
    #                 axes_class=cartopy.mpl.geoaxes.GeoAxes, 
    #                 axes_kwargs=dict(map_projection=cartopy.crs.PlateCarree()))
    # axins.add_feature(cartopy.feature.COASTLINE)
    # axins.add_feature(cartopy.feature.RIVERS)
    # axins.add_feature(cartopy.feature.LAKES )
    # axins.set_extent([lonW, lonE, latS, latN])
    # axins.stock_img()
    # axins.scatter([lon_left], [lat_bottom],
    #     color='purple', linewidth=1, marker='*', zorder= 10, s = 40
        
    #     )
    # map_ax.set_title(title_name[i], fontsize=28)



    #plt.show()
    #plt.savefig(file_path+f'{mountain_range_names[i]}_no_rainfall_all_basins_delta_ksnTEST.jpg', dpi=500,  bbox_inches='tight')
    map_ax.text(0.1, 0.1, case_labels[label_count], transform=map_ax.transAxes, color='black',bbox=dict(boxstyle="round",
                ec='Black',
                fc='White', alpha=0.8
                ))
    map_ax.text(-0.5, 0.9, subplot_letter, transform=map_ax.transAxes, 
            weight='normal')
    
