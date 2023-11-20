# Importing what I need, they are all installable from conda in case you miss one of them
import numpy as np
import pandas as pd
import xarray as xr
import xsimlab as xs
import fastscape as fst
from fastscape.processes.context import FastscapelibContext
import numba as nb
import math
import zarr
import matplotlib.pyplot as plt
import os
import helplotlib as hpl
import lsdtopytools as lsd
import cmcrameri.cm as cmc
import rasterio
import re
import pyproj
import pickle
import itertools
from functions_automate_chi_extraction import *
from matplotlib.offsetbox import AnchoredText

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Keep this or the code breaks because the HDF files aren't closed properly.



######################################################################################################
# LOAD THE DATA
## The names of the files need to be standardised so that they are read properly.
base_path = '/exports/csce/datastore/geos/users/s1440040/LSDTopoTools/data/ExampleTopoDatasets/ChiAnalysisData/dems_to_process/'
DEM_paths = [base_path+'peru/input_data/', base_path+'argentina/input_data/', base_path+'turkey/input_data/',base_path+'xian/input_data/', base_path+'colorado/input_data/', base_path+'alburz_south/input_data/', base_path+'massif_central/input_data/',base_path+'pyrenees/input_data/']# complete the list later - first try with just two cases
mountain_range_names =  ['peru','argentina', 'turkey', 'xian','colorado','alburz_south', 'massif_central', 'pyrenees']

DEM_names = ['peru_dem.bil','argentina_dem.bil', 'turkey_dem.bil', 'xian_dem.bil','colorado_dem.bil', 'alburz_south_dem.bil', 'massif_central_dem.bil', 'pyrenees_dem.bil']

#precip_names = ['andes_rainfall.bil', 'alburz_north_rainfall.bil']
# all of this precipitation needs to be in m/yr
precipitation_types = ['original_rain']#, 'rain_case1', 'rain_case2', 'rain_case3', 'rain_case4', 'rain_case5']
list_preci_file_names = []
for m, p in itertools.product(mountain_range_names,precipitation_types):
    precipitation_file_name = os.path.splitext(m)[0] + '_'+ p +'.bil'
    list_preci_file_names.append(precipitation_file_name)
    #print(m,p)

# ######################################################################################################
# ######################################################################################################
# ######################################################################################################
area_thres = 1500
n_river_threshold = 80

dem_counter = 0
rain_counter = 0
include_rain = False
for i in range(len(mountain_range_names)):
    filelist = count_outlets_in_mountain(DEM_paths[i])
    for k in range(len(filelist)):
        rain_counter = -1 # this is to fix the fact that the 'no rainfall' is not in the list of rainfall cases
        rain_on = False
        for j in range(len(precipitation_types)+1):
            csv_file_name = filelist[k]
            basin_name = get_basin_name(csv_file_name)
            print('I am processing basin: {}'.format(basin_name))

            # read the csv and convert the coordinates to the crs of the dem
            lat_outlet, lon_outlet = read_outlet_csv_file(csv_file_name)#DEM_paths[i]+csv_file_name)
            dem_crs = get_dem_crs(DEM_paths[i],DEM_names[i])
            lon_transform, lat_transform = coordinate_transform(lat_outlet, lon_outlet, dem_crs)
            print(lat_transform, lon_transform)

            ######################################################################################################
            # DEM PROCESSING STEPS

            my_dem = lsd.LSDDEM(
                path = DEM_paths[i],
                file_name = DEM_names[i]
                )

            my_dem.PreProcessing(
                filling = True,
                carving = True,
                minimum_slope_for_filling = 0.0001
                )
            if rain_on == False:
                print('No rainfall')
                my_dem.CommonFlowRoutines(
                    discharge = False
                    )
            else:
                print(f'Rainfall case: {precipitation_types[rain_counter]}')
                precipitation_file_name = os.path.splitext((mountain_range_names[i]))[0] + '_'+ precipitation_types[rain_counter] +'.tif'
                my_dem.CommonFlowRoutines(
                    discharge = True,
                    ingest_precipitation_raster = DEM_paths[i]+precipitation_file_name,
                    precipitation_raster_multiplier = 1 # the units MUST be in m/yr
                    )

            #rain_on = True
            # min_area = 1e9 without precipitation
            # min_area = 8e8 with precipitation
            #area_thres = 10000

            ###### let's comment this out so that I can test whether the fine tuning
            ###### mechanism works.
            # add the threshold fine tuning function so that the number of tributaries
            # is the same as after the theta is optimised

            my_dem, number_rivers, updated_area_threshold = fine_tune_threshold_area(my_dem, area_thres, lon_transform, lat_transform, n_river_threshold)

            ######################

            # my_dem.ExtractRiverNetwork(
            #     method = "area_threshold",
            #     area_threshold_min = area_thres # 1500 is the default value
            #     )
            #
            # my_dem.DefineCatchment(
            #     method="from_XY",
            #     X_coords = [lon_transform], # input the outlet coordinates
            #     Y_coords = [lat_transform]
            #     )
            #
            # my_dem.GenerateChi(
            #     theta=0.45,
            #     A_0 = 1
            #     )

            #total_basins = len(XY_basins_with_rain['X'])
            ######################

            my_dem.ksn_MuddEtAl2014(
                target_nodes=30,
                n_iterations=60,
                skip=1
                )
            ######
            plot_base_dem(my_dem, DEM_paths[i], DEM_names[i], basin_name+'__base_dem')
            if rain_on == False:
                plot_ksn_map(my_dem, DEM_paths[i], DEM_names[i], basin_name+'_no_rainfall_theta_0_45__ksn_map', 0.45, 'No rainfall', is_it_optimal_theta=False)
            else:
                plot_rainfall_dem_rainfall(my_dem, DEM_paths[i], DEM_names[i], precipitation_file_name, basin_name+'_'+precipitation_types[rain_counter]+'__rainfall_distribution', precipitation_types[rain_counter])
                plot_ksn_map(my_dem, DEM_paths[i], DEM_names[i], basin_name+'_'+precipitation_types[rain_counter]+'_theta_0_45__ksn_map', 0.45, precipitation_types[rain_counter], is_it_optimal_theta=False)
            ######
            #df_basins_with_rain = my_dem.df_ksn
            df_ksn_analysis = my_dem.df_ksn
            if rain_on == False:
                df_ksn_analysis.to_csv(DEM_paths[i]+f'{basin_name}_no_rainfall__chi_map_theta_0_45.csv', index=False)
            else:
                df_ksn_analysis.to_csv(DEM_paths[i]+f'{basin_name}_{precipitation_types[rain_counter]}__chi_map_theta_0_45.csv', index=False)

            # FIND OPTImal theta values based on the disorder metric
            # test reloading the dem from scratch
            #my_dem = load_dem_from_scratch(DEM_paths[i], DEM_names[i], rain_on, precipitation_file_name,mountain_range_names[i], precipitation_types[rain_counter], rain_counter)

            all_disorder, results, median_theta = theta_quick_constrain_single_basin_exp(my_dem, updated_area_threshold, lon_transform, lat_transform, n_river_threshold)
            theta_range = np.linspace(0.05,1.,38)
            theta_range = np.round(theta_range, decimals = 2)
            if rain_on == False:
                with open(DEM_paths[i]+f'{basin_name}_no_rainfall__all_disorder.pkl', 'wb') as f:
                    pickle.dump(all_disorder, f)

                with open(DEM_paths[i]+f'{basin_name}_no_rainfall__results_theta.pkl', 'wb') as f:
                    pickle.dump(results, f)

                with open(DEM_paths[i]+f'{basin_name}_no_rainfall__median_theta.pkl', 'wb') as f:
                    pickle.dump(median_theta, f)

                with open(DEM_paths[i]+f'{basin_name}_no_rainfall__theta_range.pkl', 'wb') as f:
                    pickle.dump(theta_range, f)

            else:
                with open(DEM_paths[i]+f'{basin_name}_{precipitation_types[rain_counter]}__all_disorder.pkl', 'wb') as f:
                    pickle.dump(all_disorder, f)

                with open(DEM_paths[i]+f'{basin_name}_{precipitation_types[rain_counter]}__results_theta.pkl', 'wb') as f:
                    pickle.dump(results, f)

                with open(DEM_paths[i]+f'{basin_name}_{precipitation_types[rain_counter]}__median_theta.pkl', 'wb') as f:
                    pickle.dump(median_theta, f)

                with open(DEM_paths[i]+f'{basin_name}_{precipitation_types[rain_counter]}__theta_range.pkl', 'wb') as f:
                    pickle.dump(theta_range, f)


            ###
            #theta = median_theta # this is the optimum theta found using the disorder metric.

            my_dem.GenerateChi(
                theta=median_theta,
                A_0 = 1
                )
            my_dem.ksn_MuddEtAl2014(
                target_nodes=30,
                n_iterations=60,
                skip=1
                )
            ####
            # plot again but with the new optimal median theta value
            ####
            df_ksn_optimal_theta = my_dem.df_ksn
            if rain_on == False:
                df_ksn_optimal_theta.to_csv(DEM_paths[i]+f'{basin_name}_no_rainfall__chi_map_theta_best_{median_theta}.csv', index=False)
                plot_ksn_map(my_dem, DEM_paths[i], DEM_names[i], basin_name+f'_no_rainfall_theta_best_{median_theta}__ksn_map', median_theta, 'No rainfall', is_it_optimal_theta=True)
            else:
                df_ksn_optimal_theta.to_csv(DEM_paths[i]+f'{basin_name}_{precipitation_types[rain_counter]}__chi_map_theta_best_{median_theta}.csv', index=False)
                plot_ksn_map(my_dem, DEM_paths[i], DEM_names[i], basin_name+'_'+precipitation_types[rain_counter]+f'_theta_best_{median_theta}__ksn_map', median_theta,precipitation_types[rain_counter], is_it_optimal_theta=True)
            rain_counter+=1
            rain_on = True
