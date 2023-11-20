'''
File to do the statistical analysis of the disorder metrics and
the optimised theta value obtained from it.
Plot histograms for each study site separately

Marina Ruiz Sanchez-Oro
07/10/2022
'''
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import seaborn as sns
from numpy import median

import pandas as pd
import math
import cmcrameri.cm as cmc
import rasterio
import re
from matplotlib.offsetbox import AnchoredText
import pickle
import itertools
from functions_statistics import *
import helplotlib
import re
import glob
from matplotlib.ticker import StrMethodFormatter



# Load the disorder data
# We need to carry out the analysis in pairs.

base_path = '/exports/csce/datastore/geos/users/s1440040/LSDTopoTools/data/ExampleTopoDatasets/ChiAnalysisData/dems_to_process/'
DEM_paths = [base_path+'peru/input_data/', base_path+'argentina/input_data/', base_path+'xian_all/input_data/', base_path+'turkey/input_data/',base_path+'colorado/input_data/', base_path+'alburz_south/input_data/', base_path+'massif_central/input_data/',base_path+'pyrenees/input_data/']# complete the list later - first try with just two cases

#precip_names = ['andes_rainfall.bil', 'alburz_north_rainfall.bil']
mountain_range_names =  ['peru','argentina', 'xian', 'turkey', 'colorado', 'alburz_south', 'massif_central', 'pyrenees']#['andes_north', 'alburz_north', 'alburz_south', 'massif_central', 'pyrenees']
# all of this precipitation needs to be in m/yr
# here we an include the no rainfall scenario because we are not processing any DEM files.
precipitation_types = ['original_rain']#, 'rain_case1']#, 'rain_case2', 'rain_case3', 'rain_case4', 'rain_case5']
counter = 0


ksn_ratios_list = []

for i in range(len(precipitation_types)):
    for j in range(len(mountain_range_names)):
        # plot one histogram for each of the study sites
        # ksn_ratio_theta_45 = []
        # ksn_ratio_rainfall = []
        # ksn_ratio_no_rainfall = []

        filelist = count_outlets_in_mountain(DEM_paths[j])
        for k in range(len(filelist)):
            csv_file_name = filelist[k]
            basin_name = get_basin_name(csv_file_name)
            print('I am processing basin: {}, precipitation case: {}'.format(basin_name,precipitation_types[i]))

            ###############################################
            # NO RAIN
            csv_path = get_csv_name(DEM_paths[j], basin_name, 'no_rainfall', is_it_theta_best = False)
            ratio_ksn_no_rain_theta_45 = calculate_ksn_main_steam_tribs_ratio(csv_path)
            csv_path = get_csv_name(DEM_paths[j], basin_name, 'no_rainfall', is_it_theta_best = True)
            ratio_ksn_no_rain_theta_best = calculate_ksn_main_steam_tribs_ratio(csv_path)
            #print(ratio_ksn_no_rain_theta_45, ratio_ksn_no_rain_theta_best)
            #
            # ################################################
            # # ORIGINAL RAIN
            csv_path = get_csv_name(DEM_paths[j], basin_name, precipitation_types[i], is_it_theta_best = False)
            ratio_ksn_rain_theta_45 = calculate_ksn_main_steam_tribs_ratio(csv_path)
            csv_path = get_csv_name(DEM_paths[j], basin_name, precipitation_types[i], is_it_theta_best = True)
            ratio_ksn_rain_theta_best = calculate_ksn_main_steam_tribs_ratio(csv_path)
            #print(ratio_ksn_rain_theta_45, ratio_ksn_rain_theta_best)

            ratio_diff_rainfall_theta_0_45 = ratio_ksn_no_rain_theta_45/ratio_ksn_rain_theta_45
            ratio_diff_theta_no_rain = ratio_ksn_no_rain_theta_best/ratio_ksn_no_rain_theta_45
            ratio_diff_theta_rain = ratio_ksn_rain_theta_best/ratio_ksn_rain_theta_45
            ratio_diff_rainfall_theta_best = ratio_ksn_no_rain_theta_best/ratio_ksn_rain_theta_best


            print(f'Ratio of ksn for theta 0.45 for diff rainfall scenarios: {ratio_diff_rainfall_theta_0_45}')
            print(f'Ratio of ksn for theta best for diff rainfall scenarios: {ratio_diff_rainfall_theta_best}')
            print(f'Ratio of ksn for no rainfall scenarios (between theta = 0.45 and theta=theta_best): {ratio_diff_theta_no_rain}')
            print(f'Ratio of ksn for rainfall scenarios (between theta = 0.45 and theta=theta_best): {ratio_diff_theta_rain}')



            ksn_ratios_list.append([mountain_range_names[j], ratio_diff_rainfall_theta_0_45,ratio_diff_rainfall_theta_best,ratio_diff_theta_rain,ratio_diff_theta_no_rain])

ksn_ratios_df = pd.DataFrame(ksn_ratios_list, columns = ['mountain', 'ksn_ratio_theta_45', 'ksn_ratio_theta_best','ksn_ratio_rain', 'ksn_ratio_no_rain'])
ksn_ratios_df.to_csv(base_path+'ksn_ratios_0_01.csv', index=False)



        # print(all_errors_disorder)#(base_path, basin_name, data, x_axis_label,theta_or_disorder, rainfall_case, what_color):
        # percentage_disorder_w_rain = ((np.sum(np.array(all_errors_disorder)>0))*100)/len(all_errors_disorder)
        # print(f'{percentage_disorder_w_rain}%')
