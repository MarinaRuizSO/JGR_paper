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
#from functions_statistics import *
import helplotlib
import re
import glob
from matplotlib.ticker import StrMethodFormatter
from functions_statistics_gradients import *




# Load the disorder data
# We need to carry out the analysis in pairs.

base_path = '/exports/csce/datastore/geos/users/s1440040/projects/phd-fastscape/phd-fastscape/model_outputs_for_statistics/'
ss_names = ['ss_da','ss_discharge_grad_1','ss_discharge_grad_1', 'ss_discharge_grad_2' ,'ss_discharge_grad_3', 'ss_discharge_grad_4','ss_discharge_grad_5', 'ss_discharge_grad_6','ss_discharge_grad_7', 'ss_discharge_grad_8','ss_discharge_grad_9', 'ss_discharge_grad_10']
#chi_cases = ['grad_1', 'grad_2' ,'grad_3', 'grad_4','grad_5', 'grad_6','grad_7', 'grad_8','grad_9', 'grad_10']
DEM_paths = [base_path+'ss_da_gradient/', base_path+'ss_discharge_gradient/']# complete the list later - first try with just two cases

gradient_values = [0,1,2,3,4,5,6,7,8,9,10]
ss_case = ['da','discharge_grad_1','discharge_grad_1', 'discharge_grad_2' ,'discharge_grad_3', 'discharge_grad_4','discharge_grad_5', 'discharge_grad_6','discharge_grad_7', 'discharge_grad_8','discharge_grad_9', 'discharge_grad_10']


count = 0 # for the ss_names

def get_gradient_filelist(gradient_value, ss_case, file_path):
    filelist = []
    for file in glob.glob(f'{file_path}*ss_{ss_case}_chi_{gradient_value}__chi_map_theta_0_45.csv'):
        filelist.append(file)
    return filelist



for f_name in DEM_paths:
    print(f_name)
    ksn_ratios_list = []


    #for s in range(ss_names):
    count_chi_grads = 0
    for grad in gradient_values:

        f_list = list_outlets_in_mountain(f_name, f'grad_{grad}')
        print(f_list)
        for i in range(len(f_list)):
            print(f'count ss: {count}')
            print(f'count chi: {count_chi_grads}')
            csv_file_name = f_list[i]
            basin_name = get_basin_name(csv_file_name)

            ###############################################
            #CHI NO RAIN
            print('Chi no rain')
            print(basin_name)
            # for theta = 0.45
            csv_path = get_csv_name_litho(f_name, basin_name, 'da', is_it_theta_best=False)
            ratio_ksn_no_rain_theta_45 = calculate_ksn_main_steam_tribs_ratio(csv_path)

            # for theta=theta_best
            csv_path = get_csv_name_litho(f_name, basin_name, 'da', is_it_theta_best=True)
            ratio_ksn_no_rain_theta_best = calculate_ksn_main_steam_tribs_ratio(csv_path)
            #print(ratio_ksn_no_rain_theta_45, ratio_ksn_no_rain_theta_best)

            ################################################
            # CHI RAIN
            print('Chi WITH rain')
            print(basin_name)
            # for theta = 0.45
            csv_path = get_csv_name_litho(f_name, basin_name, f'grad_{grad}', is_it_theta_best=False)
            ratio_ksn_rain_theta_45 = calculate_ksn_main_steam_tribs_ratio(csv_path)

            # for theta=theta_best
            csv_path = get_csv_name_litho(f_name, basin_name, f'grad_{grad}', is_it_theta_best=True)
            ratio_ksn_rain_theta_best = calculate_ksn_main_steam_tribs_ratio(csv_path)
            #print(ratio_ksn_rain_theta_45, ratio_ksn_rain_theta_best)
            ###################################################
            # RATIOS

            ratio_diff_rainfall_theta_0_45 = ratio_ksn_no_rain_theta_45/ratio_ksn_rain_theta_45
            ratio_diff_theta_no_rain = ratio_ksn_no_rain_theta_best/ratio_ksn_no_rain_theta_45
            ratio_diff_theta_rain = ratio_ksn_rain_theta_best/ratio_ksn_rain_theta_45
            ratio_diff_rainfall_theta_best = ratio_ksn_no_rain_theta_best/ratio_ksn_rain_theta_best

            print(f'Ratio of ksn for theta 0.45 for diff rainfall scenarios: {ratio_diff_rainfall_theta_0_45}')
            print(f'Ratio of ksn for theta best for diff rainfall scenarios: {ratio_diff_rainfall_theta_best}')
            print(f'Ratio of ksn for no rainfall scenarios (between theta = 0.45 and theta=theta_best): {ratio_diff_theta_no_rain}')
            print(f'Ratio of ksn for rainfall scenarios (between theta = 0.45 and theta=theta_best): {ratio_diff_theta_rain}')

            ksn_ratios_list.append([grad, ratio_diff_rainfall_theta_0_45,ratio_diff_rainfall_theta_best,ratio_diff_theta_rain,ratio_diff_theta_no_rain])

        count_chi_grads+=1



    ksn_ratios_df = pd.DataFrame(ksn_ratios_list, columns = ['rain_gradient', 'ksn_ratio_theta_45', 'ksn_ratio_theta_best','ksn_ratio_rain', 'ksn_ratio_no_rain'])
    if count == 0:
        ksn_ratios_df.to_csv(base_path+f'ksn_ratios_all_grad_ss_da.csv', index=False)
    else:
        ksn_ratios_df.to_csv(base_path+f'ksn_ratios_all_grad_ss_discharge.csv', index=False)
    count +=1


