'''
File to do the statistical analysis of the disorder metrics and
the optimised theta value obtained from it.
Plot histograms for each study site separately

Marina Ruiz Sanchez-Oro
16/02/2024
'''
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from numpy import median

import pandas as pd
from matplotlib.offsetbox import AnchoredText
#from functions_statistics import *
import re
import glob
from functions_statistics_gradients import list_outlets_in_mountain, get_basin_name, get_csv_name_litho




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

def obtain_basin_direction(basin_df):
    """calculates whether the basin is draining to the east or the west"""
    domain_length_x = 15000 # legth of the domain in m
    mean_x_positions = np.mean(basin_df['x'])
    if mean_x_positions > (domain_length_x/2):
        east_or_west = 'east'
    else:
        east_or_west = 'west'
    return east_or_west 




for f_name in DEM_paths:
    print(f_name)
    east_west_list = []


    #for s in range(ss_names):
    count_chi_grads = 0
    for grad in gradient_values:

        f_list = list_outlets_in_mountain(f_name, f'grad_{grad}')
        print(f_list)
        for i in range(len(f_list)):
            print(f'count ss: {count}')
            print(f'count chi: {count_chi_grads}')
            csv_file_name = f_list[i].split('__')[0]+'__chi_map_theta_0_45.csv'
            
            basin_name = get_basin_name(csv_file_name)
            basin_number = int(re.findall(r'\d+', basin_name)[0])
            basin_df = pd.read_csv(csv_file_name)
            east_or_west = obtain_basin_direction(basin_df)

            ###################################################

            print(f'Basin {basin_number}, draining to the {east_or_west}')

            east_west_list.append([basin_number, east_or_west])

        count_chi_grads+=1



    east_west_df = pd.DataFrame(east_west_list, columns = ['basin_number', 'east_or_west'])
    if count == 0:
        east_west_df.to_csv(base_path+f'east_or_west_ss_da.csv', index=False)
    else:
        east_west_df.to_csv(base_path+f'east_or_west_ss_discharge.csv', index=False)
    count +=1


