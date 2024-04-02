'''

Marina Ruiz Sanchez-Oro
16/02/2024
'''
import numpy as np

import os, glob

import pandas as pd
import glob

base_path = '/exports/csce/datastore/geos/users/s1440040/projects/phd-fastscape/phd-fastscape/model_outputs_for_statistics/'
ksn_statistics_file_da = pd.read_csv(base_path + 'ksn_ratios_all_grad_ss_da.csv')
ksn_statistics_file_discharge = pd.read_csv(base_path + 'ksn_ratios_all_grad_ss_discharge.csv')

east_west_file_da = pd.read_csv(base_path + 'east_or_west_ss_da.csv')
east_west_file_discharge = pd.read_csv(base_path + 'east_or_west_ss_discharge.csv')

# merge the files so that the east west column is swon in the ksn distortion files 

merged_da = pd.concat([ksn_statistics_file_da, east_west_file_da], axis=1)
merged_discharge = pd.concat([ksn_statistics_file_discharge, east_west_file_discharge], axis=1)


merged_da.to_csv(base_path+f'ksn_stats_with_east_or_west_ss_da.csv', index=False)
merged_discharge.to_csv(base_path+f'ksn_stats_with_east_or_west_ss_discharge.csv', index=False)