
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import seaborn as sns
import pandas as pd
import math
import cmcrameri.cm as cmc
import rasterio
import re
from matplotlib.offsetbox import AnchoredText
import pickle
import itertools
import helplotlib
from matplotlib.ticker import StrMethodFormatter
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


def load_data(base_path, mountain, rain):
    with open(base_path + mountain + '_' + rain + '__' + 'all_disorder.pkl', 'rb') as f:
        all_disorder= pickle.load(f) # this is a dictionary with only one key
    with open(base_path  + mountain + '_' + rain + '__' + 'median_theta.pkl', 'rb') as f:
        median_theta = pickle.load(f)
    with open(base_path + mountain + '_' + rain + '__' + 'results_theta.pkl', 'rb') as f:
        results_theta = pickle.load(f)
    with open(base_path + mountain + '_' + rain + '__' + 'theta_range.pkl', 'rb') as f:
        theta_range = pickle.load(f)

    return all_disorder, median_theta, results_theta, theta_range





def calculate_relative_error(value_a, value_b):
    # this could be in x or in y.
    # in this case, the x coordinate is theta, and the y coordinate is disorder
    relative_error = (value_a - value_b)/value_b

    return relative_error
def error_propagation(error_a, error_b):
    prop_error = np.sqrt(((error_a)**2)+(error_b)**2)
    return prop_error

def calculate_absolute_error(value_a, value_b):
    absolute_error = (value_a - value_b)
    return absolute_error

def get_basin_name(path):
    head, tail = os.path.split(path)
    basin_name = tail.split('__')
    return basin_name[0]

def correct_theta_values(median_D):
    new_theta_list = np.arange(0.05,1,0.025)
    median_theta = round(min(median_D),2)
    min_value = min(median_D)
    min_index=list(median_D).index(min_value)
    return min_index, new_theta_list

def format_string(string_to_format):
    formatted_string = string_to_format.replace("_", " ")
    formatted_string = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", formatted_string).strip()
    return formatted_string.capitalize()

def count_outlets_in_mountain(path):
    #print(f'{path}*outlet_csv_latlon.csv')
    filelist = glob.glob(f'{path}*outlet_csv_latlon.csv')
    return filelist


def calculate_all_disorder_data(all_disorder, results_theta):
    theta_vals = np.array(list(all_disorder.values()))
    max_per_row = np.max(results_theta, axis = 1)
    results_norm = results_theta/max_per_row[:,None]
    median_D = np.median(results_norm[:,:], axis = 0)
    first_q_D = np.percentile(results_norm[:,:], 25, axis = 0)
    third_q_D = np.percentile(results_norm[:,:], 75, axis = 0)
    stdev_D = np.std(results_norm[:,:], axis = 0)
    #########################
    return theta_vals, max_per_row, results_norm, median_D


def calculate_all_relative_errors(DEM_paths, mountain_range_names, precipitation_types):
    counter = 0
    all_relative_errors_theta = []
    all_relative_errors_disorder = []
    for i in range(len(mountain_range_names)):
        filelist = count_outlets_in_mountain(DEM_paths[i])
        for k in range(len(filelist)):
            csv_file_name = filelist[k]
            basin = get_basin_name(csv_file_name)
            print(f'basin {basin}')
            all_disorder_no_rain, median_theta_no_rain, results_theta_no_rain, theta_range_no_rain = load_data(DEM_paths[i], mountain_range_names[i], basin, 'no_rainfall')
            theta_vals_no_rain, max_per_row_no_rain, results_norm_no_rain, median_D_no_rain = calculate_all_disorder_data(all_disorder_no_rain,results_theta_no_rain)
            min_index_no_rain, new_theta_list_no_rain = correct_theta_values(median_D_no_rain)
            value_median_theta_no_rain = new_theta_list_no_rain[min_index_no_rain]
            value_median_disorder_no_rain = median_D_no_rain[min_index_no_rain]
            for j in range(len(precipitation_types)):
                print(f'Calculating relative error for {mountain_range_names[i]}, basin {basin}, precipitation {precipitation_types[j]}')
                all_disorder_rain, median_theta_rain, results_theta_rain, theta_range_rain = load_data(DEM_paths[i], mountain_range_names[i], basin, precipitation_types[j])
                theta_vals_rain, max_per_row_rain, results_norm_rain, median_D_rain = calculate_all_disorder_data(all_disorder_rain,results_theta_rain)
                min_index_rain, new_theta_list_rain = correct_theta_values(median_D_rain)
                value_median_theta_rain = new_theta_list_rain[min_index_rain]
                value_median_disorder_rain = median_D_rain[min_index_rain]

                rel_error_theta = calculate_relative_error(value_median_theta_no_rain, value_median_theta_rain)
                rel_error_disorder = calculate_relative_error(value_median_disorder_no_rain, value_median_disorder_rain)
                all_relative_errors_theta.append(rel_error_theta)
                all_relative_errors_disorder.append(rel_error_disorder)
    return new_theta_list_rain, all_relative_errors_theta, all_relative_errors_disorder, value_median_theta_no_rain,value_median_disorder_no_rain,value_median_theta_rain,value_median_disorder_rain

def calculate_all_relative_errors_per_precip_type(DEM_paths, mountain_range_names, precipitation_types):
    for i in range(len(precipitation_types)):
        counter = 0
        all_relative_errors_theta = []
        all_relative_errors_disorder = []

        all_disorder_no_rain, median_theta_no_rain, results_theta_no_rain, theta_range_no_rain = load_data(DEM_paths[i], mountain_range_names[i],basin, 'no_rainfall')
        theta_vals_no_rain, max_per_row_no_rain, results_norm_no_rain, median_D_no_rain = calculate_all_disorder_data(all_disorder_no_rain,results_theta_no_rain)
        min_index_no_rain, new_theta_list_no_rain = correct_theta_values(median_D_no_rain)
        value_median_theta_no_rain = new_theta_list_no_rain[min_index_no_rain]
        value_median_disorder_no_rain = median_D_no_rain[min_index_no_rain]
        # for j in range(len(mountain_range_names)):
        #     for k in range(len(filelist)):
        plot_histogram(all_relative_errors_theta, r'Relative Error in $\theta_{best}$')
        plt.clf()



    #return theta_list, all_relative_errors_theta, all_relative_errors_disorder, value_median_theta_no_rain,value_median_disorder_no_rain,value_median_theta_rain,value_median_disorder_rain
        text_to_display = f'{rainfall_string}, '+ r'$\theta_{best}=%.2f$' % (theta_value, )

def get_csv_name(base_path, mountain, rain, is_it_theta_best=False):
    if is_it_theta_best == True:
        csv_path = base_path + mountain + '_' + rain + '__' + 'chi_map_theta_best*.csv'
    else:
        csv_path = base_path + mountain + '_' + rain + '__' + 'chi_map_theta_0_45.csv'
    #print(csv_path)

    return csv_path

def plot_ksn_ratios(csv_file_name, ):




    return

def pd_read_pattern(pattern):
    files = glob.glob(pattern)

    df = pd.DataFrame()
    for f in files:
        df = df.append(pd.read_csv(f))

    return df.reset_index(drop=True)

def calculate_ksn_main_steam_tribs_ratio(file_name):

    initial_df = pd_read_pattern(file_name)
    # 2) Normalise Drainage Area - add extra column

    norm_drainage_area = initial_df['drainage_area']/initial_df['drainage_area'].max()
    initial_df['norm_drainage_area'] = norm_drainage_area

    # 3) Normalise k_sn -  add extra column

    norm_m_chi = initial_df['m_chi']/initial_df['m_chi'].max()
    initial_df['norm_m_chi'] = norm_m_chi

    # Basins less than 0.1DA - remove
    df_thresh = initial_df[initial_df['norm_drainage_area']>0.01]

    # split the dataframe into 2 sections - 1) the main steam and larger tributaries, 2) smaller side tributaries

    df_large_t = df_thresh[df_thresh['norm_drainage_area']>=0.8]
    df_small_t = df_thresh[df_thresh['norm_drainage_area']<0.3]

    ksn_large = df_large_t['norm_m_chi']
    ksn_small = df_small_t['norm_m_chi']

    ratio_ksn = np.median(ksn_large)/np.median(ksn_small)
    return ratio_ksn




def calculate_errors_data(all_disorder, results_theta):
    theta_vals = np.array(list(all_disorder.values()))
    max_per_row = np.max(results_theta, axis = 1)
    results_norm = results_theta/max_per_row[:,None]
    median_D = np.median(results_norm[:,:], axis = 0)
    first_q_D = np.percentile(results_norm[:,:], 25, axis = 0)
    third_q_D = np.percentile(results_norm[:,:], 75, axis = 0)
    stdev_D = np.std(results_norm[:,:], axis = 0)
    #########################
    return first_q_D, third_q_D, stdev_D

def plot_histogram_per_basin(base_path, basin_name, data, percentage_disorder, x_axis_label,theta_or_disorder, rainfall_case, what_color):
    fig, ax = helplotlib.mkfig_simple_bold(fontsize_major = 21, fontsize_minor= 14, family = "DejaVu Sans" , figsize = (7,5))
    print(data)
    #n_bins = len(np.arange(min(data), max(data), (max(data)-min(data))/10))
    y,x,_=plt.hist(data,bins = 10, color=what_color)
    #x_tick_values = np.arange(min(data), max(data), (max(data)-min(data))/10)
    #plt.xticks(x_tick_values)
    plt.xlabel(x_axis_label)
    plt.ylabel('# counts')

    props_median = dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='gray')
    plt.axvline(x=np.median(data), linestyle='--', color='gray')
    ax.margins(x=0)
    rainfall_string = rainfall_case.replace("_", " ")
    rainfall_string = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", rainfall_string).strip()
    rainfall_string = rainfall_string.capitalize()
    text_to_display = f'{rainfall_string}'
    if theta_or_disorder == 'theta':
        variable = r'\theta'
        median_value = np.round(np.median(data), 3)
        txt_median = f'$\Delta{{{variable}}}_{{median}} = {{{median_value}}} $'
        if median_value > 0:
            greater_or_less = f'$ {{{variable}}}_{{rain}}<{{{variable}}}_{{no\:rain}}$'
        elif median_value == 0:
            greater_or_less = f'$ {{{variable}}}_{{rain}}={{{variable}}}_{{no\:rain}}$'
        else:
            greater_or_less = f'$ {{{variable}}}_{{rain}}>{{{variable}}}_{{no\:rain}}$'
        math_txt = txt_median +'\n'+greater_or_less

    elif theta_or_disorder == 'disorder':
        variable = 'D'
        median_value = np.round(np.median(data), 3)
        txt_median = f'$\Delta{{{variable}}}_{{median}} = {{{median_value}}} $'
        if median_value > 0:
            greater_or_less = f'$ {{{variable}}}_{{rain}}<{{{variable}}}_{{no\:rain}}$'
        elif median_value == 0:
            greater_or_less = f'$ {{{variable}}}_{{rain}}={{{variable}}}_{{no\:rain}}$'
        else:
            greater_or_less = f'$ {{{variable}}}_{{rain}}>{{{variable}}}_{{no\:rain}}$'
        math_txt = txt_median +'\n'+greater_or_less

    elif theta_or_disorder == 'disorder_first_derivative':
        variable = 'D^{\prime}'
        median_value = np.round(np.median(data), 3)
        txt_median = f'$\Delta{{{variable}}}_{{median}} = {{{median_value}}} $'
        if median_value > 0:
            greater_or_less = f'$ {{{variable}}}_{{rain}}<{{{variable}}}_{{no\:rain}}$'
        elif median_value == 0:
            greater_or_less = f'$ {{{variable}}}_{{rain}}={{{variable}}}_{{no\:rain}}$'
        else:
            greater_or_less = f'$ {{{variable}}}_{{rain}}>{{{variable}}}_{{no\:rain}}$'
        math_txt = txt_median +'\n'+greater_or_less
    else:
        variable = 'D^{\prime\prime}'
        median_value = np.round(np.median(data), 3)
        txt_median = f'$\Delta{{{variable}}}_{{median}} = {{{median_value}}} $'
        if median_value > 0:
            greater_or_less = f'$ {{{variable}}}_{{rain}}<{{{variable}}}_{{no\:rain}}$'
        elif median_value == 0:
            greater_or_less = f'$ {{{variable}}}_{{rain}}={{{variable}}}_{{no\:rain}}$'
        else:
            greater_or_less = f'$ {{{variable}}}_{{rain}}>{{{variable}}}_{{no\:rain}}$'
        math_txt = txt_median +'\n'+greater_or_less

    if x.min()<=0 and x.max()<=0:
        # case when min and max both have the same sign
        plt.axvspan(x.min(), x.max(), facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100)
    elif x.min()>=0 and x.max()>=0:
        plt.axvspan(x.min(), x.max(), facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100)
    else:
        plt.axvspan(x.min(), 0.000, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100)
        plt.axvspan(0.000, x.max(), facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100)


    plt.figtext(0.5,1, f"Lower disorder with rain: {percentage_disorder}% of basins ", ha="center", va="center")
    basin_string = basin_name.replace("_", " ")
    basin_string = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", basin_string).strip()
    basin_string = basin_string.capitalize()

    ax.scatter([],[], label = basin_string.upper() + '\n' + text_to_display + '\n' + math_txt, alpha = 0)
    ax.legend(handlelength=0,handletextpad=0, facecolor='white', edgecolor='black')


    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}')) # No decimal places
    #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    plt.tight_layout()
    plt.savefig(f'{base_path}/statistics_results/{basin_name}_{rainfall_case}__{theta_or_disorder}_histogram_rel_errors.png', dpi = 300, bbox_inches="tight")
    plt.clf()
    plt.close(fig)
    #plt.show()

def plot_histogram(base_path, data, x_axis_label,theta_or_disorder, rainfall_case, what_color):
    fig, ax = helplotlib.mkfig_simple_bold(fontsize_major = 21, fontsize_minor= 14, family = "DejaVu Sans" , figsize = (7,5))
    print(data)
    #n_bins = len(np.arange(min(data), max(data), (max(data)-min(data))/10))
    y,x,_=plt.hist(data,bins = 10, color=what_color)
    #x_tick_values = np.arange(min(data), max(data), (max(data)-min(data))/10)
    #plt.xticks(x_tick_values)
    plt.xlabel(x_axis_label)
    plt.ylabel('# counts')

    props_median = dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='gray')
    plt.axvline(x=np.median(data), linestyle='--', color='gray')
    ax.margins(x=0)
    rainfall_string = rainfall_case.replace("_", " ")
    rainfall_string = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", rainfall_string).strip()
    rainfall_string = rainfall_string.capitalize()
    text_to_display = f'{rainfall_string}'
    if theta_or_disorder == 'theta':
        variable = r'\theta'
        median_value = np.round(np.median(data), 3)
        txt_median = f'$\Delta{{{variable}}}_{{median}} = {{{median_value}}} $'
        if median_value > 0:
            greater_or_less = f'$ {{{variable}}}_{{rain}}<{{{variable}}}_{{no\:rain}}$'
        elif median_value == 0:
            greater_or_less = f'$ {{{variable}}}_{{rain}}={{{variable}}}_{{no\:rain}}$'
        else:
            greater_or_less = f'$ {{{variable}}}_{{rain}}>{{{variable}}}_{{no\:rain}}$'
        math_txt = txt_median +'\n'+greater_or_less

    elif theta_or_disorder == 'disorder':
        variable = 'D'
        median_value = np.round(np.median(data), 3)
        txt_median = f'$\Delta{{{variable}}}_{{median}} = {{{median_value}}} $'
        if median_value > 0:
            greater_or_less = f'$ {{{variable}}}_{{rain}}<{{{variable}}}_{{no\:rain}}$'
        elif median_value == 0:
            greater_or_less = f'$ {{{variable}}}_{{rain}}={{{variable}}}_{{no\:rain}}$'
        else:
            greater_or_less = f'$ {{{variable}}}_{{rain}}>{{{variable}}}_{{no\:rain}}$'
        math_txt = txt_median +'\n'+greater_or_less

    elif theta_or_disorder == 'disorder_first_derivative':
        variable = 'D^{\prime}'
        median_value = np.round(np.median(data), 3)
        txt_median = f'$\Delta{{{variable}}}_{{median}} = {{{median_value}}} $'
        if median_value > 0:
            greater_or_less = f'$ {{{variable}}}_{{rain}}<{{{variable}}}_{{no\:rain}}$'
        elif median_value == 0:
            greater_or_less = f'$ {{{variable}}}_{{rain}}={{{variable}}}_{{no\:rain}}$'
        else:
            greater_or_less = f'$ {{{variable}}}_{{rain}}>{{{variable}}}_{{no\:rain}}$'
        math_txt = txt_median +'\n'+greater_or_less
    else:
        variable = 'D^{\prime\prime}'
        median_value = np.round(np.median(data), 3)
        txt_median = f'$\Delta{{{variable}}}_{{median}} = {{{median_value}}} $'
        if median_value > 0:
            greater_or_less = f'$ {{{variable}}}_{{rain}}<{{{variable}}}_{{no\:rain}}$'
        elif median_value == 0:
            greater_or_less = f'$ {{{variable}}}_{{rain}}={{{variable}}}_{{no\:rain}}$'
        else:
            greater_or_less = f'$ {{{variable}}}_{{rain}}>{{{variable}}}_{{no\:rain}}$'
        math_txt = txt_median +'\n'+greater_or_less

    if x.min()<=0 and x.max()<=0:
        # case when min and max both have the same sign
        plt.axvspan(x.min(), x.max(), facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100)
    elif x.min()>=0 and x.max()>=0:
        plt.axvspan(x.min(), x.max(), facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100)
    else:
        plt.axvspan(x.min(), 0.000, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100)
        plt.axvspan(0.000, x.max(), facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100)



    ax.scatter([],[], label=text_to_display.upper()+'\n'+ math_txt, alpha = 0)
    ax.legend(handlelength=0,handletextpad=0, facecolor='white', edgecolor='black')

    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}')) # No decimal places
    #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    plt.tight_layout()
    plt.savefig(f'{base_path}/statistics_results/{rainfall_case}__{theta_or_disorder}_histogram_rel_errors.png', dpi = 300)
    plt.clf()
    plt.close(fig)
    #plt.show()

def plot_two_disorder_graphs(theta_list, data_1, data_2, label_1, label_2):
    plt.plot(theta_list,data_1, label=label_1)
    plt.plot(theta_list,data_2, label=label_2)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_histogram_stdev(base_path,basin_name, data, x_axis_label,theta_or_disorder, rainfall_case,legend_title,propagated_error,what_color):
    fig, ax = helplotlib.mkfig_simple_bold(fontsize_major = 21, fontsize_minor= 14, family = "DejaVu Sans" , figsize = (7,5))
    y,x,_=plt.hist(data,bins = 10, color=what_color)
    plt.xlabel(x_axis_label)
    plt.ylabel('Counts')
    median_value = np.round(np.median(data), 3)

    props_median = dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='gray')
    plt.axvline(x=median_value, linestyle='--', color='white')
    props_theta = dict(boxstyle='round', facecolor='white', alpha=1) #f'${{{x_axis_label}}}_{{median}} = {{{median_value}}}$'
    plt.text(median_value, (y.max()-2), x_axis_label+f'$_{{,\:median}} = {{{median_value}}}$', rotation=0, verticalalignment='center',
     bbox=props_theta, fontsize=10, fontweight='normal', horizontalalignment='right')
    plt.axvline(x=median_value+propagated_error, linestyle=':', color='black')
    plt.axvline(x=median_value-propagated_error, linestyle=':', color='black')

    ax.margins(x=0)
    rainfall_string = rainfall_case.replace("_", " ")
    rainfall_string = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", rainfall_string).strip()
    rainfall_string = rainfall_string.capitalize()
    text_to_display = f'{rainfall_string}'

    ss_or_chi = '\chi'
    if theta_or_disorder == 'theta':
        variable = r'\theta'
        txt_median = f'$\Delta{{{variable}}}_{{median}} = {{{median_value}}} $'
        if median_value > 0:
            greater_or_less = f'$ {{{variable}}}_{{{{{ss_or_chi}}}_{{rain}}}}<{{{variable}}}_{{{{{ss_or_chi}}}_{{no\:rain}}}}$'
        elif median_value == 0:
            greater_or_less = f'$ {{{variable}}}_{{{{{ss_or_chi}}}_{{rain}}}}={{{variable}}}_{{{{{ss_or_chi}}}_{{no\:rain}}}}$'
        else:
            greater_or_less = f'$ {{{variable}}}_{{{{{ss_or_chi}}}_{{rain}}}}>{{{variable}}}_{{{{{ss_or_chi}}}_{{no\:rain}}}}$'
        math_txt = '\n'+greater_or_less#txt_median +'\n'+greater_or_less

    elif theta_or_disorder == 'disorder':
        variable = 'D'
        txt_median = f'$\Delta{{{variable}}}_{{median}} = {{{median_value}}} $'
        if median_value > 0:
            greater_or_less = f'$ {{{variable}}}_{{{{{ss_or_chi}}}_{{rain}}}}<{{{variable}}}_{{{{{ss_or_chi}}}_{{no\:rain}}}}$'
        elif median_value == 0:
            greater_or_less = f'$ {{{variable}}}_{{{{{ss_or_chi}}}_{{rain}}}}={{{variable}}}_{{{{{ss_or_chi}}}_{{no\:rain}}}}$'
        else:
            greater_or_less = f'$ {{{variable}}}_{{{{{ss_or_chi}}}_{{rain}}}}>{{{variable}}}_{{{{{ss_or_chi}}}_{{no\:rain}}}}$'
        math_txt = '\n'+greater_or_less#txt_median +'\n'+greater_or_less

    elif theta_or_disorder == 'disorder_first_derivative':
        variable = 'D^{\prime}'
        txt_median = f'$\Delta{{{variable}}}_{{median}} = {{{median_value}}} $'
        if median_value > 0:
            greater_or_less = f'$ {{{variable}}}_{{{{{ss_or_chi}}}_{{rain}}}}<{{{variable}}}_{{{{{ss_or_chi}}}_{{no\:rain}}}}$'
        elif median_value == 0:
            greater_or_less = f'$ {{{variable}}}_{{{{{ss_or_chi}}}_{{rain}}}}={{{variable}}}_{{{{{ss_or_chi}}}_{{no\:rain}}}}$'
        else:
            greater_or_less = f'$ {{{variable}}}_{{{{{ss_or_chi}}}_{{rain}}}}>{{{variable}}}_{{{{{ss_or_chi}}}_{{no\:rain}}}}$'
        math_txt = '\n'+greater_or_less#txt_median +'\n'+greater_or_less
    else:
        variable = 'D^{\prime\prime}'
        txt_median = f'$\Delta{{{variable}}}_{{median}} = {{{median_value}}} $'
        if median_value > 0:
            greater_or_less = f'$ {{{variable}}}_{{{{{ss_or_chi}}}_{{rain}}}}<{{{variable}}}_{{{{{ss_or_chi}}}_{{no\:rain}}}}$'
        elif median_value == 0:
            greater_or_less = f'$ {{{variable}}}_{{{{{ss_or_chi}}}_{{rain}}}}={{{variable}}}_{{{{{ss_or_chi}}}_{{no\:rain}}}}$'
        else:
            greater_or_less = f'$ {{{variable}}}_{{{{{ss_or_chi}}}_{{rain}}}}>{{{variable}}}_{{{{{ss_or_chi}}}_{{no\:rain}}}}$'
        math_txt = '\n'+greater_or_less#txt_median +'\n'+greater_or_less
    x_min, x_max = x.min(), x.max()

    plt.axvspan(x.min()-propagated_error, 0.000, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100)
    plt.axvspan(median_value-propagated_error, median_value+propagated_error, facecolor=cmc.buda.colors[0], alpha=0, zorder=-100, hatch='//')

    plt.axvspan(0.000, x.max()+propagated_error, facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100)
    plt.axvspan(median_value-propagated_error, median_value+propagated_error, facecolor=cmc.buda.colors[-1], alpha=0, zorder=-100, hatch='//')




    #ax.scatter([],[], label=text_to_display.upper()+'\n'+ math_txt, alpha = 0)
    legend_elements = [Patch(facecolor='white', edgecolor='black',hatch='//',
                         label='Error in'+'$\:$'+ x_axis_label+f'$_{{,\:median}}$')]

    legend = ax.legend(handles=legend_elements,title = text_to_display.upper()+ math_txt,handlelength=2,handletextpad=0.2, facecolor='white', edgecolor='black')
    legend._legend_box.align = "left"
    legend.get_texts()[0].set_fontsize('small')

    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}')) # 3 decimal places
    plt.yticks(np.arange(min(y), max(y)+1, 1))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
    plt.xticks(rotation=45)
    #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    plt.tight_layout()
    #plt.show()
    #plt.savefig(f'{base_path}/{rainfall_case}__{theta_or_disorder}_{ss_or_chi}_histogram.png', dpi = 300)
    plt.savefig(f'{base_path}/statistics_results/{basin_name}_{rainfall_case}__{theta_or_disorder}_histogram_with_errors.png', dpi = 300)

    plt.clf()
    plt.close(fig)
    #plt.show()
