
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
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

def load_data(base_path, basin_name, chi_case):
    with open(base_path + basin_name + '_chi_'+chi_case+'__' + 'all_disorder.pkl', 'rb') as f:
        print(base_path + basin_name + '_chi_'+chi_case+'__' + 'all_disorder.pkl')
        all_disorder= pickle.load(f) # this is a dictionary with only one key
    with open(base_path + basin_name+'_chi_'+chi_case+ '__' + 'median_theta.pkl', 'rb') as f:
        median_theta = pickle.load(f)
    with open(base_path + basin_name +'_chi_'+chi_case+ '__' + 'results_theta.pkl', 'rb') as f:
        results_theta = pickle.load(f)
    with open(base_path + basin_name +'_chi_'+chi_case+ '__' + 'theta_range.pkl', 'rb') as f:
        theta_range = pickle.load(f)

    return all_disorder, median_theta, results_theta, theta_range

def load_data_litho(base_path, basin_name, chi_case, litho_case):
    with open(base_path + basin_name +'_chi_'+chi_case+'__' + 'all_disorder.pkl', 'rb') as f:
        print(base_path + basin_name + '_chi_'+chi_case+'__' + 'all_disorder.pkl')
        all_disorder= pickle.load(f) # this is a dictionary with only one key
    with open(base_path + basin_name+ '_chi_'+chi_case+ '__' + 'median_theta.pkl', 'rb') as f:
        median_theta = pickle.load(f)
    with open(base_path + basin_name +'_chi_'+chi_case+ '__' + 'results_theta.pkl', 'rb') as f:
        results_theta = pickle.load(f)
    with open(base_path + basin_name  +'_chi_'+chi_case+ '__' + 'theta_range.pkl', 'rb') as f:
        theta_range = pickle.load(f)

    return all_disorder, median_theta, results_theta, theta_range

def load_data_large_area(base_path, basin_name, chi_case):
    with open(base_path + basin_name  +'_chi_'+chi_case+ '_70x70__'+ 'all_disorder.pkl', 'rb') as f:
        print(base_path + basin_name  +'_chi_'+chi_case+ '_70x70__'+ 'all_disorder.pkl')
        all_disorder= pickle.load(f) # this is a dictionary with only one key
    with open(base_path + basin_name+'_chi_'+chi_case+ '_70x70__'+'median_theta.pkl', 'rb') as f:
        median_theta = pickle.load(f)
    with open(base_path + basin_name +'_chi_'+chi_case+ '_70x70__' + 'results_theta.pkl', 'rb') as f:
        results_theta = pickle.load(f)
    with open(base_path + basin_name +'_chi_'+chi_case+ '_70x70__' + 'theta_range.pkl', 'rb') as f:
        theta_range = pickle.load(f)

    return all_disorder, median_theta, results_theta, theta_range

def get_csv_name_litho(base_path, basin_name, chi_case, is_it_theta_best=False):
    if is_it_theta_best == True:
        csv_path = base_path + basin_name + '_chi_' + chi_case + '__' + 'chi_map_theta_best*.csv'
    else:
        csv_path = base_path + basin_name + '_chi_' + chi_case + '__' + 'chi_map_theta_0_45.csv'
    #print(csv_path)

    return csv_path

def get_csv_name_various_theta(base_path, basin_name, chi_case, theta_type):
    if theta_type == 'best':
        csv_path = base_path + basin_name + '_chi_' + chi_case + '__' + 'chi_map_theta_best*.csv'
    elif theta_type == '0.45':
        csv_path = base_path + basin_name + '_chi_' + chi_case + '__' + 'chi_map_theta_0_45.csv'
    else:
        csv_path = base_path + basin_name + '_chi_' + chi_case + '__' + f'chi_map_theta_{theta_type}.csv'

    return csv_path


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
    df_thresh = initial_df[initial_df['norm_drainage_area']>0.1]

    # split the dataframe into 2 sections - 1) the main steam , 2) smaller side tributaries
    # excluding any tributaries that fall in between those, for a clearer signal

    df_large_t = df_thresh[df_thresh['norm_drainage_area']>=0.8]
    df_small_t = df_thresh[df_thresh['norm_drainage_area']<0.3]

    ksn_large = df_large_t['norm_m_chi']
    ksn_small = df_small_t['norm_m_chi']

    ratio_ksn = np.median(ksn_large)/np.median(ksn_small)
    return ratio_ksn




def calculate_relative_error(value_a, value_b):
    # this could be in x or in y.
    # in this case, the x coordinate is theta, and the y coordinate is disorder
    relative_error = (value_a - value_b)/value_b

    return relative_error

def calculate_absolute_error(value_a, value_b):
    # this could be in x or in y.
    # in this case, the x coordinate is theta, and the y coordinate is disorder
    absolute_error = (value_a - value_b)

    return absolute_error

def get_basin_name(path):
    head, tail = os.path.split(path)
    basin_name = tail.split('_chi_')
    return basin_name[0]

def get_only_filename(path):
    filelist = glob.glob(f'{path}*{rain_case}__theta_range.pkl')
    head, tail = os.path.split(path)
    basin_name = tail.split('_chi_')
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

def list_outlets_in_mountain(path, rain_case):
    #print(f'{path}*outlet_csv_latlon.csv')
    filelist = glob.glob(f'{path}*{rain_case}__theta_range.pkl')
    return filelist

def list_outlets_in_mountain_large_area(path, rain_case):
    #print(f'{path}*outlet_csv_latlon.csv')
    filelist = glob.glob(f'{path}*{rain_case}_70x70__theta_range.pkl')
    return filelist

def find_case_files(path, case_to_find):
    my_case_list = glob.glob(f"{path}*_chi_{case_to_find}_*.pkl")
    return my_case_list

def find_number_of_basins(files):
    list_of_files = files
    def extract_number(f):
        s = re.findall("\d+$",f)
        return (int(s[0]) if s else -1,f)

    max_file = max(list_of_files,key=extract_number)
    max_file = os.path.basename(max_file)
    number_of_basins = re.findall(r'\d+', max_file)
    return number_of_basins

def find_number_of_basins_more_cases(files):
    number_of_basins = len(files)/2

    return int(number_of_basins)


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


def calculate_all_errors(DEM_paths, mountain_range_names, precipitation_types):
    counter = 0
    all_errors_theta = []
    all_errors_disorder = []
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

                error_theta = calculate_absolute_error(value_median_theta_no_rain, value_median_theta_rain)
                error_disorder = calculate_absolute_error(value_median_disorder_no_rain, value_median_disorder_rain)
                all_errors_theta.append(rel_error_theta)
                all_errors_disorder.append(rel_error_disorder)
    return new_theta_list_rain, all_errors_theta, all_errors_disorder, value_median_theta_no_rain,value_median_disorder_no_rain,value_median_theta_rain,value_median_disorder_rain

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
        plt.close()



    #return theta_list, all_relative_errors_theta, all_relative_errors_disorder, value_median_theta_no_rain,value_median_disorder_no_rain,value_median_theta_rain,value_median_disorder_rain


def plot_histogram_simple(base_path,data, x_axis_label,theta_or_disorder, rainfall_case, what_color):
    fig, ax = helplotlib.mkfig_simple_bold(fontsize_major = 21, fontsize_minor = 14, family = "DejaVu Sans" , figsize = (7,5))
    plt.hist(data,color=what_color)
    plt.xlabel(x_axis_label)
    plt.ylabel('Frequency')
    plt.tight_layout()
    median_value = np.round(np.median(data), 3)

    ax.axvline(x=median_value, linestyle='--', color='gray')

    rainfall_string = rainfall_case.replace("_", " ")
    rainfall_string = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", rainfall_string).strip()
    rainfall_string = rainfall_string.capitalize()
    text_to_display = f'{rainfall_string}'
    ax.scatter([],[], label=text_to_display, alpha = 0)
    ax.legend(handlelength=0,handletextpad=0, facecolor='wheat')
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # No decimal places
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f'{base_path}/{rainfall_case}__{theta_or_disorder}_histogram.png', dpi = 300)
    plt.clf()
    plt.close(fig)
    #plt.show()

def plot_two_disorder_graphs(theta_list, data_1, data_2, label_1, label_2):
    plt.plot(theta_list,data_1, label=label_1)
    plt.plot(theta_list,data_2, label=label_2)
    plt.legend()
    plt.tight_layout()
    plt.show()

def error_propagation(error_a, error_b):
    prop_error = np.sqrt(((error_a)**2)+(error_b)**2)
    return prop_error

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

def plot_histogram(base_path,data, x_axis_label,theta_or_disorder, rainfall_case, all_ss_or_chi,legend_title,propagated_error,what_color):
    fig, ax = helplotlib.mkfig_simple_bold(fontsize_major = 21, fontsize_minor= 14, family = "DejaVu Sans" , figsize = (7,5))
    y,x,_=plt.hist(data,bins = 10, color=what_color)
    plt.xlabel(x_axis_label)
    plt.ylabel('Counts')
    median_value = np.round(np.median(data), 3)

    props_median = dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='gray')
    plt.axvline(x=median_value, linestyle='--', color='white')
    props_theta = dict(boxstyle='round', facecolor='white', alpha=1) #f'${{{x_axis_label}}}_{{median}} = {{{median_value}}}$'
    plt.text(median_value, (y.max()-1), x_axis_label+f'$_{{,\:median}} = {{{median_value}}}$', rotation=90, verticalalignment='center',
     bbox=props_theta, fontsize=10, fontweight='normal', horizontalalignment='center')
    plt.axvline(x=median_value+propagated_error, linestyle=':', color='black')
    plt.axvline(x=median_value-propagated_error, linestyle=':', color='black')

    ax.margins(x=0)
    rainfall_string = rainfall_case.replace("_", " ")
    rainfall_string = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", rainfall_string).strip()
    rainfall_string = rainfall_string.capitalize()
    text_to_display = f'{rainfall_string}'
    if all_ss_or_chi == 'all_chi':
        ss_or_chi = 'ss'
    else:
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
    if x.min()<=0 and x.max()<=0:
        # case when min and max both have the same sign
        if (x.min()>(median_value-propagated_error)):
            x_min = median_value-propagated_error
        if (x.max()<(median_value+propagated_error)):
            x_max = median_value+propagated_error
        if x_max >=0:
            plt.axvspan(x_min, 0.000, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100)
            plt.axvspan(median_value-propagated_error, median_value+propagated_error, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100, hatch='//', label='Error in'+ x_axis_label+f'$_{{,\:median}} = {{{median_value}}}$')
            plt.axvspan(0.000, x_max, facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100)
            plt.axvspan(median_value-propagated_error, median_value+propagated_error, facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100, hatch='//', label='Error in'+ x_axis_label+f'$_{{,\:median}} = {{{median_value}}}$')
        else:
            plt.axvspan(x_min, 0, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100)
            plt.axvspan(median_value-propagated_error, median_value+propagated_error, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100, hatch='//', label='Error in'+ x_axis_label+f'$_{{,\:median}} = {{{median_value}}}$')
    elif x.min()>=0 and x.max()>=0:
        if (x.min()>(median_value-propagated_error)):
            x_min = median_value-propagated_error
        if (x.max()<(median_value+propagated_error)):
            x_max = median_value+propagated_error
        if x_min<=0:
            plt.axvspan(x_min, 0.000, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100)
            plt.axvspan(median_value-propagated_error, median_value+propagated_error, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100, hatch='//')
            plt.axvspan(0.000, x_max, facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100)
            plt.axvspan(median_value-propagated_error, median_value+propagated_error, facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100, hatch='//')

        else:
            plt.axvspan(0, x_max, facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100)
            plt.axvspan(median_value-propagated_error, median_value+propagated_error, facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100, hatch='//')
    else:
        plt.axvspan(x.min()-propagated_error, 0.000, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100)
        plt.axvspan(median_value-propagated_error, median_value+propagated_error, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100, hatch='//')

        plt.axvspan(0.000, x.max()+propagated_error, facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100)
        plt.axvspan(median_value-propagated_error, median_value+propagated_error, facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100, hatch='//')





    legend_elements = [Patch(facecolor='white', edgecolor='black',hatch='//',
                         label='Error in'+'$\:$'+ x_axis_label+f'$_{{,\:median}}$')]

    legend = ax.legend(handles=legend_elements,title = legend_title+ math_txt,handlelength=2,handletextpad=0.2, facecolor='white', edgecolor='black')
    legend._legend_box.align = "left"
    legend.get_texts()[0].set_fontsize('small')
    #
    # ax.scatter([],[], label=legend_title+ math_txt, alpha = 0)
    # ax.legend(handlelength=0,handletextpad=1, facecolor='white', edgecolor='black')
    #
    # legend1 = ax.legend(handlelength=0,handletextpad=1, facecolor='white', edgecolor='black')
    # legend2 = ax.legend(handles=legend_elements,handlelength=1,handletextpad=1, facecolor='white', edgecolor='black')
    # ax.add_artist(legend1)
    # ax.add_artist(legend2)

    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}')) # No decimal places
    plt.yticks(np.arange(min(y), max(y)+1, 1))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
    plt.xticks(rotation=45)
    #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    plt.tight_layout()
    plt.show()
    #plt.savefig(f'{base_path}/{rainfall_case}__{theta_or_disorder}_{all_ss_or_chi}_histogram.png', dpi = 300)
    plt.clf()
    plt.close(fig)
    #plt.show()



def plot_histogram_single_case(base_path,data, x_axis_label,theta_or_disorder, rainfall_case,chi_case,legend_title,median_error,what_color):
    fig, ax = helplotlib.mkfig_simple_bold(fontsize_major = 21, fontsize_minor= 14, family = "DejaVu Sans" , figsize = (7,5))
    y,x,_=plt.hist(data,bins = 10, color=what_color)
    plt.xlabel(x_axis_label)
    plt.ylabel('Frequency')
    median_value = np.round(np.median(data), 3)
    print(median_value,median_error, median_value+median_error, median_value-median_error)

    props_median = dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='gray')
    plt.axvline(x=median_value, linestyle='--', color='gray')
    plt.axvline(x = median_value+median_error, linestyle=':', color='black')
    plt.axvline(x = median_value-median_error, linestyle=':', color='black')
    ax.margins(x=0)
    rainfall_string = rainfall_case.replace("_", " ")
    rainfall_string = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", rainfall_string).strip()
    rainfall_string = rainfall_string.capitalize()
    text_to_display = f'{rainfall_string}'

    if theta_or_disorder == 'disorder':
        variable = r'D'
        #median_value = np.round(np.median(data), 5)
        txt_median = f'${{{variable}}}_{{median}} = {{{median_value}}} $'
        math_txt = txt_median
    if chi_case =='chi_rain':
        chi_txt = '$,\: \chi_{rain}$'

    else:
        chi_txt = '$,\: \chi_{no\:rain}$'

    if x.min()<=0 and x.max()<=0:
        # case when min and max both have the same sign
        plt.axvspan(x.min(), x.max(), facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100)
    elif x.min()>=0 and x.max()>=0:
        plt.axvspan(x.min(), x.max(), facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100)
    else:
        plt.axvspan(x.min(), 0.000, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100)
        plt.axvspan(0.000, x.max(), facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100)

    ax.scatter([],[], label=legend_title+chi_txt+'\n'+ math_txt, alpha = 0)
    ax.legend(handlelength=0,handletextpad=0, facecolor='white', edgecolor='black')
    plt.ticklabel_format(style='sci', axis='x')
    plt.yticks(np.arange(min(y), max(y)+1, 1))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places

    #plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}')) # No decimal places
    plt.xticks(rotation=45)
    #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    plt.tight_layout()
    plt.savefig(f'{base_path}/{rainfall_case}__{theta_or_disorder}_{legend_title}_{chi_case}_histogram.png', dpi = 300)
    plt.clf()
    plt.close(fig)
    #plt.show()

def plot_histogram_litho(base_path,data, x_axis_label,theta_or_disorder, rainfall_case, all_ss_or_chi,legend_title,what_color, litho):
    fig, ax = helplotlib.mkfig_simple_bold(fontsize_major = 21, fontsize_minor= 14, family = "DejaVu Sans" , figsize = (7,5))
    y,x,_=plt.hist(data,bins = 10, color=what_color)
    plt.xlabel(x_axis_label)
    plt.ylabel('Counts')
    median_value = np.round(np.median(data), 3)

    props_median = dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='gray')
    plt.axvline(x=median_value, linestyle='--', color='white')
    props_theta = dict(boxstyle='round', facecolor='white', alpha=1) #f'${{{x_axis_label}}}_{{median}} = {{{median_value}}}$'
    plt.text(median_value, (y.max()-1), x_axis_label+f'$_{{,\:median}} = {{{median_value}}}$', rotation=90, verticalalignment='center',
     bbox=props_theta, fontsize=10, fontweight='normal', horizontalalignment='center')
    #plt.axvline(x=median_value+propagated_error, linestyle=':', color='black')
    #plt.axvline(x=median_value-propagated_error, linestyle=':', color='black')

    ax.margins(x=0)
    rainfall_string = rainfall_case.replace("_", " ")
    rainfall_string = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", rainfall_string).strip()
    rainfall_string = rainfall_string.capitalize()
    text_to_display = f'{rainfall_string}'
    if all_ss_or_chi == 'all_chi':
        ss_or_chi = 'ss'
    else:
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

    plt.axvspan(x.min(), 0.000, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100)

    plt.axvspan(0.000, x.max(), facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100)

    legend = ax.legend(title = legend_title+'\n'+litho.title()+' Lithology'+ math_txt,handlelength=2,handletextpad=0.2, facecolor='white', edgecolor='black')
    legend._legend_box.align = "left"
    #legend.get_texts()[0].set_fontsize('small')

    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}')) # No decimal places
    plt.yticks(np.arange(min(y), max(y)+1, 1))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{base_path}{rainfall_case}__{theta_or_disorder}_{all_ss_or_chi}_{litho}_K_histogram.png', dpi = 300)
    plt.clf()
    plt.close(fig)


def plot_two_histograms(ax, base_path,data, x_axis_label,theta_or_disorder, rainfall_case, all_ss_or_chi,legend_title,propagated_error,what_color):
    #fig, ax = helplotlib.mkfig_simple_bold(fontsize_major = 21, fontsize_minor= 14, family = "DejaVu Sans" , figsize = (7,5))
    y,x,_=plt.hist(data,bins = 10, color=what_color)
    plt.xlabel(x_axis_label)
    plt.ylabel('Counts')
    median_value = np.round(np.median(data), 3)

    props_median = dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='gray')
    plt.axvline(x=median_value, linestyle='--', color='black')
    props_theta = dict(boxstyle='round', facecolor='white', alpha=1) #f'${{{x_axis_label}}}_{{median}} = {{{median_value}}}$'
    plt.text(median_value, (y.max()-2), x_axis_label+f'$_{{,\:median}} = {{{median_value}}}$', rotation=90, verticalalignment='center',
     bbox=props_theta, fontsize=10, fontweight='normal', horizontalalignment='center')
    plt.axvline(x=median_value+propagated_error, linestyle=':', color=what_color)
    plt.axvline(x=median_value-propagated_error, linestyle=':', color=what_color)

    ax.margins(x=0)
    rainfall_string = rainfall_case.replace("_", " ")
    rainfall_string = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", rainfall_string).strip()
    rainfall_string = rainfall_string.capitalize()
    text_to_display = f'{rainfall_string}'
    ss_or_chi = '\chi'
    variable = 'D'
    txt_median = f'$\Delta{{{variable}}}_{{median}} = {{{median_value}}} $'
    less_than = f'$ {{{variable}}}_{{{{{ss_or_chi}}}_{{rain}}}}<{{{variable}}}_{{{{{ss_or_chi}}}_{{no\:rain}}}}$'
    greater_than = f'$ {{{variable}}}_{{{{{ss_or_chi}}}_{{rain}}}}>{{{variable}}}_{{{{{ss_or_chi}}}_{{no\:rain}}}}$'
    #math_txt = '\n'+greater_or_less#txt_median +'\n'+greater_or_less
    x_min, x_max = x.min(), x.max()
    if x.min()<=0 and x.max()<=0:
        # case when min and max both have the same sign
        if (x.min()>(median_value-propagated_error)):
            x_min = median_value-propagated_error
        if (x.max()<(median_value+propagated_error)):
            x_max = median_value+propagated_error
        if x_max >=0:
            #plt.axvspan(x_min, 0.000, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100)
            plt.axvspan(median_value-propagated_error, median_value+propagated_error,  alpha=0.5,facecolor=what_color, edgecolor=what_color,hatch='//')#, label='Error in'+ x_axis_label+f'$_{{,\:median}} = {{{median_value}}}$')
            #plt.axvspan(0.000, x_max, facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100)
            plt.axvspan(median_value-propagated_error, median_value+propagated_error,  alpha=0.5, facecolor=what_color, edgecolor=what_color,hatch='//')#, label='Error in'+ x_axis_label+f'$_{{,\:median}} = {{{median_value}}}$')
        else:
            #plt.axvspan(x_min, x_max, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100)
            plt.axvspan(median_value-propagated_error, median_value+propagated_error,  alpha=0.5, facecolor=what_color,edgecolor=what_color, hatch='//')#, label='Error in'+ x_axis_label+f'$_{{,\:median}} = {{{median_value}}}$')
    elif x.min()>=0 and x.max()>=0:
        if (x.min()>(median_value-propagated_error)):
            x_min = median_value-propagated_error
        if (x.max()<(median_value+propagated_error)):
            x_max = median_value+propagated_error
        if x_min<=0:
            #plt.axvspan(x_min, 0.000, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100)
            plt.axvspan(median_value-propagated_error, median_value+propagated_error,  alpha=0.5, facecolor=what_color, edgecolor=what_color,hatch='//')
            #plt.axvspan(0.000, x_max, facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100)
            plt.axvspan(median_value-propagated_error, median_value+propagated_error,  alpha=0.5,facecolor=what_color,edgecolor=what_color, hatch='//')

        else:
            #plt.axvspan(x_min, x_max, facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100)
            plt.axvspan(median_value-propagated_error, median_value+propagated_error,  alpha=0.5, facecolor=what_color,edgecolor=what_color, hatch='//')
    else:
        #plt.axvspan(x.min()-propagated_error, 0.000, facecolor=cmc.buda.colors[0], alpha=0.5, zorder=-100)
        plt.axvspan(median_value-propagated_error, median_value+propagated_error, 0.000,  alpha=0.5, facecolor=what_color,edgecolor=what_color, hatch='//')

        #plt.axvspan(0.000, x.max()+propagated_error, facecolor=cmc.buda.colors[-1], alpha=0.5, zorder=-100)
        plt.axvspan(median_value-propagated_error, median_value+propagated_error,  alpha=0.2, facecolor=what_color,edgecolor=what_color,hatch='//')

    legend_elements = [Patch(facecolor='white', edgecolor='black',hatch='//',
                         label='Error in'+'$\:$'+ x_axis_label+f'$_{{,\:median}}$')]

    legend1 = ax.legend(title = 'Incision: Discharge' +'\n'+ less_than ,handlelength=0,handletextpad=0, facecolor='white', edgecolor='gold', loc=1)
    legend1._legend_box.align = "left"
    #legend1.get_texts()[0].set_fontsize('small')

    legend2 = ax.legend(title = 'Incision: Drainage Area' +'\n'+ greater_than,handlelength=0,handletextpad=0, facecolor='white', edgecolor='mediumorchid', loc=2)
    legend2._legend_box.align = "left"
    #legend2.get_texts()[0].set_fontsize('small')

    legend3 = ax.legend(handles=legend_elements ,handlelength=2,handletextpad=0.2, facecolor='white', edgecolor='black', loc=4)
    legend3._legend_box.align = "left"
    legend3.get_texts()[0].set_fontsize('small')

    ax.add_artist(legend1)
    ax.add_artist(legend2)
    #ax.add_artist(legend3)

    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # No decimal places
    plt.yticks(np.arange(min(y), max(y)+1, 1))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
    #plt.xticks(np.linspace(min(x), max(x), 10), rotation=45)
    plt.tight_layout()
    plt.savefig(f'{base_path}/ALL{rainfall_case}__{theta_or_disorder}_{all_ss_or_chi}_histogram.png', dpi = 300)
    #plt.clf()
    #plt.close(fig)
    #plt.show()
