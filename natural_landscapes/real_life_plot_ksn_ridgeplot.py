'''
File to plot the ksn and ksn-q values and plot them with a kde distribution

Marina Ruiz Sanchez-Oro
28/09/2023
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
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import statsmodels.api as sm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import FancyBboxPatch
#from functions_statistics_gradients import list_outlets_in_mountain, get_basin_name, get_csv_name_litho, pd_read_pattern
import matplotlib.gridspec as grid_spec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 21
plt.rcParams['legend.title_fontsize'] = 21
plt.rcParams["legend.fontsize"] = 18

# Load the disorder data
# We need to carry out the analysis in pairs.

base_path = '/exports/csce/datastore/geos/users/s1440040/LSDTopoTools/data/ExampleTopoDatasets/ChiAnalysisData/dems_to_process/'
DEM_paths = [base_path+'peru/input_data/', base_path+'argentina/input_data/', base_path+'xian_all/input_data/', base_path+'turkey/input_data/',base_path+'colorado/input_data/', base_path+'alburz_south/input_data/', base_path+'massif_central/input_data/',base_path+'pyrenees/input_data/']# complete the list later - first try with just two cases

mountain_range_names =  ['peru','argentina', 'xian', 'turkey', 'colorado', 'alburz_south', 'massif_central', 'pyrenees']#['andes_north', 'alburz_north', 'alburz_south', 'massif_central', 'pyrenees']
mountain_range_names_legend = ['N. Andes', 'S. Andes', 'Qinling', 'Kaçkar', 'Rockies', 'Alburz', 'Massif'+'\n'+' Central', 'Pyrénées']
# all of this precipitation needs to be in m/yr
# here we an include the no rainfall scenario because we are not processing any DEM files.
precipitation_types = ['original_rain']
counter = 0


ksn_list_theta_0_45 = [] # ksn (chi no rain) cases
ksn_list_theta_best = []
ksn_q_list_theta_0_45 = []
ksn_q_list_theta_best = []

for i in range(len(precipitation_types)):
    for j in range(len(mountain_range_names)):

        filelist = count_outlets_in_mountain(DEM_paths[j])

        ksn_list_theta_0_45_all_basins = [] # ksn (chi no rain) cases
        ksn_list_theta_best_all_basins = []
        ksn_q_list_theta_0_45_all_basins = []
        ksn_q_list_theta_best_all_basins = []
        for k in range(len(filelist)):
            csv_file_name = filelist[k]
            basin_name = get_basin_name(csv_file_name)
            #print('I am processing basin: {}, precipitation case: {}'.format(basin_name,precipitation_types[i]))

            ###############################################
            # NO RAIN KSN

            ######## theta = 0.45 ########
            csv_path = get_csv_name(DEM_paths[j], basin_name, 'no_rainfall', is_it_theta_best = False)
            df = pd_read_pattern(csv_path)
            # remove rows with negative values
            df = df[df['m_chi'] >= 0]

            # first need to normalise k_sn, add extra column 
            norm_m_chi = df['m_chi']#/df['m_chi'].max()
            df['norm_m_chi'] = norm_m_chi
            ksn_list_theta_0_45_all_basins.append(norm_m_chi.to_numpy())

            ######## theta = BEST ########
            csv_path = get_csv_name(DEM_paths[j], basin_name, 'no_rainfall', is_it_theta_best = True)
            df = pd_read_pattern(csv_path)
            # remove rows with negative values
            df = df[df['m_chi'] >= 0]

            # first need to normalise k_sn, add extra column 
            norm_m_chi = df['m_chi']#/df['m_chi'].max()
            df['norm_m_chi'] = norm_m_chi
            ksn_list_theta_best_all_basins.append(norm_m_chi.tolist())


            #
            # ################################################
            # # ORIGINAL RAIN KSN-Q

            ######## theta = 0.45 ########


            csv_path = get_csv_name(DEM_paths[j], basin_name, precipitation_types[i], is_it_theta_best = False)

            df = pd_read_pattern(csv_path)
            df = df[df['m_chi'] >= 0]

            # first need to normalise k_sn, add extra column 
            norm_m_chi = df['m_chi']#/df['m_chi'].max()
            df['norm_m_chi'] = norm_m_chi
            ksn_q_list_theta_0_45_all_basins.append(norm_m_chi.tolist())
            ######## theta = BEST ########

            csv_path = get_csv_name(DEM_paths[j], basin_name, precipitation_types[i], is_it_theta_best = True)
            df = pd_read_pattern(csv_path)
            df = df[df['m_chi'] >= 0]

            # first need to normalise k_sn, add extra column 
            norm_m_chi = df['m_chi']#/df['m_chi'].max()
            df['norm_m_chi'] = norm_m_chi
            ksn_q_list_theta_best_all_basins.append(norm_m_chi.tolist())


        ksn_list_theta_best.append(ksn_list_theta_best_all_basins)
        ksn_list_theta_0_45.append(ksn_list_theta_0_45_all_basins)
        ksn_q_list_theta_best.append(ksn_q_list_theta_best_all_basins)
        ksn_q_list_theta_0_45.append(ksn_q_list_theta_0_45_all_basins)





def plot_ridgeplot(data, data_q, show_fig, save_fig, show_legend, title_str, figure_save_name, subplot_letter):
    cmap_blues = mcolors.LinearSegmentedColormap.from_list("", cmc.berlin.colors[:int(256/3)+1],gamma=0.5,N=len(mountain_range_names))
    colors_blues = [mcolors.rgb2hex(cmap_blues(i)) for i in range(cmap_blues.N)]
    #colors_blues= colors_blues[::-1]

    cmap_reds = mcolors.LinearSegmentedColormap.from_list("", cmc.berlin.colors[:int(256/3)+1],gamma=0.5,N=len(mountain_range_names))
    colors_reds = [mcolors.rgb2hex(cmap_reds(i)) for i in range(cmap_reds.N)]
    #colors_reds = colors_reds[::-1]

    gs = grid_spec.GridSpec(len(mountain_range_names),1)
    fig = plt.figure(figsize=(8,8))
    #mountain_range_names= [1,2]
    i = 0
    ax_objs = []
    for mountain in mountain_range_names:
        mountain = mountain_range_names[i]

        ksn = data[mountain]
        kde = sm.nonparametric.KDEUnivariate(ksn)
        kde.fit(bw=5)
        # # creating new axes object
        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
        ax_objs[-1].plot(kde.support, kde.density,color='black', alpha=0.6,lw=1, zorder=3)
        ax_objs[-1].fill_between(kde.support, kde.density, alpha=0.6,color=cmc.berlin.colors[-int(256/10)], zorder=3)



        ksn_q = data_q[mountain]
        kde_q = sm.nonparametric.KDEUnivariate(ksn_q)
        kde_q.fit(bw=5)

        #logprob = kde.score_samples(x_d[:, None])

        
        # plotting the distribution
        ax_objs[-1].plot(kde_q.support, kde_q.density,color='black', alpha=0.6,lw=1, zorder= 4)
        ax_objs[-1].fill_between(kde_q.support, kde_q.density, alpha=0.6,color=cmc.berlin.colors[int(256/4)], zorder=4)


        # setting uniform x and y lims
        ax_objs[-1].set_xlim(0,400)
        ax_objs[-1].set_ylim(0,0.03)

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax_objs[-1].tick_params(labelsize='x-large')

        if i == len(mountain_range_names)-1:
            ax_objs[-1].set_xlabel(r"$k_{sn}$", fontsize=21,fontweight="bold")
        else:
            ax_objs[-1].set_xticklabels([])
            

        spines = ["top","right","left","bottom"]
        for s in spines:
            if s =="bottom":
                ax_objs[-1].spines[s].set_visible(True)
            else:
                ax_objs[-1].spines[s].set_visible(False)

        #adj_mountain = mountain.replace(" ","\n")
        ax_objs[-1].text(-10,0,mountain_range_names_legend[i],fontweight="bold",fontsize=21,ha="right")

        #plt.xticks([])
        plt.yticks([])
        ax_objs[0].text(-0.1, 0.9, subplot_letter, transform=ax_objs[0].transAxes, 
            size=21, weight='bold')
        

        #fig.supylabel('Rainfall Gradient (m/yr)',fontweight="bold",fontsize=14 )


        i += 1
    if show_legend==True:
        legend_elements = [
                    mpatches.Patch(color=cmc.berlin.colors[-int(256/10)], label=r'$k_{sn}$', alpha=0.6 ),
                    mpatches.Patch(color=cmc.berlin.colors[int(256/4)], label=r'$k_{sn-q}$',alpha=0.6)]
        ax_objs[0].legend(handles=legend_elements, title = title_str, fontsize="21", bbox_to_anchor=(0.65, 0.25))

    gs.update(hspace=-0.7)
    ####

    #axes.set_xlabel('Channel Steepness')
    #axes.set_xlim(0)
    #axes.get_legend().remove()
    #if show_legend==True:
        #create_legend(ax_objs[0])
    #arr = ploty.lines[0].get_ydata()
    
    if show_fig == True: 
        #axes.set_ylim(0, y_axis_lim)
        #ax_objs[0].set_title(title_str)
        #fig.tight_layout()
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()
    if save_fig == True:
        #axes.set_ylim(0, y_axis_lim)
        #ax_objs[0].set_title(title_str)
        #fig.tight_layout()
        plt.savefig(figure_save_name, dpi = 300, bbox_inches ='tight')
        plt.cla()
        plt.clf()
        plt.close()
    


SAVE_FIGURES = False 
SHOW_FIGS = True
y_limit_best = 0.02
x_limit_best = 1000
y_limit_45 = 0.03
x_limit_45 = 1000


# cmap_hawaii = mcolors.LinearSegmentedColormap.from_list("", cmc.hawaii_r.colors,gamma=0.5,N=len(mountain_range_names))
# palette= cmap_hawaii(np.linspace(0,1,cmap_hawaii.N))

########################################################################################################################
# DRAINAGE AREA
########################################################################################################################

# to plot the distributions, we need to take the entries from the first 
# put the data into a dataframe so that it's easier to plot and visualise
########################################
########## KSN THETA BEST ##########
##### KSN #####
# need to flatten the list for each of the rainfall mountains 
flat_list_per_grad_ksn = []
for i in range(len(mountain_range_names)): # twice to account for the two steady state cases
    flat_list_all_basins_ksn = [item for sublist in ksn_list_theta_best[i] for item in sublist]
    flat_list_per_grad_ksn.append(flat_list_all_basins_ksn)

# initialise the dataframe STEADY STATE: DRAINAGE AREA
df_theta_best_ksn = pd.DataFrame(flat_list_per_grad_ksn[0], columns=[f'{mountain_range_names[0]}'])
df_theta_best_grads_ksn = pd.DataFrame(flat_list_per_grad_ksn[1], columns=[f'{mountain_range_names[1]}'])
new_df_ksn = pd.concat([df_theta_best_ksn, df_theta_best_grads_ksn], ignore_index=False, axis = 1)


# need to concatenate because they are different lengths 
for i in range(2, len(mountain_range_names)):
    df_theta_best_grads_ksn = pd.DataFrame(flat_list_per_grad_ksn[i], columns=[f'{mountain_range_names[i]}'])
    new_df_ksn = pd.concat([new_df_ksn, df_theta_best_grads_ksn], ignore_index=False, axis = 1)

##### KSN_Q #####
# need to flatten the list for each of the rainfall mountains 
flat_list_per_grad_ksn_q = []
for i in range(len(mountain_range_names)): # twice to account for the two steady state cases
    flat_list_all_basins_ksn_q = [item for sublist in ksn_q_list_theta_best[i] for item in sublist]
    flat_list_per_grad_ksn_q.append(flat_list_all_basins_ksn_q)

# initialise the dataframe STEADY STATE: DRAINAGE AREA

df_theta_best_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[0], columns=[f'{mountain_range_names[0]}'])
df_theta_best_grads_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[1], columns=[f'{mountain_range_names[1]}'])
new_df_ksn_q = pd.concat([df_theta_best_ksn_q, df_theta_best_grads_ksn_q], ignore_index=False, axis = 1)

for i in range(2, len(mountain_range_names)):
    df_theta_best_grads_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[i], columns=[f'{mountain_range_names[i]}'])
    new_df_ksn_q = pd.concat([new_df_ksn_q, df_theta_best_grads_ksn_q], ignore_index=False, axis = 1)


fig_name = 'real_mountains_theta_best_ksn_ridgeplot.pdf'
title_name = r'$\theta = \theta_{best}$'

plot_ridgeplot(new_df_ksn, new_df_ksn_q, SHOW_FIGS, SAVE_FIGURES,True, title_name, fig_name, 'B')


########## KSN THETA 0.45 ##########
##### KSN #####
# need to flatten the list for each of the rainfall mountains 
flat_list_per_grad_ksn = []
for i in range(len(mountain_range_names)): # twice to account for the two steady state cases
    flat_list_all_basins_ksn = [item for sublist in ksn_list_theta_0_45[i] for item in sublist]
    flat_list_per_grad_ksn.append(flat_list_all_basins_ksn)

# initialise the dataframe STEADY STATE: DRAINAGE AREA
df_theta_best_ksn = pd.DataFrame(flat_list_per_grad_ksn[0], columns=[f'{mountain_range_names[0]}'])
df_theta_best_grads_ksn = pd.DataFrame(flat_list_per_grad_ksn[1], columns=[f'{mountain_range_names[1]}'])
new_df_ksn = pd.concat([df_theta_best_ksn, df_theta_best_grads_ksn], ignore_index=False, axis = 1)


# need to concatenate because they are different lengths 
for i in range(2, len(mountain_range_names)):
    df_theta_best_grads_ksn = pd.DataFrame(flat_list_per_grad_ksn[i], columns=[f'{mountain_range_names[i]}'])
    new_df_ksn = pd.concat([new_df_ksn, df_theta_best_grads_ksn], ignore_index=False, axis = 1)

##### KSN_Q #####
# need to flatten the list for each of the rainfall mountains 
flat_list_per_grad_ksn_q = []
for i in range(len(mountain_range_names)): # twice to account for the two steady state cases
    flat_list_all_basins_ksn_q = [item for sublist in ksn_q_list_theta_0_45[i] for item in sublist]
    flat_list_per_grad_ksn_q.append(flat_list_all_basins_ksn_q)

# initialise the dataframe STEADY STATE: DRAINAGE AREA

df_theta_best_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[0], columns=[f'{mountain_range_names[0]}'])
df_theta_best_grads_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[1], columns=[f'{mountain_range_names[1]}'])
new_df_ksn_q = pd.concat([df_theta_best_ksn_q, df_theta_best_grads_ksn_q], ignore_index=False, axis = 1)

for i in range(2, len(mountain_range_names)):
    df_theta_best_grads_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[i], columns=[f'{mountain_range_names[i]}'])
    new_df_ksn_q = pd.concat([new_df_ksn_q, df_theta_best_grads_ksn_q], ignore_index=False, axis = 1)


fig_name = 'real_mountains_theta_0_45_ksn_ridgeplot.pdf'
title_name = r'$\theta = 0.45$'

plot_ridgeplot(new_df_ksn, new_df_ksn_q, SHOW_FIGS, SAVE_FIGURES,True, title_name, fig_name, 'A')