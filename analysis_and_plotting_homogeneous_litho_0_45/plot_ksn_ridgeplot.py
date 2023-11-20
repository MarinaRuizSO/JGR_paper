import numpy as np 
import pandas as pd 
import os 

import matplotlib.pyplot as plt
import glob
from numpy import median
import seaborn as sns
import helplotlib
import math
import cmcrameri.cm as cmc
import rasterio
import re
import statsmodels.api as sm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import FancyBboxPatch
from functions_statistics_gradients import list_outlets_in_mountain, get_basin_name, get_csv_name_litho, pd_read_pattern
import matplotlib.gridspec as grid_spec

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 21


# Load the disorder data
# We need to carry out the analysis in pairs.

base_path = '/exports/csce/datastore/geos/users/s1440040/projects/phd-fastscape/phd-fastscape/model_outputs_for_statistics/'
ss_paths = [base_path+'ss_da_gradient/', base_path+'ss_discharge_gradient/']

gradient_values = [1,5,10]
ss_case = ['da','discharge_grad_1', 'discharge_grad_5','discharge_grad_10']



count = 0 # for the ss_paths

# the first 11 entries (because there are 11 gradients) will correspond to the ss da case.
# the following 11 entries correspond to the ss discharge case.  
ksn_list_theta_0_45 = [] # ksn (chi no rain) cases
ksn_list_theta_best = []
ksn_q_list_theta_0_45 = []
ksn_q_list_theta_best = []




for f_name in ss_paths:
    #print(f_name)
    #ksn_ratios_list = []

    count_chi_grads = 0
    
    for grad in gradient_values:

        f_list = list_outlets_in_mountain(f_name, f'grad_{grad}')
        #print(len(f_list))
        ksn_list_theta_0_45_all_basins = [] # ksn (chi no rain) cases
        ksn_list_theta_best_all_basins = []
        ksn_q_list_theta_0_45_all_basins = []
        ksn_q_list_theta_best_all_basins = []
        

        for i in range(len(f_list)): # iterate over all the basins
            #print(f_list)
            #print(f'count ss: {count}')
            #print(f'count chi: {count_chi_grads}')
            csv_file_name = f_list[i]
            basin_name = get_basin_name(csv_file_name)
            ###############################################
            # KSN
            ######## theta = 0.45 ########
            csv_path = get_csv_name_litho(f_name, basin_name, 'da', is_it_theta_best=False)
            #print(csv_path)

            df = pd_read_pattern(csv_path)
            # remove rows with negative values
            df = df[df['m_chi'] >= 0]

            # first need to normalise k_sn, add extra column 
            norm_m_chi = df['m_chi']#/df['m_chi'].max()
            df['norm_m_chi'] = norm_m_chi
            ksn_list_theta_0_45_all_basins.append(norm_m_chi.to_numpy())

            ######## theta = BEST ########
            csv_path = get_csv_name_litho(f_name, basin_name, 'da', is_it_theta_best=True)
            #print(csv_path)

            df = pd_read_pattern(csv_path)
            # remove rows with negative values
            df = df[df['m_chi'] >= 0]

            # first need to normalise k_sn, add extra column 
            norm_m_chi = df['m_chi']#/df['m_chi'].max()
            df['norm_m_chi'] = norm_m_chi
            ksn_list_theta_best_all_basins.append(norm_m_chi.tolist())

            ###############################################
            # KSN-Q
            ######## theta = 0.45 ########
            csv_path = get_csv_name_litho(f_name, basin_name, f'grad_{grad}', is_it_theta_best=False)
            #print(csv_path)

            df = pd_read_pattern(csv_path)
            df = df[df['m_chi'] >= 0]

            # first need to normalise k_sn, add extra column 
            norm_m_chi = df['m_chi']#/df['m_chi'].max()
            df['norm_m_chi'] = norm_m_chi
            ksn_q_list_theta_0_45_all_basins.append(norm_m_chi.tolist())

            ######## theta = BEST ########
            csv_path = get_csv_name_litho(f_name, basin_name, f'grad_{grad}', is_it_theta_best=True)
            #print(csv_path)

            df = pd_read_pattern(csv_path)
            df = df[df['m_chi'] >= 0]

            # first need to normalise k_sn, add extra column 
            norm_m_chi = df['m_chi']#/df['m_chi'].max()
            df['norm_m_chi'] = norm_m_chi
            ksn_q_list_theta_best_all_basins.append(norm_m_chi.tolist())

        count_chi_grads += 1

        ksn_list_theta_best.append(ksn_list_theta_best_all_basins)
        ksn_list_theta_0_45.append(ksn_list_theta_0_45_all_basins)
        ksn_q_list_theta_best.append(ksn_q_list_theta_best_all_basins)
        ksn_q_list_theta_0_45.append(ksn_q_list_theta_0_45_all_basins)
    count+=1


def create_legend(axs):
    ### cbar for ksn (reds)###
    norm = plt.Normalize(1, gradient_values[-1])
    cmap_blues = mcolors.LinearSegmentedColormap.from_list("", cmc.vik.colors[:int(256/3)+1],gamma=0.5,N=len(gradient_values))
    colors_blues = [mcolors.rgb2hex(cmap_blues(i)) for i in range(cmap_blues.N)]
    colors_blues= colors_blues[::-1]
    sm = plt.cm.ScalarMappable(cmap=cmap_blues, norm=norm)
    sm.set_array([])
    cbaxes = axs.inset_axes([0.8, 0.9, 0.07, 0.03],zorder=4) # [x0, y0, width, height]
    cbar = axs.figure.colorbar(sm, cax=cbaxes, orientation="horizontal", shrink = 0.1)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cbaxes.annotate(r'   $k_{sn-q}$ ', xy=(10, 1), zorder=10, fontsize = 21)

    ### cbar for ksn-q (blues)###
    cmap_reds = mcolors.LinearSegmentedColormap.from_list("", cmc.vik_r.colors[:int(256/3)+1],gamma=0.5,N=len(gradient_values))
    colors_reds = [mcolors.rgb2hex(cmap_reds(i)) for i in range(cmap_reds.N)]
    colors_reds = colors_reds[::-1]

    norm = plt.Normalize(1, gradient_values[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap_reds, norm=norm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    #cbaxes = inset_axes(ax, width="30%", height="3%", loc=2) 
    cbaxes = axs.inset_axes([0.8, 0.8, 0.07, 0.03],zorder=4) # [x0, y0, width, height]
    cbar = axs.figure.colorbar(sm, cax=cbaxes, orientation="horizontal", shrink = 0.1)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cbaxes.annotate(r'   $k_{sn}$ ', xy=(10, 1), zorder=10, fontsize=21)



def plot_ridgeplot(data, data_q, show_fig, save_fig, show_legend, title_str, figure_save_name, da_or_discharge, subplot_letter):

    cmap_blues = mcolors.LinearSegmentedColormap.from_list("", cmc.vik.colors[:int(256/3)+1],gamma=0.5,N=len(gradient_values))
    colors_blues = [mcolors.rgb2hex(cmap_blues(i)) for i in range(cmap_blues.N)]
    colors_blues= colors_blues[::-1]

    cmap_reds = mcolors.LinearSegmentedColormap.from_list("", cmc.vik_r.colors[:int(256/3)+1],gamma=0.5,N=len(gradient_values))
    colors_reds = [mcolors.rgb2hex(cmap_reds(i)) for i in range(cmap_reds.N)]
    colors_reds = colors_reds[::-1]

    gs = grid_spec.GridSpec(len(gradient_values),1)
    fig = plt.figure(figsize=(10,8))
    #gradient_values= [1,2]
    i = 0
    ax_objs = []
    for gradient in gradient_values:
        gradient = gradient_values[i]

        ksn = data[data['Rain']==f'Grad {gradient}']['ksn']
        #breakpoint()
        kde = sm.nonparametric.KDEUnivariate(ksn)
        if da_or_discharge =='da':
            bw_a = 1
            bw_q = 5
        else:
            bw_a = 5
            bw_q = 1
        kde.fit(bw=bw_a)
        # creating new axes object
        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
        ax_objs[-1].plot(kde.support, kde.density,color="black",lw=1, alpha=0.6, zorder=3)
        ax_objs[-1].fill_between(kde.support, kde.density, alpha=0.6,color=colors_reds[i], zorder = 3)



        #breakpoint()
        ksn_q = data_q[data_q['Rain']==f'Grad {gradient}']['ksn']
        #breakpoint()
        kde_q = sm.nonparametric.KDEUnivariate(ksn_q)
        kde_q.fit(bw=bw_q)
        #breakpoint()

        #logprob = kde.score_samples(x_d[:, None])

        
        # plotting the distribution
        ax_objs[-1].plot(kde_q.support, kde_q.density,color="black",lw=1, alpha=0.6, zorder=4)
        ax_objs[-1].fill_between(kde_q.support, kde_q.density, alpha=0.6,color=colors_blues[i], zorder=4)


        # setting uniform x and y lims
        ax_objs[-1].set_xlim(0,1000)
        ax_objs[-1].set_ylim(0,0.03)

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels

        if i == len(gradient_values)-1:
            ax_objs[-1].set_xlabel("Channel Steepness", fontsize=21,fontweight="bold")
        else:
            ax_objs[-1].set_xticklabels([])

        spines = ["top","right","left","bottom"]
        for s in spines:
            if s =="bottom":
                ax_objs[-1].spines[s].set_visible(True)
            else:
                ax_objs[-1].spines[s].set_visible(False)

        #adj_gradient = gradient.replace(" ","\n")
        ax_objs[-1].text(-10,0,gradient,fontweight="bold",fontsize=21,ha="right")

        #plt.xticks([])
        plt.yticks([])
        plt.xticks(fontsize=21)

        fig.supylabel('Rainfall Gradient (m/yr)',fontweight="bold",fontsize=21)
        ax_objs[0].text(-0.1, 0.9, subplot_letter, transform=ax_objs[0].transAxes, 
            size=21, weight='bold')



        i += 1

    gs.update(hspace=-0.7)
    ####

    if show_legend==True:
        create_legend(ax_objs[0])
    
    if show_fig == True: 
        ax_objs[0].set_title(title_str)
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()
    if save_fig == True:
        ax_objs[0].set_title(title_str)
        plt.savefig(figure_save_name, dpi = 300)
        plt.cla()
        plt.clf()
        plt.close()



SAVE_FIGURES = True
SHOW_FIGS = False
y_limit = 0.03
########################################################################################################################
# DRAINAGE AREA
########################################################################################################################

# to plot the distributions, we need to take the entries from the first 
# put the data into a dataframe so that it's easier to plot and visualise
########################################
########## KSN THETA BEST ##########
##### KSN #####
# need to flatten the list for each of the rainfall gradients 
flat_list_per_grad_ksn = []
for i in range(2*len(gradient_values)): # twice to account for the two steady state cases
    flat_list_all_basins_ksn = [item for sublist in ksn_list_theta_best[i] for item in sublist]
    flat_list_per_grad_ksn.append(flat_list_all_basins_ksn)

# initialise the dataframe STEADY STATE: DRAINAGE AREA
df_theta_best_ksn = pd.DataFrame(flat_list_per_grad_ksn[0], columns=[f"Grad {gradient_values[0]}"])
df_theta_best_grads_ksn = pd.DataFrame(flat_list_per_grad_ksn[1], columns=[f"Grad {gradient_values[1]}"])
new_df_ksn = pd.concat([df_theta_best_ksn, df_theta_best_grads_ksn], ignore_index=False, axis = 1)
# need to concatenate because they are different lengths 
for i in range(2, len(gradient_values)):
    df_theta_best_grads_ksn = pd.DataFrame(flat_list_per_grad_ksn[i], columns=[f"Grad {gradient_values[i]}"])
    new_df_ksn = pd.concat([new_df_ksn, df_theta_best_grads_ksn], ignore_index=False, axis = 1)
new_df_ksn = new_df_ksn.melt(var_name='Rain', value_name='ksn')

##### KSN_Q #####
# need to flatten the list for each of the rainfall gradients 
flat_list_per_grad_ksn_q = []
for i in range(2*len(gradient_values)): # twice to account for the two steady state cases
    flat_list_all_basins_ksn_q = [item for sublist in ksn_q_list_theta_best[i] for item in sublist]
    flat_list_per_grad_ksn_q.append(flat_list_all_basins_ksn_q)

# initialise the dataframe STEADY STATE: DRAINAGE AREA

df_theta_best_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[0], columns=[f"Grad {gradient_values[0]}"])
df_theta_best_grads_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[1], columns=[f"Grad {gradient_values[1]}"])
new_df_ksn_q = pd.concat([df_theta_best_ksn_q, df_theta_best_grads_ksn_q], ignore_index=False, axis = 1)

for i in range(2, len(gradient_values)):
    df_theta_best_grads_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[i], columns=[f"Grad {gradient_values[i]}"])
    new_df_ksn_q = pd.concat([new_df_ksn_q, df_theta_best_grads_ksn_q], ignore_index=False, axis = 1)
new_df_ksn_q = new_df_ksn_q.melt(var_name='Rain', value_name='ksn')

########## PLOTTING #########

fig_name = 'ss_da_theta_best_ksn_ridgeplot.pdf'
title_name = r'Incision: Drainage Area. $\theta = \theta_{best}$'
plot_ridgeplot(new_df_ksn, new_df_ksn_q, SHOW_FIGS, SAVE_FIGURES,True, title_name, fig_name, 'da','B')




########## KSN THETA=0.45 ##########
flat_list_per_grad_ksn = []
for i in range(2*len(gradient_values)): # twice to account for the two steady state cases
    flat_list_all_basins_ksn = [item for sublist in ksn_list_theta_0_45[i] for item in sublist]
    flat_list_per_grad_ksn.append(flat_list_all_basins_ksn)

# initialise the dataframe STEADY STATE: DRAINAGE AREA
df_theta_best_ksn = pd.DataFrame(flat_list_per_grad_ksn[0], columns=[f"Grad {gradient_values[0]}"])
df_theta_best_grads_ksn = pd.DataFrame(flat_list_per_grad_ksn[1], columns=[f"Grad {gradient_values[1]}"])
new_df_ksn = pd.concat([df_theta_best_ksn, df_theta_best_grads_ksn], ignore_index=False, axis = 1)
# need to concatenate because they are different lengths 
for i in range(2, len(gradient_values)):
    df_theta_best_grads_ksn = pd.DataFrame(flat_list_per_grad_ksn[i], columns=[f"Grad {gradient_values[i]}"])
    new_df_ksn = pd.concat([new_df_ksn, df_theta_best_grads_ksn], ignore_index=False, axis = 1)
new_df_ksn = new_df_ksn.melt(var_name='Rain', value_name='ksn')

##### KSN_Q #####
# need to flatten the list for each of the rainfall gradients 
flat_list_per_grad_ksn_q = []
for i in range(2*len(gradient_values)): # twice to account for the two steady state cases
    flat_list_all_basins_ksn_q = [item for sublist in ksn_q_list_theta_0_45[i] for item in sublist]
    flat_list_per_grad_ksn_q.append(flat_list_all_basins_ksn_q)

# initialise the dataframe STEADY STATE: DRAINAGE AREA

df_theta_best_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[0], columns=[f"Grad {gradient_values[0]}"])
df_theta_best_grads_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[1], columns=[f"Grad {gradient_values[1]}"])
new_df_ksn_q = pd.concat([df_theta_best_ksn_q, df_theta_best_grads_ksn_q], ignore_index=False, axis = 1)

for i in range(2, len(gradient_values)):
    df_theta_best_grads_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[i], columns=[f"Grad {gradient_values[i]}"])
    new_df_ksn_q = pd.concat([new_df_ksn_q, df_theta_best_grads_ksn_q], ignore_index=False, axis = 1)
new_df_ksn_q = new_df_ksn_q.melt(var_name='Rain', value_name='ksn')


########## PLOTTING #########

fig_name = 'ss_da_theta_0_45_ksn_ridgeplot.pdf'
title_name = r'Incision: Drainage Area. $\theta = 0.45$'
plot_ridgeplot(new_df_ksn, new_df_ksn_q, SHOW_FIGS, SAVE_FIGURES,False, title_name, fig_name, 'da', 'D')



########################################################################################################################
# DISCHARGE
########################################################################################################################

########################################
########## KSN THETA BEST ##########
##### KSN #####
# need to flatten the list for each of the rainfall gradients 
flat_list_per_grad_ksn = []
for i in range(2*len(gradient_values)): # twice to account for the two steady state cases
    flat_list_all_basins_ksn = [item for sublist in ksn_list_theta_best[i] for item in sublist]
    flat_list_per_grad_ksn.append(flat_list_all_basins_ksn)

# initialise the dataframe STEADY STATE: DISCHARGE
df_theta_best_ksn = pd.DataFrame(flat_list_per_grad_ksn[len(gradient_values)], columns=[f"Grad {gradient_values[0]}"])
df_theta_best_grads_ksn = pd.DataFrame(flat_list_per_grad_ksn[len(gradient_values)+1], columns=[f"Grad {gradient_values[1]}"])
new_df_ksn = pd.concat([df_theta_best_ksn, df_theta_best_grads_ksn], ignore_index=False, axis = 1)
# need to concatenate because they are different lengths 
for i in range(len(gradient_values)+1, 2*len(gradient_values)):
    print(i)
    df_theta_best_grads_ksn = pd.DataFrame(flat_list_per_grad_ksn[i], columns=[f"Grad {gradient_values[i-len(gradient_values)]}"])
    new_df_ksn = pd.concat([new_df_ksn, df_theta_best_grads_ksn], ignore_index=False, axis = 1)
new_df_ksn = new_df_ksn.melt(var_name='Rain', value_name='ksn')
##### KSN_Q #####
# need to flatten the list for each of the rainfall gradients 
flat_list_per_grad_ksn_q = []
for i in range(2*len(gradient_values)): # twice to account for the two steady state cases
    flat_list_all_basins_ksn_q = [item for sublist in ksn_q_list_theta_best[i] for item in sublist]
    flat_list_per_grad_ksn_q.append(flat_list_all_basins_ksn_q)

# initialise the dataframe STEADY STATE: DISCHARGE

df_theta_best_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[len(gradient_values)], columns=[f"Grad {gradient_values[0]}"])
df_theta_best_grads_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[len(gradient_values)+1], columns=[f"Grad {gradient_values[1]}"])
new_df_ksn_q = pd.concat([df_theta_best_ksn_q, df_theta_best_grads_ksn_q], ignore_index=False, axis = 1)

for i in range(len(gradient_values)+1, 2*len(gradient_values)):
    df_theta_best_grads_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[i], columns=[f"Grad {gradient_values[i-len(gradient_values)]}"])
    new_df_ksn_q = pd.concat([new_df_ksn_q, df_theta_best_grads_ksn_q], ignore_index=False, axis = 1)
new_df_ksn_q = new_df_ksn_q.melt(var_name='Rain', value_name='ksn')
########## PLOTTING #########

fig_name = 'ss_discharge_theta_best_ksn_ridgeplot.pdf'
title_name = r'Incision: Discharge. $\theta = \theta_{best}$'
plot_ridgeplot(new_df_ksn, new_df_ksn_q, SHOW_FIGS, SAVE_FIGURES,False, title_name, fig_name, 'discharge', 'A')

########## KSN THETA=0.45 ##########
##### KSN #####
# need to flatten the list for each of the rainfall gradients 
flat_list_per_grad_ksn = []
for i in range(2*len(gradient_values)): # twice to account for the two steady state cases
    flat_list_all_basins_ksn = [item for sublist in ksn_list_theta_0_45[i] for item in sublist]
    flat_list_per_grad_ksn.append(flat_list_all_basins_ksn)

# initialise the dataframe STEADY STATE: DISCHARGE
df_theta_best_ksn = pd.DataFrame(flat_list_per_grad_ksn[len(gradient_values)], columns=[f"Grad {gradient_values[0]}"])
df_theta_best_grads_ksn = pd.DataFrame(flat_list_per_grad_ksn[len(gradient_values)+1], columns=[f"Grad {gradient_values[1]}"])
new_df_ksn = pd.concat([df_theta_best_ksn, df_theta_best_grads_ksn], ignore_index=False, axis = 1)
# need to concatenate because they are different lengths 
for i in range(len(gradient_values)+1, 2*len(gradient_values)):
    print(i)
    df_theta_best_grads_ksn = pd.DataFrame(flat_list_per_grad_ksn[i], columns=[f"Grad {gradient_values[i-len(gradient_values)]}"])
    new_df_ksn = pd.concat([new_df_ksn, df_theta_best_grads_ksn], ignore_index=False, axis = 1)
new_df_ksn = new_df_ksn.melt(var_name='Rain', value_name='ksn')
##### KSN_Q #####
# need to flatten the list for each of the rainfall gradients 
flat_list_per_grad_ksn_q = []
for i in range(2*len(gradient_values)): # twice to account for the two steady state cases
    flat_list_all_basins_ksn_q = [item for sublist in ksn_q_list_theta_best[i] for item in sublist]
    flat_list_per_grad_ksn_q.append(flat_list_all_basins_ksn_q)

# initialise the dataframe STEADY STATE: DISCHARGE

df_theta_best_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[len(gradient_values)], columns=[f"Grad {gradient_values[0]}"])
df_theta_best_grads_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[len(gradient_values)+1], columns=[f"Grad {gradient_values[1]}"])
new_df_ksn_q = pd.concat([df_theta_best_ksn_q, df_theta_best_grads_ksn_q], ignore_index=False, axis = 1)

for i in range(len(gradient_values)+1, 2*len(gradient_values)):
    df_theta_best_grads_ksn_q = pd.DataFrame(flat_list_per_grad_ksn_q[i], columns=[f"Grad {gradient_values[i-len(gradient_values)]}"])
    new_df_ksn_q = pd.concat([new_df_ksn_q, df_theta_best_grads_ksn_q], ignore_index=False, axis = 1)
new_df_ksn_q = new_df_ksn_q.melt(var_name='Rain', value_name='ksn')
########## PLOTTING #########

fig_name = 'ss_discharge_theta_0_45_ksn_ridgeplot.pdf'
title_name = r'Incision: Discharge. $\theta = 0.45$'
plot_ridgeplot(new_df_ksn, new_df_ksn_q, SHOW_FIGS, SAVE_FIGURES,False, title_name, fig_name, 'discharge', 'C')
