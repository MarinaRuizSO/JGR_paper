import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import helplotlib as hpl
import cmcrameri.cm as cmc
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors



os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Keep this or the code breaks because the HDF files aren't closed properly.

mountain_range = 'argentina0'
# import the pickle files with the chi and the distance data
base_path = '/exports/csce/datastore/geos/users/s1440040/LSDTopoTools/data/ExampleTopoDatasets/ChiAnalysisData/dems_to_process/argentina/input_data/'
file_name_chi_a_0_45 = f'{mountain_range}_no_rainfall__chi_map_theta_0_45.csv'
file_name_chi_a_best = f'{mountain_range}_no_rainfall__chi_map_theta_best_0.225.csv'

ksn_dataframe_a_0_45 = pd.read_csv(base_path+file_name_chi_a_0_45)
ksn_dataframe_a_best = pd.read_csv(base_path+file_name_chi_a_best)

# import the pickle files with the chi and the distance data
file_name_chi_q_0_45 = f'{mountain_range}_original_rain__chi_map_theta_0_45.csv'
file_name_chi_q_best = f'{mountain_range}_original_rain__chi_map_theta_best_0.3.csv'

ksn_dataframe_q_0_45 = pd.read_csv(base_path+file_name_chi_q_0_45)
ksn_dataframe_q_best = pd.read_csv(base_path+file_name_chi_q_best)
#ksn_dataframe.head()
hpl.mkfig_simple_bold(fontsize_major = 21, fontsize_minor= 14, family = "DejaVu Sans" , figsize = (10,8))
fig, axs = plt.subplots(2,2, figsize=(22, 16))

title_fontdict={'fontsize': 18, 'fontweight': 'bold', 'family' : "DejaVu Sans"}
color_map = cmc.batlow_r
# plot the basin shape
# plt.scatter(ksn_dataframe_q_0_45.x,ksn_dataframe_q_0_45.y, c = ksn_dataframe_q_0_45.basin_key, s = 5, lw = 2, zorder = 2)
# plt.show()

def create_legend(axs, cbar_min, cbar_max):
    ### cbar for ksn (reds)###
    norm = plt.Normalize(cbar_min, cbar_max)

    sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array([])
    cbaxes = fig.add_axes([0.9, 0.15, 0.025, 0.7], zorder=10)
    cbar = axs.figure.colorbar(sm, cax=cbaxes, orientation="vertical")#, fontsize = 14)
    cbar.ax.set_ylabel(r'$log_{10}(k_{sn})$', size=21)
    cbar.ax.tick_params(labelsize=21)




def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    #subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = fig.add_axes([x,y,width,height],facecolor=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.25
    y_labelsize *= rect[3]**0.25
    subax.xaxis.set_tick_params(labelsize=8)
    subax.yaxis.set_tick_params(labelsize=8)
    return subax

 

def plot_profiles(axis, df, xlabel, subplot_letter):
    # main_basin_m_chi = df.loc[df['source_key'] == 0, 'm_chi']
    # main_basin_chi = df.loc[df['source_key'] == 0, 'chi']
    # main_basin_elev = df.loc[df['source_key'] == 0, 'elevation']
    # main_basin_flow = df.loc[df['source_key'] == 0, 'flow_distance']
    #sc = axis.scatter(main_basin_chi.values, main_basin_elev.values, alpha = 1, c = np.log(main_basin_m_chi.values), cmap = color_map)
    
    # do some manipulations on the data - remove wrong values
    df_non_zero = df.drop(df.index[df['m_chi'] < 0])
    df_non_zero = df_non_zero.drop(df_non_zero.index[df_non_zero['elevation'] < 10])
    norm_drainage_area = df_non_zero['drainage_area']/df_non_zero['drainage_area'].max()
    df_non_zero['norm_drainage_area'] = norm_drainage_area
    df_thresh = df_non_zero[df_non_zero['norm_drainage_area']>0.01]
    # just for alburz_south2
    df_thresh = df_thresh[df_thresh['elevation']>1240]
    #breakpoint()


    min_val = np.nanmin(np.log(df_non_zero['m_chi']).values)
    max_val = np.nanmax(np.log(df_non_zero['m_chi']).values)

    sc = axis.scatter(df_thresh['chi'], df_thresh['elevation'], alpha = 1, c = np.log(df_thresh['m_chi']), cmap = color_map, s=15)
    cbar = plt.colorbar(sc, ax=axis)
    cbar.ax.set_ylabel(r'$log_{10}(k_{sn})$', size=21)

    #cbar = fig.colorbar(sc, ax=axis)

    axis.set_xlabel(r"$\chi$ (m)", fontsize=21)
    axis.set_ylabel('Elevation (m)', fontsize=21)
    axis.legend([],[], frameon=False)
    axis.text(-0.1, 0.9, subplot_letter, transform=axis.transAxes, 
            size=21, weight='bold')
    max_elevation = df_thresh['elevation'].max()
    axis.text(0, max_elevation-200, xlabel, horizontalalignment='left', size=21, 
              color='black', weight='semibold')
    axis.tick_params(axis='both', which='major', labelsize=21)
    axis.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axis.yaxis.get_offset_text().set_fontsize(18)

    rect = [0.57,0.1,0.3,0.3]
    ax1 = add_subplot_axes(axis,rect)
    #sc1=ax1.scatter(main_basin_flow.values, main_basin_elev.values, alpha = 1, c = np.log(main_basin_m_chi.values), cmap = color_map, s=10)
    sc1=ax1.scatter(df_thresh['flow_distance'], df_thresh['elevation'], alpha = 1, c = np.log(df_thresh['m_chi']), cmap = color_map, s=5)
    ax1.set_xlabel("Distance (m)", fontsize=12)
    ax1.set_ylabel('Elevation (m)', fontsize=12)
    ax1.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax1.xaxis.get_offset_text().set_fontsize(12)
    ax1.yaxis.get_offset_text().set_fontsize(12)
    
    #sc1.ax1.set_tick_params(fontweight='normal')
    return min_val, max_val
    
min_ksn_a_45, max_ksn_a_45 = plot_profiles(axs[0][0], ksn_dataframe_a_0_45, r"$\chi_{A}, \theta=0.45$", 'A')
min_ksn_a_best, max_ksn_a_best = plot_profiles(axs[0][1], ksn_dataframe_a_best, r"$\chi_{A}, \theta=\theta_{best}$", 'B')
min_ksn_q_45, max_ksn_q_45 = plot_profiles(axs[1][0], ksn_dataframe_q_0_45, r"$\chi_{Q}, \theta=0.45$", 'C')
min_ksn_q_best, max_ksn_q_best = plot_profiles(axs[1][1], ksn_dataframe_q_best, r"$\chi_{Q}, \theta=\theta_{best}$", 'D')
#plt.tight_layout()

hue_norm_max = np.max((max_ksn_a_45, max_ksn_a_best, 
                    max_ksn_q_45, max_ksn_q_best))

hue_norm_min = np.min((min_ksn_a_45, min_ksn_a_best, 
                    min_ksn_q_45, min_ksn_q_best))

fig.subplots_adjust(right=0.87)
#create_legend(axs[0][1], hue_norm_min, hue_norm_max)
#plt.show()
plt.savefig(base_path + f'chi_plots_discharge_paper_{mountain_range}.pdf', dpi = 400, bbox_inches ='tight')
