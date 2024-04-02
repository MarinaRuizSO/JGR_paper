import lsdviztools.lsdbasemaptools as bmt
from lsdviztools.lsdplottingtools import lsdmap_gdalio as gio
import pandas as pd
import lsdviztools.lsdmapwrappers as lsdmw
# plot the rainfall and the elevation swath together 
from lsdviztools.lsdplottingtools import lsdmap_swathplotting as LSDSP
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib import colors
from lsdviztools.lsdplottingtools import lsdmap_gdalio as LSDMap_IO
from lsdviztools.lsdplottingtools import lsdmap_basicplotting as LSDMap_BP
from lsdviztools.lsdplottingtools import lsdmap_pointtools as LSDMap_PD
from lsdviztools.lsdmapfigure.plottingraster import MapFigure
from lsdviztools.lsdmapfigure.plottingraster import BaseRaster
from lsdviztools.lsdmapfigure import plottinghelpers as Helper
import figure_specs_paper
import numpy as np
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Liberation Sans']
rcParams['font.size'] = 20

def plot_two_swaths(ax, fig, subplot_letter, mountain_name):
    DataDirectoryDem = f"./dems_to_process/{mountain_name}/input_data/"


    # fig = figure_specs_paper.CreateFigure(FigSizeFormat="JGR", AspectRatio=16./9.)
    # ax = fig.add_subplot()

    df = pd.read_csv(DataDirectoryDem+f"{mountain_name}_dem_swath.csv")
    distance = df["distance"].values/1000
    median_val = df["median_z"].values
    min_val = df["minimum_z"].values
    max_val = df["max_z"].values
    fq = df["first_quartile_z"].values
    tq = df["third_quartile_z"].values
    min_x_dem = distance.min()
    max_x_dem = distance.max()
    # Get the minimum and maximum distances
    X_axis_min = 0
    X_axis_max = distance[-1]
    n_target_tics = 5
    xlocs,new_x_labels = LSDMap_BP.TickConverter(X_axis_min,X_axis_max,n_target_tics)

    ax.fill_between(distance, fq, tq, facecolor='yellowgreen', alpha = 0.8, interpolate=True)
    ax.fill_between(distance, min_val, max_val, facecolor='yellowgreen', alpha = 0.2, interpolate=True)
    lns1=ax.plot(distance, median_val,"darkolivegreen", linewidth = 2, label='Elevation')
    ax.plot(distance, min_val,"darkolivegreen",distance,max_val,"darkolivegreen",linewidth = 0.5, linestyle = "dashdot")
    ax.plot(distance, fq,"darkolivegreen",distance,tq,"darkolivegreen",linewidth = 1, linestyle = "dotted")
    ax.set_ylabel('Elevation (m)', fontsize=20)

    ax2 = ax.twinx()

    df = pd.read_csv(DataDirectoryDem+f"{mountain_name}_original_rain_swath.csv")
    distance = df["distance"].values/1000
    median_val = df["median_z"].values
    min_val = df["minimum_z"].values
    max_val = df["max_z"].values
    fq = df["first_quartile_z"].values
    tq = df["third_quartile_z"].values

    ax2.fill_between(distance, fq, tq, facecolor='lightsteelblue', alpha = 0.8, interpolate=True, zorder=0)
    ax2.fill_between(distance, min_val, max_val, facecolor='lightsteelblue', alpha = 0.2, interpolate=True, zorder=0)
    lns2=ax2.plot(distance, median_val,"navy", linewidth = 2,label = 'Rainfall')
    ax2.plot(distance, min_val,"navy",distance,max_val,"navy",linewidth = 0.5, linestyle = "dashdot", zorder=1)
    ax2.plot(distance, fq,"navy",distance,tq,"navy",linewidth = 1, linestyle = "dotted", zorder=1)
    min_x_rainfall = distance.min()
    max_x_rainfall = distance.max()

    all_x_min= np.max([min_x_rainfall, min_x_dem])
    all_x_max = np.min([max_x_rainfall, max_x_dem])
    ax2.set_xlim([all_x_min,all_x_max])
    ax.set_xlabel('Distance along swath (km)', fontsize=20)
    ax2.set_ylabel('Rainfall (m/yr)', fontsize=20)
    #plt.title(f'{mountain_title}')

    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    legend=ax2.legend(lns, labs, loc=0, fontsize=18,).set_zorder(2)

    ax.margins(x=0, y=0)
    ax.text(-0.3, 0.9, subplot_letter, transform=ax.transAxes, 
                weight='normal')
    plt.tight_layout()
    #plt.show()
    plt.savefig(DataDirectoryDem+f'{mountain_name}_two_swaths.jpg',format='jpg',dpi=500, bbox_inches='tight')

mountain_name =  ['peru','argentina', 'turkey', 'xian','colorado','alburz_south', 'pyrenees', 'massif_central']

for mountain in mountain_name:
    map_fig = figure_specs_paper.CreateFigure(FigSizeFormat="JGR", AspectRatio=7./5., size_font=20)
    map_ax = plt.subplot()
    plot_two_swaths(map_ax, map_fig, 'C', mountain)