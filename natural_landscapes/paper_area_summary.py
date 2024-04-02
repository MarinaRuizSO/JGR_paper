

import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
import cartopy.crs as ccrs

from paper_plot_rainfall_maps import plot_rainfall_map
from glim_finalise_plots import plot_glim
from read_and_plot_2_swaths import plot_two_swaths
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Keep this or the code breaks because the HDF files aren't closed properly.
import figure_specs_paper
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Liberation Sans']
rcParams['font.size'] = 8

import matplotlib.gridspec as gridspec

# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2,height_ratios=[3,1], width_ratios=[2,0.5])
map_fig=figure_specs_paper.CreateFigure(FigSizeFormat="JGR", AspectRatio=16./9.)
#map_fig, map_ax = plt.subplots(2,2, dpi=500, projection='20S') #width, height in inches
crs = ccrs.UTM('20S')
ax1 = plt.subplot(gs[0, 0]) # row 0, col 0

ax2 = plt.subplot(gs[0, 1], projection=crs) # row 0, col 1

ax3 = plt.subplot(gs[1, :]) # row 1, span all columns

mountain = 'argentina'

#map_fig, map_ax = plt.subplots(2,2, dpi=500) #width, height in inches
#plt.subplots_adjust(wspace=0, hspace=0)

# run_full_ksn_delta_case_ia(map_ax[0][0],map_fig, 0,  'A')
# run_full_ksn_delta_case_iq(map_ax[0][1],map_fig, 1,  'B')
# run_full_ksn_delta_case_ii(map_ax[1][0],map_fig, 2,  'C')
plot_rainfall_map(ax1, map_fig, 'A')
plot_glim(ax2, map_fig, 'B')
plot_two_swaths(ax3, map_fig, 'C', mountain)
# #map_ax, map_fig, label_count, case_labels,cbar_labels, subplot_letter

#plt.tight_layout(h_pad=0)
base_path = '/exports/csce/datastore/geos/users/s1440040/LSDTopoTools/data/ExampleTopoDatasets/ChiAnalysisData/dems_to_process/'
# plt.show()
plt.savefig(base_path + f'{mountain}_area_summary' +'.jpg', dpi = 500, bbox_inches='tight',pad_inches = 0, facecolor='white')