

import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

from paper_plot_maps_delta_ksn_norm_no_rain import run_full_ksn_delta_case_ia
from paper_plot_maps_delta_ksn_norm import run_full_ksn_delta_case_iq
from paper_plot_maps_delta_ksn_case_ii_norm import run_full_ksn_delta_case_ii
from paper_plot_maps_delta_ksn_case_iii_norm import run_full_ksn_delta_case_iii
import cartopy.crs as ccrs

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Keep this or the code breaks because the HDF files aren't closed properly.
import figure_specs_paper
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Liberation Sans']
rcParams['font.size'] = 8
plt.rcParams['lines.linewidth'] = 1  


cbar_labels = [
    r'$|k^{*}_{sn-q,\theta=0.45}- k^{*}_{sn-q,\theta=\theta_{best}}|$',
    r'$|k^{*}_{sn,\theta=0.45}- k^{*}_{sn,\theta=\theta_{best}}|$',
    r'$|k^{*}_{sn,\theta=0.45}- k^{*}_{sn-q,\theta=0.45}|$',
    r'$|k^{*}_{sn,\theta=\theta_{best}}- k^{*}_{sn-q,\theta=\theta_{best}}|$']
case_labels = [
    r'Case $i_{{Q}}$',
    r'Case $i_{{A}}$',
    r'Case ii',
    r'Case iii'
]
label_count = 0
ss_min = []
ss_max = []
#mountain = 'argentina'
crs_list_cartopy = ['19S', '20S', '37N', '48N',  '13N','39N','31N', '31N']
file_path = '/exports/csce/datastore/geos/users/s1440040/LSDTopoTools/data/ExampleTopoDatasets/ChiAnalysisData/dems_to_process/'
DEM_paths = [file_path+'peru/input_data/', file_path+'argentina/input_data/', file_path+'turkey/input_data/',file_path+'xian/input_data/', file_path+'colorado/input_data/', file_path+'alburz_south/input_data/', file_path+'massif_central/input_data/',file_path+'pyrenees/input_data/']# complete the list later - first try with just two cases
file_names = ['peru_dem.bil', 'argentina_dem.bil', 'turkey_dem.bil',
              'xian_dem.bil', 'colorado_dem.bil', 'alburz_south_dem.bil',
              'massif_central_dem.bil', 'pyrenees_dem.bil']
title_name = ['Andes, Southern Perú', 'Andes, Northern Argentina',
               'Kaçkar Mts, Turkey', 'North Qinling Mts, China',
                'Southern Rockies, USA','Alburz Mts, Iran', 
                'Massif Central, France', 'Pyrénées, Spain-France']

mountain_range_names =  ['peru', 'argentina', 'turkey', 'xian', 'colorado', 'alburz_south', 'massif_central', 'pyrenees']
count = 0
for mountain in mountain_range_names:
    base_path = '/exports/csce/datastore/geos/users/s1440040/LSDTopoTools/data/ExampleTopoDatasets/ChiAnalysisData/dems_to_process/'

    crs = ccrs.UTM(crs_list_cartopy[count])
    #map_ax = plt.subplot(projection=crs)
    map_fig, map_ax = plt.subplots(2,2, dpi=500, subplot_kw={'projection': crs}) #width, height in inches
    plt.subplots_adjust(wspace=0, hspace=0)

    run_full_ksn_delta_case_ia(map_ax[0][1],map_fig, 1,case_labels,cbar_labels,  'B', count, DEM_paths, file_names, mountain_range_names, base_path)
    run_full_ksn_delta_case_iq(map_ax[0][0],map_fig, 0,case_labels,cbar_labels,  'A', count, DEM_paths, file_names, mountain_range_names, base_path)
    run_full_ksn_delta_case_ii(map_ax[1][0],map_fig, 2,case_labels,cbar_labels,  'C', count, DEM_paths, file_names, mountain_range_names, base_path)
    run_full_ksn_delta_case_iii(map_ax[1][1],map_fig, 3,case_labels,cbar_labels,  'D', count, DEM_paths, file_names, mountain_range_names, base_path)

    plt.tight_layout(h_pad=0)
    #plt.show()
    plt.savefig(base_path + f'{mountain}_ksn_cases_maps' +'.jpg', dpi = 500, bbox_inches='tight',pad_inches = 0, facecolor='white')
    count+=1