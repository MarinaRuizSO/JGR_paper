from rasterstats import zonal_stats
from pprint import pprint
# import modules
import osgeo.gdal as gdal
import rasterio as rio
from rasterio.features import shapes
from shapely.geometry import shape, Polygon, mapping
import fiona
import geopandas as gpd
import pandas as pd

def getNoDataValue(rasterfn):
    """This gets the nodata value from the raster

    Args:
        rasterfn (str): The filename (with path and extension) of the raster

    Returns:
        float: nodatavalue; the nodata value

    Author: SMM
    """
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    return band.GetNoDataValue()

def PolygoniseRaster(DataDirectory, RasterFile, OutputShapefile='polygons'):
    """
    This function takes in a raster and converts to a polygon shapefile using rasterio
    from https://gis.stackexchange.com/questions/187877/how-to-polygonize-raster-to-shapely-polygons/187883#187883?newreg=8b1f507529724a8488ce4789ba787363

    Args:
        DataDirectory (str): the data directory with the basin raster
        RasterFile (str): the name of the raster
        OutputShapefile (str): the name of the output shapefile WITHOUT EXTENSION. Default = 'polygons'

    Returns:
        Dictionary where key is the raster value and the value is a shapely polygon

    Author: FJC
    """
    

    # define the mask
    #mask = None
    raster_band = 1

    # get raster no data value
    NDV = getNoDataValue(DataDirectory+RasterFile)

    # load in the raster using rasterio
    with rio.open(DataDirectory+RasterFile) as src:
        image = src.read(raster_band, masked=False)

        msk = src.read_masks(1)

        results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v)
        in enumerate(
            shapes(image, mask=msk, transform=src.transform)))

    # define shapefile attributes
    print("Let me grab the coordinate reference system.")
    crs = src.crs
    print (crs)

    #crs = GetUTMEPSG(DataDirectory+RasterFile)
    schema = {'geometry': 'Polygon',
              'properties': { 'ID': 'float'}}



    # This is necessary to filter the basin results
    geoms = list(results)
    #print("Geom size is: "+str(len(geoms)))

    filtered_geoms = {}
    area_dict = {}
    for f in geoms:
        this_shape = Polygon(shape(f['geometry']))
        this_val = float(f['properties']['raster_val'])
        #print("ID is: "+str(this_val))
        this_area = this_shape.area
        if this_val in filtered_geoms.keys():
            print("Whoops. Found a repeated ID. Getting rid of the smaller one.")
            if area_dict[this_val] < this_area:
                filtered_geoms[this_val] = f
                area_dict[this_val] = this_area
                print("Found a repeated ID. Keeping the one with area of "+str(this_area))
            else:
                print("Keeping the initial ID.")
        else:
            filtered_geoms[this_val] = f
            area_dict[this_val] = this_area

    new_geoms = []
    for key,item in filtered_geoms.items():
        this_shape = Polygon(shape(item['geometry']))
        this_val = float(item['properties']['raster_val'])
        #print("ID is: "+str(this_val))
        this_area = this_shape.area
        #print("Area is: "+str(this_area))
        new_geoms.append(item)
    #print("Geom size is: "+str(len(new_geoms)))

    # transform results into shapely geometries and write to shapefile using fiona
    PolygonDict = {}

    # We use the internal fiona crs module since this is really buggy and depends on
    # the proj version
    print("I need to convert the crs to wkt format so it is resistant to stupid proj errors.")
    this_crs_in_wkt = crs.to_wkt()


    with fiona.open(DataDirectory+OutputShapefile, 'w', crs=this_crs_in_wkt, driver='ESRI Shapefile', schema=schema) as output:
        for f in new_geoms:
            this_shape = Polygon(shape(f['geometry']))
            this_val = float(f['properties']['raster_val'])
            print("ID is: "+str(this_val))
            if this_val != NDV: # remove no data values
                output.write({'geometry': mapping(this_shape), 'properties':{'ID': this_val}})
            PolygonDict[this_val] = this_shape

    return PolygonDict


#path = '/exports/csce/datastore/geos/users/s1440040/LSDTopoTools/data/ExampleTopoDatasets/ChiAnalysisData/dems_to_process/colorado/input_data/'

file_path = '/exports/csce/datastore/geos/users/s1440040/LSDTopoTools/data/ExampleTopoDatasets/ChiAnalysisData/'
DEM_paths = [file_path+'dems_to_process/massif_central/input_data/']#[file_path+'dems_to_process/peru/input_data/', file_path+'dems_to_process/argentina/input_data/', file_path+'dems_to_process/turkey/input_data/',file_path+'dems_to_process/xian/input_data/', file_path+'dems_to_process/colorado/input_data/', file_path+'dems_to_process/alburz_south/input_data/',file_path+'dems_to_process/pyrenees/input_data/', file_path+'dems_to_process/massif_central/input_data/']# complete the list later - first try with just two cases
mountain_range_names =  ['massif_central']#['peru','argentina', 'turkey', 'xian','colorado','alburz_south', 'pyrenees', 'massif_central']
file_names = ['massif_central_dem.bil']#['peru_dem.bil', 'argentina_dem.bil', 'turkey_dem.bil',
               #'xian_dem.bil', 'colorado_dem.bil', 'alburz_dem.bil',
                #'pyrenees_dem.bil','massif_central_dem.bil']
counter =0
litho_df = pd.read_csv(file_path+'glim_lithology_codes.csv', header=0)
litho_dict = dict(zip(list(litho_df.Number), list(litho_df.Code)))
for mountain in mountain_range_names:
    print(f'I am processing {mountain}')
    # # the order of the polygons is from lowest to highest ID number
    polygons_df = gpd.read_file(DEM_paths[counter]+'polygons/'+'polygons.shp')
    stats = zonal_stats(DEM_paths[counter]+'polygons/'+'polygons.shp', 
                        file_path+f'{mountain}_glim_crs_dem.tif', 
                        categorical=True, category_map = litho_dict) 
    stats_df = pd.DataFrame.from_dict(stats)
    stats_df['ID'] = polygons_df['ID']

    basins_lookup = pd.read_csv(file_path+f'{mountain}_basin_lookup_table.csv')
    stats_df['basin_number'] = basins_lookup['basin_number']
    stats_df['max'] = stats_df.idxmax(axis=1)

    stats_df.to_csv(DEM_paths[counter]+'litho_basin_stats.csv', index=False)
    counter+=1
    # will need to manually match the basin id with the basin numbers ... these are now in the {mountain}_basin_lookup_table.csv files