# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="na5xv2yL4niM"
# #USE gee to show the location and relative data resources for WWS project

# %% [markdown] id="jofcKmYp4-gT"
# ##Import the gee API

# %% id="nnn-D9Td454d"
import ee

# %% [markdown] id="jDMqjDio5LSs"
# ##Authenticate and initialize

# %% colab={"base_uri": "https://localhost:8080/"} id="Inu_JejX5SiE" outputId="c3441e6a-a8f7-4bf4-e2a2-d37c94f30ad3"
ee.Authenticate()
#Initialize the library
ee.Initialize()

from google.colab import drive
drive.mount('/content/drive')

# %% [markdown] id="CWnLyvuUjKST"
# ##Folium package

# %% colab={"base_uri": "https://localhost:8080/"} id="yfDoErsEjIAd" outputId="fc02020f-ddc8-45c2-c121-02f711ae035b"
#Need update folium to use scalecontrol
# #!pip install folium
try:
    import folium
except ImportError as e:
    # !pip install folium
    # #!pip install --upgrade folium

import folium
# !pip show folium

# %% [markdown] id="1KAG5zsm50cc"
# ##Test the API

# %% colab={"base_uri": "https://localhost:8080/"} id="4UnumVFj5457" outputId="c224cfde-b60a-4630-e636-6fb1dc1328e4"
# Print the elevation of Mount Everest.
dem30m = ee.Image('USGS/SRTMGL1_003')
xy = ee.Geometry.Point([86.9250, 27.9881])
elev = dem30m.sample(xy, 30).first().get('elevation').getInfo()
print('Mount Everest elevation (m):', elev)

# %% [markdown] id="7buCnWNOCjVP"
# ##Set boundary for WWS project

# %% id="_roHnzzCCn3-"
xmin = -126.0
ymin = 41.5
xmax = -110.5
ymax = 49.0
#for display purposes
wwsbnd = ee.Geometry.Rectangle([xmin,ymin,xmax,ymax], 'EPSG:4326')
wwscenter = ee.Geometry.Point((xmin+xmax)/2.0,(ymin+ymax)/2.0)
# dem 
dem10m = ee.Image('USGS/3DEP/10m').clip(wwsbnd)
# State boundary
statebnd = ee.FeatureCollection("TIGER/2016/States").filterBounds(wwsbnd)
# county boundary 
countybnd = ee.FeatureCollection(ee.FeatureCollection("TIGER/2016/Counties")).filterBounds(wwsbnd)
# MTBS burnt area
MTBS = ee.FeatureCollection("USFS/GTAC/MTBS/burned_area_boundaries/v1").filterBounds(wwsbnd)
# NLCD 2019
nlcd_years = [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019]
NLCD = dict()
for year in nlcd_years:
  NLCD[year] = ee.ImageCollection("USGS/NLCD_RELEASES/2019_REL/NLCD").filter(ee.Filter.eq('system:index', str(year))).first().select('landcover').clip(wwsbnd)
# Global Surface Water Mapping Layers
surfacew = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(wwsbnd)
# HUC 8
HUC08 = ee.FeatureCollection("USGS/WBD/2017/HUC08").filterBounds(wwsbnd)
# HUC 10
HUC10 = ee.FeatureCollection("USGS/WBD/2017/HUC10").filterBounds(wwsbnd)
# HUC 12
HUC12 = ee.FeatureCollection("USGS/WBD/2017/HUC12").filterBounds(wwsbnd)
# GagesII
gagesII = ee.FeatureCollection("projects/ee-mingliangliuearth/assets/gagesII_9322_sept30_2011").filterBounds(wwsbnd)

# https://www.oregon.gov/deq/wq/programs/pages/dwp-maps.aspx
# Surface water drinking water source areas in Oregon
OR_SW_DWSAs = ee.FeatureCollection("projects/ee-mingliangliuearth/assets/OR_SW_DWSAs_ORLAMBERT_Ver10_15JAN2019")
#WA
WA_SW_DWSAs = ee.FeatureCollection("projects/ee-mingliangliuearth/assets/WA_drinking_Water_source_area")


#EPA reach ER1-2
StreamEr1_2 = ee.FeatureCollection("projects/ee-mingliangliuearth/assets/Erf1_20") #.filterBounds(wwsbnd)

#WA WQ stations from WADOE
WA_WQ = ee.FeatureCollection("projects/ee-mingliangliuearth/assets/WA_WQ_points_WADOE_pro")
#OR WQ stations from ORDEQ
OR_WQ = ee.FeatureCollection("projects/ee-mingliangliuearth/assets/OR_WQ_points_ORDEQ_pro")

#McKenzie basin observations
MK_USGS_sites = ee.FeatureCollection("projects/ee-mingliangliuearth/assets/USGS_sites_McKenzie")
MK_EWEB_sites = ee.FeatureCollection("projects/ee-mingliangliuearth/assets/EWEB_sites_DOC")

#Forest to Fauset assessment
f2f2 = ee.FeatureCollection("projects/ee-mingliangliuearth/assets/f2f2_pnw")

#Gate creek subwatershed
GateCreekSubwatershed = ee.FeatureCollection("projects/ee-mingliangliuearth/assets/GateCreekSubbasin")
GateCreekReach = ee.FeatureCollection("projects/ee-mingliangliuearth/assets/Gate_creek_Reach_polyline")

global_simplify = 1000 #meter
#Transform the boundary geometry to the projection of the lines
#bounding_box_transformed = wwsbnd.transform(StreamEr1_2.first().projection())
#StreamEr1_2 = StreamEr1_2.filterBounds(bounding_box_transformed)

#projection = ee.Feature(HUC12.first()).projection()

# Print the projection information
#print('Projection:', projection.getInfo())

# %% id="fofi84-mBHyH"
default_show = False
default_overlay = True
default_control = True
from pickle import FALSE
# Import the Folium library.
# import folium
from folium import plugins
from folium.plugins import MeasureControl
from folium.plugins import BeautifyIcon
import json
import pandas as pd
# from folium.plugins import ScaleControl
# Define a method for displaying Earth Engine image tiles on a folium map.
def add_ee_layer(self, ee_object, vis_params, name):
    
    try:    
        # display ee.Image()
        if isinstance(ee_object, ee.image.Image):    
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = default_overlay,
            control = default_control,
            show = default_show
            ).add_to(self)
        # display ee.ImageCollection()
        elif isinstance(ee_object, ee.imagecollection.ImageCollection):    
            ee_object_new = ee_object.mosaic()
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = default_overlay,
            control = default_control,
            show = default_show
            ).add_to(self)
        # display ee.Geometry()
        elif isinstance(ee_object, ee.geometry.Geometry):    
            folium.GeoJson(
            data = ee_object.getInfo(),
            name = name,
            overlay = default_overlay,
            control = default_control,
            show = True
        ).add_to(self)
        # display ee.FeatureCollection()
        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):  
            ee_object_new = ee.Image().paint(ee_object, 0, 2)
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = default_overlay,
            control = default_control,
            show = True
        ).add_to(self)
    
    except:
        print("Could not display {}".format(name))
    
# Add EE drawing method to folium.
folium.Map.add_ee_layer = add_ee_layer

# Add custom basemaps to folium
basemaps = {
    'Google Maps': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Maps',
        overlay = default_overlay,
        control = default_control,
        show = default_show
    ),
    'Google Satellite': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = default_overlay,
        control = default_control,
        show = default_show
    ),
    'Google Terrain': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Terrain',
        overlay = default_overlay,
        control = default_control,
        show = True
    ),
    'Google Satellite Hybrid': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite Hybrid',
        overlay = default_overlay,
        control = default_control,
        show = default_show
    ),
    'Esri Satellite': folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = default_overlay,
        control = default_control,
        show = default_show
    )
}


# %% [markdown] id="RXXH4GL1eVJZ"
# ##Some basic functions

# %% id="lEhC2Bh7eWWa"
################################################################################
def get_map_TileLayer_names(my_map):
  layer_names = []
  for child in my_map._children.values():
    if isinstance(child, folium.TileLayer): 
      layer_names.append(child.tile_name)
  return layer_names
    
################################################################################
def list_map_childs(my_map):
  layer_names = []
  for child in my_map._children.values():
    print(child)
################################################################################
def display_map(my_map, add_basemaps=False):
  if add_basemaps == True:
      TileLayers = get_map_TileLayer_names(my_map)
      if 'Google Maps' not in TileLayers:
        basemaps['Google Maps'].add_to(my_map)
      if 'Google Satellite Hybrid' not in TileLayers:
        basemaps['Google Satellite Hybrid'].add_to(my_map)
      if 'Google Terrain' not in TileLayers:
        basemaps['Google Terrain'].add_to(my_map)
      if 'Esri Satellite' not in TileLayers:
        basemaps['Esri Satellite'].add_to(my_map)
    
  # Add a layer control panel to the map.
  # Check if the map has a layer control
  has_layer_control = any(isinstance(child, folium.LayerControl) for child in my_map._children.values())
  if not has_layer_control:
    my_map.add_child(folium.LayerControl())
    print("added LayerControl")

  # Add fullscreen button
  has_fullscreen_control = any(isinstance(child, plugins.Fullscreen) for child in my_map._children.values())
  if not has_fullscreen_control:
    plugins.Fullscreen().add_to(my_map)
    print("added Fullscreen")

  # Popup location
  #has_LatLngPopup_control = any(isinstance(child, folium.LatLngPopup) for child in my_map._children.values())
  #if not has_LatLngPopup_control:
  #  my_map.add_child(folium.LatLngPopup())
  #  print("added LatLngPopup")
  #Tool for measure
  has_MeasureControl_control = any(isinstance(child, MeasureControl) for child in my_map._children.values())
  if not has_MeasureControl_control:
    my_map.add_child(MeasureControl(position='bottomleft'))
    print("added MeasureControl")

  display(my_map)
  return my_map
################################################################################  
def simplify(feature):
    return feature.setGeometry(feature.geometry().simplify(maxError=30))
################################################################################
def simplify100(feature):
    return feature.setGeometry(feature.geometry().simplify(maxError=100))
################################################################################
def simplify500(feature):
    return feature.setGeometry(feature.geometry().simplify(maxError=500))
################################################################################
def simplify1000(feature):
    return feature.setGeometry(feature.geometry().simplify(maxError=1000))
################################################################################
#list field names
def get_featurecollection_fields(gsdata):
  fields = set()
  for feature in gsdata.limit(1).getInfo()['features']:
      properties = feature.get('properties', {})
      for field in properties.keys():
          fields.add(field)
  #print('MTBS fields:',fields)
  return fields
################################################################################
#select polygons from geojson
def get_polygons_from_geojson(gsdata):
# Iterate over the features
  #print(f"len_fc:{len(gsdata['features'])}")
  new_fc = list()
  for feature in gsdata['features']:
    if feature['geometry']['type'] == 'Polygon':
      #print(feature)
      new_fc.append(feature)
  #print(new_fc)
  gsdata['features'] = new_fc
  return gsdata
    # Check if the feature is a polygon
    #if feature['geometry']['type'] == 'Polygon':
    #    # Create a GeoJSON layer for the polygon feature
    #    folium.GeoJson(feature).add_to(m)
################################################################################
#plot polygon feature collection with popups
def add_feacturecollection_to_maplayer(fc,map,fstyle,name,simplify=0, fields = list(), block_size = 5000):
  def sp_simplify(feature):
    return feature.setGeometry(feature.geometry().simplify(maxError=simplify))
  if simplify > 0:
    fc = fc.map(sp_simplify)

  if len(fields) > 0:
    pop_fields = fields
  else:
    pop_fields = get_featurecollection_fields(fc)

  # Define the block size (e.g., 10000 features per block)
  #block_size = 5000
  # Get the total number of features
  total_features = fc.size().getInfo()
  # Calculate the number of blocks
  num_blocks = total_features // block_size
  if total_features % block_size != 0:
    num_blocks += 1
  if num_blocks > 1:
      print(f"name:{name} has num_blocks:{num_blocks}")
  # Iterate over the blocks and export to Google Drive
  for block_index in range(num_blocks):
    if num_blocks > 1:
      outname = name + '_' + str(block_index)
    else:
      outname = name
    start_index = block_index * block_size
    end_index = min((block_index + 1) * block_size, total_features)
    # Get the block of features
    block = fc.toList(block_size, start_index)

    tmp_geojson = ee.FeatureCollection(block).getInfo()
    tmp_geojson = get_polygons_from_geojson(tmp_geojson)

    highlight_function=lambda feature: {
                'fillColor': 'yellow',
                'fillOpacity': 0.1,
                'color': 'white',
                'weight': 1,
              }

    folium.GeoJson(tmp_geojson
                #,marker=markers['Ref']
                ,name=outname
                ,style_function=fstyle
                ,highlight_function=highlight_function
                ,popup=folium.GeoJsonPopup(fields=tuple(pop_fields))
                ).add_to(map)
  return map
################################################################################
def split_large_feacturecollection_to_blocks(fc,simplify=0,block_size=1000,sel_polygon = True):
  blocks = dict()
  def sp_simplify(feature):
    return feature.setGeometry(feature.geometry().simplify(maxError=simplify))
  if simplify > 0:
    fc = fc.map(sp_simplify)

  # Get the total number of features
  total_features = fc.size().getInfo()
  # Calculate the number of blocks
  num_blocks = total_features // block_size
  if total_features % block_size != 0:
    num_blocks += 1
  if num_blocks > 1:
      print(f"num_blocks:{num_blocks}")
  # Iterate over the blocks and export to Google Drive
  for block_index in range(num_blocks):
    start_index = block_index * block_size
    end_index = min((block_index + 1) * block_size, total_features)
    # Get the block of features
    block = fc.toList(block_size, start_index)
    print(f'block:{block_index} block_size:{len(block.getInfo())}')
    tmp_geojson = ee.FeatureCollection(block).getInfo()
    if sel_polygon:
      tmp_geojson = get_polygons_from_geojson(tmp_geojson)
    blocks[block_index] = tmp_geojson
  return blocks


# %% id="Ij0sZhRrLZf-"
#Seems not necessary, will try later.
def retrive_large_feature_collection(fc,simplify = False):
  # Get the total size of the collection
  total_size = fc.size().getInfo()
  print(f'total_size:{total_size}')
  
  # Convert the feature collection to a list
  feature_list = fc.toList(fc.size())
  features = []
  for i in range(feature_list.size().getInfo()):
    feature = ee.Feature(feature_list.get(i))
    if simplify:
      feature = simplify100(feature)
    features.append(feature)

  # Convert the elements_list to a single ee.FeatureCollection
  retrieved_collection = ee.FeatureCollection(features)
  #print(f' out_fc_size:{retrieved_collection.size().getInfo()}')
  return retrieved_collection



# %% [markdown] id="7VO_756NBL1Y"
# ###Interactive map

# %% id="AjUd8wd8BO35"
# Set visualization parameters.
dem_vis_params = {
  'min': 0,
  'max': 4000,
  'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}

sw_visualization = {
  'bands': ['occurrence'],
  'min': 0.0,
  'max': 100.0,
  'palette': ['ffffff', 'ffbbbb', '0000ff']}

mtbs_visParams = {
  'fillColor': 'ff8a50',
  'color': 'ff5722',
  'width': 1.0}  

# %% [markdown] id="h-qFi5s6U2t6"
#

# %% [markdown] id="wy248AQETPTP"
# ##Add MTBS data

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="GMNLUGkxTUJV" outputId="1d7474fc-0488-44fe-c1c5-9761729183bf"
from numpy.lib.shape_base import tile
# Create a folium map object.
from shapely.geometry import shape, GeometryCollection
geometry = GeometryCollection([shape(feature['geometry']) for feature in GateCreekSubwatershed.getInfo()['features']])
centroid = geometry.centroid
centroid_coords = [centroid.y, centroid.x]
#my_map = folium.Map(location=[wwscenter.coordinates().get(1).getInfo(),wwscenter.coordinates().get(0).getInfo()], zoom_start=7, tiles=None)
my_map = folium.Map(location=centroid_coords, zoom_start=10, tiles=None)
basemaps['Google Maps'].add_to(my_map)
basemaps['Google Satellite Hybrid'].add_to(my_map)
basemaps['Google Terrain'].add_to(my_map)
basemaps['Esri Satellite'].add_to(my_map)

#06022023
"""
for tyear in range(2010,2011):
  tile_layer_url = naip_layer_url #gsh_tile_layer.replace("year", f'{tyear}')  # Replace "2020" with your desired year
  print(f'{tyear}:{tile_layer_url}')
  attr = 'USDA NAIP Imagery'
  name = f'USGS National Map {tyear}'
  # Add the customized tile layer to the map object
  folium.raster_layers.WmsTileLayer(tile_layer_url, layers = "0", attr=attr, name=name, overlay=True, control=True,show=False).add_to(my_map)
"""

#Define a function to convert Unix Epoch Time to a readable string
import datetime
#add new time field
def add_time_ymd(feat):
    time_ymd = ee.Date(feat.get('Ig_Date')).format('YYYY-MM-dd')
    newtime=feat.set("Ig_TimeYMD",time_ymd)
    return newtime

MTBS = MTBS.map(add_time_ymd)

#simplify
#maxError is in meters

#MTBS = MTBS.map(simplify500)
 
start_date = ee.Date('2010-01-01')
end_date = ee.Date('2020-12-31')
#mill second after 2010
start_unix_epoch_timestamp = int(start_date.getInfo()['value'])
end_unix_epoch_timestamp = int(end_date.getInfo()['value'])

fire_size_ac_gt = 10000 #10000 #5000
selected_MTBS = MTBS.filter(ee.Filter.eq('Incid_Type', 'Wildfire')) \
                    .filter(ee.Filter.gt('BurnBndAc', fire_size_ac_gt)) \
                    .filter(ee.Filter.gt('Ig_Date',start_unix_epoch_timestamp)) \
                    .filter(ee.Filter.lt('Ig_Date', end_unix_epoch_timestamp))
#print('MTBS fields:', get_featurecollection_fields(selected_MTBS))

fstyle = lambda feature: {    
               'fillColor': '#FFA07A',           
               'color':'red', 
               'weight': 1,
               'dashArray': '4, 4'}
mtbs_pop = ['Ig_TimeYMD','Incid_Type','BurnBndAc','Incid_Name','Map_ID']
my_map = add_feacturecollection_to_maplayer(selected_MTBS,my_map,fstyle,'MTBS Burnt Area',global_simplify,mtbs_pop)
print(f"selected_MTBS:{selected_MTBS.size().getInfo()}")
#my_map = add_feacturecollection_to_maplayer(MTBS,my_map,fstyle,'MTBS Burnt Area')
my_map = display_map(my_map,True)

#list_map_childs(my_map)

# %% [markdown] id="Qxpvt5hkT1KD"
# ##Add Oregon and Washington drinking water source areas

# %% id="uJBh1iAIT6bS"
#add Oregon surface & ground drinking water source areas in Oregon
#list field names

#my_map = folium.Map(location=[wwscenter.coordinates().get(1).getInfo(),wwscenter.coordinates().get(0).getInfo()], zoom_start=7, tiles=None)

#OR_SW_DWSAs = OR_SW_DWSAs.map(simplify500)
#print('OR_SW_DWSAs fields:',get_featurecollection_fields(OR_SW_DWSAs))
fstyle = lambda feature: {    
               'fillColor': '#87CEFA',           
               'color':'#1E90FF', 
               'weight': 2,
               'dashArray': '2, 2'}
#or_wq_pop = []
my_map = add_feacturecollection_to_maplayer(OR_SW_DWSAs,my_map,fstyle,'Surface water source OR',global_simplify)

#print(OR_SW_DWSAs_geojson)
#OR_SW_DWSAs_geojson = get_polygons_from_geojson(OR_SW_DWSAs_geojson)

#add WA surface drinking water source areas in Oregon
#list field names
#print('WA_SW_DWSAs fields:',get_featurecollection_fields(WA_SW_DWSAs))

#WA_SW_DWSAs = WA_SW_DWSAs.map(simplify500)
my_map = add_feacturecollection_to_maplayer(WA_SW_DWSAs,my_map,fstyle,'Surface water source WA',global_simplify)

#my_map = display_map(my_map,False)
#list_map_childs(my_map)

# %% [markdown] id="QTUSeI0pAi57"
# ##Add USGS Gages map

# %% colab={"base_uri": "https://localhost:8080/"} id="S4rao7OzLTTF" outputId="550b1fdd-06c7-4a47-8e8b-2e7598511736"
# Load the USGS stream gauges observation data

#my_map = folium.Map(location=[wwscenter.coordinates().get(1).getInfo(),wwscenter.coordinates().get(0).getInfo()], zoom_start=7, tiles=None)

gages_feilds = get_featurecollection_fields(gagesII)

ref_gages = gagesII.filter(ee.Filter.eq('CLASS', 'Ref'))
noref_gages_gt100km = gagesII.filter(ee.Filter.neq('CLASS', 'Ref')).filter(ee.Filter.gt('DRAIN_SQKM', 250))
noref_gages_lt100km = gagesII.filter(ee.Filter.neq('CLASS', 'Ref')).filter(ee.Filter.lt('DRAIN_SQKM', 250))
#count = ref_gages.size().getInfo()
print('Number of Gages Ref:', ref_gages.size().getInfo(), 
      ' NoRef (>250km2):',noref_gages_gt100km.size().getInfo(),
      ' NoRef (<250km2):',noref_gages_lt100km.size().getInfo())

#layer = folium.GeoJson(ref_gages.getInfo())

markers = {'Ref' : folium.CircleMarker(radius = 8, # Radius in metres
                                 weight = 0, #outline weight
                                 fill_color = 'blue', 
                                 fill_opacity = 1),
           'NoRefgt250km2' : folium.CircleMarker(radius = 6, # Radius in metres
                                 weight = 0, #outline weight
                                 fill_color = 'black', 
                                 fill_opacity = 1),
           'NoReflt250km2' : folium.CircleMarker(radius = 4, # Radius in metres
                                 weight = 0, #outline weight
                                 fill_color = 'black', 
                                 fill_opacity = 1),
           'WQ' : folium.CircleMarker(radius = 8, # Radius in metres
                                 weight = 2, #outline weight
                                 fill_color = 'pink', 
                                 fill_opacity = 1),
           'MK_USGS': folium.CircleMarker(radius = 6, # Radius in metres
                                 weight = 1, #outline weight
                                 fill_color = '#050c12', 
                                 fill_opacity = 1),
           'MK_EWEB': folium.CircleMarker(radius = 8, # Radius in metres
                                 weight = 3, #outline weight
                                 fill_color = 'red', 
                                 fill_opacity = 0.7)}

folium.GeoJson(ref_gages.getInfo()
               ,marker=markers['Ref']
               ,name='GAGES-II Ref'
               ,tooltip=folium.GeoJsonTooltip(fields=('STAID',))
               ,popup=folium.GeoJsonPopup(fields=tuple(gages_feilds))).add_to(my_map)
folium.GeoJson(noref_gages_gt100km.getInfo()
               ,marker=markers['NoRefgt250km2']
               ,name='NoRef > 250km2'
               ,tooltip=folium.GeoJsonTooltip(fields=('STAID',))
               ,popup=folium.GeoJsonPopup(fields=tuple(gages_feilds))).add_to(my_map)
folium.GeoJson(noref_gages_lt100km.getInfo()
               ,marker=markers['NoReflt250km2']
               ,name='NoRef < 250km2'
               ,tooltip=folium.GeoJsonTooltip(fields=('STAID',))
               ,popup=folium.GeoJsonPopup(fields=tuple(gages_feilds))).add_to(my_map)

# water quality gage station
# square marker

#format the date column (start and end date)
def add_start_date_ymd(feat):
    time_ymd = ee.Date(feat.get('start_date')).format('YYYY-MM-dd')
    newtime=feat.set("start_date_YMD",time_ymd)
    return newtime
def add_end_date_ymd(feat):    
    time_ymd = ee.Date(feat.get('end_date')).format('YYYY-MM-dd')
    newtime=feat.set("end_date_YMD",time_ymd)
    return newtime
WA_WQ = WA_WQ.map(add_start_date_ymd)
WA_WQ = WA_WQ.map(add_end_date_ymd)
OR_WQ = OR_WQ.map(add_start_date_ymd)
OR_WQ = OR_WQ.map(add_end_date_ymd)

wa_wq_fields = get_featurecollection_fields(WA_WQ)
or_wq_fields = get_featurecollection_fields(OR_WQ)

folium.GeoJson(WA_WQ.getInfo()
               ,marker=markers['WQ']
               ,name='WA_WQ WADOE'
               ,tooltip=folium.GeoJsonTooltip(fields=('Location_N',))
               ,popup=folium.GeoJsonPopup(fields=tuple(wa_wq_fields))).add_to(my_map)
folium.GeoJson(OR_WQ.getInfo()
               ,marker=markers['WQ']
               ,name='OR_WQ ORDEQ'
               ,tooltip=folium.GeoJsonTooltip(fields=('monitori_1',))
               ,popup=folium.GeoJsonPopup(fields=tuple(or_wq_fields))).add_to(my_map)

#my_map = display_map(my_map,False)

# %% [markdown] id="uqcsbYyyJngW"
# ###Fields observations over McKenzie basin

# %% colab={"base_uri": "https://localhost:8080/"} id="Y3NCqmrnKL-_" outputId="db5eceeb-84b1-407f-e91d-96476b85488e"
#MK_USGS_sites = ee.FeatureCollection("projects/ee-mingliangliuearth/assets/USGS_sites_McKenzie")
#MK_EWEB_sites = ee.FeatureCollection("projects/ee-mingliangliuearth/assets/EWEB_sites_DOC")
MK_USGS_sites_fields = get_featurecollection_fields(MK_USGS_sites)
MK_EWEB_sites_fields = get_featurecollection_fields(MK_EWEB_sites)
#print(f"MK_USGS_sites_fields:{MK_USGS_sites_fields} \n MK_EWEB_sites_fields:{MK_EWEB_sites_fields}")

folium.GeoJson(MK_USGS_sites.getInfo()
               ,marker=markers['MK_USGS']
               ,name='McKenzie (USGS)'
               ,tooltip=folium.GeoJsonTooltip(fields=('STANAME',))
               ,popup=folium.GeoJsonPopup(fields=tuple(MK_USGS_sites_fields))).add_to(my_map)
folium.GeoJson(MK_EWEB_sites.getInfo()
               ,marker=markers['MK_EWEB']
               ,name='McKenzie (EWEB)'
               ,tooltip=folium.GeoJsonTooltip(fields=('site_name',))
               ,popup=folium.GeoJsonPopup(fields=tuple(MK_EWEB_sites_fields))).add_to(my_map)


# %% [markdown] id="eNWRCh8tK2hv"
# ## HUC units, State, and county boundary

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="gtfctROMUqhy" outputId="b47f084d-f4b5-4de8-842c-57d04345f22f"


#my_map = folium.Map(location=[wwscenter.coordinates().get(1).getInfo(),wwscenter.coordinates().get(0).getInfo()], zoom_start=7, tiles=None)
# Add the elevation model to the map object.
#my_map.add_ee_layer(dem10m.updateMask(dem10m.gt(0)), dem_vis_params, 'DEM')
#NLCD 2019
for year in nlcd_years:
  my_map.add_ee_layer(NLCD[year], {}, 'NLCD' + str(year))
#JRC surface water occurrence
my_map.add_ee_layer(surfacew, sw_visualization, 'Waterbody Occurrence') 
sim_huc = 1000 #meter

#Add HUBs
my_map.add_ee_layer(HUC12, {'palette': ['9fc2e0']}, 'HUC12')
#my_map.add_ee_layer(HUC10, {'palette': ['397bb3']}, 'HUC10')
my_map.add_ee_layer(HUC08, {'palette': ['152e43']}, 'HUC08')
huc12_fstyle = lambda feature: {    
               'fillColor': '#87CEFA',   
               'fillOpacity': 0,        
               'color':'#679fce', 
               'weight': 0.5,
               'dashArray': '2, 1'}
huc10_fstyle = lambda feature: {    
               'fillColor': '#87CEFA',   
               'fillOpacity': 0,        
               'color':'#3777ac', 
               'weight': 1,
               'dashArray': '2, 2'}
huc08_fstyle = lambda feature: {    
               'fillColor': '#87CEFA',   
               'fillOpacity': 0,        
               'color':'#234c6f', 
               'weight': 1.5,
               'dashArray': '2, 3'}
#huc12_large = retrive_large_feature_collection(HUC12,True)
#print(f"huc12_large:{huc12_large.size().getInfo()}")
#huc12_pop = ['name','huc12','areasqkm']
huc10_pop = ['name','huc10','areasqkm']
huc08_pop = ['name','huc8','areasqkm']
#print(f"huc12:{HUC12.size().getInfo()}")
#my_map = add_feacturecollection_to_maplayer(HUC12,my_map,huc12_fstyle,'HUC12pop',3000,huc12_pop)
#HUC10 = HUC10.map(simplify500)
#print(f"huc10:{HUC10.size().getInfo()}")
#my_map = add_feacturecollection_to_maplayer(HUC10,my_map,huc10_fstyle,'HUC10pop',sim_huc,huc10_pop)
print(f"huc08:{HUC08.size().getInfo()}")
my_map = add_feacturecollection_to_maplayer(HUC08,my_map,huc08_fstyle,'HUC08pop',sim_huc,huc08_pop)
print(f"HUC08 fields:{get_featurecollection_fields(HUC08)}")
print("Adding state & county...")

#Add state boundary
my_map.add_ee_layer(statebnd, {'palette': ['000000']}, 'US States')
#Add county boundary
cty_fstyle = lambda feature: {           
               'color':'black', 
               'fillOpacity': 0,
               'weight': 2,
               'dashArray': '1, 1'}
#my_map.add_ee_layer(countybnd, {'palette': ['8B8878']}, 'US Counties')
#print(f"countybnd:{countybnd.size().getInfo()}")
#countybnd = countybnd.map(simplify500)
#print(f"countybnd fields:{get_featurecollection_fields(countybnd)}")
cty_pop = ['NAME','COUNTYFP','STATEFP']
my_map = add_feacturecollection_to_maplayer(countybnd,my_map,cty_fstyle,'US Counties',global_simplify,cty_pop)



#Stream
#my_map.add_ee_layer(StreamEr1_2, {'palette': ['#00008B']}, 'Stream')

#Add Gate Creek
#gc_fstyle = lambda feature: {    
#               'fillColor': '#87CEFA',   
#               'fillOpacity': 0,        
#               'color':'#EE3B3B', 
#               'weight': 3.0,
#               'dashArray': '1, 1'}
#GateCreekSubwatershed = ee.FeatureCollection("projects/ee-mingliangliuearth/assets/GateCreek_Watersheds_polygon")
#GateCreekReach = ee.FeatureCollection("projects/ee-mingliangliuearth/assets/Gate_creek_Reach_polyline")
#my_map = add_feacturecollection_to_maplayer(GateCreekSubwatershed,my_map,gc_fstyle,'Gate Creek Subwatershed')
my_map.add_ee_layer(GateCreekReach, {'palette': ['#0d05f5']}, 'Gate Creek reach')
my_map.add_ee_layer(GateCreekSubwatershed, {'palette': ['#f50505']}, 'Gate Creek Subwatershed')


my_map = display_map(my_map,False)

# %% [markdown] id="PxFYNE5tJDl9"
# ## Forest to Faucet assessment results

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="hs3uicKcJAKg" outputId="d481e343-17a0-49a6-9b85-406ec8da8d19"
f2f2_blocks = split_large_feacturecollection_to_blocks(f2f2, 2000, block_size=4000, sel_polygon=True)

# List of column names of interest
# #!pip install geopandas
try:
  import geopandas as gpd
except ImportError as e:
  # !pip install geopandas
  import geopandas as gpd

f2f2_map = folium.Map(location=[wwscenter.coordinates().get(1).getInfo(),wwscenter.coordinates().get(0).getInfo()], zoom_start=7, tiles=None)
basemaps['Google Maps'].add_to(f2f2_map)
basemaps['Google Satellite Hybrid'].add_to(f2f2_map)
basemaps['Google Terrain'].add_to(f2f2_map)
basemaps['Esri Satellite'].add_to(f2f2_map)
for year in nlcd_years:
  f2f2_map.add_ee_layer(NLCD[year], {}, 'NLCD' + str(year))

# Get the list of column names from the Feature Collection
column_names = get_featurecollection_fields(f2f2)
#print(column_names)

per_scale = [0,15,30,50,70,85,100]
#print(myscale)

#map_imp = ['IMP_R','WFP_IMP_R']
map_imp = ['WFP_IMP_R']
max_blocks = 10
for imp in map_imp:
  if imp == 'IMP_R':
    legend_name='Important Areas for Surface Drinking Water (0-100 Quantiles)'
  elif imp == 'WFP_IMP_R':
    legend_name='Wildfire Threat to Important Surface Drinking Water Watersheds (0-100 Quantiles)'
  for block in f2f2_blocks:
    if block < max_blocks:
      gdf = gpd.GeoDataFrame.from_features(f2f2_blocks[block]['features'])
      gdf.crs = 'EPSG:4326'
      if len(f2f2_blocks) <= 1:
        name = imp
      else:
        name = f'{imp}_p{block}'
      tc = folium.Choropleth(geo_data=gdf,
                        name=name,
                        data=gdf,
                        columns=['HUC12', imp],
                        key_on='feature.properties.HUC12',
                        fill_color='YlOrRd',
                        threshold_scale=per_scale,
                        fill_opacity=0.7,
                        line_opacity=0.1,
                        legend_name=legend_name,
                        #popup=folium.GeoJsonPopup(fields=tuple(column_names))
                        )
      if block != 0:
        for key in tc._children:
          if key.startswith('color_map'):
              del(tc._children[key])
      tc.add_to(f2f2_map)

      # Create a GeoJson layer with popups
      geojson = folium.GeoJson(
            f2f2_blocks[block],
            name='popup_' + name,
            style_function=lambda feature: {
                'fillColor': 'YlGnBu',
                'fillOpacity': 0,
                'color': 'black',
                'weight': 0.1
            },
            highlight_function=lambda feature: {
                'fillColor': 'YlGnBu',
                'fillOpacity': 0.1,
                'color': 'white',
                'weight': 0.3,
            },
            tooltip=folium.features.GeoJsonTooltip(
            #popup=folium.GeoJsonPopup(
                #fields=['NAME','APCW_R', 'GW', 'SW', 'IMP_R','WFP_IMP_R'],
                fields=tuple(column_names),
                #aliases=['HUC12', 'Ability to Produce Clean Water'
                #         , 'Number of groundwater water intakes'
                #         ,'Number of surface water intakes'
                #         ,'Important Areas for Surface Drinking Water'
                #         ,'Wildfire Threat to Important Surface Drinking Water Watersheds'],
                style='background-color: white; color: #333333; font-weight: bold;',
                sticky=False
            )
      ).add_to(f2f2_map)

f2f2_map.add_ee_layer(HUC12, {'palette': ['01090f']}, 'HUC12')
f2f2_map = display_map(f2f2_map,False)

# %% [markdown] id="3GAdoGdFU5GC"
# ##Save map as html

# %% id="p4dfE5LJU77C"
import datetime
# Get the current date
current_date = datetime.date.today()
# Convert the date to a string
current_date_str = current_date.strftime('%Y-%m-%d')
html1 = "/content/drive/MyDrive/WWS/Map_" + current_date_str + ".html"
my_map.save(html1)

#new_map.save("/content/drive/MyDrive/WWS/IMP_Map_" + current_date_str + ".html")

# %% id="Sb0nvzLuEqxE"
# Save the maps to separate HTML files
html2 = "/content/drive/MyDrive/WWS/f2f2_Map_" + current_date_str + ".html"
f2f2_map.save(html2)

# Open and read the contents of map1.html and map2.html
with open(html1, "r") as file1, open(html2, "r") as file2:
    map1_html = file1.read()
    map2_html = file2.read()

# Create a combined HTML file
combined_html = f"""
<html>
<head></head>
<body>
<div style="float: left; width: 50%;">
{map1_html}
</div>
<div style="float: left; width: 50%;">
{map2_html}
</div>
</body>
</html>
"""
html3 = "/content/drive/MyDrive/WWS/Combined_Map_" + current_date_str + ".html"
# Save the combined HTML to a file
with open(html3, "w") as combined_file:
    combined_file.write(combined_html)
