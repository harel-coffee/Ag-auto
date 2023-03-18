////////
////////     Functions
////////

var addNDVI = function(image) {

  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');

  return image.addBands(ndvi);

};

////////
////////     Parameters we need
////////


var needed_bands = ['B12', 'B11', 'B10', 'B9', 
                    'B8', 'B7', 'B6', 'B5', 'B4', 
                    'B3', 'B2', 'B1']

var start_date = '2018-03-01'
var end_date = '2018-10-01'
                    
print (needed_bands)


////////
////////     shape files uploaded into assets
////////

var fifty_double_field = ee.FeatureCollection(double_region);
var fifty_apple_field = ee.FeatureCollection(apple_region);
var fifty_potato_field = ee.FeatureCollection(potato_region);
var fifty_grape_field = ee.FeatureCollection(grape_region);
var fifty_alfalfa_field = ee.FeatureCollection(alfalfa_region);

print(fifty_double_field)


////////
////////     Collect Images
////////

////   double field

var fifty_double_field_imageCollection = ee.ImageCollection('COPERNICUS/S2')
    .filterDate( start_date, end_date )
    .filterBounds(fifty_double_field)
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 0.5)
    .select(needed_bands);

print(fifty_double_field_imageCollection)

var fifty_double_field_imageCollection = fifty_double_field_imageCollection.map(addNDVI);

print(fifty_double_field_imageCollection)
    
////   Apple field

var fifty_apple_field_imageCollection = ee.ImageCollection('COPERNICUS/S2')
    .filterDate(start_date, end_date)
    .filterBounds(fifty_apple_field)
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 0.5)
    .select(needed_bands);

print (fifty_apple_field_imageCollection)

////   Potato field

var fifty_potato_field_imageCollection = ee.ImageCollection('COPERNICUS/S2')
    .filterDate(start_date, end_date)
    .filterBounds(fifty_potato_field)
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 0.5)
    .select(needed_bands); 

print (fifty_potato_field_imageCollection)

////   Grape field

var fifty_grape_field_imageCollection = ee.ImageCollection('COPERNICUS/S2')
    .filterDate(start_date, end_date)
    .filterBounds(fifty_grape_field)
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 0.5)
    .select(needed_bands); 

print(fifty_grape_field_imageCollection)

////   Alfalfa field

var fifty_alfalfa_field_imageCollection = ee.ImageCollection('COPERNICUS/S2')
    .filterDate(start_date, end_date)
    .filterBounds(fifty_alfalfa_field)
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 0.5)
    .select(needed_bands); 
    
print(fifty_alfalfa_field_imageCollection)

/////////////
/////////////   Plots, charts, adding layers to map
/////////////


/////////////   Plots
//
//  The following part would not make sense. all fields are perhaps combinedd
// 
//

// var double_crop_NDVI_chart = ui.Chart.image.doySeriesByRegion(fifty_double_field_imageCollection, 
//                                                       "NDVI", 
//                                                       fifty_double_field, 
//                                                       ee.Reducer.mean());

// print(double_crop_NDVI_chart)



////////////////////////////////////////////////////
/////////////
/////////////   double crop field map 
/////////////
////////////////////////////////////////////////////

Map.setCenter(-119.448564, 46.1463602, 9)

// Map.setCenter(-119.639383, 46.2450093, 12)
var vizParams = {
  bands: ['B12'],
  min: 4000,
  max: 1000,
  gamma: 10
};

Map.addLayer(fifty_double_field, vizParams)

/////////////   apple crop field map 

// Map.setCenter(-119.393766, 46.2298139, 12)
Map.addLayer(fifty_apple_field, vizParams)

/////////////   potato crop field map 

// Map.setCenter(-119.312542, 45.9642573, 12)
Map.addLayer(fifty_potato_field, vizParams)

////////////////////////////////////////////////////////
//////////////
//////////////      EXPORT Data  
//////////////      if this is not pixel level, then what it is?
//////////////
////////////////////////////////////////////////////////

Export.table.toDrive({
  collection: fifty_double_field_imageCollection,
  description: 'fifty_double_field_imageCollection_2018_march_oct',
  fileFormat: 'CSV',
  folder:"Sentinel_Images_off_EE/fifty_fields/"
});

Export.table.toDrive({
  collection: fifty_apple_field_imageCollection,
  description: 'fifty_apple_field_imageCollection_2018_march_oct',
  fileFormat: 'CSV',
  folder:"Sentinel_Images_off_EE/fifty_fields/"

});


Export.table.toDrive({
  collection: fifty_potato_field_imageCollection,
  description: 'fifty_potato_field_imageCollection_2018_march_oct',
  fileFormat: 'CSV',
  folder:"Sentinel_Images_off_EE/fifty_fields/"
});


Export.table.toDrive({
  collection: fifty_grape_field_imageCollection,
  description: 'fifty_grape_field_imageCollection_2018_march_oct',
  fileFormat: 'CSV',
  folder:"Sentinel_Images_off_EE/fifty_fields/"
});


Export.table.toDrive({
  collection: fifty_alfalfa_field_imageCollection,
  description: 'fifty_alfalfa_field_imageCollection_2018_march_oct',
  fileFormat: 'CSV',
  folder:"Sentinel_Images_off_EE/fifty_fields/"
});


/////////////////////////////////////////////////////////////////////////////////
/////////
/////////      Pixel level off overlow
//
//  https://gis.stackexchange.com/questions/278533/extract-complete-pixel-values-inside-a-geometry?noredirect=1&lq=1
//
/////////////////////////////////////////////////////////////////////////////////


/////////
//Added code that was improved after Jon's comments
/////////


// generate a new image containing lat/lon of the pixel and reproject it to Landsat projection
print (fifty_alfalfa_field_imageCollection)

// var coordsImage = ee.Image.pixelLonLat().reproject(landsatDownload.projection());

// var joinedImage = coordsImage.addBands(landsatDownload);

// var valuesList = joinedImage.reduceRegion({
//   reducer: ee.Reducer.toList(4),
//   geometry: myGeometry
// }).values().get(0);

// print(valuesList);




