////////////////////////////////////////////////////////////////////////
/////
/////     We can read a shapefile with many polygons, and using map()
/////     read all images corresponding to a polygon. However, the result of
/////     that is a featureCollection for which we do not know how to
/////     access to the info. inside. So, we want to do a for-loop for each polygon 
/////     now.
/////
var three_grant_regions = ee.FeatureCollection(grant_three_SF);
var first_grant_region = ee.FeatureCollection(first_grant_SF);

print(first_grant_region.geometry())

////////////////////////////////////////////////////////////////////////////////////////
///
///                           functions definitions start
///
// Function to mask clouds using the Sentinel-2 QA band.
function maskS2clouds(image) {
    var qa = image.select('QA60')

    // Bits 10 and 11 are clouds and cirrus, respectively.
    var cloudBitMask = 1 << 10;
    var cirrusBitMask = 1 << 11;

    // Both flags should be set to zero, indicating clear conditions.
    var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
                        qa.bitwiseAnd(cirrusBitMask).eq(0))

    // Return the masked and scaled data, without the QA bands.
    return image.updateMask(mask).divide(10000)
                          .select("B.*")
                          .copyProperties(image, ["system:time_start"])
}


// add Day of Year to an image
var addDate_to_image = function(image){
  var doy = image.date().getRelative('day', 'year');
  var doyBand = ee.Image.constant(doy).uint16().rename('doy')
  doyBand = doyBand.updateMask(image.select('B8').mask())

  return image.addBands(doyBand);
};

// add Day of Year to an imageCollection
var addDate_to_collection = function(collec){
  var C = collec.map(addDate_to_image)
  return C;
};

// add NDVI to an image
var addNDVI_to_image = function(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
};

// add NDVI to an imageCollection
var add_NDVI_collection = function(image_IC){
  var NDVI_IC = image_IC.map(addNDVI_to_image);
  return NDVI_IC;
}

// Extract ImageCollection from Sentinel 2- level 1C 
// dates etc are hard-coded to enable us to use map() function
var extract_sentinel_IC = function(a_feature){
    // var start_date = 
    // var end_date = 
    // var cloud_percentage = 
    // var geom = ee.Feature(feature_col).geometry();
    var geom = a_feature.geometry();
    var imageC = ee.ImageCollection('COPERNICUS/S2')
                .filterDate('2019-01-01', '2019-12-31')
                .filterBounds(geom)
                //.filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 10)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                .select(['B8', 'B4']);
    imageC = imageC.map(addDate_to_image)
  return imageC;
}
///
///                           functions definitions end
///
////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////
///
///
///
var first_field_image = ee.ImageCollection('COPERNICUS/S2')
                        .filterDate('2019-01-01', '2019-12-31')
                        .filterBounds(first_grant_region)
                        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 10)
                        //.select(['B8', 'B4']);
                
print ("first_field_image", first_field_image.size().getInfo())

var first_field_image_Doy = first_field_image.map(addDate_to_image)
print(first_field_image_Doy)

////////////////////////////////////////////////////////////////////////////////////////
///
///      Third field - get it via map() and extract_sentinel_IC
///
var first_field_image_2 = first_grant_region.map(extract_sentinel_IC)
print ("first_field_image_2", first_field_image_2.size().getInfo())
print(first_field_image_2)



////////////////////////////////////////////////////////////////////////////////////////
///
///      Third field - get it via map() and extract_sentinel_IC
///
var first_field_image_2 = first_grant_region.map(extract_sentinel_IC)
print ("first_field_image_2", first_field_image_2.size().getInfo())
print(first_field_image_2.first())

////////////////////////////////////////////////////////////////////////////////////////
// image = image.map(function(image) { return image.clip(geom); });
// var three_fields_IC = extract_sentinel_IC(three_grant_regions)
// print ("three_fields_IC", three_fields_IC.size().getInfo())

var three_fields_IC = three_grant_regions.map(extract_sentinel_IC)
print ("three_fields_IC", three_fields_IC.size().getInfo())
print(three_fields_IC)
print(three_fields_IC.first())

print(three_fields_IC.getInfo())


Map.setCenter(-119, 47.2, 9)
var vizParams = {
  bands: ['B12'],
  min: 4000,
  max: 1000,
  gamma: 10
};

// Map.addLayer(three_grant_regions, vizParams)
// Map.addLayer(first_grant_region, vizParams)
// Map.addLayer(second_grant_region, vizParams)
Map.addLayer(three_grant_regions, vizParams)


