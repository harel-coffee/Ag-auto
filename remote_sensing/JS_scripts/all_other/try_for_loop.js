
var three_grant_regions = ee.FeatureCollection(grant_three_SF);
var first_grant_region = ee.FeatureCollection(first_grant_SF);
var second_grant_region = ee.FeatureCollection(second_grant_SF);
var third_grant_region = ee.FeatureCollection(third_grant_SF);


print(three_grant_regions.geometry())
print(first_grant_region.geometry())

////////////////////////////////////////////////////////////////////////////////////////
///
///                           functions definitions start
///
var addNDVI = function(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
};

var add_NDVI_IC = function(image_IC){
  var NDVI_IC = image_IC.map(addNDVI);
  return NDVI_IC;
}

var extract_sentinel_IC = function(a_feature){
    // var start_date = 
    // var end_date = 
    // var cloud_percentage = 
    // var geom = ee.Feature(feature_col).geometry();
    var geom = a_feature.geometry();
    var image = ee.ImageCollection('COPERNICUS/S2')
                .filterDate('2019-01-01', '2019-12-31')
                .filterBounds(geom)
                .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 10)
                .select(['B8', 'B4']);
  return image;
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
////////////////////////////////////////////////////////////////////////////////////////
///
///
///
var second_field_image = ee.ImageCollection('COPERNICUS/S2')
                        .filterDate('2019-01-01', '2019-12-31')
                        .filterBounds(second_grant_region)
                        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 10)
                        .select(['B8', 'B4']);
                
print ("second_field_image", second_field_image.size().getInfo())
////////////////////////////////////////////////////////////////////////////////////////
///
///      Third field - get it directly
///
var third_field_image = ee.ImageCollection('COPERNICUS/S2')
                        .filterDate('2019-01-01', '2019-12-31')
                        .filterBounds(third_grant_region)
                        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 10)
                        .select(['B8', 'B4']);
                        
print ("third_field_image", third_field_image.size().getInfo())
print(third_field_image)

////////////////////////////////////////////////////////////////////////////////////////
///
///      Third field - get it via map() and extract_sentinel_IC
///
var third_field_image_2 = third_grant_region.map(extract_sentinel_IC)
print ("third_field_image_2", third_field_image_2.size().getInfo())
print(third_field_image_2.first())

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


