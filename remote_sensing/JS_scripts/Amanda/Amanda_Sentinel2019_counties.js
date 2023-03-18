// Load Sentinel 2 imagery, filter it to dates of interest, minimize clouds, 
    // get median composite, compute NDVI, export.
    // ***Note: Level-2A availability before 2019 is limited.
    // Drew rectangle delineating region of interest, named "geometry"

// Function to mask clouds using the Sentinel-2 QA band.
function maskS2clouds(image) {
  var qa = image.select('QA60');
  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
             qa.bitwiseAnd(cirrusBitMask).eq(0));
  // Return the masked and scaled data, without the QA bands.
  return image.updateMask(mask).divide(10000)
      .select("B.*")
      .copyProperties(image, ["system:time_start"]);
}

// Map the function over time period and take the median.
  // Load Sentinel-2 BOA reflectance data for dates of interest in AOI.
  var collection = ee.ImageCollection(Level2A)
      .filter(ee.Filter.bounds(geometry))
      .filterDate('2019-05-01', '2019-06-15')
      // Pre-filter to get less cloudy granules.
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
      .map(maskS2clouds);
  
  print('Collection: ', collection);
  
  // Get the number of images.
  var count = collection.size();
  print('Count: ', count);
  
  // Get the date range of images in the collection.
  var dates = ee.List(collection.get('date_range'));
  var dateRange = ee.DateRange(dates.get(0), dates.get(1));
  print('Date range available: ', dateRange);
      
  // Reduce the image collection by taking the median.
  var median = collection.median();
  
  // Load a Census Table of state boundaries and filter to Whitman & Garfield Counties.
var fc = ee.FeatureCollection('TIGER/2010/Blocks')
    .filter(ee.Filter.or(
         ee.Filter.eq('countyfp10', '075'),
         ee.Filter.eq('countyfp10', '023')));
  
  // clip the raster of median values
  var median_clip = median.clipToCollection(fc);

//Display the result of median for AOI in true color. 
// Can apply stretch to map layer: 100% stretch with gamma ~2 works well.
Map.setCenter(-117, 46.5, 8);
var visParam = {min: 0.0614, max: 0.3528, gamma: 2, bands: ['B4', 'B3', 'B2']};
Map.addLayer(median_clip, visParam, 'median true color');

// Display the result of median in color infrared (8,4,3).
// Can apply stretch to map layer: 100% stretch with gamma ~4 works well.
Map.setCenter(-117, 46.5, 8);
var CIRParam = {min: 0.0351, max: 0.3810, gamma: 4, bands: ['B8', 'B4', 'B3']};
Map.addLayer(median_clip, CIRParam, 'median color infrared');

// Compute Normalized Difference Vegetation Index.
// NDVI = (NIR - RED) / (NIR + RED), where
// RED is B4
// NIR is B8

// Use the normalizedDifference(A, B) to compute (A - B) / (A + B)

var addNDVI = function(median_clip) {
  var ndvi = median_clip.normalizedDifference(['B8', 'B4']).rename('Sent-NDVI');
  return median_clip.addBands(ndvi);
};

// Test the addNDVI function on a single image.
var ndvi = addNDVI(median_clip).select('Sent-NDVI');

// Display the result.
Map.setCenter(-117, 46.5, 8);
var ndviParams = {min: -1, max: 1, palette: ['blue', 'white', 'green']};
Map.addLayer(ndvi, ndviParams, 'NDVI');


//export as GEOTIFF
// Load median image and select bands.
  // In ArcMap Sentinel will be: 1=Red='B4', 2=Green='B3', 3=Blue='B2', 4=NIR='B8'
  // In ArcMap NAIP will be: 1=Red='R', 2=Green='G', 3=Blue='B', 4=NIR='N'
var sentinel = ee.Image(median_clip)
  .select(['B4', 'B3', 'B2', 'B8']);

// Export the images, specifying scale and region.
Export.image.toDrive({
  image: sentinel,
  description: 'Sentinel_counties',
  scale: 10,
  maxPixels: 1e9,
  region: geometry
  });
