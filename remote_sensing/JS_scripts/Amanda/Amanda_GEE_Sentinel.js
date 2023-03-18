// The purpose of this script is to query Sentinel-2 Level 2A imagery,
    // mask clouds, create a composite image with median operator, 
    // clip to "ROI" = polygon(s) of interest.
// In GEE, imported shapefile as table, named "ROI"

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

  // *Map the function over time period, create median composite.
    // Time period to include (here = 08/15 to 10/01)
  var imagecollection = ee.ImageCollection('COPERNICUS/S2_SR')
      .filter(ee.Filter.bounds(ROI))
      .filterDate('2019-08-15', '2019-10-01')
      // Pre-filter to get less cloudy granules. (Here: less than 20%)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
      .map(maskS2clouds);
  var median = imagecollection.median();
  var median_clip = median.clipToCollection(ROI);
  
// Set map center with lat/long and display the result in true color. (Here zoom level = 14)
  Map.setCenter(-117.079539, 46.509950, 14);
  var visParam = {min: 0.0614, max: 0.3528, gamma: 2, bands: ['B4', 'B3', 'B2']};
  Map.addLayer(median_clip, visParam, 'clipped composite image');


