var boulder = ee.FeatureCollection(boulder_poly);

///////////////////////////////////////////////////////////////////////////////
var meanDictionary_IC = function(IC){
  var x = IC.map(meanDictionary_img);
  return(x);
}

var meanDictionary_img = function(image){
  
  var x = image.reduceRegion({
           reducer: ee.Reducer.min(),
           geometry: image.geometry(),
           scale: 30,
           maxPixels: 1e9
          });
  return(x);
}
  

var addDate_to_collection = function(collec){
  var C = collec.map(addDate_to_image);
  return C;
};

var addDate_to_image = function(image){
  var doy = image.date().getRelative('day', 'year');
  var doyBand = ee.Image.constant(doy).uint16().rename('doy');
  doyBand = doyBand.updateMask(image.select('B8').mask());
  return image.addBands(doyBand);
};

// col = col.map(function(img) {
//   var doy = ee.Date(img.get('system:time_start')).getRelative('day', 'year');
//   return img.set('doy', doy);
// });

///////////////////////////////////////////////////////////////////////////////

var boulder_image_collection = ee.ImageCollection('COPERNICUS/S2')
                               .filterDate('2019-01-01', '2019-10-20')
                               .filterBounds(boulder_poly)
                               .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 
                                                "less_than", 10);
print(boulder_image_collection.size());
print(boulder_image_collection)





var boulder_image = boulder_image_collection.median();

// print(boulder_image.get('bands')) // returns nothing

boulder_image_collection = addDate_to_collection(boulder_image_collection);

print(boulder_image_collection.first());

var image = boulder_image_collection.first();
var meanDictionary = image.reduceRegion({
  reducer: ee.Reducer.min(),
  geometry: image.geometry(),
  scale: 30,
  maxPixels: 1e9
});
print(meanDictionary);
print("day of year is ", meanDictionary.get('doy'));

var date = ee.Date(image.get('system:time_start'));
print('Timestamp: ', date); // ee.Date

// print(boulder_image_collection.select('doy').getInfo())


// print((boulder_image_collection.get('system:time_start'))) // returns nothing


// Get the list of values from the mosaic image
// var freqHist = boulder_image_collection.select('doy')
//               .reduceRegion({reducer: ee.Reducer.frequencyHistogram(), 
//                               geometry: boulder, 
//                               scale: 20});

// // rewrite the distionary
// var values = ee.List(freqHist.map(function(feat){
//   feat = ee.Feature(feat);
//   var vals = ee.Dictionary(feat.get('histogram')).keys();
//   return ee.Feature(null, {vals: vals});
// }).aggregate_array('vals')).flatten().distinct();
// print(values);
