////////
////////     Read ShapeFiles from asset directory
////////

var double_LatLong_regions = ee.FeatureCollection(double_LatLong_SF);
var double_weird_regions = ee.FeatureCollection(double_weird_SF);

print(double_LatLong_regions)
print(double_weird_regions)


// print (typeof(double_weird_regions.size()))
print ('double_LatLong_regions is \n of size: ', 
                         double_LatLong_regions.size())

print ('double_weird_regions is \n of size: ', 
                              double_weird_regions.size())

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

var needed_bands = ['B8', 'B4']

var years = ['2019', '2018', '2017', '2016', 
             '2015', '2014', '2013', '2012']

var year = years[1]

var start_date = year.concat('-03-01')
var end_date = year.concat('-10-01')

print ('the chosen time window is\n from '.concat(year, 
        '-03-01\n to   ', year, '-10-01'))


// var double_LatLong_Collection = ee.ImageCollection('COPERNICUS/S2')
//     .filterDate( start_date, end_date )
//     .filterBounds(fifty_double_field)
//     .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 0.5)
//     .select(needed_bands);

