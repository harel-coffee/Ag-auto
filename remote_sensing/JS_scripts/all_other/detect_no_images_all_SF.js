////////
////////     Read ShapeFiles from asset directory
////////

var double_LatLong_regions = ee.FeatureCollection(double_LatLong_SF);

var double_2018_first_region = ee.FeatureCollection(double_first_2018)
var double_2018_second_region = ee.FeatureCollection(double_second_2018)
var double_2018_first_and_second_region = ee.FeatureCollection(
                                            double_first_and_second_2018)

var double_2018_last_region = ee.FeatureCollection(double_last_2018)

var needed_bands = ['B8', 'B4']
var year = '2018'

var start_date = year.concat('-03-01')
var end_date = year.concat('-10-01')


// print (typeof(double_weird_regions.size()))
print ('double_LatLong_regions is \n of size: ', 
                         double_LatLong_regions.size())

var list_of_regions = double_LatLong_regions.toList(double_LatLong_regions.size())

print ("list_of_regions is of length: ", list_of_regions.size())
// print ((list_of_regions.get(1)))

///////////////////////////////////////////////////
///////
/////// check whether we are getting all images or not.
///////
///////////////////////////////////////////////////

//
// first field
//
var double_first_2018_IC = ee.ImageCollection('COPERNICUS/S2')
    .filterDate( start_date, end_date )
    .filterBounds(double_2018_first_region)
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 10)
    .select(needed_bands);

print ("there are ", double_2018_first_region.size(), "regions in first region SF")
print ("number of images \n for first regions is ", 
       double_first_2018_IC.size())

//
// second field
//

var double_second_2018_IC = ee.ImageCollection('COPERNICUS/S2')
    .filterDate( start_date, end_date )
    .filterBounds(double_2018_second_region)
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 10)
    .select(needed_bands);

print ("there are ", double_2018_second_region.size(), "regions in second region SF")
print ("number of images \n for second regions is ", 
       double_second_2018_IC.size())
//
// first and second fields
//
var double_first_and_second_2018_IC = ee.ImageCollection('COPERNICUS/S2')
    .filterDate( start_date, end_date )
    .filterBounds(double_2018_first_and_second_region)
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 10)
    .select(needed_bands);

print ("there are ", double_2018_first_and_second_region.size(), 
       "regions in first and last region")

print ("number of images \n for first and second regions is ", 
       double_first_and_second_2018_IC.size())

////////////////////////////
//
// all fields
//
var double_LatLong_IC = ee.ImageCollection('COPERNICUS/S2')
    .filterDate( start_date, end_date )
    .filterBounds(double_LatLong_regions)
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 10)
    .select(needed_bands);


print ("there are ", double_LatLong_regions.size(), "regions in total")
print ("number of images for all regions is ", 
        double_LatLong_IC.size())

////////////////////////////////////////////////
//
//    last region
//
var double_last_2018_IC = ee.ImageCollection('COPERNICUS/S2')
    .filterDate( start_date, end_date )
    .filterBounds(double_2018_last_region)
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 10)
    .select(needed_bands);

print ("there are ", double_2018_last_region.size(), "regions in last region")

print ("number of images \n for last regions is ", 
       double_last_2018_IC.size())

////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////