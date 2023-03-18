////////////////////////////////////////////////////////////////////////
/////
/////  Test wether reduceregion() can be used without being have to use clipping.
/////    
/////    
/////    
/////    
/////

// define the region by reading the shapefile from asset.
var first_grant_region = ee.FeatureCollection(first_grant_SF);
var first_grant_region_geometry = first_grant_region.geometry()

////////////////// shrink the region by .buffer()
var shrunk_region = first_grant_region_geometry.buffer(-50)

// form a small circle inside the region.

var circle = ee.Geometry.Point([-119.1226, 47.46670]).buffer(30);
// print (circle)


print("area within the main region is ", 
      first_grant_region_geometry.area(), 
      " square meter and area of shrunk image is ", 
      shrunk_region.area(), " and area of circle is ",
      circle.area())

// print(first_grant_region_geometry)

// pull the images from EE
var first_field_image = ee.ImageCollection('COPERNICUS/S2')
                        .filterDate('2019-01-01', '2019-01-31')
                        .filterBounds(first_grant_region)
                        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 20)
                        .select(['B8'])

print("Number of images in the collection is ", first_field_image.size())

var image = first_field_image.first()
print(image)


// print(first_grant_region_geometry)
print ("polygon centroid is ", first_grant_region_geometry.centroid())
Map.setCenter(-119.12261, 47.46670, 15);
Map.addLayer(first_grant_region_geometry, {color: '0000FF'}, 'blue')
Map.addLayer(circle, {color: 'FF0000'}, 'red')

// Reduce the image within the given region, using a reducer that
// computes the max pixel value.  We also specify the spatial
// resolution at which to perform the computation, in this case 200
// meters.

var max_no_geom_geometry = image.reduceRegion({
                             reducer: ee.Reducer.max(),
                             scale: 200
                         });

print("The value of spatial reduction without setting a region is ", max_no_geom_geometry);

//////
//////
//////
var max_field_geometry = image.reduceRegion({
                             reducer: ee.Reducer.max(),
                             geometry: first_grant_region_geometry,
                             scale: 200
                         });

print("The value of spatial reduction by setting a region is ", max_field_geometry);


//////
//////
//////
var max_shrunk_geometry = image.reduceRegion({
                             reducer: ee.Reducer.max(),
                             geometry: shrunk_region,
                             scale: 200
                         });

print("The value of spatial reduction without setting a region is ", max_shrunk_geometry);

//////
//////
//////

var max_circle = image.reduceRegion({
                             reducer: ee.Reducer.max(),
                             geometry: circle,
                             scale: 10
                         });

// Print the result (a Dictionary) to the console.
print("The value of spatial reduction by setting a region is ", max_circle);
