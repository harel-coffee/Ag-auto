var SF = ee.FeatureCollection(four_polygons_wCentroids);

var x_pairs = ee.List([[-119.824520, 47.1593417], 
                       [-119.830150, 47.1160363],
                       [-119.925224, 46.9893742],
                       [-119.796600, 46.9978835]
                       ]);
                               
var small_tile_size = 125;

var buffer_1 = ee.Feature(ee.Geometry.Point(x_pairs.get(0)).buffer(small_tile_size).bounds());
var buffer_2 = ee.Feature(ee.Geometry.Point(x_pairs.get(1)).buffer(small_tile_size).bounds());
var buffer_3 = ee.Feature(ee.Geometry.Point(x_pairs.get(2)).buffer(small_tile_size).bounds());
var buffer_4 = ee.Feature(ee.Geometry.Point(x_pairs.get(3)).buffer(small_tile_size).bounds());


var keepProperties = ['ID', 'county', 'CropTyp', 'Irrigtn', 'LstSrvD'];
buffer_1 = buffer_1.copyProperties(SF.toList(4).get(0), keepProperties);
buffer_2 = buffer_2.copyProperties(SF.toList(4).get(1), keepProperties);
buffer_3 = buffer_3.copyProperties(SF.toList(4).get(2), keepProperties);
buffer_4 = buffer_4.copyProperties(SF.toList(4).get(3), keepProperties);

// buffer_1 = buffer_1.set('ID', "104563_WSDA_SF_2017");
// buffer_2 = buffer_2.set('ID', "105429_WSDA_SF_2017");
// buffer_3 = buffer_3.set('ID', "106054_WSDA_SF_2017");
// buffer_4 = buffer_4.set('ID', "102309_WSDA_SF_2017");

var buffers = ee.FeatureCollection([buffer_1, buffer_2, buffer_3, buffer_4]);

// print(ee.Number(x_small_list.get(1)));
print ("buffers", buffers);
print ("SF", SF);

Map.addLayer(buffers, {color: 'red'});
Map.addLayer(SF, {color: 'blue'}, 'CSV centroids');

//////////////////////////////////////////////////////
///////
///////     Functions
///////
//////////////////////////////////////////////////////
////////////////////////////////////////////////
///
///     add Year
///
function addYear_to_image(image){
  var year = image.date().get('year');
  var yearBand = ee.Image.constant(year).uint16().rename('image_year');
  return image.addBands(yearBand);
}

function addYear_to_collection(collec){
  var C = collec.map(addYear_to_image);
  return C;
}

////////////////////////////////////////////////
///
/// add Day of Year (Perhaps can be omitted and obtained in python from `system_start_time`)
///

function addDoY_to_image(image){
  // var doy = image.date().getRelative('day', 'year');
  var doy = ee.Date(image.get('system:time_start')).getRelative('day', 'year');
  var doyBand = ee.Image.constant(doy).uint16().rename('doy');
  doyBand = doyBand.updateMask(image.select('NDVI').mask());

  return image.addBands(doyBand);
}

function addDoY_to_collection(collec){
  var C = collec.map(addDoY_to_image);
  return C;
}

////////////////////////////////////////////////
///
/// add Date of image to an imageCollection
///

function add_system_start_time_image(image) {
  image = image.addBands(image.metadata('system:time_start').rename("system_start_time"));
  return image;
}

function add_system_start_time_collection(colss){
 var c = colss.map(add_system_start_time_image);
 return c;
}


////////////////////////////////////////////////
///
///         Do the Job function
///

function extract_satellite_IC(a_feature, start_date, end_date){
    var geom = a_feature.geometry(); // a_feature is a feature collection
    var newDict = {'original_polygon_1': geom};
    var imageC = ee.ImageCollection('LANDSAT/LE07/C01/T1_8DAY_NDVI')
                .filterDate(start_date, end_date)
                .select('NDVI')
                .filterBounds(geom)
                .map(function(image){return image.clip(geom)})
                .sort('system:time_start', true);

    // imageC = imageC.map(clean_clouds_from_one_image_landsat);
    imageC = addYear_to_collection(imageC); // add year as a band
    imageC = addDoY_to_collection(imageC);  // add DoY as a band
    imageC = add_system_start_time_collection(imageC);
    
    // add original geometry to each image. We do not need to do this really:
    imageC = imageC.map(function(im){return(im.set(newDict))});
    
    // add original geometry and WSDA data as a feature to the collection
    imageC = imageC.set({ 'original_polygon': geom});
  return imageC;
}

function mosaic_and_reduce_IC_mean(an_IC,a_feature,start_date,end_date){
  an_IC = ee.ImageCollection(an_IC);
  var reduction_geometry = a_feature;
  var start_date_DateType = ee.Date(start_date);
  var end_date_DateType = ee.Date(end_date);
  //######**************************************
  // Difference in days between start and end_date

  var diff = end_date_DateType.difference(start_date_DateType, 'day');

  // Make a list of all dates
  var range = ee.List.sequence(0, diff.subtract(1)).map(function(day){
                                    return start_date_DateType.advance(day,'day')});

  // Funtion for iteraton over the range of dates
  function day_mosaics(date, newlist) {
    // Cast
    date = ee.Date(date);
    newlist = ee.List(newlist);

    // Filter an_IC between date and the next day
    var filtered = an_IC.filterDate(date, date.advance(1, 'day'));

    // Make the mosaic
    var image = ee.Image(filtered.mosaic());

    // Add the mosaic to a list only if the an_IC has images
    return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist));
  }

  // Iterate over the range to make a new list, and then cast the list to an imagecollection
  var newcol = ee.ImageCollection(ee.List(range.iterate(day_mosaics, ee.List([]))));
  //print("newcol 1", newcol);
  //######**************************************

  var reduced = newcol.map(function(image){
                            return image.reduceRegions({
                                                        collection:reduction_geometry,
                                                        reducer:ee.Reducer.mean(), 
                                                        scale: 10
                                                      });
                                          }
                        ).flatten();
                          
  reduced = reduced.set({ 'original_polygon': reduction_geometry});
  
  return(reduced);
}

// remove geometry on each feature before printing or exporting

var myproperties=function(feature){
  feature=ee.Feature(feature).setGeometry(null);
  return feature;
};

//////////////////////////////////////////////////////
///////
///////     Body
///////
//////////////////////////////////////////////////////

//////////////////////////////
///////
///////     Grant
///////
//////////////////////////////
var grant_xmin = -120.080116;
var grant_ymin =  46.601701;
var grant_xmax = -118.975653;
var grant_ymax = 47.963471;

var Grant = ee.Geometry.Polygon([[grant_xmin, grant_ymin], [grant_xmin, grant_ymax], 
                                 [grant_xmax, grant_ymax], [grant_xmax, grant_ymin], 
                                 [grant_xmin, grant_ymin]]);


var SF_regions = ee.FeatureCollection(Grant);
var reduction_geometry = ee.FeatureCollection(buffers);
Map.addLayer(Grant, {color: 'gray'}, 'Grant');

// print ("Number of fields in the shapefile is", reduction_geometry.size());

var wstart_date = '2008-01-01';
var wend_date = '2021-07-01';
var cloud_perc = 70;

var imageC = extract_satellite_IC(SF_regions, wstart_date, wend_date);
var reduced = mosaic_and_reduce_IC_mean(imageC, reduction_geometry, wstart_date, wend_date);  
var featureCollection = reduced;


var outfile_name = 'Grant_4Fields_Landsat7_8day_NDVIComosite_' + wstart_date + "_" + wend_date;
Export.table.toDrive({
  collection: featureCollection,
  description:outfile_name,
  folder:"first_investigation_Aug_2021",
  fileNamePrefix: outfile_name,
  fileFormat: 'CSV',
  selectors:["ID", "county", "CropTyp",  "Irrigtn",  "LstSrvD",
             "doy", 'NDVI',
             "system_start_time", "image_year"]
});


