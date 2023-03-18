// L7_C2L2Scaled.js Jan 03, 2022

// How to share code: https://developers.google.com/earth-engine/guides/playground
// This link should provide/share the latest version of the code:
// https://code.earthengine.google.com/?scriptPath=users%2Fhnoorazar%2FNASA_trends%3A07_snapshot_RGB_Working%2FL7_C2L2Scaled
// whole repo: https://code.earthengine.google.com/?accept_repo=users/hnoorazar/NASA_trends
// Assets: https://code.earthengine.google.com/?asset=users/hnoorazar/NASA_trend

// zz_L8_C2L2Scaled_Working_Monterey_Loop, sht_3 - exportRGBuseful1
// Load a landsat image and select three bands.

/*
   Problem: https://www.google.com/intl/en_ca/earth/outreach/learn/introduction-to-google-earth-engine/
*/ 
/////////////////////////////////////////////
/////
/////      Parameters
/////
/////////////////////////////////////////////
// 'LANDSAT/LC08/C01/T1_TOA/LC08_044034_20140318'

// var data_source = 'LANDSAT/LC08/C02/T1_L2';
// var RGBbands = ['SR_B4', 'SR_B3', 'SR_B2'];
// var out_pref = "_L8_L2C2_";

var data_source = "LANDSAT/LE07/C02/T1_L2";
var RGBbands = ['SR_B3', 'SR_B2', 'SR_B1'];
var out_pref = "_L7_L2C2_";

// Define the visualization parameters.
var vizParams = {
  bands: RGBbands,
  min: 0,
  max: 0.2, // https://developers.google.com/earth-engine/tutorials/tutorial_api_04 
  // gamma: [0.1, 1.1, 1]
};

//////////////////////////////
///////
///////     Monterey
///////
//////////////////////////////
// Create a geometry representing an export region.

var xmin = -122.10;
var xmax = -120.40;
var ymin = 35.68;
var ymax = 37.00;
var xmean = (xmin + xmax) / 2.0;
var ymean = (ymin + ymax) / 2.0;
var big_rect = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax]);
Map.addLayer(big_rect, {color: 'red'}, 'big_rect');
Map.setCenter(xmean, ymean, 8);

var SF = ee.FeatureCollection(Monterey);
Map.addLayer(SF, {color: 'blue'}, 'Monterey');

var wstart_date = '2014-01-01';
var wend_date = '2015-01-01';
var outfile_name_prefix = out_pref  + wstart_date + "_" + wend_date;
var output_folder = 'snapshot_Monterey';

//////////////////////////////
///////
///////     Adam Benton
///////
//////////////////////////////

// var xmin = -120.2;
// var xmax= -117.6;
// var ymin = 45.75;
// var ymax = 47.5;
// var xmean = (xmin + xmax) / 2.0;
// var ymean = (ymin + ymax) / 2.0;
// var big_rect = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax]);
// Map.addLayer(big_rect, {color: 'red'}, 'big_rect');
// Map.setCenter(xmean, ymean, 7);

// var SF = ee.FeatureCollection(AdamBenton2016);
// Map.addLayer(SF, {color: 'blue'}, 'AdamBenton2016');

// var wstart_date = '2016-01-01';
// var wend_date = '2017-01-01';
// var outfile_name_prefix = out_pref + wstart_date + "_" + wend_date;
// var output_folder = 'snapshot_AdamBenton2016';

//////////////////////////////
///////
///////    Grant
///////
//////////////////////////////

// var xmin = -120.1;
// var xmax = -118.85;
// var ymin = 46.6;
// var ymax = 48.0;
// var xmean = (xmin + xmax) / 2.0;
// var ymean = (ymin + ymax) / 2.0;
// var big_rect = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax]);
// Map.addLayer(big_rect, {color: 'red'}, 'big_rect');
// Map.setCenter(xmean, ymean, 8);

// var SF = ee.FeatureCollection(Grant2017);
// Map.addLayer(SF, {color: 'blue'}, 'Grant2017');

// var wstart_date = '2017-01-01';
// var wend_date = '2018-01-01';
// var outfile_name_prefix = out_pref + wstart_date + "_" + wend_date;
// var output_folder = 'snapshot_Grant2017';

//////////////////////////////
///////
///////    Walla Walla
///////
//////////////////////////////

// var xmin = -119.09;
// var xmax = -117.9;
// var ymin = 45.9;
// var ymax = 46.66;
// var xmean = (xmin + xmax) / 2.0;
// var ymean = (ymin + ymax) / 2.0;
// var big_rect = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax]);
// Map.addLayer(big_rect, {color: 'red'}, 'big_rect');
// Map.setCenter(xmean, ymean, 8.5);

// var SF = ee.FeatureCollection(Walla2015);
// Map.addLayer(SF, {color: 'blue'}, 'Walla2015');

// var wstart_date = '2015-01-01';
// var wend_date = '2016-01-01';
// var outfile_name_prefix = out_pref + wstart_date + "_" + wend_date;
// var output_folder = 'snapshot_Walla2015';

//////////////////////////////
///////
///////    Franklin Yakima
///////
//////////////////////////////

// var xmin = -121.2;
// var xmax = -118.1;
// var ymin = 45.95;
// var ymax = 47.0;
// var xmean = (xmin + xmax) / 2.0;
// var ymean = (ymin + ymax) / 2.0;
// var big_rect = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax]);
// Map.addLayer(big_rect, {color: 'red'}, 'big_rect');
// Map.setCenter(xmean, ymean, 7);

// var SF = ee.FeatureCollection(FranklinYakima2018);
// Map.addLayer(SF, {color: 'blue'}, 'FranklinYakima2018');

// var wstart_date = '2018-01-01';
// var wend_date = '2019-01-01';
// var outfile_name_prefix = out_pref + wstart_date + "_" + wend_date;
// var output_folder = 'snapshot_FranklinYakima2018';

/////////////////////////////////////////////
/////
/////      Functions
/////
/////////////////////////////////////////////
function scale_the_d_bands(image){
  var red = image.select('SR_B3').multiply(0.0000275).add(-0.2);
  var green = image.select('SR_B2').multiply(0.0000275).add(-0.2);
  var blue = image.select('SR_B1').multiply(0.0000275).add(-0.2);

  return image.addBands(green, null, true)
              .addBands(red, null, true)
              .addBands(blue, null, true);
}


function add_system_start_time_image(image) {
  image = image.addBands(image.metadata('system:time_start').rename("system_start_time"));
  return image;
}

function add_system_start_time_collection(colss){
 var c = colss.map(add_system_start_time_image);
 return c;
}

function fetch(data_source, bandss, start_date, end_date, AOI){
   var an_IC = ee.ImageCollection(data_source)
                .select(bandss)
                .filterDate(wstart_date, wend_date)
                .filterBounds(AOI)
                .map(function(image){return image.clip(AOI)})
                .sort('system:time_start', true);
                
  // scale the bands
  an_IC = an_IC.map(scale_the_d_bands);

  an_IC = add_system_start_time_collection(an_IC);
  return an_IC;
}

function mosaic_IC(an_image_coll, start_date, end_date, AOI){
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

    // Filter an_image_coll between date and the next day
    var filtered = an_image_coll.filterDate(date, date.advance(1, 'day'));

    var image = ee.Image(filtered.mosaic()); // Make the mosaic
    
    image = image.copyProperties({source: filtered.first(),
                                  properties: ['system:time_start']
                                  });
    image = image.set({system_time_start: filtered.first().get('system:time_start')});

    // Add the mosaic to a list only if the an_image_coll has images
    return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist));
  }

  // Iterate over the range to make a new list, and then cast the list to an imagecollection
  var newcol = ee.ImageCollection(ee.List(range.iterate(day_mosaics, ee.List([]))));
  newcol = newcol.sort('system_time_start', true);
  return(newcol);
}

function visualize_and_export_image(image2vis){
  var imageRGB_mosaiced1 = image2vis.visualize(vizParams);
  
  imageRGB_mosaiced1 = imageRGB_mosaiced1.copyProperties({source: image2vis,
                                                          properties: ['system:time_start']
                                                          });
  imageRGB_mosaiced1 = imageRGB_mosaiced1.set({system_time_start: image2vis.get('system_time_start')});
  // imageRGB_mosaiced1 = imageRGB_mosaiced1.addBands(image2vis.get('system_time_start'));
  return imageRGB_mosaiced1;
}

function clip_with_shapefile(an_ima){
  an_ima = an_ima.clip(SF);
  return (an_ima);
}

/////////////////////////////////////////////
/////
/////      Body
/////
/////////////////////////////////////////////
var landsat_fetched = fetch(data_source, RGBbands, wstart_date, wend_date, big_rect);
var landsat_mosaiced = mosaic_IC(landsat_fetched, wstart_date, wend_date, big_rect);
var visualized_IC = landsat_mosaiced.map(visualize_and_export_image);
visualized_IC = visualized_IC.map(clip_with_shapefile);

var size = landsat_mosaiced.size();
var visualized_IC_list = visualized_IC.toList(size);
Map.addLayer(visualized_IC.first(), {color: ''}, 'RGB');

// var client_size = size.getInfo();
// print (client_size);
// for (var n=0; n < client_size; n++) {
//   var image = ee.Image(visualized_IC_list.get(n));
//   var d_time = image.get('system_time_start');
//   Export.image.toDrive({
//     image: image,
//     description: d_time.getInfo() + outfile_name_prefix,
//     folder: output_folder,
//     scale: 10,
//     region: big_rect,
//     maxPixels: 1e9,
//   });
// }

