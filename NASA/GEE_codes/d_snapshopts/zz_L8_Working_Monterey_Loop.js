// zz_Working_Monterey_Loop, sht_3 - exportRGBuseful1
// Load a landsat image and select three bands.

/////////////////////////////////////////////
/////
/////      Parameters
/////
/////////////////////////////////////////////
// 'LANDSAT/LC08/C01/T1_TOA/LC08_044034_20140318'
var data_source = 'LANDSAT/LC08/C01/T1_TOA';
var bands = ['B4', 'B3', 'B2'];

var wstart_date = '2014-01-01';
var wend_date = '2015-01-01';
var outfile_name_prefix = 'L8_C01_T1_TOA_' + wstart_date + "_" + wend_date + "_";

// Define the visualization parameters.
var vizParams = {
  bands: ['B4', 'B3', 'B2'],
  min: 0,
  max: [0.2, 0.2, 0.2],
  // gamma: [0.1, 1.1, 1]
};

// Create a geometry representing an export region.
var xmin = -122.20;
var xmax = -120.45;
var ymin = 35.75;
var ymax = 37.1;

var xmin = -122.50;
var xmax = -120.00;
var ymin = 35.65;
var ymax = 37.2;

var xmean = (xmin + xmax) / 2.0;
var ymean = (ymin + ymax) / 2.0;

var big_rect = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax]);
Map.addLayer(big_rect, {color: 'red'}, 'big_rect');
Map.addLayer(Monterey, {color: 'blue'}, 'Monterey');
Map.setCenter(xmean, ymean, 8);

/////////////////////////////////////////////
/////
/////      Functions
/////
/////////////////////////////////////////////
function add_system_start_time_image(image) {
  image = image.addBands(image.metadata('system:time_start').rename("system_start_time"));
  return image;
}

function add_system_start_time_collection(colss){
 var c = colss.map(add_system_start_time_image);
 return c;
}

function fetch(data_source, bands, start_date, end_date, AOI){
   var an_IC = ee.ImageCollection(data_source)
                .select(bands)
                .filterDate(wstart_date, wend_date)
                .filterBounds(AOI)
                .map(function(image){return image.clip(AOI)})
                .sort('system:time_start', true);

  an_IC = add_system_start_time_collection(an_IC);
  return an_IC;
}

function mosaic_IC(an_image_coll, bands, start_date, end_date, AOI){
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
  an_ima = an_ima.clip(Monterey);
  return (an_ima);
}

/////////////////////////////////////////////
/////
/////      Body
/////
/////////////////////////////////////////////
var landsat_fetched = fetch(data_source, bands, wstart_date, wend_date, big_rect);
var landsat_mosaiced = mosaic_IC(landsat_fetched, bands, wstart_date, wend_date, big_rect);
var visualized_IC = landsat_mosaiced.map(visualize_and_export_image);
visualized_IC = visualized_IC.map(clip_with_shapefile);

var size = landsat_mosaiced.size();
var visualized_IC_list = visualized_IC.toList(size);


// add to the end of sht_3

var client_size = size.getInfo();
print (client_size, visualized_IC_list);

for (var n=0; n < client_size; n++) {
  var image = ee.Image(visualized_IC_list.get(n));
  var d_time = image.get('system_time_start');
  Export.image.toDrive({
    image: image,
    description: outfile_name_prefix + d_time.getInfo(),
    folder: 'snapshot_Monterey',
    scale: 30,
    region: big_rect,
    maxPixels: 1e9,
  });
}

