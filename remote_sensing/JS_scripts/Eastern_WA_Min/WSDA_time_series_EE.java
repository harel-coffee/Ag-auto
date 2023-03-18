
var SF = ee.FeatureCollection("users/mingliangliuearth/Eastern_noGrant_Irrigated_2017"),
    Bullrun = ee.FeatureCollection("users/mingliangliuearth/BullRun");
    
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

function addDoY_to_image(image){
  var doy = image.date().getRelative('day', 'year');
  var doyBand = ee.Image.constant(doy).uint16().rename('doy');
  doyBand = doyBand.updateMask(image.select('B8').mask());

  return image.addBands(doyBand);
}

function addDoY_to_collection(collec){
  var C = collec.map(addDoY_to_image);
  return C;
}

function add_Date_Time_image(image) {
  image = image.addBands(image.metadata('system:time_start')); 
  return image;
}

function add_Date_Time_collection(colss){
 var c = colss.map(add_Date_Time_image);
 return c;
}

function addNDVI_to_image(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
}

function add_NDVI_collection(image_IC){
  var NDVI_IC = image_IC.map(addNDVI_to_image);
  return NDVI_IC;
}

function addEVI_to_image(image) {
  var evi = image.expression(
                      '2.5 * ((NIR - RED) / (NIR + (6 * RED) - (7.5 * BLUE) + 1.0))', {
                      'NIR': image.select('B8'),
                      'RED': image.select('B4'),
                      'BLUE': image.select('B2')
                  }).rename('EVI');
  return image.addBands(evi);
}

function add_EVI_collection(image_IC){
  var EVI_IC = image_IC.map(addEVI_to_image);
  return EVI_IC;
}

function extract_sentinel_IC(a_feature,start_date,end_date){
    var geom = a_feature.geometry(); //a_feature is a feature collection
    var newDict = {'original_polygon_1': geom};
    var imageC = ee.ImageCollection('COPERNICUS/S2')
                .filterDate(start_date, end_date)
                .filterBounds(geom)
                //.filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", cloud_perc)
                .map(function(image){return image.clip(geom)})
                .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_perc))
                //.filter('CLOUDY_PIXEL_PERCENTAGE < 70')
                .sort('system:time_start', true);
    
    // toss out cloudy pixels
    imageC = imageC.map(maskS2clouds);
    
    // pick up some bands
    // imageC = imageC.select(['B8', 'B4', 'B3', 'B2']);
    
    // add DoY as a band
    imageC = addDoY_to_collection(imageC);
    
    imageC = add_Date_Time_collection(imageC);
    
    // add NDVI as a band
    imageC = add_NDVI_collection(imageC);
    
    // add EVI as a band
    imageC = add_EVI_collection(imageC);

    // add original geometry to each image
    // we do not need to do this really:
    imageC = imageC.map(function(im){return(im.set(newDict))});
    
    // add original geometry and WSDA data as a feature to the collection
    imageC = imageC.set({ 'original_polygon': geom,
                          'WSDA':a_feature
                        });
    // imageC = imageC.sort('system:time_start', true);

    
    //imageC = imageC.map(add_NDVI_collection)
  return imageC;
}

function mosaic_and_reduce_IC_mean(an_IC,a_feature,start_date,end_date){
  an_IC = ee.ImageCollection(an_IC);
  //print('mosaic_start_date:',start_date);
  //var reduction_geometry = ee.Feature(ee.Geometry(an_IC.get('original_polygon')));
  var reduction_geometry = a_feature;
  var WSDA = an_IC.get('WSDA');
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
                                                        scale: 10//,
                                                        //tileScale: 16
                                                      });
                                          }
                        ).flatten();
                          
  reduced = reduced.set({ 'original_polygon': reduction_geometry,
                          'WSDA':WSDA
                      });
  WSDA = ee.Feature(WSDA);
  WSDA = WSDA.toDictionary();
  
  // var newDict = {'WSDA':WSDA};
  reduced = reduced.map(function(im){return(im.set(WSDA))}); 
  return(reduced);
}
// remove geometry on each feature before printing or exporting
var myproperties=function(feature){
  feature=ee.Feature(feature).setGeometry(null);
  return feature;
};


var xmin=-125.0;
var ymin = 45.0;
var xmax=-116.0;
var ymax = 49.0;
//var xmin=-120.0;
//var ymin = 47.0;
//var xmax=-119.9;
//var ymax = 47.1;
var xmed1 = (xmin + xmax) / 2.0;
var xmed2 = (xmin + xmax) / 2.0;
//var WA1 = ee.Geometry.Polygon([[[xmin, ymin], [xmin, ymax], [xmed1, ymax], [xmed1, ymin], [xmin, ymin]],
//                              [[xmed2, ymin], [xmed2, ymax], [xmax, ymax], [xmax, ymin], [xmed2, ymin]]
//                             ]);
var WA1 = ee.Geometry.Polygon([[xmin, ymin], [xmin, ymax], [xmed1, ymax], [xmed1, ymin], [xmin, ymin]]);
var WA2 = ee.Geometry.Polygon([[xmed2, ymin], [xmed2, ymax], [xmax, ymax], [xmax, ymin], [xmed2, ymin]]);
var WA = [WA1,WA2];
//WA = ee.FeatureCollection(WA);

//var SF_regions = ee.FeatureCollection(SF);
//var SF_regions = ee.FeatureCollection(Bullrun);
var SF_regions = ee.FeatureCollection(WA);
var reduction_geometry = ee.FeatureCollection(SF);
//var reduction_geometry = ee.FeatureCollection(WA);

print ("Number of fields in the shapefile is", SF_regions.size());


var wstart_date = '2017-01-01';
//var end_date = '2017-01-31';
var wend_date = '2017-12-31';
var cloud_perc = 70;

var end_days = ee.List(["31","28","31","30","31","30","31","31","30","31","30","31"]); 
var smonth = ee.List(["01","02","03","04","05","06","07","08","09","10","11","12"]); 
//var imonth = ee.List.sequence(0, 11);

//var loc_end_days = ["31","28","31","30","31","30","31","31","30","31","30","31"]; 
//var loc_smonth = ["01","02","03","04","05","06","07","08","09","10","11","12"]; 
var imonth = ee.List.sequence(0, 11);

//print('smonth[0]:',smonth[0]);

//var t = 'month ' + smonth[0];
//print('t',t.toString());
//List<String> list = Arrays.asList(new String[]{"foo", "bar"});

var compute = function(month) {
  //var end_days = ["31","28","31","30","31","30","31","31","30","31","30","31"]; 
  //var smonth = ["01","02","03","04","05","06","07","08","09","10","11","12"]; 
  //var smonth = ee.List(['01','02','03','04','05','06','07','08','09','10','11','12']);
  // We define the operation using the EE API.
  //print("smonth:",smonth.get(month));
  var ed = ee.String(end_days.get(month));
  var m = ee.String(smonth.get(month));
  //print('t',t);
  var start_date = ee.String('2017-').cat(m).cat('-01');
  //var start_date = ee.String('my_file_').cat(t);
  var end_date = ee.String('2017-').cat(m).cat('-').cat(ed);
  //print('start_date',start_date.toString());
  //print('end_date',end_date);
  var imageC = extract_sentinel_IC(SF_regions, start_date, end_date);
  //print('imageC',imageC);
  var reduced = mosaic_and_reduce_IC_mean(imageC,reduction_geometry,start_date,end_date);  
  //var featureCollection = reduced.map(myproperties);
  //var outfile_name = ee.String('newtest_monthly_m').cat(m);
  // var temp = outfile_name.toString();
  //print('typeof(temp):', typeof(temp));
  //print('velue(outfile_name):', outfile_name);
  //var outfile_name = 'test_monthly_' + t; //smonth[month];

  //Export.table.toDrive({
  //  collection: featureCollection,
  //  description: 'test',
  //  folder:"Irrigated_NotCorrectYrs" + cloud_perc + "_cloud",
  //  fileNamePrefix: outfile_name,
  //  fileFormat: 'CSV'
  //});
  return(reduced);
};

//var out = compute(0);
var out = imonth.map(compute);
print('out:',out);

//var serverString = ee.String('This is on the server.');
//var test = serverString.toString();
//var locserverString = 'This is on the cliend.';
//print('typeof(serverString):', typeof(serverString));
//print('typeof(locserverString):', typeof(locserverString));
//print('typeof(test):', typeof(test));

//print('serverString:', serverString);
//print('locserverString:', locserverString);




//var Grant_2017_allFs_notCorrectYears = SF_regions.map(extract_sentinel_IC);
//print("Grant_2017_allFs_notCorrectYears:",Grant_2017_allFs_notCorrectYears);
//var all_fields_TS = Grant_2017_allFs_notCorrectYears.map(mosaic_and_reduce_IC_mean);
//all_fields_TS = all_fields_TS.flatten();
//print("all_fields_TS:",all_fields_TS);



//print ("Number of fields in the all_fields_TS is", all_fields_TS.size());

//maping 
//var empty = ee.Image().byte();
//var outlinectOriginal = empty.paint({
//  featureCollection: SF_regions,
//  color: 1,
//  width: 2
//});

//var outlinectOutput = empty.paint({
//  featureCollection: all_fields_TS,
//  color: 3,
//  width: 3
//});

//Map.addLayer(outlinectOriginal, {palette: '0000ff'}, 'SF_regions');
//Map.addLayer(outlinectOutput, {palette: '00ff00'}, 'SF_Output');
//Map.setCenter(bounds.getCenterLonLat(),12); 
//var styling = {color: 'red', fillColor: '00000000'};
//Map.addLayer(SF_regions.style(styling));
//Map.addLayer(Grant_2017_allFs_notCorrectYears.style(styling))
//Map.setCenter(-120.18914719665702,47.39204393356499,16);
    //print("SF_regions:",SF_regions);
    //var geom = SF_regions.first().geometry();
    
    
    
    //var geom = SF_regions.geometry();
    //var newDict = {'original_polygon_1': geom};
    //var imageC = ee.ImageCollection('COPERNICUS/S2')
    //            .filterDate(start_date, end_date)
    //            .filterBounds(geom)
    //            .map(function(image){return image.clip(geom)})
    //            .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_perc))
    //            .sort('system:time_start', true);
    ////print("imageC:",imageC);        
    //
    //var withNDVI = imageC.map(function(img){
    //var red = ee.Image(img.select('B4'));
    //var nir = ee.Image(img.select('B8'));
    //var ndvi = (nir.subtract(red)).divide(nir.add(red)).rename('ndvi');
    //return img.addBands(ndvi);
    //});
    //
    //
    //imageC = imageC.map(maskS2clouds);
    ////print("Clouds_imageC:",imageC);    
    //
    //// add DoY as a band
    //imageC = addDoY_to_collection(imageC);
    ////print("DoY_imageC:",imageC);  
    //imageC = add_Date_Time_collection(imageC);
    ////print("_Date_Time_imageC:",imageC);  
    //// add NDVI as a band
    //imageC = add_NDVI_collection(imageC);
    ////print("NDVI_imageC:",imageC);  
    //// add EVI as a band
    //imageC = add_EVI_collection(imageC);
    ////print("EVI_imageC:",imageC);  
    //// add original geometry to each image
    //// we do not need to do this really:
    //imageC = imageC.map(function(im){return(im.set(newDict))});
    ////print("newDict_imageC:",imageC);  
    //
    //// add original geometry and WSDA data as a feature to the collection
    //imageC = imageC.set({ 'original_polygon': geom,
    //                      'WSDA':SF_regions
    //                    });
    //                    
   // print("set_imageC:",imageC);  
   
   
    //var imageC = extract_sentinel_IC(SF_regions,start_date,end_date);
    
    
  //  var an_IC = ee.ImageCollection(imageC);
  //
  ////var reduction_geometry = ee.Feature(ee.Geometry(an_IC.get('original_polygon')));
  ////var reduction_geometry = SF_regions;
  //var reduction_geometry = ee.FeatureCollection(SF);
  //
  //
  ////print("reduction_geometry:",reduction_geometry);
  //var WSDA = an_IC.get('WSDA');
  //var start_date_DateType = ee.Date(start_date);
  //var end_date_DateType = ee.Date(end_date);
  ////######**************************************
  //// Difference in days between start and end_date
//
  //var diff = end_date_DateType.difference(start_date_DateType, 'day');
  //print("diff:",diff);
//
  //// Make a list of all dates
  //var range = ee.List.sequence(0, diff.subtract(1)).map(function(day){
  //                                  return start_date_DateType.advance(day,'day')});
  //print("range:",range);
  //// Funtion for iteraton over the range of dates
  //function day_mosaics(date, newlist) {
  //  // Cast
  //  date = ee.Date(date);
  //  newlist = ee.List(newlist);
//
  //  // Filter an_IC between date and the next day
  //  var filtered = an_IC.filterDate(date, date.advance(1, 'day'));
//
  //  // Make the mosaic
  //  var image = ee.Image(filtered.mosaic());
//
  //  // Add the mosaic to a list only if the an_IC has images
  //  return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist));
  //}
//
  //// Iterate over the range to make a new list, and then cast the list to an imagecollection
  //var newcol = ee.ImageCollection(ee.List(range.iterate(day_mosaics, ee.List([]))));
  ////print("newcol:",newcol);
  ////print("newcol 1", newcol);
  ////######**************************************
//
  //var reduced = newcol.map(function(image){
  //                          return image.reduceRegions({
  //                                                      collection:reduction_geometry,
  //                                                      reducer:ee.Reducer.mean(), 
  //                                                      scale: 10
  //                                                    });
  //                                        }
  //                      ).flatten();
  //                        
  //reduced = reduced.set({ 'original_polygon': reduction_geometry,
  //                        'WSDA':WSDA
  //                    });
  //WSDA = ee.Feature(WSDA);
  //WSDA = WSDA.toDictionary();
  //
  //// var newDict = {'WSDA':WSDA};
  //reduced = reduced.map(function(im){return(im.set(WSDA))}); 
  ////print("reduced:",reduced);
    
  //var reduced = mosaic_and_reduce_IC_mean(imageC,reduction_geometry);  
    

    
    //Map.addLayer(SF_regions, {}, 'SF_regions');

    // use quality mosaic to get the per pixel maximum NDVI values and corresponding bands
  // var ndviQual = withNDVI.qualityMosaic('ndvi');
    //print("ndviQual:",ndviQual)
   // Map.addLayer(ndviQual, {min:0, max: 5000, bands: ['B4', 'B3', 'B2']})
    

                

//print("collected images:",Grant_2017_allFs_notCorrectYears);
//Map.addLayer(SF_regions, {}, 'SF_regions');
//Map.addLayer(Grant_2017_allFs_notCorrectYears, {}, 'Images');
//Map.addLayer(SF_regions, {}, 'SF_regions');
//Map.addLayer(all_fields_TS, {}, 'all_fields_TS');
//Map.addLayer(reduced, {}, 'reduced');
//Map.addLayer(all_fields_TS.style(styling));

//print ("Export maps...\n");

// Make a feature without geometry and set the properties to the dictionary of means.
//var feature = ee.Feature(null, reduced);

var featureCollection = ee.FeatureCollection(out.get(0));
for (var mon = 1; mon < 12; mon++){
  featureCollection.merge(out.get(mon));
}


//var featureCollection = featureCollection.map(myproperties);
var outfile_name = 'test_monthly_fc_2';// + smonth[month];

Export.table.toDrive({
  collection: featureCollection,
  //description:'Eastern_NoGrant_2017_irrigated_notCorrectYrs_' + cloud_perc + 'cloud',
  description:outfile_name,
  folder:"Irrigated_NotCorrectYrs" + cloud_perc + "_cloud",
  fileNamePrefix: outfile_name,
  fileFormat: 'CSV'
});

//}

