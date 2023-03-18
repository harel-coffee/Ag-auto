// Mapping Winter Crops in China with Multi-Source Satellite Imagery and Phenology-Based Algorithm
// define time windows

var monthsmin=['2017-10-1','2017-11-10','2018-5-20','2018-7-1']; // for low NDVI period 
var monthsmax=['2017-12-1','2018-3-20']; // for high NDVI period
// define geographical range

var geometry = ee.Geometry.Rectangle([106,28,122.8,42]);

////////////////////////////////////////////
//                                        //
//          define function               //
//                                        //
////////////////////////////////////////////

// Landsat cloud mask function
var cloudMask = function(img) {
   var cloudscore = ee.Algorithms.Landsat.simpleCloudScore(img).select('cloud'); 
   return img.updateMask(cloudscore.lt(50));
};

// Sentinel-2 cloud mask function
function maskS2clouds(image) {
   var qa = image.select('QA60');
   var cloudBitMask = ee.Number(2).pow(10).int();
   var cirrusBitMask = ee.Number(2).pow(11).int();
   var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and( qa.bitwiseAnd(cirrusBitMask).eq(0));
   return image.updateMask(mask).divide(10000);
}

// NDVI function
var addNDVI = function(image){
	return image.addBands(image.normalizedDifference(['nir','red']).rename('ndvi'));
};

var L8_BANDS = ['B2','B3','B4','B5','B6','B7']; // Landsat 8 name 
var L7_BANDS = ['B1','B2','B3','B4','B5','B7'];// Landsat 7 name 
var STD_NAMES = ['blue','green','red','nir','swir1','swir2'];
var s2b10 = ['B2', 'B3', 'B4', 'B8'];
var STD_NAMES_s2 = ['blue','green','red','nir'];

// compute maximum NDVI value (named 'ndvi_max') in high NDVI period 
// Acquire Landsat data collection
var L8max = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')
              .filterDate(monthsmax[0], monthsmax[1])
              .filterBounds(geometry)
              .map(cloudMask)
              .select(L8_BANDS, STD_NAMES); // new names

var L7max = ee.ImageCollection('LANDSAT/LE07/C01/T1_TOA')
              .filterDate(monthsmax[0], monthsmax[1])
              .filterBounds(geometry)
              .map(cloudMask)
              .select(L7_BANDS,STD_NAMES);

var L_collectionmax = ee.ImageCollection(L7max.merge(L8max)); 
var L_collectionmax = L_collectionmax.map(addNDVI).select('ndvi');

// Acquire Sentinel-2 data collection
var s2max = ee.ImageCollection('COPERNICUS/S2')
              .filterDate(monthsmax[0], monthsmax[1])
              .filterBounds(geometry)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
              .map(maskS2clouds)
              .select(s2b10,STD_NAMES_s2);

var s2_collectionmax = s2max.map(addNDVI).select('ndvi');

//combine Landsat collection and sentinel-2 collection
var NDVI_max_collection = ee.ImageCollection(L_collectionmax.merge(s2_collectionmax));

// compute maximum NDVI
var ndvi_max = NDVI_max_collection.max().clip(geometry).rename('ndvimax');

// Landsat and Sentinel-2 observation frequency for the high NDVI period 
var Lobservation_max = L_collectionmax.count().int8().clip(geometry);
var Sobservation_max = s2_collectionmax.count().int8().clip(geometry);

//download 'ndvi_max' 
Export.image.toDrive( { image:ndvi_max, 
                        description: 'ndvi_max', 
                        fileNamePrefix: 'ndvi_max', 
                        scale: 30,
                        region: geometry, 
                        maxPixels: 999999999999
                    });

// download 'Lobservation_max' 
Export.image.toDrive( { image:Lobservation_max, 
	                    description:'Lobservation_max ', 
	                    fileNamePrefix:'Lobservation_max ', 
	                    scale: 30,
                        region: geometry, 
                        maxPixels: 999999999999});

// download 'Sobservation'
Export.image.toDrive( { image:Sobservation_max, 
                        description:'Sobservation_max ', 
                        fileNamePrefix:'Sobservation_max ', 
                        scale: 30,
                        region: geometry, 
                        maxPixels: 999999999999 });


//compute minimum and median NDVI value (named 'ndvi_min', 'ndvi_minmed') in high NDVI period
//Acquire Landsat data collection
var L8min = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')
              .filterDate(monthsmin[0], monthsmin[1])
              .filterBounds(geometry)
              .map(cloudMask)
              .select(L8_BANDS,STD_NAMES);

var L7min = ee.ImageCollection('LANDSAT/LE07/C01/T1_TOA')
              .filterDate(monthsmin[0], monthsmin[1])
              .filterBounds(geometry)
              .map(cloudMask)
              .select(L7_BANDS,STD_NAMES); // new names

var Lmin_collection = ee.ImageCollection(L7min.merge(L8min));

for (var j = 2; j < monthsmin.length-1; j=j+2){
   var L8min = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')
                   .filterDate(monthsmin[j], monthsmin[j+1])
                   .filterBounds(geometry)
                   .map(cloudMask)
                   .select(L8_BANDS,STD_NAMES);
                    
   var L7min = ee.ImageCollection('LANDSAT/LE07/C01/T1_TOA')
                   .filterDate(monthsmin[j], monthsmin[j+1])
                   .filterBounds(geometry)
                   .map(cloudMask)
                   .select(L7_BANDS,STD_NAMES); // new names
   var Lmin_collection = ee.ImageCollection(Lmin_collection.merge(L8min)); 
   var Lmin_collection = ee.ImageCollection(Lmin_collection.merge(L7min));
 }

var Lmin_collection = Lmin_collection.map(addNDVI).select('ndvi');

//Acquire Sentinel-2 data collection
var s2min_collection = ee.ImageCollection('COPERNICUS/S2')
                           .filterDate(monthsmin[0], monthsmin[1])
                           .filterBounds(geometry)
                           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                           .map(maskS2clouds)
                           .select(s2b10,STD_NAMES_s2);

for (var i = 2; i < monthsmin.length-1; i=i+2){
  var s2min = ee.ImageCollection('COPERNICUS/S2')
                  .filterDate(monthsmin[i], monthsmin[i+1])
                  .filterBounds(geometry)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .map(maskS2clouds)
                  .select(s2b10,STD_NAMES_s2);
  
  var s2min_collection = ee.ImageCollection(s2min_collection.merge(s2min)); 
}

var s2min_collection = s2min_collection.map(addNDVI).select('ndvi');


// Combine Landsat and sentinel-2 data collection
var NDVI_min_collection = ee.ImageCollection(Lmin_collection.merge(s2min_collection));

// Compute minimum and median NDVI value in low NDVI period
var ndvi_min = NDVI_min_collection.min().clip(geometry).rename('ndvimin');
var ndvi_minmed = NDVI_min_collection.median().clip(geometry).rename('ndvimin_med');

// Landsat and Sentinel-2 observation frequency for the low NDVI period 
var Sobservation_min = s2min_collection.count().int8().clip(geometry);
var Lobservation_min = Lmin_collection.count().int8().clip(geometry);

Export.image.toDrive( { image:ndvi_min, 
                        description:'ndvi_min', 
                        fileNamePrefix:'ndvi_min', scale: 30,
                        region: geometry,
                        maxPixels: 999999999999
                        });

Export.image.toDrive( { image:ndvi_minmed,
                        description: 'ndvi_minmed ', 
                        fileNamePrefix: 'ndvi_minmed', 
                        scale: 30,
                        region: geometry,
                        maxPixels: 999999999999
});

Export.image.toDrive( { image:Sobservation_min, 
                        description:'Sobservation_min', 
                        fileNamePrefix:'Sobservation_min', 
                        scale: 30,
                        region: geometry,
                        maxPixels: 999999999999
});

Export.image.toDrive( { image:Lobservation_min, 
                        description:'Lobservation_min', 
                        fileNamePrefix:'Lobservation_min', 
                        scale: 30,
                        region: geometry,
                        maxPixels: 999999999999
                        });

//slope
var srtm = ee.Image('USGS/SRTMGL1_003');
var slope = ee.Terrain.slope(srtm).clip(geometry);

Export.image.toDrive( { image:slope, description:'slope', 
                        fileNamePrefix:'slope', 
                        scale: 30,
                        region: geometry, 
                        maxPixels: 999999999999});

var diff = ndvi_max.subtract(ndvi_minmed).rename('diff');
// combine bands
var composite = ndvi_max.addBands(ndvi_min)
                        .addBands(diff)
                        .addBands(ndvi_minmed)
                        .addBands(slope); 

Map.addLayer(composite, {min: 0, max: 0.7}); 
Map.centerObject(composite, 10);



