
 var vizParams = {
   bands: ['B4', 'B3', 'B2'],
   min: 0,
   max: 0.5,
   gamma: [0.95, 1.1, 1]
 };
 Map.addLayer(Grant_2018_IC_first.get(1), vizParams, 'false color composite');



 print(first_region.geometry().centroid())
 Look at here: 
 // https:www.google.com/maps/place/47%C2%B031'33.2%22N+119%C2%B031
 // '30.0%22W/@47.5259,-119.525,813m/data=!3m1!1e3!4m5!3m4!1s0x0:0x0!
 // 8m2!3d47.5259!4d-119.525

#############################################################

                        Export data

#############################################################

#############################################################

      Parameters (maybe) needed for exporting files


 var gg = first_region.geometry()
 var gg = first_region.geometry().bounds().getInfo();

 print ("gg");
 print (gg);

var k = Grant_2018_IC_first.size()
var Grant_2018_IC_first_list = Grant_2018_IC_first.toList(k)

 print (Grant_2018_IC_first)
 print (typeof(Grant_2018_IC_first))

 var Grant_2018_IC_first_flatten = Grant_2018_IC_first.flatten()
 print (Grant_2018_IC_first_flatten)

#############################################################

Export.table.toDrive({
 collection: Grant_2018_IC_first_flatten,
 scale: 10,
 description:'Grant_2018_IC_first_flatten',
 folder:"Grant_2018_IC_first",
 fileFormat: 'CSV'
});

 var batch = require('users/fitoprincipe/geetools:batch');
 (batch.Download.ImageCollection.toDrive(Grant_2018_IC_first, 
                                         'LLL', 
                                         {scale:10}));

 batch.Download.ImageCollection.toDrive(Grant_2018_IC_first, 
                                       'LLL', 
                                       {scale: 10 ,
                                         region: gg // or geometry.getInfo()
                                     });

#############################################################

 Export.table.toDrive({collection: Grant_2018_IC_first, 
                       fileFormat: 'GeoJSON',
                       description:'Grant_2018_IC_first',
                       folder:"Grant_2018_IC_first",
                       });

   #############################################################

 ## We can NOT use the following, since we have imageCollection. 
 ## Not an image
 Export.image.toDrive({
   image: Grant_2018_IC_first,
   description: 'Grant_2018_IC_first',
   maxPixels: 1e9,
   scale:10,
   folder:"Grant_2018_IC_first"
 });


