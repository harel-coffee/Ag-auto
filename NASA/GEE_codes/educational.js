//////////
////////// Create buffer arround some points in 2 ways:
//////////

var x_small_list = ee.List([-119.796600, -119.824520, -119.830150, -119.925224]);
var y_small_list = ee.List([46.9978835,   47.1593417,  47.1160363, 46.9893742]);

var small_tiles = ee.FeatureCollection(x_small_list.map(function(xcor){
    var feat = ee.FeatureCollection(y_small_list.map(function(ycor){
    return ee.Feature(ee.Geometry.Point([xcor,ycor]).buffer(small_tile_size).bounds());
}));
return feat;
})).flatten();
 
var buffer_1 = ee.Feature(ee.Geometry.Point([ee.Number(x_small_list.get(0)),
                              ee.Number(y_small_list.get(0))]).buffer(small_tile_size).bounds());
                              
var buffer_2 = ee.Feature(ee.Geometry.Point([ee.Number(x_small_list.get(1)),
                              ee.Number(y_small_list.get(1))]).buffer(small_tile_size).bounds());

var buffer_3 = ee.Feature(ee.Geometry.Point([ee.Number(x_small_list.get(2)),
                              ee.Number(y_small_list.get(2))]).buffer(small_tile_size).bounds());

var buffer_4 = ee.Feature(ee.Geometry.Point([ee.Number(x_small_list.get(3)),
                              ee.Number(y_small_list.get(3))]).buffer(small_tile_size).bounds());

var buffers = ee.FeatureCollection([buffer_1, buffer_2, buffer_3, buffer_4]);


///////////// OR 


var x_pairs = ee.List([[-119.796600, 46.9978835], 
                       [-119.824520, 47.1593417], 
                       [-119.830150, 47.1160363],
                       [-119.925224, 46.9893742]
                       ]);
                               
var small_tile_size = 125;

var buffer_1 = ee.Feature(ee.Geometry.Point(x_pairs.get(0)).buffer(small_tile_size).bounds());
var buffer_2 = ee.Feature(ee.Geometry.Point(x_pairs.get(1)).buffer(small_tile_size).bounds());
var buffer_3 = ee.Feature(ee.Geometry.Point(x_pairs.get(2)).buffer(small_tile_size).bounds());
var buffer_4 = ee.Feature(ee.Geometry.Point(x_pairs.get(3)).buffer(small_tile_size).bounds());


var buffers = ee.FeatureCollection([buffer_1, buffer_2, buffer_3, buffer_4]);
