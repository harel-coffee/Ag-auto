var triplets = dataset.map(function(image) {
  return image.reduceRegions({
    collection: points, 
    reducer: ee.Reducer.first().setOutputs(image.bandNames()), 
    scale: 250,
  }).map(function(feature) {
    return feature.set({
      'imageID': image.id(),
      'timeMillis': image.get('system:time_start')
    })
  });
}).flatten();
print(triplets) 

var format = function(table, rowId, colId, rowProperty, colProperty) {
  var rows = table.distinct(rowId); 
  var joined = ee.Join.saveAll('matches').apply({
    primary: rows, 
    secondary: table, 
    condition: ee.Filter.equals({
      leftField: rowId, 
      rightField: rowId
    })
  });
  return joined.map(function(row) {
      var values = ee.List(row.get('matches'))
        .map(function(feature) {
          feature = ee.Feature(feature);
          return [feature.get(colId), feature.get(colProperty)];
        }).flatten();
      return row.select([rowId, rowProperty]).set(ee.Dictionary(values));
    });
};

var results = format(triplets, 'imageID', 'featureID', 'timeMillis', 'NDVI');
print(results)

// Note that there's a dummy feature in there for the points ('null').
var transpose = format(triplets, 'featureID', 'imageID', 'null', 'NDVI');
print(transpose)

Export.table.toDrive({
  collection: results, 
  description: 'foo', 
  fileNamePrefix: 'foo', 
  fileFormat: 'CSV'
});
