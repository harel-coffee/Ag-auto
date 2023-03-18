// This is a copy of stackoverflow code
// https://gis.stackexchange.com/questions/280156/mosaicking-a-image-collection-by-date-day-in-google-earth-engine
var start = ee.Date('2014-10-01');
var finish = ee.Date('2018-03-31');

var collection = ee.ImageCollection('COPERNICUS/S1_GRD')
                 .filterDate(start, finish)
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                 .filter(ee.Filter.eq('instrumentMode', 'IW'))
                 .filterMetadata('resolution_meters', 'equals', 10)
                 .filterBounds(poly);

// Difference in days between start and finish
var diff = finish.difference(start, 'day')

// Make a list of all dates
var range = ee.List.sequence(0, diff.subtract(1)).map(function(day){return start.advance(day,'day')})

// Funtion for iteraton over the range of dates
function day_mosaics(date, newlist) {
  // Cast
  date = ee.Date(date)
  newlist = ee.List(newlist)

  // Filter collection between date and the next day
  var filtered = collection.filterDate(date, date.advance(1,'day'))

  // Make the mosaic
  var image = ee.Image(filtered.mosaic())

  // Add the mosaic to a list only if the collection has images
  return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist))
}

// Iterate over the range to make a new list, and then cast the list to an imagecollection
var newcol = ee.ImageCollection(ee.List(range.iterate(day_mosaics, ee.List([]))))
print(newcol)






