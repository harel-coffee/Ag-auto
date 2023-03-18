library(geosphere)

stations <- read.csv('data/ag_stations_summary.csv')[c('Station_name', 'AG_lat', 'AG_long')]
n <- nrow(stations)
distances <- matrix(nrow = n, ncol = n)
for (i in 1:n)
  for (j in 1:n) {
    distances[i, j] <- distm(
      c(stations$AG_long[i], stations$AG_lat[i]), c(stations$AG_long[j], stations$AG_lat[j]),
      fun = distHaversine) / 1000
  }

df <- data.frame(distances)
colnames(df) <- stations$Station_name
rownames(df) <- stations$Station_name
write.csv(df, 'data/station_distances.csv')
