


in_dir = "/Users/hn/Documents/GitHub/Kirti/codling_moth/code/parameters/"
counties <- data.table(read.table(paste0(in_dir, "/CropParamCRB.csv"), header = T, sep=","))

counties <- subset(counties, select=c("lat", "long", "countyname"))
counties$location <- paste0(counties$lat, "_", counties$long)
counties<- within(counties, remove(lat, long))

out_dir <- "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/"
write.table(counties, file = paste0(out_dir, "counties.csv"), sep=",", col.names=T, row.names=F)