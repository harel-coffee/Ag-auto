
data_dir <- paste0("/Users/hn/Desktop/Desktop/Ag/", 
                   "check_point/pre_microsoft_meeting/")

mins_file_dir <- paste0(data_dir, "/Mins_files/wsda2018shp/")
mins_file <- readOGR(paste0(mins_file_dir, "/WSDACro_2018.shp"),
                     layer = "WSDACro_2018", 
                     GDAL1_integer64_policy = TRUE)

mins_file_centoirds <- rgeos::gCentroid(WSDACrop2018_doublecrop, byid=TRUE)

# these parameters came from shape file itself.
# you may see differnet things online!
# if these are not exact, of course results are not exact
crs <- CRS("+proj=lcc 
           +lat_1=45.83333333333334 
           +lat_2=47.33333333333334 
           +lat_0=45.33333333333334 
           +lon_0=-120.5 +datum=WGS84")

centroid_coord <- spTransform(double_crops_centroids, 
                              CRS("+proj=longlat +datum=WGS84"))

centroid_coord_dt <- data.table(centroid_coord@coords)
