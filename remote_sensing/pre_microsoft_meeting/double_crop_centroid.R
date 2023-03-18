
data_dir <- paste0("/Users/hn/Desktop/Desktop/Ag/", 
                   "check_point/pre_microsoft_meeting/",
                   "filtered_shape_files/Min_double_crops/")

Min_double_crops <- readOGR(paste0(data_dir, "/Min_DoubleCrop.shp"),
                     layer = "Min_DoubleCrop", 
                     GDAL1_integer64_policy = TRUE)

double_crops_center <- rgeos::gCentroid(Min_double_crops, byid=TRUE)

# these parameters came from shape file itself.
# you may see differnet things online!
# if these are not exact, of course results are not exact
crs <- CRS("+proj=lcc 
           +lat_1=45.83333333333334 
           +lat_2=47.33333333333334 
           +lat_0=45.33333333333334 
           +lon_0=-120.5 +datum=WGS84")

centroid_coord <- spTransform(double_crops_center, 
                              CRS("+proj=longlat +datum=WGS84"))

centroid_coord_dt <- data.table(centroid_coord@coords)

centroid_coord_dt[,(c("x", "y")) := round(.SD,5), .SDcols=c("x", "y")]

setnames(centroid_coord_dt, 
         old=c("x", "y"), 
         new=c("longitude", "latitude"))

centroid_coord_dt$location <- paste0(centroid_coord_dt$latitude, " E, ",
                                     abs(centroid_coord_dt$longitude), " W")

write.table(centroid_coord_dt, 
            paste0(data_dir, "double_crop_centroid.csv"), 
            row.names = FALSE, col.names = TRUE, sep=",")



