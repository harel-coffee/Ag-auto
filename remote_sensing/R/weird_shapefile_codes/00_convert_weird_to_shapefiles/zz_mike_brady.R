rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
library(sf)
library(foreign)

##########
########## Directories
##########
data_dir <- paste0("/Users/hn/Desktop/Desktop/Ag/", 
                   "check_point/remote_sensing/")

##########
########## TRUE SHAPEFILE
##########
##########***************************************************
##########
##########  Min's file is the shapefile of weird 2018 data.
##########
##########***************************************************
start_time <- Sys.time()
mins_file_dir <- paste0(data_dir, "/Mins_files/wsda2018shp/")
mins_file <- readOGR(paste0(mins_file_dir, "/WSDACro_2018.shp"),
                     layer = "WSDACro_2018", 
                     GDAL1_integer64_policy = TRUE)
print (paste0("Reading Min's shapefile takes:", 
              Sys.time() - start_time ))
##########
##########  weird shape files
##########

##########
########## read weird 2012-2017
##########
start_time <- Sys.time()
weird_2012_2018_dir <- paste0(data_dir, 
                              "2012_2018_weird_shapefile.gdb")

# list the layer names in their to read desired layer
ogrListLayers(weird_2012_2018_dir);
gdb <- path.expand(weird_2012_2018_dir)
WSDACrop_2012 <- readOGR(gdb, "WSDACrop_2012")
WSDACrop_2013 <- readOGR(gdb, "WSDACrop_2013")
WSDACrop_2014 <- readOGR(gdb, "WSDACrop_2014")
WSDACrop_2015 <- readOGR(gdb, "WSDACrop_2015")
WSDACrop_2016 <- readOGR(gdb, "WSDACrop_2016")
WSDACrop_2017 <- readOGR(gdb, "WSDACrop_2017")
WSDACrop_2018 <- readOGR(gdb, "WSDACrop_2018")

print (paste0("Reading 2012-2017 shapefile takes:", 
              Sys.time() - start_time ))

##########
########## read 2018 survey
##########
start_time <- Sys.time()
weird_2018_dir <- paste0(data_dir, 
                         "2018_weird_shape_file/", 
                         "2018WSDACrop.gdb")

ogrListLayers(weird_2018_dir)
gdb <- path.expand(weird_2018_dir)
WSDACrop_2018 <- readOGR(gdb, "WSDACrop_2018")

print (paste0("Reading 2018 shapefile takes:", 
              Sys.time() - start_time ))

summary(WSDACrop_2018)
##########
##########      convert to lowet the notes
##########
##########
WSDACrop_2012_data <- WSDACrop_2012@data
WSDACrop_2013_data <- WSDACrop_2013@data
WSDACrop_2014_data <- WSDACrop_2014@data
WSDACrop_2015_data <- WSDACrop_2015@data
WSDACrop_2016_data <- WSDACrop_2016@data
WSDACrop_2017_data <- WSDACrop_2017@data
WSDACrop_2018_data <- WSDACrop_2018@data

rm(WSDACrop_2012, WSDACrop_2013, WSDACrop_2014,
   WSDACrop_2015, WSDACrop_2016, WSDACrop_2017,
   WSDACrop_2018)

WSDACrop_2012_data$Notes <- tolower(WSDACrop_2012_data$Notes)
WSDACrop_2013_data$Notes <- tolower(WSDACrop_2013_data$Notes)
WSDACrop_2014_data$Notes <- tolower(WSDACrop_2014_data$Notes)
WSDACrop_2015_data$Notes <- tolower(WSDACrop_2015_data$Notes)
WSDACrop_2016_data$Notes <- tolower(WSDACrop_2016_data$Notes)
WSDACrop_2018_data$Notes <- tolower(WSDACrop_2018_data$Notes)


########## bind weird files
##########

# The followint takes too long. Who knows how long!
# start_time <- Sys.time()
# combined_2017_2018 <- raster::bind(WSDACrop_2017, WSDACrop_2018)
# print (paste0("binding weird files takes:", 
#                Sys.time() - start_time ))

##
## Number of columns are different!!!
##
sort(colnames(WSDACrop_2018_data))
sort(colnames(WSDACrop_2017_data))
#
# The 2018 data has extra columns: CoverCrop, Notes
#

WSDACrop_2018@data <- within(WSDACrop_2018_data, 
                             remove("CoverCrop", "Notes"))

setnames(WSDACrop_2018_data, 
         new=c("SHAPE_Area", "SHAPE_Length", 
               "Source", "RotationCropType"), 
         old=c("Shape_Area", "Shape_Length", 
               "DataSource", "RotationCrop"))

start_time <- Sys.time()
combined_2017_2018 <- rbind(WSDACrop_2012, 
                            WSDACrop_2013,
                            WSDACrop_2014,
                            WSDACrop_2015,
                            WSDACrop_2016,
                            WSDACrop_2017, 
                            WSDACrop_2018)

print (paste0("binding weird files takes:", 
               Sys.time() - start_time ))

write_dir <- paste0(data_dir, "combined_2017_2018")
if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

writeOGR(obj = combined_2017_2018, 
         dsn = write_dir, 
         layer="combined_2017_2018", 
         driver="ESRI Shapefile")

## writeOGR issue: 
## Field names abbreviated for ESRI Shapefile driver
## https://gis.stackexchange.com/questions/30785/how-to-stop-writeogr-from-abbreviating-field-names-when-using-esri-shapefile-d
combined_2017_2018_data <- data.table(combined_2017_2018_data)
saveRDS(combined_2017_2018_data,
        paste0(data_dir, "combined_2017_2018_data.rds"))


##########
##########  Just keep the date, not the time/hour in surveyDate 
########## columns
##########
combined_2017_2018_data <- readRDS(paste0(data_dir, 
                                          "/combined_2017_2018_data.rds"))

combined_2017_2018_data[, c("InitialSurveyDate", "InitialSurveytime") := tstrsplit(InitialSurveyDate, " ", fixed=TRUE)]
combined_2017_2018_data[, c("LastSurveyDate", "LastSurveytime") := tstrsplit(LastSurveyDate, " ", fixed=TRUE)]
combined_2017_2018_data <- within(combined_2017_2018_data, 
                                  remove("LastSurveytime", 
                                         "InitialSurveytime"))

combined_2017_2018_data$LastSurveyDate <- as.Date(combined_2017_2018_data$LastSurveyDate)
combined_2017_2018_data$InitialSurveyDate <- as.Date(combined_2017_2018_data$InitialSurveyDate)

combined_2017_2018_data$InitialSurveyYear <- substr(combined_2017_2018_data$InitialSurveyDate, 1, 4)
combined_2017_2018_data$LastSurveyYear <- substr(combined_2017_2018_data$LastSurveyDate, 1, 4)


print (dim(combined_2017_2018_data))

# Rotation crops
rotation_crops <- sort(unique(combined_2017_2018_data$RotationCrop))[18:187]

double_cp_data <- combined_2017_2018_data %>%
                  filter(RotationCrop %in% rotation_crops)%>%
                  data.table()

print (dim(double_cp_data))
################################################################
##########
########## data w/ weirdd stuff in rotationCrop col.
##########
no_rotation_cp_data <- combined_2017_2018_data %>%
                       filter(!(RotationCrop %in% rotation_crops))%>%
                       data.table()
dim(no_rotation_cp_data)
no_rotation_cp_data <- na.omit(no_rotation_cp_data, 
                               cols=c("RotationCrop"))
dim(no_rotation_cp_data)
no_rotation_cp_data <- no_rotation_cp_data %>%
                       filter(RotationCrop != " ") %>%
                       data.table()
dim(no_rotation_cp_data)
#################################################################
#
#           table for total double crop area per year
#
years <- sort(unique(double_cp_data$LastSurveyYear))
col_names <- c("year", "double_cp_area")
double_crop_area_table <- setNames(data.table(matrix(nrow = length(years), 
                                   ncol = length(col_names))), 
                                   col_names)

double_crop_area_table$year <- as.numeric(double_crop_area_table$year)
double_crop_area_table$double_cp_area <- as.numeric(double_crop_area_table$double_cp_area)
row_count <- 1
for (year in years){
  curr_dt <- double_cp_data %>%
             filter(LastSurveyYear == year)%>%
             data.table()
  
  double_crop_area_table[row_count, 1] <- as.numeric(year)
  double_crop_area_table[row_count, 2] <- as.numeric(sum(curr_dt$ExactAcres))
  row_count <- row_count + 1
}
write.table(double_crop_area_table, 
            file = paste0(data_dir, 
                          "double_crop_area_table.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")



