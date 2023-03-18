library(foreign)
library(data.table)
library(dplyr)
library(rgdal)
library(sp)
library(rrd)


###### strada part

strada_dir <- "/Users/hn/Documents/01_research_data/remote_sensing/NASS_CDL/useless/WA_strata/"

strada_shp <- readOGR(paste0(strada_dir, "WA_strata_dd.shp"),
                      layer = paste0("WA_strata_dd"), 
                      GDAL1_integer64_policy = TRUE)
#
# nothing here ^^^
#
rm (strada_shp, strada_dir)

######
######
######
data_dir = paste0("/Users/hn/Documents/01_research_data/", 
                  "remote_sensing/NASS_CDL/2019_30m_cdls/")


ArcGIS10_7 <- read.dbf(paste0(data_dir, "ESRI_attribute_files/ArcGIS10.7.0_2019_30m_cdls.img.vat.dbf"), as.is=T)
ArcGIS10_3 <- read.dbf(paste0(data_dir, "ESRI_attribute_files/ArcGIS10.3.1_2019_30m_cdls.img.vat.dbf"), as.is=T)

#
# nothing here ^^^
#
rm(ArcGIS10_7, ArcGIS10_3)


RRD_file <- read_rdd(paste0(data_dir, "2019_30m_cdls.rde"))

######
######
######

######### CropScape from Kirti
CDL_dir = paste0("/Users/hn/Documents/01_research_data/", 
                  "remote_sensing/NASS_CDL/CDL_2019_53/")

CDL_2019_53 <- read.dbf(paste0(CDL_dir, "/CDL_2019_53.tif.vat.dbf"), as.is=T)

ArcGIS2019 <- read.dbf(paste0(CDL_dir, "/ArcGIS10.7.0_2019_30m_cdls.img.vat.dbf"), as.is=T)




######### CropScape from Kirti
library(cdlTools)

download_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                      "remote_sensing/NASS_CDL/download_by_cdlTools/")

cdlTools::getCDL(x = 'Washington', 
                 year = 2018, 
                 location = download_dir,
                 ssl.verifypeer = FALSE)

CDL_2019 <- data.table(read.dbf(paste0(download_dir, "/CDL_2019_53.tif.vat.dbf"), as.is=T))
CDL_2018 <- data.table(read.dbf(paste0(download_dir, "/CDL_2018_53.tif.vat.dbf"), as.is=T))
CDL_2017 <- data.table(read.dbf(paste0(download_dir, "/CDL_2017_53.tif.vat.dbf"), as.is=T))

CDL_2019$CLASS_NAME <- tolower(CDL_2019$CLASS_NAME)
CDL_2018$CLASS_NAME <- tolower(CDL_2018$CLASS_NAME)
CDL_2017$CLASS_NAME <- tolower(CDL_2017$CLASS_NAME)

dim(CDL_2019)
dim(CDL_2018)
dim(CDL_2017)

CDL_2019 <- CDL_2019[grepl('dbl', CDL_2019$CLASS_NAME), ]
CDL_2018 <- CDL_2018[grepl('dbl', CDL_2018$CLASS_NAME), ]
CDL_2017 <- CDL_2017[grepl('dbl', CDL_2017$CLASS_NAME), ]

dim(CDL_2019)
dim(CDL_2018)
dim(CDL_2017)
###############################################
a_dir <- "/Users/hn/Documents/01_research_data/remote_sensing/NASS_CDL/three_yearsCropScape/"
Scape_2019 <- data.table(read.dbf(paste0(a_dir, "/CDL_2019_clip_20200723132005_408635628.tif.vat.dbf"), as.is=T))

Scape_2019_tiff <- readTIFF(paste0(a_dir, "CDL_2019_clip_20200723132005_408635628.tif"), as.is=TRUE)

