####
#### This module generates #s in the following link ()
#### https://docs.google.com/document/d/18KX24FkL70_Xhxagwx9EBRWeQmz-Ud-iuTXqnf9YXnk/edit?usp=sharing
####

rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
# library(sf)
library(foreign)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)

##############################################################################

data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/00_shapeFiles/0002_final_shapeFiles/000_Eastern_WA/")

##############################################################################

grant_year = 2017
WSDA <- readOGR(paste0(data_dir, "Eastern_", grant_year, "/Eastern_", grant_year, ".shp"),
                layer = paste0("Eastern_", grant_year), 
                GDAL1_integer64_policy = TRUE)


Grant <- WSDA[grepl('Grant', WSDA$county), ]

Grant <- Grant@data
Grant <- filter_out_non_irrigated_datatable(Grant)
Grant <- filter_lastSrvyDate_DataTable(Grant, grant_year)
Grant <- Grant[Grant$DataSrc != "nass", ]
Grant_Acr <- Grant %>% 
             group_by(county, CropTyp) %>% 
             summarise(sum_ExctAcr = sum(ExctAcr)) %>% 
             data.table()
Grant_Acr$year = grant_year

adambenton_year = 2016
WSDA <- readOGR(paste0(data_dir, "Eastern_", adambenton_year, "/Eastern_", adambenton_year, ".shp"),
                       layer = paste0("Eastern_", adambenton_year), 
                       GDAL1_integer64_policy = TRUE)

Adam <- WSDA[grepl('Adam', WSDA$county), ]
Benton <- WSDA[grepl('Benton', WSDA$county), ]
Adam <- raster::bind(Adam, Benton)
Adam <- Adam@data
Adam <- filter_out_non_irrigated_datatable(Adam)
Adam <- filter_lastSrvyDate_DataTable(Adam, adambenton_year)
Adam <- Adam[Adam$DataSrc != "nass", ]
AdamBenton_Acr <- Adam %>% 
                  group_by(county, CropTyp) %>% 
                  summarise(sum_ExctAcr = sum(ExctAcr)) %>% 
                  data.table()
AdamBenton_Acr$year = adambenton_year


FrankYakima_year = 2018
WSDA <- readOGR(paste0(data_dir, "Eastern_", FrankYakima_year, "/Eastern_", FrankYakima_year, ".shp"),
                       layer = paste0("Eastern_", FrankYakima_year), 
                       GDAL1_integer64_policy = TRUE)

Franklin <- WSDA[grepl('Franklin', WSDA$county), ]
Yakima <- WSDA[grepl('Yakima', WSDA$county), ]
Franklin <- raster::bind(Franklin, Yakima)

Franklin <- Franklin@data
Franklin <- filter_out_non_irrigated_datatable(Franklin)
Franklin <- filter_lastSrvyDate_DataTable(Franklin, FrankYakima_year)
unique(Franklin$DataSrc)
Franklin <- Franklin[Franklin$DataSrc != "nass", ]
unique(Franklin$DataSrc)
FranklinYakima_Acr <- Franklin %>% 
                      group_by(county, CropTyp) %>% 
                      summarise(sum_ExctAcr = sum(ExctAcr)) %>% 
                      data.table()
FranklinYakima_Acr$year = FrankYakima_year



walla_yr = 2015
WSDA <- readOGR(paste0(data_dir, "Eastern_", walla_yr, "/Eastern_", walla_yr, ".shp"),
                       layer = paste0("Eastern_", walla_yr), 
                       GDAL1_integer64_policy = TRUE)

walla <- WSDA[grepl('Walla Walla', WSDA$county), ]
walla <- walla@data
walla <- filter_out_non_irrigated_datatable(walla)
walla <- filter_lastSrvyDate_DataTable(walla, walla_yr)
unique(walla$DataSrc)
walla <- walla[walla$DataSrc != "nass", ]
unique(walla$DataSrc)

walla_Acr <- walla %>% 
             group_by(county, CropTyp) %>% 
             summarise(sum_ExctAcr = sum(ExctAcr)) %>% 
             data.table()
walla_Acr$year = walla_yr


# monteray

Mont <- readOGR("/Users/hn/Documents/01_research_data/NASA/shapefiles/Monterey/2014_Crop_Monterey_CDL.shp",
                 layer = "2014_Crop_Monterey_CDL", 
                 GDAL1_integer64_policy = TRUE)

Mont <- Mont@data
# Mont <- Mont %>% filter(County == "Monterey") %>% data.table()
Mont$County <- "Monterey"
#
# rename the columns
#
setnames(Mont, old=c("Crop2014", "Acres", "County"), new=c("CropTyp", "ExctAcr", "county"))


Mont_Acr <- Mont %>% 
            group_by(county, CropTyp) %>% 
            summarise(sum_ExctAcr = sum(ExctAcr)) %>% 
            data.table()
Mont_Acr$year = 2014


unique(Mont_Acr$county)
unique(walla_Acr$county)
unique(FranklinYakima_Acr$county)
unique(AdamBenton_Acr$county)
unique(Grant_Acr$county)


sort(unique(walla$Irrigtn))
sort(unique(Franklin$Irrigtn))
sort(unique(Adam$Irrigtn))
sort(unique(Grant$Irrigtn))


sort(unique(walla$LstSrvD))
sort(unique(Franklin$LstSrvD))
sort(unique(Adam$LstSrvD))
sort(unique(Grant$LstSrvD))


countyCropAcr <- rbind(walla_Acr, AdamBenton_Acr, Grant_Acr, FranklinYakima_Acr, Mont_Acr)
countyCropAcr <- countyCropAcr[order(county, CropTyp)]

countyCropAcr <- countyCropAcr[, c("county", "year", "CropTyp", "sum_ExctAcr")]
countyCropAcr[,(c("sum_ExctAcr")) := round(.SD,0), .SDcols=c("sum_ExctAcr")]
write.csv(countyCropAcr, 
          file = "/Users/hn/Documents/01_research_data/NASA/SF_train_countyCropAcr_correctYR_irr_NoNASS.csv",
          row.names=FALSE)


countyAcr <- countyCropAcr %>% 
              group_by(county, year) %>% 
              summarise(sum_ExctAcr = sum(sum_ExctAcr)) %>% 
              data.table()
countyAcr[,(c("sum_ExctAcr")) := round(.SD,0), .SDcols=c("sum_ExctAcr")]
write.csv(countyAcr, 
          file = "/Users/hn/Documents/01_research_data/NASA/SF_train_countyAcr_correctYR_irr_NoNASS.csv",
          row.names=FALSE)

crop_types <- tolower(countyCropAcr$CropTyp)
write.csv(crop_types, 
          file = "/Users/hn/Documents/01_research_data/NASA/SF_train_unique_crops_NoNASS.csv",
          row.names=FALSE)



###############################################################################################################
###############################################################################################################
###############################################################################################################
#######
#######   only irrigated field and no other filter
#######
grant_year = 2017
WSDA <- readOGR(paste0(data_dir, "Eastern_", grant_year, "/Eastern_", grant_year, ".shp"),
                       layer = paste0("Eastern_", grant_year), 
                       GDAL1_integer64_policy = TRUE)


Grant <- WSDA[grepl('Grant', WSDA$county), ]

Grant <- Grant@data
Grant <- filter_out_non_irrigated_datatable(Grant)
Grant_Acr <- Grant %>% 
             group_by(county, CropTyp) %>% 
             summarise(sum_ExctAcr = sum(ExctAcr)) %>% 
             data.table()
Grant_Acr$year = grant_year

adambenton_year = 2016
WSDA <- readOGR(paste0(data_dir, "Eastern_", adambenton_year, "/Eastern_", adambenton_year, ".shp"),
                       layer = paste0("Eastern_", adambenton_year), 
                       GDAL1_integer64_policy = TRUE)

Adam <- WSDA[grepl('Adam', WSDA$county), ]
Benton <- WSDA[grepl('Benton', WSDA$county), ]
Adam <- raster::bind(Adam, Benton)
Adam <- Adam@data
Adam <- filter_out_non_irrigated_datatable(Adam)
AdamBenton_Acr <- Adam %>% 
                  group_by(county, CropTyp) %>% 
                  summarise(sum_ExctAcr = sum(ExctAcr)) %>% 
                  data.table()
AdamBenton_Acr$year = adambenton_year


FrankYakima_year = 2018
WSDA <- readOGR(paste0(data_dir, "Eastern_", FrankYakima_year, "/Eastern_", FrankYakima_year, ".shp"),
                       layer = paste0("Eastern_", FrankYakima_year), 
                       GDAL1_integer64_policy = TRUE)

Franklin <- WSDA[grepl('Franklin', WSDA$county), ]
Yakima <- WSDA[grepl('Yakima', WSDA$county), ]
Franklin <- raster::bind(Franklin, Yakima)

Franklin <- Franklin@data
Franklin <- filter_out_non_irrigated_datatable(Franklin)
unique(Franklin$DataSrc)
FranklinYakima_Acr <- Franklin %>% 
                      group_by(county, CropTyp) %>% 
                      summarise(sum_ExctAcr = sum(ExctAcr)) %>% 
                      data.table()
FranklinYakima_Acr$year = FrankYakima_year



walla_yr = 2015
WSDA <- readOGR(paste0(data_dir, "Eastern_", walla_yr, "/Eastern_", walla_yr, ".shp"),
                       layer = paste0("Eastern_", walla_yr), 
                       GDAL1_integer64_policy = TRUE)

walla <- WSDA[grepl('Walla Walla', WSDA$county), ]
walla <- walla@data
walla <- filter_out_non_irrigated_datatable(walla)
unique(walla$DataSrc)

walla_Acr <- walla %>% 
             group_by(county, CropTyp) %>% 
             summarise(sum_ExctAcr = sum(ExctAcr)) %>% 
             data.table()
walla_Acr$year = walla_yr


# monteray

Mont <- readOGR("/Users/hn/Documents/01_research_data/NASA/shapefiles/Monterey/2014_Crop_Monterey_CDL.shp",
                       layer = "2014_Crop_Monterey_CDL", 
                       GDAL1_integer64_policy = TRUE)

Mont <- Mont@data
# Mont <- Mont %>% filter(County == "Monterey") %>% data.table()
Mont$County <- "Monterey"

#
# rename the columns
#
setnames(Mont, old=c("Crop2014", "Acres", "County"), new=c("CropTyp", "ExctAcr", "county"))


Mont_Acr <- Mont %>% 
            group_by(county, CropTyp) %>% 
            summarise(sum_ExctAcr = sum(ExctAcr)) %>% 
            data.table()
Mont_Acr$year = 2014


unique(Mont_Acr$county)
unique(walla_Acr$county)
unique(FranklinYakima_Acr$county)
unique(AdamBenton_Acr$county)
unique(Grant_Acr$county)


sort(unique(walla$Irrigtn))
sort(unique(Franklin$Irrigtn))
sort(unique(Adam$Irrigtn))
sort(unique(Grant$Irrigtn))


sort(unique(walla$LstSrvD))
sort(unique(Franklin$LstSrvD))
sort(unique(Adam$LstSrvD))
sort(unique(Grant$LstSrvD))


countyCropAcr <- rbind(walla_Acr, AdamBenton_Acr, Grant_Acr, FranklinYakima_Acr, Mont_Acr)
countyCropAcr <- countyCropAcr[order(county, CropTyp)]

countyCropAcr <- countyCropAcr[, c("county", "year", "CropTyp", "sum_ExctAcr")]
countyCropAcr[,(c("sum_ExctAcr")) := round(.SD,0), .SDcols=c("sum_ExctAcr")]

write.csv(countyCropAcr, 
          file = "/Users/hn/Documents/01_research_data/NASA/SF_train_countyCropAcr_irr.csv",
          row.names=FALSE)


countyAcr <- countyCropAcr %>% 
              group_by(county, year) %>% 
              summarise(sum_ExctAcr = sum(sum_ExctAcr)) %>% 
              data.table()

countyAcr[,(c("sum_ExctAcr")) := round(.SD,0), .SDcols=c("sum_ExctAcr")]

write.csv(countyAcr, 
          file = "/Users/hn/Documents/01_research_data/NASA/SF_train_countyAcr_irr.csv",
          row.names=FALSE)

crop_types <- tolower(countyCropAcr$CropTyp)
write.csv(crop_types, 
          file = "/Users/hn/Documents/01_research_data/NASA/SF_train_unique_crops.csv",
          row.names=FALSE)

