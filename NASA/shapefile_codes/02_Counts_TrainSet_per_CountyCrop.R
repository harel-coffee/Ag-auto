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

Grant_count <- Grant %>% 
               group_by(county, CropTyp) %>% 
               summarise(count = n_distinct(ID)) %>% 
               data.table()

Grant_count$year = grant_year

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
Adam_count <- Adam %>% 
              group_by(county, CropTyp) %>% 
              summarise(count = n_distinct(ID)) %>% 
              data.table()
Adam_count$year = adambenton_year


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
Franklin_count <- Franklin %>% 
                  group_by(county, CropTyp) %>% 
                  summarise(count = n_distinct(ID)) %>% 
                  data.table()
Franklin_count$year = FrankYakima_year

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

walla_count <- walla %>% 
              group_by(county, CropTyp) %>% 
              summarise(count = n_distinct(ID)) %>% 
              data.table()
walla_count$year = walla_yr


#  monteray

Mont <- readOGR("/Users/hn/Documents/01_research_data/NASA/shapefiles/clean_Monterey/clean_Monterey.shp",
                 layer = "clean_Monterey", 
                 GDAL1_integer64_policy = TRUE)

Mont <- Mont@data
# Mont <- Mont %>% filter(County == "Monterey") %>% data.table()
Mont$county <- "Monterey"
#
# rename the  columns
#
setnames(Mont, old=c("Crop2014", "Acres"), new=c("CropTyp", "ExctAcr"))


Mont_count <- Mont %>% 
              group_by(county, CropTyp) %>% 
              summarise(count = n_distinct(ID)) %>% 
              data.table()
Mont_count$year = 2014


unique(Mont_count$county)
unique(walla_count$county)
unique(Franklin_count$county)
unique(Adam_count$county)
unique(Grant_count$county)

countyCropCount <- rbind(walla_count, Adam_count, Grant_count, Franklin_count, Mont_count)
countyCropCount <- countyCropCount[order(county, CropTyp)]

countyCropCount <- countyCropCount[, c("county", "year", "CropTyp", "count")]
write.csv(countyCropCount, 
          file = "/Users/hn/Documents/01_research_data/NASA/field_counts/SF_train_countyCropCount_correctYR_irr_NoNASS.csv",
          row.names=FALSE)


countyCount <- countyCropCount %>% 
               group_by(county, year) %>% 
               summarise(count = sum(count)) %>% 
               data.table()
countyCount <- countyCount[order(county)]
write.csv(countyCount, 
          file = "/Users/hn/Documents/01_research_data/NASA/field_counts/SF_train_countyCount_correctYR_irr_NoNASS.csv",
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
Grant_count <- Grant %>% 
             group_by(county, CropTyp) %>% 
             summarise(count = n_distinct(ID)) %>% 
             data.table()
Grant_count$year = grant_year

adambenton_year = 2016
WSDA <- readOGR(paste0(data_dir, "Eastern_", adambenton_year, "/Eastern_", adambenton_year, ".shp"),
                       layer = paste0("Eastern_", adambenton_year), 
                       GDAL1_integer64_policy = TRUE)

Adam <- WSDA[grepl('Adam', WSDA$county), ]
Benton <- WSDA[grepl('Benton', WSDA$county), ]
Adam <- raster::bind(Adam, Benton)
Adam <- Adam@data
Adam <- filter_out_non_irrigated_datatable(Adam)
AdamBenton_count <- Adam %>% 
                  group_by(county, CropTyp) %>% 
                  summarise(count = n_distinct(ID)) %>% 
                  data.table()
AdamBenton_count$year = adambenton_year

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
FranklinYakima_count <- Franklin %>% 
                      group_by(county, CropTyp) %>% 
                      summarise(count = n_distinct(ID)) %>% 
                      data.table()
FranklinYakima_count$year = FrankYakima_year

walla_yr = 2015
WSDA <- readOGR(paste0(data_dir, "Eastern_", walla_yr, "/Eastern_", walla_yr, ".shp"),
                       layer = paste0("Eastern_", walla_yr), 
                       GDAL1_integer64_policy = TRUE)

walla <- WSDA[grepl('Walla Walla', WSDA$county), ]
walla <- walla@data
walla <- filter_out_non_irrigated_datatable(walla)
unique(walla$DataSrc)

walla_count <- walla %>% 
             group_by(county, CropTyp) %>% 
             summarise(count = n_distinct(ID)) %>% 
             data.table()
walla_count$year = walla_yr


#  monteray

Mont <- readOGR("/Users/hn/Documents/01_research_data/NASA/shapefiles/clean_Monterey/clean_Monterey.shp",
                layer = "clean_Monterey",
                GDAL1_integer64_policy = TRUE)

Mont <- Mont@data
Mont$County <- "Monterey"
setnames(Mont, old=c("Crop2014", "Acres", "County"), new=c("CropTyp", "ExctAcr", "county"))


Mont_count <- Mont %>% 
            group_by(county, CropTyp) %>% 
            summarise(count = n_distinct(ID)) %>% 
            data.table()
Mont_count$year = 2014


unique(Mont_count$county)
unique(walla_count$county)
unique(FranklinYakima_count$county)
unique(AdamBenton_count$county)
unique(Grant_count$county)

countyCrop_count <- rbind(walla_count, AdamBenton_count, Grant_count, FranklinYakima_count, Mont_count)
countyCrop_count <- countyCrop_count[order(county, CropTyp)]

countyCrop_count <- countyCrop_count[, c("county", "year", "CropTyp", "count")]

write.csv(countyCrop_count, 
          file = "/Users/hn/Documents/01_research_data/NASA/field_counts/SF_train_countyCropCount_irr.csv",
          row.names=FALSE)


county_count <- countyCrop_count %>% 
                group_by(county, year) %>% 
                summarise(count = sum(count)) %>% 
                data.table()
county_count <- county_count[order(county)]
write.csv(county_count, 
          file = "/Users/hn/Documents/01_research_data/NASA/field_counts/SF_train_countyCount_irr.csv",
          row.names=FALSE)

