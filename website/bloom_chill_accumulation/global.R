# Bloom - Vince
library(scales)
library(lattice)
# library(ggmap)
library(jsonlite)
library(raster)

library(data.table)
library(shiny)
library(shinydashboard)
library(shinyBS)
library(maps)
library(rgdal)    # for readOGR and others
library(sp)       # for spatial objects
library(leaflet)  # for interactive maps (NOT leafletR here)
library(dplyr)    # for working with data frames
library(ggplot2)  # for plotting
library(reshape2)
library(RColorBrewer)
# library(plotly)
# library(Hmisc)

data_dir = "/data/hnoorazar/codling_moth/"

# bloom <- data.table(readRDS(paste0(data_dir, "/bloom_data.rds")))
# bloom_rcp45 <- data.table(readRDS(paste0(data_dir, "/bloom_data_rcp45.rds")))

# bloom <- data.table(readRDS(paste0(data_dir, "/bloom_rcp85_50_new.rds")))
# bloom_rcp45 <- data.table(readRDS(paste0(data_dir, "/bloom_rcp45_50_new.rds")))

# print (colnames(bloom_rcp45))
# ##########################
# bloom_rcp85_100 <- data.table(readRDS(paste0(data_dir, "/bloom_rcp85_100_new.rds")))
# bloom_rcp45_100 <- data.table(readRDS(paste0(data_dir, "/bloom_rcp45_100_new.rds")))

# bloom_rcp85_95 <- data.table(readRDS(paste0(data_dir, "/bloom_rcp85_95_new.rds")))
# bloom_rcp45_95 <- data.table(readRDS(paste0(data_dir, "/bloom_rcp45_95_new.rds")))

bloom_rcp85_50 <- data.table(readRDS(paste0(data_dir, "/bloom_rcp85_50_new.rds")))
bloom_rcp45_50 <- data.table(readRDS(paste0(data_dir, "/bloom_rcp45_50_new.rds")))
print (bloom_rcp45_50$ClimateGroup)

######################################
###################################### clear above
######################################
##########################
########################## For Analog Map
##########################

#########################################################
# read county shapefile
shapefile_dir <- "/data/hnoorazar/bloom/shape_files/tl_2017_us_county/"
shapefile_dir <- "/data/hnoorazar/bloom/shape_files/tl_2017_us_county_simple/"
counties <- rgdal::readOGR(dsn=path.expand(shapefile_dir), 
                           layer = "tl_2017_us_county")

# Extract just the three states OR: 41, WA:53, ID: 16
counties <- counties[counties@data$STATEFP %in% c("16", "41", "53"), ]

#
# Compute states like so, to put border around states
states <- aggregate(counties[, "STATEFP"], 
                    by = list(ID = counties@data$STATEFP), 
                    FUN = unique, dissolve = T)

interest_counties <- c("16027", "53001", "53021", "53071",
                       "41021", "53005", "53025", "53077", 
                       "41027", "53007", "53037",  
                       "41049", "53013", "53039", 
                       "41059", "53017", "53047")

counties <- counties[counties@data$GEOID %in% interest_counties, ]

spatial_bcf_dir <- "/data/hnoorazar/bloom/just_CM_locs/"
spatial_bcf <- readRDS(paste0(spatial_bcf_dir, "cm_spatial_bcf.rds")) %>% 
                       group_by(location, lat, long)

################################################################################


# analog_param_dir <- "/data/hnoorazar/bloom/params/"
# st_cnty_names <- read.csv(paste0(analog_param_dir, 
#                                  "17_counties_fips_unique.csv"),
#                           header=T,
#                           as.is=T) %>% 
#                  data.table()

# emissions <- c("RCP 8.5" = "rcp85",
#                "RCP 4.5" = "rcp45")

######################################################

RdBu_reverse <- rev(brewer.pal(11, "RdBu"))

