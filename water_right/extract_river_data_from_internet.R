  
.libPaths("/data/hydro/R_libs35")
.libPaths()
# library(rgdal)
#===============
# LOAD PACKAGES
#===============
library(tidyverse)
library(maptools)

options(digit=9)
options(digits=9)

# Time the processing of this batch of files
start_time <- Sys.time()

url.river_data <- url("http://sharpsightlabs.com/wp-content/datasets/usa_rivers.RData")

# LOAD DATA
# - this will retrieve the data from the URL
load(url.river_data)

lines.rivers <- subset(lines.rivers, 
                       (STATE %in% c('WA')))

lines.rivers <- subset(lines.rivers, 
                       !(FEATURE %in% c("Shoreline",
                                        "Shoreline Intermittent",
                                        "Null", 
                                        "Closure Line",
                                        "Apparent Limit"
                                        )))
simple_shapefile_dir <- paste0("/data/hydro/users/", 
                               "Hossein/water_right/shapefile/")

all_streams_dir <- paste0(simple_shapefile_dir,
                          "rivers_USA/")

if (dir.exists(file.path(all_streams_dir)) == F){
  dir.create(path=file.path(all_streams_dir), recursive=T)
}

writeOGR(obj = all_streams_sp, 
         dsn = all_streams_dir,
         layer="rivers_USA", 
         driver="ESRI Shapefile")








