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

data_dir <- paste0("/Users/hn/Documents/01_research_data/NASA/shapefiles/Monterey/")

##############################################################################

Monterey <- readOGR(paste0(data_dir, "2014_Crop_Monterey_CDL.shp"),
                    layer = "2014_Crop_Monterey_CDL", 
                    GDAL1_integer64_policy = TRUE)

# Monterey$County[is.na(Monterey$County)] <- "na"
# NA_county <- Monterey[grepl('na', Monterey$County), ]
# Monterey <- Monterey[grepl('Monterey', Monterey$County), ]

# there is only one source: "Land IQ, LLC"
# there is only one comment:  NA
# there is only one County at this point:  Monterey
# Value and Majority are the same
# there is only one DateApply:  "July, 2014"
Monterey@data <- within(Monterey@data, remove(Comments, ModBy, Source, County, Value, DateApply))

add_identifier <- function(dt_df, year){
  dt_df@data <- tibble::rowid_to_column(dt_df@data, "ID")
  dt_df@data$ID <- paste0("Monterey", year, "_", dt_df@data$ID)
  return(dt_df)
}

Monterey <- add_identifier(Monterey, 2014)
setnames(Monterey@data, old=c("Crop2014", "Acres"), new=c("CropTyp", "ExctAcr"))

write_dir <- paste0("/Users/hn/Documents/01_research_data/NASA/shapefiles/")

writeOGR(obj = Monterey, 
         dsn = paste0(write_dir, "/", "clean_Monterey"),
         layer = "clean_Monterey", 
         driver = "ESRI Shapefile")



Monterey <- Monterey@data
# Monterey$Crop2014 <- tolower(Monterey$Crop2014)
# Monterey$Crop2014 <- sub("/", "_", Monterey$Crop2014)
# Monterey$Crop2014 <- sub(" ", "-", Monterey$Crop2014)
# Monterey$Crop2014 <- sub(" ", "-", Monterey$Crop2014)
# Monterey$Crop2014 <- sub(" ", "-", Monterey$Crop2014)
# Monterey$Crop2014 <- sub(" ", "-", Monterey$Crop2014)
# Monterey$Crop2014 <- sub(" ", "-", Monterey$Crop2014)
# Monterey$Crop2014 <- sub(",", "", Monterey$Crop2014)
# Mont$County <- "Monterey"
write.csv(Monterey, 
          "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/Monterey2014.csv", row.names = F)

