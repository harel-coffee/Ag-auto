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

##############################################################################

data_dir <- "/Users/hn/Documents/01_research_data/RangeLand/Data/Supriya/Nov30_HerbRatio/"

##############################################################################
#### ***NOTE***

## Supriya had named this file County_State.shp. However, it is averages over 
## counties. So, I renamed it to county_herb_ratio.shp.
## Then we got on state level as well which was called state.shp
## I renamed it state_herb_ratio.shp
county_herb_ratio <- read_sf(paste0(data_dir, "county_herb_ratio.shp"))

herbRatio <- data.table(data.frame(county_herb_ratio$GEOID, county_herb_ratio$Herb_Avgme,
                                   county_herb_ratio$Herb_SDstd, county_herb_ratio$Pixelscoun))

setnames(herbRatio, 
         old = c('County_State.GEOID','County_State.Pixelscoun','County_State.Herb_SDstd', 'County_State.Herb_Avgme'), 
         new = c('county_fips','pixel_count','herb_std', 'herb_avg'))

write.csv(herbRatio, file = paste0(data_dir, "county_herb_ratio.csv"), row.names=FALSE)

# nc = st_read(system.file(paste0(data_dir, "County_State.shp"), package="sf"))

#################################################################
#### I did the state level on Kamiak. Did not want
#### to install R on laptop when I was sick and at home.
#### Did it on CoLab.

module load r/4.3.0
R
library(sp)


data_dir <- "/data/project/agaid/h.noorazar/rangeland/supriya/Nov30_HerbRatio/"
state_herb_ratio <- sf::read_sf(paste0(data_dir, "state_herb_ratio.shp"))


