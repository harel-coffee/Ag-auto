
rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
# library(sf)
library(foreign)
library(ggplot2)
source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)

##############################################################################

data_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                    "remote_sensing/00_shapeFiles/0002_final_shapeFiles/", 
                    "000_Eastern_WA/Eastern_2018/")

##############################################################################

WSDA_2018 <- readOGR(paste0(data_dir, "Eastern_2018.shp"),
                     layer = "Eastern_2018", 
                     GDAL1_integer64_policy = TRUE)

IDs <- c("101258_WSDA_SF_2018")

subset <- WSDA_2018[WSDA_2018@data$ID %in% IDs,]

SF_centers <- rgeos::gCentroid(subset, byid=TRUE)

####### the following is not necessary in our case. All are on the same page!
crs <- CRS("+proj=lcc 
            +lat_1=45.83333333333334 
            +lat_2=47.33333333333334 
            +lat_0=45.33333333333334 
            +lon_0=-120.5 +datum=WGS84")

centroid_coord <- spTransform(SF_centers, 
                            CRS("+proj=longlat +datum=WGS84"))

centroid_coord_dt <- data.table(centroid_coord@coords)
setnames(centroid_coord_dt, old=c("x", "y"), new=c("long", "lat"))

"""
centroid_coord_dt
          long        lat
3: -119.168055, 46.3681265    46.3681265, -119.168055
"""


states <- map_data("state")
states_cluster <- subset(states, region %in% c("washington"))

subset_map <- ggplot() +
              geom_polygon(data = states_cluster, aes(x=long, y=lat, group=group), fill = "grey", color = "black") +
              geom_polygon(data = subset, aes(x=long, y=lat, group=group), fill = "grey47", color = "red") +
              coord_sf(xlim = c(-119.60, -119.00), ylim = c(45.80, 47.00))
plot_w <- 5
plot_h <- 5
ggsave(filename = paste0("3fields_map.pdf"),
       plot = subset_map, 
       width = plot_w, height = plot_h, units = "in", limitsize = FALSE,
       dpi = 400, device = "pdf",
       path = "/Users/hn/Desktop/")



subset_map <- ggplot() +
              geom_polygon(data = states_cluster, aes(x=long, y=lat, group=group), fill = "grey", color = "black") +
              geom_polygon(data = subset, aes(x=long, y=lat, group=group), fill = "grey47", color = "red")
plot_w <- 8
plot_h <- 5
ggsave(filename = paste0("3fields_map_WA.pdf"),
       plot = subset_map, 
       width = plot_w, height = plot_h, units = "in", limitsize = FALSE,
       dpi = 400, device = "pdf",
       path = "/Users/hn/Desktop/why_God_why/")



writeOGR(obj = subset, 
         dsn = paste0("/Users/hn/Desktop/why_God_why/", "/three_Fields_2018/"), 
         layer="three_Fields_2018", 
         driver="ESRI Shapefile")


