# plot chill sites in the map

rm(list=ls())
library(data.table)
library(dplyr)
library(ggmap)
library(ggplot2)
library(leaflet)
options(digit=9)
options(digits=9)

source_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_plot_core.R"
source(source_path)

##########################################################################################
###                                                                                    ###
###                                    Driver                                          ###
###                                                                                    ###
##########################################################################################
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"
locations_wanted <- read.csv(paste0(param_dir, "LocationGroups_NoMontana.csv"), header=T, as.is=T)

########################## Needs Google API stuff.
# # getting the map
# mapgilbert <- get_map(location = c(lon = mean(locations_wanted$lon), lat = mean(locations_wanted$lat)), zoom = 4,
#                       maptype = "satellite", scale = 2)

# # plotting the map with some points on it
# ggmap(mapgilbert) +
#   geom_point(data = locations_wanted, aes(x = lon, y = lat, fill = "red", alpha = 0.8), size = 5, shape = 21) +
#   guides(fill=FALSE, alpha=FALSE, size=FALSE)



## coordinates(locations_wanted) <- ~longitude+latitude
leaflet(locations_wanted) %>% addCircles() %>% addTiles()

sites_on_Esri <- leaflet(locations_wanted) %>% 
                 addCircles() %>% 
                 addRectangles( lng1=-124.513197, lat1=41.995776,
                                lng2=-115.84610, lat2=49.057503, 
                                fillColor = "transparent") %>%
                 addProviderTiles(providers$Esri.WorldStreetMap, options= providerTileOptions(opacity = 0.8)) %>%
                 setView(lat = 46.5, lng = -119, zoom = 7)

mapshot(sites_on_map, file = "/Users/hn/Documents/sites_on_map.png")

saveWidget(sites_on_map, "temp.html", selfcontained = FALSE)
webshot("temp.html", 
        file = "sites_on_map.png",
        cliprect = "viewport")


sites_on_sattelite <- leaflet(locations_wanted) %>% 
                      addCircles(color="red") %>% 
                      addTiles(urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                               attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
                               layerId = "Satellite",
                               options= providerTileOptions(opacity = 0.8)) %>%
                      setView(lat = 46.5, lng = -119, zoom = 7)

sites_on_sattelite




sites_on_simple <- leaflet(locations_wanted) %>% addCircles() %>% addTiles()
sites_on_simple


#####
map(col="grey80", border = "grey40", fill = TRUE,
  xlim = c(-124.513197, -115.84610), ylim = c(41.99, 49.05), mar = rep(0.1, 4))

  