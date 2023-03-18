library(leaflet)
library(rgdal)
library(dplyr)


dir <- "/Users/hn/Documents/GitHub/Kirti/analogy/05_website/just_analog_practice/"

counties <- readOGR(paste0(dir, "/tl_2017_us_county/tl_2017_us_county.shp"),
                    layer = "tl_2017_us_county", GDAL1_integer64_policy = TRUE)

states <- readOGR(paste0(dir, "/tl_2017_us_state/tl_2017_us_state.shp"),
                    layer = "tl_2017_us_state", GDAL1_integer64_policy = TRUE)

####################################################################################
# SCRATCH
# library(sf)
# counties <- st_read(paste0(dir, "/tl_2017_us_county.shp"))
# counties <- counties %>% filter(STATEFP %in% c(16, 41, 53))
# counties <- as(counties, 'Spatial')
####################################################################################

# counties <- counties[counties$STATEFP != "02", ] # filter Alaska
# counties <- counties[counties$STATEFP != "15", ] # filter Hawaii
# counties <- counties[counties$STATEFP != "60", ] # filter Samoa
# counties <- counties[counties$STATEFP != "66", ] # filter Guam
# counties <- counties[counties$STATEFP != "69", ] # filter WHERE?
# counties <- counties[counties$STATEFP != "72", ] # filter Puerto Rico
# counties <- counties[counties$STATEFP != "78", ] # filter VIRGIN ISLANDS

# Extract just the three states [Idaho: 16],  [OR: 41], [WA: 53]
counties <- counties[counties@data$STATEFP %in% c("16", "41", "53"), ]

#####################################################################################
#
# Simplify polygons/shapefile for making the website faster
#
counties <- rmapshaper::ms_simplify(counties)

#####################################################################################
#
#      Good color combinations: (to my eye)
#
# (filling, border, highlight)
# 0 azure, black, red
# 1 lightblue, black, red
# 2 slategrey, white, black
# 3 green, black, white
#####################################################################################
# We have to either make the state borders thicker, or have three
# states in different colors so they are distinguishable

factpal <- colorFactor(topo.colors(5), counties$category)

counties %>%
leaflet() %>%
# addTiles() %>% 
setView(lng = -118.4942, lat = 47.2149, zoom = 5) %>%
addPolygons( fillColor = ~factpal(STATEFP), fillOpacity = 0.5,
             # The following line is associated with borders
             color = "black", opacity = 1.0, weight = .6, smoothFactor = 0.5,
             highlightOptions = highlightOptions(color="white", weight=2, bringToFront = TRUE),
             label= ~ NAME)

counties %>%
leaflet() %>%
# addTiles() %>% 
setView(lng = -118.4942, lat = 47.2149, zoom = 5) %>%
addPolygons( fillColor = ~colorBin("YlOrRd", NAME)(NAME), fillOpacity = 0.5,
             # The following line is associated with borders
             color = "black", opacity = 1.0, weight = .6, smoothFactor = 0.5,
             highlightOptions = highlightOptions(color="white", weight=2, bringToFront = TRUE),
             label= ~ NAME) 



%>%
addProviderTiles(providers$OpenStreetMap.HOT,
                 options = providerTileOptions(noWrap = TRUE))

%>%
addMarkers(~as.numeric(INTPTLON) , ~as.numeric(INTPTLAT), 
	       data = counties@data , 
	       clusterOptions = markerClusterOptions())


m <- leaflet() %>% 
     addTiles() %>% 
     setView(lng = -118.4942, lat = 47.2149, zoom = 6) %>%
     # addProviderTiles(providers$Esri.NatGeoWorldMap) %>%
     

