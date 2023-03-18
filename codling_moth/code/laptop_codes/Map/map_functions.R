library(shiny)
library(shinydashboard)
library(shinyBS)
library(rgdal)    # for readOGR and others
library(maps)
library(sp)       # for spatial objects
library(leaflet)  # for interactive maps (NOT leafletR here)
library(dplyr)    # for working with data frames
library(ggplot2)  # for plotting
library(data.table)
library(reshape2)
library(RColorBrewer)


constructMap <- function(mapLayerData, layerlist, palColumn, legendVals, title, gradient = "RdBu") {
  pal <- colorNumeric(
                      # palette = "Blues,Paired,Greens,Purples",
                      # check - https://www.nceas.ucsb.edu/~frazier/RSpatialGuides/colorPaletteCheatsheet.pdf
                      palette = gradient, # Spectral_reverse, #"Spectral",
                      domain = legendVals
                      )
    myLabelFormat = function(..., dates=FALSE){ 
      if(dates){ function(type = "numeric", cuts){ 
                          format(as.Date(cuts, origin = "2018-01-01"), "%b %d")
                        }  
      } else {
        labelFormat(...)
      }
    } 

    map <- leaflet() %>% 
           addTiles() %>% 
           addProviderTiles(providers$CartoDB.Positron) %>% 
           setView(# lat = 46.2, lng = -119.53, zoom = 7, 
                   lat = 47.40, lng = -119.53, zoom = 7)
    
    for(i in 1:length(mapLayerData)){
                loc = tstrsplit(mapLayerData[[i]]$location, "_")
                mapLayerData[[i]]$latitude = as.numeric(unlist(loc[1]))
                mapLayerData[[i]]$longitude = as.numeric(unlist(loc[2]))
                map <- addCircleMarkers(map,
                                        data = mapLayerData[[i]],
                                        lat = mapLayerData[[i]]$latitude, lng = mapLayerData[[i]]$longitude,
                                        stroke = FALSE,
                                        radius = 7,
                                        fillOpacity = 0.9,
                                        color = ~pal(mapLayerData[[i]][, get(palColumn)]) #,
                                        #popup=paste0("<b>", title, ": </b>", 
                                        #              round(mapLayerData[[i]][, get(palColumn)], 1),
                                        #              "<br/><b>Latitude: </b> ", mapLayerData[[i]]$latitude, 
                                        #              "<br/><b>Longitude: </b> ", mapLayerData[[i]]$longitude)
                                        )
              }
    
    if(title == "Median Day of Year") { map = addLegend(map, "bottomleft", pal = pal, values = legendVals,
                                                        title = title,
                                                        labFormat = myLabelFormat(prefix = "  ", dates=TRUE),
                                                        opacity = 0.7) 
    } else { map = addLegend(map, "bottomleft", pal = pal, values = legendVals,
                             title = title,
                             labFormat = myLabelFormat(prefix = " "),
                             opacity = 0.7)
          }
    map
  }

# addTiles() %>% 
# addTiles(urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png", attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>') %>% 
# addTiles() %>% addProviderTiles(providers$Esri.NatGeoWorldMap, group = "Nat Geo") %>%
# addTiles() %>% addProviderTiles(providers$Stamen.TonerLite, group = "Toner Lite") %>%
# addProviderTiles(providers$CartoDB.Positron) %>% 

