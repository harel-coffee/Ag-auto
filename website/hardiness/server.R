# Hardiness

library(scales)
library(lattice)
# library(ggmap)
library(jsonlite)

library(data.table)
library(shiny)
library(shinydashboard)
library(shinyBS)
library(maps)
library(raster)
library(rgdal)    # for readOGR and others
library(sp)       # for spatial objects
library(leaflet)  # for interactive maps (NOT leafletR here)
library(dplyr)    # for working with data frames
library(ggplot2)  # for plotting
library(reshape2)
library(RColorBrewer)

shinyServer(function(input, output, session) {
  #############################################
  #############################################

  #
  # Create the map
  #
  output$hard_map <- renderLeaflet({
    pal <- colorBin(palette = "plasma", reverse = TRUE,
                    domain = spatial_hardiness_locs$color, 
                    bins = 8, pretty=TRUE)
    # This is hardiness

    factpal <- colorFactor(palette = "RdBu", 
                           levels = sort(unique(spatial_hardiness_locs$freezing_years)))

    leaflet() %>%
    addTiles(# urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
             # attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>'
             ) %>%
    setView(lat = 47, lng = -120, zoom = 7) %>%
    # addCircleMarkers(data = spatial_hardiness_locs, 
    #                  lng = ~ long, lat = ~ lat,
    #                  label = ~ location,
    #                  layerId = ~ location,
    #                  radius = 3,
    #                  color = ~ pal(color),
    #                  stroke  = FALSE,
    #                  fillOpacity = .95) %>%

    addCircleMarkers(data = spatial_hardiness_locs, 
                     lng = ~ long, lat = ~ lat,
                     label = ~ location,
                     layerId = ~ location,
                     radius = 5,
                     color = ~ factpal(freezing_years),
                     stroke  = FALSE,
                     fillOpacity = .95) %>%
    addLegend(position="bottomleft", 
              pal = factpal, 
              # colors = c("royalblue3", "steelblue1", "maroon3", "red", "black"),
              values = unique(spatial_hardiness_locs$freezing_years),
              labels = unique(spatial_hardiness_locs$freezing_years),
              title = "No. years w/\ndamaging events")
  })

  #
  # Bloom Dashboard plots
  #
  observeEvent(input$hard_map_marker_click, 
           { p <- input$hard_map_marker_click$id
             lat <- substr(as.character(p), 1, 8)
             long <- substr(as.character(p), 13, 21)
             toggleModal(session,
                         modalId = "hard_graph", 
                         toggle =  "open")

             output$hard_plot <- renderImage({image_name <- paste0(lat, "-", long, ".png")
                                              filename <- normalizePath(file.path(observed_plot_dir, image_name))
                                               list(src = filename, width = 650, height = 400)
                                              }, 
                                              deleteFile = FALSE
                                             )
          })

  #######################################################

})
