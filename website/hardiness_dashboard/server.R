# hardiness dashboard

###  Shiny Server  ###
shinyServer(function(input, output, session) {
  ##################
  ##
  ##  Build Map ####
  ##
  ##################

  output$hard_map <- renderLeaflet({
    factpal <- colorFactor(palette = "RdBu", 
                           levels = sort(unique(spatial_hardiness_locs$freezing_years)))

    if (input$map_tile_ == "World Street"){
            leaflet() %>%
            addProviderTiles(providers$Esri.WorldStreetMap, 
                             options= providerTileOptions(opacity = 0.8))%>%
            setView(lat = 47, lng = -120, zoom = 7) %>%
            addCircleMarkers(data = spatial_hardiness_locs, 
                             lng = ~ long, lat = ~ lat,
                             label = ~ location,
                             layerId = ~ location,
                             radius = 6,
                             color = ~ factpal(freezing_years),
                             stroke  = FALSE,
                             fillOpacity = .95) %>%
            addLegend(position="bottomleft", 
                      pal = factpal, 
                      # colors = c("royalblue3", "steelblue1", "maroon3", "red", "black"),
                      values = unique(spatial_hardiness_locs$freezing_years),
                      labels = unique(spatial_hardiness_locs$freezing_years),
                      title = "No. years w/ damaging events")
        
          } else if (input$map_tile_ == "Sattelite") {
            leaflet() %>%
            addTiles(urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
                     layerId = "Satellite",
                     options= providerTileOptions(opacity = 0.8)) %>%
            setView(lat = 47, lng = -120, zoom = 7) %>%
            addCircleMarkers(data = spatial_hardiness_locs, 
                             lng = ~ long, lat = ~ lat,
                             label = ~ location,
                             layerId = ~ location,
                             radius = 6,
                             color = ~ factpal(freezing_years),
                             stroke  = FALSE,
                             fillOpacity = .95) %>%
            addLegend(position="bottomleft", 
                      pal = factpal, 
                      # colors = c("royalblue3", "steelblue1", "maroon3", "red", "black"),
                      values = unique(spatial_hardiness_locs$freezing_years),
                      labels = unique(spatial_hardiness_locs$freezing_years),
                      title = "No. years w/ damaging events")

          }else if (input$map_tile_ == "Open Topo"){
            leaflet() %>%
            addProviderTiles(providers$OpenTopoMap,
                             options= providerTileOptions(opacity = 0.8))%>%
            setView(lat = 47, lng = -120, zoom = 7) %>%
            addCircleMarkers(data = spatial_hardiness_locs, 
                             lng = ~ long, lat = ~ lat,
                             label = ~ location,
                             layerId = ~ location,
                             radius = 6,
                             color = ~ factpal(freezing_years),
                             stroke  = FALSE,
                             fillOpacity = .95) %>%
            addLegend(position="bottomleft", 
                      pal = factpal, 
                      # colors = c("royalblue3", "steelblue1", "maroon3", "red", "black"),
                      values = unique(spatial_hardiness_locs$freezing_years),
                      labels = unique(spatial_hardiness_locs$freezing_years),
                      title = "No. years w/ damaging events")
        

          } else {
            leaflet() %>%
            addTiles(urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
                     attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>') %>%
            setView(lat = 47, lng = -120, zoom = 7) %>%
            addCircleMarkers(data = spatial_hardiness_locs, 
                             lng = ~ long, lat = ~ lat,
                             label = ~ location,
                             layerId = ~ location,
                             radius = 6,
                             color = ~ factpal(freezing_years),
                             stroke  = FALSE,
                             fillOpacity = .95) %>%
            addLegend(position="bottomleft", 
                      pal = factpal, 
                      # colors = c("royalblue3", "steelblue1", "maroon3", "red", "black"),
                      values = unique(spatial_hardiness_locs$freezing_years),
                      labels = unique(spatial_hardiness_locs$freezing_years),
                      title = "No. years w/ damaging events")
    }


  })

  # plot part of bloom
  observeEvent(input$hard_map_marker_click, 
             { p <- input$hard_map_marker_click$id
               lat <- substr(as.character(p), 1, 8)
               long <- substr(as.character(p), 13, 21)
               print(p)
               file_dir_string <- observed_plot_dir
               toggleModal(session, modalId = "hard_graphs", toggle = "open")

              
               output$hard_plot <- renderImage({
                     image_name <- paste0(lat, "-", long, ".png")

                     filename <- normalizePath(file.path(file_dir_string, image_name))
                     # Return a list containing the filename and alt text
                     list(src = filename, width = 800, height = 550)
                                              }, deleteFile = FALSE
                                              )
            })
})
