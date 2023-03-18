shinyServer(function(input, output, session) {
  
  observe({
       water_resource <- input$water_source_type
       if (water_resource == "surfaceWater") {
           curr_spatial <- spatial_wtr_right %>% 
                           filter(WaRecRCWCl == "surfaceWater")
           curr_spatial <- data.table(curr_spatial)

           } else if (water_resource == "groundwater"){
            curr_spatial <- spatial_wtr_right %>% 
                            filter(WaRecRCWCl == "groundwater")
            curr_spatial <- data.table(curr_spatial)

           } else {
            curr_spatial <- spatial_wtr_right %>% 
                            filter(WaRecRCWCl %in% c("surfaceWater", 
                                                     "groundwater")
                                  )
            curr_spatial <- data.table(curr_spatial)
       }
      
      target_date <- as.Date(input$cut_date)
      curr_spatial[, color := ifelse(right_date < target_date, 
                                     "#FF3333", "#0080FF")]

      leafletProxy("a_map", data = curr_spatial) %>%
      clearShapes() %>%
      addCircleMarkers(data = curr_spatial, 
                       lng = ~long, lat = ~lat,
                       label = ~ popup,
                       # layerId = ~ location,
                       radius = 3,
                       color = ~ color,
                       stroke  = FALSE,
                       fillOpacity = .95 
                       )
  })

  output$a_map <- renderLeaflet({
     leaflet() %>%
     addTiles(urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
              attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
              layerId = "Satellite",
              options= providerTileOptions(opacity = 0.9)) %>%

     setView(lat = 47, lng = -120, zoom = 7)
  })
  
})
