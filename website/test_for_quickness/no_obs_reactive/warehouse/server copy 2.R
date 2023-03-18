#
# from 
# https://shiny.rstudio.com/reference/shiny/latest/updateSelectInput.html
#


shinyServer(function(input, output, session) {
  curr_spatial <- spatial_wtr_right

  output$water_right_map <- renderLeaflet({
    target_date <- as.Date(input$cut_date)
    curr_spatial[, colorr := ifelse(right_date < target_date, 
                                    "#FF3333", "#0080FF")]

    #########################################################
    if (input$water_source_type == "surfaceWater") {
      curr_spatial <- spatial_wtr_right %>%
                      filter(WaRecRCWCl == "surfaceWater")%>%
                      data.table()

     } else if (input$water_source_type == "groundwater"){
      curr_spatial <- spatial_wtr_right %>%
                      filter(WaRecRCWCl == "groundwater")%>%
                      data.table()
       } else {
          curr_spatial <- spatial_wtr_right
    }
    #########################################################
    #########################################################
    # observeEvent(input$countyType_id, {
      #######################
      curr_spatial <- curr_spatial %>% 
                      filter(county_type == input$countyType_id) %>% 
                      data.table()
      #######################
    
      subbasins <- sort(unique(curr_spatial$subbasin))
     
      # Can also set the label and select items
      updateSelectInput(session,
                        inputId = 'subbasins_id',
                        choices = subbasins,
                        selected = head(subbasins, 1)
                        )
    # })
    ############################################# 
    #############################################
    observeEvent(input$subbasins_id, {
      curr_spatial <- curr_spatial %>% 
                      filter(subbasin %in% input$subbasins_id) %>% 
                      data.table()
      print (unique(curr_spatial$subbasin))
    })
  #############################################
    mean_lat <- mean(curr_spatial$lat) 
    mean_long <- mean(curr_spatial$long)  
    
    leaflet() %>%
    addTiles(urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
             attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
             layerId = "Satellite",
             options= providerTileOptions(opacity = 0.9)) %>%
    setView(lat = mean_lat, lng = mean_long, zoom = 7) %>%
    addCircleMarkers(data = curr_spatial, 
                     lng = ~long, lat = ~lat,
                     label = ~ popup,
                     # layerId = ~ location,
                     radius = 3,
                     color = ~ colorr,
                     stroke  = FALSE,
                     fillOpacity = .95 
                      )
 
  })

})

