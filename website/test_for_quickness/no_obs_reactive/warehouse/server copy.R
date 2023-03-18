shinyServer(function(input, output, session) {

  
  output$water_right_map <- renderLeaflet({
    target_date <- as.Date(input$cut_date)

    # water_resource <- input$water_source_type
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
  
    curr_spatial <- curr_spatial %>% 
                  filter(county_type == input$countyType_id) %>% 
                  data.table()

    observeEvent(input$countyType_id, {
        #
        # from 
        # https://shiny.rstudio.com/reference/shiny/latest/updateSelectInput.html
        #
        subbasins <- sort(unique(curr_spatial$subbasin))
     
        # Can also set the label and select items
        updateSelectInput(session,
                          inputId = 'subbasins_id',
                          # label = '2. Select subbasins',
                          choices = subbasins,
                          selected = head(subbasins, 1)
                          )
    })

    observeEvent(input$subbasins_id, {

      print ("subbasins_id changed now and updated subbasins are")
      curr_spatial <- curr_spatial %>% 
                      filter(subbasin %in% input$subbasins_id) %>% 
                      data.table()

      print (unique(curr_spatial$subbasin))
      print("_______++++++++_________________")

    })

    curr_spatial[, colorr := ifelse(right_date < target_date, 
                                    "#FF3333", "#0080FF")]
    
    print("_______++++++++_________________")
    print ('after observe (unique(curr_spatial$subbasin))')
    print (unique(curr_spatial$subbasin))
    print("_______++++++++_________________")

    leaflet() %>%
    addTiles(urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
             attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
             layerId = "Satellite",
             options= providerTileOptions(opacity = 0.9)) %>%
    setView(lat = 47, lng = -120, zoom = 7) %>%
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

