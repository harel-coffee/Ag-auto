# stats dashboard

col_inds <- list('CDI' = 9, 'no switch rate' = 7, "DDₕ" = 6, 'switch day' = 8, 'avg endo slope' = 10,
                 'avg eco slope' = 11, 'max biweekly avg endo slope' = 12, 'max biweekly avg eco slope' = 13)
leg_titles <- list(
  'CDI' = 'CDI', 'no switch rate' = 'no ecodormancy rate', "DDₕ" = "DDₕ",
  'switch day' = "days since Sep 1", 'avg endo slope' = 'slope', 'avg eco slope' = 'slope',
  'max biweekly avg endo slope' = 'slope', 'max biweekly avg eco slope' = 'slope')
scenarios <- list('RCP4.5' = 'rcp45', 'RCP8.5' = 'rcp85')
###  Shiny Server  ###
shinyServer(function(input, output, session) {
  ##################
  ##
  ##  Build Map ####
  ##
  ##################
  tile_opts <- providerTileOptions(opacity = 0.8)
  output$hard_map <- renderLeaflet({
    mask <- stats$model == input$model_
    if (input$model_ != obs_hist & input$scenario_ != 'N/A' & input$scenario_ != '1979-2016')
      mask <- mask & !is.na(stats$scenario) & stats$scenario == scenarios[[input$scenario_]] & 
        stats$range == input$time_frame_
    
    df <- stats[mask, ]
    names(df)[col_inds[[input$variable_]]] <- 'val'
    if (sum(!is.na(df$val)) > 0) {
      val_range <- range(df$val, na.rm = T)  
    } else {
      val_range <- 0:1
    }
    pal <- colorNumeric(palette = c('white', 'red'), na.color = "#000000", domain = val_range)
    leg_title <- leg_titles[[input$variable_]]
    # if (input$scenario_ == 'obs_hist') {
    #   disable('scenario_')
    # }
    if (input$map_tile_ == "World Street"){
      base_map <- leaflet() %>% addProviderTiles(providers$Esri.WorldStreetMap, options=tile_opts)
    } else if (input$map_tile_ == "Sattelite") {
      base_map <- leaflet() %>% addTiles(
        urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
        layerId = "Satellite", options=tile_opts)
    } else if (input$map_tile_ == "Open Topo"){
      base_map <- leaflet() %>% addProviderTiles(providers$OpenTopoMap, options=tile_opts)
    } else {
      base_map <- leaflet() %>% addTiles(
        urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
        attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>', options=tile_opts)
    }
    base_map %>% setView(lat = 47, lng = -120, zoom = 7) %>% addCircleMarkers(
      data = df, lng = ~ long, lat = ~ lat, label = ~ location, layerId = ~ location,
      radius = 6, color = ~ pal(val), stroke  = FALSE, fillOpacity = .95) %>% 
      addLegend(position="bottomleft", pal = pal, values = df$val, title = leg_title)
  })

  observeEvent(input$model_,{
    if (input$model_ != obs_hist) {
      updateSelectInput(session,"time_frame_", choices=time_frames[2:4], selected=ifelse(
        input$time_frame_ == time_frames[1], time_frames[2], input$time_frame_))
      updateSelectInput(session,"scenario_", choices=scenario_types, selected=ifelse(
        input$scenario_ == 'N/A', scenario_types[2], input$scenario_))
    } else {
      updateSelectInput(session,"time_frame_", choices=time_frames[1])
      updateSelectInput(session,"scenario_", choices='N/A')
    }
  })
  
  # output$scenario_ <- renderUI({
  #   if (input$model_ != obs_hist) {
  #     selectInput("scenario_", label="", choices=scenarios, selected = scenario[2], selectize = F)
  #   } else {
  #     return(NULL)
  #   }
  # })
  # 
  # output$scenario_ <- renderUI({
  #   if (input$model_ != obs_hist) {
  #     selectInput("time_frame_", label="", choices=time_frames, selected = time_frames[1], selectize = F)
  #   } else {
  #     return(NULL)
  #   }
  # })
  
  # observe({
  #   if (input$static == "A") {
  #     values$dyn <- input$dynamic
  #   } else {
  #     values$dyn <- NULL
  #   }
  # })
  
  observeEvent(input$hard_map_marker_click, 
             { p <- input$hard_map_marker_click$id
               lat <- substr(as.character(p), 1, 8)
               long <- substr(as.character(p), 13, 21)
               print(p)
               file_dir_string <- plot_dir
               toggleModal(session, modalId = "hard_graphs", toggle = "open")

              
               output$hard_plot <- renderImage({
                     image_name <- paste0('stats_', lat, "_", long, ".png")

                     filename <- normalizePath(file.path(file_dir_string, image_name))
                     # Return a list containing the filename and alt text
                     list(src = filename, width = 800, height = 550)
                                              }, deleteFile = FALSE
                                              )
            })
})
