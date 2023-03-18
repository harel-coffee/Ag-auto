# stats DIFF dashboard
library(scales)
library(lattice)
# library(ggmap)
library(jsonlite)

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

na <- 'N/A'
rendered <- list('CDI' = F, 'DD_h' = F, 'no_eco_rate' = F, 'switch_day' = F, 'slope' = F)
###  Shiny Server  ###
shinyServer(function(input, output, session) {
  render_map <- function(base_map, grape_variety, model, time_frame, scenario, col, DDh, is_percent, leg_title, palette) {
    if (grape_variety == "Cabernet_Sauvignon"){
      stats = cabernet_stats
      hist <- cabernet_hist

    } else if(grape_variety == "Chardonnay"){
      stats = chardonnay_stats
      hist <- chardonnay_hist

    }
    else if(grape_variety == "Merlot"){
      stats = merlot_stats
      hist <- merlot_hist

    } else if(grape_variety == "Reisling"){
      stats = reisling_stats
      hist <- reisling_hist
    }
    # stats <- cabernet_stats
    # hist <- cabernet_hist

    mask <- (is.na(stats$DD_h) | stats$DD_h >= DDh) &
      stats$scenario == scenario & stats$range == time_frame
    if (model != 'avg')
      mask <- mask & stats$model == model
    df <- stats[mask, ]
    names(df)[names(df) == col] <- 'val'
    
    
    df <- merge(df[c('model', 'lat', 'long', 'location', 'val')], hist[c('location', col)], all.x=T)
    df$val <- df$val - df[, col]
    df <- aggregate(val ~ lat+long+location, data=df, mean)
    
    if (grepl('slope', col))
      col <- 'slope'
    rendered[[col]] <- T
    
    if (sum(!is.na(df$val)) > 0){# & !is_percent) {
      min_val <- min(df$val, na.rm = T)
      max_val <- max(df$val, na.rm = T)
      if (min_val < 0 & max_val > 0) {
        max_abs_val <- max(-min_val, max_val)
        val_range <- c(-max_abs_val, 0, max_abs_val)
      } else if (min_val >= 0) {
        val_range <- c(0, max_val)
        palette <- palette[2:3]
      } else if (max_val <= 0) {
        val_range <- c(min_val, 0)
        palette <- palette[1:2]
      } else if (min_val == 0 & max_val == 0) {
        val_range <- 0
        palette <- palette[2]
      }
    } else {
      val_range <- 0:2 # if either all values are NA (dummy range needed) #or the variable is a percent
    }
    if (endsWith(col, 'endo')) {
      palette <- palette[length(palette):1]
    }
    pal <- colorNumeric(palette = palette, na.color = "#000000", domain = val_range)
    
    if (is_percent) {
      lab_format <- labelFormat(suffix = "%", transform = function(x) 100 * x)
    } else {
      lab_format <- labelFormat()
    }

    if (sum(mask) > 0) {
      base_map %>% 
      clearControls() %>% 
      clearMarkers() %>% 
      addCircleMarkers(data = df, lng = ~long, lat = ~lat, label = ~paste(val, location, sep='; '), layerId = ~location,
                      radius = 6, color = ~pal(val), stroke  = FALSE, fillOpacity = 1) %>% 
      addLegend(position="bottomleft", pal=pal, values=val_range, title=leg_title, labFormat=lab_format, opacity=1) %>%
      addPolylines(data = states, color = "red", opacity = 1, weight = 2) %>%
      addPolylines(data = rocky, color = "blue", opacity = 1, weight = 2)

    } else {
      base_map %>% clearControls() %>% clearMarkers()
    }
  }
  
  render_map_wrapper <- function(base_map, grp_vrty, m_tf_s, col, DDh, is_percent, leg_title, palette=c('blue', 'white', 'red')) {
    render_map(base_map, grp_vrty, m_tf_s[1], m_tf_s[2], m_tf_s[3], col, DDh, is_percent, leg_title, palette)
  }
  ##################
  ##
  ##  Build Map ####
  ##
  ##################
#  tile_opts <- providerTileOptions(opacity = 1)
  output$map_CDI <- renderLeaflet(make_base_map(input$map_tile_cdi))
  output$map_DDh <- renderLeaflet(make_base_map(input$map_tile_ddh))
  output$map_no_switch <- renderLeaflet(make_base_map(input$map_tile_no_switch))
  output$map_switch_day <- renderLeaflet(make_base_map(input$map_tile_switch_day))
  output$map_slope <- renderLeaflet(make_base_map(input$map_tile_slope))
  observeEvent({input$scope_cdi
                input$grape_variety_cdi
                input$model_cdi
                input$DDh_cdi
                input$map_tile_cdi}, 
                render_map_wrapper(leafletProxy("map_CDI"), 
                                   input$grape_variety_cdi,
                                   extract_scope(input$scope_cdi, input$model_cdi), 
                                   'CDI', 
                                   input$DDh_cdi, T, ''))
  
  observeEvent({input$scope_ddh
                input$grape_variety_ddh
                input$model_ddh
                input$DDh_ddh
                input$map_tile_ddh}, 
                render_map_wrapper(leafletProxy("map_DDh"), 
                                   input$grape_variety_ddh,
                                   extract_scope(input$scope_ddh, input$model_ddh), 
                                   'DD_h', 
                                   input$DDh_ddh, F, 'DDₕ'))
  
  observeEvent({input$scope_no_switch
                input$grape_variety_no_switch
                input$model_no_switch
                input$DDh_no_switch
                input$map_tile_no_switch}, 
                render_map_wrapper(leafletProxy("map_no_switch"),
                                   input$grape_variety_no_switch, 
                                   extract_scope(input$scope_no_switch, input$model_no_switch), 
                                   'no_eco_rate', 
                                   input$DDh_no_switch, T, ''))
  
  observeEvent({input$scope_switch_day
                input$grape_variety_switch_day
                input$model_switch_day
                input$DDh_switch_day
                input$map_tile_switch_day}, 
                render_map_wrapper(leafletProxy("map_switch_day"),
                                   input$grape_variety_switch_day,
                                   extract_scope(input$scope_switch_day, input$model_switch_day), 
                                   'switch_day', input$DDh_switch_day, F, 'days since Sep-01'))
  
  observeEvent({input$scope_slope
                input$grape_variety_slope
                input$model_slope
                input$DDh_slope
                input$acclim
                input$averaging
                input$map_tile_slope},
                render_map_wrapper(leafletProxy("map_slope"), 
                                   input$grape_variety_slope,
                                   extract_scope(input$scope_slope, input$model_slope), 
                                   paste(input$averaging, 'slope', input$acclim, sep='_'),
                                   input$DDh_slope, 
                                   F, 'slope'))

    observeEvent(input$nav, {
    if(input$nav == "CDI" & !rendered$CDI){
      render_map_wrapper(leafletProxy("map_CDI"),
                         input$grape_variety_cdi,
                         extract_scope(input$scope_cdi, input$model_cdi), 
                         'CDI', input$DDh_cdi, T, '')

    } else if(input$nav == "DDh" & !rendered$DD_h){
      render_map_wrapper(leafletProxy("map_DDh"),
                         input$grape_variety_ddh,
                         extract_scope(input$scope_ddh, input$model_ddh), 
                         'DD_h', input$DDh_ddh, F, 'DDₕ')

    } else if(input$nav == "no switch" & !rendered$no_eco_rate){
      
      render_map_wrapper(leafletProxy("map_no_switch"), 
                         input$grape_variety_no_switch,
                         extract_scope(input$scope_no_switch, input$model_no_switch), 
                         'no_eco_rate', input$DDh_no_switch, T, '')

    } else if(input$nav == "switch day" & !rendered$switch_day){
      render_map_wrapper(leafletProxy("map_switch_day"), 
                         input$grape_variety_switch_day,
                         extract_scope(input$scope_switch_day, input$model_switch_day), 
                         'switch_day', input$DDh_switch_day, F, 'days since Sep-01')
    
    } else if(input$nav == "slope" & !rendered$slope){
      render_map_wrapper(leafletProxy("map_slope"),
                         input$grape_variety_slope,
                         extract_scope(input$scope_slope, input$model_slope), 
                         paste(input$averaging, 'slope', input$acclim, sep='_'),
                         input$DDh_slope, F, 'slope')
    }
  })

})


extract_scope <- function(scope, model) {
  if (scope == obs_hist) {
    model <- obs_hist
    time_frame <- '1979-2020'
    scenario <- na
  } else {
    time_frame_scenario <- unlist(strsplit(scope, "_"))
    time_frame <- time_frame_scenario[1]
    scenario <- time_frame_scenario[2]
  }
  
  c(model, time_frame, scenario)
}

make_base_map <- function(map_tile) {
  if (map_tile == "ws"){
    base_map <- leaflet() %>% addProviderTiles(providers$Esri.WorldStreetMap)

  } else if (map_tile == "sat") {
    base_map <- leaflet() %>% 
                addTiles(urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                         attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
                         layerId = "Satellite")

  } else if (map_tile == "ot"){
    base_map <- leaflet() %>% addProviderTiles(providers$OpenTopoMap)#, options=tile_opts
  }
  
  base_map %>% setView(lat = 45.5, lng = -115, zoom = 6)
  
}


