# Lagoon

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
  ############################################# Bloom W/ Global map
  #############################################

  # Show page on click event for chill frost

  #
  # Create the map
  #
  output$lagoon_map <- renderLeaflet({
    # right order c("royalblue3", "steelblue1", "maroon3", "red", "black"), 
    factpal <- colorFactor(palette = c("royalblue3", "steelblue1", 
                                       "maroon3", "red", "black"), 
                           levels = sort(unique(spatial_lagoon$cluster)))

    leaflet() %>%
    addTiles(urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
             attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>') %>%
    addPolygons(data = skagit, fill = FALSE, 
                stroke = 1, color = 'black',
                smoothFactor = 2) %>% 
    
    addPolygons(data = whatcom, fill = FALSE, 
                stroke = 10, color = 'black',
                smoothFactor = 2) %>% 
    
    addPolygons(data = snohomish, fill = FALSE, 
                stroke = 10, color = 'black',
                smoothFactor = 2) %>% 
    
    setView(lat = 48, lng = -122, zoom = 8) %>%
    addCircleMarkers(data = spatial_lagoon, 
                     lng = ~ long, lat = ~ lat,
                     label = ~ location,
                     layerId = ~ location,
                     radius = 4,
                     color = ~ factpal(cluster),
                     stroke  = FALSE,
                     fillOpacity = .95) %>%
    addLegend(position="bottomleft", 
              pal = factpal, 
              # colors = c("royalblue3", "steelblue1", "maroon3", "red", "black"),
              values = unique(spatial_lagoon$cluster),
              labels = unique(spatial_lagoon$cluster),
              title = "Cluster Subregions") 
  })

  #
  # Bloom Dashboard plots
  #
  observeEvent(input$lagoon_map_marker_click, 
           { p <- input$lagoon_map_marker_click$id
             lat <- substr(as.character(p), 1, 8)
             long <- substr(as.character(p), 13, 21)
             file_dir_string <- paste0("/data/hnoorazar/", 
                                       "bloom/plots/", 
                                       "CM_locs/bloom_thresh_in_one",
                                       "/no_obs/apple/thresh_75/")
             toggleModal(session,
                         modalId = "lagoon_graphs", 
                         toggle =  "open")

             output$lagoon_plot <- renderImage({
                   image_name <- paste0(lat, "_-", long, "_", 
                                        gsub(" ", "_", input$em_scenario), 
                                        "_", 
                                        input$lagoon_plot_fruit_type,
                                        ".png")

                   filename <- normalizePath(file.path(file_dir_string, 
                                                       image_name))
                   # Return a list containing the filename and alt text
                   list(src = filename, 
                        width = 620, height = 370)
                                            }, deleteFile = FALSE
                                            )
          })  
  
  ###################################################
  ###################################################
  ###################################################
  output$location_group <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/maps/',
                                                                            'clust_map_4_web.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, 
                                             width = 600, height = 400)}, 
                                        deleteFile = FALSE)

  output$cluster_visualization <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/maps/',
                                                                            'cluster_visualization_5.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 300, height = 550)}, 
                                        deleteFile = FALSE)

  ################################
  ########
  ######## Regional Plots - RCP 8.5
  ########
  ########
  ###
  ###   annual
  ###
  output$rain_85 <- renderImage({
                                 filename <- normalizePath(file.path(plot_dir, 
                                                                     '/precip/wtr_yr/',
                                                                     'wtr_yr_rain_85.png'))
                                 # Return a list containing the filename and alt text
                                 list(src = filename, width = 600, height = 600)}, 
                                      deleteFile = FALSE)

  output$rain_45 <- renderImage({
                                 filename <- normalizePath(file.path(plot_dir, 
                                                                     '/precip/wtr_yr/',
                                                                     'wtr_yr_rain_45.png'))
                                 # Return a list containing the filename and alt text
                                 list(src = filename, width = 600, height = 600)}, 
                                      deleteFile = FALSE)

  output$runoff_85 <- renderImage({
                                   filename <- normalizePath(file.path(plot_dir, 
                                                                       '/runoff/wtr_yr/',
                                                                       'wtr_yr_RCP85.png'))
                                   # Return a list containing the filename and alt text
                                   list(src = filename, width = 600, height = 400)}, 
                                   deleteFile = FALSE)

  output$runoff_45 <- renderImage({
                                   filename <- normalizePath(file.path(plot_dir, 
                                                                       '/runoff/wtr_yr/',
                                                                       'wtr_yr_RCP45.png'))
                                   # Return a list containing the filename and alt text
                                   list(src = filename, width = 600, height = 400)}, 
                                   deleteFile = FALSE)

  output$storm_85 <- renderImage({
                                  filename <- normalizePath(file.path(plot_dir, 
                                                                      '/storm/',
                                                                      'storm_85.png'))
                                  # Return a list containing the filename and alt text
                                  list(src = filename, width = 600, height = 400)}, 
                                  deleteFile = FALSE
                                )

  output$storm_45 <- renderImage({
                                  filename <- normalizePath(file.path(plot_dir, 
                                                                      '/storm/',
                                                                      'storm_45.png'))
                                  # Return a list containing the filename and alt text
                                  list(src = filename, width = 600, height = 400)}, 
                                  deleteFile = FALSE
                                )
  ################################
  ########
  ######## Western_Coastal
  ########
  ###
  ###   annual
  ###
  output$Western_coastal_annual_rain_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/wtr_yr/',
                                                                            'Western_coastal_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 600)}, 
                                        deleteFile = FALSE)
  output$Western_coastal_annual_runoff_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/wtr_yr/',
                                                                            'Western_coastal_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)
  output$Western_coastal_storm_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/storm/',
                                                                            'storm_Western_coastal_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)

  # output$Western_coastal_dsi_rcp85 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/storm/',
  #                                                                           'Western_coastal_85.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 350)}, 
  #                                       deleteFile = FALSE)
  ###
  ###   seasonal
  ###
  output$Western_coastal_seasonal_rain_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/seasonal/',
                                                                            'Western_coastal_rain_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)
  output$Western_coastal_seasonal_runoff_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/seasonal/',
                                                                            'Western_coastal_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)

  ###
  ###   monthly
  ###
  output$Western_coastal_monthly_rain_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/monthly/',
                                                                            'Western_coastal_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)
  output$Western_coastal_monthly_runoff_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/monthly/',
                                                                            'Western_coastal_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)
  
  #######################################################
  ########
  ######## Cascade_foothills
  ########
  ###
  ###   annual
  ###
  output$Cascade_foothills_annual_rain_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/wtr_yr/',
                                                                            'Cascade_foothills_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 600)}, 
                                        deleteFile = FALSE)
  output$Cascade_foothills_annual_runoff_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/wtr_yr/',
                                                                            'Cascade_foothills_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)
  output$Cascade_foothills_storm_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/storm/',
                                                                            'storm_Cascade_foothills_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)
  # output$Cascade_foothills_Annual_rain_rcp85 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/precip/wtr_yr/',
  #                                                                           'wtr_yr_rain_85.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 600)}, 
  #                                       deleteFile = FALSE)

  # output$Cascade_foothills_Annual_runoff_rcp85 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/runoff/wtr_yr/',
  #                                                                           'wtr_yr_RCP85.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 400)}, 
  #                                       deleteFile = FALSE)

  # output$Cascade_foothills_dsi_rcp85 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/storm/',
  #                                                                           'Cascade_foothills_85.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 350)}, 
  #                                       deleteFile = FALSE)
  ###
  ###   seasonal
  ###
  output$Cascade_foothills_seasonal_rain_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/seasonal/',
                                                                            'Cascade_foothills_rain_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Cascade_foothills_seasonal_runoff_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/seasonal/',
                                                                            'Cascade_foothills_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)

  ###
  ###   monthly
  ###
  output$Cascade_foothills_monthly_rain_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/monthly/',
                                                                            'Cascade_foothills_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Cascade_foothills_monthly_runoff_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/monthly/',
                                                                            'Cascade_foothills_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)
  
  ##########################################
  ########
  ######## Northwest_Cascades
  ########
  ########
  ###
  ###   annual
  ###
  output$Northwest_Cascades_annual_rain_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/wtr_yr/',
                                                                            'Northwest_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 600)}, 
                                        deleteFile = FALSE)
  output$Northwest_Cascades_annual_runoff_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/wtr_yr/',
                                                                            'Northwest_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)
  output$Northwest_Cascades_storm_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/storm/',
                                                                            'storm_Northwest_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)
  # output$Northwest_Cascades_Annual_rain_rcp85 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/precip/wtr_yr/',
  #                                                                           'wtr_yr_rain_85.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 600)}, 
  #                                       deleteFile = FALSE)

  # output$Northwest_Cascades_Annual_runoff_rcp85 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/runoff/wtr_yr/',
  #                                                                           'wtr_yr_RCP85.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 400)}, 
  #                                       deleteFile = FALSE)

  # output$Northwest_Cascades_dsi_rcp85 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/storm/',
  #                                                                           'Northwest_Cascades_85.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 350)}, 
  #                                       deleteFile = FALSE)
  ###
  ###   seasonal
  ###
  output$Northwest_Cascades_seasonal_rain_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/seasonal/',
                                                                            'Northwest_Cascades_rain_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Northwest_Cascades_seasonal_runoff_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/seasonal/',
                                                                            'Northwest_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)

  ###
  ###   monthly
  ###
  output$Northwest_Cascades_monthly_rain_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/monthly/',
                                                                            'Northwest_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Northwest_Cascades_monthly_runoff_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/monthly/',
                                                                            'Northwest_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)
  
  ##########################################
  ########
  ######## Northcentral_Cascades
  ########
  ########
  ###
  ###   annual
  ###
  output$Northcentral_Cascades_annual_rain_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/wtr_yr/',
                                                                            'Northcentral_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 600)}, 
                                        deleteFile = FALSE)
  output$Northcentral_Cascades_annual_runoff_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/wtr_yr/',
                                                                            'Northcentral_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)
  output$Northcentral_Cascades_storm_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/storm/',
                                                                            'storm_Northcentral_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)
  # output$Northcentral_Cascades_Annual_rain_rcp85 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/precip/wtr_yr/',
  #                                                                           'wtr_yr_rain_85.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 600)}, 
  #                                       deleteFile = FALSE)

  # output$Northcentral_Cascades_Annual_runoff_rcp85 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/runoff/wtr_yr/',
  #                                                                           'wtr_yr_RCP85.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 400)}, 
  #                                       deleteFile = FALSE)

  # output$Northcentral_Cascades_dsi_rcp85 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/storm/',
  #                                                                           'Northcentral_Cascades_85.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 350)}, 
  #                                       deleteFile = FALSE)
  ###
  ###   seasonal
  ###
  output$Northcentral_Cascades_seasonal_rain_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/seasonal/',
                                                                            'Northcentral_Cascades_rain_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Northcentral_Cascades_seasonal_runoff_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/seasonal/',
                                                                            'Northcentral_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)

  ###
  ###   monthly
  ###
  output$Northcentral_Cascades_monthly_rain_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/monthly/',
                                                                            'Northcentral_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Northcentral_Cascades_monthly_runoff_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/monthly/',
                                                                            'Northcentral_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)
  ##########################################
  ########
  ######## Northeast_Cascades
  ########
  ###
  ###   annual
  ###
  output$Northeast_Cascades_annual_rain_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/wtr_yr/',
                                                                            'Northeast_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 600)}, 
                                        deleteFile = FALSE)
  output$Northeast_Cascades_annual_runoff_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/wtr_yr/',
                                                                            'Northeast_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)
  output$Northeast_Cascades_storm_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/storm/',
                                                                            'storm_Northeast_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)
  # output$Northeast_Cascades_Annual_rain_rcp85 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/precip/wtr_yr/',
  #                                                                           'wtr_yr_rain_85.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 600)}, 
  #                                       deleteFile = FALSE)

  # output$Northeast_Cascades_Annual_runoff_rcp85 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/runoff/wtr_yr/',
  #                                                                           'wtr_yr_RCP85.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 400)}, 
  #                                       deleteFile = FALSE)

  # output$Northeast_Cascades_dsi_rcp85 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/storm/',
  #                                                                           'Northeast_Cascades_85.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 350)}, 
  #                                       deleteFile = FALSE)
  ###
  ###   seasonal
  ###
  output$Northeast_Cascades_seasonal_rain_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/seasonal/',
                                                                            'Northeast_Cascades_rain_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Northeast_Cascades_seasonal_runoff_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/seasonal/',
                                                                            'Northeast_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)

  ###
  ###   monthly
  ###
  output$Northeast_Cascades_monthly_rain_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/monthly/',
                                                                            'Northeast_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Northeast_Cascades_monthly_runoff_rcp85 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/monthly/',
                                                                            'Northeast_Cascades_85.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)

  ################################
  ########
  ######## Regional Plots - RCP 4.5
  ########
  ################################
  ########
  ######## Western_Coastal
  ########
  ###
  ###   annual
  ###
  output$Western_coastal_annual_rain_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/wtr_yr/',
                                                                            'Western_coastal_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 600)}, 
                                        deleteFile = FALSE)
  output$Western_coastal_annual_runoff_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/wtr_yr/',
                                                                            'Western_coastal_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)
  output$Western_coastal_storm_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/storm/',
                                                                            'storm_Western_coastal_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)

  # output$Western_coastal_dsi_rcp45 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/storm/',
  #                                                                           'Western_coastal_45.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 350)}, 
  #                                       deleteFile = FALSE)
  ###
  ###   seasonal
  ###
  output$Western_coastal_seasonal_rain_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/seasonal/',
                                                                            'Western_coastal_rain_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Western_coastal_seasonal_runoff_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/seasonal/',
                                                                            'Western_coastal_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)

  ###
  ###   monthly
  ###
  output$Western_coastal_monthly_rain_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/monthly/',
                                                                            'Western_coastal_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Western_coastal_monthly_runoff_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/monthly/',
                                                                            'Western_coastal_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)
  

  #######################################################
  ########
  ######## Cascade_foothills
  ########
  ###
  ###   annual
  ###
  output$Cascade_foothills_annual_rain_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/wtr_yr/',
                                                                            'Cascade_foothills_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 600)}, 
                                        deleteFile = FALSE)
  output$Cascade_foothills_annual_runoff_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/wtr_yr/',
                                                                            'Cascade_foothills_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)
  output$Cascade_foothills_storm_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/storm/',
                                                                            'storm_Cascade_foothills_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)

  # output$Cascade_foothills_Annual_rain_rcp45 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/precip/wtr_yr/',
  #                                                                           'wtr_yr_rain_45.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 600)}, 
  #                                       deleteFile = FALSE)

  # output$Cascade_foothills_Annual_runoff_rcp45 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/runoff/wtr_yr/',
  #                                                                           'wtr_yr_RCP45.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 400)}, 
  #                                       deleteFile = FALSE)

  # output$Cascade_foothills_dsi_rcp45 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/storm/',
  #                                                                           'Cascade_foothills_45.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 350)}, 
  #                                       deleteFile = FALSE)
  ###
  ###   seasonal
  ###
  output$Cascade_foothills_seasonal_rain_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/seasonal/',
                                                                            'Cascade_foothills_rain_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Cascade_foothills_seasonal_runoff_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/seasonal/',
                                                                            'Cascade_foothills_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)

  ###
  ###   monthly
  ###
  output$Cascade_foothills_monthly_rain_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/monthly/',
                                                                            'Cascade_foothills_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Cascade_foothills_monthly_runoff_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/monthly/',
                                                                            'Cascade_foothills_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)
  
  ##########################################
  ########
  ######## Northwest_Cascades
  ########
  ########
  ###
  ###   annual
  ###
  output$Northwest_Cascades_annual_rain_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/wtr_yr/',
                                                                            'Northwest_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 600)}, 
                                        deleteFile = FALSE)
  output$Northwest_Cascades_annual_runoff_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/wtr_yr/',
                                                                            'Northwest_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)
  output$Northwest_Cascades_storm_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/storm/',
                                                                            'storm_Northwest_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)
  # output$Northwest_Cascades_Annual_rain_rcp45 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/precip/wtr_yr/',
  #                                                                           'wtr_yr_rain_45.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 600)}, 
  #                                       deleteFile = FALSE)

  # output$Northwest_Cascades_Annual_runoff_rcp45 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/runoff/wtr_yr/',
  #                                                                           'wtr_yr_RCP45.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 400)}, 
  #                                       deleteFile = FALSE)

  # output$Northwest_Cascades_dsi_rcp45 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/storm/',
  #                                                                           'Northwest_Cascades_45.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 350)}, 
  #                                       deleteFile = FALSE)
  ###
  ###   seasonal
  ###
  output$Northwest_Cascades_seasonal_rain_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/seasonal/',
                                                                            'Northwest_Cascades_rain_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Northwest_Cascades_seasonal_runoff_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/seasonal/',
                                                                            'Northwest_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)

  ###
  ###   monthly
  ###
  output$Northwest_Cascades_monthly_rain_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/monthly/',
                                                                            'Northwest_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Northwest_Cascades_monthly_runoff_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/monthly/',
                                                                            'Northwest_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)
  
  ##########################################
  ########
  ######## Northcentral_Cascades
  ########
  ########
  ###
  ###   annual
  ###
  output$Northcentral_Cascades_annual_rain_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/wtr_yr/',
                                                                            'Northcentral_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 600)}, 
                                        deleteFile = FALSE)
  output$Northcentral_Cascades_annual_runoff_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/wtr_yr/',
                                                                            'Northcentral_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)
  output$Northcentral_Cascades_storm_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/storm/',
                                                                            'storm_Northcentral_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)

  # output$Northcentral_Cascades_Annual_rain_rcp45 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/precip/wtr_yr/',
  #                                                                           'wtr_yr_rain_45.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 600)}, 
  #                                       deleteFile = FALSE)

  # output$Northcentral_Cascades_Annual_runoff_rcp45 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/runoff/wtr_yr/',
  #                                                                           'wtr_yr_RCP45.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 400)}, 
  #                                       deleteFile = FALSE)

  # output$Northcentral_Cascades_dsi_rcp45 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/storm/',
  #                                                                           'Northcentral_Cascades_45.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 350)}, 
  #                                       deleteFile = FALSE)
  ###
  ###   seasonal
  ###
  output$Northcentral_Cascades_seasonal_rain_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/seasonal/',
                                                                            'Northcentral_Cascades_rain_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Northcentral_Cascades_seasonal_runoff_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/seasonal/',
                                                                            'Northcentral_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)

  ###
  ###   monthly
  ###
  output$Northcentral_Cascades_monthly_rain_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/monthly/',
                                                                            'Northcentral_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Northcentral_Cascades_monthly_runoff_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/monthly/',
                                                                            'Northcentral_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)
  ##########################################
  ########
  ######## Northeast_Cascades
  ########
  ###
  ###   annual
  ###
  output$Northeast_Cascades_annual_rain_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/wtr_yr/',
                                                                            'Northeast_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 600)}, 
                                        deleteFile = FALSE)
  output$Northeast_Cascades_annual_runoff_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/wtr_yr/',
                                                                            'Northeast_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)
  output$Northeast_Cascades_storm_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/storm/',
                                                                            'storm_Northeast_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 350, height = 400)}, 
                                        deleteFile = FALSE)

  # output$Northeast_Cascades_Annual_rain_rcp45 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/precip/wtr_yr/',
  #                                                                           'wtr_yr_rain_45.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 600)}, 
  #                                       deleteFile = FALSE)

  # output$Northeast_Cascades_Annual_runoff_rcp45 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/runoff/wtr_yr/',
  #                                                                           'wtr_yr_RCP45.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 400)}, 
  #                                       deleteFile = FALSE)

  # output$Northeast_Cascades_dsi_rcp45 <- renderImage({
  #                                       filename <- normalizePath(file.path(plot_dir, 
  #                                                                           '/storm/',
  #                                                                           'Northeast_Cascades_45.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 600, height = 350)}, 
  #                                       deleteFile = FALSE)
  ###
  ###   seasonal
  ###
  output$Northeast_Cascades_seasonal_rain_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/seasonal/',
                                                                            'Northeast_Cascades_rain_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Northeast_Cascades_seasonal_runoff_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/seasonal/',
                                                                            'Northeast_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)

  ###
  ###   monthly
  ###
  output$Northeast_Cascades_monthly_rain_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/precip/monthly/',
                                                                            'Northeast_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)

  output$Northeast_Cascades_monthly_runoff_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            '/runoff/monthly/',
                                                                            'Northeast_Cascades_45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 400)}, 
                                        deleteFile = FALSE)
  ####################
  #
  #     map images
  #
  
  output$storm_diff_45_16inch_diffGradient <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            'maps/storm/different_gradient/',
                                                                            'storm_diff_45_16inch_diffGradient.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 1000, height = 350)}, 
                                        deleteFile = FALSE)
  output$storm_diff_45_16inch_idenGradient <- renderImage({
                                               filename <- normalizePath(file.path(plot_dir, 
                                                                            'maps/storm/ident_gradient/',
                                                                            'storm_diff_45_16inch.png'))
                                               # Return a list containing the filename and alt text
                                               list(src = filename, width = 1000, height = 350)}, 
                                               deleteFile = FALSE)
  output$storm_diff_85_16inch_diffGradient <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            'maps/storm/different_gradient/',
                                                                            'storm_diff_85_16inch_diffGradient.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 1000, height = 350)}, 
                                        deleteFile = FALSE)
  output$storm_diff_85_16inch_idenGradient <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            'maps/storm/ident_gradient/',
                                                                            'storm_diff_85_16inch.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 1000, height = 350)}, 
                                        deleteFile = FALSE)
  
  output$precip_diff_45_16_diffGradient <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            'maps/precip/different_gradient/',
                                                                            'unbias_45_16_inch_wide.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 1000, height = 350)}, 
                                        deleteFile = FALSE)
  output$precip_diff_45_16_idenGradient <- renderImage({
                                               filename <- normalizePath(file.path(plot_dir, 
                                                                            'maps/precip/ident_gradient/',
                                                                            'unbias_45_16_inch_wide.png'))
                                               # Return a list containing the filename and alt text
                                               list(src = filename, width = 1000, height = 350)}, 
                                               deleteFile = FALSE)
  output$precip_diff_85_16_diffGradient <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            'maps/precip/different_gradient/',
                                                                            'unbias_85_16_inch_wide.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 1000, height = 350)}, 
                                        deleteFile = FALSE)
  output$precip_diff_85_16_idenGradient <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            'maps/precip/ident_gradient/',
                                                                            'unbias_85_16_inch_wide.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 1000, height = 350)}, 
                                        deleteFile = FALSE)

  output$runoff_diff_45_16_diffGradient <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            'maps/runoff/different_gradient/',
                                                                            'unbias_45_16_inch_wide.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 1000, height = 350)}, 
                                        deleteFile = FALSE)
  output$runoff_diff_45_16_idenGradient <- renderImage({
                                               filename <- normalizePath(file.path(plot_dir, 
                                                                            'maps/runoff/ident_gradient/',
                                                                            'unbias_45_16_inch_wide.png'))
                                               # Return a list containing the filename and alt text
                                               list(src = filename, width = 1000, height = 350)}, 
                                               deleteFile = FALSE)
  output$runoff_diff_85_16_diffGradient <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            'maps/runoff/different_gradient/',
                                                                            'unbias_85_16_inch_wide.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 1000, height = 350)}, 
                                        deleteFile = FALSE)
  output$runoff_diff_85_16_idenGradient <- renderImage({
                                        filename <- normalizePath(file.path(plot_dir, 
                                                                            'maps/runoff/ident_gradient/',
                                                                            'unbias_85_16_inch_wide.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 1000, height = 350)}, 
                                        deleteFile = FALSE)
  

  
  #######################################################
  #
  #     Functions
  #
  ####################################################################
  constructMap <- function(mapLayerData, layerlist, palColumn, 
                           legendVals, title, gradient = "RdBu") {
    pal <- colorNumeric(
      palette = gradient, #Spectral_reverse, #"Spectral",
      domain = legendVals
    )
    myLabelFormat = function(...,dates=FALSE){ 
      if(dates){ 
        function(type = "numeric", cuts){ 
          #as.Date(cuts, origin="1970-01-01")
          format(as.Date(cuts, origin = "2018-01-01"), "%b %d")
        }  
      }else{
        labelFormat(...)
      }
    } 
    
    map <- leaflet() %>% 
      addTiles(
        urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
        attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>'
      ) %>% 
      setView(lat = 47.40, lng = -119.53, zoom = 7)
    
    for(i in 1:length(mapLayerData)) {
      loc = tstrsplit(mapLayerData[[i]]$location, "_")
      mapLayerData[[i]]$latitude = as.numeric(unlist(loc[1]))
      mapLayerData[[i]]$longitude = as.numeric(unlist(loc[2]))
      
      map <- addCircleMarkers(map,
                              data = mapLayerData[[i]],
                              lat = mapLayerData[[i]]$latitude, 
                              lng = mapLayerData[[i]]$longitude,
                              stroke = FALSE,
                              radius = 7,
                              fillOpacity = 0.9,
                              color = ~pal(mapLayerData[[i]][, get(palColumn)]),
                              popup=paste0("<b>", title, ": </b>", 
                                           round(mapLayerData[[i]][, get(palColumn)], 1),
                                           "<br/><b>Latitude: </b> ", 
                                           mapLayerData[[i]]$latitude, 
                                           "<br/><b>Longitude: </b> ", 
                                           mapLayerData[[i]]$longitude))
    }
    
    if(title == "Median Day of Year") {
      map = addLegend(map, "bottomleft", pal = pal, values = legendVals,
                       title = title,
                       labFormat = myLabelFormat(prefix = "  ", dates=TRUE),
                       opacity = 0.7) 
    }
    else {
       map = addLegend(map, "bottomleft", pal = pal, values = legendVals,
                       title = title,
                       labFormat = myLabelFormat(prefix = " "),
                       opacity = 0.7)
    }

    map
  }


})
