# Bloom - Vince

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
  spatial_bcf_data <- reactive({
    spatial_bcf
  })

  #
  # Create the map
  #
  output$bcf_map <- renderLeaflet({
    pal <- colorBin(palette = "plasma", reverse = TRUE,
                    domain = spatial_bcf_data()$lat, bins = 8, pretty=TRUE)
    leaflet() %>%
    addTiles(# urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
             # attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>'
             ) %>%
    setView(lat = 47, lng = -120, zoom = 7) %>%
    addCircleMarkers(data = spatial_bcf_data(), 
                     lng = ~ long, lat = ~ lat,
                     label = ~ location,
                     layerId = ~ location,
                     radius = 4,
                     color = ~ pal(lat),
                     stroke  = FALSE,
                     fillOpacity = .95)
  })

  #
  # Bloom Dashboard plots
  #
  observeEvent(input$bcf_map_marker_click, 
           { p <- input$bcf_map_marker_click$id
             lat <- substr(as.character(p), 1, 8)
             long <- substr(as.character(p), 13, 21)
             # file_dir_string <- paste0("/data/hnoorazar/", 
             #                           "bloom/plots/", 
             #                           "CM_locs/bloom_thresh_in_one",
             #                           "/no_obs/apple/thresh_75/")
             toggleModal(session,
                         modalId = "bcf_graphs", 
                         toggle =  "open")

             output$bcf_plot <- renderImage({

                    if (!(input$bcf_plot_fruit_type %in% c("pear", "cherry"))){
                       fruit_t = "apple" 
                       } else if (input$bcf_plot_fruit_type == "pear"){
                        fruit_t = "pear"
                       } else if (input$bcf_plot_fruit_type == "cherry"){
                         fruit_t = "cherry"
                    }
             
                    file_dir_string <- paste0("/data/hnoorazar/", 
                                              "bloom/plots/", 
                                              "CM_locs/bloom_thresh_in_one",
                                              "/no_obs/", 
                                              fruit_t, "/",
                                              input$cp_thresh, "/"
                                              )

                    if (input$bcf_plot_fruit_type == "pear"){
                      image_name <- paste0(lat, "_-", long, "_", 
                                           gsub(" ", "_", input$em_scenario), 
                                           "_Cripps_Pink",
                                           ".png")

                      } else if (input$bcf_plot_fruit_type == "cherry") {
                        image_name <- paste0(lat, "_-", long, "_", 
                                             gsub(" ", "_", input$em_scenario), 
                                             "_Cripps_Pink",
                                             ".png")
                      } else {
                        image_name <- paste0(lat, "_-", long, "_", 
                                      gsub(" ", "_", input$em_scenario), 
                                      "_", 
                                      input$bcf_plot_fruit_type,
                                      ".png")
                    }

                   filename <- normalizePath(file.path(file_dir_string, 
                                                       image_name))
                   
                   # Return a list containing the filename and alt text
                   list(src = filename, 
                        width = 620, height = 370)
                                            }, deleteFile = FALSE
                                            )
          })

  ########################### ANALOG W/ just side bar
  ########################### to choose County names from.
  #############################################
  ############################################# ANALOG WITH Global map
  #############################################
  #
  # Create the map
  #
  # output$analog_front_page <- renderLeaflet({
  #   map <- counties %>%
  #          leaflet(options = leafletOptions(zoomControl = TRUE,
  #                  minZoom = 4, maxZoom = 20, dragging = TRUE))  %>%
  #          addTiles() %>%
  #          setView(lng = -118.4942, lat = 46, zoom = 6) %>%
  #          addPolygons( fillColor = "green", fillOpacity = 0.5,
  #                      color = "black", opacity = 1.0, weight = .6, smoothFactor = 0.5,
  #                      highlightOptions = highlightOptions(color="white", 
  #                                                          weight=2, 
  #                                                          bringToFront = TRUE),
  #                      label= ~ NAME) %>%
  #          addPolylines(data = states, color = "black", opacity = 1, weight = 1.5)
  # })
  #############

  # output$map_bloom_doy <- renderLeaflet({
  #   # c("Historical", "2040's", "2060's", "2080's")
  #   layerlist = levels(bloom$ClimateGroup)  # diap$ClimateGroup

  #   if(input$cg_bloom == "Historical") {
  #     climate_group = input$cg_bloom
  #     future_version = "rcp85"
  #    }
  #    else {
  #     temp = tstrsplit(input$cg_bloom, "_")
  #     climate_group = unlist(temp[1])
  #     future_version = unlist(temp[2])
  #   }

  #   if(future_version == "rcp45") {
  #    bloom_d = bloom_rcp45
  #    }
  #    else {
  #    bloom_d = bloom
  #   }

  #   sub_bloom = subset(bloom_d, apple_type == input$apple_type & ClimateGroup == climate_group)
  #   sub_bloom$location = paste0(sub_bloom$latitude, "_", sub_bloom$longitude)
    
  #   medBloom = list(hist = subset(sub_bloom, ClimateGroup == layerlist[1]),
  #                   `2040` = subset(sub_bloom, ClimateGroup == layerlist[2]),
  #                   `2060` = subset(sub_bloom, ClimateGroup == layerlist[3]),
  #                   `2080` = subset(sub_bloom, ClimateGroup == layerlist[4]))
    
  #   BloomMap <- constructMap(medBloom, layerlist, palColumn = "medDoY", 
  #                            legendVals = seq(85, 165), "Median Day of Year")
  #   BloomMap
  # })
  ##########################################################
  # output$map_bloom_doy_100 <- renderLeaflet({
  #   layerlist = levels(bloom$ClimateGroup) #diap$ClimateGroup

  #   if(input$cg_bloom_100 == "Historical") {
  #        climate_group = input$cg_bloom_100
  #        future_version = "rcp85"
  #         } else {
  #        temp = tstrsplit(input$cg_bloom_100, "_")
  #        climate_group = unlist(temp[1])
  #        future_version = unlist(temp[2])
  #   }

  #   if(future_version == "rcp45") {
  #        bloom_d = bloom_rcp45_100
  #         } else { 
  #        bloom_d = bloom_rcp85_100
  #   }

  #   sub_bloom = subset(bloom_d, apple_type == input$apple_type & 
  #                      ClimateGroup == climate_group)
  #   sub_bloom$location = paste0(sub_bloom$latitude, "_", sub_bloom$longitude)
    
  #   medBloom = list( hist = subset(sub_bloom, ClimateGroup == layerlist[1]),
  #                    `2040` = subset(sub_bloom, ClimateGroup == layerlist[2]),
  #                    `2060` = subset(sub_bloom, ClimateGroup == layerlist[3]),
  #                    `2080` = subset(sub_bloom, ClimateGroup == layerlist[4]))
    
  #   BloomMap <- constructMap(medBloom, layerlist, 
  #                            palColumn = "medDoY", 
  #                            legendVals = seq(85,165), 
  #                            "Median Day of Year")
  #   BloomMap
  # })
  ##########################################################
  # output$map_bloom_doy_95 <- renderLeaflet({
  #   layerlist = levels(bloom$ClimateGroup) # diap$ClimateGroup

  #   if(input$cg_bloom_95 == "Historical") {
  #     climate_group = input$cg_bloom_95
  #     future_version = "rcp85"
  #     } else {
  #     temp = tstrsplit(input$cg_bloom_95, "_")
  #     climate_group = unlist(temp[1])
  #     future_version = unlist(temp[2])
  #   }

  #   if(future_version == "rcp45") {
  #     bloom_d = bloom_rcp45_95
  #     } else { 
  #     bloom_d = bloom_rcp85_95
  #   }

  #   sub_bloom = subset(bloom_d, apple_type == input$apple_type & 
  #                      ClimateGroup == climate_group)
  #   sub_bloom$location = paste0(sub_bloom$latitude, "_", sub_bloom$longitude)
    
  #   medBloom = list( hist = subset(sub_bloom, ClimateGroup == layerlist[1]),
  #                    `2040` = subset(sub_bloom, ClimateGroup == layerlist[2]),
  #                    `2060` = subset(sub_bloom, ClimateGroup == layerlist[3]),
  #                    `2080` = subset(sub_bloom, ClimateGroup == layerlist[4]))
    
  #   BloomMap <- constructMap(medBloom, 
  #                           layerlist, 
  #                           palColumn = "medDoY", 
  #                           legendVals = seq(85,165), 
  #                           "Median Day of Year")
  #   BloomMap
  # })

  output$map_bloom_doy_50 <- renderLeaflet({
    layerlist = levels(bloom_rcp45_50$ClimateGroup) # diap$ClimateGroup

    if(input$cg_bloom_50 == "Historical") {
      climate_group = input$cg_bloom_50
      future_version = "rcp85"
     } else {
      temp = tstrsplit(input$cg_bloom_50, "_")
      climate_group = unlist(temp[1])
      future_version = unlist(temp[2])
    }

    if(future_version == "rcp45") {
     bloom_d = bloom_rcp45_50
      } else { 
     bloom_d = bloom_rcp85_50
    }
    print (bloom_d$ClimateGroup)
    print (climate_group)
    sub_bloom = subset(bloom_d, 
                       apple_type == input$apple_type & 
                       ClimateGroup == climate_group)
    
    sub_bloom$location = paste0(sub_bloom$latitude, "_", 
                                sub_bloom$longitude)
    sub_bloom <- na.omit(sub_bloom)
    medBloom = list( hist = subset(sub_bloom, ClimateGroup == layerlist[1]),
                     `2040` = subset(sub_bloom, ClimateGroup == layerlist[2]),
                     `2060` = subset(sub_bloom, ClimateGroup == layerlist[3]),
                     `2080` = subset(sub_bloom, ClimateGroup == layerlist[4]))

    BloomMap <- constructMap(medBloom, 
                             layerlist, 
                             palColumn = "medDoY", 
                             legendVals = seq(50,140), #seq(min(sub_bloom$medDoY), max(sub_bloom$medDoY)), # seq(50,165), 
                             "Median Day of Year")
    BloomMap
  })

  #######################################################
  #######################################################
  #######################################################
  output$map_bloom_diff <- renderLeaflet({
    # diffType = as.integer(input$emerg_diff_type)
    
    temp = tstrsplit(input$cg_bloom_diff, "_")
    climate_group = unlist(temp[1])
    future_version = unlist(temp[2])

    if(future_version == "rcp45") {
        data = bloom_rcp45_50
        }
       else {
        data = bloom_rcp85_50
    }

    layerdiff = c("2040's - Historical", 
                  "2060's - Historical", 
                  "2080's - Historical")

    layerlist = levels(data$ClimateGroup)
    data$location = paste0(data$latitude, "_", data$longitude)

    data <- na.omit(data)
    sub_Bloom <- data %>% 
                 filter(apple_type == input$apple_type_diff) %>%
                 data.table()
    sub_Bloom <- subset(sub_Bloom, 
                        select = c(ClimateGroup, location, medDoY))
    
    # sub_Bloom <- subset(data, !is.na(ClimateGroup) & 
    #                          apple_type == input$apple_type_diff, 
    #                    select = c(ClimateGroup, location, medDoY))
    
    cgBloom = list(subset(sub_Bloom, ClimateGroup == layerlist[1]),
                   subset(sub_Bloom, ClimateGroup == layerlist[2]),
                   subset(sub_Bloom, ClimateGroup == layerlist[3]),
                   subset(sub_Bloom, ClimateGroup == layerlist[4]))
    
    if(layerdiff[1] == climate_group) {
      diffBloom = list(merge(cgBloom[[2]], cgBloom[[1]], by = c("location")))
    }
    else if(layerdiff[2] == climate_group) {
      diffBloom = list(merge(cgBloom[[3]], cgBloom[[1]], by = c("location")))
    }
    else if(layerdiff[3] == climate_group) {
      diffBloom = list(merge(cgBloom[[4]], cgBloom[[1]], by = c("location")))
    }
    diffBloom[[1]]$diff = diffBloom[[1]]$medDoY.y - diffBloom[[1]]$medDoY.x
    diffDomain = diffBloom[[1]]$diff

    BloomDiffMap <- constructMap(diffBloom, 
                                 layerdiff, 
                                 palColumn = "diff", 
                                 legendVals = seq(min(diffDomain), max(diffDomain)), # seq(8,48), 
                                 HTML("Median calendar day <br />difference from historical"), 
                                 RdBu_reverse)
    BloomDiffMap
  })


  # # Show page on click event for analog
  # observeEvent(input$analog_front_page_shape_click, 
  #              { 
  #                p <- input$analog_front_page_shape_click
  #                toggleModal(session, modalId = "Graphs", toggle = "open")
  #                county <- rgdal::readOGR("/data/hnoorazar/bloom/shape_files/simle_county/", 
  #                                          layer = "simpleCounty")
                 
  #                # get polygon of current selected county(boundary)
  #                dat <- data.frame(Longitude = c(p$lng), Latitude =c(p$lat))
  #                coordinates(dat) <- ~ Longitude + Latitude
  #                proj4string(dat) <- proj4string(county)
  #                current_county_name <- toString(over(dat, county)$NAME)
  #                current_state_fip <- toString(over(dat, county)$STATEFP)
  #                current_state_name <- st_cnty_names[st_cnty_names$state_fip == current_state_fip,]$state[1]

  #               output$Plot <- renderImage({if (input$detail_level == "all_models"){
  #                                             image_name <- paste0("all_mods_", 
  #                                                                  current_state_name, "_", 
  #                                                                  current_county_name, 
  #                                                                  ".png")
  #                                             curr_emission <- input$arrow_emission
  #                                              } else {
  #                                                image_name <- paste0("triple_", 
  #                                                                     current_county_name, "_",
  #                                                                     current_state_name, "_",
  #                                                                     input$climate_model, "_",
  #                                                                     input$time_period,
  #                                                                     ".png")
  #                                                curr_emission <- input$emission
  #                                            }
  #                                            file_dir_string <- paste0("/data/hnoorazar/bloom/plots/regional_plots/analog_plots/", 
  #                                                                      "1_sigma_", 
  #                                                                      curr_emission, "/", 
  #                                                                      image_name)

  #                                            filename <- normalizePath(file.path(file_dir_string))
  #                                            # Return a list containing the filename and alt text
  #                                             list(src = filename, width = 600, height = 600)
  #                                           }, 
  #                                          deleteFile = FALSE
  #                                          )
  # })
  #
  # Dashboard plots
  #
  # output$analog_plot <- renderImage({ image_name <- paste0(input$county, "_w_precip_", input$emission, ".png")
  #                                     filename <- normalizePath(file.path('/data/hnoorazar/bloom/plots/regional_plots/analog_plots', image_name))
  #                                     # Return a list containing the filename and alt text
  #                                     list(src = filename, width = 600, height = 600)}, 
  #                                    deleteFile = FALSE
  #                                    )
  
  ###################################################################
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
        # urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
        # attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>'
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
                              radius = 5,
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
      map = addLegend(map, "bottomleft", 
                      pal = pal, 
                      values = legendVals,
                      title = title,
                      labFormat = myLabelFormat(prefix = "  ", dates=TRUE),
                      opacity = 0.7) 
    }
    else {
       map = addLegend(map, "bottomleft", 
                       pal = pal, 
                       values = legendVals,
                       title = title,
                       labFormat = myLabelFormat(prefix = " "),
                       opacity = 0.7)
    }
    map
  }

})
