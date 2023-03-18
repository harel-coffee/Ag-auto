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
  ###################################################
  ################################################### ANALOG WITH Global map
  ###################################################
  #
  # Create the map
  #
  output$analog_front_page <- renderLeaflet({
    map <- counties %>%
           leaflet(options = leafletOptions(zoomControl = TRUE,
                   minZoom = 4, maxZoom = 20, dragging = TRUE))  %>%
           addTiles() %>%
           setView(lng = -118.4942, lat = 46, zoom = 6) %>%
           addPolygons( fillColor = "green", fillOpacity = 0.5,
                       color = "black", opacity = 1.0, weight = .6, smoothFactor = 0.5,
                       highlightOptions = highlightOptions(color="white", 
                                                           weight=2, 
                                                           bringToFront = TRUE),
                       label= ~ NAME) %>%
           addPolylines(data = states, color = "black", opacity = 1, weight = 1.5)
  })
  #
  # Dashboard plots
  #
  # Show page on click event...
  observeEvent(input$analog_front_page_shape_click, 
               { 
                 p <- input$analog_front_page_shape_click
                 toggleModal(session, modalId = "Graphs", toggle = "open")
                 county <- rgdal::readOGR("/data/codmoth_data/analog/simle_county/", 
                                           layer = "simpleCounty")
                 
                 # get polygon of current selected county(boundary)
                 dat <- data.frame(Longitude = c(p$lng), Latitude =c(p$lat))
                 coordinates(dat) <- ~ Longitude + Latitude
                 proj4string(dat) <- proj4string(county)
                 current_county_name <- toString(over(dat, county)$NAME)
                 current_state_fip <- toString(over(dat, county)$STATEFP)
                 current_state_name <- st_cnty_names[st_cnty_names$state_fip == current_state_fip,]$state[1]

                output$Plot <- renderImage({if (input$detail_level == "all_models"){
                                              image_name <- paste0("all_mods_", 
                                                                   current_state_name, "_", 
                                                                   current_county_name, 
                                                                   ".png")
                                              curr_emission <- input$arrow_emission
                                               } else {
                                                 image_name <- paste0("triple_", 
                                                                      current_county_name, "_",
                                                                      current_state_name, "_",
                                                                      input$climate_model, "_",
                                                                      input$time_period,
                                                                      ".png")
                                                 curr_emission <- input$emission
                                             }
                                             file_dir_string <- paste0("./plots/analog_plots/", 
                                                                       "1_sigma_", 
                                                                       curr_emission, "/", 
                                                                       image_name)

                                             filename <- normalizePath(file.path(file_dir_string))
                                             # Return a list containing the filename and alt text
                                              list(src = filename, width = 600, height = 600)
                                            }, 
                                           deleteFile = FALSE
                                           )
  })

  # observe({

  # })

  ########################### ANALOG WITH just side bar
  ########################### to choose County names from.
  output$analog_plot <- renderImage({ image_name <- paste0(input$county, "_w_precip_", input$emission, ".png")
                                      filename <- normalizePath(file.path('./plots/analog_plots', image_name))
                                      # Return a list containing the filename and alt text
                                      list(src = filename, width = 600, height = 600)}, 
                                     deleteFile = FALSE
                                     )
  ###################################################
  ###################################################
  ###################################################
  output$location_group <- renderImage({filename <- normalizePath(file.path('./plots/', 'location-group.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 600)}, 
                                        deleteFile = FALSE)
  
  # output$ag_vplot <- renderImage({
  #   filename <- normalizePath(file.path('./plots/LarvaAdult', 'agenerationVsDoY.png'))
    
  #   # Return a list containing the filename and alt text
  #   list(src = filename, width = 1200, height = 800)
    
  # }, deleteFile = FALSE) # Hossein
  
  # output$lg_vplot <- renderImage({filename <- normalizePath(file.path('./plots/LarvaAdult', 
  #                                                                     'lgenerationVsDoY.png'))
  #                                # Return a list containing the filename and alt text
  #                                list(src = filename, width = 1200, height = 800)}, 
  #                                deleteFile = FALSE) # Hossein commented out
  
  output$ag_bplot <- renderImage({
                                  filename <- normalizePath(file.path('./plots/', 
                                                                      'Adult_Gen_Aug_rcp85.png'))
                                  # Return a list containing the filename and alt text
                                  list(src = filename, width = 600, height = 450)}, 
                                  deleteFile = FALSE)
  
  output$lg_bplot <- renderImage({
                                  filename <- normalizePath(file.path('./plots/', 
                                                                      'Larva_Gen_Aug_rcp85.png'))
                                  # Return a list containing the filename and alt text
                                  list(src = filename, width = 600, height = 450)}, 
                                  deleteFile = FALSE)
  
  # output$ag_vplot_rcp45 <- renderImage({filename <- normalizePath(file.path('./plots/LarvaAdult', 
  #                                                                 'agenerationVsDoY_rcp45.png'))
  #                                      # Return a list containing the filename and alt text
  #                                      list(src = filename, width = 1200, height = 800)}, 
  #                                      deleteFile = FALSE) # Hossein
  
  # output$lg_vplot_rcp45 <- renderImage({filename <- normalizePath(file.path('./plots/LarvaAdult', 
  #                                                               'lgenerationVsDoY_rcp45.png'))
  #                                       # Return a list containing the filename and alt text
  #                                       list(src = filename, width = 1200, height = 800)}, 
  #                                       deleteFile = FALSE) # Hossein commented out
  
  output$ag_bplot_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path('./plots/', 
                                                                 'Adult_Gen_Aug_rcp45.png'))
                                        # Return a list containing the filename and alt text
                                        list(src = filename, width = 600, height = 450)}, 
                                        deleteFile = FALSE)
  
  output$lg_bplot_rcp45 <- renderImage({
                                        filename <- normalizePath(file.path('./plots/', 
                                                            'Larva_Gen_Aug_rcp45.png'))
                                        # Return a list containing the filename and alt text
                                      list(src = filename, width = 600, height = 450)}, 
                                      deleteFile = FALSE)

  #  output$ap_vplot <- renderImage({
  #    filename <- normalizePath(file.path('./plots/LarvaAdult', 'aMonthVsPop.png'))
  #    
  #    # Return a list containing the filename and alt text
  #    list(src = filename, width = 1200, height = 900)
  #    
  #  }, deleteFile = FALSE)
  #  
  #  output$lp_vplot <- renderImage({
  #    filename <- normalizePath(file.path('./plots/LarvaAdult', 'lMonthVsPop.png'))
  #    
  #    # Return a list containing the filename and alt text
  #    list(src = filename, width = 1200, height = 900)
  #    
  #  }, deleteFile = FALSE)

  output$gen_pop_plot <- renderImage({filename <- normalizePath(file.path('./plots/LarvaAdult', 
                                                                          'gen_pop.png'))
                                      # Return a list containing the filename and alt text
                                      list(src = filename, width = 1200, height = 900)}, 
                                      deleteFile = FALSE)

  output$gen_pop_plot1 <- renderImage({filename <- normalizePath(file.path('./plots/LarvaAdult', 
                                                                           'gen_pop1.png'))
                                      # Return a list containing the filename and alt text
                                      list(src = filename, width = 1200, height = 900)}, 
                                      deleteFile = FALSE)
  
#  output$ap_bplot <- renderImage({
#    filename <- normalizePath(file.path('./plots/LarvaAdult', 'apop_year.png'))
#    
#    # Return a list containing the filename and alt text
#    list(src = filename, width = 1200, height = 900)
#    
#  }, deleteFile = FALSE)
#  
#  output$lp_bplot <- renderImage({
#    filename <- normalizePath(file.path('./plots/LarvaAdult', 'lpop_year.png'))
#    
#    # Return a list containing the filename and alt text
#    list(src = filename, width = 1200, height = 900)
#    
#  }, deleteFile = FALSE)
  
  output$e_vplot <- renderImage({filename <- normalizePath(file.path('./plots/', 
                                                                     'adult_emergence_rcp85.png'))
                                 # Return a list containing the filename and alt text
                                 list(src = filename, width = 600, height = 450)}, 
                                 deleteFile = FALSE)
  output$e_vplot_rcp45 <- renderImage({filename <- normalizePath(file.path('./plots/', 
                                                                           'adult_emergence_rcp45.png'))
                                       # Return a list containing the filename and alt text
                                       list(src = filename, width = 600, height = 450)}, 
                                       deleteFile = FALSE)
  
  output$e_bplot <- renderImage({filename <- normalizePath(file.path('./plots', 'edoy_year.png'))
                                 # Return a list containing the filename and alt text
                                 list(src = filename, width = 1200, height = 900)}, 
                                 deleteFile = FALSE)
  
  output$d_vplot <- renderImage({filename <- normalizePath(file.path('./plots', 'diapPop.png'))
                                 # Return a list containing the filename and alt text
                                 list(src = filename, width = 900, height = 700)}, 
                                 deleteFile = FALSE)
  
  output$d_bplot <- renderImage({filename <- normalizePath(file.path('./plots', 'dpop_year.png'))
                                 # Return a list containing the filename and alt text
                                 list(src = filename, width = 1200, height = 900)}, 
                                 deleteFile = FALSE)
 
  ########### 
  output$cumDD_mongrps_magdiff <- renderImage({filename <- normalizePath(file.path('./plots/DegreeDays', 
                                                                      'cumDD_month_groups_magdiff.png'))
                                               # Return a list containing the filename and alt text
                                               list(src = filename, width = 1200, height = 900)}, 
                                               deleteFile = FALSE)
  
  output$cumDD_mons_magdiff <- renderImage({filename <- normalizePath(file.path('./plots/DegreeDays', 
                                                                   'cumDD_months_magdiff.png'))
                                            # Return a list containing the filename and alt text
                                            list(src = filename, width = 1200, height = 1200)}, 
                                            deleteFile = FALSE)

  output$DD_mongrps_magdiff1 <- renderImage({filename <- normalizePath(file.path('./plots/DegreeDays',
                                                                    'DD_month_groups_magdiff1.png'))
                                             # Return a list containing the filename and alt text
                                             list(src = filename, width = 1200, height = 900)}, 
                                             deleteFile = FALSE)

  output$DD_mongrps <- renderImage({filename <- normalizePath(file.path('./plots/DegreeDays', 
                                                                        'DD_month_groups.png'))
                                    # Return a list containing the filename and alt text
                                    list(src = filename, width = 1200, height = 900)}, 
                                    deleteFile = FALSE)

  output$cumDD_mongrps_magdiff_rcp45 <- renderImage({filename <- normalizePath(file.path('./plots/DegreeDays', 
                                                                    'cumDD_month_groups_magdiff_rcp45.png'))
                                                     # Return a list containing the filename and alt text
                                                     list(src = filename, width = 1200, height = 900)}, 
                                                     deleteFile = FALSE)
  
  output$cumDD_mons_magdiff_rcp45 <- renderImage({filename <- normalizePath(file.path('./plots/DegreeDays', 
                                                                             'cumDD_months_magdiff_rcp45.png'))
                                                  # Return a list containing the filename and alt text
                                                  list(src = filename, width = 1200, height = 1200)}, 
                                                  deleteFile = FALSE)

  output$DD_mongrps_magdiff_rcp45 <- renderImage({filename <- normalizePath(file.path('./plots/DegreeDays', 
                                                                        'DD_month_groups_magdiff_rcp45.png'))
                                                  # Return a list containing the filename and alt text
                                                  list(src = filename, width = 1200, height = 900)}, 
                                                  deleteFile = FALSE)

  output$DD_mongrps_rcp45 <- renderImage({filename <- normalizePath(file.path('./plots/DegreeDays', 
                                                                              'DD_month_groups_rcp45.png'))
                                      # Return a list containing the filename and alt text
                                      list(src = filename, width = 1200, height = 900)}, 
                                      deleteFile = FALSE)
  ###########

  # output$rel_pop_cumdd <- renderImage({filename <- normalizePath(file.path('./plots/Diapause', 
  #                                                                          'rel_pop_cumdd.png'))
  #                                      # Return a list containing the filename and alt text
  #                                      list(src = filename, width = 800, height = 600)}, 
  #                                      deleteFile = FALSE)

  # output$rel_pop_doy <- renderImage({filename <- normalizePath(file.path('./plots/Diapause', 
  #                                                                       'rel_pop_doy.png'))
  #                                    # Return a list containing the filename and alt text
  #                                     list(src = filename, width = 800, height = 900)}, 
  #                                     deleteFile = FALSE)

  output$abs_pop_cumdd <- renderImage({
    filename <- normalizePath(file.path('./plots/', 'diapause_abs_rcp85.png'))
    
    # Return a list containing the filename and alt text
    list(src = filename, width = 800, height = 600)
    
  }, deleteFile = FALSE)

  output$abs_pop_doy <- renderImage({
    filename <- normalizePath(file.path('./plots/Diapause', 'abs_pop_doy.png'))
    
    # Return a list containing the filename and alt text
    list(src = filename, width = 800, height = 900)
    
  }, deleteFile = FALSE)

  # output$rel_pop_cumdd_rcp45 <- renderImage({ # hossein commented out
  #   filename <- normalizePath(file.path('./plots/Diapause', 'rel_pop_cumdd_rcp45.png'))
    
  #   # Return a list containing the filename and alt text
  #   list(src = filename, width = 800, height = 600)
    
  # }, deleteFile = FALSE)

  output$rel_pop_doy_rcp45 <- renderImage({
    filename <- normalizePath(file.path('./plots/Diapause', 'rel_pop_doy_rcp45.png'))
    
    # Return a list containing the filename and alt text
    list(src = filename, width = 1200, height = 900)
    
  }, deleteFile = FALSE)

  output$abs_pop_cumdd_rcp45 <- renderImage({
    filename <- normalizePath(file.path('./plots/', 'diapause_abs_rcp45.png'))
    
    # Return a list containing the filename and alt text
    list(src = filename, width = 800, height = 600)
    
  }, deleteFile = FALSE)

  # output$abs_pop_doy_rcp45 <- renderImage({ # Hossein commented out
  #   filename <- normalizePath(file.path('./plots/Diapause', 'abs_pop_doy_rcp45.png'))
    
  #   # Return a list containing the filename and alt text
  #   list(src = filename, width = 1200, height = 900)
    
  # }, deleteFile = FALSE)
  ############
  output$full_bloom <- renderImage({
    #filename <- normalizePath(file.path('./plots', 'FullBloom.png'))
    filename <- normalizePath(file.path('./plots', 'bloom_rcp85.png'))
    
    # Return a list containing the filename and alt text
    list(src = filename, width = 600, height = 500)
    
  }, deleteFile = FALSE)

  output$cumdd <- renderImage({
    filename <- normalizePath(file.path('./plots', 'cumdd_rcp85.png'))
    
    # Return a list containing the filename and alt text
    list(src = filename, width = 600, height = 450)
    
  }, deleteFile = FALSE)

  output$cum_larva_pop <- renderImage({
    filename <- normalizePath(file.path('./plots', 'eggHatch_rcp85.png'))
    
    # Return a list containing the filename and alt text
    list(src = filename, width = 600, height = 450)
    
  }, deleteFile = FALSE)

  output$full_bloom_rcp45 <- renderImage({
    #filename <- normalizePath(file.path('./plots', 'FullBloom_rcp45.png'))
    filename <- normalizePath(file.path('./plots', 'bloom_rcp45.png'))
    
    # Return a list containing the filename and alt text
    list(src = filename, width = 600, height = 500)
    
  }, deleteFile = FALSE)

  output$cumdd_rcp45 <- renderImage({
    filename <- normalizePath(file.path('./plots', 'cumdd_rcp45.png'))
    # Return a list containing the filename and alt text
    list(src = filename, width = 600, height = 450)
    
  }, deleteFile = FALSE)

  output$cum_larva_pop_rcp45 <- renderImage({
         filename <- normalizePath(file.path('./plots', 'eggHatch_rcp45.png'))
         # Return a list containing the filename and alt text
         list(src = filename, width = 600, height = 450)
         }, 
        deleteFile = FALSE)
  #############

  #output$county_groups <- renderImage({
  #  filename <- normalizePath(file.path('./plots', 'county_groups.png'))
    
    # Return a list containing the filename and alt text
  #  list(src = filename, width = 600, height = 400)
    
  #}, deleteFile = FALSE)

  output$map_adult_med_doy <- renderLeaflet({
    type = "A"
    genPct = paste0(type, input$adult_gen, "_", input$adult_percent)
    #domainVal = ifelse(input$type == "A", seq(105, 180), seq(125,200));
    #print(genPct)
    domainVal = seq(85, 365)
    if(input$cg_adult_med_doy == "Historical") {
      climate_group = input$cg_adult_med_doy
      future_version = "rcp85"
    }
    else {
      temp = tstrsplit(input$cg_adult_med_doy, "_")
      climate_group = unlist(temp[1])
      future_version = unlist(temp[2])
    }
    #print(input$fver_med_doy)
    if(future_version == "rcp45") {
      data = d_rcp45
      #names(data)[names(data) == "ClimateGroup"] = "timeFrame"
    }
    else {
      data = d
    }
    layerlist = levels(data$timeFrame) # c("Historical", "2040's", "2060's", "2080's")
    
    sub_Gen = subset(data, !is.na(timeFrame) & 
                     !is.na(get(genPct)) & 
                     timeFrame == climate_group, 
                     select = c(timeFrame, year, location, get(genPct)))
    sub_Gen = sub_Gen[, .(medianDoY = as.integer(median( get(genPct) ))), 
                      by = c("timeFrame", "location")]
    
    medianGen = list( hist = subset(sub_Gen, timeFrame == layerlist[1]),
                             `2040` = subset(sub_Gen, timeFrame == layerlist[2]),
                             `2060` = subset(sub_Gen, timeFrame == layerlist[3]),
                             `2080` = subset(sub_Gen, timeFrame == layerlist[4]))
    
    # GenMap <- constructMap(medianGen, layerlist, palColumn = "medianDoY", 
    #                        legendVals = sub_Gen$medianDoY, "Median Day of Year")
    GenMap <- constructMap(medianGen, layerlist, 
                           palColumn = "medianDoY", 
                           legendVals = domainVal, "Median Day of Year")
    GenMap
  })

  output$map_larvae_med_doy <- renderLeaflet({
    type = "L"
    genPct = paste0(type, input$larvae_gen, "_", input$larvae_percent)
    #domainVal = ifelse(input$type == "A", seq(105, 180), seq(125,200));
    #print(genPct)
    domainVal = seq(110, 345)
    if(input$cg_larvae_med_doy == "Historical") {
      climate_group = input$cg_larvae_med_doy
      future_version = "rcp85"
     }
     else {
      temp = tstrsplit(input$cg_larvae_med_doy, "_")
      climate_group = unlist(temp[1])
      future_version = unlist(temp[2])
    }
    #print(input$fver_med_doy)
    if(future_version == "rcp45") {
      data = d_rcp45
      #names(data)[names(data) == "ClimateGroup"] = "timeFrame"
     }
     else {
      data = d
    }
    layerlist = levels(data$timeFrame) #c("Historical", "2040's", "2060's", "2080's")
    
    sub_Gen = subset(data, 
                     !is.na(timeFrame) & 
                     !is.na(get(genPct)) & 
                     timeFrame == climate_group, 
                     select = c(timeFrame, year, location, 
                                get(genPct)))
    sub_Gen = sub_Gen[, .(medianDoY = as.integer(median( get(genPct) ))), 
                      by = c("timeFrame", "location")]
    
    medianGen = list( hist = subset(sub_Gen, timeFrame == layerlist[1]),
                             `2040` = subset(sub_Gen, timeFrame == layerlist[2]),
                             `2060` = subset(sub_Gen, timeFrame == layerlist[3]),
                             `2080` = subset(sub_Gen, timeFrame == layerlist[4]))
    
    # GenMap <- constructMap(medianGen, 
    #                        layerlist, 
    #                        palColumn = "medianDoY", 
    #                        legendVals = sub_Gen$medianDoY, 
    #                        "Median Day of Year")

    GenMap <- constructMap(medianGen, layerlist, 
                           palColumn = "medianDoY", 
                           legendVals = domainVal, "Median Day of Year")
    GenMap
  })
  
  output$map_adult_diff_doy <- renderLeaflet({
    #diffType = as.integer(input$diff_type)
    type_diff = "A"
    diffType = 3
    genPct = paste0(type_diff, input$adult_gen_diff, "_", input$adult_percent_diff)
    #print(c(diffType, genPct))

    temp = tstrsplit(input$cg_adult_doy_diff, "_")
    climate_group = unlist(temp[1])
    future_version = unlist(temp[2])

    if(future_version == 'rcp45') {
      data = d_rcp45
      #names(data)[names(data) == "ClimateGroup"] = "timeFrame"
     }
     else {
      data = d
    }
    layerdiff = c("2040's - Historical", "2060's - Historical", "2080's - Historical")
    layerlist = levels(data$timeFrame) #c("Historical", "2040's", "2060's", "2080's")
    
    sub_Gen = subset(data, 
                     !is.na(timeFrame) & 
                     !is.na(get(genPct)), 
                     select = c(timeFrame, year, location, 
                                get(genPct)))
    sub_Gen = sub_Gen[, 
                      .(value = as.integer(quantile( get(genPct), names = FALSE )[diffType])), 
                      by = c("timeFrame", "location")]
    
    tfGen = list(subset(sub_Gen, timeFrame == layerlist[1]),
                    subset(sub_Gen, timeFrame == layerlist[2]),
                    subset(sub_Gen, timeFrame == layerlist[3]),
                    subset(sub_Gen, timeFrame == layerlist[4]))
    
    # diffGen = list(merge(tfGen[[2]], tfGen[[1]], by = c("location")),
    #                merge(tfGen[[3]], tfGen[[1]], by = c("location")),
    #                merge(tfGen[[4]], tfGen[[1]], by = c("location")))
        
    # for(i in 1:length(diffGen)) {
    #  diffGen[[i]]$diff = diffGen[[i]]$value.y - diffGen[[i]]$value.x
    # }
    # diffDomain = c(diffGen[[1]]$diff, diffGen[[2]]$diff, diffGen[[3]]$diff)
    if(layerdiff[1] == climate_group) {
      diffGen = list(merge(tfGen[[2]], tfGen[[1]], by = c("location")))
     }
     else if(layerdiff[2] == climate_group) {
      diffGen = list(merge(tfGen[[3]], tfGen[[1]], by = c("location")))
    }
    else if(layerdiff[3] == climate_group) {
      diffGen = list(merge(tfGen[[4]], tfGen[[1]], by = c("location")))
    }
    diffGen[[1]]$diff = diffGen[[1]]$value.y - diffGen[[1]]$value.x
    diffDomain = diffGen[[1]]$diff
    
    # GenDiffMap <- constructMap(diffGen, 
    #                           layerdiff, 
    #                           palColumn = "diff", 
    #                           legendVals = diffDomain, 
    #                           "Median calendar day difference from historical", 
    #                           RdBu_reverse)
    GenDiffMap <- constructMap(diffGen, 
                               layerdiff, palColumn = "diff", 
                               legendVals = seq(0,115), 
                               HTML("Median calendar day<br />difference from historical"), 
                               RdBu_reverse)
    GenDiffMap
  })

  output$map_larvae_diff_doy <- renderLeaflet({
    #diffType = as.integer(input$diff_type)
    type_diff = "L"
    diffType = 3
    genPct = paste0(type_diff, input$larvae_gen_diff, "_", input$larvae_percent_diff)
    #print(c(diffType, genPct))

    temp = tstrsplit(input$cg_larvae_doy_diff, "_")
    climate_group = unlist(temp[1])
    future_version = unlist(temp[2])

    if(future_version == 'rcp45') {
      data = d_rcp45
      #names(data)[names(data) == "ClimateGroup"] = "timeFrame"
     }
      else {
      data = d
    }
    layerdiff = c("2040's - Historical", "2060's - Historical", "2080's - Historical")
    layerlist = levels(data$timeFrame) #c("Historical", "2040's", "2060's", "2080's")
    
    sub_Gen = subset(data, !is.na(timeFrame) & !is.na(get(genPct)), 
                     select = c(timeFrame, year, location, get(genPct)))
    sub_Gen = sub_Gen[, .(value = as.integer(quantile( get(genPct), names = FALSE )[diffType])), 
                      by = c("timeFrame", "location")]
    
    tfGen = list(subset(sub_Gen, timeFrame == layerlist[1]),
                    subset(sub_Gen, timeFrame == layerlist[2]),
                    subset(sub_Gen, timeFrame == layerlist[3]),
                    subset(sub_Gen, timeFrame == layerlist[4]))
    
    #diffGen = list(merge(tfGen[[2]], tfGen[[1]], by = c("location")),
    #                merge(tfGen[[3]], tfGen[[1]], by = c("location")),
    #                merge(tfGen[[4]], tfGen[[1]], by = c("location")))
        
    #for(i in 1:length(diffGen)) {
    #  diffGen[[i]]$diff = diffGen[[i]]$value.y - diffGen[[i]]$value.x
    #}
    #diffDomain = c(diffGen[[1]]$diff, diffGen[[2]]$diff, diffGen[[3]]$diff)
    if(layerdiff[1] == climate_group) {
      diffGen = list(merge(tfGen[[2]], tfGen[[1]], by = c("location")))
    }
    else if(layerdiff[2] == climate_group) {
      diffGen = list(merge(tfGen[[3]], tfGen[[1]], by = c("location")))
    }
    else if(layerdiff[3] == climate_group) {
      diffGen = list(merge(tfGen[[4]], tfGen[[1]], by = c("location")))
    }
    diffGen[[1]]$diff = diffGen[[1]]$value.y - diffGen[[1]]$value.x
    diffDomain = diffGen[[1]]$diff
    
    #GenDiffMap <- constructMap(diffGen, 
    #                           layerdiff, 
    #                           palColumn = "diff",
    #                           legendVals = diffDomain, 
    #                           "Median calendar day difference from historical", 
    #                           RdBu_reverse)
    GenDiffMap <- constructMap(diffGen, 
                               layerdiff, palColumn = "diff", 
                               legendVals = seq(0,70), 
                               HTML("Median calendar day<br />difference from historical"), 
                               RdBu_reverse)
    GenDiffMap
  })
  
#  output$map_med_pop <- renderLeaflet({
#    pop_mon = paste0(input$type_pop, input$pop_month)
#    #print(pop_mon)
#    layerlist = levels(d$timeFrame) #c("Historical", "2040's", "2060's", "2080's")
#    
#    sub_Pop = subset(d, !is.na(timeFrame) & !is.na(get(pop_mon)), 
#                     select = c(timeFrame, year, location, get(pop_mon)))
#    sub_Pop[, (pop_mon) := get(pop_mon) * 100]
#    sub_Pop = sub_Pop[, .(medianPop = median( get(pop_mon) )), by = c("timeFrame", "location")]
#    
#    medianPop = list( hist = subset(sub_Pop, timeFrame == layerlist[1]),
#                      `2040` = subset(sub_Pop, timeFrame == layerlist[2]),
#                      `2060` = subset(sub_Pop, timeFrame == layerlist[3]),
#                      `2080` = subset(sub_Pop, timeFrame == layerlist[4]))
#    
#    PopMap <- constructMap(medianPop, layerlist, 
#                           palColumn = "medianPop", 
#                           legendVals = seq(0, 100), "Population (%)", 
#                           RdBu_reverse)
#
#    PopMap
#  })
  output$map_med_pop <- renderLeaflet({
    typeGen = paste0("Perc", input$type_pop, input$type_pop_gen)
    pop_mon = input$pop_month
    #print(pop_mon)
    
    if(input$fver_pop_med == 'rcp45') {
      data = d1_rcp45
     }
     else {
      data = d1
    }

    layerlist = levels(data$ClimateGroup)
    
    sub_Pop = subset(data, !is.na(ClimateGroup) & 
                     month == pop_mon, 
                     select = c(ClimateGroup, month, 
                                year, location, latitude, 
                                longitude, get(typeGen)))
    
    sub_Pop[, (typeGen) := get(typeGen) * 100]
    sub_Pop = sub_Pop[, .(medianPop = median( get(typeGen) )), 
                      by = c("ClimateGroup", "latitude", "longitude", "location")]
    
    # sub_Pop$location = paste0(sub_Pop$latitude, "_", sub_Pop$longitude)
        
    medianPop = list( hist = subset(sub_Pop, ClimateGroup == layerlist[1]),
                      `2040` = subset(sub_Pop, ClimateGroup == layerlist[2]),
                      `2060` = subset(sub_Pop, ClimateGroup == layerlist[3]),
                      `2080` = subset(sub_Pop, ClimateGroup == layerlist[4]))
    
    PopMap <- constructMap(medianPop, layerlist, 
                           palColumn = "medianPop", 
                           legendVals = seq(0, 100), 
                           "Population (%)", RdBu_reverse)
    PopMap
  })
  
  output$map_diff_pop <- renderLeaflet({
    diffType = as.integer(input$pop_diff_type)
    typeGen = paste0("Perc", input$type_pop_diff, input$pop_diff_type_gen)
    pop_mon = input$pop_diff_month
    #print(c(diffType, pop_mon))

    if(input$fver_pop_diff == 'rcp45') {
      data = d1_rcp45
    }
    else {
      data = d1
    }
    
    layerdiff = c("2040's - Historical", "2060's - Historical", "2080's - Historical")
    layerlist = levels(data$ClimateGroup) #c("Historical", "2040's", "2060's", "2080's")
    
    sub_Pop = subset(data, !is.na(ClimateGroup) & month == pop_mon, 
                     select = c(ClimateGroup, month, year, 
                                location, latitude, longitude, 
                                get(typeGen)))

    sub_Pop[, (typeGen) := get(typeGen) * 100]
    sub_Pop = sub_Pop[, .(value = quantile( get(typeGen), names = FALSE )[diffType]), 
                      by = c("ClimateGroup", "latitude", "longitude", "location")]
    
    # sub_Pop$location = paste0(sub_Pop$latitude, "_", sub_Pop$longitude)

    tfPop = list(subset(sub_Pop, ClimateGroup == layerlist[1]),
                 subset(sub_Pop, ClimateGroup == layerlist[2]),
                 subset(sub_Pop, ClimateGroup == layerlist[3]),
                 subset(sub_Pop, ClimateGroup == layerlist[4]))
    
    diffPop = list(merge(tfPop[[2]], tfPop[[1]], by = c("location")),
                   merge(tfPop[[3]], tfPop[[1]], by = c("location")),
                   merge(tfPop[[4]], tfPop[[1]], by = c("location")))
    
    for(i in 1:length(diffPop)) {
      diffPop[[i]]$diff = diffPop[[i]]$value.x - diffPop[[i]]$value.y
    }
    diffDomain = c(diffPop[[1]]$diff, diffPop[[2]]$diff, diffPop[[3]]$diff)
    
    PopDiffMap <- constructMap(diffPop, 
                               layerdiff, palColumn = "diff", 
                               legendVals = diffDomain, 
                               "Population(%) Difference from Historical", 
                               RdBu_reverse)
    PopDiffMap
  })
  
  output$map_risk <- renderLeaflet({
    #input$type_risk = "L"
    type_risk = "L"
    genPct = paste0(type_risk, input$gen_risk, "_", input$percent_risk)
    #print(genPct)
    if(input$cg_risk == "Historical") {
      climate_group = input$cg_risk
      future_version = "rcp85"
      }
     else {
      temp = tstrsplit(input$cg_risk, "_")
      climate_group = unlist(temp[1])
      future_version = unlist(temp[2])
    }

    if(future_version == "rcp45") {
      data = d_rcp45
      }
      else {
     data = d
    }

    layerlist = levels(data$timeFrame)
    
    freq_data = subset(data, !is.na(timeFrame) & 
                       timeFrame == climate_group, 
                       select = c(timeFrame, year, location, 
                                  get(genPct)))
    freq_data_melted = melt(freq_data, id.vars = c("timeFrame", "location", "year"), na.rm = FALSE)
    f1 = freq_data_melted[, .(years_range = (max(year) - min(year) + 1)), 
                          by = list(timeFrame, location)][order(timeFrame, location)]
    
    f2 = freq_data_melted[complete.cases(freq_data_melted$value), 
                         .(years_freq = uniqueN(year)), 
                         by = list(timeFrame, location)][order(timeFrame, location)]

    # left join - merge both tables
    f = merge(f1, f2, by = c("timeFrame", "location"), all.x = TRUE)
    # replace na values by 0
    f[is.na(years_freq), years_freq := 0]
    # not working # set(f, i = which(is.na(f$years_freq)), j = years_freq, value = 0)
    f$percentage = (f$years_freq / f$years_range) * 100
    
    riskGen = list( hist = subset(f, timeFrame == layerlist[1]),
                      `2040` = subset(f, timeFrame == layerlist[2]),
                      `2060` = subset(f, timeFrame == layerlist[3]),
                      `2080` = subset(f, timeFrame == layerlist[4]))
    
    GenMap <- constructMap(riskGen, 
                           layerlist, 
                           palColumn = "percentage", 
                           legendVals = seq(0, 100), 
                           HTML("Percentage(%) of<br />Years Occurred"), 
                           RdBu_reverse)
    GenMap
  })
  
  output$map_risk1 <- renderLeaflet({
    typeGen = paste0("Perc", input$type_risk1, input$gen_risk1)
    #print(typeGen)
    if(input$fver_risk == "rcp45") {
     data = d1_rcp45
      }
     else {
      data = d1
    }
     
    climate_scenario = input$clim_scen
    freq_data = subset(data, !is.na(ClimateGroup) & 
                       ClimateScenario == climate_scenario, 
                       select = c(ClimateGroup, ClimateScenario, 
                                  latitude, longitude, County, 
                                  year, month, 
                                  get(typeGen)))
    layerlist = unique(as.character(freq_data$ClimateGroup))
    
    freq_data_melted = melt(freq_data, 
                            id.vars = c("ClimateGroup", "ClimateScenario", 
                                        "latitude", "longitude", "County", 
                                        "year", "month"), 
                            na.rm = FALSE)
    
    f1 = freq_data_melted[, .(years_range = (max(year) - min(year) + 1)), 
                          by = list(ClimateGroup, ClimateScenario, 
                                    latitude, longitude)][order(ClimateGroup, ClimateScenario, 
                                                                latitude, longitude)]
    
    f2 = freq_data_melted[ freq_data_melted$month == 'October' & 
                           freq_data_melted$value >= input$percent_risk1, 
                           .(years_freq = uniqueN(year)), 
                           by = list(ClimateGroup, ClimateScenario, 
                                     latitude, longitude)][order(ClimateGroup, ClimateScenario, 
                                                                 latitude, longitude)]
    
    # left join - merge both tables
    f = merge(f1, f2, by = c("ClimateGroup", "ClimateScenario", 
                             "latitude", "longitude"), all.x = TRUE)
    # replace na values by 0
    f[is.na(years_freq), years_freq := 0]
    f$percentage = (f$years_freq / f$years_range) * 100
    f[, percentage := mean(percentage), 
        by = c("ClimateGroup", "latitude", "longitude")]
    f = unique(f)
    f$location = paste0(f$latitude, "_", f$longitude)
    
    riskGen = list( hist = subset(f, ClimateGroup == layerlist[1]),
                    `2040` = subset(f, ClimateGroup == layerlist[2]),
                    `2060` = subset(f, ClimateGroup == layerlist[3]),
                    `2080` = subset(f, ClimateGroup == layerlist[4]))
    
    GenMap1 <- constructMap(riskGen, 
                            layerlist, palColumn = "percentage", 
                            legendVals = seq(0,100), 
                            "Percentage(%) of Years Occurred", 
                            RdBu_reverse)
    GenMap1
  })

  output$map_emerg_doy <- renderLeaflet({
    col = "Emergence"

    if(input$cg_emerg_doy == "Historical") {
      climate_group = input$cg_emerg_doy
      future_version = "rcp85"
     }
     else {
      temp = tstrsplit(input$cg_emerg_doy, "_")
      climate_group = unlist(temp[1])
      future_version = unlist(temp[2])
    }

    if(future_version == "rcp45") {
      data = d_rcp45
       }
        else {
      data = d
    }

    layerlist = levels(data$timeFrame) #c("Historical", "2040's", "2060's", "2080's")
    
    sub_Emerg = subset(data, !is.na(timeFrame) & 
                       !is.na(get(col)) & timeFrame == climate_group, 
                       select = c(timeFrame, year, location, 
                                  get(col)))
    sub_Emerg = sub_Emerg[, .(medianDoY = as.integer(median( get(col) ))), 
                          by = c("timeFrame", "location")]
    
    medianEmerg =list( hist = subset(sub_Emerg, timeFrame == layerlist[1]),
                      `2040` = subset(sub_Emerg, timeFrame == layerlist[2]),
                      `2060` = subset(sub_Emerg, timeFrame == layerlist[3]),
                      `2080` = subset(sub_Emerg, timeFrame == layerlist[4]))
    
    EmergMap <- constructMap(medianEmerg, layerlist, 
                             palColumn = "medianDoY", 
                             legendVals = seq(65,145), 
                             "Median Day of Year")
    EmergMap
  })
  
  output$map_diff_emerg <- renderLeaflet({
    #diffType = as.integer(input$emerg_diff_type)
    diffType = 3
    col = "Emergence"
    
    temp = tstrsplit(input$cg_diff_emerg, "_")
    climate_group = unlist(temp[1])
    future_version = unlist(temp[2])

    if(future_version == "rcp45") {
      data = d_rcp45
     }
     else {
      data = d
    }

    layerdiff = c("2040's - Historical", "2060's - Historical", "2080's - Historical")
    layerlist = levels(data$timeFrame) #c("Historical", "2040's", "2060's", "2080's")
    
    sub_Emerg = subset(data, !is.na(timeFrame) & 
                       !is.na(get(col)), 
                       select = c(timeFrame, year, location, get(col)))

    sub_Emerg = sub_Emerg[, .(value = as.integer(quantile( get(col), names = FALSE )[diffType])), 
                          by = c("timeFrame", "location")]
    
    tfEmerg = list(subset(sub_Emerg, timeFrame == layerlist[1]),
                  subset(sub_Emerg, timeFrame == layerlist[2]),
                  subset(sub_Emerg, timeFrame == layerlist[3]),
                  subset(sub_Emerg, timeFrame == layerlist[4]))
    
    # diffEmerg = list(merge(tfEmerg[[2]], tfEmerg[[1]], by = c("location")),
    #                merge(tfEmerg[[3]], tfEmerg[[1]], by = c("location")),
    #                merge(tfEmerg[[4]], tfEmerg[[1]], by = c("location")))
    
    
    #for(i in 1:length(diffEmerg)) {
    #  diffEmerg[[i]]$diff = diffEmerg[[i]]$value.y - diffEmerg[[i]]$value.x
    #}
    # diffDomain = c(diffEmerg[[1]]$diff, diffEmerg[[2]]$diff, diffEmerg[[3]]$diff)
    if(layerdiff[1] == climate_group) {
      diffEmerg = list(merge(tfEmerg[[2]], tfEmerg[[1]], by = c("location")))
     }
     else if(layerdiff[2] == climate_group) {
      diffEmerg = list(merge(tfEmerg[[3]], tfEmerg[[1]], by = c("location")))
     }
     else if(layerdiff[3] == climate_group) {
      diffEmerg = list(merge(tfEmerg[[4]], tfEmerg[[1]], by = c("location")))
    }
    diffEmerg[[1]]$diff = diffEmerg[[1]]$value.y - diffEmerg[[1]]$value.x
    diffDomain = diffEmerg[[1]]$diff

    EmergDiffMap <- constructMap(diffEmerg, 
                                 layerdiff, 
                                 palColumn = "diff", 
                                 legendVals = seq(0,45),
                                 HTML("Median calendar day<br />difference from historical"), 
                                 RdBu_reverse)
    # EmergDiffMap <- constructMap(diffEmerg, layerdiff, 
    #                             palColumn = "diff", 
    #                             legendVals = diffDomain, 
    #                             "Median calendar day difference from historical", 
    #                             RdBu_reverse)
    EmergDiffMap
  })
  
  output$map_diap_pop <- renderLeaflet({
    #$input$diap_pop = "RelPct"
    col = paste0("RelPct", input$diapaused, if(input$diap_gen == "all") "" else input$diap_gen)

    if(input$cg_diap == "Historical") {
      climate_group = input$cg_diap
      future_version = "rcp85"
     }
     else {
      temp = tstrsplit(input$cg_diap, "_")
      climate_group = unlist(temp[1])
      future_version = unlist(temp[2])
    }

    if(future_version == "rcp45") {
      diap_d = diap_rcp45
     }
     else {
      diap_d = diap
    }

    layerlist = levels(diap_d$ClimateGroup) #c("Historical", "2040's", "2060's", "2080's")

    sub_Diap = subset(diap_d, ClimateGroup == climate_group, 
                      select = c(ClimateGroup, CountyGroup, 
                                 latitude, longitude, get(col)))
    sub_Diap$location = paste0(sub_Diap$latitude, "_", sub_Diap$longitude)
    
    meanDiap = list( hist = subset(sub_Diap, ClimateGroup == layerlist[1]),
                     `2040` = subset(sub_Diap, ClimateGroup == layerlist[2]),
                     `2060` = subset(sub_Diap, ClimateGroup == layerlist[3]),
                     `2080` = subset(sub_Diap, ClimateGroup == layerlist[4]))
    
    DiapMap <- constructMap(meanDiap, 
                            layerlist, 
                            palColumn = col, 
                            legendVals = seq(0, 100), 
                            "Percentage (%)", 
                            RdBu_reverse)
    DiapMap
  })

  output$map_diff_diap <- renderLeaflet({
    diffType = as.integer(input$diap_diff_type)
    col = "Diapause"
    
    layerdiff = c("2040's - Historical", "2060's - Historical", "2080's - Historical")
    layerlist = levels(d$timeFrame) 
    
    sub_Diap = subset(d, !is.na(timeFrame) & !is.na(get(col)), 
                      select = c(timeFrame, year, location, get(col)))
    sub_Diap[, (col) := get(col) * 100]
    sub_Diap = sub_Diap[, .(value = quantile( get(col), names = FALSE )[diffType]), 
                        by = c("timeFrame", "location")]
    
    tfDiap = list(subset(sub_Diap, timeFrame == layerlist[1]),
                  subset(sub_Diap, timeFrame == layerlist[2]),
                  subset(sub_Diap, timeFrame == layerlist[3]),
                  subset(sub_Diap, timeFrame == layerlist[4]))
    
    diffDiap = list(merge(tfDiap[[2]], tfDiap[[1]], by = c("location")),
                    merge(tfDiap[[3]], tfDiap[[1]], by = c("location")),
                    merge(tfDiap[[4]], tfDiap[[1]], by = c("location")))
    
    
    for(i in 1:length(diffDiap)) {
      diffDiap[[i]]$diff = diffDiap[[i]]$value.y - diffDiap[[i]]$value.x
    }
    diffDomain = c(diffDiap[[1]]$diff, diffDiap[[2]]$diff, diffDiap[[3]]$diff)
    
    DiapDiffMap <- constructMap(diffDiap, 
                                layerdiff, 
                                palColumn = "diff", 
                                legendVals = diffDomain, 
                                "Population(%) Difference from Historical", 
                                RdBu_reverse)
    DiapDiffMap
  })


  output$map_bloom_doy <- renderLeaflet({
    layerlist = levels(diap$ClimateGroup) # c("Historical", "2040's", "2060's", "2080's")

    if(input$cg_bloom == "Historical") {
      climate_group = input$cg_bloom
      future_version = "rcp85"
     }
     else {
      temp = tstrsplit(input$cg_bloom, "_")
      climate_group = unlist(temp[1])
      future_version = unlist(temp[2])
    }

    if(future_version == "rcp45") {
     bloom_d = bloom_rcp45
     }
     else {
     bloom_d = bloom
    }

    sub_bloom = subset(bloom_d, apple_type == input$apple_type & ClimateGroup == climate_group)
    sub_bloom$location = paste0(sub_bloom$latitude, "_", sub_bloom$longitude)
    
    medBloom = list(hist = subset(sub_bloom, ClimateGroup == layerlist[1]),
                    `2040` = subset(sub_bloom, ClimateGroup == layerlist[2]),
                    `2060` = subset(sub_bloom, ClimateGroup == layerlist[3]),
                    `2080` = subset(sub_bloom, ClimateGroup == layerlist[4]))
    
    BloomMap <- constructMap(medBloom, layerlist, palColumn = "medDoY", 
                             legendVals = seq(85, 165), "Median Day of Year")
    BloomMap
  })
  ##########################################################
  output$map_bloom_doy_100 <- renderLeaflet({
    layerlist = levels(diap$ClimateGroup) # c("Historical", "2040's", "2060's", "2080's")

    if(input$cg_bloom_100 == "Historical") {
      climate_group = input$cg_bloom_100
      future_version = "rcp85"
     } else {
      temp = tstrsplit(input$cg_bloom_100, "_")
      climate_group = unlist(temp[1])
      future_version = unlist(temp[2])
    }

    if(future_version == "rcp45") {
     bloom_d = bloom_rcp45_100
      } else { 
     bloom_d = bloom_rcp85_100
    }

    sub_bloom = subset(bloom_d, apple_type == input$apple_type & 
                       ClimateGroup == climate_group)
    sub_bloom$location = paste0(sub_bloom$latitude, "_", sub_bloom$longitude)
    
    medBloom = list( hist = subset(sub_bloom, ClimateGroup == layerlist[1]),
                     `2040` = subset(sub_bloom, ClimateGroup == layerlist[2]),
                     `2060` = subset(sub_bloom, ClimateGroup == layerlist[3]),
                     `2080` = subset(sub_bloom, ClimateGroup == layerlist[4]))
    
    BloomMap <- constructMap(medBloom, layerlist, 
                             palColumn = "medDoY", 
                             legendVals = seq(85,165), 
                             "Median Day of Year")
    BloomMap
  })
  ##########################################################
  output$map_bloom_doy_95 <- renderLeaflet({
  layerlist = levels(diap$ClimateGroup)

  if(input$cg_bloom_95 == "Historical") {
    climate_group = input$cg_bloom_95
    future_version = "rcp85"
    } else {
    temp = tstrsplit(input$cg_bloom_95, "_")
    climate_group = unlist(temp[1])
    future_version = unlist(temp[2])
  }

  if(future_version == "rcp45") {
    bloom_d = bloom_rcp45_95
    } else { 
    bloom_d = bloom_rcp85_95
  }

  sub_bloom = subset(bloom_d, apple_type == input$apple_type & 
                     ClimateGroup == climate_group)
  sub_bloom$location = paste0(sub_bloom$latitude, "_", sub_bloom$longitude)
  
  medBloom = list( hist = subset(sub_bloom, ClimateGroup == layerlist[1]),
                   `2040` = subset(sub_bloom, ClimateGroup == layerlist[2]),
                   `2060` = subset(sub_bloom, ClimateGroup == layerlist[3]),
                   `2080` = subset(sub_bloom, ClimateGroup == layerlist[4]))
  
  BloomMap <- constructMap(medBloom, 
                          layerlist, 
                          palColumn = "medDoY", 
                          legendVals = seq(85,165), 
                          "Median Day of Year")
  BloomMap
  })

  output$map_bloom_doy_50 <- renderLeaflet({
    layerlist = levels(diap$ClimateGroup) # c("Historical", "2040's", "2060's", "2080's")

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

    sub_bloom = subset(bloom_d, apple_type == input$apple_type & ClimateGroup == climate_group)
    sub_bloom$location = paste0(sub_bloom$latitude, "_", sub_bloom$longitude)
    
    medBloom = list( hist = subset(sub_bloom, ClimateGroup == layerlist[1]),
                     `2040` = subset(sub_bloom, ClimateGroup == layerlist[2]),
                     `2060` = subset(sub_bloom, ClimateGroup == layerlist[3]),
                     `2080` = subset(sub_bloom, ClimateGroup == layerlist[4]))
    
    BloomMap <- constructMap(medBloom, layerlist, 
                             palColumn = "medDoY", 
                             legendVals = seq(85,165), 
                             "Median Day of Year")
    BloomMap
  })

  #######################################################
  #######################################################
  #######################################################
  output$map_bloom_diff <- renderLeaflet({
    #diffType = as.integer(input$emerg_diff_type)
    
    temp = tstrsplit(input$cg_bloom_diff, "_")
    climate_group = unlist(temp[1])
    future_version = unlist(temp[2])

    if(future_version == "rcp45") {
  data = bloom_rcp45
    }
    else {
  data = bloom
    }

    layerdiff = c("2040's - Historical", "2060's - Historical", "2080's - Historical")
    layerlist = levels(data$ClimateGroup) #c("Historical", "2040's", "2060's", "2080's")
    data$location = paste0(data$latitude, "_", data$longitude)
    
    sub_Bloom = subset(data, !is.na(ClimateGroup) & 
                       apple_type == input$apple_type_diff, 
                       select = c(ClimateGroup, location, medDoY))
    
    cgBloom = list(subset(sub_Bloom, ClimateGroup == layerlist[1]),
                  subset(sub_Bloom, ClimateGroup == layerlist[2]),
                  subset(sub_Bloom, ClimateGroup == layerlist[3]),
                  subset(sub_Bloom, ClimateGroup == layerlist[4]))
    
    #diffEmerg = list(merge(tfEmerg[[2]], tfEmerg[[1]], by = c("location")),
    #                merge(tfEmerg[[3]], tfEmerg[[1]], by = c("location")),
    #                merge(tfEmerg[[4]], tfEmerg[[1]], by = c("location")))
    
    
    #for(i in 1:length(diffEmerg)) {
    #  diffEmerg[[i]]$diff = diffEmerg[[i]]$value.y - diffEmerg[[i]]$value.x
    #}
    #diffDomain = c(diffEmerg[[1]]$diff, diffEmerg[[2]]$diff, diffEmerg[[3]]$diff)
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
                                 legendVals = seq(5,45), 
                                 HTML("Median calendar day <br />difference from historical"), 
                                 RdBu_reverse)
    BloomDiffMap
  })
  
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
