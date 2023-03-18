######################################################
######################################################
#
#                   UI part
#
######################################################
######################################################
#
# Dry
#
##************************************************ Done
menuItem("Bloom and Chill Portions", 
         tabName = "bcf_map", 
         icon = icon("tint"))

menuItem("Dry Days", tabName = "dry_map", icon = icon("tint"))

##************************************************ Done
tabItem(tabName = "bcf_map",
            box(id = "bcf", width = NULL,
                #  Main Map  #
                tabPanel("Map",
                         div(class = "outer",
                             tags$style(type = "text/css", 
                                        "#bcf_map {height: calc(100vh - 125px) !important;}"),
                             leafletOutput("bcf_map")
                            )
                        )
                )
            )

tabItem(tabName = "dry_map",
            box(id = "dry_days",
                width = NULL,
                #  Main Map  #
                tabPanel("Map",
                         div(class = "outer",
                             tags$style(type = "text/css", "#dry_map {height: calc(100vh - 125px) !important;}"),
                             leafletOutput("dry_map"),
                             absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                                           draggable = TRUE, top = 90, left = "auto", right = 40, bottom = "auto",
                                           width = 250, height = "auto",
                                           h3(tags$b("Explorer")),
                                           radioButtons("dry_days_map_climate_proj", 
                                                        label = "Climate Projection", 
                                                        choices = list("RCP 4.5" = "rcp45", 
                                                                       "RCP 8.5" = "rcp85")),
                                           radioButtons("dry_days_map_exceedance", 
                                                        label = "Exceedance Value", 
                                                        choices = list("20th Percentile" = "prob_80", 
                                                                       "10th Percentile" = "prob_90",
                                                                       "5th Percentile" = "prob_95")),
                                           radioButtons("dry_days_map_climate_group", 
                                                        label = "Climate Group", 
                                                        choices = list("2040s" = "2040s", 
                                                                       "2060s" = "2060s", 
                                                                       "2080s" = "2080s")))
                            )
                        )
                )
            )

##################################################
bsModal(title="Dry Days Summary", id = "dry_days_graphs", trigger = NULL, size = "large",
          fluidPage(fluidRow(column(2, radioButtons("dry_days_plot_climate_proj", label = h3("Climate Projection"), 
                                                    choices = list("RCP 4.5" = "rcp45", 
                                                                   "RCP 8.5" = "rcp85"),
                                                    selected = "rcp85")),
                             column(10, offset = 0, plotOutput("dry_days_plot", height = 550))
                            )
                   )
          )

bsModal(title = "Bloom vs. CP, and frost", 
          id = "bcf_graphs", 
          trigger = NULL, 
          size = "large",
          fluidPage(fluidRow(column(2, 
                                    radioButtons("bcf_plot_climate_proj", 
                                                 label = h4("Climate Projection"), 
                                                 choices = list("RCP 4.5" = "rcp45", 
                                                                "RCP 8.5" = "rcp85"),
                                                 selected = "rcp85"),

                                    radioButtons("fcp_plot_fruit_type", 
                                                 label = h4("Fruit type"), 
                                                 choices = list("Cripps Pink" = "cripps_pink", 
                                                                "Gala" = "gala", 
                                                                "Red Delicious" = "red_deli"),
                                                 selected = "cripps_pink")),
                             column(10, offset = 0, plotOutput("fcp_plot", height = 550))
                            )
                    )
) # ,
  

######################################################
######################################################
#
#                   server part
#
####################################################################
####################################################################
#
# done 
#
spatial_bcf_data <- reactive({
  spatial_bcf
})

dry_days_map_data <- reactive({
    spatial_dry_days %>% 
    filter(climate_proj == input$dry_days_map_climate_proj,
           group == input$dry_days_map_climate_group, 
           exceedance == input$dry_days_map_exceedance)
})
######################################################################

# done

output$bcf_map <- renderLeaflet({
  pal <- colorBin(palette = "plasma", reverse = TRUE,
                  domain = spatial_bcf()$lat, bins = 8, pretty=TRUE)
  leaflet() %>%
  addTiles(urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
           attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>') %>%
  addPolygons(data = skagit, fill = FALSE, stroke = 1, color = 'black') %>% 
  addPolygons(data = whatcom, fill = FALSE, stroke = 1, color = 'black') %>% 
  addPolygons(data = snohomish, fill = FALSE, stroke = 1, color = 'black') %>% 
  setView(lat = 48.35, lng = -121.5, zoom = 8) %>%
  addCircleMarkers(data = spatial_bcf(), 
                   lng = ~ lng, lat = ~ lat,
                   label = ~ file_name,
                   layerId = ~ file_name,
                   radius = 6,
                   color = ~ pal(lat),
                   stroke  = FALSE,
                   fillOpacity = .95)
})

output$dry_map <- renderLeaflet({
  pal <- colorBin(palette = "plasma", reverse = TRUE,
                  domain = dry_days_map_data()$prob_median, bins = 8, pretty=TRUE)
  
  leaflet() %>%
  addTiles(urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
           attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>') %>%
  addPolygons(data = skagit, fill = FALSE, stroke = 1, color = 'black') %>% 
  addPolygons(data = whatcom, fill = FALSE, stroke = 1, color = 'black') %>% 
  addPolygons(data = snohomish, fill = FALSE, stroke = 1, color = 'black') %>% 
  setView(lat = 48.35, lng = -121.5, zoom = 8) %>%
  addCircleMarkers(data = dry_days_map_data(), 
                   lng = ~ lng, lat = ~ lat,
                   label = ~ file_name,
                   layerId = ~ file_name,
                   radius = 6,
                   color = ~ pal(prob_median),
                   stroke  = FALSE,
                   fillOpacity = .95) %>% 
  addLegend("bottomleft", 
            pal = pal, 
            values = NULL, 
            title = "Difference from Exceedance Probability") 
})
######################################################################
#
# done
#
observeEvent(input$fcp_map_marker_click, {
  toggleModal(session, modalId = "bcf_graphs", toggle = "open")
  })

observeEvent(input$dry_map_marker_click, {
  toggleModal(session, modalId = "dry_days_graphs", toggle = "open")
  })

######################################################################

fcp_data <- reactive({

  # Test if location is selected
  validate(
            need(!is.null(input$fcp_map_marker_click$id), 
            "Please select a location")
          )
  # load data
  readRDS(paste0("/data/pruett/combined/data/", 
                 input$fcp_map_marker_click$id))
  
})

dry_days_data <- reactive({

  # Test if location is selected
  validate(
            need(!is.null(input$dry_map_marker_click$id), 
            "Please select a location")
          )
  # load data
  readRDS(paste0("/data/pruett/combined/data/", 
                 input$dry_map_marker_click$id))
  
})

  output$dry_days_plot <- renderPlot({
      
      # p_month <- plot_monthly_prob(dry_days_data_month(), "Monthly Probability")
      # p_octmar <- plot_octmar_prob(dry_days_data_octmar())
      
      # plot_grid(p_month, p_octmar, nrow = 1, align = "vh", rel_widths = c(4, 1), axis = 'b')
    
    plot_drydays_boxplot(dry_days_data(), input$dry_days_plot_climate_proj)
    
  }, res = 70, width = 400)
#########################################################
#########################################################
#
# Global part
#
#########################################################
#########################################################
data_dir <- "/data/hnoorazar/bloom_thresh_frost/"

bloom_f_name <- "fullbloom_50percent_day.rds"
frost_f_name <- "first_frost.rds"
CP_f_name <- "sept_summary_comp.rds"

bloom_dt <- readRDS(paste0(data_dir, bloom_f_name)) %>% 
            group_by(location, lat, lng)

frost_dt <- readRDS(paste0(data_dir, frost_f_name)) %>% 
            group_by(location, lat, lng)

thresh_dt <- readRDS(paste0(data_dir, CP_f_name)) %>% 
            group_by(location, lat, lng)



#### from analog:

observeEvent(input$fcp_map_marker_click, 
             { 
                 p <- input$fcp_map_marker_click$id
                 lat <- substr(as.character(p), 1, 8)
                 long <- substr(as.character(p), 13, 20)
                 file_dir_string <- paste0("/home/hnoorazar/ShinyApps/", 
                                           "bloom_thresh_frost/plots/", 
                                           "CM_locs/bloom_thresh_in_one", 
                                           "/no_obs/apple/thresh_75/")
                 
                 toggleModal(session, modalId = "bcf_graphs", toggle = "open")

                 curr_emission <- gsub(" ", "_", input$bcf_plot_climate_proj)
                 output$fcp_plot <- renderImage({
                       image_name <- paste0(lat, "_", long, "_", 
                                            curr_emission, "_", 
                                            input$fcp_plot_fruit_type,
                                            ".png")

                       filename <- normalizePath(file.path(file_dir_string))
                       # Return a list containing the filename and alt text
                       list(src = filename, width = 600, height = 600)
                                                }, deleteFile = FALSE
                                                )
            })


