           tabPanel(tags$b("Bloom vs. Chill Portion"),
                    fluidPage( id = "nav", inverse=FALSE, 
                               fluid=FALSE, title="Tool",
                               div( class="outer",
                                    tags$head(includeCSS("styles.css"),
                                              includeScript("gomap.js")
                                              ),
                                    leafletOutput("bcf_map", 
                                                   width="100%", height="100%")
                                  )
                              ),
                    fluidPage(bsModal(
                                       id="bcf_graphs",
                                       trigger = NULL,
                                       title = "",
                                       size = "large",                                       
                                       dashboardPage( dashboardHeader(title = "Plots"),
                                                      dashboardSidebar(
                                                                       radioButtons(inputId = "em_scenario",
                                                                                    label = tags$b("Scenario"),
                                                                                    choices = c("RCP 8.5" = "rcp85",
                                                                                                "RCP 4.5" = "rcp45"), 
                                                                                    selected = "RCP 8.5"),
                                                                       radioButtons("bcf_plot_fruit_type", 
                                                                                    label = h4("Fruit type"), 
                                                                                    choices = list("Cripps Pink" = "Cripps_Pink", 
                                                                                                   "Gala" = "Gala", 
                                                                                                   "Red Delicious" = "Red_Deli"),
                                                                                    selected = "Cripps_Pink")
                                                                       ),
                                                      #####################
                                                      #
                                                      # End of side bar of dashboard of analog maps
                                                      #
                                                      #####################
                                                      dashboardBody(
                                                                     tags$head(tags$style(HTML('.content-wrapper, 
                                                                                               .right-side {
                                                                                               background-color: #252d38;
                                                                                                 }'
                                                                                              ))
                                                                               ),
                                                                    plotOutput("bcf_plot"),
                                                                    br(), br(),
                                                                    br(), br(),
                                                                    br(), br(),
                                                                    br(), br(),
                                                                    br(), br(),                                                                    
                                                                    p((span("A county with", style = "color:white")),
                                                                      tags$b(span("yellow", style = "color:#fff200")), # "color:GoldenRod"
                                                                      (span(" border is the best analog", style = "color:white")),
                                                                      (span(" for a given county.", style = "color:white"))
                                                                      )
                                                                    )
                                                    )
                                      )
                              )
                    )

##############
  # plot part of bloom
observeEvent(input$bcf_map_marker_click, 
           { p <- input$bcf_map_marker_click$id
             lat <- substr(as.character(p), 1, 8)
             long <- substr(as.character(p), 13, 21)
             print(p)
             file_dir_string <- paste0("/data/hnoorazar/", 
                                       "bloom_thresh_frost/plots/", 
                                       "CM_locs/bloom_thresh_in_one",
                                       "/no_obs/apple/thresh_75/")
             toggleModal(session,
                         modalId = "bcf_graphs", 
                         toggle = "open")

             curr_emission <- gsub(" ", "_", input$bcf_plot_climate_proj)
             output$bcf_plot <- renderImage({
                   image_name <- paste0(lat, "_-", long, "_", 
                                        curr_emission, "_", 
                                        input$bcf_plot_fruit_type,
                                        ".png")

                   filename <- normalizePath(file.path(file_dir_string, image_name))
                   # Return a list containing the filename and alt text
                   list(src = filename, width = 800, height = 550)
                                            }, deleteFile = FALSE
                                            )
          })

# Server part
observeEvent(input$analog_front_page_shape_click, 
               { 
                 p <- input$analog_front_page_shape_click
                 toggleModal(session, 
                             modalId = "BCFGraphs", 
                             toggle = "open")
                 county <- rgdal::readOGR("/data/hnoorazar/bloom/shape_files/simle_county/", 
                                           layer = "simpleCounty")
                 
                 # get polygon of current selected county(boundary)
                 dat <- data.frame(Longitude = c(p$lng), Latitude =c(p$lat))
                 coordinates(dat) <- ~ Longitude + Latitude
                 proj4string(dat) <- proj4string(county)
                 current_county_name <- toString(over(dat, county)$NAME)
                 current_state_fip <- toString(over(dat, county)$STATEFP)
                 current_state_name <- st_cnty_names[st_cnty_names$state_fip == current_state_fip,]$state[1]

                output$bcf_plot <- renderImage({
                                                if (input$detail_level == "all_models"){
                                                    image_name <- paste0("all_mods_", 
                                                                   current_state_name, "_", 
                                                                   current_county_name, 
                                                                   ".png")
                                                    curr_emission <- input$em_scenario
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