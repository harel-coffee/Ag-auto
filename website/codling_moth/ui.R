# codling moth

library(leaflet)
library(shinyBS)
library(shiny)
library(plotly)
library(shinydashboard)

navbarPage(title = div("",
                       img(src='csanr_logo.png', style='width:100px;height:35px;'), 
                       img(src='WSU-DAS-log.png', style='width:100px;height:35px;'),
                       img(src='NW-Climate-Hub-Logo.jpg', style='width:100px;height:35px;')
                       ),
           id="nav", 
           windowTitle = "Codling Moth",
           #
           ############## Home Begin
           #
           tabPanel(tags$b("Home"),
                    navlistPanel(tabPanel(tags$b("About"), 
                                          tags$div(style="width:950px", 
                                                   includeHTML("home-page/about.html")
                                                   )
                                          ),
                                 ########################################################
                                 tabPanel(tags$b("People"), 
                                          tags$div(style="width:950px", 
                                                   includeHTML("home-page/people.html")
                                                   )
                                          ),
                                 tabPanel(tags$b("Codling Moth Life Cycle and Management"), 
                                          tags$div(style = "width: 950px", 
                                                   includeHTML("home-page/life-cycle.html")
                                                   )
                                          ),

                                 tabPanel(tags$b("Climate Data"), 
                                          tags$div(style="width:950px", 
                                                   includeHTML("home-page/climate-change-projections.html")
                                                   )
                                          ),

                                 tabPanel(tags$b("What's the story?"), 
                                                 tags$div(style="width: 950px", 
                                                          includeHTML("home-page/changing-pest-pressures.html")
                                                          )
                                                 ),

                                 tabPanel(tags$b("Contact"), 
                                          tags$div(style="width:950px", 
                                               includeHTML("home-page/contact.html")
                                               )
                                          ),

                                 tabPanel(tags$b("Take a tour! (video)"), 
                                          tags$div(style="width:950px", 
                                                   includeHTML("home-page/take-a-tour.html")
                                                   )
                                          ),
                                 widths = c(2,10)
                                 )
                    ),
           #
           ############## Home End
           #
           #
           ############## BLOOM start
           #
           # navbarMenu(tags$b("Bloom"),
           #            tabPanel("Median Day of Year", 
           #                     div(class="outer",
           #                         tags$head(includeCSS("styles.css")),
           #                         leafletOutput("map_bloom_doy", width="100%", height="100%"),
           #                         absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
           #                                       draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
           #                                       width = 250, height = "auto",
           #                                       h3(tags$b("Explorer")), 
           #                                       gsub("cg", "cg_bloom", includeHTML("explorer_climate_group.html")),
           #             selectInput("apple_type", label = h4(tags$b("Select Apple Variety")),
           #                                       choices = list("Cripps Pink" = "cripps_pink",
           #                                                      "Gala" = "gala", 
           #                                                      "Red Delicious" = "red_deli"),
           #                                       selected = "cripps_pink")))),

           #            tabPanel("Median Day of Year (new params. 100%)", 
           #                     div(class="outer",
           #                         tags$head(includeCSS("styles.css")),
           #                         leafletOutput("map_bloom_doy_100", width="100%", height="100%"),
           #                         absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
           #                                       draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
           #                                       width = 250, height = "auto",
           #                                       h3(tags$b("Explorer")), 
           #                                       gsub("cg", "cg_bloom_100", includeHTML("explorer_climate_group.html")),
           #             selectInput("apple_type", label = h4(tags$b("Select Apple Variety")),
           #                                       choices = list("Cripps Pink" = "cripps_pink", 
           #                                                      "Gala" = "gala", 
           #                                                      "Red Delicious" = "red_deli"),
           #                                       selected = "cripps_pink")))),

           #            tabPanel("Median Day of Year (new params. 95%)", 
           #                     div(class="outer",
           #                         tags$head(includeCSS("styles.css")),
           #                         leafletOutput("map_bloom_doy_95", width="100%", height="100%"),
           #                         absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
           #                                       draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
           #                                       width = 250, height = "auto",
           #                                       h3(tags$b("Explorer")), 
           #                                       gsub("cg", "cg_bloom_95", includeHTML("explorer_climate_group.html")),
           #             selectInput("apple_type", label = h4(tags$b("Select Apple Variety")),
           #                                       choices = list("Cripps Pink" = "cripps_pink", 
           #                                                      "Gala" = "gala", 
           #                                                      "Red Delicious" = "red_deli"),
           #                                       selected = "cripps_pink")))),

           #            tabPanel("Median Day of Year (new params. 50%)", 
           #                     div(class="outer",
           #                         tags$head(includeCSS("styles.css")),
           #                         leafletOutput("map_bloom_doy_50", width="100%", height="100%"),
           #                         absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
           #                                       draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
           #                                       width = 250, height = "auto",
           #                                       h3(tags$b("Explorer")), 
           #                                       gsub("cg", "cg_bloom_50", includeHTML("explorer_climate_group.html")),
           #             selectInput("apple_type", label = h4(tags$b("Select Apple Variety")),
           #                                       choices = list("Cripps Pink" = "cripps_pink", 
           #                                                      "Gala" = "gala", 
           #                                                      "Red Delicious" = "red_deli"),
           #                                       selected = "cripps_pink")))),

           #            ######################################################
           #            ######################################################
           #            ######################################################
           #            tabPanel("Difference from Historical", 
           #                     div(class="outer",
           #                         tags$head(
           #                           # Include our custom CSS
           #                           includeCSS("styles.css")
           #                           #includeScript("gomap.js")
           #                         ),
           #                         leafletOutput("map_bloom_diff", width="100%", height="100%"),
           #                         absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
           #                                       draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
           #                                       width = 250, height = "auto",
                                                 
           #                                       h3(tags$b("Explorer")),
           #                                        gsub("cg", "cg_bloom_diff", includeHTML("explorer_climate_group_diff.html")),

           #                      selectInput("apple_type_diff", label = h4(tags$b("Select Apple Variety")),
           #                                             choices = list( "Cripps Pink" = "cripps_pink",
           #              "Gala" = "gala", 
           #                                "Red Delicious" = "red_deli"),
           #             selected = "cripps_pink"))))
           #            ),
           #
           ############## BLOOM END
           #
           #
           ############## CM Flight START
           #
           navbarMenu(tags$b("Flight"),
                      tabPanel("Median Day of Year (First Flight)", 
                               div(class="outer",
                                   tags$head(includeCSS("styles.css")),
                                   leafletOutput("map_emerg_doy", width="100%", height="100%"),
                                   absolutePanel(id = "controls", 
                                                 class = "panel panel-default", fixed = TRUE,
                                                 draggable = TRUE, 
                                                 top = 60, left = "auto", 
                                                 right = 20, bottom = "auto",
                                                 width = 250, height = "auto",
                                                 h3(tags$b("Median Day of Year (First Flight)")),
                                                 gsub("cg", "cg_emerg_doy", includeHTML("explorer_climate_group.html"))
                                                )
                                  )
                              ),
                      tabPanel("Difference from Historical (First Flight)",
                               div(class="outer",
                                   tags$head(includeCSS("styles.css")),
                                   leafletOutput("map_diff_emerg", width="100%", height="100%"),
                                   absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                                                 draggable = TRUE, 
                                                 top = 60, left = "auto", 
                                                 right = 20, bottom = "auto",
                                                 width = 250, height = "auto",
                                                 h3(tags$b("Difference from Historical (First Flight)")),
                                                 gsub("cg", "cg_diff_emerg", includeHTML("explorer_climate_group_diff.html"))
                                                 )
                                   )
                               ),
                      tabPanel("Median Day Of Year (By Generation)", 
                               div(class="outer",
                                   tags$head(includeCSS("styles.css")),
                                   leafletOutput("map_adult_med_doy", width="100%", height="100%"),
                                   absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                                                 draggable = TRUE, 
                                                 top = 60, left = "auto", 
                                                 right = 20, bottom = "auto",
                                                 width = 250, height = "auto",
                                                 h3(tags$b("Flight - Median Day Of Year (By Generation)")),
                                                 gsub("cg", "cg_adult_med_doy", includeHTML("explorer_climate_group.html")),
                                                 selectInput("adult_gen", 
                                                             label = h4(tags$b("Select Generation")),
                                                             choices = list("Generation 1" = "Gen1", 
                                                                            "Generation 2" = "Gen2",
                                                                            "Generation 3" = "Gen3", 
                                                                            "Generation 4" = "Gen4"),
                                                             selected = "Gen1"),
                                                 radioButtons("adult_percent", 
                                                              label = h4(tags$b("Select % Population that has completed the growth stage")),
                                                              choices = list("25 %" = "0.25", "50 %" = "0.5", "75 %" = "0.75"),
                                                              selected = "0.25", inline = TRUE
                                                              )
                                                 )
                                   )
                               ),
                      tabPanel("Difference from Historical (By Generation)",
                               div(class="outer",
                                   tags$head(includeCSS("styles.css")),
                                   leafletOutput("map_adult_diff_doy", width="100%", height="100%"),
                                   absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                                                 draggable = TRUE, 
                                                 top = 60, left = "auto", 
                                                 right = 20, bottom = "auto",
                                                 width = 250, height = "auto",
                                                 h3(tags$b("Flight - Difference from Historical (By Generation)")),
                                                 gsub("cg", "cg_adult_doy_diff", includeHTML("explorer_climate_group_diff.html")),
                                                 selectInput("adult_gen_diff", 
                                                             label = h4(tags$b("Select Generation")),
                                                             choices = list("Generation 1" = "Gen1", 
                                                                            "Generation 2" = "Gen2", 
                                                                            "Generation 3" = "Gen3", 
                                                                            "Generation 4" = "Gen4"),
                                                             selected = "Gen1"),
                                                 radioButtons("adult_percent_diff", 
                                                              label = h4(tags$b("Select % Population that has completed the growth stage")),
                                                              choices = list("25 %" = "0.25", "50 %" = "0.5", "75 %" = "0.75"),
                                                              selected = "0.25", inline = TRUE)
                                                 )
                                   )
                               )
                     ),
           #
           ############## CM Flight END
           #
           #
           ############## CM Egg Hatch START
           #
           navbarMenu(tags$b("Egg Hatch"),
                      tabPanel("Pest Risk",
                               div(class="outer",
                                   tags$head(includeCSS("styles.css")),
                                   leafletOutput("map_risk", width="100%", height="100%"),
                                   absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                                                 draggable = TRUE, 
                                                 top = 60, left = "auto", 
                                                 right = 20, bottom = "auto",
                                                 width = 250, height = "auto",
                                                 h3(tags$b("Pest Risk")),
                                                 gsub("cg", "cg_risk", includeHTML("explorer_climate_group.html")),
                                                 selectInput("gen_risk", 
                                                             label = h4(tags$b("Select Generation")),
                                                             choices = list("Generation 3" = "Gen3", 
                                                                            "Generation 4" = "Gen4"),
                                                             selected = "Gen3"
                                                             ),
                                                radioButtons("percent_risk", 
                                                             label = h4(tags$b("Select Proportion of Eggs Hatched")),
                                                             choices = list("25 %" = "0.25", "50 %" = "0.5", "75 %" = "0.75"),
                                                             selected = "0.75", 
                                                             inline = TRUE
                                                             )
                                                )
                                  )
                              ),
                      tabPanel("Median Day Of Year (By Generation)", 
                               div(class="outer",
                                   tags$head(includeCSS("styles.css")),
                                   leafletOutput("map_larvae_med_doy", width="100%", height="100%"),
                                   absolutePanel(id = "controls", 
                                                 class = "panel panel-default", 
                                                 fixed = TRUE,
                                                 draggable = TRUE, 
                                                 top = 60, left = "auto", 
                                                 right = 20, bottom = "auto",
                                                 width = 250, height = "auto",
                                                 h3(tags$b("Egg Hatch - Median Day Of Year (By Generation)")),
                                                 gsub("cg", "cg_larvae_med_doy", includeHTML("explorer_climate_group.html")),
                                                 selectInput("larvae_gen", 
                                                             label = h4(tags$b("Select Generation")),
                                                             choices = list("Generation 1" = "Gen1", 
                                                                            "Generation 2" = "Gen2",
                                                                            "Generation 3" = "Gen3", 
                                                                            "Generation 4" = "Gen4"),
                                                             selected = "Gen1"),
                                                 radioButtons("larvae_percent", 
                                                               label = h4(tags$b("Select Proportion of Eggs hatched")),
                                                               choices = list("25 %" = "0.25", 
                                                                              "50 %" = "0.5", 
                                                                              "75 %" = "0.75"),
                                                               selected = "0.25", inline = TRUE
                                                               )
                                                 )
                                   )
                              ),
                      tabPanel("Difference from Historical (By Generation)",
                               div(class="outer",
                                   tags$head(includeCSS("styles.css")),
                                   leafletOutput("map_larvae_diff_doy", width="100%", height="100%"),
                                   absolutePanel(id = "controls", 
                                                 class = "panel panel-default", 
                                                 fixed = TRUE,
                                                 draggable = TRUE, 
                                                 top = 60, left = "auto", 
                                                 right = 20, bottom = "auto",
                                                 width = 250, height = "auto",
                                                 h3(tags$b("Egg Hatch - Difference from Historical (By Generation)")),
                                                 gsub("cg", "cg_larvae_doy_diff", includeHTML("explorer_climate_group_diff.html")),
                                                 selectInput("larvae_gen_diff", label = h4(tags$b("Select Generation")),
                                                             choices = list("Generation 1" = "Gen1", 
                                                                            "Generation 2" = "Gen2", 
                                                                            "Generation 3" = "Gen3", 
                                                                            "Generation 4" = "Gen4"),
                                                             selected = "Gen1"),
                                                 radioButtons("larvae_percent_diff", 
                                                              label = h4(tags$b("Select Proportion of Eggs hatched")),
                                                              choices = list("25 %" = "0.25", 
                                                                             "50 %" = "0.5", 
                                                                             "75 %" = "0.75"),
                                                              selected = "0.25", 
                                                              inline = TRUE
                                                              )
                                                 )
                                   )
                              )
                      ),
           #
           ############## CM Egg Hatch END
           #
           #
           ############## CM Diapause START
           #
           tabPanel(tags$b("Diapause"),
                    div(class="outer",
                        tags$head(includeCSS("styles.css")),
                        leafletOutput("map_diap_pop", width="100%", height="100%"),
                        absolutePanel(id = "controls", class = "panel panel-default", 
                                      fixed = TRUE, draggable = TRUE, 
                                      top = 60, left = "auto", 
                                      right = 20, bottom = "auto",
                                      width = 250, height = "auto",
                                      h3(tags$b("Diapause")),
                                      gsub("cg", "cg_diap", includeHTML("explorer_climate_group.html")),
                                      radioButtons("diapaused", 
                                                   label = h4(tags$b("Select Diapause/Non-Diapause")),
                                                   choices = list("Diapause Escaped" = "NonDiap", 
                                                                  "Diapause Induced" = "Diap"),
                                                   selected = "NonDiap", 
                                                   inline = FALSE
                                                   ),
                                      selectInput("diap_gen",
                                                  label = h4(tags$b("Select Generation")),
                                                  choices = list("Generation 1" = "Gen1", 
                                                                 "Generation 2" = "Gen2",
                                                                 "Generation 3" = "Gen3", 
                                                                 "Generation 4" = "Gen4",
                                                                 "All" = "all"),
                                                   selected = "Gen1"
                                                   )
                                      )
                        )
                    ),
           #
           ############## CM Diapause END
           #
           #
           ############## Regional Plots START
           #
           tabPanel(tags$b("Regional Plots"),
                    navlistPanel(
                                 ####### Subregion Groups start
                                 # tabPanel("Subregions Map", imageOutput("location_group")
                                 # # tabPanel("Summary", verbatimTextOutput("location_summary"))
                                 #  ),
                                 tabPanel("Subregions Map", 
                                          tags$b("Sites are divided into two subregions:"),
                                          br(),br(),
                                          imageOutput("location_group"),
                                          br(),
                                          tags$h3("We use statistical methods ", 
                                                  "to group the regions into two", 
                                                  " areas based on average historical", 
                                                  " growing degree day accumulation.",
                                                  " The warmer areas have an avg.", 
                                                  " annual historical GDD of XX ", 
                                                  "degree day F (provide a range) ", 
                                                  "and the cooler areas have a ", 
                                                  "historical GDD of YY degree ", 
                                                  "days F (provide a range).")
                                          ),
                                 # tabPanel(tags$b("Subregion Map HTML"), 
                                 #          tags$div(style="width:950px", 
                                 #                   includeHTML("HTML_plots/subregion_groups.html")
                                 #                   )),

                                 # tabsetPanel(
                                 #             tabPanel("Subregions Map", imageOutput("location_group")),
                                 #             tabPanel("Summary", verbatimTextOutput("subregion_summary"))),
                                 ####### Subregion Groups End
                                 HTML("<b>RCP 8.5</b>"),
                                 ####### BLOOM start
                                 # tabPanel("Bloom", imageOutput("full_bloom")),
                                 ####### BLOOM END

                                 ####### No. of generations 85 start
                                 tabPanel("No. of Generations",
                                          fluidRow(tabBox(tabPanel("Adult", 
                                                                   column(7, imageOutput("Adult_Gen_Aug_rcp85")),
                                                                   br(),br(),br(),
                                                                   column(1, imageOutput("location_group_1")),
                                                                   br(),br(), br(),br(), br(),br(),
                                                                   br(),br(), br(),br(), br(),br(),
                                                                   br(),br(), br(),br(), br(),br(),
                                                                   br(),br(), br(),br(), br(),br(),
                                                                   height = 1700
                                                                   # column(10,
                                                                   #        div(style="display: inline-block; width: 70%;",img(src="Adult_Gen_Aug_rcp85.png", width=500, height=400)),
                                                                   #        div(style="display: inline-block; width: 29%;",img(src="location-group.png", width=350, height=400))
                                                                   #       ),
                                                                   # br(),br(), br(),br(), br(),br(),
                                                                   # br(),br(), br(),br(), br(),br(),
                                                                   # br(),br(), br(),br(), br(),br(),
                                                                   # br(),br(), br(),br(), br(),br(),
                                                                   # br(),br(), br(),br(), br(),br()
                                                                   ),
                                                          tabPanel("Larva", 
                                                                   # imageOutput("Larva_Gen_Aug_rcp85"), 
                                                                   # br(), br(),
                                                                   # height=700
                                                                   column(7, imageOutput("Larva_Gen_Aug_rcp85")),
                                                                   br(),br(),br(),
                                                                   column(1, imageOutput("location_group_2")),
                                                                   br(),br(), br(),br(), br(),br(),
                                                                   br(),br(), br(),br(), br(),br(),
                                                                   br(),br(), br(),br(), br(),br(),
                                                                   br(),br(), br(),br(), br(),br(),
                                                                   height = 1700
                                                                   ), 
                                                          width = 12
                                                          )
                                                  )
                                          ),
                                 ####### No. of generations 85 END
                                 
                                 ####### DD 85 start

                                 # tabPanel("LLL", 
                                    #       column(7, imageOutput("Larva_Gen_Aug_rcp85")),
                                    #       br(),br(),
                                    #       column(1, imageOutput("location_group_3")),
                                    #       br(),br(), br(),br(), br(),br(),
                                    #       br(),br(), br(),br(), br(),br(),
                                    #       br(),br(), br(),br(), br(),br(),
                                    #       br(),br(), br(),br(), br(),br()
                                    #       ), 

                                 tabPanel("Degree Days", 
                                           # imageOutput("cumdd")
                                          column(7, imageOutput("cumdd")),
                                          br(),br(),br(),br(),
                                          column(1, imageOutput("location_group_3")),
                                          height = 1700
                                           ),
                                 ####### DD 85 END
                                 
                                 ####### Adult Flight 85 start
                                 tabPanel("Adult First Flight", 
                                          # imageOutput("adult_emergence_rcp85")
                                          column(7, imageOutput("adult_emergence_rcp85")),
                                          br(),br(),br(),br(),
                                          column(1, imageOutput("location_group_4")),
                                          height = 1700
                                          ),
                                 ####### Adult Flight 85 END

                                 ####### Egg Hatch into Larva start
                                 tabPanel("Egg Hatch into Larva", 
                                           # imageOutput("cum_larva_pop")
                                          column(7, imageOutput("cum_larva_pop")),
                                          br(),br(),br(),br(),
                                          column(1, imageOutput("location_group_5")),
                                          height = 1700
                                           ),

                                 # tabPanel("Egg Hatch into Larva",
                                 #          fluidRow(tabBox(tabPanel("Cumulative Larva Population Fraction", imageOutput("cum_larva_pop")),
                                 #                          width = 12
                                 #                          )
                                 #                   )
                                 #          ),
                                 ####### Egg Hatch into Larva END

                                 ####### DIAPAUSE 85 start
                                 tabPanel("Diapause", 
                                           # imageOutput("diapause_abs_rcp85")
                                          column(9, imageOutput("diapause_abs_rcp85")),
                                          br(),br(),br(),br(),br(),br(),
                                          column(1, imageOutput("location_group_6")),
                                          height = 1700
                                           ),
                                 
                                 # tabPanel("Diapause",
                                 #          fluidRow(tabBox(# tabPanel("Relative Population Vs Cumulative DD", imageOutput("rel_pop_cumdd")), # Hossein
                                 #                          tabPanel("Absolute Population Vs Cumulative DD", imageOutput("diapause_abs_rcp85")),
                                 #                          width = 12
                                 #                          )
                                 #                  )
                                 #          ),
                                 ####### DIAPAUSE 85 END
                                 HTML("<b>RCP 4.5</b>"),
                                 ####### BLOOM 45 start
                                 # tabPanel("Bloom", imageOutput("full_bloom_rcp45")),
                                 ####### BLOOM 45 END

                                 ####### No. of generations 45 start
                                 tabPanel("No. of Generations",
                                          fluidRow(tabBox(tabPanel("Adult", 
                                                                   # imageOutput("Adult_Gen_Aug_rcp45"),
                                                                   # br(), br(),
                                                                   # height=700
                                                                   column(7, imageOutput("Adult_Gen_Aug_rcp45")),
                                                                          br(),br(),br(),
                                                                          column(1, imageOutput("location_group_7")),
                                                                          br(),br(), br(),br(), br(),br(),
                                                                          br(),br(), br(),br(), br(),br(),
                                                                          br(),br(), br(),br(), br(),br(),
                                                                          br(),br(), br(),br(), br(),br(),
                                                                          height = 1700
                                                                   ),

                                                          tabPanel("Larva", 
                                                                   # imageOutput("Larva_Gen_Aug_rcp45"), 
                                                                   # br(), br(),
                                                                   # height=700
                                                                   column(7, imageOutput("Larva_Gen_Aug_rcp45")),
                                                                          br(),br(),br(),
                                                                          column(1, imageOutput("location_group_8")),
                                                                          br(),br(), br(),br(), br(),br(),
                                                                          br(),br(), br(),br(), br(),br(),
                                                                          br(),br(), br(),br(), br(),br(),
                                                                          br(),br(), br(),br(), br(),br(),
                                                                          height = 1700
                                                                   ),
                                                          width = 12
                                                          )
                                                  )
                                          ),

                                 # tabPanel("No. of Generations", imageOutput("Adult_Gen_Aug_rcp45")),
                                 ####### No. of generations 45 END

                                 ####### DD 45 start
                                 tabPanel("Degree Days", 
                                 	      # imageOutput("cumdd_rcp45")
                                 	        column(7, imageOutput("cumdd_rcp45")),
                                          br(),br(),br(),br(),
                                          column(1, imageOutput("location_group_9")),
                                          height = 1700
                                 	      ),
                                 ####### DD 45 END

                                 ####### Adult Flight 45 start
                                 tabPanel("Adult First Flight", 
                                 	        # imageOutput("adult_emergence_rcp45")
                                 	        column(7, imageOutput("adult_emergence_rcp45")),
                                          br(),br(),br(),br(),
                                          column(1, imageOutput("location_group_10")),
                                          height = 1700
                                 	      ),
                                 ####### Adult Flight 45 END

                                 ####### Egg Hatch 45 start
                                 tabPanel("Egg Hatch into Larva", 
                                 	      # imageOutput("cum_larva_pop_rcp45")
                                 	        column(7, imageOutput("cum_larva_pop_rcp45")),
                                          br(),br(),br(),br(),
                                          column(1, imageOutput("location_group_11")),
                                          height = 1700
                                 	      ),
                                 # tabPanel("Egg Hatch into Larva",
                                 #          fluidRow(tabBox(
                                 #                          tabPanel("Cumulative Larva Population Fraction", 
                                 #                                   imageOutput("cum_larva_pop_rcp45")),
                                 #                          width = 12
                                 #                          )
                                 #                  )
                                 #          ),
                                 ####### Egg Hatch 45 END

                                 ####### Diapause 45 START
                                 tabPanel("Diapause", 
                                 	      # imageOutput("diapause_abs_rcp45")
                                 	      column(9, imageOutput("diapause_abs_rcp45")),
                                          br(),br(),br(),br(),br(),br(),
                                          column(1, imageOutput("location_group_12")),
                                          height = 1700
                                 	      ),
                                 # tabPanel("Diapause",
                                 #          fluidRow(tabBox(# tabPanel("Relative Population Vs Cumulative DD", imageOutput("rel_pop_cumdd_rcp45")), # Hossein
                                 #                          tabPanel("Absolute Population Vs Cumulative DD", imageOutput("abs_pop_cumdd_rcp45")),
                                 #                          width = 12
                                 #                          )
                                 #                   )
                                 #          ),
                                 ####### Diapause 45 END
                                 widths = c(2,10)
                                 )
                   ),
           #
           ############## Regional Plots END
           #
           #
           ############## Analogs Map Front Page start
           #
           tabPanel(tags$b("Analog Map"),
                    fluidPage( id = "nav", inverse=FALSE, fluid=FALSE, title="Tool",
                               div( class="outer",
                                    tags$head(includeCSS("styles.css"),
                                              includeScript("gomap.js")
                                              ),
                                    leafletOutput("analog_front_page", 
                                                   width="100%", height="100%")
                                  )
                              ),
                    fluidPage(bsModal( "Graphs", trigger = NULL, title = "", size = "large",
                                       dashboardPage( dashboardHeader(title = "Plots"),
                                                      dashboardSidebar(
                                                                       selectInput(inputId = "detail_level",
                                                                                   label = tags$b("Detail Level"),
                                                                                   choices = detail_levels, 
                                                                                   selected = detail_levels[1]
                                                                                   ),
                                                                       conditionalPanel(condition = "input.detail_level == 'all_models'",
                                                                                        radioButtons(inputId = "arrow_emission",
                                                                                                     label = tags$b("Scenario"),
                                                                                                     choices = emissions, 
                                                                                                     selected = emissions[1]
                                                                                                     )
                                                                                        ),

                                                                       # Only show this panel if the level of detail is more_details
                                                                       conditionalPanel(condition = "input.detail_level == 'more_details'",
                                                                                        radioButtons(inputId = "emission", 
                                                                                                     label = tags$b("Scenario"),
                                                                                                     choices = emissions, 
                                                                                                     selected = emissions[1]),

                                                                                        radioButtons(inputId = "time_period", 
                                                                                                     label = tags$b("Time Period"),
                                                                                                     choices = time_periods,
                                                                                                     selected = time_periods[3]),

                                                                                        selectInput(inputId = "climate_model", 
                                                                                                    label = tags$b("Climate Model"),
                                                                                                    choices = climate_models, 
                                                                                                    selected = climate_models[1]
                                                                                                    )
                                                                                        ) 
                                                                       
                                                                       ),
                                                      #####################
                                                      #
                                                      # End of side bar of dashboard of analog maps
                                                      #
                                                      #####################
                                                      dashboardBody(tags$head(tags$style(HTML('.content-wrapper, 
                                                                                               .right-side {
                                                                                               background-color: #252d38;
                                                                                                 }
                                                                                              ')
                                                                                        )
                                                                              ),
                                                                    plotOutput("Plot"), # ,
                                                                    br(), br(),
                                                                    br(), br(),
                                                                    br(), br(),
                                                                    br(), br(),
                                                                    br(), br(),
                                                                    p(tags$b(span("Best-Analogs", style = "color:white")),
                                                                      (span(" plot includes most similar", style = "color:white")),
                                                                      (span(" county in all models.", style = "color:white"))
                                                                      ),
                                                                    
                                                                    p((span("A county with", style = "color:white")),
                                                                      tags$b(span("red", style = "color:red")),
                                                                      (span(" border is in the future.", style = "color:white"))
                                                                      ), 
                                                                    
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
           #
           ############## Analogs Map Front page END
           #
           

       )
