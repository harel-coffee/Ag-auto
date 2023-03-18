# Bloom - Vince

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
           windowTitle = "Bloom",
           #
           ############## Home Begin
           #
           tabPanel(tags$b("Home"),
                    navlistPanel(tabPanel(tags$b("About"), 
                                          tags$div(style="width:950px", 
                                                   includeHTML("home-page/about.html")
                                                   )
                                          ),
                                 
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

                                 # tabPanel(tags$b("Take a tour! (video)"), 
                                 #          tags$div(style="width:950px", 
                                 #                   includeHTML("home-page/take-a-tour.html")
                                 #                   )
                                 #          ),
                                 widths = c(2,10)
                                 )
                    ),
           #
           ############## Home End
           #
           #
           ############## BLOOM start
           #
           navbarMenu(tags$b("Bloom"),
                      # tabPanel("Median Day of Year", 
                      #          div(class="outer",
                      #              tags$head(includeCSS("styles.css")),
                      #              leafletOutput("map_bloom_doy", width="100%", height="100%"),
                      #              absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                      #                            draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
                      #                            width = 250, height = "auto",
                      #                            h3(tags$b("Bloom Median DoY")), 
                      #                            gsub("cg", "cg_bloom", includeHTML("explorer_climate_group.html")),
                      #  selectInput("apple_type", label = h4(tags$b("Select Apple Variety")),
                      #                            choices = list("Cripps Pink" = "cripps_pink",
                      #                                           "Gala" = "gala", 
                      #                                           "Red Delicious" = "red_deli"),
                      #                            selected = "cripps_pink")))),

                      # tabPanel("Median Day of Year (new params. 100%)", 
                      #          div(class="outer",
                      #              tags$head(includeCSS("styles.css")),
                      #              leafletOutput("map_bloom_doy_100", width="100%", height="100%"),
                      #              absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                      #                            draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
                      #                            width = 250, height = "auto",
                      #                            h3(tags$b("Explorer")), 
                      #                            gsub("cg", "cg_bloom_100", includeHTML("explorer_climate_group.html")),
                      #  selectInput("apple_type", label = h4(tags$b("Select Apple Variety")),
                      #                            choices = list("Cripps Pink" = "cripps_pink", 
                      #                                           "Gala" = "gala", 
                      #                                           "Red Delicious" = "red_deli"),
                      #                            selected = "cripps_pink")))),

                      # tabPanel("Median Day of Year (new params. 95%)", 
                      #          div(class="outer",
                      #              tags$head(includeCSS("styles.css")),
                      #              leafletOutput("map_bloom_doy_95", width="100%", height="100%"),
                      #              absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                      #                            draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
                      #                            width = 250, height = "auto",
                      #                            h3(tags$b("Explorer")), 
                      #                            gsub("cg", "cg_bloom_95", includeHTML("explorer_climate_group.html")),
                      #  selectInput("apple_type", label = h4(tags$b("Select Apple Variety")),
                      #                            choices = list("Cripps Pink" = "cripps_pink", 
                      #                                           "Gala" = "gala", 
                      #                                           "Red Delicious" = "red_deli"),
                      #                            selected = "cripps_pink")))),

                      tabPanel("Median Day of Year", 
                               div(class="outer",
                                   tags$head(includeCSS("styles.css")),
                                   leafletOutput("map_bloom_doy_50", width="100%", height="100%"),
                                   absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                                                 draggable = TRUE, top = 60, 
                                                 left = "auto", right = 20, bottom = "auto",
                                                 width = 250, height = "auto",
                                                 h3(tags$b("Bloom - Median Day of Year")), 
                                                 gsub("cg", "cg_bloom_50", includeHTML("explorer_climate_group.html")),
                       selectInput("apple_type", label = h4(tags$b("Select Apple Variety")),
                                                 choices = list("Cripps Pink" = "cripps_pink", 
                                                                "Gala" = "gala", 
                                                                "Red Delicious" = "red_deli"),
                                                 selected = "cripps_pink")))),

                      ######################################################
                      ######################################################
                      ######################################################
                      tabPanel("Difference from Historical", 
                               div(class="outer",
                                   tags$head(
                                     # Include our custom CSS
                                     includeCSS("styles.css")
                                     #includeScript("gomap.js")
                                   ),
                                   leafletOutput("map_bloom_diff", width="100%", height="100%"),
                                   absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                                                 draggable = TRUE, 
                                                 top = 60, left = "auto", right = 20, 
                                                 bottom = "auto",
                                                 width = 250, height = "auto",
                                                 
                                                 h3(tags$b("Bloom - Difference from Historical")),
                                                 gsub("cg", "cg_bloom_diff", 
                                                      includeHTML("explorer_climate_group_diff.html")),

                                                 selectInput("apple_type_diff", 
                                                             label = h4(tags$b("Select Apple Variety")),
                                                             choices = list( "Cripps Pink" = "cripps_pink",
                                                                             "Gala" = "gala", 
                                                                             "Red Delicious" = "red_deli"),
                                                             selected = "cripps_pink"))))
                      ),
           #
           ############## BLOOM END
           #
           #
           ############## Bloom, CP, frost Map Front Page start
           #
           tabPanel(tags$b("Bloom vs. Chill Portion"),
                    fluidPage( id = "nav", inverse=FALSE, 
                               fluid=FALSE, title="Tool",
                               div( class="outer",
                                    tags$head(includeCSS("styles.css"),
                                              includeScript("gomap.js")
                                              ),
                                    leafletOutput("bcf_map", 
                                                   width="100%", 
                                                   height="100%")
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
                                                                                    choices = c("RCP 8.5" = "RCP 8.5",
                                                                                                "RCP 4.5" = "RCP 4.5"), 
                                                                                    selected = "RCP 8.5"),
                                                                       radioButtons(inputId="bcf_plot_fruit_type", 
                                                                                    label = h4("Fruit type"), 
                                                                                    choices = list("Cripps Pink" = "Cripps_Pink", 
                                                                                                   "Gala" = "Gala", 
                                                                                                   "Red Delicious" = "Red_Deli",
                                                                                                   "Pear" = "pear",
                                                                                                   "Cherry" = "cherry"),
                                                                                    selected = "Cripps_Pink"),
                                                                       radioButtons(inputId = "cp_thresh",
                                                                                    label = tags$b("CP threshold"),
                                                                                    choices = c("75 CP" = "thresh_75",
                                                                                                "70 CP" = "thresh_70",
                                                                                                "65 CP" = "thresh_65",
                                                                                                "60 CP" = "thresh_60",
                                                                                                "55 CP" = "thresh_55",
                                                                                                "50 CP" = "thresh_50"), 
                                                                                    selected = "thresh_75")
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
                                                                                              ))),
                                                                     # plotOutput("bcf_plot")
                                                                     fluidPage(fluidRow(column(2, offset = 0, plotOutput("bcf_plot", height = 550)))
                                                                              )

                                                                    )
                                                    )
                                      )
                              )
                    )

       )
