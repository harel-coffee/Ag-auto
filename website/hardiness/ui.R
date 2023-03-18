# Hardiness

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
           windowTitle = "Cold Hardidness",
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
           ############## Bloom, CP, frost Map Front Page start
           #
           tabPanel(tags$b("Cold Hardiness Plots"),
                    fluidPage( id = "nav", inverse=FALSE, 
                               fluid=FALSE, title="Tool",
                               div( class="outer",
                                    tags$head(includeCSS("styles.css"),
                                              includeScript("gomap.js")
                                              ),
                                    leafletOutput("hard_map", 
                                                   width="100%", 
                                                   height="100%")
                                  )
                              ),
                    fluidPage(bsModal(
                                       id="hard_graph",
                                       trigger = NULL,
                                       title = "",
                                       size = "large",                                       
                                       dashboardPage( dashboardHeader(title = "Plots"),
                                                      dashboardSidebar(
                                                                       radioButtons(inputId = "em_scenario",
                                                                                    label = tags$b("Scenario"),
                                                                                    choices = c(
                                                                                    	        # "RCP 8.5" = "RCP 8.5",
                                                                                                # "RCP 4.5" = "RCP 4.5",
                                                                                                "Observed" = "Observed"), 
                                                                                    selected = "Observed")
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
                                                                     fluidPage(fluidRow(column(2, 
                                                                                               offset = 0, 
                                                                                               plotOutput("hard_plot", height = 550)
                                                                                               )
                                                                                        )
                                                                              )

                                                                    )
                                                    )
                                      )
                              )
                    )

       )
