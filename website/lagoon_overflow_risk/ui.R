# Lagoon

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
           windowTitle = "Lagoon Overflow Risk",
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
           #     Subregion Map start
           #
           tabPanel(tags$b("Grouping Locations"), 
                    # tags$b(
                           paste0("The grids are clustered", 
                                  "into 5 subregions based on ", 
                                  "elevation and average annual ", 
                                  "precipitation."),
                           #),
                    # br(),br(),
                    imageOutput("location_group"),
                    # tags$h4(
                            paste0("The variables", 
                                   " distribution is given below ", 
                                   "and they correspond to ", 
                                   "historical data:"),
                            #),
                    imageOutput("cluster_visualization"),
                    br(), br(), br(), br(), br(), br()
                    ),

           # tabPanel(tags$b("Subregion Map"),
           #          navlistPanel(
           #                       ####### Western Coastal 85 start
                                 
           #                       fluidRow(tabBox(
           #                                       tabPanel("Annual", 
           #                                                tags$blockquote("Below we see annual precipitation ranges."),
           #                                                imageOutput("rain_85"),
           #                                                br(),br(), br(),br(),br(), br(),
           #                                                br(),br(), br(),br(),br(), br(),
           #                                                tags$blockquote("Below we see annual runoff ranges."),
           #                                                imageOutput("runoff_85"),
           #                                                br(),br(),
           #                                                tags$blockquote("and finally 25-year/24-hour design storm intensity"),
           #                                                imageOutput("storm_85")
           #                                                ),
           #                                         width = 12
           #                                        )
           #                                 )
           #                                ),
           #                       widths = c(2,10)
           #                       )
           #         ),
           #############
           #
           #     Subregion Map End
           #
           ##############
           ############## Annual Plots start
           #
           tabPanel(tags$b("Annual Plots"),
                    navlistPanel(
                                 ####### Western Coastal 85 start
                                 tabPanel("RCP 8.5",
                                          fluidRow(tabBox(
                                                          tabPanel("Annual", 
                                                                   tags$blockquote("Below we see annual precipitation ranges."),
                                                                   imageOutput("rain_85"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see annual runoff ranges."),
                                                                   imageOutput("runoff_85"),
                                                                   br(),br(),
                                                                   tags$blockquote("and finally 25-year/24-hour design storm intensity"),
                                                                   imageOutput("storm_85")
                                                                   ),
                                                          width = 12
                                                          )
                                                  )
                                          ),
                                 tabPanel("RCP 4.5",
                                          fluidRow(tabBox(
                                                          tabPanel("Annual", 
                                                                   tags$blockquote("Below we see annual precipitation ranges."),
                                                                   imageOutput("rain_45"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see annual runoff ranges."),
                                                                   imageOutput("runoff_45"),
                                                                   br(),br(),
                                                                   tags$blockquote("and finally 25-year/24-hour design storm intensity"),
                                                                   imageOutput("storm_45")
                                                                   ),
                                                          width = 12
                                                          )
                                                  )
                                          ),
                                 widths = c(2,10)
                                 )
                   ),
           ############## Annual Plots End
           #
           ############## Regional Plots START
           #
           tabPanel(tags$b("Monthly and Seasonal Plots"),
                    navlistPanel(
                                 ##### Subregion Groups start
                                 # tabPanel(tags$b("Subregion Map"),
                                 #          tags$div(style="width:950px", 
                                 #                   includeHTML("./HTML_plots/subregion_groups.html")
                                 #                   )
                                 #          ),

                                 # tabPanel(tags$b("Subregion Map"),
                                 #          tags$h4(paste0("We have clustered the grids ", 
                                 #                         "into 5 subregions based on ", 
                                 #                         "elevation and average annual ", 
                                 #                         "precipitation.")),
                                 #          imageOutput("location_group"),
                                 #          tags$h4(paste0("The variables", 
                                 #                         " distribution is given below ", 
                                 #                         "and they correspond to ", 
                                 #                         "historical data:")),
                                 #          imageOutput("cluster_visualization"),
                                 #          br(),br()
                                 #          ),
                                 ####### Subregion Groups End
                                 HTML("<b>RCP 8.5</b>"),
                                 
                                 ####### Western Coastal 85 start
                                 tabPanel("Western coastal",
                                          fluidRow(tabBox(
                                                          tabPanel("Annual", 
                                                                   tags$blockquote("Below we see annual precipitation ranges."),
                                                                   imageOutput("Western_coastal_annual_rain_rcp85"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see annual runoff ranges."),
                                                                   imageOutput("Western_coastal_annual_runoff_rcp85"),
                                                                   br(),br(), br(),
                                                                   tags$blockquote("and finally 25-year/24-hour design storm intensity"),
                                                                   imageOutput("Western_coastal_storm_rcp85")
                                                                   ),

                                                          tabPanel("Seasonal", 
                                                                   tags$blockquote("Below we see seasonal precipitation ranges."),
                                                                   imageOutput("Western_coastal_seasonal_rain_rcp85"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see seasonal runoff ranges."),
                                                                   imageOutput("Western_coastal_seasonal_runoff_rcp85")
                                                                   ),

                                                          tabPanel("Monthly", 
                                                                   tags$blockquote("Below we see monthly precipitation ranges."),
                                                                   imageOutput("Western_coastal_monthly_rain_rcp85"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see monthly runoff ranges."),
                                                                   imageOutput("Western_coastal_monthly_runoff_rcp85")
                                                                   ),
                                                          width = 12
                                                          )
                                                  )
                                          ),
                                 ####### Western Coastal 85 END

                                 ####### Cascade foothills 85 start
                                 tabPanel("Cascade foothills",
                                          fluidRow(tabBox(tabPanel("Annual", 
                                                                   tags$blockquote("Below we see annual precipitation ranges."),
                                                                   imageOutput("Cascade_foothills_annual_rain_rcp85"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see annual runoff ranges."),
                                                                   imageOutput("Cascade_foothills_annual_runoff_rcp85"),
                                                                   br(),br(), br(),
                                                                   tags$blockquote("and finally 25-year/24-hour design storm intensity"),
                                                                   imageOutput("Cascade_foothills_storm_rcp85")
                                                                   ),

                                                          tabPanel("Seasonal", 
                                                                   tags$blockquote("Below we see seasonal precipitation ranges."),
                                                                   imageOutput("Cascade_foothills_seasonal_rain_rcp85"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see seasonal runoff ranges."),
                                                                   imageOutput("Cascade_foothills_seasonal_runoff_rcp85")
                                                                   ),

                                                          tabPanel("Monthly", 
                                                                   tags$blockquote("Below we see monthly precipitation ranges."),
                                                                   imageOutput("Cascade_foothills_monthly_rain_rcp85"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see monthly runoff ranges."),
                                                                   imageOutput("Cascade_foothills_monthly_runoff_rcp85")
                                                                   ),
                                                          width = 12
                                                          )
                                                  )
                                          ),
                                 ####### Cascade foothills 85 END

                                 ####### Northwest Cascades 85 start
                                 tabPanel("Northwest Cascades",
                                          fluidRow(tabBox(
                                                          tabPanel("Annual", 
                                                                   tags$blockquote("Below we see annual precipitation ranges."),
                                                                   imageOutput("Northwest_Cascades_annual_rain_rcp85"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see annual runoff ranges."),
                                                                   imageOutput("Northwest_Cascades_annual_runoff_rcp85"),
                                                                   br(),br(), br(),
                                                                   tags$blockquote("and finally 25-year/24-hour design storm intensity"),
                                                                   imageOutput("Northwest_Cascades_storm_rcp85")
                                                                   ), 
                                                          tabPanel("Seasonal", 
                                                                   tags$blockquote("Below we see seasonal precipitation ranges."),
                                                                   imageOutput("Northwest_Cascades_seasonal_rain_rcp85"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see seasonal runoff ranges."),
                                                                   imageOutput("Northwest_Cascades_seasonal_runoff_rcp85")
                                                                   ),

                                                          tabPanel("Monthly", 
                                                                   tags$blockquote("Below we see monthly precipitation ranges."),
                                                                   imageOutput("Northwest_Cascades_monthly_rain_rcp85"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see monthly runoff ranges."),
                                                                   imageOutput("Northwest_Cascades_monthly_runoff_rcp85")
                                                                   ),
                                                           width = 12
                                                           )
                                                   )
                                          ),
                                 ####### Northwest Cascades 85 END

                                 ####### Northcentral Cascades 85 start
                                 tabPanel("Northcentral Cascades",
                                          fluidRow(tabBox(tabPanel("Annual", 
                                                                   tags$blockquote("Below we see annual precipitation ranges."),
                                                                   imageOutput("Northcentral_Cascades_annual_rain_rcp85"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see annual runoff ranges."),
                                                                   imageOutput("Northcentral_Cascades_annual_runoff_rcp85"),
                                                                   br(),br(), br(),
                                                                   tags$blockquote("and finally 25-year/24-hour design storm intensity"),
                                                                   imageOutput("Northcentral_Cascades_storm_rcp85")
                                                                   ),
                                                          tabPanel("Seasonal", 
                                                                   tags$blockquote("Below we see seasonal precipitation ranges."),
                                                                   imageOutput("Northcentral_Cascades_seasonal_rain_rcp85"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see seasonal runoff ranges."),
                                                                   imageOutput("Northcentral_Cascades_seasonal_runoff_rcp85")
                                                                   ),

                                                          tabPanel("Monthly", 
                                                                   tags$blockquote("Below we see monthly precipitation ranges."),
                                                                   imageOutput("Northcentral_Cascades_monthly_rain_rcp85"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see monthly runoff ranges."),
                                                                   imageOutput("Northcentral_Cascades_monthly_runoff_rcp85")
                                                                   ),
                                                           width = 12
                                                           )
                                                   )
                                          ),
                                 ####### Northcentral Cascades 85 END

                                 ####### Northeast Cascades 85 start
                                 tabPanel("Northeast Cascades",
                                          fluidRow(tabBox(tabPanel("Annual", 
                                                                   tags$blockquote("Below we see annual precipitation ranges."),
                                                                   imageOutput("Northeast_Cascades_annual_rain_rcp85"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see annual runoff ranges."),
                                                                   imageOutput("Northeast_Cascades_annual_runoff_rcp85"),
                                                                   br(),br(), br(),
                                                                   tags$blockquote("and finally 25-year/24-hour design storm intensity"),
                                                                   imageOutput("Northeast_Cascades_storm_rcp85")
                                                                   ),
                                                          tabPanel("Seasonal", 
                                                                   tags$blockquote("Below we see seasonal precipitation ranges."),
                                                                   imageOutput("Northeast_Cascades_seasonal_rain_rcp85"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see seasonal runoff ranges."),
                                                                   imageOutput("Northeast_Cascades_seasonal_runoff_rcp85")
                                                                   ),

                                                          tabPanel("Monthly", 
                                                                   tags$blockquote("Below we see monthly precipitation ranges."),
                                                                   imageOutput("Northeast_Cascades_monthly_rain_rcp85"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see monthly runoff ranges."),
                                                                   imageOutput("Northeast_Cascades_monthly_runoff_rcp85")
                                                                   ),
                                                           width = 12
                                                           )
                                                   )
                                          ),
                                 ####### Northeast Cascades 85 END
                                 #######
                                 #######
                                 #######     RCP 4.5
                                 #######
                                 #######
                                 HTML("<b>RCP 4.5</b>"),
                                 ####### Western Coastal 45 start
                                 tabPanel("Western coastal",
                                          fluidRow(tabBox(tabPanel("Annual", 
                                                                   tags$blockquote("Below we see annual precipitation ranges."),
                                                                   imageOutput("Western_coastal_annual_rain_rcp45"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see annual runoff ranges."),
                                                                   imageOutput("Western_coastal_annual_runoff_rcp45"),
                                                                   br(),br(), br(),
                                                                   tags$blockquote("and finally 25-year/24-hour design storm intensity"),
                                                                   imageOutput("Western_coastal_storm_rcp45")
                                                                   ),
                                                          tabPanel("Seasonal", 
                                                                   tags$blockquote("Below we see seasonal precipitation ranges."),
                                                                   imageOutput("Western_coastal_seasonal_rain_rcp45"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see seasonal runoff ranges."),
                                                                   imageOutput("Western_coastal_seasonal_runoff_rcp45")
                                                                   ),

                                                          tabPanel("Monthly", 
                                                                   tags$blockquote("Below we see monthly precipitation ranges."),
                                                                   imageOutput("Western_coastal_monthly_rain_rcp45"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see monthly runoff ranges."),
                                                                   imageOutput("Western_coastal_monthly_runoff_rcp45")
                                                                   ),
                                                          width = 12
                                                          )
                                                  )
                                          ),
                                 ####### Western Coastal 45 END

                                 ####### Cascade foothills 45 start
                                 tabPanel("Cascade foothills",
                                          fluidRow(tabBox(tabPanel("Annual", 
                                                                   tags$blockquote("Below we see annual precipitation ranges."),
                                                                   imageOutput("Cascade_foothills_annual_rain_rcp45"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see annual runoff ranges."),
                                                                   imageOutput("Cascade_foothills_annual_runoff_rcp45"),
                                                                   br(),br(), br(),
                                                                   tags$blockquote("and finally 25-year/24-hour design storm intensity"),
                                                                   imageOutput("Cascade_foothills_storm_rcp45")
                                                                   ),
                                                          tabPanel("Seasonal", 
                                                                   tags$blockquote("Below we see seasonal precipitation ranges."),
                                                                   imageOutput("Cascade_foothills_seasonal_rain_rcp45"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see seasonal runoff ranges."),
                                                                   imageOutput("Cascade_foothills_seasonal_runoff_rcp45")
                                                                   ),

                                                          tabPanel("Monthly", 
                                                                   tags$blockquote("Below we see monthly precipitation ranges."),
                                                                   imageOutput("Cascade_foothills_monthly_rain_rcp45"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see monthly runoff ranges."),
                                                                   imageOutput("Cascade_foothills_monthly_runoff_rcp45")
                                                                   ),
                                                          width = 12
                                                          )
                                                  )
                                          ),
                                 ####### Cascade foothills 45 END

                                 ####### Northwest Cascades 45 start
                                 tabPanel("Northwest Cascades",
                                          fluidRow(tabBox(tabPanel("Annual", 
                                                                   tags$blockquote("Below we see annual precipitation ranges."),
                                                                   imageOutput("Northwest_Cascades_annual_rain_rcp45"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see annual runoff ranges."),
                                                                   imageOutput("Northwest_Cascades_annual_runoff_rcp45"),
                                                                   br(),br(), br(),
                                                                   tags$blockquote("and finally 25-year/24-hour design storm intensity"),
                                                                   imageOutput("Northwest_Cascades_storm_rcp45")
                                                                   ),
                                                          tabPanel("Seasonal", 
                                                                   tags$blockquote("Below we see seasonal precipitation ranges."),
                                                                   imageOutput("Northwest_Cascades_seasonal_rain_rcp45"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see seasonal runoff ranges."),
                                                                   imageOutput("Northwest_Cascades_seasonal_runoff_rcp45")
                                                                   ),

                                                           tabPanel("Monthly", 
                                                                   tags$blockquote("Below we see monthly precipitation ranges."),
                                                                   imageOutput("Northwest_Cascades_monthly_rain_rcp45"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see monthly runoff ranges."),
                                                                   imageOutput("Northwest_Cascades_monthly_runoff_rcp45")
                                                                   ),
                                                           width = 12
                                                           )
                                                   )
                                          ),
                                 ####### Northwest Cascades 45 END

                                 ####### Northcentral Cascades 45 start
                                 tabPanel("Northcentral Cascades",
                                          fluidRow(tabBox(tabPanel("Annual", 
                                                                   tags$blockquote("Below we see annual precipitation ranges."),
                                                                   imageOutput("Northcentral_Cascades_annual_rain_rcp45"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see annual runoff ranges."),
                                                                   imageOutput("Northcentral_Cascades_annual_runoff_rcp45"),
                                                                   br(),br(), br(),
                                                                   tags$blockquote("and finally 25-year/24-hour design storm intensity"),
                                                                   imageOutput("Northcentral_Cascades_storm_rcp45")
                                                                   ),

                                                          tabPanel("Seasonal", 
                                                                   tags$blockquote("Below we see seasonal precipitation ranges."),
                                                                   imageOutput("Northcentral_Cascades_seasonal_rain_rcp45"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see seasonal runoff ranges."),
                                                                   imageOutput("Northcentral_Cascades_seasonal_runoff_rcp45")
                                                                   ),

                                                           tabPanel("Monthly", 
                                                                   tags$blockquote("Below we see monthly precipitation ranges."),
                                                                   imageOutput("Northcentral_Cascades_monthly_rain_rcp45"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see monthly runoff ranges."),
                                                                   imageOutput("Northcentral_Cascades_monthly_runoff_rcp45")
                                                                   ),
                                                           width = 12
                                                           )
                                                   )
                                          ),
                                 ####### Northcentral Cascades 45 END

                                 ####### Northeast Cascades 45 start
                                 tabPanel("Northeast Cascades",
                                          fluidRow(tabBox(tabPanel("Annual", 
                                                                   tags$blockquote("Below we see annual precipitation ranges."),
                                                                   imageOutput("Northeast_Cascades_annual_rain_rcp45"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see annual runoff ranges."),
                                                                   imageOutput("Northeast_Cascades_annual_runoff_rcp45"),
                                                                   br(),br(), br(),
                                                                   tags$blockquote("and finally 25-year/24-hour design storm intensity"),
                                                                   imageOutput("Northeast_Cascades_storm_rcp45")
                                                                   ),

                                                          tabPanel("Seasonal", 
                                                                   tags$blockquote("Below we see seasonal precipitation ranges."),
                                                                   imageOutput("Northeast_Cascades_seasonal_rain_rcp45"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see seasonal runoff ranges."),
                                                                   imageOutput("Northeast_Cascades_seasonal_runoff_rcp45")
                                                                   ),

                                                          tabPanel("Monthly", 
                                                                   tags$blockquote("Below we see monthly precipitation ranges."),
                                                                   imageOutput("Northeast_Cascades_monthly_rain_rcp45"),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   br(),br(), br(),br(),br(), br(),
                                                                   tags$blockquote("Below we see monthly runoff ranges."),
                                                                   imageOutput("Northeast_Cascades_monthly_runoff_rcp45")
                                                                   ),
                                                           width = 12
                                                           )
                                                   )
                                          ),
                                 ####### Northeast Cascades 45 END


                                 widths = c(2,10)
                                 )
                   ),
           #
           ############## Regional Plots END
           #
           #
           ############## Bloom, CP, frost Map Front Page start
           #
           # tabPanel(tags$b("Some Insight"),
           #          fluidPage( id = "nav", inverse=FALSE, 
           #                     fluid=FALSE, title="Tool",
           #                     div( class="outer",
           #                          tags$head(includeCSS("styles.css"),
           #                                    includeScript("gomap.js")
           #                                    ),
           #                          leafletOutput("lagoon_map", 
           #                                         width="100%", 
           #                                         height="100%")
           #                        )
           #                    ),
           #          fluidPage(bsModal(
           #                             id="lagoon_graphs",
           #                             trigger = NULL,
           #                             title = "",
           #                             size = "large",                                       
           #                             dashboardPage( dashboardHeader(title = "Plots"),
           #                                            dashboardSidebar(
           #                                                             radioButtons(inputId = "em_scenario",
           #                                                                          label = tags$b("Scenario"),
           #                                                                          choices = c("RCP 8.5" = "RCP 8.5",
           #                                                                                      "RCP 4.5" = "RCP 4.5"), 
           #                                                                          selected = "RCP 8.5"),
           #                                                             radioButtons(inputId="lagoon_plot_fruit_type", 
           #                                                                          label = h4("Info. type"), 
           #                                                                          choices = list("Precipitation" = "precip", 
           #                                                                                         "Runoff" = "runoff", 
           #                                                                                         "Design storm" = "design_storm"),
           #                                                                          selected = "precip")
           #                                                             ),
           #                                            #####################
           #                                            #
           #                                            # End of side bar of dashboard of analog maps
           #                                            #
           #                                            #####################
           #                                            dashboardBody(
           #                                                           tags$head(tags$style(HTML('.content-wrapper, 
           #                                                                                     .right-side {
           #                                                                                     background-color: #252d38;
           #                                                                                       }'
           #                                                                                    ))),
           #                                                           # plotOutput("lagoon_plot")
           #                                                           fluidPage(fluidRow(column(2, offset = 0, 
           #                                                                                     plotOutput("lagoon_plot", height = 550)))
           #                                                                    )

           #                                                          )
           #                                          )
           #                            )
           #                    )
           #          )
           tabPanel(tags$b("Maps"),
                    navlistPanel(
                                 ####### 
                                 tabPanel("RCP 8.5",
                                          fluidRow(tabBox(
                                                          tabPanel("Desgin Storm Intensity", 
                                                                   tags$h5("Below we see the map ", 
                                                                           "of storm design intensity where the ",
                                                                           "color scales are different accross ",
                                                                           "time periods to maximize visual ", 
                                                                           " distinction between color of different points:"),
                                                                   imageOutput("storm_diff_85_16inch_diffGradient"),
                                                                   tags$h5(
                                                                    "Below we see the map ", 
                                                                                   "of storm design intensity--",
                                                                                   "the same plot as above--",
                                                                                   "with color scales being identical ",
                                                                                   "across time periods in order for ",
                                                                                   "see the changes accross time windows:"
                                                                                   ),
                                                                   imageOutput("storm_diff_85_16inch_idenGradient"),
                                                                   br(),br()                                                                   
                                                                   ),
                                                          tabPanel("Precipitation", 
                                                                   tags$h5("Below we see the map ", 
                                                                           "of precipitation differences where the ",
                                                                           "color scales are different accross ",
                                                                           "time periods to maximize visual ", 
                                                                           " distinction between color of different points:"),
                                                                   imageOutput("precip_diff_85_16_diffGradient"),
                                                                   tags$h5(
                                                                    "Below we see the map ", 
                                                                                   "of precipitation differences--",
                                                                                   "the same plot as above--",
                                                                                   "with color scales being identical ",
                                                                                   "across time periods in order for ",
                                                                                   "see the changes accross time windows:"
                                                                                   ),
                                                                   imageOutput("precip_diff_85_16_idenGradient"),
                                                                   br(),br()
                                                                   ),
                                                          tabPanel("Runoff", 
                                                                   tags$h5("Below we see the map ", 
                                                                           "of runoff differences where the ",
                                                                           "color scales are different accross ",
                                                                           "time periods to maximize visual ", 
                                                                           " distinction between color of different points:"),
                                                                   imageOutput("runoff_diff_85_16_diffGradient"),
                                                                   tags$h5(
                                                                    "Below we see the map ", 
                                                                                   "of runoff differences--",
                                                                                   "the same plot as above--",
                                                                                   "with color scales being identical ",
                                                                                   "across time periods in order for ",
                                                                                   "see the changes accross time windows:"
                                                                                   ),
                                                                   imageOutput("runoff_diff_85_16_idenGradient"),
                                                                   br(),br()
                                                                   ),
                                                          width = 12
                                                          )
                                                  )
                                          ),
                                 tabPanel("RCP 4.5",
                                          fluidRow(tabBox(
                                                          tabPanel("Desgin Storm Intensity", 
                                                                   tags$h5("Below we see the map ", 
                                                                                   "of storm design intensity where the ",
                                                                                   "color scales are different accross ",
                                                                                   "time periods to maximize visual ", 
                                                                                   " distinction between color of different points:"),
                                                                   imageOutput("storm_diff_45_16inch_diffGradient"),
                                                                   tags$h5(
                                                                    "Below we see the map ", 
                                                                                   "of storm design intensity--",
                                                                                   "the same plot as above--",
                                                                                   "with color scales being identical ",
                                                                                   "across time periods in order for ",
                                                                                   "see the changes accross time windows:"
                                                                                   ),
                                                                   imageOutput("storm_diff_45_16inch_idenGradient"),
                                                                   br(),br()
                                                                   ),
                                                          tabPanel("Precipitation", 
                                                                   tags$h5("Below we see the map ", 
                                                                           "of precipitation differences where the ",
                                                                           "color scales are different accross ",
                                                                           "time periods to maximize visual ", 
                                                                           " distinction between color of different points:"),
                                                                   imageOutput("precip_diff_45_16_diffGradient"),
                                                                   tags$h5(
                                                                    "Below we see the map ", 
                                                                                   "of precipitation differences--",
                                                                                   "the same plot as above--",
                                                                                   "with color scales being identical ",
                                                                                   "across time periods in order for ",
                                                                                   "see the changes accross time windows:"
                                                                                   ),
                                                                   imageOutput("precip_diff_45_16_idenGradient"),
                                                                   br(),br()
                                                                   ),
                                                          tabPanel("Runoff", 
                                                                   tags$h5("Below we see the map ", 
                                                                           "of runoff differences where the ",
                                                                           "color scales are different accross ",
                                                                           "time periods to maximize visual ", 
                                                                           " distinction between color of different points:"),
                                                                   imageOutput("runoff_diff_45_16_diffGradient"),
                                                                   tags$h5(
                                                                    "Below we see the map ", 
                                                                                   "of runoff differences--",
                                                                                   "the same plot as above--",
                                                                                   "with color scales being identical ",
                                                                                   "across time periods in order for ",
                                                                                   "see the changes accross time windows:"
                                                                                   ),
                                                                   imageOutput("runoff_diff_45_16_idenGradient"),
                                                                   br(),br()
                                                                   ),
                                                           width = 12
                                                          )
                                                  )
                                          ),
                                 widths = c(2,10)
                                 )
                   )

           # tabPanel(tags$b("Maps"),
           #          navlistPanel(
           #                       ####### Subregion Groups start
           #                       tabPanel("RCP 8.5", 
           #                                "Below we see the map of storm design intensity.",
           #                                br(),br(),
           #                                imageOutput("storm_diff_85_14inch_diffGradient"),
           #                                "We use statistical methods ", 
           #                                        "to group the regions into two", 
           #                                        " areas based on average historical", 
           #                                        " growing degree day accumulation.",
           #                                        " The warmer areas have an avg.", 
           #                                        " annual historical GDD of XX ", 
           #                                        "degree day F (provide a range) ", 
           #                                        "and the cooler areas have a ", 
           #                                        "historical GDD of YY degree ", 
           #                                        "days F (provide a range)."
           #                                ),
                                
           #                       ####### No. of generations 85 start
           #                       tabPanel("RCP 4.5", 
           #                                tags$b("Below we see the map of storm design intensity."),
           #                                br(),br(),
           #                                imageOutput("storm_diff_45_14inch_diffGradient"),
           #                                tags$h3("We use statistical methods ", 
           #                                        "to group the regions into two", 
           #                                        " areas based on average historical", 
           #                                        " growing degree day accumulation.",
           #                                        " The warmer areas have an avg.", 
           #                                        " annual historical GDD of XX ", 
           #                                        "degree day F (provide a range) ", 
           #                                        "and the cooler areas have a ", 
           #                                        "historical GDD of YY degree ", 
           #                                        "days F (provide a range).")
           #                                ),
           #                       ####### No. of generations 85 END
           #                       widths = c(2,10)
           #                       )
           #         )

       )
