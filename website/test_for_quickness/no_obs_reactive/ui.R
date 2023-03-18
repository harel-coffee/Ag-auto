# Water Rights

# library(leaflet)
# library(shinyBS)
# library(shiny)
# library(plotly)
# library(shinydashboard)

navbarPage(title = div(""),
           id="nav", 
           windowTitle = "Water Right",
           #
           tabPanel(tags$b("Water Right"),
                    div(class="outer",
                        tags$head(includeCSS("styles.css")),
                        leafletOutput("water_right_map", width="100%", height="100%"),
                        absolutePanel(id = "controlsss", 
                                      class = "panel panel-default", 
                                      fixed = TRUE,
                                      draggable = TRUE, 
                                      top = 60, right = 20,
                                      left = "auto", bottom = "auto",
                                      width = 330, height = "auto",
                                      
                                      h4("Earlier in red, later in blue"),
                                      sliderInput(inputId = "cut_date",
                                                  label = "Dates:",
                                                  min = as.Date("1800-01-01","%Y-%m-%d"),
                                                  max = as.Date("2015-12-30","%Y-%m-%d"),
                                                  value=as.Date("1800-01-01"),
                                                  timeFormat="%Y-%m-%d"),

                                      selectInput(inputId = "purpose_id", 
                                                  label = "0. Purpose", 
                                                  choices = c("Points of Diversion" = "pod",
                                                              "Place of Use" = "pou"), 
                                                  selected = "pod"),

                                      selectInput(inputId = "water_source_type", 
                                                  label = "1. Water Resource", 
                                                  choices = c("Surface Water" = "surfaceWater",
                                                              "Ground Water" = "groundwater",
                                                              "Both" = "both_water_resource"), 
                                                  selected = "both_water_resource"),
                                      
                                      selectInput(inputId = 'countyType_id',
                                                  label = '2. Select a basin',
                                                  choices = all_basins
                                                  ),

                                      selectInput(inputId = 'subbasins_id',
                                                  label = '3. Select creek',
                                                  choices = subbasins, 
                                                  selected = subbasins,
                                                  multiple = TRUE,
                                                  selectize=TRUE
                                                  )
                                   )
                    )
           )
           
)


