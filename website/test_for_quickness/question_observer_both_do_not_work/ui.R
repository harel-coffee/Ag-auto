navbarPage(title = div(""),
           id="nav", 
           windowTitle = "Q",
           #
           tabPanel(tags$b("Q"),
                    div(class="outer",
                        tags$head(includeCSS("styles.css")),
                        leafletOutput("a_map", width="100%", height="100%"),
                        absolutePanel(id = "controls", 
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
                                      
                                      selectInput(inputId = "water_source_type", 
                                                  label = "Water Resource", 
                                                  choices = c("Surface Water" = "surfaceWater",
                                                              "Ground Water" = "groundwater",
                                                              "Both" = "both_water_resource"), 
                                                  selected = "both_water_resource")
                                      
                        )
                    )
           )
           
)


