library(scales)
library(lattice)
library(jsonlite)
library(raster)
library(data.table)
library(shiny)
library(shinydashboard)
library(shinyBS)
library(maps)
library(rgdal)   
library(sp)       
library(leaflet)  
library(dplyr)    
library(ggplot2) 
library(reshape2)
library(RColorBrewer)

######################################################

d <- paste0("path_to_file", 
            "water_right_attributes.rds")

spatial_wtr_right <- readRDS(d) %>% data.table()
spatial_wtr_right$color <- "#ffff00"

all_basins <- sort(unique(spatial_wtr_right$county_type))
state.name <- c("Washington", "Oregon")

subbasins <- c("Ahtanum Creek", 
               "Lmumu-Burbank",
               "Lower Yakima tributaries",
               "tributaries", 
               "Satus Creek",
               "Taneum-Manastash",
               "Toppenish Creek",
               "Wilson-Cherry")


#############
#############
############# server
#############
#############

shinyServer(function(input, output, session) {
  output$water_right_map <- renderLeaflet({
    target_date <- as.Date(input$cut_date)

    water_resource <- input$water_source_type
    if (water_resource == "surfaceWater") {
      curr_spatial <- spatial_wtr_right %>%
                      filter(WaRecRCWCl == "surfaceWater")%>%
                      data.table()

     } else if (water_resource == "groundwater"){
      curr_spatial <- spatial_wtr_right %>%
                      filter(WaRecRCWCl == "groundwater")%>%
                      data.table()
       } else {
          curr_spatial <- spatial_wtr_right
    }
  
    curr_spatial <- curr_spatial %>% 
                  filter(county_type == input$countyType_id) %>% 
                  data.table()

    observeEvent(input$countyType_id, {
        #
        # from 
        # https://shiny.rstudio.com/reference/shiny/latest/updateSelectInput.html
        #
        subbasins <- sort(unique(curr_spatial$subbasin))
     
        # Can also set the label and select items
        updateSelectInput(session,
                          inputId = 'subbasins_id',
                          # label = '2. Select subbasins',
                          choices = subbasins,
                          selected = head(subbasins, 1)
                          )
    })

    observeEvent(input$subbasins_id, {

      print ("subbasins_id changed now and updated subbasins are")
      curr_spatial <- curr_spatial %>% 
                      filter(subbasin %in% input$subbasins_id) %>% 
                      data.table()

      print (unique(curr_spatial$subbasin))
      print("_______++++++++_________________")

    })

    curr_spatial[, colorr := ifelse(right_date < target_date, 
                                    "#FF3333", "#0080FF")]
    
    print("_______++++++++_________________")
    print ('after observe (unique(curr_spatial$subbasin))')
    print (unique(curr_spatial$subbasin))
    print("_______++++++++_________________")

    leaflet() %>%
    addTiles(urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
             attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
             layerId = "Satellite",
             options= providerTileOptions(opacity = 0.9)) %>%
    setView(lat = 47, lng = -120, zoom = 7) %>%
    addCircleMarkers(data = curr_spatial, 
                     lng = ~long, lat = ~lat,
                     label = ~ popup,
                     # layerId = ~ location,
                     radius = 3,
                     color = ~ colorr,
                     stroke  = FALSE,
                     fillOpacity = .95 
                      )
 
  })

})



##############
##############
# UI

navbarPage(title = div(""),
           id="nav", 
           windowTitle = "Water Right",
           #
           tabPanel(tags$b("Water Right"),
                    div(class="outer",
                        tags$head(includeCSS("styles.css")),
                        leafletOutput("water_right_map", width="100%", height="100%"),
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
                                                  label = "0. Water Resource", 
                                                  choices = c("Surface Water" = "surfaceWater",
                                                              "Ground Water" = "groundwater",
                                                              "Both" = "both_water_resource"), 
                                                  selected = "both_water_resource"),
                                      
                                      selectInput(inputId = 'countyType_id',
                                                  label = '1. Select a basin',
                                                  choices = all_basins
                                                  ),

                                      selectizeInput(inputId = 'subbasins_id',
                                                     label = '2. Select subbasins', 
                                                     choices = subbasins, 
                                                     selected = head(subbasins, 1),
                                                     multiple = TRUE
                                                    )
                                   )
                    )
           )
           
)


