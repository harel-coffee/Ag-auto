# Per_MS Meeting

library(shiny)
library(leaflet)

# Define server logic required to draw a histogram
shinyServer(function(input, output, session) {

  observe({
       if (input$map_tile_ == "Esri.WorldStreetMap"){
         output$mymap = renderLeaflet({ 
                                   leaflet() %>%
                                   # Esri.WorldStreetMap or OpenTopoMap
                                   addProviderTiles(providers$Esri.WorldStreetMap, 
                                                    options= providerTileOptions(opacity = 0.8))%>%
                                   setView(lat = 46.5, lng = -119, zoom = 7) %>%
                                   addPolygons(data = Min_sp,
                                               stroke = TRUE, 
                                               fillOpacity = 0.1, 
                                               smoothFactor = 0.9)%>% 
                                  # addMarkers(data = centroids)
                                  addCircleMarkers(data = centroids, 
                                                   lng = ~ longitude, 
                                                   lat = ~ latitude,
                                                   label = ~ location,
                                                   layerId = ~ location,
                                                   radius = 4,
                                                   # color = ~ pal(latitude),
                                                   stroke  = FALSE,
                                                   fillOpacity = .95)

                                    })
            } else if (input$map_tile_ == "OpenTopoMap"){
             output$mymap = renderLeaflet({ 
                            leaflet() %>%
                                             # Esri.WorldStreetMap or OpenTopoMap
                            addProviderTiles(providers$OpenTopoMap,
                                            options= providerTileOptions(opacity = 0.8))%>%
                            setView(lat = 46.5, lng = -119, zoom = 7) %>%
                            addPolygons(data = Min_sp,
                                        stroke = TRUE, 
                                        fillOpacity = 0.1, 
                                       smoothFactor = 0.9)%>% 
                            # addMarkers(data = centroids)
                            addCircleMarkers(data = centroids, 
                                             lng = ~ longitude, 
                                             lat = ~ latitude,
                                             label = ~ location,
                                             layerId = ~ location,
                                             radius = 4,
                                             # color = ~ pal(latitude),
                                             stroke  = FALSE,
                                             fillOpacity = .95)
                            })

            } else if (input$map_tile_ == "Sattelite"){
               output$mymap = renderLeaflet({ 
                                      leaflet() %>%
                                      addTiles(urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                                               attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
                                               layerId = "Satellite",
                                               options= providerTileOptions(opacity = 0.8)) %>%
                                      setView(lat = 46.5, lng = -119, zoom = 7) %>%
                                      addPolygons(data = Min_sp,
                                                  stroke = TRUE, 
                                                  fillOpacity = 0.1, 
                                                 smoothFactor = 0.9) %>% 
                                      # addMarkers(data = centroids)
                                      addCircleMarkers(data = centroids, 
                                                       lng = ~ longitude, 
                                                       lat = ~ latitude,
                                                       label = ~ location,
                                                       layerId = ~ location,
                                                       radius = 4,
                                                       # color = ~ pal(latitude),
                                                       stroke  = FALSE,
                                                       fillOpacity = .95)
                                            }) 

       }
  })
           
})



