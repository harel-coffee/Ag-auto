# Per_MS Meeting

vars <- c( "Sattelite", "Esri.WorldStreetMap", "OpenTopoMap")

navbarPage("Pre-MS", id="nav",

  tabPanel("Map of polygons of double crops filtered by Notes and RotationCrop",
    div(class="outer",

      tags$head(
        # Include CSS
        includeCSS("styles.css"),
        includeScript("gomap.js")
      ),

      # If not using custom CSS, set height of 
      # leafletOutput to a number instead of percent
      leafletOutput("mymap", width="100%", height="100%"),
      absolutePanel(id = "controls", 
                    class = "panel panel-default", 
                    fixed = TRUE,
                    draggable = TRUE, 
                    top = 60, right = 20,
                    left = "auto", bottom = "auto",
                    width = 330, height = "auto",

                    h2("Tile Choice"),
                    selectInput("map_tile_", label=" ", 
                                choices=vars,
                                selected = vars[1])
                  ),

      tags$div(id="cite",
               'Data compiled for ', 
               tags$em(paste0('Coming Apart: The State of White America, 1960–2010', 
                       ' by Charles Murray (Crown Forum, 2012).')
                      )
               )
    )
  )
)


