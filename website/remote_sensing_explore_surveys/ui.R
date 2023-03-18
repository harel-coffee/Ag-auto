# Per_MS Meeting

tile_vars <- c( "Sattelite", "Esri.WorldStreetMap", "OpenTopoMap")
year_vars <- c(2015, 2016, 2017, 2018)
field_vars <- c("All fields", "Double-cropped potentials")

navbarPage("Explore WSDA surveys", id="nav",

  tabPanel("Map of fields",
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

                    h2("Your Options"),
                    selectInput(label = h4(tags$b("Tile choices")), 
                                "map_tile_", 
                                choices=tile_vars,
                                selected = tile_vars[1]),

                    selectInput(label = h4(tags$b("Survey year")), 
                                "Survey_Year" , 
                                choices = year_vars,
                                selected = year_vars[3]
                                ),

                    selectInput(label = h4(tags$b("Field type")), 
                                "Field_type" , 
                                choices = field_vars,
                                selected = field_vars[2]
                                )
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


