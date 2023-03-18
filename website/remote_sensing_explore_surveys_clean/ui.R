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
                    radioButtons(inputId = "Survey_Year",
                                 label = tags$b("Survey year"),
                                 choices = c("2015" = "2015",
                                             "2016" = "2016",
                                             "2017" = "2017",
                                             "2018" = "2018"
                                             ), 
                                 selected = "2017"),

                    # selectInput(label = h4(tags$b("Survey year")), 
                    #             "Survey_Year" , 
                    #             choices = year_vars,
                    #             selected = year_vars[3]
                    #             ),

                    radioButtons(inputId = "Field_type",
                                 label = tags$b("Field type"),
                                 choices = c("Double-cropped potentials" = "Double-cropped potentials",
                                             "All fields" = "All fields"
                                             ), 
                                 selected = "Double-cropped potentials")

                    # selectInput(label = h4(tags$b("Field type")), 
                    #             "Field_type" , 
                    #             choices = field_vars,
                    #             selected = field_vars[2]
                    #             )
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


