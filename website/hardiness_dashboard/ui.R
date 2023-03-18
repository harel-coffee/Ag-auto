# hardiness dashboard

####  Header  ####
vars <- c( "Sattelite", "World Street", "Open Topo", "Default")

header <- dashboardHeader(
  title = tags$div(tags$img(src='WSU_logo.png', height = 30), "CAHNRS"))

####  Sidebar  ####
sidebar <- dashboardSidebar(
  collapsed = TRUE,
  sidebarMenu(
    menuItem("Cold Hardiness", 
             tabName = "hard_map", 
             icon = icon("tint"))
  )
)
####################################
####        ########################
####  Body  ########################
####        ########################
####################################
body <- dashboardBody(
  # tags$head(tags$style(
  #   HTML('.wrapper {height: auto !important; position:relative; overflow-x:hidden; overflow-y:hidden}')
  # )),
  ##################
  ####         #####
  ####   CSS   #####
  ####         #####
  ##################
  tags$head(
    tags$link(rel = "stylesheet", 
              type = "text/css", 
              href = "custom.css")),
  
  ## Tabs ##
  tabItems(
    
    # Home Tab #
    tabItem(tabName = "home"),
    tabItem(tabName = "about", fluidPage(fluidRow(column(8, offset = 2, includeMarkdown("pages/about.Rmd"))))),
    
    # bloom_vs_CP tab
    tabItem(tabName = "hard_map",
            box(id = "hard_box_id", width = NULL,
                #  Main Map  #
                tabPanel("Map",
                         div(class = "outer",
                             tags$style(type = "text/css", 
                                        "#hard_map {height: calc(100vh - 125px) !important;}"),
                             leafletOutput("hard_map")
                            )
                        ),
                absolutePanel(id = "controls", 
                    class = "panel panel-default", fixed = TRUE,
                    draggable = TRUE, 
                    left = "auto", right = 20, 
                    top = 40, bottom = "auto",
                    width = "auto", height = "auto",

                    # h4("Tile"),
                    selectInput("map_tile_", label="", 
                                choices=vars,
                                selected = vars[1])
                  )
                )
        )
    ),
  
  # plot Modal
  bsModal(title = "Cold Hardiness", 
          id = "hard_graphs", trigger = NULL, size = "large",
          fluidPage(fluidRow(
                              column(1, 
                                     offset = 2, 
                                     plotOutput("hard_plot", 
                                                height = 800,
                                                width= 800))
                             )
                    )
         )
  )

####  Combine Dashboard Elements  ####
dashboardPage(
  title="Cold Hardiness",
  header,
  sidebar,
  body
)


