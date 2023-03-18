# stats dashboard
source("global.R")
####  Header  ####
map_types <- c( "Sattelite", "World Street", "Open Topo", "Default")
var_types <- c('CDI', 'no switch rate', "DDₕ", 'switch day', 'avg endo slope', 'avg eco slope',
               'max biweekly avg endo slope', 'max biweekly avg eco slope')
header <- dashboardHeader(
  title = tags$div(tags$img(src='WSU_logo.png', height = 30), "CAHNRS"))

####  Sidebar  ####
sidebar <- dashboardSidebar(
  collapsed = TRUE,
  sidebarMenu(menuItem("stats", tabName = "hard_map", icon = icon("tint")))
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
                    top = 60, bottom = "auto",
                    width = "auto", height = "auto",
                    selectInput("model_", label="model", choices=models, selected=models[1]),
                    selectInput("scenario_", label="scenario", choices=scenario_types, selected = scenario_types[2]),
                    selectInput("time_frame_", label="time frame", choices=time_frames, selected = time_frames[1]),
                    selectInput("variable_", label="statistics", choices=var_types, selected=var_types[1]),
                    selectInput("map_tile_", label="map", choices=map_types, selected = map_types[1])
                ),
                # absolutePanel(id = "controls",
                #     class = "panel panel-default", fixed = TRUE,
                #     draggable = TRUE,
                #     left = "auto", right = 20,
                #     top = 40, bottom = "auto",
                #     width = "auto", height = "auto",
                #     selectInput("model_", label="",
                #                 choices=models,
                #                 selected=models[1])
                #   ),
                # absolutePanel(id = "controls",
                #     class = "panel panel-default", fixed = TRUE,
                #     draggable = TRUE,
                #     left = "auto", right = 20,
                #     top = 40, bottom = "auto",
                #     width = "auto", height = "auto",
                #     #uiOutput(outputId = "scenario_")
                #     selectInput("scenario_", label="", choices=scenario_types, selected = scenario_types[2])
                #   ),
                # absolutePanel(id = "controls",
                #               class = "panel panel-default", fixed = TRUE,
                #               draggable = TRUE,
                #               left = "auto", right = 20,
                #               top = 40, bottom = "auto",
                #               width = "auto", height = "auto",
                #               #uiOutput(outputId = "time_frame_")
                #               selectInput("time_frame_", label="", choices=time_frames, selected = time_frames[1])
                # ),
                # absolutePanel(id = "controls",
                #     class = "panel panel-default", fixed = TRUE,
                #     draggable = TRUE,
                #     left = "auto", right = 20,
                #     top = 40, bottom = "auto",
                #     width = "auto", height = "auto",
                #     selectInput("variable_", label="",
                #                 choices=var_types,
                #                 selected=var_types[1])
                #   )
                )
        )
    ),
  
  # plot Modal
  bsModal(title = 'Stats', 
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
  title="Stats",
  header,
  sidebar,
  body
)


