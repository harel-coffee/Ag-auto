# stats dashboard
####  Header  ####
library(leaflet)
library(shinyBS)
library(shiny)
library(plotly)
library(shinydashboard)

navbarPage(
  title = div("",
                       img(src='csanr_logo.png', style='width:100px;height:35px;')
                       # img(src='WSU-DAS-log.png', style='width:100px;height:35px;'),
                       # img(src='NW-Climate-Hub-Logo.jpg', style='width:100px;height:35px;')
),
id="nav",
windowTitle = "grapevine model differences",
#
############## CM CDI START
#
tabPanel(tags$b("CDI"), value='CDI', 
         div(class="outer",
             tags$head(includeCSS("styles.css")),
             leafletOutput("map_CDI", width="100%", height="100%"),
             absolutePanel(id = "controls", class = "panel panel-default",
                           fixed = TRUE, draggable = TRUE,
                           top = 60, left = "auto",
                           right = 20, bottom = "auto",
                           width = 250, height = "auto",
                           h3(tags$b("CDI")),
                           #textOutput('test'),
                           gsub('"model"', '"model_cdi"', includeHTML("model.html")),
                            gsub('"scope"', '"scope_cdi"', includeHTML("scope.html")),
                           gsub('"DDh"', '"DDh_cdi"', includeHTML("DDh.html")),
                           gsub('"map_tile"', '"map_tile_cdi"', includeHTML("map_tile.html"))
                           
             )
         )
),
#
############## CM CDI END
#
############## CM DDh START
#
tabPanel(tags$b("heat degree-days"), value='DDh', 
         div(class="outer",
             tags$head(includeCSS("styles.css")),
             leafletOutput("map_DDh", width="100%", height="100%"),
             absolutePanel(id = "controls", class = "panel panel-default",
                           fixed = TRUE, draggable = TRUE,
                           top = 60, left = "auto",
                           right = 20, bottom = "auto",
                           width = 250, height = "auto",
                           h3(tags$b("heat degree-days")),
                           gsub('"scope"', '"scope_ddh"', includeHTML("scope.html")),
                           gsub('"model"', '"model_ddh"', includeHTML("model.html")),
                           gsub('"DDh"', '"DDh_ddh"', includeHTML("DDh.html")),
                           gsub('"map_tile"', '"map_tile_ddh"', includeHTML("map_tile.html"))
                           
             )
         )
),
#
############## CM DDh END
#
#
############## CM NO SWITCH START
#
tabPanel(tags$b("no switch"), value='no switch', 
         div(class="outer",
             tags$head(includeCSS("styles.css")),
             leafletOutput("map_no_switch", width="100%", height="100%"),
             absolutePanel(id = "controls", class = "panel panel-default",
                           fixed = TRUE, draggable = TRUE,
                           top = 60, left = "auto",
                           right = 20, bottom = "auto",
                           width = 250, height = "auto",
                           h3(tags$b("no switch proportion")),
                           gsub('"scope"', '"scope_no_switch"', includeHTML("scope.html")),
                           gsub('"model"', '"model_no_switch"', includeHTML("model.html")),
                           gsub('"DDh"', '"DDh_no_switch"', includeHTML("DDh.html")),
                           gsub('"map_tile"', '"map_tile_no_switch"', includeHTML("map_tile.html"))
                           
             )
         )
),
#
############## CM NO SWITCH END
#
#
############## CM SWITCH DAY START
#
tabPanel(tags$b("switch day"),value='switch day', 
         div(class="outer",
             tags$head(includeCSS("styles.css")),
             leafletOutput("map_switch_day", width="100%", height="100%"),
             absolutePanel(id = "controls", class = "panel panel-default",
                           fixed = TRUE, draggable = TRUE,
                           top = 60, left = "auto",
                           right = 20, bottom = "auto",
                           width = 250, height = "auto",
                           h3(tags$b("switch day")),
                           gsub('"scope"', '"scope_switch_day"', includeHTML("scope.html")),
                           gsub('"model"', '"model_switch_day"', includeHTML("model.html")),
                           gsub('"DDh"', '"DDh_switch_day"', includeHTML("DDh.html")),
                           gsub('"map_tile"', '"map_tile_switch_day"', includeHTML("map_tile.html"))
                           
             )
         )
),
#
############## CM SWITCH DAY END
#
#
############## CM SLOPE START
#
tabPanel(tags$b("de-/acclimation slopes"),value='slope', 
         div(class="outer",
             tags$head(includeCSS("styles.css")),
             leafletOutput("map_slope", width="100%", height="100%"),
             absolutePanel(id = "controls", class = "panel panel-default", 
                           fixed = TRUE, draggable = TRUE, 
                           top = 60, left = "auto", 
                           right = 20, bottom = "auto",
                           width = 250, height = "auto",
                           h3(tags$b("de-/acclimation slopes")),
                           radioButtons("acclim", 
                                        label = '',#h4(tags$b("de-/acclimation")),
                                        choices = list("acclimation" = "endo", 
                                                       "deacclimation" = "eco"),
                                        selected = "endo"#, inline = FALSE
                           ),
                           radioButtons("averaging", 
                                        label = h4(tags$b("averaging")),
                                        choices = list("mean" = "avg", 
                                                       "max biweekly mean" = "biweekly"),
                                        selected = "avg"#, inline = FALSE
                           ),
                           gsub('"scope"', '"scope_slope"', includeHTML("scope.html")),
                           gsub('"model"', '"model_slope"', includeHTML("model.html")),
                           gsub('"DDh"', '"DDh_slope"', includeHTML("DDh.html")),
                           gsub('"map_tile"', '"map_tile_slope"', includeHTML("map_tile.html"))
             )
         )
)
#
############## CM SLOPE END
#

)