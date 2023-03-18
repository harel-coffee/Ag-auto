shinyUI(
navbarPage("Climate Visualization Tool",
tabPanel("Home",
fluidPage(id="nav", inverse=FALSE, fluid=FALSE,
title="WSU Columbia River Basin Climate Tool",
div(class="outer",
tags$head(
# Include our custom CSS
includeCSS("style.css"),
includeScript("gomap.js"),
tags$style(type="text/css", "
#loadmessage {
position: fixed;
top: 0px;
left: 0px;
width: 100%;
padding: 5px 0px 50px 0px;
text-align: center;
font-weight: blod;
font-size: 100%;
color: #000000;
background-color: #CCFF66;
z-index: 105;
}
")
), 
leafletOutput("map", width="100%", height="100%"),
absolutePanel(id="menuPanel_main", draggable = TRUE, width=330, height= "auto", left="auto",
right=20, bottom="auto", top=60, fixed=TRUE, inverse=TRUE,
h2("Map Key"),
selectInput("tileSelect", "Map Overlay", c("Satellite","Topographic", "Basic")),
selectInput("cropSelect", "Locations with:", crop_names, selectize = FALSE, multiple = TRUE, selected=crop_names),
actionButton("cropSelectAll", "Select All")
)
),
bsModal(id="graphs", trigger=NULL, title="", size="large",
dashboardPage(
dashboardHeader(title = "Climate Visualization"),
dashboardSidebar(
sidebarMenu(
menuItem("Long Term Climate Change",  tabName = "longTerm"),
#menuItem("Short Term (Coming Soon)", tabName = "shortTerm_alt"),
#menuItem("Codling Moth (Coming Soon)",  tabName = "codlingMoth_alt")
radioButtons("mod", "Model", c(#"bcc-csm1-1" = "bcc-csm1-1", 
                               "BNU-ESM" = "BNU-ESM", 
                               "CanESM2" = "CanESM2", 
                               "CNRM-CM5" = "CNRM-CM5",
                               "GFDL-ESM2G" = "GFDL-ESM2G",
                               "GFDL-ESM2M" = "GFDL-ESM2M")
                          ), 
              radioButtons("rcp", "RCP", c("rcp85" = "rcp85", 
                                          "rcp45" = "rcp45")
                          )

)
),
dashboardBody(
tabItems(
  tabItem(tabName = "shortTerm_alt"),

### REPLACE THE ABOVE WITH THIS ONCE SHORT TERM IS READY
# tabItem(tabName = "shortTerm",
# fluidRow(
# tabBox(width = "100%",
# tabPanel("GDD",
# sidebarLayout(position="right",
#     sidebarPanel(id="menuPanel_shortterm", width = 3,
#                  h2("Plot Parameters"),
#                  selectInput("crop_name",
#                              label = "Crops:",
#                              choices =i selected_crops$name,
#                              selected = default_crop_name),
#                  dateInput("planting_date",
#                            label="Default Planting Date",
#                            value = paste0(this_year, "-04-10")),
#                  sliderInput("maturity_range", "", -30, 30, value=0),
#                  actionButton("gddButton", "Go!")
#     ),
#     mainPanel(width=9, class="main",
#               p("This figure shows the growing degree accumulation starting from a planting date. Historical conditions and this year's forecast are displayed. Use the panel on the side to change the planting dates. The timing of emergence and maturity for the selected crop -  based on phenology assumptions -  is also displayed. Use the panel on the side to change the crop and replot."),
#               h3("Growing Degree Day Forecast"),
#               plotlyOutput("gdd_forecast_plotly", width = "800px", height = "500px")
#     )
# )
# ),
# tabPanel("Precipitation",
# p("The figures below plot monthly accumulation of precipitation for snow, rain, and total precipitation, allowing a comparison between historical ranges and short and long term forecasts. The historical and forecast data provide only precipitation, so rain and snow are estimated based on temperatures using the Pipes and Quick linear split method. This method had limitations, but is an approximation used in several hydrology models."),
# h3("Monthly Snow Accumulation Forecast and Historical Range"),
# plotlyOutput("precip_forecast_snow",  width = "800px"),
# h3("Monthly Rain Accumulation Forecast and Historical Range"),
# plotlyOutput("precip_forecast_rain",  width = "800px"),
# h3("Monthly Total Precipitation Accumulation Forecast and Historical Range"),
# plotlyOutput("precip_forecast_total",  width = "800px")
# ),
# tabPanel("Historical Similarity",
# p("The figures below plot historical yield (per acre) over time, and highlight the five past years that are most 'similar' to the current year in terms of GDD or precipitation. Detrended plots remove the linear trend to show variance. While the correlation between similarlity and yield is not strong, the figures provide a sense of yields in past years with similar GDD and precipitation to the current year. Similarity is indicated in increasing color, where red and blue are most similar, yellow and light blue less so. Gray years are not in the five most similar years."),
# h3("GDD Similarity between Historical and Current Years"),
# plotlyOutput("gdd_similarity_plotly", width = "800px"),
# h3("GDD Similarity, Detrended"),
# plotlyOutput("gdd_similarity_detrended_plotly", width = "800px"),
# h3("Precipitation Similarity between Historical and Current Years"),
# plotlyOutput("precip_similarity_plotly", width = "800px"),
# h3("Precipitation Similarity, Detrended"),
# plotlyOutput("precip_similarity_detrended_plotly", width = "800px")
# )
# )
# )  
# ),
  tabItem(tabName = "longTerm",
    fluidRow(
      tabBox(width = "100%",
       tabPanel("GDD",
          sidebarLayout(position="right",
            sidebarPanel(id="menuPanel_longterm", width = 3,
              h2("Plot Parameters"),
              selectInput("crop_name_longterm",
                label = "Crop",
                choices = selected_crops$name,
                selected = default_crop_name),
              dateInput("planting_date_longterm",
                label="Default Planting Date",
                value = paste0(this_year, "-04-10")),
              sliderInput("maturity_range_longterm", "Default Maturity % Offset:", -30, 30, value=0),
              dateInput("planting_date_longterm2",
                label="Shifted Planting Date",
                value = paste0(this_year, "-03-10")),
              sliderInput("maturity_range_longterm2", "Shifted Maturity % Offset:", -30, 30, value=0),
              actionButton("gddButton2", "Go!")
            ),
            mainPanel(width=9, class="main",
              p("This figure shows the growing degree accumulation for historical and future climate conditions. Use the panel on the side to change the planting dates. The timing of emergence and maturity based on phenology assumptions by crop is also displayed. Use the panel on the side to change the crop"),
              p("With warming going into the future, there is accelerated growing degree day accumulation and earlier crop maturity. This can lead to potential reduction in yields, which could be mitigated by switching to slower growing crop varieties. Earlier  maturity might also result in an additional cutting  of hay crops, or faculitate a double cropping system, if the growing season is long enough."),
              h3("Long-Term Growing Degree Day Trend"),
              plotlyOutput("gdd_outlook_plotly", width = "800px", height = "600px") 
            )
          )  
        ),
        #tabPanel("Evapotransporation",
        #         sidebarLayout(position="right",
        #                       sidebarPanel(id="menuPanel_ET", width = 3,
        #                                    h2("ET Parameters")
        #                                    # selectInput("analogue_run",
        #                                    #             label = "Analogue Variables",
        #                                    #             choices = analogue_runs,
        #                                    #             selected = default_analogue_run)
        #                       ),
        #                       mainPanel(width=9, class='main',
        #                                 h3("Referenced Evapotransporation"),
        #                                 h4("Historical and future evapotransporation for this location"),
        #                                 plotlyOutput("ET_plotly", width = "800px", height="500px")  #textOutput("ET_plotly")  
        #                       )
        #         )
        #),

        tabPanel("Temperature",
          sidebarLayout(position="right",
            sidebarPanel(id="menuPanel_shortterm", width = 3,
              h2("Plot Parameters"),
              sliderInput("heat_risk_range", "Daily Max Temperature Threshold (F)",
                min = 80, max = 100, value = 90, sep='')
            ),
            mainPanel(width=9, class="main",
              p("In this tab, you can visualize the changes expected to daily minimum temperatures (T-min), daily maximum temperatures (T-max), and daily average temperatures (T-gvg) into the future. You can also visualize expected changes to heat and frost risks."),
              p("In the Pacific Northwest region, we expect temperatures to increase going into the future and this signal is robust across all climate model projections. This warming results in a longer available growing season. Even though a longer growing season is available, the actual time-to-maturity for crops is expected to be shorted due to accelerated growing degree day (GDD) accumulation (see GDD tab). This can have a negative impact on crop yields, but can be adapted to, by shifting to crop varieties that have longer times to maturity, or where possible, by shifting to double cropping systems."),
              p("The range of years considered include 1979 to 2015 (historical), 2025 to 2055 (2040s), 2045 to 2075 (2060s) and 2065 to 2095(2080s)."),
              h3("Expected changes to T-min"),
              plotlyOutput("temp_outlook3", width = "800px", height= "400px"),
              h3("Expected changes to T-avg"),
              plotlyOutput("temp_outlook2", width = "800px"),
              h3("Expected changes to T-max"),
              plotlyOutput("temp_outlook1", width = "800px"),
              h3("Expected changes to  heat risk"),
              p("This figure shows the number of days in each month, temperatures higher than a threshold are expected.  Use the option on the side panel to change the thresholds and replot the figure."),
              plotlyOutput("temp_heat_risk", width = "800px"),
              h3("Expected changes to  frost risk"),
              p("This figure shows the number of days in each month, a frost event is expected."),
              plotlyOutput("temp_frost_risk", width = "800px")
            )
          )
        ),
        tabPanel("Precipitation",
          sidebarLayout(position="right",
            sidebarPanel(id="menuPanel_longterm", width = 3,
              h2("Plot Parameters"),
              sliderInput("change_risk_range", "Daily Precipitation Threshold (in/day)",
                min = 0, max = 4, value = 0.25, step = .25, sep='')
            ),
            mainPanel(width=9, class="main",
              p("In this tab, you can visualize the changes expected to  total precipitation, rain, and snow levels, going into the future. Total precipitaion is split into rain and snow using the Pipes and Quick linear split method. This method had limitations, but is an approximation used in several hydrology models. You can also visualize the frequency of daily precipitation events exceeding a user specified threshold."),
              p("In the Pacific Northwest region, we generally expect a warming related shift in precipitation from snow to rain (decreases in snow levels and increases in rain levels). This signal is robust across multiple climate model projections. Total prepicipation is generally expected to stay about the same except perhaps certain months in early spring or winter. There is also variability among climate models in this aspect."),
              p("The range of years considered include 1979 to 2015 (historical), 2025 to 2055 (2040s), 2045 to 2075 (2060s) and 2065 to 2095(2080s)."),
              h3("Expected changed to snow levels"),
              plotlyOutput("precip_outlook_snow",  width = "800px"),
              h3("Expected changed to rain levels"),
              plotlyOutput("precip_outlook_rain",  width = "800px"),
              h3("Expected changes to total precipitation levels"),
              plotlyOutput("precip_outlook_total",  width = "800px"),
              h3("Expected changes to  daily precipitation event frequencies"),
              p("This figure shows the number of days in each month that that daily precipitation exceed a certain threshold.  Use the option on the side panel to change the thresholds and replot the figure."),
              plotlyOutput("precip_outlook_frequency",  width = "800px")
            )
          )
        ),
        tabPanel("Climate Analogues",
          sidebarLayout(position="right",
            sidebarPanel(id="menuPanel_longterm", width = 3,
              h2("Analogue County Similarity Data"),
              radioButtons("matchyear", "Year",
                        c("2040" = "2040",
                          "2060" = "2060",
                          "2080" = "2080")),
             tableOutput("matches_table")
             # tableOutput("match_county_table")
            ),
            mainPanel(width=9, class='main',
              p("Analogs can help users think about projected future changes in the context of what is currently experienced in other locations. An analog of a selected location is another location whose current growing conditions are similar to those expected in the selected location in the future. The analog maps below provide this information based on similarity of soil and climatic parameters. Analogs are provided for 2040s (2025-2055), 2060s (2045-2075) and 2080s(2065-2095) averages. This analysis is at a county level.  So similarity is calculated between historical parameters (1979-2015) of the county containing the selected location and all other counties in the Western United States."),
              h3("County Climate Analogues"),
              p("The selected county is outlined in green. The three counties with historical soil and climate most similar to the future climate of the selected county are highlighted in orange, with more intense orange indicating a closer match."),
              imageOutput("match_county_map", width = "100%", height="auto")
            )
          )
        ) 
      )
    )     
  )
)
)
)
)                 
)
),
tabPanel("About",
  p("This product from the Center for Sustaining Ag and Natural Resources (CSANR) at Washington State University, is a result of funding from the Northwest Climate Hub and USDA NIFA's Regional Approaches to Climate Change (REACHH) project. We would also like to acknowledge John Abatzoglou and Katherine Hegewisch from University of Idaho  for the climate data."),
  p("Here, you can visualize climate data from an agricultural prespective. This prototype is currently for agricultural areas in the Columbia River Basin. Select your locaton and crop of interest, and view historical climate, future climate projections, and seasonal climate forecasts in terms of temperature, precipitation, growing season, growing degree day accumulation, and heat/frost risk outlooks. You can visualize relevant historical crop yield statistics and analogs of growing conditions. You will also find case studies on how changes in climate can affect various aspect of agriculture in the region."),
  p("This is a protype tool under development. We hope to improve the interface, and add more case studies, and look forward to comments/feedback from you. You can reach us at kirti@wsu.edu")
),
tabPanel("Team",
  h3("Funding provided by:"),
  img(src='kcd_logo.png', align = "left", width = "200", height = "200"), 
  img(src='scd_logo.png', width = "200", height = "200"), 
  h3("Personnel:"),
  p("This online tool is primarily the result of efforts of the following people at Washington State University:"),
  tags$ul(
    tags$ol(strong("Kirti Rajagopalan (CSANR)")),
    tags$ol(strong("Nicholas Potter (School of Economic Science)")),
    tags$ol(strong("Trevor Mozingo (School of Electrical Engineering and Computer Science)"))
  ),
  p("Computer Science Senior Design Project Team:"),
  tags$ul(
    tags$ol("Austin Schorsch"),
    tags$ol("Des Marks"),
    tags$ol("Michael Antosz"),
    tags$ol("Trevor Mozingo")
  ),
  p("Other people involved in shaping the current and upcoming content include:"),
  tags$ul(
    tags$ol("Chad Kruger (CSANR)"),
    tags$ol("Tim Ewing (CSANR)"),
    tags$ol("Von Walden (Department of Civil Engineering)"),
    tags$ol("Vincent Jones (Tree Fruit Research Center)"),
    tags$ol("Michael Brady (School of Economic Sciences)"),
    tags$ol("Claudio Stockle (Department of Biological Systems Engineering)"),
    tags$ol("Sepid Mazrouee (School of Electrical Engineering and Computer Science)")
  )
),
tabPanel("Contact",
  p("This is a prototype tool under development. We look forward to comments and feedback from you. You can reach us at kirtir@wsu.edu.")
)
)
)




