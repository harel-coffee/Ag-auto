rm(list=ls())
library(shiny)
library(leaflet)
r_colors <- rgb(t(col2rgb(colors()) / 255))
names(r_colors) <- colors()


shinyUI(
	fluidPage(leafletOutput("mymap")
              )
)
