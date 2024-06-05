rm(list=ls())

library(ggmap)
library(ggpubr)
library(lubridate)
library(purrr)
library(scales)
library(tidyverse)
library(maps)
library(data.table)
library(dplyr)

options(digits=9)
options(digit=9)


data_dir = "/Users/hn/Documents/01_research_data/RangeLand/Data/for_paper/"

States_on_Map <- read.csv(paste0(data_dir, "States_on_Map.csv"), 
                          header=T, sep=",", as.is=T)

States_on_Map$region = tolower(States_on_Map$state_full)

states <- map_data("state")

SoI_states <- subset(states, region %in% States_on_Map$region)


map(database = "state")
map(database = "state", regions = tolower(States_on_Map$state_full), 
    col = "blue", fill=T, add=TRUE)
#########
states_2 = full_join(states, States_on_Map, by = "region")

ggplot() + 
     geom_polygon(data = states_2, aes(x=long, y=lat, group=group),
                  fill = ifelse(states_2$rangeland_acre > 10 ,"red", "green"),
                  linewidth = 0.2)

ggplot(states_2, aes(map_id = region)) + 
  geom_map(aes(fill = rangeland_acre), map = states)+
  expand_limits(x = states$long, y = states$lat)


# Get centroids
centroid_labels <- usmapdata::centroid_labels("states")

# Join centroids to data
state_labels <- merge(States_on_Map, centroid_labels, by.x = "state", by.y = "abbr")

ggplot(states_2, aes(map_id = region)) + 
  geom_map(aes(fill = ifelse(rangeland_acre > 0 ,"red", "black")), map = states) +
  scale_fill_manual(values = c(`red` = "dodgerblue", `black` = "black")) + 
  geom_polygon(data=states_2, aes(x=long, y=lat, group = group), colour='black', fill=NA, size = 0.35) + 
  expand_limits(x = states$long, y = states$lat) +
  # geom_text(data = states_2, aes(x=long, y=lat, group=region, label=region), size = 3, hjust=0, vjust=-1) + 
  # There is no x ad y here. We need to adjust and build on this.
  # geom_text(data = state_labels, aes(x = x, y = y, label = state_full), color = "white") 
  theme(axis.title.y = element_blank(),
                 axis.title.x = element_blank(),
                 axis.ticks.y = element_blank(), 
                 axis.ticks.x = element_blank(),
                 axis.text.x = element_blank(),
                 axis.text.y = element_blank(),
                 panel.grid.major = element_line(size = 0.1),
                 legend.position = 'none', 
                 legend.text=element_text(size=12), #, face="bold"
                 legend.title=element_text(size=15),
                 strip.text = element_text(size=12, face="bold"),
                 plot.margin = margin(t=0.03, b=0.03, r=0.2, l=0.2, unit = 'cm')
                 )






data %>% ggplot() +
         geom_polygon(data = SoI_states, aes(x=long, y=lat, group = group),
                       fill = "grey", color = "black") +
            # aes_string to allow naming of column in function 
           geom_point(aes_string(x = "long", y = "lat", color = color_col), alpha = 0.4, size=.1) +
           coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
           facet_grid(~ emission ~ time_period) +
           labs(x = "longitude (degree)", y = "latitude (degree)") +
           scale_y_continuous(breaks = seq(40, 50, by = 5)) + 
           scale_x_continuous(breaks = seq(-125, -110, by = 10)) + 
           theme(axis.title.y = element_text(color = "black", face="bold"),
                 axis.title.x = element_text(color = "black", face="bold"),
                 # axis.ticks.y = element_blank(), 
                 # axis.ticks.x = element_blank(),
                 axis.text.x = element_text(color = "black"),
                 axis.text.y = element_text(color = "black"),
                 panel.grid.major = element_line(size = 0.1),
                 legend.position="bottom", 
                 legend.text=element_text(size=12), #, face="bold"
                 legend.title=element_text(size=15),
                 strip.text = element_text(size=12, face="bold"),
                 plot.margin = margin(t=0.03, b=0.03, r=0.2, l=0.2, unit = 'cm')
                 ) + 
           labs(fill=guide_legend(title=legend_d)) + 
           scale_color_gradient2(midpoint = 0, mid = "white", 
                                 high = "blue", low = "red", 
                                 guide = "colourbar", space = "Lab",
                                 limit = c(low_lim, up_lim),
                                 breaks = c(-200, -75, -40, -25, 0, 25, 40, 75, 200),
                                 name=legend_d)