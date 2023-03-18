rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
library(maps)

source_path_1 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

options(digit=9)
options(digits=9)
# --------------------------------------------------------------------
# location and elevations

#######################      Precip-Elevation 5 clusters
#######################
in_dir <- "/Users/hn/Documents/GitHub/Ag/Lagoon/parameters/"
observed <- readRDS(paste0(in_dir, "5_clust_with_elevation_info.rds")) %>%
            data.table()

observed <- get_ridof_canada(observed)

geo_map_of_clusters_4_web <- function(obs_w_clusters){
  obs_w_clusters <- subset(obs_w_clusters, 
                           select=c(location, cluster))
  obs_w_clusters <- unique(obs_w_clusters)

  x <- sapply(obs_w_clusters$location, 
              function(x) strsplit(x, "_")[[1]], 
              USE.NAMES=FALSE)
  lat = as.numeric(x[1, ]); long = as.numeric(x[2, ])
  obs_w_clusters$lat <- lat; obs_w_clusters$long <- long
  obs_w_clusters <- within(obs_w_clusters, remove(location))

  # states <- map_data("state")
  # WA_state <- subset(states, region %in% c("washington"))
  # WA_state <- WA_state %>% filter(subregion == "main")
  
  WA_counties <- map_data("county", "washington")
  WA_counties <- WA_counties %>% 
                 filter(subregion %in% c("whatcom", "skagit", "snohomish", "island"
                                         #, "okanogan" "chelan"
                                         ))%>% 
                 data.table()

  # color_ord = c("red", "purple", "dodgerblue2", "blue4")
  if (length(unique(obs_w_clusters$cluster))==4){
     color_ord = c("royalblue3", "steelblue1", "maroon3", "red")
     } else if(length(unique(obs_w_clusters$cluster))== 5){
     color_ord = c("royalblue3", "steelblue1", "maroon3", "red", "black")
  }
  
  the_theme <- theme(plot.margin = unit(c(t=.2, r=.1, b=.2, l=0.2), "cm"),
                     plot.title = element_text(size = 13, face = "bold"),
                     plot.subtitle = element_text(size = 10, face = "bold"),
                     panel.border = element_rect(fill=NA, size=.3),
                     legend.position = "right",
                     legend.key.size = unit(.6, "line"),
                     legend.spacing.x = unit(.1, 'line'),
                     legend.background=element_blank(),
                     # legend.key = element_blank(),
                     legend.text = element_text(size=16, face="bold"),
                     legend.margin = margin(t=.4, r=0, b=0, l=0, unit = 'line'),
                     legend.title = element_blank(),
                     axis.ticks = element_blank(),
                     axis.text.y = element_blank(),
                     axis.text.x = element_blank(),
                     axis.title.y = element_blank(),
                     axis.title.x = element_blank())

  cluster_plot <- obs_w_clusters %>%
                  ggplot() +
                  geom_polygon(data = WA_counties, 
                               aes(x=long, y=lat, group = group),
                               fill = "grey", color = "black", size=0.5) +
                  geom_polygon(data=WA_counties, 
                               aes(x=long, y=lat,group = group), 
                               fill = NA, colour = "black", size=0.0000001) + 
                  geom_point(aes_string(x = "long", y = "lat", color="cluster"), 
                            alpha = 1, size=2.5) + 
                  scale_color_manual(values = color_ord, name = "Precip.") + 
                  the_theme +
                  # size of dot inside the legend
                  guides(colour = guide_legend(override.aes = list(size=3))) + 
                  # labs(title = "Groups of grids based on annual precip.",
                  #      subtitle = "averaged over 38 years.") + 
                  # ggtitle("Groups of grids") + 
                  # size of dot inside the legend box
                  guides(colour = guide_legend(override.aes = list(size=2.5)))

  return(cluster_plot)
}

PE_5_plt <- geo_map_of_clusters_4_web(observed) 

plot_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
ggsave(filename = "clust_map_4_web.png", 
       plot = PE_5_plt, device = "png",
       width = 10, height = 6, 
       units = "in", dpi=600,
       path = plot_dir)
