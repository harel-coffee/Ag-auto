rm(list=ls())
library(ggmap)
library(ggpubr)
library(plyr)
library(lubridate)
library(purrr)
library(scales)
library(tidyverse)
options(diigt=9)

#####################################################################################
####                                                                             ####
####                           Functions                                         ####
####                                                                             ####
#####################################################################################
observed_map <- function(data, month_col, min, max) {
  data %>% ggplot() +
           geom_polygon(data = states_cluster, aes(x = long, y = lat, group = group),
                        fill = "grey", color = "black") +
           geom_point(aes_string(x = "long", y = "lat",
                                 color = month_col), alpha = 0.4) +
           scale_color_viridis_c(option = "plasma", name = "Median", direction = -1,
                                 limits = c(min, max),
                                 breaks = pretty_breaks(n = 4)) +
           coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
           ggtitle("Observed historical") + 
           theme_bw()
}

model_map <- function(data, model_name, scenario_name, month_col, min, max) {
  data %>% filter(model == model_name,
  	              scenario == scenario_name | scenario == "historical") %>%
           ggplot() +
           geom_polygon(data=states_cluster, aes(x = long, y = lat, group = group),
                        fill="grey", color="black") +
            # aes_string to allow naming of column in function 
           geom_point(aes_string(x = "long", y = "lat",
                                 color = month_col), alpha = 0.4) +
           scale_color_viridis_c(option = "plasma", name = "Median", direction = -1,
                                 limits = c(min, max),
                                 breaks = pretty_breaks(n = 4)) +
           coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
           facet_wrap(~ scenario, nrow = 1) +
           ggtitle(paste0(model_name)) + 
           theme_bw()
}

# A function to make a map from the averaged dataset. Note that it uses the
# ensemble data frame.
ensemble_map <- function(data, month_col, min, max) {
  data %>% ggplot() +
           geom_polygon(data = states_cluster, aes(x = long, y = lat, group = group),
                        fill = "grey", color = "black") +
           # aes_string to allow naming of column in function 
           geom_point(aes_string(x = "long", y = "lat",
                                 color = month_col), alpha = 0.4) +
           scale_color_viridis_c(option = "plasma", name = "Median", direction = -1,
                                 limits = c(min, max),
                                 breaks = pretty_breaks(n = 4)) +
           coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
           facet_wrap(~ scenario, nrow = 1) +
           ggtitle("Ensemble means") + 
           theme_bw()
}

#####################################################################################
dynamic_data_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/overlapping/dynamic_model_stats/slopes/"
utah_data_dir= "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/overlapping/utah_model_stats/slopes/"

model_names = c("utah", "dynamic")
model = model_names[1]

for (model in model_names){
  if (model=="dynamic"){
    data_dir = dynamic_data_dir
  } else {
    data_dir = utah_data_dir
  }

  observed_slopes = data.table(readRDS(paste0(data_dir, model, "_observed_slopes.rds")))
  modeled_slopes = data.table(readRDS(paste0(data_dir, model, "_modeled_slopes.rds")))

  observed_slopes$lat = sapply(strsplit(observed_slopes$location, "-"), function(x) x[1])
  observed_slopes$long = paste0("-", sapply(strsplit(observed_slopes$location, "-"), function(x) x[2]))
  modeled_slopes$lat = sapply(strsplit(modeled_slopes$location, "-"), function(x) x[1])
  modeled_slopes$long = paste0("-", sapply(strsplit(modeled_slopes$location, "-"), function(x) x[2]))

  observed_slopes$lat = as.numeric(observed_slopes$lat)
  observed_slopes$long = as.numeric(observed_slopes$long)

  modeled_slopes$lat = as.numeric(modeled_slopes$lat)
  modeled_slopes$long = as.numeric(modeled_slopes$long)


  observed_slopes = within(observed_slopes, remove(location))
  modeled_slopes = within(modeled_slopes, remove(location))

  states <- map_data("state")
  states_cluster <- subset(states, region %in% c("oregon", "washington", "idaho"))

  # Take a mean across models for ensemble plot
  ensemble_slopes <- modeled_slopes %>%
                     group_by(lat, long, scenario) %>%
                     summarize_if(.predicate = is.numeric, mean)%>% data.table()

  modeled_min = min(ensemble_slopes$slope)
  modeled_max = max(ensemble_slopes$slope)

  observed_min = min(observed_slopes$slope)
  observed_max = max(observed_slopes$slope)

  # universal_min = min(modeled_min, observed_min)
  # universal_max = max(modeled_max, observed_max)

  # universal_min = -8.25
  # universal_max = 1.09

  ens_plot <- ensemble_map(data=ensemble_slopes, month_col = "slope", 
  	                       min = modeled_min, max = modeled_max) 

  obs_plot <- observed_map(data=observed_slopes, month_col = "slope", 
                           min = observed_min, max = observed_max)


  assembeled <- ggarrange(plotlist = list(obs_plot, ens_plot), 
                          ncol = 2, nrow = 1, 
                          widths = c(1, 2.3) , heights = 1,
                          common.legend = TRUE)

  ggsave(filename = paste0(model, "_slopes.png"), 
         plot = assembeled, 
         width = 8, height = 4, units = "in", 
         dpi=400, device = "png",
         path="/Users/hn/Desktop/")
}

