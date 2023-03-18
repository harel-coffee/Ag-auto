# Script for creating chill accumulation & threshold maps.
# Intended to work with create-model-maps.sh script.

# 1. Load packages --------------------------------------------------------
rm(list=ls())
library(ggmap)
library(ggpubr)
# library(plyr)
library(lubridate)
library(purrr)
library(scales)
library(tidyverse)
library(maps)
library(dplyr)
library(data.table)
options(digits=9)
options(digit=9)

# 2. Pull data from current directory -------------------------------------

# define the color spectrum so they are consistent
dynamic_chill_min = 40
dynamic_chill_max = 180
dynamic_thresh_min = 10
dynamic_thresh_max = 185

##### either choose utah or dynamic

start = "sept"
in_dir <- paste0("/Users/hn/Documents/01_research_data/chilling/sum_stats_4_maps/", start, "/")
setwd(in_dir)
plot_path <- in_dir
getwd()

the_dir <- dir()
the_dir <- the_dir[grep(pattern = ".txt", x = the_dir)] # remove filenames that aren't data
the_dir_stats <- the_dir[grep(pattern = "summary_stats", x = the_dir)] # Data for maps

stats_comp <- lapply(the_dir_stats, read.table, header = T, as.is=T, stringsAsFactors=FALSE) # Compile the summary stats files for mapping
stats_comp <- do.call(bind_rows, stats_comp)

# str(stats_comp)
# stats_comp %>% select(model, scenario) %>% unique()

# stats_comp <- stats_comp %>% data.table()
stats_comp$time_period[is.na(stats_comp$time_period)] <- "Historical"

########################################################################
#
#    Get rid of modeled historical and 2005_2024
#
########################################################################
stats_comp_obs <- stats_comp %>% filter(model == "observed")
stats_comp_F <- stats_comp %>% filter(time_period %in% c("2025_2050", "2051_2075", "2076_2100"))
stats_comp <- rbind(stats_comp_obs, stats_comp_F)
rm(stats_comp_F, stats_comp_obs)
########################################################################
#
#    Edit time periods
#
########################################################################
stats_comp$time_period[stats_comp$time_period == "2025_2050"] <- "2026-2050"
stats_comp$time_period[stats_comp$time_period == "2051_2075"] <- "2051-2075"
stats_comp$time_period[stats_comp$time_period == "2076_2100"] <- "2076-2099"

setnames(stats_comp, old=c("scenario"), new=c("emission"))
stats_comp$emission[stats_comp$emission == "historical"] <- "Historical"
stats_comp$emission[stats_comp$emission == "rcp45"] <- "RCP 4.5"
stats_comp$emission[stats_comp$emission == "rcp85"] <- "RCP 8.5"

stats_comp$model[stats_comp$model == "observed"] <- "Observed"


stats_comp$time_period <- factor(stats_comp$time_period, 
                                 levels=c("Historical", "2026-2050", "2051-2075", "2076-2099"),
                                 order=TRUE)

param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"
LocationGroups_NoMontana <- read.csv(paste0(param_dir, "LocationGroups_NoMontana.csv"), 
                                     header=T, sep=",", as.is=T)

# stats_comp <- remove_montana_add_warm_cold(stats_comp, LocationGroups_NoMontana)
# stats_comp <- within(stats_comp, remove(warm_cold))

remove_montana <- function(data_dt, LocationGroups_NoMontana){
  if (!("location" %in% colnames(data_dt))){
    data_dt$location <- paste0(data_dt$lat, "_", data_dt$long)
  }
  data_dt <- data_dt %>% filter(location %in% LocationGroups_NoMontana$location)
  return(data_dt)
}

stats_comp <- remove_montana(stats_comp, LocationGroups_NoMontana)

# stats_comp$time_period <- as.character(stats_comp$time_period)
# stats_comp$time_period[stats_comp$model== "observed"] <- "Observed historical"

# Take a mean across models
stats_comp_ensemble_mean <- stats_comp %>%
                            group_by(lat, long, location, emission, time_period) %>%
                            summarize_if(.predicate = is.numeric, mean) %>%
                            data.table()

stats_comp_ensemble_median <- stats_comp %>%
                              group_by(lat, long, location, emission, time_period) %>%
                              summarize_if(.predicate = is.numeric, median) %>%
                              data.table()


# str(stats_comp_ensemble)

# # Data for plots
# the_dir_summary <- the_dir[-grep(pattern = "summary_stats", x = the_dir)]
# # Compile the data files for plotting
# summary_comp <- lapply(the_dir_summary, read.table, header = T)
# summary_comp <- do.call(bind_rows, summary_comp)

# str(summary_comp)
# # Remove incomplete model runs
# # summary_comp <- summary_comp[-grep(x = summary_comp$model, pattern = "incomplete"),]
# str(summary_comp)

# 3. Mapping --------------------------------------------------------------

# Pull base layers
states <- map_data("state")
states_cluster <- subset(states, region %in% c("oregon", "washington", "idaho"))

# Base map for following figures
# base_map <- ggplot() +
#  geom_polygon(data = states_cluster, aes(x = long, y = lat, group = group),
#               fill = "grey", color = "black")

# 3a. Map functions -------------------------------------------------------

# Create a map of observed historical data. I'm not clear exactly how ggarrange()
# works with common.legend to harmonize legends between plots. To be safe, I
# make sure all plots have same limits up front and thus make a new observed
# hist map for each time ggarrange is called, using function below.

# Make a function to plot the future scenarios. Note that it requires the main
# data frame to be called stats_comp and the base map to be called
#states_cluster.

observed_hist_map <- function(dt=stats_comp, minn, maxx, month_col) {
  
  dt %>%
  filter(model == "Observed") %>%
  ggplot() +
  geom_polygon(data = states_cluster, 
               aes(x = long, y = lat, group = group),
               fill = "grey", color = "black") +
  
  geom_point(aes_string(x = "long", y = "lat",
                        color = month_col), 
                        alpha = 0.4, size=0.1) +
  scale_color_viridis_c(option = "plasma", direction = -1,
                        limits = c(minn, maxx),
                        name = "Mean",
                        breaks = pretty_breaks(n = 4)) +

  coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
  facet_wrap(~ emission ~ time_period, nrow = 1) +
  theme(axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(), 
        axis.ticks.x = element_blank(),
        legend.title = element_blank(),
        plot.margin = margin(t=-0.5, r=0.2, b=0, l=0.2, unit = 'cm'),
        strip.text = element_text(size=12, face="bold"))
}

# A function to make a map from the averaged dataset. Note that it uses the
# ensemble data frame.
ensemble_map <- function(dt, scenario_name, month_col, legend_label) {
  minn <- min(dt[, get(month_col)])
  maxx <- max(dt[, get(month_col)])
  dt <- dt %>%
        filter(emission %in% c("Historical", scenario_name)) %>%
        data.table()
  dt$emission <- scenario_name
  
  dt %>% ggplot() +
         geom_polygon(data = states_cluster, aes(x = long, y = lat, group = group),
                     fill = "grey", color = "black") +
         # aes_string to allow naming of column in function 
         geom_point(aes_string(x = "long", y = "lat",
                               color = month_col), alpha = 0.4, size=0.1) +
         scale_color_viridis_c(option = "plasma", name = legend_label, direction = -1,
                               limits = c(minn, maxx),
                               breaks = pretty_breaks(n = 4)) +
         coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
         facet_grid(~ emission ~ time_period) + # facet_wrap came with nrow = 1
         theme(axis.title.y = element_blank(),
               axis.title.x = element_blank(),
               axis.ticks.y = element_blank(), 
               axis.ticks.x = element_blank(),
                axis.text.x = element_blank(),
               axis.text.y = element_blank(),
               panel.grid.major = element_line(size = 0.1),
               legend.position="bottom", 
               strip.text = element_text(size=12, face="bold"),
               plot.margin = margin(t=-0.5, r=0.2, b=-0.5, l=0.2, unit = 'cm'),
               legend.title = element_blank()
               )
}


mean_of_models <- ensemble_map(dt = stats_comp_ensemble_mean, 
                               scenario_name= "RCP 8.5", 
                               month_col = "median_A1", 
                               legend_label = "Mean")

median_of_models <- ensemble_map(dt = stats_comp_ensemble_median, 
                                 scenario_name = "RCP 8.5", 
                                 month_col = "median_A1", 
                                 legend_label = "Median")

plot_base <- paste0("/Users/hn/Documents/00_GitHub/ag_papers/chill_paper/02_Springer_2/figure_200/cp_accum_map/")
if (dir.exists(plot_base) == F) {dir.create(path = plot_base, recursive = T)}

plt_width <- 10
plt_height <- 3.3
qual = 400
ggsave(filename = paste0("accumu_CP_Sept1March31_85_mean_of_models_median_of_years.pdf"), 
       plot=mean_of_models, 
       width=plt_width, height=plt_height, units="in", 
       dpi = qual, device="pdf", 
       path=plot_base)

ggsave(filename = paste0("accumu_CP_Sept1March31_85_median_of_models_median_of_years.pdf"), 
       plot=median_of_models, 
       width=plt_width, height=plt_height, units="in", 
       dpi = qual, device="pdf", 
       path=plot_base)


mean_of_models <- ensemble_map(dt = stats_comp_ensemble_mean, 
                               scenario_name= "RCP 4.5", 
                               month_col = "median_A1",
                               legend_label = "Mean")

median_of_models <- ensemble_map(dt = stats_comp_ensemble_median, 
                                 scenario_name = "RCP 4.5", 
                                 month_col = "median_A1", 
                                 legend_label = "Median")

ggsave(filename = paste0("accumu_CP_Sept1March31_45_mean_of_models_median_of_years.pdf"), 
       plot=mean_of_models, 
       width=plt_width, height=plt_height, units="in", 
       dpi = qual, device="pdf", 
       path=plot_base)

ggsave(filename = paste0("accumu_CP_Sept1March31_45_median_of_models_median_of_years.pdf"), 
       plot=median_of_models, 
       width=plt_width, height=plt_height, units="in", 
       dpi = qual, device="pdf", 
       path=plot_base)

plot_base


ensemble_map_2_rcps <- function(dt, month_col, legend_label) {
  minn <- min(dt[, get(month_col)])
  maxx <- max(dt[, get(month_col)])

  dt_F <- dt %>%
        filter(emission != "Historical") %>%
        data.table()

  d_hist <- dt %>%
            filter(emission =="Historical") %>%
            data.table()

  d_hist_45 <- d_hist

  d_hist_45$emission <- "RCP 4.5"
  d_hist$emission <- "RCP 8.5"

  dt <- rbind(d_hist_45, d_hist, dt_F)

  dt$emission <- factor(dt$emission, 
                        levels=c("RCP 8.5", "RCP 4.5"),
                        order=TRUE)
  
  dt %>% ggplot() +
         geom_polygon(data = states_cluster, aes(x = long, y = lat, group = group),
                     fill = "grey", color = "black") +
         # aes_string to allow naming of column in function 
         geom_point(aes_string(x = "long", y = "lat",
                               color = month_col), alpha = 0.4, size=0.1) +
         scale_color_viridis_c(option = "plasma", name = legend_label, direction = -1,
                               limits = c(minn, maxx),
                               breaks = pretty_breaks(n = 4)) +
         coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
         facet_grid(~ emission ~ time_period) + # facet_wrap came with nrow = 1
         theme(axis.title.y = element_blank(),
               axis.title.x = element_blank(),
               axis.ticks.y = element_blank(), 
               axis.ticks.x = element_blank(),
                axis.text.x = element_blank(),
               axis.text.y = element_blank(),
               panel.grid.major = element_line(size = 0.1),
               legend.position="bottom", 
               strip.text = element_text(size=12, face="bold"),
               plot.margin = margin(t=-0.5, r=0.2, b=-0.5, l=0.2, unit = 'cm'),
               legend.title = element_blank()
               )
}

both_RCPs_mean <- ensemble_map_2_rcps(dt = stats_comp_ensemble_median, 
                                      month_col = "median_A1", 
                                      legend_label = "Mean")

ggsave(filename = paste0("accumu_CP_Sept1March31.pdf"), 
       plot=both_RCPs_mean, 
       width=plt_width, height=5.6, units="in", 
       dpi = qual, device="pdf", 
       path=plot_base)





