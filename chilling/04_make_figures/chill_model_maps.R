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
#library(data.table)
options(digits=9)
options(digit=9)

# 2. Pull data from current directory -------------------------------------

# print to check
getwd()

# define the color spectrum so they are consistent
dynamic_chill_min = 40
dynamic_chill_max = 180
dynamic_thresh_min = 10
dynamic_thresh_max = 185

utah_chill_min = -140
utah_chill_max = 3000
utah_thresh_min = 20
utah_thresh_max = 150

##### either choose utah or dynamic

model_name = "dynamic"

if (model_name=="utah"){
  #setwd("/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/utah_model_stats/")
  #plot_path = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/utah_model_stats/"
  chill_min = utah_chill_min
  chill_max = utah_chill_max
  thresh_min = utah_thresh_min
  thresh_max = utah_thresh_max
 } else if (model_name=="dynamic"){
  #setwd("/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/dynamic_model_stats/")
  #plot_path = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/dynamic_model_stats/"
  chill_min = dynamic_chill_min
  chill_max = dynamic_chill_max
  thresh_min = dynamic_thresh_min
  thresh_max = dynamic_thresh_max
}

start = "sept"
in_dir <- paste0("/Users/hn/Documents/01_research_data/Ag_check_point/chilling/sum_stats_4_maps/", start, "/")
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

# remove time period 2005_2024
stats_comp <- stats_comp %>% filter(time_period != "2005_2024")
stats_comp$time_period <- factor(stats_comp$time_period)

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

stats_comp$time_period <- as.character(stats_comp$time_period)
stats_comp$time_period[stats_comp$model== "observed"] <- "Observed historical"

## stats_comp$time_period[stats_comp$time_period== "2005_2024"] = "2005-2024"
stats_comp$time_period[stats_comp$time_period== "2025_2050"] = "2025-2050"
stats_comp$time_period[stats_comp$time_period== "2051_2075"] = "2051-2075"
stats_comp$time_period[stats_comp$time_period== "2076_2100"] = "2076-2100"

stats_comp$time_period <- factor(stats_comp$time_period, 
                                 levels = c("Observed historical", "Historical", "2025-2050", "2051-2075", "2076-2100"),
                                 order=T)

# Take a mean across models
stats_comp_ensemble <- stats_comp %>%
                       filter(model != "observed") %>%
                       group_by(lat, long, scenario, time_period) %>%
                       summarize_if(.predicate = is.numeric, mean)

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
#base_map <- ggplot() +
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

model_map <- function(model_name, scenario_name, month_col, min, max) {
  if (scenario_name == "rcp45"){
    NN = "RCP 4.5"
    } else if(scenario_name=="rcp85") {
    NN = "RCP 8.5"
   } else {
    NN = "Figrue it out"
  }
  
  stats_comp %>%
  filter(model == model_name,
         scenario == scenario_name | scenario == "historical") %>%
  ggplot() +
  geom_polygon(data = states_cluster, aes(x = long, y = lat, group = group),
               fill = "grey", color = "black") +
  # aes_string to allow naming of column in function 
  geom_point(aes_string(x = "long", y = "lat",
                        color = month_col), alpha = 0.4) +
  scale_color_viridis_c(option = "plasma", name = "Mean", direction = -1,
                        limits = c(min, max),
                        breaks = pretty_breaks(n = 4)) +
  coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
  facet_wrap(~ time_period, nrow = 1) +
  theme(axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.y = element_blank(), 
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.x = element_blank(),
        strip.text = element_text(size=12, face="bold")) + 
  ggtitle(paste0(model_name))
}

observed_hist_map <- function(min, max, month_col) {
  
  stats_comp %>%
  filter(model == "observed") %>%
  ggplot() +
  geom_polygon(data = states_cluster, 
               aes(x = long, y = lat, group = group),
               fill = "grey", color = "black") +
  
  geom_point(aes_string(x = "long", y = "lat",
                        color = month_col), alpha = 0.4) +
  scale_color_viridis_c(option = "plasma", direction = -1,
                        limits = c(min, max),
                        name = "Mean",
                        breaks = pretty_breaks(n = 4)) +

  coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
  facet_wrap(~ time_period, nrow = 1) +
  theme(axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(), 
        axis.ticks.x = element_blank(),
        plot.margin = margin(t=-0.5, r=0.2, b=0, l=0.2, unit = 'cm'),
        strip.text = element_text(size=12, face="bold"))
}

# A function to make a map from the averaged dataset. Note that it uses the
# ensemble data frame.
ensemble_map <- function(scenario_name, month_col, min, max) {
  if (scenario_name == "rcp45"){
    NN = "RCP 4.5"
   } else if(scenario_name=="rcp85") {
    NN = "RCP 8.5"
  } else{
    NN = "Figrue it out"
  }

  stats_comp_ensemble %>%
  filter(scenario == scenario_name | scenario == "historical") %>%
  ggplot() +
  geom_polygon(data = states_cluster, aes(x = long, y = lat, group = group),
               fill = "grey", color = "black") +
  # aes_string to allow naming of column in function 
  geom_point(aes_string(x = "long", y = "lat",
                        color = month_col), alpha = 0.4) +
  scale_color_viridis_c(option = "plasma", name = "Mean", direction = -1,
                        limits = c(min, max),
                        breaks = pretty_breaks(n = 4)) +
  coord_fixed(xlim = c(-124.5, -111.4),  ylim = c(41, 50.5), ratio = 1.3) +
  facet_wrap(~ time_period, nrow = 1) +
  theme(axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.y = element_blank(), 
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        strip.text = element_text(size=12, face="bold"),
        plot.margin = margin(t=-0.5, r=0.2, b=0, l=0.2, unit = 'cm'),
        )
}

# 3b. RCP45 monthly accum figs --------------------------------------------


################################################################
# January, rcp45

# Going to make a common scale using min/max of all models in Jan @ rcp45 OR
# either modeled or observed historical

df_45 <- filter(stats_comp, scenario == "rcp45" | scenario == "historical")

accum_jan45_min <- min(df_45$median_J1)
accum_jan45_max <- max(df_45$median_J1)

accum_jan45_min = chill_min
accum_jan45_max = chill_max

#*** To go with Sept-Jan 85 be indentical for comparison
accum_jan45_min = 31
accum_jan45_max = 75

cat("accum_jan45_min= ", accum_jan45_min, "-- accum_jan45_max= ", accum_jan45_max)

# Create and automatically assign a map object for each model
for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "jan45", sep="_"),
         value ={ model_map(model = h, scenario_name = "rcp45", month_col = "median_J1",
                     min = accum_jan45_min, max = accum_jan45_max)
                }
       )
}

# Separately created maps for observed historical and ensemble mean
observed_map_jan45 <- observed_hist_map(min = accum_jan45_min, max = accum_jan45_max,
                                        month_col = "median_J1")

ensemble_map_jan45 <- ensemble_map(scenario_name = "rcp45", 
                                   month_col = "median_J1", 
                                   min = accum_jan45_min,
                                   max = accum_jan45_max)

hist_ensemble_jan45_figs <- ggarrange(plotlist = list(observed_map_jan45, ensemble_map_jan45),
                                      ncol = 2, widths = c(0.26, 1), 
                                      nrow = 1,
                                      common.legend = TRUE)

ggsave(filename = paste0("accum_", start, "_jan_45_obs_ensemble_mean_models_median_years.png"), 
       plot = hist_ensemble_jan45_figs, 
       device = "png",
       width = 10, height = 3, units = "in", 
       dpi=600, path=plot_path)

# Need to add historical observed to this:
# accum_jan45_figs <- ggarrange(plotlist = list(observed_map_jan45,
#                                               ensemble_map_jan45,
#                                               bcc_csm1_1_m_map_jan45,
#                                               bcc_csm1_1_map_jan45,
#                                               BNU_ESM_map_jan45,
#                                               CanESM2_map_jan45,
#                                               CCSM4_map_jan45, 
#                                               CNRM_CM5_map_jan45,
#                                               CSIRO_Mk3_6_0_map_jan45,
#                                               GFDL_ESM2G_map_jan45,
#                                               GFDL_ESM2M_map_jan45,
#                                               HadGEM2_CC365_map_jan45,
#                                               HadGEM2_ES365_map_jan45,
#                                               inmcm4_map_jan45,
#                                               IPSL_CM5A_LR_map_jan45, 
#                                               IPSL_CM5A_MR_map_jan45,
#                                               IPSL_CM5B_LR_map_jan45,
#                                               MIROC_ESM_CHEM_map_jan45,
#                                               MIROC5_map_jan45, 
#                                               MRI_CGCM3_map_jan45,
#                                               NorESM1_M_map_jan45),
#                               ncol = 2, nrow = 11,
#                               common.legend = TRUE)

# accum_jan45_figs <- annotate_figure(p = accum_jan45_figs,
#                                     top = text_grob(label = "Median accumulated chill units by Jan. 1 under RCP 4.5",
#                                                    face = "bold", size = 18))

# ggsave(filename = "accum_jan45.png", plot = accum_jan45_figs, device = "png",
#        width = 15, height = 40, units = "in", dpi=400, path=plot_path)



rm(list = ls(pattern = "jan45"))
######################################################
# February, rcp 45

accum_feb45_min <- min(df_45$median_F1)
accum_feb45_max <- max(df_45$median_F1)

accum_feb45_min <- chill_min
accum_feb45_max <- chill_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "feb45", sep="_"),
         value = { model_map(model = h, scenario_name = "rcp45", month_col = "median_F1",
                             min = accum_feb45_min, max = accum_feb45_max)}
          )
}
observed_map_feb45 <- observed_hist_map(min = accum_feb45_min, max = accum_feb45_max,
                                        month_col = "median_F1")

ensemble_map_feb45 <- ensemble_map(scenario_name = "rcp45", 
                                   month_col = "median_F1", min = accum_feb45_min,
                                   max = accum_feb45_max)

# Need to add historical observed to this:
accum_feb45_figs <- ggarrange(plotlist = list(observed_map_feb45,
                                              ensemble_map_feb45,
                                              bcc_csm1_1_m_map_feb45,
                                              bcc_csm1_1_map_feb45,
                                              BNU_ESM_map_feb45,
                                              CanESM2_map_feb45,
                                              CCSM4_map_feb45,
                                              CNRM_CM5_map_feb45,
                                              CSIRO_Mk3_6_0_map_feb45,
                                              GFDL_ESM2G_map_feb45,
                                              GFDL_ESM2M_map_feb45,
                                              HadGEM2_CC365_map_feb45,
                                              HadGEM2_ES365_map_feb45,
                                              inmcm4_map_feb45,
                                              IPSL_CM5A_LR_map_feb45, 
                                              IPSL_CM5A_MR_map_feb45,
                                              IPSL_CM5B_LR_map_feb45,
                                              MIROC_ESM_CHEM_map_feb45,
                                              MIROC5_map_feb45,
                                              MRI_CGCM3_map_feb45,
                                              NorESM1_M_map_feb45),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

accum_feb45_figs<- annotate_figure(p = accum_feb45_figs,
                                   top = text_grob(label = "Median accumulated chill units by Feb. 1 under RCP 4.5",
                                   face = "bold", size = 18))

ggsave(filename = "accum_feb45.png", plot = accum_feb45_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path = plot_path)

rm(list = ls(pattern = "feb45"))
################################################
#######
#######          March, rcp 45
#######
################################################
accum_mar45_min <- min(df_45$median_M1)
accum_mar45_max <- max(df_45$median_M1)

accum_mar45_min <- chill_min
accum_mar45_min <- chill_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "mar45", sep="_"),
         value =
         {
           model_map(model = h, scenario_name = "rcp45", month_col = "median_M1",
                     min = accum_mar45_min, max = accum_mar45_max)
         }
  ) 
}

observed_map_mar45 <- observed_hist_map(min = accum_mar45_min, max = accum_mar45_max,
                                        month_col = "median_M1")

ensemble_map_mar45 <- ensemble_map(scenario_name = "rcp45", 
                                   month_col = "median_M1", min = accum_mar45_min,
                                   max = accum_mar45_max)

# Need to add historical observed to this:
accum_mar45_figs <- ggarrange(plotlist = list(observed_map_mar45,
                                              ensemble_map_mar45,
                                              bcc_csm1_1_m_map_mar45,
                                              bcc_csm1_1_map_mar45,
                                              BNU_ESM_map_mar45,
                                              CanESM2_map_mar45,
                                              CCSM4_map_mar45,
                                              CNRM_CM5_map_mar45,
                                              CSIRO_Mk3_6_0_map_mar45,
                                              GFDL_ESM2G_map_mar45,
                                              GFDL_ESM2M_map_mar45,
                                              HadGEM2_CC365_map_mar45,
                                              HadGEM2_ES365_map_mar45,
                                              inmcm4_map_mar45,
                                              IPSL_CM5A_LR_map_mar45, 
                                              IPSL_CM5A_MR_map_mar45,
                                              IPSL_CM5B_LR_map_mar45,
                                              MIROC_ESM_CHEM_map_mar45,
                                              MIROC5_map_mar45,
                                              MRI_CGCM3_map_mar45,
                                              NorESM1_M_map_mar45),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

accum_mar45_figs<- annotate_figure(p = accum_mar45_figs,
                   top = text_grob(label = "Median accumulated chill units by March 1 under RCP 4.5",
                                   face = "bold", size = 18))

ggsave(filename = "accum_mar45.png", plot = accum_mar45_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "mar45"))

######################################################
# April, rcp 45

accum_apr45_min <- min(df_45$median_A1)
accum_apr45_max <- max(df_45$median_A1)

accum_apr45_min = chill_min
accum_apr45_max = chill_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "apr45", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp45", month_col = "median_A1",
                    min = accum_apr45_min, max = accum_apr45_max)}
          )
}

observed_map_apr45 <- observed_hist_map(min = accum_apr45_min, max = accum_apr45_max,
                                        month_col = "median_A1")

ensemble_map_apr45 <- ensemble_map(scenario_name = "rcp45", 
                                   month_col = "median_A1", min = accum_apr45_min,
                                   max = accum_apr45_max)

# Need to add historical observed to this:
accum_apr45_figs <- ggarrange(plotlist = list(observed_map_apr45,
                                              ensemble_map_apr45,
                                              bcc_csm1_1_m_map_apr45,
                                              bcc_csm1_1_map_apr45,
                                              BNU_ESM_map_apr45,
                                              CanESM2_map_apr45,
                                              CCSM4_map_apr45,
                                              CNRM_CM5_map_apr45,
                                              CSIRO_Mk3_6_0_map_apr45,
                                              GFDL_ESM2G_map_apr45,
                                              GFDL_ESM2M_map_apr45,
                                              HadGEM2_CC365_map_apr45,
                                              HadGEM2_ES365_map_apr45,
                                              inmcm4_map_apr45,
                                              IPSL_CM5A_LR_map_apr45, 
                                              IPSL_CM5A_MR_map_apr45,
                                              IPSL_CM5B_LR_map_apr45,
                                              MIROC_ESM_CHEM_map_apr45,
                                              MIROC5_map_apr45,
                                              MRI_CGCM3_map_apr45,
                                              NorESM1_M_map_apr45),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

accum_apr45_figs <- annotate_figure(p = accum_apr45_figs,
                                    top = text_grob(label = "Median accumulated chill units by April 1 under RCP 4.5",
                                    face = "bold", size = 18))

ggsave(filename = "accum_apr45.png", plot = accum_apr45_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "apr45"))

# 3b. RCP 85 monthly accum figs --------------------------------------------
################################################
################################################ January, rcp85
################################################
# Going to make a common scale using min/max of all models in Jan @ rcp85 OR
# either modeled or observed historical

df_85 <- filter(stats_comp, scenario == "rcp85" | scenario == "historical")

accum_jan85_min <- min(df_85$median_J1)
accum_jan85_max <- max(df_85$median_J1)

# accum_jan85_min = chill_min 
# accum_jan85_max = chill_max

accum_jan85_min = 31
accum_jan85_max = 75

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "jan85", sep="_"),
         value ={model_map(model = h, scenario_name = "rcp85", month_col = "median_J1",
                     min = accum_jan85_min, max = accum_jan85_max)}
         )
}

observed_map_jan85 <- observed_hist_map(min = accum_jan85_min, max = accum_jan85_max,
                                        month_col = "median_J1")

ensemble_map_jan85 <- ensemble_map(scenario_name = "rcp85", 
                                   month_col = "median_J1", min = accum_jan85_min,
                                   max = accum_jan85_max)

hist_ensemble_jan85_figs <- ggarrange(plotlist = list(observed_map_jan85, ensemble_map_jan85),
                                      ncol = 2, widths = c(0.26, 1), 
                                      nrow = 1,
                                      common.legend = TRUE)
annotate_figure(hist_ensemble_jan85_figs,
               # top = text_grob("Visualizing Tooth Growth", color = "red", face = "bold", size = 14),
               # bottom = text_grob("Data source: \n ToothGrowth data set", color = "blue",
               #                    hjust = 1, x = 1, face = "italic", size = 10),
               # left = text_grob("Figure arranged using ggpubr", color = "green", rot = 90),
               # right = "I'm done, thanks :-)!",
               fig.lab = "RCP 8.5", fig.lab.face = "bold")

ggsave(filename = paste0("accum_", start, "_jan_85_obs_ensemble_mean_models_median_years.png"), 
       plot = hist_ensemble_jan85_figs, 
       device = "png",
       width = 10, height = 3, units = "in", 
       dpi=600, path=plot_path)

# Need to add historical observed to this:
accum_jan85_figs <- ggarrange(plotlist = list(observed_map_jan85,
                                              ensemble_map_jan85,
                                              bcc_csm1_1_m_map_jan85,
                                              bcc_csm1_1_map_jan85,
                                              BNU_ESM_map_jan85,
                                              CanESM2_map_jan85,
                                              CCSM4_map_jan85,
                                              CNRM_CM5_map_jan85,
                                              CSIRO_Mk3_6_0_map_jan85,
                                              GFDL_ESM2G_map_jan85,
                                              GFDL_ESM2M_map_jan85,
                                              HadGEM2_CC365_map_jan85,
                                              HadGEM2_ES365_map_jan85,
                                              inmcm4_map_jan85,
                                              IPSL_CM5A_LR_map_jan85, 
                                              IPSL_CM5A_MR_map_jan85,
                                              IPSL_CM5B_LR_map_jan85,
                                              MIROC_ESM_CHEM_map_jan85,
                                              MIROC5_map_jan85,
                                              MRI_CGCM3_map_jan85,
                                              NorESM1_M_map_jan85),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

accum_jan85_figs<- annotate_figure(p = accum_jan85_figs,
                   top = text_grob(label = "Median accumulated chill units by Jan 1 under RCP 8.5",
                                   face = "bold", size = 18))

ggsave(filename = "accum_jan85.png", 
       plot = accum_jan85_figs, device = "png",
       width = 15, height = 40, units = "in", path=plot_path)

rm(list = ls(pattern = "jan85"))
################################################
################################################ February, rcp 85
################################################

accum_feb85_min <- min(df_85$median_F1)
accum_feb85_max <- max(df_85$median_F1)

accum_feb85_min = chill_min
accum_feb85_max = chill_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "feb85", sep="_"),
         value = {model_map(model = h, scenario_name = "rcp85", month_col = "median_F1",
                    min = accum_feb85_min, max = accum_feb85_max)})
}

observed_map_feb85 <- observed_hist_map(min = accum_feb85_min, max = accum_feb85_max,
                                        month_col = "median_F1")

ensemble_map_feb85 <- ensemble_map(scenario_name = "rcp85", 
                                   month_col = "median_F1", min = accum_feb85_min,
                                   max = accum_feb85_max)
# Need to add historical observed to this:
accum_feb85_figs <- ggarrange(plotlist = list(observed_map_feb85,
                                              ensemble_map_feb85,
                                              bcc_csm1_1_m_map_feb85,
                                              bcc_csm1_1_map_feb85,
                                              BNU_ESM_map_feb85,
                                              CanESM2_map_feb85,
                                              CCSM4_map_feb85,
                                              CNRM_CM5_map_feb85,
                                              CSIRO_Mk3_6_0_map_feb85,
                                              GFDL_ESM2G_map_feb85,
                                              GFDL_ESM2M_map_feb85,
                                              HadGEM2_CC365_map_feb85,
                                              HadGEM2_ES365_map_feb85,
                                              inmcm4_map_feb85,
                                              IPSL_CM5A_LR_map_feb85, 
                                              IPSL_CM5A_MR_map_feb85,
                                              IPSL_CM5B_LR_map_feb85,
                                              MIROC_ESM_CHEM_map_feb85,
                                              MIROC5_map_feb85,
                                              MRI_CGCM3_map_feb85,
                                              NorESM1_M_map_feb85),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

accum_feb85_figs<- annotate_figure(p = accum_feb85_figs,
                   top = text_grob(label = "Median accumulated chill units by Feb 1 under RCP 8.5",
                                   face = "bold", size = 18))

ggsave(filename = "accum-feb85.png", plot = accum_feb85_figs, device = "png",
       width = 15, height = 40, units = "in", path=plot_path)

rm(list = ls(pattern = "feb85"))
################################################ March, rcp 85

accum_mar85_min <- min(df_85$median_M1)
accum_mar85_max <- max(df_85$median_M1)

accum_mar85_min = chill_min
accum_mar85_max = chill_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "mar85", sep="_"),
         value ={model_map(model = h, scenario_name = "rcp85", month_col = "median_M1",
                     min = accum_mar85_min, max = accum_mar85_max)})
}

observed_map_mar85 <- observed_hist_map(min = accum_mar85_min, max = accum_mar85_max,
                                        month_col = "median_M1")

ensemble_map_mar85 <- ensemble_map(scenario_name = "rcp85", 
                                   month_col = "median_M1", min = accum_mar85_min,
                                   max = accum_mar85_max)

# Need to add historical observed to this:
accum_mar85_figs <- ggarrange(plotlist = list(observed_map_mar85,
                                              ensemble_map_mar85,
                                              bcc_csm1_1_m_map_mar85,
                                              bcc_csm1_1_map_mar85,
                                              BNU_ESM_map_mar85,
                                              CanESM2_map_mar85,
                                              CCSM4_map_mar85,
                                              CNRM_CM5_map_mar85,
                                              CSIRO_Mk3_6_0_map_mar85,
                                              GFDL_ESM2G_map_mar85,
                                              GFDL_ESM2M_map_mar85,
                                              HadGEM2_CC365_map_mar85,
                                              HadGEM2_ES365_map_mar85,
                                              inmcm4_map_mar85,
                                              IPSL_CM5A_LR_map_mar85, 
                                              IPSL_CM5A_MR_map_mar85,
                                              IPSL_CM5B_LR_map_mar85,
                                              MIROC_ESM_CHEM_map_mar85,
                                              MIROC5_map_mar85,
                                              MRI_CGCM3_map_mar85,
                                              NorESM1_M_map_mar85),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

accum_mar85_figs<- annotate_figure(p = accum_mar85_figs,
                   top = text_grob(label = "Median accumulated chill units by March 1 under RCP 8.5",
                                   face = "bold", size = 18))

ggsave(filename = "accum-mar85.png", plot = accum_mar85_figs, device = "png",
       width = 15, height = 40, units = "in", path=plot_path)

rm(list = ls(pattern = "mar85"))
################################################ April, rcp 85
accum_apr85_min <- min(df_85$median_A1)
accum_apr85_max <- max(df_85$median_A1)

accum_apr85_min = chill_min
accum_apr85_max = chill_max

for(h in unique(stats_comp$model)) {  
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "apr85", sep="_"),
         value ={model_map(model = h, scenario_name = "rcp85", month_col = "median_A1",
                     min = accum_apr85_min, max = accum_apr85_max)})
}

observed_map_apr85 <- observed_hist_map(min = accum_apr85_min, max = accum_apr85_max,
                                        month_col = "median_A1")

ensemble_map_apr85 <- ensemble_map(scenario_name = "rcp85", 
                                   month_col = "median_A1", min = accum_apr85_min,
                                   max = accum_apr85_max)

# Need to add historical observed to this:
accum_apr85_figs <- ggarrange(plotlist = list(observed_map_apr85,
                                              ensemble_map_apr85,
                                              bcc_csm1_1_m_map_apr85,
                                              bcc_csm1_1_map_apr85,
                                              BNU_ESM_map_apr85,
                                              CanESM2_map_apr85,
                                              CCSM4_map_apr85,
                                              CNRM_CM5_map_apr85,
                                              CSIRO_Mk3_6_0_map_apr85,
                                              GFDL_ESM2G_map_apr85,
                                              GFDL_ESM2M_map_apr85,
                                              HadGEM2_CC365_map_apr85,
                                              HadGEM2_ES365_map_apr85,
                                              inmcm4_map_apr85,
                                              IPSL_CM5A_LR_map_apr85, 
                                              IPSL_CM5A_MR_map_apr85,
                                              IPSL_CM5B_LR_map_apr85,
                                              MIROC_ESM_CHEM_map_apr85,
                                              MIROC5_map_apr85,
                                              MRI_CGCM3_map_apr85,
                                              NorESM1_M_map_apr85),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

accum_apr85_figs <- annotate_figure(p = accum_apr85_figs,
                   top = text_grob(label = "Median accumulated chill units by April 1 under RCP 8.5",
                                   face = "bold", size = 18))

ggsave(filename = "accum-apr85.png", plot = accum_apr85_figs, device = "png",
       width = 15, height = 40, units = "in", path=plot_path)

rm(list = ls(pattern = "apr85"))
###################################################################################
###########************************************************************************
###########
###########               Threshold Figs. :(
###########
###########************************************************************************
###################################################################################

# 3c. RCP45 threshold figs ------------------------------------------------

##############################################################################
#############
############# Thresh 20, RCP 45
#############
thresh20_45_min <- min(df_45$median_20)
thresh20_45_max <- max(df_45$median_20)

thresh20_45_min = thresh_min
thresh20_45_min = thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh20_45", sep="_"),
         value ={model_map(model = h, scenario_name = "rcp45", month_col = "median_20",
                    min = thresh20_45_min, max = thresh20_45_max)})
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh20_45 <- observed_hist_map(min = thresh20_45_min,
                                              max = thresh20_45_max,
                                              month_col = "median_20")

ensemble_map_thresh20_45 <- ensemble_map(scenario_name = "rcp45", 
                                   month_col = "median_20",
                                   min = thresh20_45_min,
                                   max = thresh20_45_max)

# Need to add historical observed to this:
thresh20_45_figs <- ggarrange(plotlist = list(observed_map_thresh20_45,
                                              ensemble_map_thresh20_45,
                                              bcc_csm1_1_m_map_thresh20_45,
                                              bcc_csm1_1_map_thresh20_45,
                                              BNU_ESM_map_thresh20_45,
                                              CanESM2_map_thresh20_45,
                                              CCSM4_map_thresh20_45, 
                                              CNRM_CM5_map_thresh20_45,
                                              CSIRO_Mk3_6_0_map_thresh20_45,
                                              GFDL_ESM2G_map_thresh20_45,
                                              GFDL_ESM2M_map_thresh20_45,
                                              HadGEM2_CC365_map_thresh20_45,
                                              HadGEM2_ES365_map_thresh20_45,
                                              inmcm4_map_thresh20_45,
                                              IPSL_CM5A_LR_map_thresh20_45, 
                                              IPSL_CM5A_MR_map_thresh20_45,
                                              IPSL_CM5B_LR_map_thresh20_45,
                                              MIROC_ESM_CHEM_map_thresh20_45,
                                              MIROC5_map_thresh20_45, 
                                              MRI_CGCM3_map_thresh20_45,
                                              NorESM1_M_map_thresh20_45),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)
lab = "Median days to reach 20 chill unit threshold under RCP 4.5"
thresh20_45_figs <- annotate_figure(p = thresh20_45_figs,
                                    top = text_grob(label = lab, 
                                                    face = "bold", size = 18))

ggsave(filename = "thresh20_45.png", plot = thresh20_45_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh20_45"))
##############################################################################
#############
############# Thresh 25, RCP 45
#############
thresh25_45_min <- min(df_45$median_25)
thresh25_45_max <- max(df_45$median_25)

thresh25_45_min <- thresh_min
thresh25_45_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh25_45", sep="_"),
         value ={model_map(model = h, scenario_name = "rcp45", month_col = "median_25",
                    min = thresh25_45_min, max = thresh25_45_max)})
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh25_45 <- observed_hist_map(min = thresh25_45_min,
                                              max = thresh25_45_max,
                                              month_col = "median_20")

ensemble_map_thresh25_45 <- ensemble_map(scenario_name = "rcp45", 
                                         month_col = "median_25",
                                         min = thresh25_45_min,
                                         max = thresh25_45_max)

# Need to add historical observed to this:
thresh25_45_figs <- ggarrange(plotlist = list(observed_map_thresh25_45,
                                              ensemble_map_thresh25_45,
                                              bcc_csm1_1_m_map_thresh25_45,
                                              bcc_csm1_1_map_thresh25_45,
                                              BNU_ESM_map_thresh25_45,
                                              CanESM2_map_thresh25_45,
                                              CCSM4_map_thresh25_45, 
                                              CNRM_CM5_map_thresh25_45,
                                              CSIRO_Mk3_6_0_map_thresh25_45,
                                              GFDL_ESM2G_map_thresh25_45,
                                              GFDL_ESM2M_map_thresh25_45,
                                              HadGEM2_CC365_map_thresh25_45,
                                              HadGEM2_ES365_map_thresh25_45,
                                              inmcm4_map_thresh25_45,
                                              IPSL_CM5A_LR_map_thresh25_45, 
                                              IPSL_CM5A_MR_map_thresh25_45,
                                              IPSL_CM5B_LR_map_thresh25_45,
                                              MIROC_ESM_CHEM_map_thresh25_45,
                                              MIROC5_map_thresh25_45, 
                                              MRI_CGCM3_map_thresh25_45,
                                              NorESM1_M_map_thresh25_45),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)
lab = "Median days to reach 25 chill unit threshold under RCP 4.5"
thresh25_45_figs <- annotate_figure(p = thresh25_45_figs,
                                    top = text_grob(label = lab, 
                                                    face = "bold", size = 18))

ggsave(filename = "thresh25_45.png", plot = thresh25_45_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh25_45"))
##############################################################################
#############
############# Thresh 30, RCP 45
#############

thresh30_45_min <- min(df_45$median_30)
thresh30_45_max <- max(df_45$median_30)

thresh30_45_min = thresh_min
thresh30_45_max = thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh30_45", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp45", month_col = "median_30",
                    min = thresh30_45_min, max = thresh30_45_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh30_45 <- observed_hist_map(min = thresh30_45_min,
                                              max = thresh30_45_max,
                                              month_col = "median_30")

ensemble_map_thresh30_45 <- ensemble_map(scenario_name = "rcp45", 
                                         month_col = "median_30",
                                         min = thresh30_45_min,
                                         max = thresh30_45_max)

# Need to add historical observed to this:
thresh30_45_figs <- ggarrange(plotlist = list(observed_map_thresh30_45,
                                              ensemble_map_thresh30_45,
                                              bcc_csm1_1_m_map_thresh30_45,
                                              bcc_csm1_1_map_thresh30_45,
                                              BNU_ESM_map_thresh30_45,
                                              CanESM2_map_thresh30_45,
                                              CCSM4_map_thresh30_45, 
                                              CNRM_CM5_map_thresh30_45,
                                              CSIRO_Mk3_6_0_map_thresh30_45,
                                              GFDL_ESM2G_map_thresh30_45,
                                              GFDL_ESM2M_map_thresh30_45,
                                              HadGEM2_CC365_map_thresh30_45,
                                              HadGEM2_ES365_map_thresh30_45,
                                              inmcm4_map_thresh30_45,
                                              IPSL_CM5A_LR_map_thresh30_45, 
                                              IPSL_CM5A_MR_map_thresh30_45,
                                              IPSL_CM5B_LR_map_thresh30_45,
                                              MIROC_ESM_CHEM_map_thresh30_45,
                                              MIROC5_map_thresh30_45, 
                                              MRI_CGCM3_map_thresh30_45,
                                              NorESM1_M_map_thresh30_45),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)
lab = "Median days to reach 30 chill unit threshold under RCP 4.5"
thresh30_45_figs <- annotate_figure(p = thresh30_45_figs,
                                    top = text_grob(label = lab, 
                                                    face = "bold", size = 18))

ggsave(filename = "thresh30_45.png", plot = thresh30_45_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh30_45"))
##############################################################################
#############
############# Thresh 35, RCP 45
#############

thresh35_45_min <- min(df_45$median_35)
thresh35_45_max <- max(df_45$median_35)

thresh35_45_min = thresh_min
thresh35_45_max = thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh35_45", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp45", month_col = "median_35",
                    min = thresh35_45_min, max = thresh35_45_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh35_45 <- observed_hist_map(min = thresh35_45_min,
                                              max = thresh35_45_max,
                                              month_col = "median_20")

ensemble_map_thresh35_45 <- ensemble_map(scenario_name = "rcp45", 
                                         month_col = "median_35",
                                         min = thresh35_45_min,
                                         max = thresh35_45_max)

# Need to add historical observed to this:
thresh35_45_figs <- ggarrange(plotlist = list(observed_map_thresh35_45,
                                              ensemble_map_thresh35_45,
                                              bcc_csm1_1_m_map_thresh35_45,
                                              bcc_csm1_1_map_thresh35_45,
                                              BNU_ESM_map_thresh35_45,
                                              CanESM2_map_thresh35_45,
                                              CCSM4_map_thresh35_45, 
                                              CNRM_CM5_map_thresh35_45,
                                              CSIRO_Mk3_6_0_map_thresh35_45,
                                              GFDL_ESM2G_map_thresh35_45,
                                              GFDL_ESM2M_map_thresh35_45,
                                              HadGEM2_CC365_map_thresh35_45,
                                              HadGEM2_ES365_map_thresh35_45,
                                              inmcm4_map_thresh35_45,
                                              IPSL_CM5A_LR_map_thresh35_45, 
                                              IPSL_CM5A_MR_map_thresh35_45,
                                              IPSL_CM5B_LR_map_thresh35_45,
                                              MIROC_ESM_CHEM_map_thresh35_45,
                                              MIROC5_map_thresh35_45, 
                                              MRI_CGCM3_map_thresh35_45,
                                              NorESM1_M_map_thresh35_45),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)
lab = "Median days to reach 35 chill unit threshold under RCP 4.5"
thresh35_45_figs <- annotate_figure(p = thresh35_45_figs,
                                    top = text_grob(label = lab, 
                                                    face = "bold", size = 18))

ggsave(filename = "thresh35_45.png", plot = thresh35_45_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh35_45"))
##############################################################################
#############
############# Thresh 40, RCP 45
#############

thresh40_45_min <- min(df_45$median_40)
thresh40_45_max <- max(df_45$median_40)

thresh40_45_min = thresh_min
thresh40_45_max = thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh40_45", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp45", month_col = "median_40",
                    min = thresh40_45_min, max = thresh40_45_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh40_45 <- observed_hist_map(min = thresh40_45_min,
                                              max = thresh40_45_max,
                                              month_col = "median_40")

ensemble_map_thresh40_45 <- ensemble_map(scenario_name = "rcp45", 
                                         month_col = "median_40",
                                         min = thresh40_45_min,
                                         max = thresh40_45_max)

# Need to add historical observed to this:
thresh40_45_figs <- ggarrange(plotlist = list(observed_map_thresh40_45,
                                              ensemble_map_thresh40_45,
                                              bcc_csm1_1_m_map_thresh40_45,
                                              bcc_csm1_1_map_thresh40_45,
                                              BNU_ESM_map_thresh40_45,
                                              CanESM2_map_thresh40_45,
                                              CCSM4_map_thresh40_45, 
                                              CNRM_CM5_map_thresh40_45,
                                              CSIRO_Mk3_6_0_map_thresh40_45,
                                              GFDL_ESM2G_map_thresh40_45,
                                              GFDL_ESM2M_map_thresh40_45,
                                              HadGEM2_CC365_map_thresh40_45,
                                              HadGEM2_ES365_map_thresh40_45,
                                              inmcm4_map_thresh40_45,
                                              IPSL_CM5A_LR_map_thresh40_45, 
                                              IPSL_CM5A_MR_map_thresh40_45,
                                              IPSL_CM5B_LR_map_thresh40_45,
                                              MIROC_ESM_CHEM_map_thresh40_45,
                                              MIROC5_map_thresh40_45, 
                                              MRI_CGCM3_map_thresh40_45,
                                              NorESM1_M_map_thresh40_45),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)
lab = "Median days to reach 40 chill unit threshold under RCP 4.5"
thresh40_45_figs <- annotate_figure(p = thresh40_45_figs,
                                    top = text_grob(label = lab, 
                                                    face = "bold", size = 18))

ggsave(filename = "thresh40_45.png", plot = thresh40_45_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh40_45"))
##############################################################################
#############
############# Thresh 45, RCP 45
#############
thresh45_45_min <- min(df_45$median_45)
thresh45_45_max <- max(df_45$median_45)

thresh45_45_min = thresh_min
thresh45_45_max = thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh45_45", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp45", month_col = "median_45",
                    min = thresh45_45_min, max = thresh45_45_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh45_45 <- observed_hist_map(min = thresh45_45_min,
                                              max = thresh45_45_max,
                                              month_col = "median_20")

ensemble_map_thresh45_45 <- ensemble_map(scenario_name = "rcp45", 
                                         month_col = "median_45",
                                         min = thresh45_45_min,
                                         max = thresh45_45_max)

# Need to add historical observed to this:
thresh45_45_figs <- ggarrange(plotlist = list(observed_map_thresh45_45,
                                              ensemble_map_thresh45_45,
                                              bcc_csm1_1_m_map_thresh45_45,
                                              bcc_csm1_1_map_thresh45_45,
                                              BNU_ESM_map_thresh45_45,
                                              CanESM2_map_thresh45_45,
                                              CCSM4_map_thresh45_45, 
                                              CNRM_CM5_map_thresh45_45,
                                              CSIRO_Mk3_6_0_map_thresh45_45,
                                              GFDL_ESM2G_map_thresh45_45,
                                              GFDL_ESM2M_map_thresh45_45,
                                              HadGEM2_CC365_map_thresh45_45,
                                              HadGEM2_ES365_map_thresh45_45,
                                              inmcm4_map_thresh45_45,
                                              IPSL_CM5A_LR_map_thresh45_45, 
                                              IPSL_CM5A_MR_map_thresh45_45,
                                              IPSL_CM5B_LR_map_thresh45_45,
                                              MIROC_ESM_CHEM_map_thresh45_45,
                                              MIROC5_map_thresh45_45, 
                                              MRI_CGCM3_map_thresh45_45,
                                              NorESM1_M_map_thresh45_45),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)
lab = "Median days to reach 45 chill unit threshold under RCP 4.5"
thresh45_45_figs <- annotate_figure(p = thresh45_45_figs,
                                    top = text_grob(label = lab, 
                                                    face = "bold", size = 18))

ggsave(filename = "thresh45_45.png", plot = thresh45_45_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)
rm(list = ls(pattern = "thresh45_45"))
##############################################################################
#############
############# Thresh 50, RCP 45
#############
thresh50_45_min <- min(df_45$median_50)
thresh50_45_max <- max(df_45$median_50)

thresh50_45_min <- thresh_min
thresh50_45_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh50_45", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp45", month_col = "median_50",
                    min = thresh50_45_min, max = thresh50_45_max)}
         )
}

# Separately created maps for observed historical and ensemble mean
observed_map_thresh50_45 <- observed_hist_map(min = thresh50_45_min,
                                              max = thresh50_45_max,
                                              month_col = "median_50")

ensemble_map_thresh50_45 <- ensemble_map(scenario_name = "rcp45", 
                                   month_col = "median_50",
                                   min = thresh50_45_min,
                                   max = thresh50_45_max)

# Need to add historical observed to this:
thresh50_45_figs <- ggarrange(plotlist = list(observed_map_thresh50_45,
                                              ensemble_map_thresh50_45,
                                              bcc_csm1_1_m_map_thresh50_45,
                                              bcc_csm1_1_map_thresh50_45,
                                              BNU_ESM_map_thresh50_45,
                                              CanESM2_map_thresh50_45,
                                              CCSM4_map_thresh50_45, 
                                              CNRM_CM5_map_thresh50_45,
                                              CSIRO_Mk3_6_0_map_thresh50_45,
                                              GFDL_ESM2G_map_thresh50_45,
                                              GFDL_ESM2M_map_thresh50_45,
                                              HadGEM2_CC365_map_thresh50_45,
                                              HadGEM2_ES365_map_thresh50_45,
                                              inmcm4_map_thresh50_45,
                                              IPSL_CM5A_LR_map_thresh50_45, 
                                              IPSL_CM5A_MR_map_thresh50_45,
                                              IPSL_CM5B_LR_map_thresh50_45,
                                              MIROC_ESM_CHEM_map_thresh50_45,
                                              MIROC5_map_thresh50_45, 
                                              MRI_CGCM3_map_thresh50_45,
                                              NorESM1_M_map_thresh50_45),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

thresh50_45_figs <- annotate_figure(p = thresh50_45_figs,
                                    top = text_grob(label = "Median days to reach 50 chill unit threshold under rcp 45",
                                                    face = "bold", size = 18))

ggsave(filename = "thresh50_45.png", plot = thresh50_45_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh50_45"))
##############################################################################
#############
############# Thresh 55, RCP 45
#############
thresh55_45_min <- min(df_45$median_55)
thresh55_45_max <- max(df_45$median_55)

thresh55_45_min <- thresh_min
thresh55_45_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh55_45", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp45", month_col = "median_55",
                    min = thresh55_45_min, max = thresh55_45_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh55_45 <- observed_hist_map(min = thresh55_45_min,
                                              max = thresh55_45_max,
                                              month_col = "median_55")
ensemble_map_thresh55_45 <- ensemble_map(scenario_name = "rcp45", 
                                   month_col = "median_55",
                                   min = thresh55_45_min,
                                   max = thresh55_45_max)

# Need to add historical observed to this:
thresh55_45_figs <- ggarrange(plotlist = list(observed_map_thresh55_45,
                                              ensemble_map_thresh55_45,
                                              bcc_csm1_1_m_map_thresh55_45,
                                              bcc_csm1_1_map_thresh55_45,
                                              BNU_ESM_map_thresh55_45,
                                              CanESM2_map_thresh55_45,
                                              CCSM4_map_thresh55_45, 
                                              CNRM_CM5_map_thresh55_45,
                                              CSIRO_Mk3_6_0_map_thresh55_45,
                                              GFDL_ESM2G_map_thresh55_45,
                                              GFDL_ESM2M_map_thresh55_45,
                                              HadGEM2_CC365_map_thresh55_45,
                                              HadGEM2_ES365_map_thresh55_45,
                                              inmcm4_map_thresh55_45,
                                              IPSL_CM5A_LR_map_thresh55_45, 
                                              IPSL_CM5A_MR_map_thresh55_45,
                                              IPSL_CM5B_LR_map_thresh55_45,
                                              MIROC_ESM_CHEM_map_thresh55_45,
                                              MIROC5_map_thresh55_45, 
                                              MRI_CGCM3_map_thresh55_45,
                                              NorESM1_M_map_thresh55_45),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

thresh55_45_figs <- annotate_figure(p = thresh55_45_figs,
                                    top = text_grob(label = "Median days to reach 55 chill unit threshold under rcp 45",
                                                    face = "bold", size = 18))

ggsave(filename = "thresh55_45.png", plot = thresh55_45_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh55_45"))
##############################################################################
#############
############# Thresh 60, RCP 45
#############
thresh60_45_min <- min(df_45$median_60)
thresh60_45_max <- max(df_45$median_60)

thresh60_45_min <- thresh_min
thresh60_45_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh60_45", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp45", month_col = "median_60",
                    min = thresh60_45_min, max = thresh60_45_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh60_45 <- observed_hist_map(min = thresh60_45_min,
                                              max = thresh60_45_max,
                                              month_col = "median_60")
ensemble_map_thresh60_45 <- ensemble_map(scenario_name = "rcp45", 
                                   month_col = "median_60",
                                   min = thresh60_45_min,
                                   max = thresh60_45_max)

# Need to add historical observed to this:
thresh60_45_figs <- ggarrange(plotlist = list(observed_map_thresh60_45,
                                              ensemble_map_thresh60_45,
                                              bcc_csm1_1_m_map_thresh60_45,
                                              bcc_csm1_1_map_thresh60_45,
                                              BNU_ESM_map_thresh60_45,
                                              CanESM2_map_thresh60_45,
                                              CCSM4_map_thresh60_45, 
                                              CNRM_CM5_map_thresh60_45,
                                              CSIRO_Mk3_6_0_map_thresh60_45,
                                              GFDL_ESM2G_map_thresh60_45,
                                              GFDL_ESM2M_map_thresh60_45,
                                              HadGEM2_CC365_map_thresh60_45,
                                              HadGEM2_ES365_map_thresh60_45,
                                              inmcm4_map_thresh60_45,
                                              IPSL_CM5A_LR_map_thresh60_45, 
                                              IPSL_CM5A_MR_map_thresh60_45,
                                              IPSL_CM5B_LR_map_thresh60_45,
                                              MIROC_ESM_CHEM_map_thresh60_45,
                                              MIROC5_map_thresh60_45, 
                                              MRI_CGCM3_map_thresh60_45,
                                              NorESM1_M_map_thresh60_45),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

thresh60_45_figs <- annotate_figure(p = thresh60_45_figs,
                                    top = text_grob(label = "Median days to reach 60 chill unit threshold under rcp 45",
                                                    face = "bold", size = 18))

ggsave(filename = "thresh60_45.png", plot = thresh60_45_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh60_45"))
##############################################################################
#############
############# Thresh 65, RCP 45
#############
thresh65_45_min <- min(df_45$median_65)
thresh65_45_max <- max(df_45$median_65)

thresh65_45_min <- thresh_min
thresh65_45_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh65_45", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp45", month_col = "median_65",
                    min = thresh65_45_min, max = thresh65_45_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh65_45 <- observed_hist_map(min = thresh65_45_min,
                                              max = thresh65_45_max,
                                              month_col = "median_65")
ensemble_map_thresh65_45 <- ensemble_map(scenario_name = "rcp45", 
                                   month_col = "median_65",
                                   min = thresh65_45_min,
                                   max = thresh65_45_max)

# Need to add historical observed to this:
thresh65_45_figs <- ggarrange(plotlist = list(observed_map_thresh65_45,
                                              ensemble_map_thresh65_45,
                                              bcc_csm1_1_m_map_thresh65_45,
                                              bcc_csm1_1_map_thresh65_45,
                                              BNU_ESM_map_thresh65_45,
                                              CanESM2_map_thresh65_45,
                                              CCSM4_map_thresh65_45, 
                                              CNRM_CM5_map_thresh65_45,
                                              CSIRO_Mk3_6_0_map_thresh65_45,
                                              GFDL_ESM2G_map_thresh65_45,
                                              GFDL_ESM2M_map_thresh65_45,
                                              HadGEM2_CC365_map_thresh65_45,
                                              HadGEM2_ES365_map_thresh65_45,
                                              inmcm4_map_thresh65_45,
                                              IPSL_CM5A_LR_map_thresh65_45, 
                                              IPSL_CM5A_MR_map_thresh65_45,
                                              IPSL_CM5B_LR_map_thresh65_45,
                                              MIROC_ESM_CHEM_map_thresh65_45,
                                              MIROC5_map_thresh65_45, 
                                              MRI_CGCM3_map_thresh65_45,
                                              NorESM1_M_map_thresh65_45),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

thresh65_45_figs <- annotate_figure(p = thresh65_45_figs,
                                    top = text_grob(label = "Median days to reach 65 chill unit threshold under rcp 45",
                                                    face = "bold", size = 18))

ggsave(filename = "thresh65_45.png", plot = thresh65_45_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh65_45"))
##############################################################################
#############
############# Thresh 70, RCP 45
#############
thresh70_45_min <- min(df_45$median_70)
thresh70_45_max <- max(df_45$median_70)

thresh70_45_min <- thresh_min
thresh70_45_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh70_45", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp45", month_col = "median_70",
                    min = thresh70_45_min, max = thresh70_45_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh70_45 <- observed_hist_map(min = thresh70_45_min,
                                              max = thresh70_45_max,
                                              month_col = "median_60")
ensemble_map_thresh70_45 <- ensemble_map(scenario_name = "rcp45", 
                                   month_col = "median_60",
                                   min = thresh70_45_min,
                                   max = thresh70_45_max)

# Need to add historical observed to this:
thresh70_45_figs <- ggarrange(plotlist = list(observed_map_thresh70_45,
                                              ensemble_map_thresh70_45,
                                              bcc_csm1_1_m_map_thresh70_45,
                                              bcc_csm1_1_map_thresh70_45,
                                              BNU_ESM_map_thresh70_45,
                                              CanESM2_map_thresh70_45,
                                              CCSM4_map_thresh70_45, 
                                              CNRM_CM5_map_thresh70_45,
                                              CSIRO_Mk3_6_0_map_thresh70_45,
                                              GFDL_ESM2G_map_thresh70_45,
                                              GFDL_ESM2M_map_thresh70_45,
                                              HadGEM2_CC365_map_thresh70_45,
                                              HadGEM2_ES365_map_thresh70_45,
                                              inmcm4_map_thresh70_45,
                                              IPSL_CM5A_LR_map_thresh70_45, 
                                              IPSL_CM5A_MR_map_thresh70_45,
                                              IPSL_CM5B_LR_map_thresh70_45,
                                              MIROC_ESM_CHEM_map_thresh70_45,
                                              MIROC5_map_thresh70_45, 
                                              MRI_CGCM3_map_thresh70_45,
                                              NorESM1_M_map_thresh70_45),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

thresh70_45_figs <- annotate_figure(p = thresh70_45_figs,
                                    top = text_grob(label = "Median days to reach 70 chill unit threshold under RCP 4.5",
                                                    face = "bold", size = 18))

ggsave(filename = "thresh70_45.png", plot = thresh70_45_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh70_45"))
##############################################################################
#############
############# Thresh 75, RCP 45
#############
thresh75_45_min <- min(df_45$median_75)
thresh75_45_max <- max(df_45$median_75)

thresh75_45_min <- thresh_min
thresh75_45_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh75_45", sep="_"),
         value =
         {
           model_map(model = h, scenario_name = "rcp45", month_col = "median_75",
                     min = thresh75_45_min, max = thresh75_45_max)
         }
  )
}

# Separately created maps for observed historical and ensemble mean
observed_map_thresh75_45 <- observed_hist_map(min = thresh75_45_min,
                                              max = thresh75_45_max,
                                              month_col = "median_75")

ensemble_map_thresh75_45 <- ensemble_map(scenario_name = "rcp45", 
                                         month_col = "median_75",
                                         min = thresh75_45_min,
                                         max = thresh75_45_max)

# Need to add historical observed to this:
thresh75_45_figs <- ggarrange(plotlist = list(observed_map_thresh75_45,
                                              ensemble_map_thresh75_45,
                                              bcc_csm1_1_m_map_thresh75_45,
                                              bcc_csm1_1_map_thresh75_45,
                                              BNU_ESM_map_thresh75_45,
                                              CanESM2_map_thresh75_45,
                                              CCSM4_map_thresh75_45, 
                                              CNRM_CM5_map_thresh75_45,
                                              CSIRO_Mk3_6_0_map_thresh75_45,
                                              GFDL_ESM2G_map_thresh75_45,
                                              GFDL_ESM2M_map_thresh75_45,
                                              HadGEM2_CC365_map_thresh75_45,
                                              HadGEM2_ES365_map_thresh75_45,
                                              inmcm4_map_thresh75_45,
                                              IPSL_CM5A_LR_map_thresh75_45, 
                                              IPSL_CM5A_MR_map_thresh75_45,
                                              IPSL_CM5B_LR_map_thresh75_45,
                                              MIROC_ESM_CHEM_map_thresh75_45,
                                              MIROC5_map_thresh75_45, 
                                              MRI_CGCM3_map_thresh75_45,
                                              NorESM1_M_map_thresh75_45),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

thresh75_45_figs <- annotate_figure(p = thresh75_45_figs,
                                    top = text_grob(label = "Median days to reach 75 chill unit threshold under rcp 45",
                                                    face = "bold", size = 18))

ggsave(filename = "thresh75_45.png", plot = thresh75_45_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)
rm(list = ls(pattern = "thresh75_45"))

#***********************                        ***********************
#*********************** RCP85 threshold figs   ***********************
#***********************                        ***********************
##############################################################################
#############
############# Thresh 20, RCP 85
#############
thresh20_85_min <- min(df_85$median_20)
thresh20_85_max <- max(df_85$median_20)

thresh20_85_min <- thresh_min
thresh20_85_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh20_85", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp85", month_col = "median_20",
                    min = thresh20_85_min, max = thresh20_85_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh20_85 <- observed_hist_map(min = thresh20_85_min,
                                              max = thresh20_85_max,
                                              month_col = "median_20")

ensemble_map_thresh20_85 <- ensemble_map(scenario_name = "rcp85", 
                                   month_col = "median_20",
                                   min = thresh20_85_min,
                                   max = thresh20_85_max)

# Need to add historical observed to this:
thresh20_85_figs <- ggarrange(plotlist = list(observed_map_thresh20_85,
                                              ensemble_map_thresh20_85,
                                              bcc_csm1_1_m_map_thresh20_85,
                                              bcc_csm1_1_map_thresh20_85,
                                              BNU_ESM_map_thresh20_85,
                                              CanESM2_map_thresh20_85,
                                              CCSM4_map_thresh20_85, 
                                              CNRM_CM5_map_thresh20_85,
                                              CSIRO_Mk3_6_0_map_thresh20_85,
                                              GFDL_ESM2G_map_thresh20_85,
                                              GFDL_ESM2M_map_thresh20_85,
                                              HadGEM2_CC365_map_thresh20_85,
                                              HadGEM2_ES365_map_thresh20_85,
                                              inmcm4_map_thresh20_85,
                                              IPSL_CM5A_LR_map_thresh20_85, 
                                              IPSL_CM5A_MR_map_thresh20_85,
                                              IPSL_CM5B_LR_map_thresh20_85,
                                              MIROC_ESM_CHEM_map_thresh20_85,
                                              MIROC5_map_thresh20_85, 
                                              MRI_CGCM3_map_thresh20_85,
                                              NorESM1_M_map_thresh20_85),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)
lab = "Median days to reach 20 chill unit threshold under RCP 8.5"
thresh20_85_figs <- annotate_figure(p = thresh20_85_figs,
                                    top = text_grob(label = lab, 
                                                    face = "bold", size = 18))

ggsave(filename = "thresh20_85.png", plot = thresh20_85_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh20_85"))
##############################################################################
#############
############# Thresh 25, RCP 85
#############
thresh25_85_min <- min(df_85$median_25)
thresh25_85_max <- max(df_85$median_25)

thresh25_85_min <- thresh_min
thresh25_85_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh25_85", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp85", month_col = "median_25",
                    min = thresh25_85_min, max = thresh25_85_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh25_85 <- observed_hist_map(min = thresh25_85_min,
                                              max = thresh25_85_max,
                                              month_col = "median_20")

ensemble_map_thresh25_85 <- ensemble_map(scenario_name = "rcp85", 
                                         month_col = "median_25",
                                         min = thresh25_85_min,
                                         max = thresh25_85_max)

# Need to add historical observed to this:
thresh25_85_figs <- ggarrange(plotlist = list(observed_map_thresh25_85,
                                              ensemble_map_thresh25_85,
                                              bcc_csm1_1_m_map_thresh25_85,
                                              bcc_csm1_1_map_thresh25_85,
                                              BNU_ESM_map_thresh25_85,
                                              CanESM2_map_thresh25_85,
                                              CCSM4_map_thresh25_85, 
                                              CNRM_CM5_map_thresh25_85,
                                              CSIRO_Mk3_6_0_map_thresh25_85,
                                              GFDL_ESM2G_map_thresh25_85,
                                              GFDL_ESM2M_map_thresh25_85,
                                              HadGEM2_CC365_map_thresh25_85,
                                              HadGEM2_ES365_map_thresh25_85,
                                              inmcm4_map_thresh25_85,
                                              IPSL_CM5A_LR_map_thresh25_85, 
                                              IPSL_CM5A_MR_map_thresh25_85,
                                              IPSL_CM5B_LR_map_thresh25_85,
                                              MIROC_ESM_CHEM_map_thresh25_85,
                                              MIROC5_map_thresh25_85, 
                                              MRI_CGCM3_map_thresh25_85,
                                              NorESM1_M_map_thresh25_85),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)
lab = "Median days to reach 25 chill unit threshold under RCP 8.5"
thresh25_85_figs <- annotate_figure(p = thresh25_85_figs,
                                    top = text_grob(label = lab, 
                                                    face = "bold", size = 18))

ggsave(filename = "thresh25_85.png", plot = thresh25_85_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)
rm(list = ls(pattern = "thresh25_85"))
##############################################################################
#############
############# Thresh 30, RCP 85
#############
thresh30_85_min <- min(df_85$median_30)
thresh30_85_max <- max(df_85$median_30)

thresh30_85_min <- thresh_min
thresh30_85_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh30_85", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp85", month_col = "median_30",
                    min = thresh30_85_min, max = thresh30_85_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh30_85 <- observed_hist_map(min = thresh30_85_min,
                                              max = thresh30_85_max,
                                              month_col = "median_30")

ensemble_map_thresh30_85 <- ensemble_map(scenario_name = "rcp85", 
                                         month_col = "median_30",
                                         min = thresh30_85_min,
                                         max = thresh30_85_max)

# Need to add historical observed to this:
thresh30_85_figs <- ggarrange(plotlist = list(observed_map_thresh30_85,
                                              ensemble_map_thresh30_85,
                                              bcc_csm1_1_m_map_thresh30_85,
                                              bcc_csm1_1_map_thresh30_85,
                                              BNU_ESM_map_thresh30_85,
                                              CanESM2_map_thresh30_85,
                                              CCSM4_map_thresh30_85, 
                                              CNRM_CM5_map_thresh30_85,
                                              CSIRO_Mk3_6_0_map_thresh30_85,
                                              GFDL_ESM2G_map_thresh30_85,
                                              GFDL_ESM2M_map_thresh30_85,
                                              HadGEM2_CC365_map_thresh30_85,
                                              HadGEM2_ES365_map_thresh30_85,
                                              inmcm4_map_thresh30_85,
                                              IPSL_CM5A_LR_map_thresh30_85, 
                                              IPSL_CM5A_MR_map_thresh30_85,
                                              IPSL_CM5B_LR_map_thresh30_85,
                                              MIROC_ESM_CHEM_map_thresh30_85,
                                              MIROC5_map_thresh30_85, 
                                              MRI_CGCM3_map_thresh30_85,
                                              NorESM1_M_map_thresh30_85),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)
lab = "Median days to reach 30 chill unit threshold under RCP 8.5"
thresh30_85_figs <- annotate_figure(p = thresh30_85_figs,
                                    top = text_grob(label = lab, 
                                                    face = "bold", size = 18))

ggsave(filename = "thresh30_85.png", plot = thresh30_85_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh30_85"))
##############################################################################
#############
############# Thresh 35, RCP 85
#############

thresh35_85_min <- min(df_85$median_35)
thresh35_85_max <- max(df_85$median_35)

thresh35_85_min <- thresh_min
thresh35_85_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh35_85", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp85", month_col = "median_35",
                    min = thresh35_85_min, max = thresh35_85_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh35_85 <- observed_hist_map(min = thresh35_85_min,
                                              max = thresh35_85_max,
                                              month_col = "median_20")

ensemble_map_thresh35_85 <- ensemble_map(scenario_name = "rcp85", 
                                         month_col = "median_35",
                                         min = thresh35_85_min,
                                         max = thresh35_85_max)

# Need to add historical observed to this:
thresh35_85_figs <- ggarrange(plotlist = list(observed_map_thresh35_85,
                                              ensemble_map_thresh35_85,
                                              bcc_csm1_1_m_map_thresh35_85,
                                              bcc_csm1_1_map_thresh35_85,
                                              BNU_ESM_map_thresh35_85,
                                              CanESM2_map_thresh35_85,
                                              CCSM4_map_thresh35_85, 
                                              CNRM_CM5_map_thresh35_85,
                                              CSIRO_Mk3_6_0_map_thresh35_85,
                                              GFDL_ESM2G_map_thresh35_85,
                                              GFDL_ESM2M_map_thresh35_85,
                                              HadGEM2_CC365_map_thresh35_85,
                                              HadGEM2_ES365_map_thresh35_85,
                                              inmcm4_map_thresh35_85,
                                              IPSL_CM5A_LR_map_thresh35_85, 
                                              IPSL_CM5A_MR_map_thresh35_85,
                                              IPSL_CM5B_LR_map_thresh35_85,
                                              MIROC_ESM_CHEM_map_thresh35_85,
                                              MIROC5_map_thresh35_85, 
                                              MRI_CGCM3_map_thresh35_85,
                                              NorESM1_M_map_thresh35_85),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)
lab = "Median days to reach 35 chill unit threshold under RCP 8.5"
thresh35_85_figs <- annotate_figure(p = thresh35_85_figs,
                                    top = text_grob(label = lab, 
                                                    face = "bold", size = 18))

ggsave(filename = "thresh35_85.png", plot = thresh35_85_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)
rm(list = ls(pattern = "thresh35_85"))
##############################################################################
#############
############# Thresh 40, RCP 85
#############
thresh40_85_min <- min(df_85$median_40)
thresh40_85_max <- max(df_85$median_40)

# thresh40_85_min <- thresh_min
# thresh40_85_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh40_85", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp85", month_col = "median_40",
                    min = thresh40_85_min, max = thresh40_85_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh40_85 <- observed_hist_map(min = thresh40_85_min,
                                              max = thresh40_85_max,
                                              month_col = "median_40")

ensemble_map_thresh40_85 <- ensemble_map(scenario_name = "rcp85", 
                                         month_col = "median_40",
                                         min = thresh40_85_min,
                                         max = thresh40_85_max)

# Need to add historical observed to this:
thresh40_85_figs <- ggarrange(plotlist = list(observed_map_thresh40_85,
                                              ensemble_map_thresh40_85,
                                              bcc_csm1_1_m_map_thresh40_85,
                                              bcc_csm1_1_map_thresh40_85,
                                              BNU_ESM_map_thresh40_85,
                                              CanESM2_map_thresh40_85,
                                              CCSM4_map_thresh40_85, 
                                              CNRM_CM5_map_thresh40_85,
                                              CSIRO_Mk3_6_0_map_thresh40_85,
                                              GFDL_ESM2G_map_thresh40_85,
                                              GFDL_ESM2M_map_thresh40_85,
                                              HadGEM2_CC365_map_thresh40_85,
                                              HadGEM2_ES365_map_thresh40_85,
                                              inmcm4_map_thresh40_85,
                                              IPSL_CM5A_LR_map_thresh40_85, 
                                              IPSL_CM5A_MR_map_thresh40_85,
                                              IPSL_CM5B_LR_map_thresh40_85,
                                              MIROC_ESM_CHEM_map_thresh40_85,
                                              MIROC5_map_thresh40_85, 
                                              MRI_CGCM3_map_thresh40_85,
                                              NorESM1_M_map_thresh40_85),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)
lab = "Median days to reach 40 chill unit threshold under RCP 8.5"
thresh40_85_figs <- annotate_figure(p = thresh40_85_figs,
                                    top = text_grob(label = lab, 
                                                    face = "bold", size = 18))

ggsave(filename = "thresh40_85.png", plot = thresh40_85_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh40_85"))
##############################################################################
#############
############# Thresh 45, RCP 85
#############
thresh45_85_min <- min(df_85$median_45)
thresh45_85_max <- max(df_85$median_45)

thresh45_85_min <- thresh_min
thresh45_85_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh45_85", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp45", month_col = "median_45",
                    min = thresh45_85_min, max = thresh45_85_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh45_85 <- observed_hist_map(min = thresh45_85_min,
                                              max = thresh45_85_max,
                                              month_col = "median_20")

ensemble_map_thresh45_85 <- ensemble_map(scenario_name = "rcp85", 
                                         month_col = "median_45",
                                         min = thresh45_85_min,
                                         max = thresh45_85_max)

# Need to add historical observed to this:
thresh45_85_figs <- ggarrange(plotlist = list(observed_map_thresh45_85,
                                              ensemble_map_thresh45_85,
                                              bcc_csm1_1_m_map_thresh45_85,
                                              bcc_csm1_1_map_thresh45_85,
                                              BNU_ESM_map_thresh45_85,
                                              CanESM2_map_thresh45_85,
                                              CCSM4_map_thresh45_85, 
                                              CNRM_CM5_map_thresh45_85,
                                              CSIRO_Mk3_6_0_map_thresh45_85,
                                              GFDL_ESM2G_map_thresh45_85,
                                              GFDL_ESM2M_map_thresh45_85,
                                              HadGEM2_CC365_map_thresh45_85,
                                              HadGEM2_ES365_map_thresh45_85,
                                              inmcm4_map_thresh45_85,
                                              IPSL_CM5A_LR_map_thresh45_85,
                                              IPSL_CM5A_MR_map_thresh45_85,
                                              IPSL_CM5B_LR_map_thresh45_85,
                                              MIROC_ESM_CHEM_map_thresh45_85,
                                              MIROC5_map_thresh45_85,
                                              MRI_CGCM3_map_thresh45_85,
                                              NorESM1_M_map_thresh45_85),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)
lab = "Median days to reach 45 chill unit threshold under RCP 8.5"
thresh45_85_figs <- annotate_figure(p = thresh45_85_figs,
                                    top = text_grob(label = lab, 
                                                    face = "bold", size = 18))

ggsave(filename = "thresh45_85.png", plot = thresh45_85_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)
rm(list = ls(pattern = "thresh45_85"))
##############################################################################
#############
############# Thresh 50, RCP 85
#############
thresh50_85_min <- min(df_85$median_50)
thresh50_85_max <- max(df_85$median_50)

thresh50_85_min <- thresh_min
thresh50_85_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh50_85", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp85", month_col = "median_50",
                    min = thresh50_85_min, max = thresh50_85_max)}
         )
}

# Separately created maps for observed historical and ensemble mean
observed_map_thresh50_85 <- observed_hist_map(min = thresh50_85_min,
                                              max = thresh50_85_max,
                                              month_col = "median_50")

ensemble_map_thresh50_85 <- ensemble_map(scenario_name = "rcp85", 
                                   month_col = "median_50",
                                   min = thresh50_85_min,
                                   max = thresh50_85_max)

# Need to add historical observed to this:
thresh50_85_figs <- ggarrange(plotlist = list(observed_map_thresh50_85,
                                              ensemble_map_thresh50_85,
                                              bcc_csm1_1_m_map_thresh50_85,
                                              bcc_csm1_1_map_thresh50_85,
                                              BNU_ESM_map_thresh50_85,
                                              CanESM2_map_thresh50_85,
                                              CCSM4_map_thresh50_85, 
                                              CNRM_CM5_map_thresh50_85,
                                              CSIRO_Mk3_6_0_map_thresh50_85,
                                              GFDL_ESM2G_map_thresh50_85,
                                              GFDL_ESM2M_map_thresh50_85,
                                              HadGEM2_CC365_map_thresh50_85,
                                              HadGEM2_ES365_map_thresh50_85,
                                              inmcm4_map_thresh50_85,
                                              IPSL_CM5A_LR_map_thresh50_85, 
                                              IPSL_CM5A_MR_map_thresh50_85,
                                              IPSL_CM5B_LR_map_thresh50_85,
                                              MIROC_ESM_CHEM_map_thresh50_85,
                                              MIROC5_map_thresh50_85, 
                                              MRI_CGCM3_map_thresh50_85,
                                              NorESM1_M_map_thresh50_85),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

thresh50_85_figs <- annotate_figure(p = thresh50_85_figs,
                                    top = text_grob(label = "Median days to reach 50 chill unit threshold under rcp 85",
                                                    face = "bold", size = 18))

ggsave(filename = "thresh50_85.png", plot = thresh50_85_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh50_85"))
##############################################################################
#############
############# Thresh 55, RCP 85
#############
thresh55_85_min <- min(df_85$median_55)
thresh55_85_max <- max(df_85$median_55)

thresh55_85_min <- thresh_min
thresh55_85_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh55_85", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp85", month_col = "median_55",
                    min = thresh55_85_min, max = thresh55_85_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh55_85 <- observed_hist_map(min = thresh55_85_min,
                                              max = thresh55_85_max,
                                              month_col = "median_55")
ensemble_map_thresh55_85 <- ensemble_map(scenario_name = "rcp85", 
                                   month_col = "median_55",
                                   min = thresh55_85_min,
                                   max = thresh55_85_max)

# Need to add historical observed to this:
thresh55_85_figs <- ggarrange(plotlist = list(observed_map_thresh55_85,
                                              ensemble_map_thresh55_85,
                                              bcc_csm1_1_m_map_thresh55_85,
                                              bcc_csm1_1_map_thresh55_85,
                                              BNU_ESM_map_thresh55_85,
                                              CanESM2_map_thresh55_85,
                                              CCSM4_map_thresh55_85, 
                                              CNRM_CM5_map_thresh55_85,
                                              CSIRO_Mk3_6_0_map_thresh55_85,
                                              GFDL_ESM2G_map_thresh55_85,
                                              GFDL_ESM2M_map_thresh55_85,
                                              HadGEM2_CC365_map_thresh55_85,
                                              HadGEM2_ES365_map_thresh55_85,
                                              inmcm4_map_thresh55_85,
                                              IPSL_CM5A_LR_map_thresh55_85, 
                                              IPSL_CM5A_MR_map_thresh55_85,
                                              IPSL_CM5B_LR_map_thresh55_85,
                                              MIROC_ESM_CHEM_map_thresh55_85,
                                              MIROC5_map_thresh55_85, 
                                              MRI_CGCM3_map_thresh55_85,
                                              NorESM1_M_map_thresh55_85),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

thresh55_85_figs <- annotate_figure(p = thresh55_85_figs,
                                    top = text_grob(label = "Median days to reach 55 chill unit threshold under rcp 85",
                                                    face = "bold", size = 18))

ggsave(filename = "thresh55_85.png", plot = thresh55_85_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh55_85"))
##############################################################################
#############
############# Thresh 60, RCP 8.5
#############
thresh60_85_min <- min(df_85$median_60)
thresh60_85_max <- max(df_85$median_60)

thresh60_85_min <- thresh_min
thresh60_85_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh60_85", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp85", month_col = "median_60",
                    min = thresh60_85_min, max = thresh60_85_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh60_85 <- observed_hist_map(min = thresh60_85_min,
                                              max = thresh60_85_max,
                                              month_col = "median_60")
ensemble_map_thresh60_85 <- ensemble_map(scenario_name = "rcp85", 
                                   month_col = "median_60",
                                   min = thresh60_85_min,
                                   max = thresh60_85_max)

# Need to add historical observed to this:
thresh60_85_figs <- ggarrange(plotlist = list(observed_map_thresh60_85,
                                              ensemble_map_thresh60_85,
                                              bcc_csm1_1_m_map_thresh60_85,
                                              bcc_csm1_1_map_thresh60_85,
                                              BNU_ESM_map_thresh60_85,
                                              CanESM2_map_thresh60_85,
                                              CCSM4_map_thresh60_85, 
                                              CNRM_CM5_map_thresh60_85,
                                              CSIRO_Mk3_6_0_map_thresh60_85,
                                              GFDL_ESM2G_map_thresh60_85,
                                              GFDL_ESM2M_map_thresh60_85,
                                              HadGEM2_CC365_map_thresh60_85,
                                              HadGEM2_ES365_map_thresh60_85,
                                              inmcm4_map_thresh60_85,
                                              IPSL_CM5A_LR_map_thresh60_85, 
                                              IPSL_CM5A_MR_map_thresh60_85,
                                              IPSL_CM5B_LR_map_thresh60_85,
                                              MIROC_ESM_CHEM_map_thresh60_85,
                                              MIROC5_map_thresh60_85, 
                                              MRI_CGCM3_map_thresh60_85,
                                              NorESM1_M_map_thresh60_85),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

thresh60_85_figs <- annotate_figure(p = thresh60_85_figs,
                                    top = text_grob(label = "Median days to reach 60 chill unit threshold under RCP 8.5",
                                                    face = "bold", size = 18))

ggsave(filename = "thresh60_85.png", plot = thresh60_85_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh60_85"))
##############################################################################
#############
############# Thresh 65, RCP 8.5
#############
thresh65_85_min <- min(df_85$median_65)
thresh65_85_max <- max(df_85$median_65)

# thresh65_85_min <- thresh_min
# thresh65_85_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh65_85", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp85", month_col = "median_65",
                    min = thresh65_85_min, max = thresh65_85_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh65_85 <- observed_hist_map(min = thresh65_85_min,
                                              max = thresh65_85_max,
                                              month_col = "median_65")
ensemble_map_thresh65_85 <- ensemble_map(scenario_name = "rcp85", 
                                   month_col = "median_65",
                                   min = thresh65_85_min,
                                   max = thresh65_85_max)

# Need to add historical observed to this:
thresh65_85_figs <- ggarrange(plotlist = list(observed_map_thresh65_85,
                                              ensemble_map_thresh65_85,
                                              bcc_csm1_1_m_map_thresh65_85,
                                              bcc_csm1_1_map_thresh65_85,
                                              BNU_ESM_map_thresh65_85,
                                              CanESM2_map_thresh65_85,
                                              CCSM4_map_thresh65_85, 
                                              CNRM_CM5_map_thresh65_85,
                                              CSIRO_Mk3_6_0_map_thresh65_85,
                                              GFDL_ESM2G_map_thresh65_85,
                                              GFDL_ESM2M_map_thresh65_85,
                                              HadGEM2_CC365_map_thresh65_85,
                                              HadGEM2_ES365_map_thresh65_85,
                                              inmcm4_map_thresh65_85,
                                              IPSL_CM5A_LR_map_thresh65_85, 
                                              IPSL_CM5A_MR_map_thresh65_85,
                                              IPSL_CM5B_LR_map_thresh65_85,
                                              MIROC_ESM_CHEM_map_thresh65_85,
                                              MIROC5_map_thresh65_85, 
                                              MRI_CGCM3_map_thresh65_85,
                                              NorESM1_M_map_thresh65_85),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

thresh65_85_figs <- annotate_figure(p = thresh65_85_figs,
                                    top = text_grob(label = "Median days to reach 65 chill unit threshold under RCP 8.5",
                                                    face = "bold", size = 18))

ggsave(filename = "thresh65_85.png", plot = thresh65_85_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh65_85"))
##############################################################################
#############
############# Thresh 70, RCP 8.5
#############
thresh70_85_min <- min(df_85$median_70)
thresh70_85_max <- max(df_85$median_70)

# thresh70_85_min <- thresh_min
# thresh70_85_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh70_85", sep="_"),
         value ={
          model_map(model = h, scenario_name = "rcp85", month_col = "median_70",
                    min = thresh70_85_min, max = thresh70_85_max)}
         )
}
# Separately created maps for observed historical and ensemble mean
observed_map_thresh70_85 <- observed_hist_map(min = thresh70_85_min,
                                              max = thresh70_85_max,
                                              month_col = "median_60")
ensemble_map_thresh70_85 <- ensemble_map(scenario_name = "rcp85", 
                                   month_col = "median_60",
                                   min = thresh70_85_min,
                                   max = thresh70_85_max)

# Need to add historical observed to this:
thresh70_85_figs <- ggarrange(plotlist = list(observed_map_thresh70_85,
                                              ensemble_map_thresh70_85,
                                              bcc_csm1_1_m_map_thresh70_85,
                                              bcc_csm1_1_map_thresh70_85,
                                              BNU_ESM_map_thresh70_85,
                                              CanESM2_map_thresh70_85,
                                              CCSM4_map_thresh70_85, 
                                              CNRM_CM5_map_thresh70_85,
                                              CSIRO_Mk3_6_0_map_thresh70_85,
                                              GFDL_ESM2G_map_thresh70_85,
                                              GFDL_ESM2M_map_thresh70_85,
                                              HadGEM2_CC365_map_thresh70_85,
                                              HadGEM2_ES365_map_thresh70_85,
                                              inmcm4_map_thresh70_85,
                                              IPSL_CM5A_LR_map_thresh70_85, 
                                              IPSL_CM5A_MR_map_thresh70_85,
                                              IPSL_CM5B_LR_map_thresh70_85,
                                              MIROC_ESM_CHEM_map_thresh70_85,
                                              MIROC5_map_thresh70_85, 
                                              MRI_CGCM3_map_thresh70_85,
                                              NorESM1_M_map_thresh70_85),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

thresh70_85_figs <- annotate_figure(p = thresh70_85_figs,
                                    top = text_grob(label = "Median days to reach 70 chill unit threshold under RCP 4.5",
                                                    face = "bold", size = 18))

ggsave(filename = "thresh70_85.png", plot = thresh70_85_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh70_85"))
##############################################################################
#############
############# Thresh 70, RCP 8.5
#############
thresh75_85_min <- min(df_85$median_75)
thresh75_85_max <- max(df_85$median_75)

thresh75_85_min <- thresh_min
thresh75_85_max <- thresh_max

for(h in unique(stats_comp$model)) {
  assign(x = paste(gsub(pattern = "-", replacement = "_", x = h), "map",
                   "thresh75_85", sep="_"),
         value ={
           model_map(model = h, scenario_name = "rcp85", month_col = "median_75",
                     min = thresh75_85_min, max = thresh75_85_max)}
           )
}

# Separately created maps for observed historical and ensemble mean
observed_map_thresh75_85 <- observed_hist_map(min = thresh75_85_min,
                                              max = thresh75_85_max,
                                              month_col = "median_75")

ensemble_map_thresh75_85 <- ensemble_map(scenario_name = "rcp85", 
                                         month_col = "median_75",
                                         min = thresh75_85_min,
                                         max = thresh75_85_max)

# Need to add historical observed to this:
thresh75_85_figs <- ggarrange(plotlist = list(observed_map_thresh75_85,
                                              ensemble_map_thresh75_85,
                                              bcc_csm1_1_m_map_thresh75_85,
                                              bcc_csm1_1_map_thresh75_85,
                                              BNU_ESM_map_thresh75_85,
                                              CanESM2_map_thresh75_85,
                                              CCSM4_map_thresh75_85, 
                                              CNRM_CM5_map_thresh75_85,
                                              CSIRO_Mk3_6_0_map_thresh75_85,
                                              GFDL_ESM2G_map_thresh75_85,
                                              GFDL_ESM2M_map_thresh75_85,
                                              HadGEM2_CC365_map_thresh75_85,
                                              HadGEM2_ES365_map_thresh75_85,
                                              inmcm4_map_thresh75_85,
                                              IPSL_CM5A_LR_map_thresh75_85, 
                                              IPSL_CM5A_MR_map_thresh75_85,
                                              IPSL_CM5B_LR_map_thresh75_85,
                                              MIROC_ESM_CHEM_map_thresh75_85,
                                              MIROC5_map_thresh75_85, 
                                              MRI_CGCM3_map_thresh75_85,
                                              NorESM1_M_map_thresh75_85),
                              ncol = 2, nrow = 11,
                              common.legend = TRUE)

thresh75_85_figs <- annotate_figure(p = thresh75_85_figs,
                                    top = text_grob(label = "Median days to reach 75 chill unit threshold under RCP 8.5",
                                                    face = "bold", size = 18))

ggsave(filename = "thresh75_85.png", plot = thresh75_85_figs, device = "png",
       width = 15, height = 40, units = "in", dpi=400, path=plot_path)

rm(list = ls(pattern = "thresh75_85"))

