rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
options(digit=9)
options(digits=9)

source_path_1 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)
############################################################################

data_base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/rain_snow_fractions/"
AV_fileNs <- "seasonal_fracs"

AV_y_lab <- "cum. precip. (mm)"
AV_title <- paste0("seasonal precip.")
AV_tg_col <- "seasonal_cum_precip"

AVs <- readRDS(paste0(data_base,"seasonal_fracs.rds")) %>% data.table()
AVs <- subset(AVs, select = c("location", "cluster", "year", "time_period", 
                              "model", "emission",
                              "seasonal_cum_precip", "rain_fraction", "snow_fraction",
                              "season"))

AVs <- remove_observed(AVs)
AVs <- remove_current_timeP(AVs) # remove 2006-2025
# update clusters labels
AVs <- convert_5_numeric_clusts_to_alphabet(data_tb = AVs) 

AVs_45 <- AVs %>% filter(emission=="RCP 4.5") %>% data.table()
AVs_85 <- AVs %>% filter(emission=="RCP 8.5") %>% data.table(); rm(AVs)
cluster_types <- unique(AVs_85$cluster)
clust_g <- cluster_types[1]

for (clust_g in cluster_types){
  subttl <- paste0(" (", clust_g, ")")
  curr_AVs_85 <- AVs_85 %>% filter(cluster == clust_g) %>% data.table()
  curr_AVs_45 <- AVs_45 %>% filter(cluster == clust_g) %>% data.table()

  #########
  ######### Actual value plots
  #########
  quans_85 <- find_quantiles(curr_AVs_85, tgt_col= AV_tg_col, time_type="seasonal")
  quans_45 <- find_quantiles(curr_AVs_45, tgt_col= AV_tg_col, time_type="seasonal")
  
  AV_box_85 <- seasonal_cum_box_season_x(dt = curr_AVs_85, tgt_col = AV_tg_col,
                                         y_lab = AV_y_lab) +
               ggtitle(label= paste0(AV_title, subttl)) +
               coord_cartesian(ylim = c(quans_85[1], quans_85[2]))

  AV_box_45 <- seasonal_cum_box_season_x(dt = curr_AVs_45, tgt_col = AV_tg_col,
                                         y_lab = AV_y_lab) +
               ggtitle(label= paste0(AV_title, subttl)) + 
               coord_cartesian(ylim = c(quans_45[1], quans_45[2]))
  #########
  ######### rain plot
  #########
  plot_dir <- paste0(data_base, "narrowed_rain_snow_fractions/seasonal/season_x/")
  if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
  print (plot_dir)

  box_title <- paste0("fraction of precip. fell as rain (", clust_g, ")")
  quans_85 <- 100 * find_quantiles(curr_AVs_85, tgt_col= "rain_fraction", time_type="seasonal")
  quans_45 <- 100 * find_quantiles(curr_AVs_45, tgt_col= "rain_fraction", time_type="seasonal")

  rain_frac_85 <- seasonal_fraction_season_x(data_tb = curr_AVs_85,
                                             y_lab = "rain fraction (%)", 
                                             tgt_col="rain_fraction") +
                  ggtitle(box_title) +
                  coord_cartesian(ylim = c(max(-2, quans_85[1]), 
                                           min(quans_85[2], 110)))

  rain_85 <- ggarrange(plotlist = list(AV_box_85, rain_frac_85),
                       ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")
  
  ggsave(filename = paste0(clust_g, "_rain_85.png"),
         plot = rain_85, width = 5.5, height = 3, units = "in", 
         dpi=400, device = "png", path = plot_dir)

  rain_frac_45 <- seasonal_fraction_season_x(data_tb = curr_AVs_45,
                                             y_lab = "rain fraction (%)", 
                                             tgt_col="rain_fraction") +
                  ggtitle(box_title) + 
                  coord_cartesian(ylim = c(max(quans_45[1], -2),
                                           min(quans_45[2], 110)))

  rain_45 <- ggarrange(plotlist = list(AV_box_45, rain_frac_45),
                       ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")
  ggsave(filename = paste0(clust_g, "_rain_45.png"),
         plot = rain_45, width =5.5, height =3, units = "in", 
         dpi=400, device = "png", path = plot_dir)
  ##############################################################################
  ##############################################################################
  # box_title <- paste0("snow fraction (", clust_g, ")")
  # quans_85 <- 100 * find_quantiles(curr_AVs_85, tgt_col= "snow_fraction", time_type="seasonal")
  # quans_45 <- 100 * find_quantiles(curr_AVs_45, tgt_col= "snow_fraction", time_type="seasonal")

  # snow_frac_85 <- seasonal_fraction_season_x(data_tb = curr_AVs_85,
  #                                            y_lab = "snow fraction (%)", 
  #                                            tgt_col="snow_fraction") +
  #                 ggtitle(box_title) +
  #                 coord_cartesian(ylim = c(quans_85[1], 110))

  # snow_85 <- ggarrange(plotlist = list(AV_box_85, snow_frac_85),
  #                      ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")
  # ggsave(filename = paste0(clust_g, "_snow_85.png"),
  #        plot = snow_85, width=5.5, height=3, units = "in", 
  #        dpi=400, device = "png", path = plot_dir)

  # snow_frac_45 <- seasonal_fraction_season_x(data_tb = curr_AVs_45,
  #                                            y_lab = "snow fraction (%)", 
  #                                            tgt_col="snow_fraction") +
  #                 ggtitle(box_title) + 
  #                 coord_cartesian(ylim = c(quans_45[1], 110))

  # snow_45 <- ggarrange(plotlist = list(AV_box_45, snow_frac_45),
  #                      ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")
  # ggsave(filename = paste0(clust_g, "_snow_45.png"),
  #        plot = snow_45, width=5.5, height=3, units = "in",
  #        dpi=400, device = "png", path = plot_dir)
  # print(paste0(clust_g, "_snow_45.png"))
  print(plot_dir)
}


