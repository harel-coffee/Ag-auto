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
#
# This is a lost cause. sick, set in specific ways, jumping to conclusions.
# not spending time to think.
#
source_path_1 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)
start_time <- Sys.time()
############################################################################

data_base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/rain_snow_fractions/"
in_dir <- data_base
############################################################################
AV_title <- "monthly precip."
AV_y_lab <- "cum. precip. (mm)"
AV_tg_col <- "monthly_cum_precip"

AVs <- readRDS(paste0(in_dir, "monthly_fracs.rds")) %>% data.table()

AVs <- remove_observed(AVs)
AVs <- remove_current_timeP(AVs) # remove 2006-2025
AVs <- convert_5_numeric_clusts_to_alphabet(data_tb = AVs)# update clusters labels
AVs <- na.omit(AVs)
# AVs <- AVs %>% filter(!(month %in% c(5, 6, 7, 8)))
plot_dir <- paste0(in_dir, "narrowed_rain_snow_fractions/monthly/final//") # 5-8-gone
if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
print (plot_dir)

AVs <- within(AVs, remove(day, rain_portion, monthly_cum_rain, monthly_cum_snow))
AVs$rain_fraction <- AVs$rain_fraction * 100
AVs$snow_fraction <- AVs$snow_fraction * 100

AVs_45 <- AVs %>% filter(emission=="RCP 4.5") %>% data.table()
AVs_85 <- AVs %>% filter(emission=="RCP 8.5") %>% data.table()
timeP_ty <- 1
cluster_types <- unique(AVs$cluster)
clust_g <- cluster_types[1]

for (clust_g in cluster_types){
  print(clust_g)
  curr_AVs_45 <- AVs_45 %>% filter(cluster == clust_g) %>% data.table()
  curr_AVs_85 <- AVs_85 %>% filter(cluster == clust_g) %>% data.table()
  #########
  ######### Actual value plots
  #########
  quans_85 <- find_quantiles(curr_AVs_85, tgt_col= AV_tg_col, time_type="monthly")
  quans_45 <- find_quantiles(curr_AVs_45, tgt_col= AV_tg_col, time_type="monthly")

  AV_box_85 <- box_trend_monthly_cum(dt = curr_AVs_85, p_type="box",
                                     y_lab = AV_y_lab, tgt_col = AV_tg_col) +
               ggtitle(AV_title) + 
               coord_cartesian(ylim = c(quans_85[1], quans_85[2]))

  AV_box_45 <- box_trend_monthly_cum(dt = curr_AVs_45, p_type="box",
                                     y_lab = AV_y_lab, tgt_col = AV_tg_col) + 
               ggtitle(AV_title) + 
               coord_cartesian(ylim = c(quans_45[1], quans_45[2]))
  #########
  ######### rain fracitons
  #########
  box_title <- paste0("fraction of precip. fell as rain")
  quans_85 <- find_quantiles(curr_AVs_85, tgt_col= "rain_fraction", time_type="monthly")
  quans_45 <- find_quantiles(curr_AVs_45, tgt_col= "rain_fraction", time_type="monthly")
             
  rain_85 <- box_trend_monthly_cum(dt = curr_AVs_85, p_type="box",
                                   y_lab = "rain fraction (%)", 
                                   tgt_col = "rain_fraction") + 
             ggtitle("fraction of precip. fell as rain") + 
             coord_cartesian(ylim = c(quans_85[1], 105))
  
  rain_85 <- ggarrange(plotlist = list(AV_box_85, rain_85),
                       ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")

  ggsave(filename = paste0(clust_g, "_rain_85.png"),
         plot = rain_85, width=9, height=5, units = "in", 
         dpi=400, device = "png", path = plot_dir)
  ####################################################################
  rain_45 <- box_trend_monthly_cum(dt = curr_AVs_45, p_type="box",
                                   y_lab = "rain fraction (%)", 
                                   tgt_col = "rain_fraction") + 
             ggtitle("fraction of precip. fell as rain") + 
             coord_cartesian(ylim = c(quans_45[1], 105))

  rain_45 <- ggarrange(plotlist = list(AV_box_45, rain_45),
                       ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")
  
  ggsave(filename = paste0(clust_g, "_rain_45.png"),
         plot = rain_45, width=9, height=5, units = "in", 
         dpi=400, device = "png", path = plot_dir)
  ####################################################################
  # box_title <- paste0("snow fraction")
  # quans_85 <- find_quantiles(curr_AVs_85, tgt_col= "snow_fraction", time_type="monthly")
  # quans_45 <- find_quantiles(curr_AVs_45, tgt_col= "snow_fraction", time_type="monthly")

  # snow_85 <- box_trend_monthly_cum(dt = curr_AVs_85, p_type="box",
  #                                  y_lab = "snow fraction (%)",
  #                                  tgt_col = "snow_fraction") + 
  #            ggtitle("fracion of precip. fell as snow") +
  #            coord_cartesian(ylim = c(0, 105)) # quans_85[1]
  
  # snow_85 <- ggarrange(plotlist = list(AV_box_85, snow_85),
  #                      ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")
  # ggsave(filename = paste0(clust_g, "_snow_85.png"),
  #        plot = snow_85, width=9, height=5, units = "in", 
  #        dpi=400, device = "png", path = plot_dir)
  ####################################################################  
  # snow_45 <- box_trend_monthly_cum(dt = curr_AVs_45, p_type="box",
  #                                  y_lab = "snow fraction (%)",
  #                                  tgt_col = "snow_fraction") + 
  #            ggtitle("fracion of precip. fell as snow") +
  #            coord_cartesian(ylim = c(0, 105)) # quans_45[1]

  # snow_45 <- ggarrange(plotlist = list(AV_box_45, snow_45),
  #                      ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")
  
  # ggsave(filename = paste0(clust_g, "_snow_45.png"),
  #        plot = snow_45, width=9, height=5, units = "in", 
  #        dpi=400, device = "png", path = plot_dir)
  # print(paste0(clust_g, "_snow_45.png"))
  ##################################################################################
  
}

print (Sys.time() - start_time)