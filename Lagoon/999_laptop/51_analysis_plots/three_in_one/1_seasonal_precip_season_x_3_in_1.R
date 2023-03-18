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
##########################################################################
base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
data_base <- paste0(base, "rain_snow_fractions/")
diff_dir <- paste0(base, "/precip/02_med_diff_med_no_bias/")
##########################################################################
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
############################################################
unbias_diff <- readRDS(paste0(diff_dir, 
                              "detail_med_diff_med_seasonal_precip.rds")) %>% 
                   data.table()
unbias_diff <- na.omit(unbias_diff)
unbias_diff <- remove_observed(unbias_diff)
unbias_diff <- remove_current_timeP(unbias_diff) # remove 2006-2025
unbias_diff <- convert_5_numeric_clusts_to_alphabet(data_tb = unbias_diff) # update clusters labels
unbias_diff_45 <- unbias_diff %>% filter(emission=="RCP 4.5") %>% data.table()
unbias_diff_85 <- unbias_diff %>% filter(emission=="RCP 8.5") %>% data.table(); rm(unbias_diff)

############################################################
cluster_types <- unique(AVs_85$cluster)
clust_g <- cluster_types[1]

for (clust_g in cluster_types){
  subttl <- paste0(" (", clust_g, ")")
  curr_AVs_85 <- AVs_85 %>% filter(cluster == clust_g) %>% data.table()
  curr_AVs_45 <- AVs_45 %>% filter(cluster == clust_g) %>% data.table()

  curr_diff_85 <- unbias_diff_85 %>% filter(cluster == clust_g) %>% data.table()
  curr_diff_45 <- unbias_diff_45 %>% filter(cluster == clust_g) %>% data.table()
  #########
  ######### Actual value plots
  #########
  quans_85 <- find_quantiles(curr_AVs_85, tgt_col=AV_tg_col, time_type="seasonal")
  quans_45 <- find_quantiles(curr_AVs_45, tgt_col=AV_tg_col, time_type="seasonal")
  
  AV_box_85 <- seasonal_cum_box_season_x(dt = curr_AVs_85, 
                                         tgt_col = AV_tg_col,
                                         y_lab = AV_y_lab) +
               ggtitle(label= paste0(AV_title, subttl)) +
               coord_cartesian(ylim = c(quans_85[1], quans_85[2]))

  AV_box_45 <- seasonal_cum_box_season_x(dt = curr_AVs_45, 
                                         tgt_col = AV_tg_col,
                                         y_lab = AV_y_lab) +
               ggtitle(label= paste0(AV_title, subttl)) + 
               coord_cartesian(ylim = c(quans_45[1], quans_45[2]))
  #########
  ######### difference plot
  #########
  box_title <- paste0("percentage differences between future time periods and historical", subttl)
  quans_85 <- find_quantiles(curr_diff_85, tgt_col= "perc_diff", time_type="seasonal")
  quans_45 <- find_quantiles(curr_diff_45, tgt_col= "perc_diff", time_type="seasonal")

  unbias_perc_diff_85 <- seasonal_cum_box_season_x(dt = curr_diff_85,
                                                  y_lab = "differences (%)",
                                                  tgt_col = "perc_diff") + 
                         ggtitle(box_title) +
                         coord_cartesian(ylim=c(quans_85[1], quans_85[2]))

  unbias_perc_diff_45 <- seasonal_cum_box_season_x(dt = curr_diff_45,
                                                  y_lab = "differences (%)",
                                                  tgt_col = "perc_diff") + 
                         ggtitle(box_title) + 
                         coord_cartesian(ylim=c(quans_45[1], quans_45[2]))
  #########
  #########
  ######### rain plot
  #########
  box_title <- paste0("fraction of precip. fell as rain (", clust_g, ")")
  quans_85 <- 100 * find_quantiles(curr_AVs_85, tgt_col="rain_fraction", time_type="seasonal")
  quans_45 <- 100 * find_quantiles(curr_AVs_45, tgt_col="rain_fraction", time_type="seasonal")

  rain_frac_85 <- seasonal_fraction_season_x(data_tb = curr_AVs_85,
                                             y_lab = "rain fraction (%)", 
                                             tgt_col="rain_fraction") +
                  ggtitle(box_title) +
                  coord_cartesian(ylim = c(max(quans_85[1], -2), 
                                           min(quans_85[2], 110)))

  rain_frac_45 <- seasonal_fraction_season_x(data_tb = curr_AVs_45,
                                             y_lab = "rain fraction (%)", 
                                             tgt_col="rain_fraction") +
                  ggtitle(box_title) + 
                  coord_cartesian(ylim = c(max(quans_45[1], -2), 
                                           min(quans_45[2], 110)))
  #######################################################################
  ############
  ############
  rain_85 <- ggarrange(plotlist = list(AV_box_85, 
                                       unbias_perc_diff_85,
                                       rain_frac_85),
                       ncol=1, nrow=3, 
                       common.legend = TRUE, legend="bottom")
  
  rain_45 <- ggarrange(plotlist = list(AV_box_45,
                                       unbias_perc_diff_45,
                                       rain_frac_45),
                       ncol=1, nrow=3,
                       common.legend=TRUE, legend="bottom")

  DD <- "narrowed_rain_snow_fractions/seasonal/season_x/3_in_1/"
  plot_dir <- paste0(data_base, DD)
  if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
  print (plot_dir)
  
  ggsave(filename = paste0(clust_g, "_rain_85.png"),
         plot = rain_85, width = 5.5, height=5, units = "in", 
         dpi=400, device = "png", path = plot_dir)
  
  ggsave(filename = paste0(clust_g, "_rain_45.png"),
         plot = rain_45, width =5.5, height=5, units = "in", 
         dpi=400, device = "png", path = plot_dir)
  ######################################################
  ########
  ########    Save just plots; 
  ########    AV_box_45, rain_frac_45, unbias_perc_diff_45
  ########
  ######################################################
  ########
  ########   just frac 
  ########
  DD <- "narrowed_rain_snow_fractions/seasonal/season_x/just_frac/"
  just_frac_plt_dir <- paste0(data_base, DD)
  if (dir.exists(just_frac_plt_dir) == F){
     dir.create(path = just_frac_plt_dir, recursive = T)}

  ggsave(filename = paste0(clust_g, "_rain_85.png"),
         plot = rain_frac_85, width=5.5, height=1.5, units = "in", 
         dpi=400, device = "png", path = just_frac_plt_dir)


  ggsave(filename = paste0(clust_g, "_rain_45.png"),
         plot = rain_frac_45, width =5.5, height =1.5, units = "in", 
         dpi=400, device = "png", path = just_frac_plt_dir)
  ######################################################
  ########
  ########   just AVs
  ########
  DD <- "narrowed_rain_snow_fractions/seasonal/season_x/just_AVs/"
  just_av_plt_dir <- paste0(data_base, DD)
  if (dir.exists(just_av_plt_dir) == F) {
    dir.create(path = just_av_plt_dir, recursive = T)}
  
  ggsave(filename = paste0(clust_g, "_rain_85.png"),
         plot = AV_box_85, width=5.5, height=1.5, units = "in", 
         dpi=400, device = "png", path = just_av_plt_dir)

  ggsave(filename = paste0(clust_g, "_rain_45.png"),
         plot = AV_box_45, width =5.5, height =1.5, units = "in", 
         dpi=400, device = "png", path = just_av_plt_dir)

  ######################################################
  ########
  ########   just diffs
  ########
  DD <- "narrowed_rain_snow_fractions/seasonal/season_x/just_diff/"
  just_diff_plt_dir <- paste0(data_base, DD)
  if (dir.exists(just_diff_plt_dir) == F) {
    dir.create(path = just_diff_plt_dir, recursive = T)}
  
  ggsave(filename = paste0(clust_g, "_rain_85.png"),
         plot = unbias_perc_diff_85, width=5.5, height=1.5, units = "in", 
         dpi=400, device = "png", path = just_diff_plt_dir)

  ggsave(filename = paste0(clust_g, "_rain_45.png"),
         plot = unbias_perc_diff_45, width =5.5, height =1.5, units = "in", 
         dpi=400, device = "png", path = just_diff_plt_dir) 
  
}


