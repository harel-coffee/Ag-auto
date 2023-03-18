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
##############################################################
base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
data_base <- paste0(base, "rain_snow_fractions/")

plot_dir <- paste0(base, "plots/precip/seasonal/clust_x/")
if (dir.exists(plot_dir)==F){
	dir.create(path = plot_dir, recursive = T)}
print (plot_dir)

diff_dir <- paste0(base, "/precip/02_med_diff_med_no_bias/")
#############################################################

AV_fileNs <- "seasonal_fracs"
AV_y_lab <- "precipitation (mm)"
AV_tg_col <- "seasonal_cum_precip"

AVs <- readRDS(paste0(data_base,"seasonal_fracs.rds")) %>% 
       data.table()
AVs <- subset(AVs, select = c("location", "cluster", "year",
                              "time_period","emission", 
                              "model","seasonal_cum_precip", 
                              "rain_fraction", 
                              "snow_fraction", "season"))
AVs <- remove_observed(AVs)
AVs <- remove_current_timeP(AVs) # remove 2006-2025
# update clusters labels
AVs <- convert_5_numeric_clusts_to_alphabet(data_tb = AVs) 

AVs_45 <- AVs %>% 
          filter(emission=="RCP 4.5") %>% 
          data.table()
AVs_85 <- AVs %>% 
          filter(emission=="RCP 8.5") %>% 
          data.table()
rm(AVs)
############################################################
diff_name <- "detail_med_diff_med_seasonal_precip.rds"
unbias_diff <- readRDS(paste0(diff_dir, diff_name)) %>% 
               data.table()
unbias_diff <- na.omit(unbias_diff)
unbias_diff <- remove_observed(unbias_diff)
 # remove 2006-2025
unbias_diff <- remove_current_timeP(unbias_diff)
# update clusters labels
unbias_diff <- convert_5_numeric_clusts_to_alphabet(unbias_diff) 

unbias_diff_45 <- unbias_diff %>% 
                  filter(emission=="RCP 4.5") %>% 
                  data.table()
unbias_diff_85 <- unbias_diff %>% 
                  filter(emission=="RCP 8.5") %>% 
                  data.table()
rm(unbias_diff)

############################################################
season_types <- c("fall", "winter", "spring", "summer")
season_g <- "fall"

for (season_g in season_types){
  AV_title <- paste0(season_g, 
                     " precipitation for historical and ",
                     "three future time frames")
  
  curr_AVs_85 <- AVs_85 %>% 
                 filter(season == season_g) %>% 
                 data.table()
  curr_AVs_45 <- AVs_45 %>% 
                 filter(season == season_g) %>% 
                 data.table()

  curr_diff_85 <- unbias_diff_85 %>% 
                  filter(season == season_g) %>% 
                  data.table()
  curr_diff_45 <- unbias_diff_45 %>% 
                  filter(season == season_g) %>% 
                  data.table()
  ######################################################
  #########
  ######### Actual value plots
  #########
  ######################################################
  quans_85 <- find_quantiles(curr_AVs_85, 
                             tgt_col= AV_tg_col, 
                             time_type="seasonal")
  quans_45 <- find_quantiles(curr_AVs_45, 
                             tgt_col= AV_tg_col, 
                             time_type="seasonal")
  
  AV_box_85 <- seasonal_cum_box_clust_x(dt = curr_AVs_85, 
                                     tgt_col = AV_tg_col,
                                     y_lab = AV_y_lab) +
               ggtitle(label= paste0(AV_title)) +
        coord_cartesian(ylim = c(quans_85[1], quans_85[2]))

  AV_box_45 <- seasonal_cum_box_clust_x(dt = curr_AVs_45, 
                                        tgt_col = AV_tg_col,
                                        y_lab = AV_y_lab) +
               ggtitle(label= paste0(AV_title)) + 
          coord_cartesian(ylim = c(quans_45[1], quans_45[2]))
  #############################################
  #########
  ######### difference plot
  #########
  #############################################
  box_title <- paste0("% difference between future and historical ", 
                      season_g, 
                      " precipitation")

  quans_85 <- find_quantiles(curr_diff_85, 
  	                         tgt_col="perc_diff", 
  	                         time_type="seasonal")
  quans_45 <- find_quantiles(curr_diff_45, 
                             tgt_col="perc_diff", 
                             time_type="seasonal")

  unbias_perc_diff_85<-seasonal_cum_box_clust_x(dt=curr_diff_85,
                                        y_lab="differences (%)",
                                        tgt_col="perc_diff") + 
                         ggtitle(box_title) +
                         coord_cartesian(ylim=c(quans_85[1],
                                                quans_85[2]))

  unbias_perc_diff_45 <- seasonal_cum_box_clust_x(dt=curr_diff_45,
                                          y_lab="differences (%)",
                                          tgt_col="perc_diff") + 
                         ggtitle(box_title) + 
                         coord_cartesian(ylim = c(quans_45[1], 
                                                  quans_45[2]))
  #############################################
  #########
  ######### rain plot
  #########
  #############################################
  box_title <- paste0("proportion (%) of ", 
                      season_g, 
                      " precipitation in rain form")
  
  quans_85 <- 100 * find_quantiles(curr_AVs_85, 
                                   tgt_col= "rain_fraction", 
                                   time_type="seasonal")
  quans_45 <- 100 * find_quantiles(curr_AVs_45, 
                                   tgt_col="rain_fraction", 
                                   time_type="seasonal")

  rain_frac_85 <- seasonal_fraction_clust_x(data_tb=curr_AVs_85,
                                      y_lab = "rain portion (%)", 
                                      tgt_col="rain_fraction") +
                  ggtitle(box_title) + 
              coord_cartesian(ylim = c(max(-2, quans_85[1]), 
                                    min(quans_85[2], 110)))

  rain_frac_45 <- seasonal_fraction_clust_x(data_tb = curr_AVs_45,
                                      y_lab = "rain portion (%)", 
                                      tgt_col="rain_fraction") +
                  ggtitle(box_title) +
             coord_cartesian(ylim = c(max(-2, quans_45[1]), 
                                      min(quans_45[2], 110)))
  ######################################################
  ########
  ########    Save just plots
  ########
  ######################################################
  ########
  ########   just frac
  ########
  just_frac_plt_dir <- paste0(plot_dir, "/just_frac/")
  if (dir.exists(just_frac_plt_dir) == F) {
    dir.create(path = just_frac_plt_dir, recursive = T)}
  
  ggsave(filename = paste0(season_g, "_rain_85.png"),
         plot = rain_frac_85, 
         width=5.5, height=1.5, units = "in", 
         dpi=600, device = "png", 
         path = just_frac_plt_dir)

  ggsave(filename = paste0(season_g, "_rain_45.png"),
         plot = rain_frac_45, 
         width =5.5, height =1.5, units = "in", 
         dpi=600, device = "png", 
         path=just_frac_plt_dir)
  
  ######################################################
  ########
  ########   just AVs
  ########
  just_av_plt_dir <- paste0(plot_dir, "/just_AVs/")
  if (dir.exists(just_av_plt_dir) == F) {
    dir.create(path = just_av_plt_dir, recursive = T)}
  
  ggsave(filename = paste0(season_g, "_rain_85.png"),
         plot = AV_box_85, 
         width=5.5, height=1.5, units = "in", 
         dpi=600, device = "png", path = just_av_plt_dir)

  ggsave(filename = paste0(season_g, "_rain_45.png"),
         plot = AV_box_45, 
         width=5.5, height=1.5, units = "in", 
         dpi=600, device = "png", path = just_av_plt_dir)

  ######################################################
  ########
  ########   just diffs
  ########
  just_diff_plt_dir <- paste0(plot_dir, "/just_diff/")
  if (dir.exists(just_diff_plt_dir) == F) {
    dir.create(path = just_diff_plt_dir, recursive = T)}
  
  ggsave(filename = paste0(season_g, "_rain_85.png"),
         plot = unbias_perc_diff_85, 
         width=5.5, height=1.5, units = "in", 
         dpi=600, device = "png", path = just_diff_plt_dir)

  ggsave(filename = paste0(season_g, "_rain_45.png"),
         plot = unbias_perc_diff_45, 
         width =5.5, height =1.5, units = "in", 
         dpi=600, device = "png", path = just_diff_plt_dir)
  
  ######################################################
  ########
  ########    precip_n_diff
  ########
  precip_n_diff_dir <- paste0(plot_dir, "/precip_&_diff/")
  if (dir.exists(precip_n_diff_dir) == F) {
    dir.create(path = precip_n_diff_dir, recursive = T)}
  
  rain_45 <- ggarrange(plotlist = list(AV_box_45, 
  	                                   unbias_perc_diff_45),
                       ncol = 1, nrow = 2, 
                       common.legend = TRUE, legend="bottom")

  rain_85 <- ggarrange(plotlist = list(AV_box_85,
                                       unbias_perc_diff_85),
                       ncol = 1, nrow = 2, 
                       common.legend = TRUE, legend="bottom")

  ggsave(filename = paste0(season_g, "_rain_85.png"),
         plot = rain_85, 
         width=5.5, height=3.5, units = "in", 
         dpi=600, device = "png", path = precip_n_diff_dir)

  ggsave(filename = paste0(season_g, "_rain_45.png"),
         plot = rain_45, 
         width=5.5, height=3.5, units = "in", 
         dpi=600, device = "png", path = precip_n_diff_dir)
  ######################################################
  ########
  ########    precip_n_frac
  ########
  precip_n_frac <- paste0(plot_dir, "/precip_&_frac/")
  if (dir.exists(precip_n_frac) == F) {
    dir.create(path = precip_n_frac, recursive = T)}
  rain_45 <- ggarrange(plotlist = list(AV_box_45, rain_frac_45),
                       ncol = 1, nrow = 2, 
                       common.legend = TRUE, legend="bottom")
  rain_85 <- ggarrange(plotlist = list(AV_box_85, rain_frac_85),
                       ncol = 1, nrow = 2, 
                       common.legend = TRUE, legend="bottom")

  ggsave(filename = paste0(season_g, "_rain_85.png"),
         plot = rain_85, 
         width=5.5, height=3.5, units = "in", 
         dpi=600, device = "png", path = precip_n_frac)

  ggsave(filename = paste0(season_g, "_rain_45.png"),
         plot = rain_45, 
         width =5.5, height=3.5, units = "in", 
         dpi=600, device = "png", path = precip_n_frac)
  ######################################################
  ########
  ########    Save 3 in 1
  ########
  plot_dir_3_in_1 <- paste0(plot_dir, "/3_in_1/")
  if (dir.exists(plot_dir_3_in_1) == F) {
    dir.create(path = plot_dir_3_in_1, recursive = T)}
  rain_45 <- ggarrange(plotlist = list(AV_box_45, 
                                       unbias_perc_diff_45,
                                       rain_frac_45),
                       ncol = 1, nrow = 3, 
                       common.legend = TRUE, legend="bottom")
  rain_85 <- ggarrange(plotlist = list(AV_box_85,
                                       unbias_perc_diff_85,
                                       rain_frac_85),
                       ncol = 1, nrow = 3, 
                       common.legend = TRUE, legend="bottom")

  ggsave(filename = paste0(season_g, "_rain_85.png"),
         plot = rain_85, 
         width=5.5, height=5, units = "in", 
         dpi=600, device = "png", path = plot_dir_3_in_1)

  ggsave(filename = paste0(season_g, "_rain_45.png"),
         plot = rain_45, 
         width =5.5, height=5, units = "in", 
         dpi=600, device = "png", path = plot_dir_3_in_1)
}



